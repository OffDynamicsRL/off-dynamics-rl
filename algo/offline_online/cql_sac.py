import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints

from torch.distributions.transforms import Transform


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False, repeat=None):
        if repeat is not None:
            x = extend_and_repeat(x, 1, repeat)
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=-1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        
        return action * self.max_action, logprob, mean * self.max_action

class DoubleQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        multiple_actions = False
        batch_size = state.shape[0]
        if action.ndim == 3 and state.ndim == 2:
            multiple_actions = True
            state = extend_and_repeat(state, 1, action.shape[1]).reshape(
                -1, state.shape[-1]
            )
            action = action.reshape(-1, action.shape[-1])
        x = torch.cat([state, action], dim=-1)
        q1 = torch.squeeze(self.network1(x), dim=-1)
        q2 = torch.squeeze(self.network2(x), dim=-1)
        if multiple_actions:
            q1 = q1.reshape(batch_size, -1)
            q2 = q2.reshape(batch_size, -1)
        return q1, q2


class CQLSAC(object):

    def __init__(self,
                 config,
                 device,
                 target_entropy=None,
                 ):
        self.config=  config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.target_entropy = target_entropy if target_entropy else -config['action_dim']
        self.update_interval = config['update_interval']

        self.total_it = 0

        # aka critic
        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        # aka temperature
        if config['temperature_opt']:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([0.2])).to(self.device)
        
        # aka cql temperature
        self.log_alpha_prime = torch.log(torch.FloatTensor([1.0])).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
        self.temp_prime_optimizer = torch.optim.Adam([self.log_alpha_prime], lr=config['actor_lr'])
    
    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            q_target = q_target.unsqueeze(1)
            if self.config['backup_entropy']:
                value_target = reward_batch + not_done_batch * self.discount * (q_target - self.alpha * logprobs_batch)
            else:
                value_target = reward_batch + not_done_batch * self.discount * q_target
            # value_target = reward_batch + not_done_batch * self.discount * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        q_1, q_2 = q_1.unsqueeze(1), q_2.unsqueeze(1)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
            writer.add_scalar('train/logprob', logprobs_batch.mean(), self.total_it)
        loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)
        return loss
    
    def update_cql_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None):
        with torch.no_grad():
            if self.config['cql_max_target_backup']:
                nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True, repeat=self.config['cql_n_qctions'])
                q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
                q_target, max_target_indices = torch.max(torch.min(q_t1, q_t2), dim=-1)
                logprobs_batch = torch.gather(logprobs_batch, -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
            else:
                nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
                q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
                # take min to mitigate positive bias in q-function training
                q_target = torch.min(q_t1, q_t2)
            if self.config['backup_entropy']:
                q_target = q_target.squeeze() - self.alpha * logprobs_batch.squeeze()
            value_target = reward_batch.squeeze() + not_done_batch.squeeze() * self.discount * q_target.squeeze()
            
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)

        # add CQL loss
        batch_size = action_batch.shape[0]
        action_dim = action_batch.shape[-1]
        cql_random_actions = action_batch.new_empty((batch_size, self.config['cql_n_actions'], action_dim), requires_grad=False).uniform_(-1, 1)
        cql_current_actions, cql_current_log_prob, _ = self.policy(state_batch, get_logprob=True, repeat=self.config['cql_n_actions'])
        cql_next_actions,    cql_next_log_prob,    _ = self.policy(nextstate_batch, get_logprob=True, repeat=self.config['cql_n_actions'])
        cql_current_actions, cql_current_log_prob = (cql_current_actions.detach(), cql_current_log_prob.detach())
        cql_next_actions,    cql_next_log_prob    = (cql_next_actions.detach(), cql_next_log_prob.detach())

        cql_q1_rand,            cql_q2_rand            = self.q_funcs(state_batch, cql_random_actions)
        cql_q1_current_actions, cql_q2_current_actions = self.q_funcs(state_batch, cql_current_actions)
        cql_q1_next_actions,    cql_q2_next_actions    = self.q_funcs(state_batch, cql_next_actions)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q_1, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q_2, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.config['cql_importance_sample']:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat([cql_q1_rand - random_density,
                                    cql_q1_next_actions - cql_next_log_prob.detach().squeeze(),
                                    cql_q1_current_actions - cql_current_log_prob.detach().squeeze()], dim=1,)
            cql_cat_q2 = torch.cat([cql_q2_rand - random_density,
                                    cql_q2_next_actions - cql_next_log_prob.detach().squeeze(),
                                    cql_q2_current_actions - cql_current_log_prob.detach().squeeze()], dim=1,)

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.config['cql_temp'], dim=1) * self.config['cql_temp']
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.config['cql_temp'], dim=1) * self.config['cql_temp']

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(cql_qf1_ood - q_1, self.config['cql_clip_diff_min'], self.config['cql_clip_diff_max']).mean()
        cql_qf2_diff = torch.clamp(cql_qf2_ood - q_2, self.config['cql_clip_diff_min'], self.config['cql_clip_diff_max']).mean()

        if self.config['cql_lagrange']:
            alpha_prime = torch.clamp(self.alpha_prime, min=0.0, max=1000000.0)
            cql_min_qf1_loss = alpha_prime * self.config['cql_alpha'] * (cql_qf1_diff - self.config['cql_target_action_gap'])
            cql_min_qf2_loss = alpha_prime * self.config['cql_alpha'] * (cql_qf2_diff - self.config['cql_target_action_gap'])
            self.temp_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.temp_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.config['cql_alpha']
            cql_min_qf2_loss = cql_qf2_diff * self.config['cql_alpha']
            alpha_prime_loss = state_batch.new_tensor(0.0)
            alpha_prime = state_batch.new_tensor(0.0)

        loss += cql_min_qf1_loss.squeeze() + cql_min_qf2_loss.squeeze()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/cql q1', q_1.mean(), self.total_it)
            writer.add_scalar('train/cql logprob', logprobs_batch.mean(), self.total_it)
        
        return loss

    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):

        self.total_it += 1

        if src_replay_buffer.size < 2*batch_size or tar_replay_buffer.size < batch_size:
            return

        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)
        
        # source domain data with CQL loss and target domain data with SAC loss

        tar_q_loss_step = self.update_q_functions(tar_state, tar_action, tar_reward, tar_next_state, tar_not_done, writer)
        src_q_loss_step = self.update_cql_q_functions(src_state, src_action, src_reward, src_next_state, src_not_done, writer)

        q_loss = tar_q_loss_step + src_q_loss_step

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.update_target()

        # update policy and temperature parameter
        for p in self.q_funcs.parameters():
            p.requires_grad = False
        
        state = torch.cat([src_state, tar_state], 0)
        pi_loss_step, a_loss_step = self.update_policy_and_temp(state)
        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()

        if self.config['temperature_opt']:
            self.temp_optimizer.zero_grad()
            a_loss_step.backward()
            self.temp_optimizer.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    @property
    def alpha_prime(self):
        return self.log_alpha_prime.exp()

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
