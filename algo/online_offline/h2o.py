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

# domain classifier for H2O, akin to DARC
class Classifier(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256, gaussian_noise_std=1.0):
        super(Classifier, self).__init__()
        self.action_dim = action_dim
        self.gaussian_noise_std = gaussian_noise_std
        self.sa_classifier = MLPNetwork(state_dim + action_dim, 2, hidden_size)
        self.sas_classifier = MLPNetwork(2*state_dim + action_dim, 2, hidden_size)

    def forward(self, state_batch, action_batch, nextstate_batch, with_noise):
        sas = torch.cat([state_batch, action_batch, nextstate_batch], -1)

        if with_noise:
            sas += torch.randn_like(sas, device=state_batch.device) * self.gaussian_noise_std
        sas_logits = torch.nn.Softmax()(self.sas_classifier(sas))

        sa = torch.cat([state_batch, action_batch], -1)

        if with_noise:
            sa += torch.randn_like(sa, device=state_batch.device) * self.gaussian_noise_std
        sa_logits = torch.nn.Softmax()(self.sa_classifier(sa))

        return sas_logits, sa_logits


class H2O(object):

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
        
        # aka classifier
        self.classifier = Classifier(config['state_dim'], config['action_dim'], config['hidden_sizes'], config['gaussian_noise_std']).to(self.device)
        
        # aka cql temperature
        self.log_alpha_prime = torch.log(torch.FloatTensor([1.0])).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
        self.temp_prime_optimizer = torch.optim.Adam([self.log_alpha_prime], lr=config['actor_lr'])
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['actor_lr'])
    
    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()
    
    def update_classifier(self, src_replay_buffer, tar_replay_buffer, batch_size, writer=None):
        src_state, src_action, src_next_state, _, _ = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, _, _ = tar_replay_buffer.sample(batch_size)

        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)

        # set labels for different domains
        label = torch.cat([torch.zeros(size=(batch_size,)), torch.ones(size=(batch_size,))], dim=0).long().to(self.device)

        indexs = torch.randperm(label.shape[0])
        state_batch, action_batch, nextstate_batch = state[indexs], action[indexs], next_state[indexs]
        label = label[indexs]

        sas_logits, sa_logits = self.classifier(state_batch, action_batch, nextstate_batch, with_noise=True)
        loss_sas = F.cross_entropy(sas_logits, label)
        loss_sa =  F.cross_entropy(sa_logits, label)
        classifier_loss = loss_sas + loss_sa
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

        # log necessary information if the logger is not None
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/sas classifier loss', loss_sas, global_step=self.total_it)
            writer.add_scalar('train/sa classifier loss', loss_sa, global_step=self.total_it)

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
            value_target = reward_batch + not_done_batch * self.discount * (q_target.unsqueeze(dim=1) - self.alpha * logprobs_batch)
            # value_target = reward_batch + not_done_batch * self.discount * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
            writer.add_scalar('train/logprob', logprobs_batch.mean(), self.total_it)
        loss = F.mse_loss(q_1.unsqueeze(dim=1), value_target) + F.mse_loss(q_2.unsqueeze(dim=1), value_target)
        return loss

    def update_cql_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch,
                                     tar_state_batch, tar_action_batch, writer=None):
        # calculate importance sampling coefficient (IS coef) in H2O
        with torch.no_grad():
            sas_logits, sa_logits = self.classifier(state_batch, action_batch, nextstate_batch, with_noise=False)
            sas_probs, sa_probs = F.softmax(sas_logits, -1), F.softmax(sa_logits, -1)
            sas_log_probs, sa_log_probs = torch.log(sas_probs + 1e-10), torch.log(sa_probs + 1e-10)
            log_importance_weighting = sas_log_probs[:, 1:] - sa_log_probs[:, 1:] - sas_log_probs[:, :1] + sa_log_probs[:,:1]
            importance_weighting = torch.exp(log_importance_weighting)
            # clip to make the training more stable
            importance_weighting  = torch.clip(importance_weighting, min=1e-5, max=1.)

            if writer is not None and self.total_it % 5000 == 0:
                writer.add_scalar('train/importance sampling weighting', importance_weighting.mean(), global_step=self.total_it)

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
            q_target = q_target.unsqueeze(dim=1)
            if self.config['backup_entropy']:
                q_target = q_target - self.alpha * logprobs_batch
            value_target = reward_batch + not_done_batch * self.discount * q_target
            
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss = (importance_weighting * (q_1.unsqueeze(dim=1) - value_target)**2).mean() + (importance_weighting * (q_2.unsqueeze(dim=1) - value_target)**2).mean()

        # add CQL loss of H2O
        action_dim = action_batch.shape[-1]

        cql_q1, cql_q2 = self.q_funcs(state_batch, action_batch)
        tar_q1, tar_q2 = self.q_funcs(tar_state_batch, tar_action_batch)
        # measure the weights
        obs = torch.repeat_interleave(state_batch, self.config['n_state'], dim=0)
        act = torch.repeat_interleave(action_batch, self.config['n_state'], dim=0)
        next_obs = torch.repeat_interleave(nextstate_batch, self.config['n_state'], dim=0)
        # use batch target state for measuring std, can also use the whole dataset
        noise_std = tar_state_batch.std(0,keepdims=False) + 1e-3
        next_obs += torch.randn(next_obs.shape, device=self.device) * noise_std
        # measuring the log importance weighting
        with torch.no_grad():
            sas_logits, sa_logits = self.classifier(obs, act, next_obs, with_noise=False)
            sas_probs, sa_probs = F.softmax(sas_logits, -1), F.softmax(sa_logits, -1)
            sas_log_probs, sa_log_probs = torch.log(sas_probs + 1e-10), torch.log(sa_probs + 1e-10)
            log_importance_weighting = sas_log_probs[:, 1:] - sa_log_probs[:, 1:] - sas_log_probs[:, :1] + sa_log_probs[:,:1]
            log_importance_weighting = log_importance_weighting.reshape((-1, self.config['n_state']))
            u_sa = torch.clamp(log_importance_weighting.mean(dim=1), 1e-45, 10) # from H2O repo
        omega = u_sa / u_sa.sum()

        cql_q1 += torch.log(omega)
        cql_q2 += torch.log(omega)
        cql_q1_gap = torch.logsumexp(cql_q1 / self.config['cql_temp'], dim=0) * self.config['cql_temp']
        cql_q2_gap = torch.logsumexp(cql_q2 / self.config['cql_temp'], dim=0) * self.config['cql_temp']

        # clip
        cql_qf1_diff = torch.clamp(cql_q1_gap - tar_q1.mean(), -1000, 1000)
        cql_qf2_diff = torch.clamp(cql_q2_gap - tar_q2.mean(), -1000, 1000)

        if self.config['cql_lagrange']:
            alpha_prime = torch.clamp(self.alpha_prime, min=0.0, max=1000000.0)
            cql_min_qf1_loss = (alpha_prime * self.config['cql_alpha'] * (cql_qf1_diff - self.config['cql_target_action_gap'])).mean()
            cql_min_qf2_loss = (alpha_prime * self.config['cql_alpha'] * (cql_qf2_diff - self.config['cql_target_action_gap'])).mean()
            self.temp_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.temp_prime_optimizer.step()
        else:
            cql_min_qf1_loss = (cql_qf1_diff * self.config['cql_alpha']).mean()
            cql_min_qf2_loss = (cql_qf2_diff * self.config['cql_alpha']).mean()
            alpha_prime_loss = state_batch.new_tensor(0.0)
            alpha_prime = state_batch.new_tensor(0.0)

        loss += cql_min_qf1_loss + cql_min_qf2_loss

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
        
        if self.total_it % self.config['tar_env_interact_freq'] == 0:
            self.update_classifier(src_replay_buffer, tar_replay_buffer, batch_size, writer)

        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        tar_q_loss_step = self.update_q_functions(tar_state, tar_action, tar_reward, tar_next_state, tar_not_done, writer)
        # this is different from H2O under offline-online setting
        src_q_loss_step = self.update_cql_q_functions(src_state, src_action, src_reward, src_next_state, src_not_done,
                                                      tar_state, tar_action,  writer)

        q_loss_step = tar_q_loss_step + src_q_loss_step

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
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
