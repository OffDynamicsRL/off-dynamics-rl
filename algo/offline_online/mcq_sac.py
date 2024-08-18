import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints, kl_divergence

from torch.distributions.transforms import Transform

def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

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

# MCQ relies on VAE for producing pseudo-targets
class VAE(torch.nn.Module):
    def __init__(self, state_dim, action_dim, vae_features, max_action=1.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = 2 * self.action_dim
        self.max_action = max_action

        self.encoder = MLPNetwork(self.state_dim + self.action_dim, 2 * self.latent_dim, vae_features)
        self.decoder = MLPNetwork(self.state_dim + self.latent_dim, self.action_dim, vae_features)
        self.noise = MLPNetwork(self.state_dim + self.action_dim, self.action_dim, vae_features)

    def encode(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        mu, logstd = torch.chunk(self.encoder(state_action), 2, dim=-1)
        logstd = torch.clamp(logstd, -4, 15)
        std = torch.exp(logstd)
        return Normal(mu, std)

    def decode(self, state, z=None):
        if z is None:
            param = next(self.parameters())
            z = torch.randn((*state.shape[:-1], self.latent_dim)).to(param)
            z = torch.clamp(z, -0.5, 0.5)

        action = self.decoder(torch.cat([state, z], dim=-1))
        action = self.max_action * torch.tanh(action)

        return action
    
    def decode_multiple(self, state, z=None, num=10):
        if z is None:
            param = next(self.parameters())
            z = torch.randn((num, *state.shape[:-1], self.latent_dim)).to(param)
            z = torch.clamp(z, -0.5, 0.5)
        state = state.repeat((num,1,1))

        action = self.decoder(torch.cat([state, z], dim=-1))
        action = self.max_action * torch.tanh(action)
        # shape: (num, batch size, state shape+action shape)
        return action

    def forward(self, state, action):
        dist = self.encode(state, action)
        z = dist.rsample()
        action = self.decode(state, z)
        return dist, action

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
        if len(state.shape) == 3:
            x = torch.cat((state, action), dim=2)
        else:
            x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class MCQSAC(object):

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

        # MCQ hyperparameter, lambda for balancing in-dis q and ood q
        self.lam = config['lam']
        # MCQ hyperparameter, number of sampled actions
        self.num_sample_action = config['num_sample_action']

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

        # aka vae
        self.vae = VAE(config['state_dim'], config['action_dim'], config['vae_features'], config['max_action']).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=config['actor_lr'])
    
    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def _get_tensor_values(self, state_batch, vae=None, network=None):
        batch_size = state_batch.shape[0]
        state_batch_repeat = state_batch.repeat((self.num_sample_action,1,1)).reshape(-1, state_batch.shape[-1])
        if vae is None:
            repeat_actions, _, _ = self.policy(state_batch_repeat)
            pred_q1, pred_q2 = network(state_batch_repeat, repeat_actions)
        else:
            repeat_actions = vae.decode_multiple(state_batch, num=self.num_sample_action)
            repeat_actions = repeat_actions.reshape(self.num_sample_action*batch_size, -1)
            pred_q1, pred_q2 = network(state_batch_repeat, repeat_actions)
            pred_q1 = pred_q1.reshape(self.num_sample_action, batch_size, 1)
            pred_q2 = pred_q2.reshape(self.num_sample_action, batch_size, 1)
            pred_q1 = torch.max(pred_q1, dim=0)[0]
            pred_q2 = torch.max(pred_q2, dim=0)[0]
            pred_q1 = pred_q1.clamp(min=0).repeat((self.num_sample_action,1,1)).reshape(-1,1)
            pred_q2 = pred_q2.clamp(min=0).repeat((self.num_sample_action,1,1)).reshape(-1,1)
        return pred_q1, pred_q2, repeat_actions.view(self.num_sample_action, batch_size, -1)

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
            if self.config['entropy_backup']:
                value_target = reward_batch + not_done_batch * self.discount * (q_target - self.alpha * logprobs_batch)
            else:
                value_target = reward_batch + not_done_batch * self.discount * q_target
            # value_target = reward_batch + not_done_batch * self.discount * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
            writer.add_scalar('train/logprob', logprobs_batch.mean(), self.total_it)
        loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)
        return loss
    
    def update_mcq_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + not_done_batch * self.discount * (q_target - self.alpha * logprobs_batch)
        # current q
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        
        ## OOD Q1
        q1_ood_curr_pred, q2_ood_curr_pred, q_ood_curr_act = self._get_tensor_values(state_batch, network=self.q_funcs)
        q1_ood_next_pred, q2_ood_next_pred, q_ood_next_act = self._get_tensor_values(nextstate_batch, network=self.q_funcs)
        q1_ood_pred = torch.cat([q1_ood_curr_pred, q1_ood_next_pred],dim=0)
        q2_ood_pred = torch.cat([q2_ood_curr_pred, q2_ood_next_pred],dim=0)

        pesudo_q1_curr_target, pesudo_q2_curr_target, q_curr_act = self._get_tensor_values(state_batch, network=self.q_funcs, vae=self.vae)
        pesudo_q1_next_target, pesudo_q2_next_target, q_next_act = self._get_tensor_values(nextstate_batch, network=self.q_funcs, vae=self.vae)
        pesudo_q1_target = torch.cat([pesudo_q1_curr_target, pesudo_q1_next_target],dim=0)
        pesudo_q2_target = torch.cat([pesudo_q2_curr_target, pesudo_q2_next_target],dim=0)
        
        pesudo_q1_target = pesudo_q1_target.detach()
        pesudo_q2_target = pesudo_q2_target.detach()

        assert pesudo_q1_target.shape[0] == q1_ood_pred.shape[0]
        assert pesudo_q2_target.shape[0] == q2_ood_pred.shape[0]

        # stop vanilla pessimistic estimate becoming large
        pesudo_q_target = torch.min(pesudo_q1_target, pesudo_q2_target)
        qf1_deviation = q1_ood_pred - pesudo_q_target
        qf2_deviation = q2_ood_pred - pesudo_q_target
        qf1_deviation[qf1_deviation <= 0] = 0
        qf2_deviation[qf2_deviation <= 0] = 0

        qf1_ood_loss = torch.mean(qf1_deviation**2)
        qf2_ood_loss = torch.mean(qf2_deviation**2)

        qf1_loss = self.lam * F.mse_loss(q_1, value_target) + (1-self.lam) * qf1_ood_loss
        qf2_loss = self.lam * F.mse_loss(q_2, value_target) + (1-self.lam) * qf2_ood_loss

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/mcq q1', q_1.mean(), self.total_it)

        mcq_loss = qf1_loss + qf2_loss

        return mcq_loss

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

        # update VAE
        dist, _action = self.vae(tar_state, tar_action)
        kl_loss = kl_divergence(dist, Normal(0, 1)).sum(dim=-1).mean()
        recon_loss = ((tar_action - _action) ** 2).sum(dim=-1).mean()
        vae_loss = kl_loss + recon_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        
        # source domain data with MCQ loss and target domain data with SAC loss

        tar_q_loss_step = self.update_q_functions(tar_state, tar_action, tar_reward, tar_next_state, tar_not_done, writer)
        src_q_loss_step = self.update_mcq_q_functions(src_state, src_action, src_reward, src_next_state, src_not_done, writer)

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
