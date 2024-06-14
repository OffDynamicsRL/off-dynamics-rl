import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from typing import Callable, Optional
from torch.nn.modules.dropout import Dropout

from torch.distributions.transforms import Transform

# RLPD, two key contributions: layer normalization, and update-to-data ratio

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

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_sizes,
        n_layers,
        activations: Callable = nn.ReLU,
        activate_final: int = False,
        layernorm: int = False,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()

        self.affines = []
        self.affines.append(nn.Linear(input_dim, hidden_sizes))
        assert n_layers >= 2
        for i in range(n_layers-2):
            self.affines.append(nn.Linear(hidden_sizes, hidden_sizes))
        self.affines.append(nn.Linear(hidden_sizes, output_dim))
        self.affines = nn.ModuleList(self.affines)

        self.activations = activations()
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)
        # layernorm
        self.use_layernorm = layernorm
        self.layernorm = nn.LayerNorm(hidden_sizes)

    def forward(self, x):
        for i in range(len(self.affines)):
            x = self.affines[i](x)
            if i != len(self.affines)-1 or self.activate_final:
                if self.use_layernorm:
                    x = self.layernorm(x)
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = self.dropout(x)
        return x


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hidden_size=256, n_layers=2, layernorm=False):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size, n_layers=n_layers, layernorm=layernorm)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
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

class EnsembleQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256, n_layers=2, layernorm=True, num_q=10):
        super(EnsembleQFunc, self).__init__()
        self.nets = []
        self.num_q = num_q
        for i in range(self.num_q):
            net = MLPNetwork(state_dim + action_dim, 1, hidden_size, n_layers=n_layers, layernorm=layernorm)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        return torch.stack(tuple(net(sa) for net in self.nets), dim=1)


class RLPD(object):

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
        self.q_funcs = EnsembleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes'], n_layers=config['n_layers'], layernorm=True, num_q=config['num_q']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes'], n_layers=config['n_layers'], layernorm=False).to(self.device)

        # aka temperature
        if config['temperature_opt']:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([0.2])).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
    
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
            # sample idx, num_backup_q differs across different domains, e.g., in antmaze, it is 1, while in mujoco, it is 2 for CDQ
            sample_idxs = np.random.choice(self.config['num_q'], self.config['num_backup_q'], replace=False)
            q_t = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            qt_list = []
            for i in range(len(sample_idxs)):
                qt_list.append(q_t[:,sample_idxs[i]])
            q_t_sample = torch.cat(qt_list, dim=1)
            q_target = q_t_sample.min(dim=1, keepdim=True)[0]
            if self.config['entropy_backup']:
                value_target = reward_batch + not_done_batch * self.discount * (q_target - self.alpha * logprobs_batch)
            else:
                value_target = reward_batch + not_done_batch * self.discount * q_target
            # value_target = reward_batch + not_done_batch * self.discount * q_target
        q = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q', q.mean(), self.total_it)
            writer.add_scalar('train/logprob', logprobs_batch.mean(), self.total_it)
        value_target = value_target.expand(-1, self.config['num_q'])
        loss = F.mse_loss(q.squeeze(), value_target)
        return loss

    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q = self.q_funcs(state_batch, action_batch)
        # RLPD does not randomly select Q for actor update and instead use Q mean to update
        qval_batch = q.mean()
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):

        self.total_it += 1

        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            return

        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        reward = torch.cat([src_reward, tar_reward], 0)
        not_done = torch.cat([src_not_done, tar_not_done], 0)

        q_loss_step = self.update_q_functions(state, action, reward, next_state, not_done, writer)

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        # update policy and temperature parameter
        for p in self.q_funcs.parameters():
            p.requires_grad = False
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