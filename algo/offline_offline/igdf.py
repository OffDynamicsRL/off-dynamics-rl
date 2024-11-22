import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.distributions.transforms import Transform

from typing import Dict, List, Tuple, Union, Optional, Type
from functools import partial
from tqdm import trange



def weight_init(m: nn.Module, gain: int = 1) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    if isinstance(m, LinearEnsemble):
        for i in range(m.ensemble_size):
            # Orthogonal initialization doesn't care about which axis is first
            # Thus, we can just use ortho init as normal on each matrix.
            nn.init.orthogonal_(m.weight.data[i], gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class LinearEnsemble(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int = 3,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        An Ensemble linear layer.
        For inputs of shape (B, H) will return (E, B, H) where E is the ensemble size
        See https://github.com/pytorch/pytorch/issues/54147
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty((ensemble_size, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((ensemble_size, 1, out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The default torch init for Linear is a complete mess
        # https://github.com/pytorch/pytorch/issues/57109
        # If we use the same init, we will end up scaling incorrectly
        # 1. Compute the fan in of the 2D tensor = dim 1 of 2D matrix (0 index)
        # 2. Comptue the gain with param=math.sqrt(5.0)
        #   This returns math.sqrt(2.0 / 6.0) = sqrt(1/3)
        # 3. Compute std = gain / math.sqrt(fan) = sqrt(1/3) / sqrt(in).
        # 4. Compute bound as math.sqrt(3.0) * std = 1 / in di
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 2:
            input = input.repeat(self.ensemble_size, 1, 1)
        elif len(input.shape) > 3:
            raise ValueError("LinearEnsemble layer does not support inputs with more than 3 dimensions.")
        return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "ensemble_size={}, in_features={}, out_features={}, bias={}".format(
            self.ensemble_size, self.in_features, self.out_features, self.bias is not None
        )


class LayerNormEnsemble(nn.Module):
    """
    This is a re-implementation of the Pytorch nn.LayerNorm module with suport for the Ensemble dim.
    We need this custom class since we need to normalize over normalize dims, but have multiple weight/bais
    parameters for the ensemble.

    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: int,
        ensemble_size: int = 3,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert isinstance(normalized_shape, int), "Currently EnsembleLayerNorm only supports final dim int shapes."
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.ensemble_size = ensemble_size
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty((self.ensemble_size, 1) + self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty((self.ensemble_size, 1) + self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.repeat(self.ensemble_size, 1, 1)
        elif len(x.shape) > 3:
            raise ValueError("LayerNormEnsemble layer does not support inputs with more than 3 dimensions.")
        x = F.layer_norm(x, self.normalized_shape, None, None, self.eps)  # (E, B, *normalized shape)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


class EnsembleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int = 3,
        hidden_layers: List[int] = [256, 256],
        act: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        output_act: Optional[Type[nn.Module]] = None,
    ):
        """
        An ensemble MLP
        Returns values of shape (E, B, H) from input (B, H)
        """
        super().__init__()
        # Change the normalization type to work over ensembles
        assert normalization is None or normalization is LayerNormEnsemble, "Ensemble only support EnsembleLayerNorm"
        net = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(LinearEnsemble(last_dim, dim, ensemble_size=ensemble_size))
            if dropout > 0.0:
                net.append(nn.Dropout(dropout))
            if normalization is not None:
                net.append(normalization(dim, ensemble_size=ensemble_size))
            net.append(act())
            last_dim = dim
        net.append(LinearEnsemble(last_dim, output_dim, ensemble_size=ensemble_size))
        if output_act is not None:
            net.append(output_act())
        self.net = nn.Sequential(*net)
        self._has_output_act = False if output_act is None else True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def last_layer(self) -> torch.Tensor:
        if self._has_output_act:
            return self.net[-2]
        else:
            return self.net[-1]



class ContrastiveInfo(nn.Module):
    def __init__(
        self, 
        state_dim:          int,
        action_dim:         int,
        repr_dim:           int,
        ensemble_size:      int = 2,
        repr_norm:          bool = False,
        repr_norm_temp:     bool = True,
        ortho_init:         bool = False,
        output_gain:        Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.state_dim          = state_dim
        self.action_dim         = action_dim        
        self.repr_dim           = repr_dim
        self.ensemble_size      = ensemble_size
        self.repr_norm          = repr_norm
        self.repr_norm_temp     = repr_norm_temp
        
        input_dim_for_sa        = self.state_dim + self.action_dim
        input_dim_for_ss        = self.state_dim

        if self.ensemble_size > 1:
            self.encoder_sa     = EnsembleMLP(input_dim_for_sa, repr_dim, ensemble_size=ensemble_size, **kwargs)
            self.encoder_ss     = EnsembleMLP(input_dim_for_ss, repr_dim, ensemble_size=ensemble_size, **kwargs)
        else:
            self.encoder_sa     = MLPNetwork(input_dim_for_sa, repr_dim, **kwargs)
            self.encoder_ss     = MLPNetwork(input_dim_for_ss, repr_dim, **kwargs)

        self.ortho_init    = ortho_init
        self.output_gain   = output_gain
        self.register_parameter()

    def register_parameter(self) -> None:
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))
    
    def encode(self, obs: torch.Tensor, action: torch.Tensor, ss: torch.Tensor) -> torch.Tensor:
        sa_repr      = self.encoder_sa(torch.cat([obs, action], dim=-1))
        ss_repr      = self.encoder_ss(ss)
        if self.repr_norm:
            sa_repr     =   sa_repr / torch.linalg.norm(sa_repr, dim=-1, keepdim=True)
            ss_repr     =   ss_repr / torch.linalg.norm(ss_repr, dim=-1, keepdim=True)
            if self.repr_norm_temp:
                raise NotImplementedError("The Running normalization is not implemented")
        return sa_repr, ss_repr

    def combine_repr(self, sa_repr: torch.Tensor, ss_repr: torch.Tensor) -> torch.Tensor:
        if len(sa_repr.shape) ==2 and len(ss_repr.shape) ==2:
            return torch.einsum('iz,jz->ij', sa_repr, ss_repr)
        else:
            return torch.einsum('eiz,ejz->eij', sa_repr, ss_repr)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, ss: torch.Tensor, return_repr: bool = False) -> torch.Tensor:
        sa_repr, ss_repr = self.encode(obs, action, ss)    #   [E, B1, Z], [E, B2, Z]
        if return_repr:
            return self.combine_repr(sa_repr, ss_repr), sa_repr, ss_repr
        else:
            return self.combine_repr(sa_repr, ss_repr)           #   [E, B1, B2]


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
    
    def bc_loss(self, state, action):
        mu_logstd = self.network(state)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        pred_action = torch.tanh(mu)

        return (pred_action - action)**2

class DoubleQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)

class ValueFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ValueFunc, self).__init__()
        self.network = MLPNetwork(state_dim, 1, hidden_size)

    def forward(self, state):
        return self.network(state)

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class IGDF(object):

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

        # IQL hyperparameter
        self.lam = config['lam']
        self.temp = config['temp']
        
        self.total_it = 0

        # aka critic
        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka value
        self.v_func = ValueFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)

        # aka actor
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.v_optimizer = torch.optim.Adam(self.v_func.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])

        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, config['max_step'])


        self.info = ContrastiveInfo(config['state_dim'], config['action_dim'], config['repr_dim'], config['ensemble_size'],
                                    config['repr_norm'], config['repr_norm_temp'], config['ortho_init'], config['output_gain'],).to(self.device)
        self.info_optimizer = torch.optim.Adam(self.info.parameters(), lr=config['actor_lr'])

    
    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()
    
    def update_info(self, src_replay_buffer, tar_replay_buffer, batch_size, writer=None):

        info_step = 0

        for train_step in trange(self.config['info_update_step'], desc="Training"):
            info_step += 1
            tar_s, tar_a, tar_ss, _, _ = tar_replay_buffer.sample(batch_size) 
            _, _, src_ss, _, _ = src_replay_buffer.sample(batch_size - 1) # src_ss = [127, state_dim]
            
            tar_s = tar_s.unsqueeze(1) # [128, 1, state_dim]
            tar_a = tar_a.unsqueeze(1) # [128, 1, action_dim]
            tar_ss = tar_ss.unsqueeze(1) # [128, 1, state_dim]
            src_ss = src_ss.unsqueeze(0) # [1, 127, state_dim]
            src_ss = src_ss.expand(batch_size, -1, -1) # [128, 127, state_dim]
            ss = torch.cat((tar_ss, src_ss), dim = 1) # [128, 128, state_dim]

            logits = self.info(tar_s, tar_a, ss) # [128, 1, 128]
            logits = logits.squeeze(1)
            matrix = torch.zeros((batch_size, batch_size), dtype = torch.float32, device = self.device)
            matrix[:, 0] = 1
            
            info_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, matrix)
            info_loss = torch.mean(info_loss)

            if writer is not None and info_step % 100 == 0:
                writer.add_scalar('train/info loss', info_loss, info_step)

            self.info_optimizer.zero_grad()
            info_loss.backward()
            self.info_optimizer.step()

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_v_function(self, state_batch, action_batch, writer=None):
        with torch.no_grad():
            q_t1, q_t2 = self.target_q_funcs(state_batch, action_batch)
            q_t = torch.min(q_t1, q_t2)
            
        v = self.v_func(state_batch)
        adv = q_t - v
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/adv', adv.mean(), self.total_it)
            writer.add_scalar('train/value', v.mean(), self.total_it)
        v_loss = asymmetric_l2_loss(adv, self.lam)
        return v_loss, adv

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, mask, writer=None):
        with torch.no_grad():
            v_t = self.v_func(nextstate_batch)
            value_target = reward_batch + not_done_batch * self.discount * v_t
            
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
        
        # from IGDF code, use mask for weighting Q loss
        loss = (mask * (q_1 - value_target)**2).mean() + (mask * (q_2 - value_target)**2).mean()
        return loss

    def update_policy(self, advantage_batch, state_batch, action_batch):
        exp_adv = torch.exp(self.temp * advantage_batch.detach()).clamp(max=100.0)
        bc_loss = self.policy.bc_loss(state_batch, action_batch)
        policy_loss = torch.mean(exp_adv * bc_loss)
        return policy_loss

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):

        self.total_it += 1

        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        # perform data filtering
        if self.config['repr_norm']:
            logits = self.info(src_state, src_action, src_next_state)
            diagonal_elements = torch.diag(logits).reshape(-1, 1)
            src_info = diagonal_elements
        else:
            logits, srcsa_repr, srcss_repr = self.info(src_state, src_action, src_next_state, return_repr = True)
            srcsa_repr = torch.linalg.norm(srcsa_repr, dim=-1, keepdim=True)  # [128, 1]
            srcss_repr = torch.linalg.norm(srcss_repr, dim=-1, keepdim=True)  # [128, 1]
            diagonal_elements = torch.diag(logits).reshape(-1, 1)
            src_info = diagonal_elements / (srcsa_repr * srcss_repr) # [128, 1]
        sorted_indices = torch.argsort(src_info[:, 0])

        sorted_num = - int(batch_size * float(self.config['xi']))
        top_half_indices = sorted_indices[sorted_num:]
        src_state = src_state[top_half_indices]
        src_action = src_action[top_half_indices]
        src_next_state = src_next_state[top_half_indices]
        src_reward = src_reward[top_half_indices]
        src_not_done = src_not_done[top_half_indices]

        info_temp = torch.exp(src_info[top_half_indices] * self.config['importance_weight'])

        mask = torch.ones((batch_size - sorted_num, 1)).to(self.device)
        mask[:-sorted_num] = info_temp

        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        reward = torch.cat([src_reward, tar_reward], 0)
        not_done = torch.cat([src_not_done, tar_not_done], 0)

        v_loss_step, adv = self.update_v_function(state, action, writer)
        self.v_optimizer.zero_grad()
        v_loss_step.backward()
        self.v_optimizer.step()

        q_loss_step = self.update_q_functions(state, action, reward, next_state, not_done, mask, writer)

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        # update policy and temperature parameter
        for p in self.q_funcs.parameters():
            p.requires_grad = False
        pi_loss_step = self.update_policy(adv, state, action)
        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.v_func.state_dict(), filename + "_value")
        torch.save(self.v_optimizer.state_dict(), filename + "_value_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.policy_lr_schedule.state_dict(), filename + "_actor_lr_scheduler")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.v_func.load_state_dict(torch.load(file_name + "_value"))
        self.v_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.policy_lr_schedule.load_state_dict(torch.load(filename + "_actor_lr_scheduler"))