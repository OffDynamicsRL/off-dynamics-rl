from typing import List, Dict, Callable, List, Dict, Tuple, Union
import numpy as np
import itertools
import random
from collections    import deque
from operator       import itemgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


def call_env_terminal_func(env_name: str) -> Callable:
    def is_terminal_region_for_hp(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
        if isinstance(state, np.ndarray):
            height = state[..., 0:1]
            angle = state[..., 1:2]
            not_done = np.isfinite(state).all(axis=-1)[..., np.newaxis] \
                        & np.abs(state[..., 1:] < 100).all(axis=-1)[..., np.newaxis] \
                        & (height > .7) \
                        & (np.abs(angle) < .2)
        elif isinstance(state, torch.Tensor):
            height = state[..., 0:1]
            angle = state[..., 1:2]
            not_done = torch.isfinite(state).all(dim=-1, keepdim=True) \
                        & torch.abs(state[..., 1:] < 100).all(dim=-1, keepdim=True) \
                        & (height > .7) \
                        & (torch.abs(angle) < .2)
        else:
            raise ValueError
        if return_done:
            done = ~not_done
            return done
        else:
            return not_done

    def is_terminal_region_for_hc(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
        if isinstance(state, np.ndarray):
            not_done = np.ones((*state.shape[:-1], 1), dtype=np.bool8)
        elif isinstance(state, torch.Tensor):
            not_done = torch.ones((*state.shape[:-1], 1)).bool()
            not_done = not_done.to(state.device)
        else:
            raise ValueError
            
        if return_done:
            done = ~not_done
            return done
        else:
            return not_done

    def is_terminal_region_for_at(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
        if isinstance(state, np.ndarray):
            is_finite = np.isfinite(state).all(-1)[..., np.newaxis]
            is_healthy= (0.2 <= state[..., 0:1]) & (state[..., 0:1] <= 1.0)     # height - z
            not_done = is_finite & is_healthy
        elif isinstance(state, torch.Tensor):
            is_finite = torch.isfinite(state).all(-1, keepdim=True)
            is_healthy= (0.2 <= state[..., 0:1]) & (state[..., 0:1] <= 1.0)
            not_done = is_finite & is_healthy
        else:
            raise ValueError
            
        if return_done:
            done = ~not_done
            return done
        else:
            return not_done

    def is_terminal_region_for_wk(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
        if isinstance(state, np.ndarray):
            height = state[..., 0:1]
            angle = state[..., 1:2]
            not_done = (0.8 < height) & (height < 2.0) & (-1.0 < angle) & (angle < 1.0)
        elif isinstance(state, torch.Tensor):
            height = state[..., 0:1]
            angle = state[..., 1:2]
            not_done = (0.8 < height) & (height < 2.0) & (-1.0 < angle) & (angle < 1.0)
        else:
            raise ValueError
            
        if return_done:
            done = ~not_done
            return done
        else:
            return not_done
    
    def is_terminal_region_for_others(state: Union[np.ndarray, torch.Tensor], return_done: bool) -> Union[np.array, torch.tensor]:
        not_done = True
        return not_done
    
    env_name = env_name.lower()
        
    if "hopper" in env_name:        
        return is_terminal_region_for_hp
    elif "halfcheetah" in env_name:
        return is_terminal_region_for_hc
    elif "ant" in env_name:
        return is_terminal_region_for_at
    elif "walker" in env_name:
        return is_terminal_region_for_wk
    else:
        return is_terminal_region_for_others




def init_weight(layer, initializer="he normal"):
    if isinstance(layer, nn.Module):
        if initializer == "xavier uniform":
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif initializer == 'xavier normal':
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif initializer == "he normal":
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif initializer == 'orthogonal':
            nn.init.orthogonal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif initializer == 'truncated normal':
            nn.init.trunc_normal_(layer.weight, std=1/(2*np.sqrt(layer.weight.shape[1])))
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Parameter):
        if initializer == "xavier uniform":
            nn.init.xavier_uniform_(layer)
        elif initializer == 'xavier normal':
            nn.init.xavier_normal_(layer)
        elif initializer == "he normal":
            nn.init.kaiming_normal_(layer)
        elif initializer == 'orthogonal':
            nn.init.orthogonal_(layer)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def call_activation(name: str) -> nn.Module:
    if name == 'Identity':
        return nn.Identity
    elif name == 'ReLU':
        return nn.ReLU
    elif name == 'Tanh':
        return nn.Tanh
    elif name == 'Sigmoid':
        return nn.Sigmoid
    elif name == 'SoftMax':
        return nn.Softmax
    elif name == 'ELU':
        return nn.ELU
    elif name == 'LeakyReLU':
        return nn.LeakyReLU
    elif name == 'Swish':
        return Swish
    else:
        raise NotImplementedError(f"Invalid activation name: {name}")


def call_mlp(
    in_dim: int, 
    out_dim: int, 
    hidden_layers: List[int],
    inner_activation: str = 'ReLU',
    output_activation: str = 'Tanh',
    initializer: str = 'he normal',
    layer_factory: Callable = None,
) -> nn.Module:
    module_seq = []
    InterActivation = call_activation(inner_activation)
    OutActivation   = call_activation(output_activation)
    
    if not layer_factory:
        factory = nn.Linear
    else:
        factory = layer_factory

    last_dim = in_dim
    for hidden in hidden_layers:
        linear = factory(last_dim, hidden)
        init_weight(linear, initializer)
        module_seq += [linear, InterActivation()]
        last_dim = hidden

    linear = factory(last_dim, out_dim)
    init_weight(linear)
    module_seq += [linear, OutActivation()]

    return nn.Sequential(*module_seq)


class Module(nn.Module):
    def save(self, f: str, prefix: str = '', keep_vars: bool = False) -> None:
        state_dict = self.state_dict(prefix= prefix, keep_vars=keep_vars)
        torch.save(state_dict, f)

    def load(self, f: str, map_location, strict: bool = True) -> None:
        state_dict = torch.load(f, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)


class QFunction(Module):
    def __init__(
        self, 
        s_dim: int, 
        a_dim: int, 
        hidden_layers: List[int], 
        inner_nonlinear: str, 
        initializer: str
    ) -> None:
        super(QFunction, self).__init__()
        self.s_dim, self.a_dim = s_dim, a_dim
        self._model = call_mlp(
            in_dim              =   s_dim + a_dim,
            out_dim             =   1,
            hidden_layers       =   hidden_layers,
            inner_activation    =   inner_nonlinear,
            output_activation   =   'Identity',
            initializer         =   initializer
        )
    
    def forward(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        return self._model(torch.cat([state, action], dim=-1))


class QEnsemble(Module):
    def __init__(
        self,
        ensemble_size: int,
        s_dim: int,
        a_dim: int,
        hiddens: List[int],
        inner_nonlinear: str,
        initializer: str
    ) -> None:
        super().__init__()
        self.ensemble_size  = ensemble_size
        self.s_dim          = s_dim 
        self.a_dim          = a_dim
        self.ensemble       = nn.ModuleList(
            [QFunction(s_dim, a_dim, hiddens, inner_nonlinear, initializer) for _ in range(ensemble_size)]
        )

    def forward(self, state: torch.tensor, action: torch.tensor) -> Tuple:
        all_q_values = [qfunction(state, action) for qfunction in self.ensemble]
        return tuple(all_q_values)

    def forward_single(self, state: torch.tensor, action: torch.tensor, index: int) -> torch.tensor:
        return self.ensemble[index](state, action)
    

class SquashedGaussianPolicy(Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        hidden_layers: List[int],
        inner_nonlinear: str,
        log_std_min: float,
        log_std_max: float,
        initializer: str
    ) -> None:
        super().__init__()
        self.s_dim, self.a_dim = s_dim, a_dim
        self._model = call_mlp(
            s_dim,
            a_dim * 2,
            hidden_layers,
            inner_nonlinear,
            'Identity',
            initializer
        )
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def sample_action(self, state: torch.tensor, with_noise: bool) -> torch.tensor:
        with torch.no_grad():
            mix = self._model(state)
            mean, log_std = torch.chunk(mix, 2, dim=-1)
        if with_noise:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            action = dist.sample()
        else:
            action = mean
        return torch.tanh(action)

    def forward(self, state: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.distributions.Distribution]:
        mix             =   self._model(state)
        mean, log_std   =   torch.chunk(mix, 2, dim=-1)
        log_std         =   torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std             =   torch.exp(log_std)

        dist                =   Normal(mean, std)
        arctanh_actions     =   dist.rsample()
        log_prob            =   dist.log_prob(arctanh_actions).sum(-1, keepdim=True)

        action              =   torch.tanh(arctanh_actions)
        squashed_correction =   torch.log(1 - action**2 + 1e-6).sum(-1, keepdim=True)
        log_prob            =   log_prob - squashed_correction

        return action, log_prob, dist
    


class Normalizer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim        = dim
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.zeros(dim))

    def fit(self, X: torch.tensor) -> None:
        assert len(X.shape)      == 2
        assert X.shape[1]   == self.dim
        device  =   self.mean.device
        self.mean.data.copy_(
            torch.mean(X, axis=0, keepdims=False)
        )
        self.std.data.copy_(
            torch.std(X, axis=0, keepdims=False)
        )
        self.std[self.std < 1e-12]  =   1.0

    def transform(self, x: Union[np.array, torch.tensor]) -> Union[np.array, torch.tensor]:
        if isinstance(x, np.ndarray):
            device  =   self.mean.device
            x       =   torch.from_numpy(x).float().to(device)
            return ((x - self.mean) / self.std).cpu().numpy()
        elif isinstance(x, torch.Tensor):
            return ((x - self.mean) / self.std)



def init_weights(m):
    def truncated_normal_init(
        t:  nn.Module, 
        mean:   float = 0.0, 
        std:    float = 0.01
    ):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)



class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        ensemble_size: int, 
        weight_decay: float = 0., 
        bias: bool = True
    ) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class EnsembleModel(nn.Module):
    def __init__(
        self, 
        state_size:     int, 
        action_size:    int, 
        reward_size:    int, 
        ensemble_size:  int, 
        hidden_size:    int   = 200, 
        learning_rate:  float = 1e-3, 
        use_decay:      bool  = False,
        device:         str   = 'cuda'
    ):
        super(EnsembleModel, self).__init__()
        self.device      = device
        self.hidden_size = hidden_size
        self.output_dim  = state_size + reward_size
        # trunk layers
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay
        # Add variance output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)
        # min / max log var bounds
        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # weight init
        self.apply(init_weights)
        self.swish = Swish()

    def forward(
        self, 
        x:           torch.tensor, 
        ret_log_var: bool = False
    ):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean    = nn5_output[:, :, :self.output_dim]
        logvar  = nn5_output[:, :, self.output_dim:]

        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def loss(
        self, 
        mean:           torch.tensor, 
        logvar:         torch.tensor, 
        labels:         torch.tensor, 
        inc_var_loss:   bool = True
    ):
        """
            mean, logvar: [ensemble_size,  batch_size, |S| + |A|]
            labels:       [ensemble_size,  batch_size, |S| + 1]
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        # Average over batch and dim, sum over ensembles.
        if inc_var_loss:
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel(Module):
    def __init__(
        self, 
        network_size:   int, 
        elite_size:     int, 
        state_size:     int, 
        action_size:    int, 
        reward_size:    int = 1, 
        hidden_size:    int = 200, 
        use_decay:      bool= False,
        device:         str = 'cuda',
    ):
        super(EnsembleDynamicsModel, self).__init__()
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay, device=device)
        self.scaler = Normalizer(dim=state_size + action_size)
        self.device = device

    def train(
        self, 
        inputs:                     torch.tensor, 
        labels:                     torch.tensor, 
        batch_size:                 int     = 256, 
        holdout_ratio:              float   = 0., 
        max_epochs_since_update:    int     = 5
    ):
        self._max_epochs_since_update   = max_epochs_since_update
        self._epochs_since_update       = 0
        self._state                     = {}
        self._snapshots                 = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout     = int(inputs.shape[0] * holdout_ratio)
        permutation     = torch.randperm(inputs.shape[0])
        inputs, labels  = inputs[permutation], labels[permutation]

        train_inputs, train_labels      = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels  = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs    = self.scaler.transform(train_inputs)
        holdout_inputs  = self.scaler.transform(holdout_inputs)

        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])
        # for log
        all_holdout_losses  =   []

        for epoch in itertools.count():
            train_idx = torch.vstack([torch.randperm(train_inputs.shape[0]) for _ in range(self.network_size)])
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: min(start_pos + batch_size, train_inputs.shape[0])]
                train_input = torch.from_numpy(train_inputs.cpu().numpy()[idx]).float().to(self.device)
                train_label = torch.from_numpy(train_labels.cpu().numpy()[idx]).float().to(self.device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar    = self.ensemble_model(holdout_inputs, ret_log_var = True)
                _, holdout_mse_losses           = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses              = holdout_mse_losses.detach().cpu().numpy()
                all_holdout_losses.append(np.mean(holdout_mse_losses))
                # rank and select the elite models
                sorted_loss_idx                 = np.argsort(holdout_mse_losses)
                self.elite_model_idxes          = sorted_loss_idx[:self.elite_size].tolist()
                # check if any model improves
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
                    
        self._track_head_loss(all_holdout_losses)

    def _save_best(
        self, 
        epoch:          int, 
        holdout_losses: torch.tensor    # [ensemble_size]
    ):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def _track_head_loss(
        self,
        holdout_losses:  List    # [<= max_train_epoch]
    )   -> None:
        self._current_mean_ensemble_losses = np.mean(holdout_losses)

    def predict(
        self,
        inputs:                 torch.Tensor, 
        batch_size:             float   = 1024, 
        factor_ensemble:        bool    = True
    ):
        if inputs.ndim == 2:
            B       = inputs.shape[0]
            inputs  = self.scaler.transform(inputs)
            ensemble_mean, ensemble_var = [], []
            for i in range(0, B, batch_size):
                input = inputs[i:min(i + batch_size, B)]
                b_mean, b_var = self.ensemble_model(
                    input[None, :, :].repeat([self.network_size, 1, 1]), 
                    ret_log_var=False
                )
                ensemble_mean.append(b_mean)
                ensemble_var.append(b_var)
            ensemble_mean   = torch.cat(ensemble_mean, dim=1)    # concat along the batch_size axis
            ensemble_var    = torch.cat(ensemble_var, dim=1)

            if factor_ensemble:
                return ensemble_mean, ensemble_var              # [ensemble_size, batch_size, |S|+1]
            else:
                mean    = torch.mean(ensemble_mean, dim=0)
                var     = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
                return mean, var
        elif inputs.ndim == 3:
            assert inputs.shape[0] == self.network_size
            B       = inputs.shape[1]
            inputs  = self.scaler.transform(inputs)
            ensemble_mean, ensemble_var = [], []
            for i in range(0, B, batch_size):
                input = inputs[:, i:min(i + batch_size, B), :]
                b_mean, b_var = self.ensemble_model(
                    input,
                    ret_log_var=False
                )
                ensemble_mean.append(b_mean)
                ensemble_var.append(b_var)
            ensemble_mean   = torch.cat(ensemble_mean, dim=1)    # concat along the batch_size axis
            ensemble_var    = torch.cat(ensemble_var, dim=1)

            if factor_ensemble:
                return ensemble_mean, ensemble_var              # [ensemble_size, batch_size, |S|+1]
            else:
                mean    = torch.mean(ensemble_mean, dim=0)
                var     = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
                return mean, var
        else:
            raise ValueError            


def soft_update(src_model: nn.Module, tar_model: nn.Module, tau: float) -> None:
    for param_src, param_tar in zip(src_model.parameters(), tar_model.parameters()):
        param_tar.data.copy_(tau * param_src.data + (1 - tau) * param_tar.data)


class VGDF:
    def __init__(self, 
                 config: Dict,
                 device,
                 target_entropy=None,
                 ) -> None:
        self.config             =       config
        self.model_config       =       config['model_config']
        self.s_dim              =       config['state_dim']
        self.a_dim              =       config['action_dim']
        self.device             =       config['device']

        self.lr                 =       config['lr']
        self.gamma              =       config['gamma']
        self.tau                =       config['tau']
        self.alpha              =       config['alpha']
        self.training_delay     =       config['training_delay']

        self.batch_size         =       config['batch_size']
        
        self.ac_gradient_clip   =       config['ac_gradient_clip']

        self.dynamics_batch_size                    =       config['dynamics_batch_size']
        self.dynamics_holdout_ratio                 =       config['dynamics_holdout_ratio']
        self.dynamics_max_epochs_since_update       =       config['dynamics_max_epochs_since_update']
        self.dynamics_train_freq                    =       config['dynamics_train_freq']
        self.max_epochs_since_update_decay_interval =       config['max_epochs_since_update_decay_interval']
        
        self.start_gate_src_sample          =       config['start_gate_src_sample']
        self.likelihood_gate_threshold      =       config['likelihood_gate_threshold']

        self.terminal_func                  =       call_env_terminal_func(config['env_name'])

        self.training_count =       0
        self.loss_log       =       {}

        # adaptive alpha
        if self.alpha is None:
            self.train_alpha    =   True
            self.log_alpha      =   torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha          =   torch.exp(self.log_alpha)
            self.target_entropy =   target_entropy if target_entropy is not None else - torch.tensor(config['model_config']['a_dim'], dtype=torch.float64)
            self.optimizer_alpha=   optim.Adam([self.log_alpha], lr= self.lr)
        else:
            self.train_alpha    =   False

        # policy
        self.policy         =       SquashedGaussianPolicy(
            s_dim           =   self.s_dim,
            a_dim           =   self.a_dim,
            hidden_layers   =   self.model_config['policy_hiddens'],
            inner_nonlinear =   self.model_config['policy_nonlinear'],
            log_std_min     =   self.model_config['policy_log_std_min'],
            log_std_max     =   self.model_config['policy_log_std_max'],
            initializer     =   self.model_config['policy_initializer']
        ).to(self.device)
        self.optimizer_policy   =   optim.Adam(self.policy.parameters(), self.lr)
        
        # optimistic policy
        self.optimistic_policy  =   SquashedGaussianPolicy(
            s_dim           =   self.s_dim,
            a_dim           =   self.a_dim,
            hidden_layers   =   self.model_config['policy_hiddens'],
            inner_nonlinear =   self.model_config['policy_nonlinear'],
            log_std_min     =   self.model_config['policy_log_std_min'],
            log_std_max     =   self.model_config['policy_log_std_max'],
            initializer     =   self.model_config['policy_initializer']
        ).to(self.device)
        self.optimizer_opt_policy   =   optim.Adam(self.optimistic_policy.parameters(), self.lr)

        # value functions
        self.QFunction      =       QEnsemble(
            ensemble_size   =   2,
            s_dim           =   self.s_dim,
            a_dim           =   self.a_dim,
            hiddens         =   self.model_config['value_hiddens'],
            inner_nonlinear =   self.model_config['value_nonlinear'],
            initializer     =   self.model_config['value_initializer']
        ).to(self.device)
        self.QFunction_tar  =      QEnsemble(
            ensemble_size   =   2,
            s_dim           =   self.s_dim,
            a_dim           =   self.a_dim,
            hiddens         =   self.model_config['value_hiddens'],
            inner_nonlinear =   self.model_config['value_nonlinear'],
            initializer     =   self.model_config['value_initializer']
        ).to(self.device)
        self.QFunction_tar.load_state_dict(self.QFunction.state_dict())
        self.optimizer_value    =   optim.Adam(self.QFunction.parameters(), self.lr)

        # batch dynamics model
        self.dynamics   =   EnsembleDynamicsModel( 
            network_size=   self.model_config['dynamics_ensemble_size'],
            elite_size  =   self.model_config['dynamics_elite_size'],
            state_size  =   self.s_dim,
            action_size =   self.a_dim,
            reward_size =   1,
            hidden_size =   self.model_config['dynamics_hidden_size'],
            use_decay   =   True,
            device      =   config['device']
        ).to(self.device)

    def select_action(self, s: np.array, test = True) -> np.array:
        pi = self.optimistic_policy if self.config['optimistic'] else self.policy
        with torch.no_grad():
            s               =   torch.from_numpy(s).float().to(self.device).view(1,-1)
            if test:
                action      =   pi.sample_action(s, with_noise=False)
            else:
                action      =   pi.sample_action(s, with_noise=True)
        return action.detach().squeeze().cpu().numpy()

    def train_model(self, src_replay_buffer, tar_replay_buffer,  current_step: int,) -> None:
        if src_replay_buffer.size < self.dynamics_batch_size:
            return

        # decay the max epochs since update coefficient
        current_dynamics_max_epochs_since_update    =   max(
            0,
            self.dynamics_max_epochs_since_update - int(current_step / self.max_epochs_since_update_decay_interval)
        )

        s_batch, a_batch, next_s_batch, r_batch, not_done_batch = tar_replay_buffer.sample(tar_replay_buffer.size)
        delta_s_batch = next_s_batch - s_batch
        inputs      = torch.cat((s_batch, a_batch), dim=-1)
        labels      = torch.cat((r_batch, delta_s_batch), dim=-1)
        self.dynamics.train(
            inputs                  =   inputs,
            labels                  =   labels,
            batch_size              =   self.dynamics_batch_size,
            holdout_ratio           =   self.dynamics_holdout_ratio,
            max_epochs_since_update =   current_dynamics_max_epochs_since_update
        )
        self.loss_log[f'loss_dynamics']                             =   self.dynamics._current_mean_ensemble_losses
        self.loss_log['current_dynamics_max_epochs_since_update']   =   current_dynamics_max_epochs_since_update

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None) -> None:
        self.training_count += 1

        # train dynamics model
        if self.training_count % self.dynamics_train_freq == 0:
            self.train_model(src_replay_buffer, tar_replay_buffer, self.training_count)

        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            return
        src_s, src_a, src_next_s, src_r, src_not_done       =   src_replay_buffer.sample(batch_size)
        tar_s, tar_a, tar_next_s, tar_r, tar_not_done       =   tar_replay_buffer.sample(batch_size)
        
        assert len(src_not_done.shape) == len(src_r.shape) == len(tar_r.shape) == len(tar_not_done.shape) == 2

        # training q function
        loss_value, chosen_src_sample_idx      =       self.update_q_functions(
            src_s, src_a, src_r, src_not_done, src_next_s,
            tar_s, tar_a, tar_r, tar_not_done, tar_next_s,
            self.terminal_func,
            self.training_count
        )
        self.optimizer_value.zero_grad()
        loss_value.backward()
        critic_total_norm = nn.utils.clip_grad_norm_(self.QFunction.parameters(), self.ac_gradient_clip)
        self.optimizer_value.step()
        self.loss_log['loss_value']         = loss_value.cpu().item()
        self.loss_log['value_total_norm']   = critic_total_norm.detach().cpu().item()

        if self.training_count % self.training_delay == 0:
            # train policy
            loss_policy, new_a_log_prob = self.update_policy(torch.cat([src_s, tar_s], dim=0))
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            policy_total_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.ac_gradient_clip)
            self.optimizer_policy.step()
            self.loss_log['loss_policy']    = loss_policy.cpu().item()
            self.loss_log['policy_total_norm'] = policy_total_norm.detach().cpu().item()

            # train optimistic policy
            loss_opt_policy = self.update_opt_policy(torch.cat([src_s, tar_s], dim=0))
            self.optimizer_opt_policy.zero_grad()
            loss_opt_policy.backward()
            opt_policy_total_norm = nn.utils.clip_grad_norm_(self.optimistic_policy.parameters(), self.ac_gradient_clip)
            self.optimizer_opt_policy.step()
            self.loss_log['loss_opt_policy']    = loss_opt_policy.cpu().item()
            self.loss_log['opt_policy_total_norm'] = opt_policy_total_norm.detach().cpu().item()

            if self.train_alpha:
                loss_alpha  =  (- torch.exp(self.log_alpha) * (new_a_log_prob.detach() + self.target_entropy)).mean()
                self.optimizer_alpha.zero_grad()
                loss_alpha.backward()
                self.optimizer_alpha.step()
                self.alpha = torch.exp(self.log_alpha)
                
                self.loss_log['alpha'] = self.alpha.detach().cpu().item()
                self.loss_log['loss_alpha'] = loss_alpha.detach().cpu().item()

        # soft update target networks
        soft_update(self.QFunction, self.QFunction_tar, self.tau)
        
        # write
        if writer is not None and self.training_count % 5000 == 0:
            for k_log, v_log in list(self.loss_log.items()):
                writer.add_scalar(f'train/{k_log}', v_log, self.training_count)

    def update_q_functions(
        self, 
        src_s: torch.tensor, src_a: torch.tensor, src_r: torch.tensor, src_not_done: torch.tensor, src_next_s: torch.tensor,
        tar_s: torch.tensor, tar_a: torch.tensor, tar_r: torch.tensor, tar_not_done: torch.tensor, tar_next_s: torch.tensor,
        terminal_func: Callable,
        current_step: int,
        src_hist_next_a: torch.tensor = None
    ) -> torch.tensor:
        # first calculate the value loss wrt samples from target domain
        with torch.no_grad():
            tar_next_a, tar_next_a_logprob, _   = self.policy(tar_next_s)
            tar_next_sa_q1, tar_next_sa_q2      = self.QFunction_tar(tar_next_s, tar_next_a)
            tar_next_sa_q                       = torch.min(tar_next_sa_q1, tar_next_sa_q2)
            tar_value_target                    = tar_r + tar_not_done * self.gamma * (tar_next_sa_q - self.alpha * tar_next_a_logprob)
        tar_q1, tar_q2  =   self.QFunction(tar_s, tar_a)
        tar_q_loss      =   F.mse_loss(tar_q1, tar_value_target) + F.mse_loss(tar_q2, tar_value_target)

        # then calculate the value loss wrt samples from source domain
        ## 1. obtain the value target of source domain trans
        with torch.no_grad():
            src_next_a, src_next_a_logprob, _   = self.policy(src_next_s)
            src_next_sa_q1, src_next_sa_q2      = self.QFunction_tar(src_next_s, src_next_a)
            src_next_sa_q                       = torch.min(src_next_sa_q1, src_next_sa_q2)
            src_value_target                    = src_r + src_not_done * self.gamma * (src_next_sa_q - self.alpha * src_next_a_logprob)
        ## 2. obtain the src value targets for comparison
            src_value_target_for_compare    = src_r + src_not_done * self.gamma * src_next_sa_q
        ## 3. expand the dynamics model (of target domain) for TD-h value target distribution
            tar_TD_H_value_target               = self._value_expansion(src_s, src_a, terminal_func)            # [ensemble_size, batch_size, 1]
            tar_TD_H_value_mean                 = torch.mean(tar_TD_H_value_target, dim=0, keepdim=False)       # [batch_size, 1]
            tar_TD_H_value_std                  = torch.std(tar_TD_H_value_target, dim=0, keepdim=False)        # [batch_size, 1]
            tar_TD_H_value_dist                 = torch.distributions.Normal(loc=tar_TD_H_value_mean, scale=tar_TD_H_value_std + 1e-8)
            self.loss_log['generated_value_mean'] = tar_TD_H_value_mean.mean().detach().item()
            self.loss_log['generated_value_std']  = tar_TD_H_value_mean.std().detach().item()
            self.loss_log['generated_value_max']  = tar_TD_H_value_mean.max().detach().item()
            self.loss_log['generated_value_min']  = tar_TD_H_value_mean.min().detach().item()
            ## obtain the value difference
            value_difference                        = torch.abs(tar_TD_H_value_mean - src_value_target_for_compare)
            self.loss_log['value_difference_mean']  = value_difference.mean().detach().item()
            self.loss_log['value_difference_std']   = value_difference.std().detach().item()
            self.loss_log['value_difference_max']   = value_difference.max().detach().item()
            self.loss_log['value_difference_min']   = value_difference.min().detach().item()
            ## obtain the likelihood
            src_value_target_in_dist_likelihood = torch.exp(tar_TD_H_value_dist.log_prob(src_value_target_for_compare)) # [batch_size, 1]
            self.loss_log['src_value_target_likelihood_mean']   =   src_value_target_in_dist_likelihood.mean().detach().item()
            self.loss_log['src_value_target_likelihood_std']    =   src_value_target_in_dist_likelihood.std().detach().item()
        # 4. reject sampling the src samples with likelihood under threshold
            if current_step > self.start_gate_src_sample:
                threshold_likelihood    =   torch.quantile(
                    src_value_target_in_dist_likelihood,
                    q   = self.likelihood_gate_threshold,
                )   # []
                accept_gate             =   (src_value_target_in_dist_likelihood > threshold_likelihood)
            else:
                accept_gate             =   torch.ones_like(src_value_target, device=self.device)
            src_chosen_sample_idx       =   torch.where(accept_gate[:, 0] > 0)[0]
            self.loss_log['accept_gated_ratio']    =   torch.sum(accept_gate.int()).detach().item() / np.prod(accept_gate.shape)
            
        # 5. obtain the loss wrt src samples
        src_q1, src_q2  =   self.QFunction(src_s, src_a)
        src_q_loss      =   (accept_gate * (src_q1 - src_value_target) ** 2).mean() + (accept_gate * (src_q2 - src_value_target) ** 2).mean()
        
        self.loss_log['q_loss_src'] = src_q_loss.detach().item()
        self.loss_log['q_loss_tar'] = tar_q_loss.detach().item()
        return tar_q_loss + src_q_loss, src_chosen_sample_idx
        
    def _value_expansion(self, src_s: torch.tensor, src_a: torch.tensor, terminal_func: Callable) -> torch.tensor:
        # imagine the next s in target domain
        dyna_pred_mean, dyna_pred_var = self.dynamics.predict(inputs=torch.cat([src_s, src_a], dim=-1), factor_ensemble=True)   # [ensemble_size, batch_size, 1 + |S|]
        dyna_pred_samples               =   dyna_pred_mean + torch.ones_like(dyna_pred_var, device=self.device) * dyna_pred_var
        dyna_pred_r, dyna_pred_delta_s  =   dyna_pred_samples[:, :, :1], dyna_pred_samples[:, :, 1:]
        dyna_pred_next_s                =   src_s + dyna_pred_delta_s 

        # expand via dynamics model
        state               =   dyna_pred_next_s
        cumulate_r          =   dyna_pred_r
        discount            =   self.gamma
        notdone             =   terminal_func(dyna_pred_next_s, return_done=False)

        # value target of imagined target domain transition
        final_pred_next_s   =   state
        final_action        =   self.policy.sample_action(final_pred_next_s, with_noise=False)
        final_q1, final_q2  =   self.QFunction_tar(final_pred_next_s, final_action)
        final_value         =   torch.min(final_q1, final_q2)   # [ensemble_size, batch_size, 1]
    
        TD_H_Value  =   cumulate_r + notdone * discount * final_value
        return TD_H_Value

    def update_policy(self, s: torch.tensor,) -> torch.tensor:
        a, a_log_prob, _    =   self.policy(s)
        q1_value, q2_value  =   self.QFunction(s, a)
        q_value             =   torch.min(q1_value, q2_value)
        neg_entropy         =   a_log_prob
        return (- q_value + self.alpha * neg_entropy).mean(), a_log_prob

    def update_opt_policy(self, s: torch.tensor,) -> torch.tensor:
        a, a_log_prob, _    =   self.optimistic_policy(s)
        q1_value, q2_value  =   self.QFunction(s, a)
        q_value             =   torch.max(q1_value, q2_value)
        neg_entropy         =   a_log_prob
        return (- q_value + self.alpha * neg_entropy).mean()

    def save(self, filename: str) -> None:
        torch.save(self.dynamics.state_dict(), filename + "_dynamics")
        torch.save(self.QFunction.state_dict(), filename + "_critic")
        torch.save(self.optimizer_value.state_dict(), filename + "_critic_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.optimizer_policy.state_dict(), filename + "_actor_optimizer")
        torch.save(self.optimistic_policy.state_dict(), filename + "_actor_optimistic")
        torch.save(self.optimizer_opt_policy.state_dict(), filename + "_actor_optimistic_optimizer")
    
    def load(self, filename: str) -> None:
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.optimizer_policy.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.QFunction.load_state_dict(torch.load(filename + "_critic"))
        self.optimizer_value.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.dynamics.load_state_dict(torch.load(filename + "_dynamics"))
        self.optimistic_policy.load_state_dict(torch.load(filename + "_actor_optimistic"))
        self.optimizer_opt_policy.load_state_dict(torch.load(filename + "_actor_optimistic_optimizer"))