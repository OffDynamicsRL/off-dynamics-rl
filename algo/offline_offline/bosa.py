from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import copy

TensorBatch = List[torch.Tensor]

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def weights_init(m: nn.Module, init_w: float = 3e-3):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-init_w, init_w)
        m.bias.data.uniform_(-init_w, init_w)


class VAE_Policy(nn.Module):
    # Vanilla Variational Auto-Encoder

    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim,
        max_action,
        hidden_dim,
        device,
    ):
        super(VAE_Policy, self).__init__()
        if latent_dim is None:
            latent_dim = 2 * action_dim
        self.encoder_shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action
        self.latent_dim = latent_dim

        self.device = device

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def importance_sampling_estimator(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        beta: float,
        num_samples: int = 500,
    ) -> torch.Tensor:
        # * num_samples correspond to num of samples L in the paper
        # * note that for exact value for \hat \log \pi_\beta in the paper
        # we also need **an expection over L samples**
        mean, std = self.encode(state, action)

        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_enc + std_enc * torch.randn_like(std_enc)  # [B x S x D]

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(state, z)
        std_dec = np.sqrt(beta / 4)

        # Find q(z|x)
        log_qzx = td.Normal(loc=mean_enc, scale=std_enc).log_prob(z)
        # Find p(z)
        mu_prior = torch.zeros_like(z).to(self.device)
        std_prior = torch.ones_like(z).to(self.device)
        log_pz = td.Normal(loc=mu_prior, scale=std_prior).log_prob(z)
        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz = td.Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        ll = w.logsumexp(dim=-1) - np.log(num_samples)
        return ll

    def encode(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder_shared(torch.cat([state, action], -1))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std

    def decode(
        self,
        state: torch.Tensor,
        z: torch.Tensor = None,
    ) -> torch.Tensor:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = (
                torch.randn((state.shape[0], self.latent_dim))
                .to(self.device)
                .clamp(-0.5, 0.5)
            )
        x = torch.cat([state, z], -1)
        return self.max_action * self.decoder(x)



# class EnsembleFC(nn.Module):
#     __constants__ = ['in_dim', 'out_dim']
#     in_dim: int
#     out_dim: int
#     ensemble_size: int
#     weight: torch.Tensor

#     def __init__(
#         self, 
#         in_dim: int, 
#         out_dim: int, 
#         ensemble_size: int, 
#         weight_decay: float = 0., 
#         bias: bool = True
#     ) -> None:
#         super(EnsembleFC, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.ensemble_size = ensemble_size
#         self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_dim, out_dim))
#         self.weight_decay = weight_decay
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(ensemble_size, 1, out_dim))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         pass

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         print(input.shape)
#         print(self.weight.shape)
#         w_times_x = torch.bmm(input, self.weight)
#         return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

#     def extra_repr(self) -> str:
#         return 'in_dim={}, out_dim={}, bias={}'.format(self.in_dim, self.out_dim, self.bias is not None)

class EnsembleFC(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        ensemble_size,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = torch.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class VAE_Dynamics_Ensemble(nn.Module):
    # Variational Auto-Encoder for Dynamics
    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim,
        hidden_dim,
        ensemble_size,
        device,
    ):
        super(VAE_Dynamics_Ensemble, self).__init__()
        if latent_dim is None:
            latent_dim = 2 * state_dim

        self.encoder_shared = nn.Sequential(
            EnsembleFC(state_dim * 2 + action_dim, hidden_dim, ensemble_size=ensemble_size),
            nn.ReLU(),
            EnsembleFC(hidden_dim, hidden_dim, ensemble_size=ensemble_size),
            nn.ReLU(),
        )
        self.device = device

        self.mean = EnsembleFC(hidden_dim, latent_dim, ensemble_size=ensemble_size)
        self.log_std = EnsembleFC(hidden_dim, latent_dim, ensemble_size=ensemble_size)

        self.decoder = nn.Sequential(
            EnsembleFC(state_dim + action_dim + latent_dim, hidden_dim, ensemble_size=ensemble_size),
            nn.ReLU(),
            EnsembleFC(hidden_dim, hidden_dim, ensemble_size=ensemble_size),
            nn.ReLU(),
            EnsembleFC(hidden_dim, state_dim, ensemble_size=ensemble_size),
        )

        # self.max_action = max_action
        self.latent_dim = latent_dim
        self.ensemble_size = ensemble_size

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # reshape inputs
        state = state.repeat(self.ensemble_size, 1, 1)
        action= action.repeat(self.ensemble_size, 1, 1)
        next_state= next_state.repeat(self.ensemble_size, 1, 1)

        mean, std = self.encode(state, action, next_state)  # [E, B, D]
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, action, z)
        return u, mean, std

    def importance_sampling_estimator(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        beta: float,
        num_samples: int = 500,
    ) -> torch.Tensor:
        # * num_samples correspond to num of samples L in the paper
        # * note that for exact value for \hat \log \pi_\beta in the paper
        # reshape inputs
        state = state.repeat(self.ensemble_size, 1, 1)             #   [E, B, D]
        action= action.repeat(self.ensemble_size, 1, 1)
        next_state= next_state.repeat(self.ensemble_size, 1, 1)

        # we also need **an expectation over L samples**
        mean, std = self.encode(state, action, next_state)  # [E. B, D]

        mean_enc  = mean.repeat(num_samples, 1, 1, 1)   #   [S, E, B, D]
        std_enc   = std.repeat(num_samples, 1, 1, 1)    #   [S, E, B, D]
        z         = mean_enc + std_enc * torch.randn_like(std_enc)  #   [S, E, B, D]

        state     = state.repeat(num_samples, 1, 1, 1)
        action    = action.repeat(num_samples, 1, 1, 1)
        next_state= next_state.repeat(num_samples, 1, 1, 1)    #   [S, E, B, D]

        mean_dec  = self.decode(state, action, z)           #   [S, E, B, D]
        std_dec   = np.sqrt(beta / 4)                       #   [S, E, B, D]

        # q(z|x)
        log_qzx   = td.Normal(loc=mean_enc, scale=std_enc).log_prob(z)
        # p(z)
        mu_prior  = torch.zeros_like(z).to(self.device)
        std_prior = torch.ones_like(z).to(self.device)
        log_pz    = td.Normal(loc=mu_prior, scale=std_prior).log_prob(z)
        # p(x|z)
        std_dec   = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz   = td.Normal(loc=mean_dec, scale=std_dec).log_prob(next_state)

        w         = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)  #   [S, E, B]  log( p(x|z) * p(z) / p(z|x) )
        ll        = w.logsumexp(dim=0) - np.log(num_samples)            #   [E, B], average over sample dim
        return ll

    def encode(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z       = self.encoder_shared(torch.cat([state, action, next_state], -1))
        mean    = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std     = torch.exp(log_std)
        return mean, std

    def decode(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor = None,
    ) -> torch.Tensor:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = (
                torch.randn((state.shape[0], state.shape[1], self.latent_dim))  # [E, B, D]
                .to(self.device)
                .clamp(-0.5, 0.5)
            )
        x = torch.cat([state, action, z], -1)
        return self.decoder(x)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
    ):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class BOSA:
    def __init__(
        self,
        config: Dict, 
        device,
    ):
        self.config = config
        self.device = device
        self.max_action = config['max_action']
        self.discount = config['gamma']
        self.tau = config['tau']
        self.policy_noise = config['expl_noise']
        self.noise_clip = config['noise_clip']
        self.policy_freq = config['update_interval']

        self.vae_policy_beta = config['vae_policy_beta']
        self.vae_dyna_beta   = config['vae_dyna_beta']
        self.lamda_policy    = config['lamda_policy']
        self.lamda_dyna      = config['lamda_dyna']

        self.vae_iteration   = config['vae_iteration']

        self.epsilon_policy_exp = config['epsilon_policy_exp']
        self.epsilon_dyna_exp   = config['epsilon_dyna_exp']

        self.conservation_coef  = config['conservation_coef']

        self.num_samples = config['num_samples']
        state_dim, action_dim = config['state_dim'], config['action_dim']
        self.vae_policy              = VAE_Policy(
            state_dim, action_dim, 
            2*action_dim, 
            self.max_action, 
            config['vae_policy_hidden_dim'],
            self.device,
        ).to(self.device)
        self.vae_policy_optimizer    = torch.optim.Adam(self.vae_policy.parameters(), lr=config['vae_policy_lr'])
        self.vae_dyna                = VAE_Dynamics_Ensemble(
            state_dim, action_dim, 
            2*state_dim,
            config['vae_dyna_hidden_dim'],
            config['vae_dyna_ensemble'],
            self.device,
        ).to(self.device)
        self.vae_dyna_optimizer      = torch.optim.Adam(self.vae_dyna.parameters(), lr=config['vae_dyna_lr'])

        self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])

        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=config['critic_lr'])
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=config['critic_lr'])

        # target networks
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = 0

    def select_action(self, state, test=False):
        with torch.no_grad():
            action = self.actor(torch.Tensor(state).view(1,-1).to(self.device))
        return action.squeeze().cpu().numpy()
    
    def elbo_policy_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        beta: float,
        num_samples: int = 1,
    ) -> torch.Tensor:
        mean, std = self.vae_policy.encode(state, action)   #   [B, D]

        mean_s    = mean.repeat(num_samples, 1, 1).permute(1, 0, 2) # [B, S, D]
        std_s     = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z         = mean_s + std_s * torch.randn_like(std_s)

        state     = state.repeat(num_samples, 1, 1).permute(1, 0, 2)
        action    = action.repeat(num_samples, 1, 1).permute(1, 0, 2)
        u         = self.vae_policy.decode(state, z)        #   [B, S, D]

        recon_loss= ((u - action) ** 2).mean(dim=(1, 2))    #   [B]
        KL_loss   = - 0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1) # [B]
        
        vae_loss = recon_loss + self.vae_policy_beta * KL_loss
        return vae_loss

    def iwae_policy_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        beta: float,
        num_samples: int = 10,
    ) -> torch.Tensor:
        ll = self.vae_policy.importance_sampling_estimator(state, action, beta, num_samples)
        return -ll

    def elbo_dyna_loss(
        self, 
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        beta: float,
        num_samples: int = 10,
    ) -> torch.Tensor:
        state       = state.repeat(self.vae_dyna.ensemble_size, 1, 1)
        action      = action.repeat(self.vae_dyna.ensemble_size, 1, 1)
        next_state  = next_state.repeat(self.vae_dyna.ensemble_size, 1, 1)  #   [E, B, D]

        mean, std   = self.vae_dyna.encode(state, action, next_state)       #   [E, B, D]
        mean_s      = mean.repeat(num_samples, 1, 1, 1)         #   [S, E, B, D]
        std_s       = std.repeat(num_samples, 1, 1, 1)  
        z           = mean_s + std_s * torch.randn_like(std_s)  #   [S, E, B, D]

        state       = state.repeat(num_samples, 1, 1, 1)
        action      = action.repeat(num_samples, 1, 1, 1)       #   [S, E, B, D]
        u           = self.vae_dyna.decode(state, action, z)

        recon_loss  = ((u - action) ** 2).mean(dim=(0, 3))      #   [E, B]
        KL_loss     = - 0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)   #   [E, B]

        vae_loss    = recon_loss + self.vae_dyna_beta * KL_loss               #   [E, B]
        return vae_loss

    def iwae_dyna_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        beta: float,
        num_samples: int = 10,
    ) -> torch.Tensor:
        ll = self.vae_dyna.importance_sampling_estimator(state, action, next_state, beta, num_samples)  #   [E, B]
        return -ll

    def vae_models_train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1
        loss_dict_vae_policy    =   self.vae_policy_train(batch)
        loss_dict_vae_dynamics  =   self.vae_dyna_train(batch)
        log_dict.update(loss_dict_vae_policy)
        log_dict.update(loss_dict_vae_dynamics)
        return log_dict

    def vae_policy_train(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, _, _, _ = batch
        # Variational Auto-Encoder Training
        recon, mean, std    = self.vae_policy(state, action)
        recon_loss          = F.mse_loss(recon, action)
        KL_loss             = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss            = recon_loss + self.vae_policy_beta * KL_loss

        self.vae_policy_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_policy_optimizer.step()

        return {
            "VAE_Policy/reconstruction_loss": recon_loss.item(),
            "VAE_Policy/KL_loss": KL_loss.item(),
            "VAE_Policy/vae_loss": vae_loss.item()
        }

    def vae_dyna_train(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, next_state, _, _ = batch
        # Variational Auto-Encoder Training
        recon, mean, std    = self.vae_dyna(state, action, next_state)
        recon_loss          = F.mse_loss(recon, next_state.repeat(int(self.config['vae_dyna_ensemble']), 1, 1))
        KL_loss             = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss            = recon_loss + self.vae_dyna_beta * KL_loss

        self.vae_dyna_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_dyna_optimizer.step()

        return {
            "VAE_Dynamics/reconstruction_loss": recon_loss.item(),
            "VAE_Dynamics/KL_loss": KL_loss.item(),
            "VAE_Dynamics/vae_loss": vae_loss.item()
        }

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):
        self.total_it += 1
        
        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            return
        batch_src, batch_tar = src_replay_buffer.sample(batch_size), tar_replay_buffer.sample(batch_size)
        batch_mix            = [torch.cat([b_tar, b_src], dim=0) for b_tar, b_src in zip(batch_tar, batch_src)]
        
        if self.total_it < self.vae_iteration:
            # vae model pretrain
            log_dict = self.vae_models_train(batch_mix)
        else:
            log_dict = {}
            state_tar, action_tar, next_state_tar, reward_tar, done_tar = batch_tar
            state_src, action_src, next_state_src, reward_src, done_src = batch_src
            state_mix, action_mix, next_state_mix, reward_mix, done_mix = batch_mix

            # train value functions
            with torch.no_grad():
                # Select action according to actor and add clipped noise
                noise           = (torch.randn_like(action_mix) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action_mix = (self.actor_target(next_state_mix) + noise).clamp(-self.max_action, self.max_action)
                # Compute the target Q value
                target_q1 = self.critic_1_target(next_state_mix, next_action_mix)
                target_q2 = self.critic_2_target(next_state_mix, next_action_mix)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward_mix + (1 - done_mix) * self.discount * target_q
            # Get current Q estimates
            mix_q1  = self.critic_1(state_mix, action_mix)
            mix_q2  = self.critic_2(state_mix, action_mix)
            # Get the mask concerning the dynamics inference ll
            with torch.no_grad():
                dyna_ll                     = self.vae_dyna.importance_sampling_estimator(state_mix, action_mix, next_state_mix, beta=self.vae_dyna_beta, num_samples=self.num_samples) #   [E, B]
                min_dyna_ll_over_ensemble   = dyna_ll.min(dim=0, keepdim=False)[0]
                min_dyna_ll_over_ensemble   = min_dyna_ll_over_ensemble.squeeze().cpu().numpy()
                # mask                        = (min_dyna_ll_over_ensemble > np.log(self.epsilon_dyna_exp)).float().to(self.device)
                mask                        = (min_dyna_ll_over_ensemble > np.log(self.epsilon_dyna_exp))
                mask                        = np.array(mask, dtype=np.float32)
                mask                        = torch.FloatTensor(mask).to(self.device)
                log_dict['critic_mask_ratio'] = (mask.sum() / mask.shape[0]).item()
            # Conservative Value for Source domain data
            src_q1  =   self.critic_1(state_src, action_src)
            src_q2  =   self.critic_2(state_src, action_src)
            # Compute critic loss
            critic_td_loss  =  (0.5 * (mask.unsqueeze(-1) * (mix_q1 - target_q)**2)).mean() + (0.5 * mask.unsqueeze(-1) * (mix_q2 - target_q)**2).mean()
            critic_cons_loss=  src_q1.mean() + src_q2.mean()
            critic_loss     =  critic_td_loss + self.conservation_coef * critic_cons_loss
            log_dict["critic_loss"]         = critic_loss.item()
            log_dict["critic_td_loss"]      = critic_td_loss.item()
            log_dict["critic_cons_loss"]    = critic_cons_loss.item()
            # Optimize the critic
            self.critic_1_optimizer.zero_grad()
            self.critic_2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.step()

            # Delayed actor updates
            if self.total_it % self.policy_freq == 0:
                # Compute actor loss
                pi           = self.actor(state_mix)
                q            = self.critic_1(state_mix, pi)

                neg_log_beta = self.iwae_policy_loss(state_mix, pi, self.vae_policy_beta, self.num_samples)

                norm_q       = 1 / q.abs().mean().detach()

                actor_loss   = -norm_q * q.mean() + self.lamda_policy * neg_log_beta.mean()

                log_dict["actor_loss"]          = actor_loss.item()
                log_dict["neg_log_beta_mean"]   = neg_log_beta.mean().item()
                log_dict["neg_log_beta_max"]    = neg_log_beta.max().item()
                log_dict["lamda_policy"]        = self.lamda_policy

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                soft_update(self.critic_1_target, self.critic_1, self.tau)
                soft_update(self.critic_2_target, self.critic_2, self.tau)
                soft_update(self.actor_target, self.actor, self.tau)

        # write
        if writer is not None and self.total_it % 5000 == 0:
            for k_log, v_log in list(log_dict.items()):
                writer.add_scalar(f'train/{k_log}', v_log, self.total_it)

    def save(self, filename) -> Dict[str, Any]:
        torch.save(self.vae_policy.state_dict(), filename + "_vae_policy")
        torch.save(self.vae_policy_optimizer.state_dict(), filename + "_vae_policy_optimizer")
        torch.save(self.vae_dyna.state_dict(), filename + "_vae_dynamics")
        torch.save(self.vae_dyna_optimizer.state_dict(), filename + "_vae_dynamics_optimizer")
        torch.save(self.critic_1.state_dict(), filename + "_critic_1")
        torch.save(self.critic_1_optimizer.state_dict(), filename + "_critic_1_optimizer")
        torch.save(self.critic_2.state_dict(), filename + "_critic_2")
        torch.save(self.critic_2_optimizer.state_dict(), filename + "__critic_2_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")  

    def load(self, filename):
        self.vae_policy.load_state_dict(torch.load(filename + "_vae_policy"))
        self.vae_policy_optimizer.load_state_dict(torch.load(filename + "_vae_policy_optimizer"))
        self.vae_dyna.load_state_dict(torch.load(filename + "_vae_dynamics"))
        self.vae_dyna_optimizer.load_state_dict(torch.load(filename + "_vae_dynamics_optimizer"))

        self.critic_1.load_state_dict(torch.load(filename + "_critic_1"))
        self.critic_1_optimizer.load_state_dict(torch.load(filename + "_critic_1_optimizer"))
        
        self.critic_2.load_state_dict(torch.load(filename + "_critic_2"))
        self.critic_2_optimizer.load_state_dict(torch.load(filename + "_critic_2_optimizer"))

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(filename + "_actor_optimizer")
