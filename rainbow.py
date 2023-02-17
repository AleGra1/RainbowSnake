from noisy_net import NoisyLinear

import os
import numpy as np
import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from buffers import ReplayBuffer, PrioritizedReplayBuffer

class RainbowNet(nn.Module):
    def __init__(self, input_dims, n_actions, atom_size, support, name, checkpoint_dir='checkpoints'):
        super(RainbowNet, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.support = support.to(self.device)
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.atom_size = atom_size
        
        self.feature_layer = nn.Sequential(nn.Linear(*input_dims, 128))
        
        self.adv_hidden_layer = NoisyLinear(128, 128)
        self.adv_layer = NoisyLinear(128, self.n_actions * self.atom_size)
        
        self.val_hidden_layer = NoisyLinear(128, 128)
        self.val_layer = NoisyLinear(128, atom_size)
        
        self.optimizer = optim.Adam(self.parameters())
        self.loss = nn.SmoothL1Loss()
        
        self.to(self.device)
        
    def forward(self, state):
        dist = self.dist(state)
        return T.sum(dist * self.support, dim=2)
    
    def dist(self, state):
        feature = self.feature_layer(state)
        adv_hid = F.relu(self.adv_hidden_layer(feature))
        val_hid = F.relu(self.val_hidden_layer(feature))
        
        advantage = self.adv_layer(adv_hid).view(
            -1, self.n_actions, self.atom_size
        )
        value = self.val_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.adv_hidden_layer.reset_noise()
        self.adv_layer.reset_noise()
        self.val_hidden_layer.reset_noise()
        self.val_layer.reset_noise()
        
    def save_checkpoint(self):
        print("Saving checkpoint to %s" % self.checkpoint_file)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint from %s" % self.checkpoint_file)
        self.load_state_dict(T.load(self.checkpoint_file))
        
class Agent:
    def __init__(self, mem_size, batch_size, n_actions, input_dims, tau, gamma, alpha, beta, eps_prior, v_min, v_max, atom_size, n_step):
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps_prior = eps_prior
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.n_step = n_step
        self.mem_size = mem_size
        
        self.learn_step_counter = 0
        
        self.memory = PrioritizedReplayBuffer(self.input_dims, self.mem_size, self.batch_size, self.alpha)
        self.memory_n = ReplayBuffer(self.mem_size, self.input_dims, self.n_step, self.gamma)        
        
        self.support = T.linspace(self.v_min, self.v_max, self.atom_size)
                
        self.dqn = RainbowNet(self.input_dims, self.n_actions, self.atom_size, self.support, 'rainbow')
        self.dqn_target = RainbowNet(self.input_dims, self.n_actions, self.atom_size, self.support, 'rainbow_target')
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        self.support = self.support.to(self.dqn.device)
                
        
    def choose_action(self, observation):
        state = T.tensor(np.array(observation), dtype=T.float).to(self.dqn.device)
        reward = self.dqn.forward(state)
        action = T.argmax(reward).item()
        return action
    
    def store_transition(self, state, action, state_, reward, done):
        transition = self.memory_n.store_transition(state, action, state_, reward, done)
        if transition:
            self.memory.store_transition(*transition)
            
    def replace_target_network(self):
        if self.learn_step_counter % self.tau == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())
            
    def save_models(self):
        self.dqn.save_checkpoint()
        self.dqn_target.save_checkpoint()

    def load_models(self):
        self.dqn.load_checkpoint()
        self.dqn_target.load_checkpoint()

    def learn(self, fraction_idx_total):
        fraction_idx_total = min(fraction_idx_total, 1.0)
        self.beta = self.beta + fraction_idx_total*(1.0 - self.beta)
        
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, state_, done, weight, indices = self.memory.sample(self.beta)
        states = state.to(self.dqn.device)
        actions = action.reshape(-1, 1).to(self.dqn.device)
        rewards = reward.reshape(-1, 1).to(self.dqn.device)
        states_ = state_.to(self.dqn.device)
        dones = done.reshape(-1, 1).to(self.dqn.device)
        weights = weight.reshape(-1, 1).to(self.dqn.device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with T.no_grad():
            # Double DQN
            next_action = self.dqn(states_).argmax(1)
            next_dist = self.dqn_target.dist(states_)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = rewards + (~dones) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                T.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.dqn.device)
            )

            proj_dist = T.zeros(next_dist.size(), device=self.dqn.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(states)
        log_p = T.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)
        loss = T.mean(elementwise_loss*weights)        
        
        gamma = self.gamma ** self.n_step
        state, action, reward, state_, done = self.memory_n.sample_from_indices(indices)
        states = state.to(self.dqn.device)
        actions = action.reshape(-1, 1).to(self.dqn.device)
        rewards = reward.reshape(-1, 1).to(self.dqn.device)
        states_ = state_.to(self.dqn.device)
        dones = done.reshape(-1, 1).to(self.dqn.device)
        weights = weight.reshape(-1, 1).to(self.dqn.device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with T.no_grad():
            # Double DQN
            next_action = self.dqn(states_).argmax(1)
            next_dist = self.dqn_target.dist(states_)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = rewards + (~dones) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                T.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.dqn.device)
            )

            proj_dist = T.zeros(next_dist.size(), device=self.dqn.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(states)
        log_p = T.log(dist[range(self.batch_size), action])
        elementwise_loss += -(proj_dist * log_p).sum(1)
        loss = T.mean(elementwise_loss * weights)
        
        self.dqn.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.dqn.optimizer.step()
        
        # PER: update priorities
        #loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = elementwise_loss + self.eps_prior
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()
