import os
import math
import numpy as np
import torch.nn as nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.max_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros(
            (self.max_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.max_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.max_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.max_size, dtype=np.float32)
        self.done_memory = np.zeros(self.max_size, dtype=np.uint8)

    def store_transition(self, state, action, state_, reward, done):
        index = self.mem_counter % self.max_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.mem_counter += 1

    def sample(self, batch_size):
        indices = np.random.choice(np.arange(
            min(self.mem_counter, self.max_size)), size=batch_size, replace=False)
        states = T.tensor(self.state_memory[indices])
        actions = T.tensor(self.action_memory[indices])
        rewards = T.tensor(self.reward_memory[indices])
        states_ = T.tensor(self.new_state_memory[indices])
        dones = T.tensor(self.done_memory[indices], dtype=T.bool)

        return states, actions, rewards, states_, dones

class NoisyLinear(nn.Module):
    def __init__(self, input_dims, n_actions, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.std_init = std_init

        self.weight_mu = nn.Parameter(T.Tensor(n_actions, *input_dims))
        self.weight_sigma = nn.Parameter(
            T.Tensor(n_actions, *input_dims)
        )
        self.register_buffer(
            "weight_epsilon", T.Tensor(n_actions, *input_dims)
        )

        self.bias_mu = nn.Parameter(T.Tensor(n_actions))
        self.bias_sigma = nn.Parameter(T.Tensor(n_actions))
        self.register_buffer("bias_epsilon", T.Tensor(n_actions))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(len(self.input_dims).flatten())
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(len(self.input_dims).flatten())
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.n_actions)
        )

    def reset_noise(self):
        epsilon_in = self.scale_noise(len(self.input_dims).flatten())
        epsilon_out = self.scale_noise(self.n_actions)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size):
        x = T.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
class NoisyNetwork(nn.Module):
    def __init__(self, n_actions, name, input_dims, checkpoint_dir):
        super(NoisyNetwork, self).__init__()
        
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.network = nn.Sequential(nn.Linear(*input_dims, 128),
                                     nn.ReLU(),
                                     NoisyLinear(128, 128),
                                     nn.ReLU(),
                                     NoisyLinear(128, n_actions))
        self.optimizer = optim.Adam(self.parameters())
        self.loss = nn.SmoothL1Loss()
        self.to(self.device)

    def forward(self, state):
        return self.network(state)
    
    def reset_noise(self):
        self.network[2].reset_noise()
        self.network[4].reset_noise()
        
    def save_checkpoint(self):
        print("Saving checkpoint to %s" % self.checkpoint_file)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint from %s" % self.checkpoint_file)
        self.load_state_dict(T.load(self.checkpoint_file))
