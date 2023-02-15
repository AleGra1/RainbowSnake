import os
import numpy as np
import torch.nn as nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.max_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.max_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.max_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.max_size, dtype=np.float32)
        self.done_memory = np.zeros(self.max_size, dtype=np.uint8)
    
    def store_transition(self, state, action, state_, reward, done):
        index = self.mem_counter%self.max_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.mem_counter += 1
    
    def sample(self, batch_size):
        indices = np.random.choice(np.arange(min(self.mem_counter, self.max_size)), size=batch_size, replace=False)
        states = T.tensor(self.state_memory[indices])
        actions = T.tensor(self.action_memory[indices])
        rewards = T.tensor(self.reward_memory[indices])
        states_ = T.tensor(self.new_state_memory[indices])
        dones = T.tensor(self.done_memory[indices], dtype=T.bool)
        
        return states, actions, rewards, states_, dones

class DQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, checkpoint_dir):
        super(DQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.join.path(checkpoint_dir, name)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.network = nn.Sequential(nn.Linear(*input_dims, 128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Linear(128, n_actions))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, state):
        return self.network(state)
    
    def save_checkpoint(self):
        print("Saving checkpoint to %s" % self.checkpoint_file)
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print("Loading checkpoint from %s" % self.checkpoint_file)
        self.load_state_dict(T.load(self.checkpoint_file))