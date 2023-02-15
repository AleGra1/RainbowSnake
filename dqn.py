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

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.1, eps_dec=5e-7, tau=1000, checkpoint_dir='checkpoints'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.tau = tau
        self.checkpoint_dir = checkpoint_dir
        self.action_space = np.arange(self.n_actions)

        self.memory = ReplayBuffer(self.mem_size, self.input_dims)

        self.dqn = DQN(self.lr, self.n_actions, 'dqn', self.input_dims, self.checkpoint_dir)
        self.dqn_target = DQN(self.lr, self.n_actions, 'dqn_target', self.input_dims, self.checkpoint_dir)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array(observation), dtype=T.float).to(self.dqn.device)
            _, advantage = self.dqn.forward(state)
            action = T.argmax(advantage).item()
            return action
        return np.random.choice(self.action_space)

    def store_transition(self, state, action, state_, reward, done):
        self.memory.store_transition(state, action, state_, reward, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.tau == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.dqn.save_checkpoint()
        self.dqn_target.save_checkpoint()

    def load_models(self):
        self.dqn.load_checkpoint()
        self.dqn_target.load_checkpoint()

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        state, action, reward, state_, done = self.memory.sample(self.batch_size)
        
