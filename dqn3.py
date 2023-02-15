import os
import numpy as np
import torch.nn as nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from prb import PrioritizedReplayBuffer


class DQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, checkpoint_dir):
        super(DQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, name)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.network = nn.Sequential(nn.Linear(*input_dims, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, n_actions))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss(reduction="none")
        self.to(self.device)

    def forward(self, state):
        return self.network(state)

    def save_checkpoint(self):
        print("Saving checkpoint to %s" % self.checkpoint_file)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint from %s" % self.checkpoint_file)
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.1, eps_dec=5e-7, tau=1000, checkpoint_dir='checkpoints', alpha=0.2, beta=0.6, eps_prior=1e-6):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.learn_step_counter = 0
        self.tau = tau
        self.checkpoint_dir = checkpoint_dir
        self.action_space = np.arange(self.n_actions)
        self.alpha = alpha
        self.beta = beta
        self.eps_prior = eps_prior

        self.memory = PrioritizedReplayBuffer(self.input_dims, self.mem_size, self.batch_size, self.alpha)

        self.dqn = DQN(self.lr, self.n_actions, 'dqn',
                       self.input_dims, self.checkpoint_dir)
        self.dqn_target = DQN(self.lr, self.n_actions,
                              'dqn_target', self.input_dims, self.checkpoint_dir)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array(observation),
                             dtype=T.float).to(self.dqn.device)
            reward = self.dqn.forward(state)
            action = T.argmax(reward).item()
            return action
        return np.random.choice(self.action_space)

    def store_transition(self, state, action, state_, reward, done):
        self.memory.store_transition(state, action, state_, reward, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.tau == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - \
            self.eps_dec if self.epsilon > self.eps_min else self.eps_min

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
        

        curr_q_value = self.dqn.forward(states).gather(1, actions)
        next_q_value = self.dqn_target.forward(
            states_
        ).max(dim=1, keepdim=True)[0].detach()
        mask = ~dones
        target = (rewards + self.gamma * next_q_value *
                  mask).to(self.dqn.device)

        elem_loss = self.dqn.loss(curr_q_value, target).to(self.dqn.device)
        loss = T.mean(elem_loss*weights)        
        
        self.dqn.optimizer.zero_grad()
        loss.backward()        
        self.dqn.optimizer.step()
        
        loss_prior = elem_loss.detach()
        self.memory.update_priorities(indices, loss_prior + self.eps_prior)
        
        self.replace_target_network()
        self.learn_step_counter += 1
        self.decrement_epsilon()
