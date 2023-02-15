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

class DDQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, checkpoint_dir):
        super(DDQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.A = nn.Sequential(nn.Linear(*input_dims, 512),
                            nn.ReLU(),
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, n_actions))
        self.V = nn.Sequential(nn.Linear(*input_dims, 512),
                            nn.ReLU(),
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, 1))
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, state):
        return self.V(state), self.A(state)
    
    def save_checkpoint(self):
        print("Saving checkpoint to %s" % self.checkpoint_file)
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print("Loading checkpoint from %s" % self.checkpoint_file)
        self.load_state_dict(T.load(self.checkpoint_file))
        
class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, tau=1000, checkpoint_dir='checkpoints'):
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
        
        self.memory = ReplayBuffer(self.mem_size, self.input_dims)
        
        self.q_eval = DDQN(self.lr, self.n_actions, name='snake_eval', input_dims=self.input_dims, checkpoint_dir=self.checkpoint_dir)
        self.q_next = DDQN(self.lr, self.n_actions, name='snake_next', input_dims=self.input_dims, checkpoint_dir=self.checkpoint_dir)
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array(observation), dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
            return action
        return np.random.choice(self.action_space)
    
    def store_transition(self, state, action, state_, reward, done):
        self.memory.store_transition(state, action, state_, reward, done)
    
    def replace_target_network(self):
        if self.learn_step_counter % self.tau == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
    
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
        
    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()
        
        self.replace_target_network()
        
        state, action, reward, state_, done = self.memory.sample(self.batch_size)
        states = state.to(self.q_eval.device)
        actions = action.to(self.q_eval.device)
        rewards = reward.to(self.q_eval.device)
        states_ = state_.to(self.q_eval.device)
        dones = done.to(self.q_eval.device)
        
        indices = np.arange(self.batch_size)
        
        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        
        V_s_eval, A_s_eval = self.q_eval.forward(states_)
        
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        
        max_actions = T.argmax(q_eval, dim=1)
        
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]
        
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()