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


class CategoricalDQN(nn.Module):
    def __init__(self, atom_size, support, n_actions, name, input_dims, checkpoint_dir):
        super(CategoricalDQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, name)        

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.support = support.to(self.device)
        self.atom_size = atom_size
        self.n_actions = n_actions

        self.network = nn.Sequential(nn.Linear(*input_dims, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.n_actions * self.atom_size))
        self.optimizer = optim.Adam(self.parameters())
        self.loss = nn.SmoothL1Loss()
        self.to(self.device)

    def forward(self, state):
        return T.sum(self.dist(state) * self.support, dim=2)
    
    def dist(self, state):
        """Get distribution for atoms."""
        q_atoms = self.network(state).view(-1, self.n_actions, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist.to(self.device)

    def save_checkpoint(self):
        print("Saving checkpoint to %s" % self.checkpoint_file)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint from %s" % self.checkpoint_file)
        self.load_state_dict(T.load(self.checkpoint_file))
        

class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.1, eps_dec=5e-7, tau=1000, v_min=0.0, v_max=200.0, atom_size=51, checkpoint_dir='checkpoints'):
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.tau = tau
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.checkpoint_dir = checkpoint_dir
        self.learn_step_counter = 0
        
        self.action_space = np.arange(self.n_actions)
        
        self.memory = ReplayBuffer(self.mem_size, self.input_dims)        
                      
        self.support = T.linspace(self.v_min, self.v_max, self.atom_size)
        
        self.dqn = CategoricalDQN(self.atom_size, self.support, self.n_actions, 'categorical', self.input_dims, self.checkpoint_dir)  
        self.dqn_target = CategoricalDQN(self.atom_size, self.support, self.n_actions, 'categorical_target', self.input_dims, self.checkpoint_dir)    
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
        
    def loss(self, states, actions, rewards, states_, dones):
        self.support = self.support.to(self.dqn.device)
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with T.no_grad():
            next_action = self.dqn_target(states_).argmax(1)
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
        log_p = T.log(dist[range(self.batch_size), actions])

        return -(proj_dist * log_p).sum(1).mean()


    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        state, action, reward, state_, done = self.memory.sample(
            self.batch_size)
        
        states = state.to(self.dqn.device)
        actions = action.reshape(-1, 1).to(self.dqn.device)
        rewards = reward.reshape(-1, 1).to(self.dqn.device)
        states_ = state_.to(self.dqn.device)
        dones = done.reshape(-1, 1).to(self.dqn.device)
        
        loss = self.loss(states, actions, rewards, states_, dones)
        self.dqn.optimizer.zero_grad()
        loss.backward()
        self.dqn.optimizer.step()

        self.replace_target_network()        
        self.learn_step_counter += 1
        self.decrement_epsilon()
