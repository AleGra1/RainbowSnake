import os
import numpy as np
import torch.nn as nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_step, gamma):
        self.max_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros(
            (self.max_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.max_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.max_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.max_size, dtype=np.float32)
        self.done_memory = np.zeros(self.max_size, dtype=np.uint8)
        
        self.n_step = n_step
        self.gamma = gamma
        
        self.n_step_buffer = deque(maxlen=self.n_step)

    def store_transition(self, state, action, state_, reward, done):
        self.n_step_buffer.append((state, action, state_, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        
        state_, reward, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        state, action = self.n_step_buffer[0][:2]
        
        index = self.mem_counter % self.max_size
        
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.mem_counter += 1
        
        return self.n_step_buffer[0]

    def sample(self, batch_size):
        indices = np.random.choice(np.arange(min(self.mem_counter, self.max_size)), size=batch_size, replace=False)
        states = T.tensor(self.state_memory[indices])
        actions = T.tensor(self.action_memory[indices])
        rewards = T.tensor(self.reward_memory[indices])
        states_ = T.tensor(self.new_state_memory[indices])
        dones = T.tensor(self.done_memory[indices], dtype=T.bool)

        return states, actions, rewards, states_, dones, indices
    
    def sample_from_indices(self, indices):
        states = T.tensor(self.state_memory[indices])
        actions = T.tensor(self.action_memory[indices])
        rewards = T.tensor(self.reward_memory[indices])
        states_ = T.tensor(self.new_state_memory[indices])
        dones = T.tensor(self.done_memory[indices], dtype=T.bool)

        return states, actions, rewards, states_, dones
    
    def _get_n_step_info(self, n_step_buffer, gamma):
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        # state, action, state_, reward, done
        next_obs, rew, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            n_o, r, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return next_obs, float(rew), done
    
    