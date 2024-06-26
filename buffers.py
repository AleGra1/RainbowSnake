import random
import numpy as np
import torch.nn as nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from segment_tree import MinSegmentTree, SumSegmentTree

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
    
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, input_dims, max_size, batch_size=32, alpha=0.6):
        super(PrioritizedReplayBuffer, self).__init__(max_size, input_dims, 1, 1.0)
        self.batch_size = batch_size
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store_transition(self, state, action, state_, reward, done):
        super().store_transition(state, action, state_, reward, done)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample(self, beta):
        indices = self._sample_proportional()

        states = T.tensor(self.state_memory[indices])
        actions = T.tensor(self.action_memory[indices])
        rewards = T.tensor(self.reward_memory[indices])
        states_ = T.tensor(self.new_state_memory[indices])
        dones = T.tensor(self.done_memory[indices], dtype=T.bool)
        weights = T.tensor([self._calculate_weight(i, beta) for i in indices], dtype=T.float32)

        return states, actions, rewards, states_, dones, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        indices = []
        p_total = self.sum_tree.sum(0, min(self.mem_counter+1, self.max_size) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * min(self.mem_counter+1, self.max_size)) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * min(self.mem_counter+1, self.max_size)) ** (-beta)
        weight = weight / max_weight

        return weight
