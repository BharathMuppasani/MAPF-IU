import numpy as np
import torch
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        
    def push(self, state, action, reward, next_state, done):
        """Ensure state has correct shape before storing."""
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        # assert state.shape == (4, 84, 84), f"Invalid state shape: {state.shape}"
        # assert next_state.shape == (4, 84, 84), f"Invalid next state shape: {next_state.shape}"

        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Ensure batch has correct shape when sampled."""
        experiences = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).to(self.device).unsqueeze(1)

        
        # print(f"Sampled state shape: {states.shape}")  # Expect (batch_size, 4, 84, 84)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, device, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-5
        self.max_priority = 1.0

        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        """
        Push a new transition into the buffer. We'll keep each of
        state, next_state as np.float32 arrays.
        """
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        
        experience = Experience(state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.next_idx] = experience

        # Assign max priority for newly inserted transition
        self.priorities[self.next_idx] = self.max_priority
        self.next_idx = (self.next_idx + 1) % self.capacity

    def get_probabilities(self):
        # Compute scaled priorities and normalize to get probabilities
        scaled_priorities = self.priorities[:len(self.buffer)] ** self.alpha
        return scaled_priorities / np.sum(scaled_priorities)

    def get_importance_weights(self, probs):
        # Compute beta schedule
        beta = min(1.0, 
                   self.beta_start + (1.0 - self.beta_start) 
                   * (self.frame / self.beta_frames))
        # Compute weights
        weights = (len(self.buffer) * probs) ** (-beta)
        # Normalize by max weight
        return torch.FloatTensor(weights / weights.max()).to(self.device)

    def sample(self, batch_size):
        """
        Samples a batch of transitions with priorities.

        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """

        probs = self.get_probabilities()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = self.get_importance_weights(probs[indices])

        experiences = [self.buffer[idx] for idx in indices]


        states = torch.FloatTensor(np.stack([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.stack([e.action for e in experiences])) \
                       .to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(np.stack([e.reward for e in experiences])) \
                       .to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.stack([e.next_state for e in experiences])) \
                          .to(self.device)
        dones = torch.FloatTensor(np.stack([e.done for e in experiences])) \
                     .to(self.device).unsqueeze(1)


        self.frame += 1
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        # Make sure it's 1D
        td_errors = td_errors.squeeze()  # shape: (batch_size,)
        td_errors = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = np.minimum(td_errors, 100.0)
        self.max_priority = max(self.max_priority, np.max(td_errors))


    def __len__(self):
        return len(self.buffer)
    

class DictReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """
        state and next_state are dictionaries with keys like 'grid', 'direction', 'distance'.
        """
        # We assume that state and next_state are already in the desired format.
        # Optionally, you can enforce type conversion here.
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Samples a batch of transitions.
        
        Returns:
            states_dict: dict-of-tensors with keys 'grid', 'direction', 'distance'
            actions: tensor of shape (batch_size, 1)
            rewards: tensor of shape (batch_size, 1)
            next_states_dict: dict-of-tensors
            dones: tensor of shape (batch_size, 1)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        # Gather each field from experiences into lists.
        state_list = [e.state for e in experiences]
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1).to(self.device)
        next_state_list = [e.next_state for e in experiences]
        dones = torch.FloatTensor([e.done for e in experiences]).unsqueeze(1).to(self.device)
        
        # Convert list-of-dicts into a dict-of-tensors.
        states_dict = self._aggregate_dicts(state_list)
        next_states_dict = self._aggregate_dicts(next_state_list)
        
        return states_dict, actions, rewards, next_states_dict, dones

    def _aggregate_dicts(self, dict_list):
        """
        Given a list of dictionaries (each with the same keys), converts them into
        a dictionary where each key maps to a batched torch tensor.
        Assumes each dictionary entry is a NumPy array.
        """
        batch = {}
        # For each key, stack all values along a new 0th dimension.
        for key in dict_list[0]:
            values = [np.array(d[key], dtype=np.float32) for d in dict_list]
            # Convert to torch tensor.
            batch[key] = torch.tensor(np.stack(values, axis=0), dtype=torch.float32).to(self.device)
        return batch

    def __len__(self):
        return len(self.buffer)


class PrioritizedDictReplayBuffer:
    # If you wish to use a prioritized variant, you can follow a similar approach.
    def __init__(self, capacity, device, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-5
        self.max_priority = 1.0

        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        state = state  # assume state is a dict already
        next_state = next_state  # assume next_state is a dict
        experience = Experience(state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.next_idx] = experience

        self.priorities[self.next_idx] = self.max_priority
        self.next_idx = (self.next_idx + 1) % self.capacity

    def get_probabilities(self):
        scaled_priorities = self.priorities[:len(self.buffer)] ** self.alpha
        return scaled_priorities / np.sum(scaled_priorities)

    def get_importance_weights(self, probs):
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        weights = (len(self.buffer) * probs) ** (-beta)
        return torch.FloatTensor(weights / weights.max()).to(self.device)

    def sample(self, batch_size):
        probs = self.get_probabilities()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = self.get_importance_weights(probs[indices])
        experiences = [self.buffer[idx] for idx in indices]
        
        state_list = [e.state for e in experiences]
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1).to(self.device)
        next_state_list = [e.next_state for e in experiences]
        dones = torch.FloatTensor([e.done for e in experiences]).unsqueeze(1).to(self.device)
        
        states_dict = self._aggregate_dicts(state_list)
        next_states_dict = self._aggregate_dicts(next_state_list)
        
        self.frame += 1
        return states_dict, actions, rewards, next_states_dict, dones, indices, weights

    def _aggregate_dicts(self, dict_list):
        batch = {}
        for key in dict_list[0]:
            values = [np.array(d[key], dtype=np.float32) for d in dict_list]
            batch[key] = torch.tensor(np.stack(values, axis=0), dtype=torch.float32).to(self.device)
        return batch

    def update_priorities(self, indices, td_errors):
        td_errors = td_errors.squeeze()
        td_errors = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = np.minimum(td_errors, 100.0)
        self.max_priority = max(self.max_priority, np.max(td_errors))

    def __len__(self):
        return len(self.buffer)