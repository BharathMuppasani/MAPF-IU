import torch
import torch.nn.functional as F
import numpy as np
from dqn import ResNetDQN, GraphDQN       # network architecture
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, DictReplayBuffer, PrioritizedDictReplayBuffer
# For batching graph data:
from torch_geometric.data import Batch

from utils.grid_env_wrapper import grid_to_graph

class DQNAgent:
    def __init__(self, 
                 state_dim,
                 action_dim,
                 device,
                 learning_rate=3e-4,
                 gamma=0.99,
                 buffer_size=100000,
                 batch_size=128,
                 target_update=10,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 use_double=False,
                 use_priority=False,
                 custom_flag=False):  # custom_flag indicates using GraphDQN.
        
        self.custom_flag = custom_flag  # When True, we use and dictionary replay buffers.
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.use_double = use_double
        self.use_priority = use_priority
        self.learning_rate = learning_rate

        # Choose network based on flags.
        if custom_flag:
            # print("Using custom GraphDQN architecture.")
            network = ResNetDQN # GraphDQN
            node_feature_dim = 4

        if network == GraphDQN:
            self.policy_net = network(node_feature_dim, action_dim).to(device)
            self.target_net = network(node_feature_dim, action_dim).to(device)
        else:
            self.policy_net = network(state_dim[0], action_dim).to(device)
            self.target_net = network(state_dim[0], action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, eps=1e-5)
        

        if custom_flag:
            if use_priority:
                self.memory = PrioritizedDictReplayBuffer(buffer_size, device)  # Or a prioritized dict replay buffer.
            else:
                self.memory = DictReplayBuffer(buffer_size, device)
        else:
            if use_priority:
                self.memory = PrioritizedReplayBuffer(buffer_size, device)
            else:
                self.memory = ReplayBuffer(buffer_size, device)
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.training_step = 0
        self.tau = 0.95 # Soft update parameter

    ############ --- NEW HELPER FUNCTIONS FOR GRAPH CONVERSION --- ############

    def _dict_to_graph(self, obs):
        # Ensure grid is a numpy array.
        grid = obs['grid']
        if isinstance(grid, torch.Tensor):
            grid = grid.cpu().numpy()
        # Now compute agent positions.
        agent_positions = np.argwhere(grid == 1)
        goal_positions = np.argwhere((grid == 2) | (grid == 3))
        agent_pos = tuple(agent_positions[0]) if len(agent_positions) > 0 else None
        goal_pos = tuple(goal_positions[0]) if len(goal_positions) > 0 else None

        # Ensure direction and distance are numpy arrays.
        direction = obs['direction']
        if isinstance(direction, torch.Tensor):
            direction = direction.cpu().numpy()
        distance = obs['distance']
        if isinstance(distance, torch.Tensor):
            distance = distance.cpu().numpy()

        low = np.concatenate([direction, distance]).reshape(1, 3)
        graph_data = grid_to_graph(grid, agent_pos, goal_pos, low)
        return graph_data


    def _dict_to_graph_batch(self, obs):
        """
        Converts a batched dictionary observation to a batched PyTorch Geometric Data object.
        The input `obs` is a dictionary with keys 'grid', 'direction', 'distance',
        where each key has a batch of items (e.g., shape (batch_size, ...)).
        """
        data_list = []
        # Determine the batch size from one of the keys.
        batch_size = obs['grid'].shape[0]
        for i in range(batch_size):
            # Create a dictionary for the i-th observation.
            single_obs = {
                'grid': obs['grid'][i],
                'direction': obs['direction'][i],
                'distance': obs['distance'][i]
            }
            data = self._dict_to_graph(single_obs)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        return batch

    ############ --- END HELPER FUNCTIONS --- ############

    def _dict_to_batch_tensors(self, obs):
        """
        For non-graph cases; kept for legacy networks.
        """
        grid = torch.FloatTensor(np.array(obs['grid'])).unsqueeze(0).to(self.device)
        direction = torch.FloatTensor(np.array(obs['direction'])).unsqueeze(0).to(self.device)
        distance = torch.FloatTensor(np.array(obs['distance'])).unsqueeze(0).to(self.device)
        return {'grid': grid, 'direction': direction, 'distance': distance}


    def _batch_dict_list(self, obs_batch):
        """
        Converts a list of obs dicts into a batch of tensors for training.
        """
        grids = torch.FloatTensor([obs['grid'] for obs in obs_batch]).unsqueeze(1).to(self.device)        # [B, 1, H, W]
        directions = torch.FloatTensor([obs['direction'] for obs in obs_batch]).to(self.device)           # [B, 2]
        distances = torch.FloatTensor([obs['distance'] for obs in obs_batch]).to(self.device)             # [B, 1]
        return {'grid': grids, 'direction': directions, 'distance': distances}


    def select_action(self, state, valid_actions=None):
        """
        Select an action given the current state.
        """
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                if self.custom_flag: # for GraphDQN
                    # Convert the single observation to a batched graph (batch size 1).
                    # state_graph = self._dict_to_graph(state)
                    # state_batch = Batch.from_data_list([state_graph]).to(self.device)
                    # Batch it:
                    state_batch = self._dict_to_batch_tensors(state)
                    q_values = self.policy_net(state_batch)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor)

                # Filter by non_oscillating_actions if provided
                if valid_actions:
                    mask = torch.full(q_values.shape, float('-inf')).to(self.device)
                    mask[0, valid_actions] = 0
                    filtered_q_values = q_values + mask
                    return filtered_q_values.max(1)[1].item()
                else:
                    return q_values.max(1)[1].item()
                
                return q_values.max(1)[1].item()
        else:
            # Use valid actions for random selection.
            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                return np.random.randint(self.action_dim)

            # return np.random.randint(self.action_dim)

    def select_actions(self, states, valid_actions_list):
        """
        Batch‐version of select_action:
          - `states`: list of obs dicts, length B
          - `valid_actions_list`: list of lists of valid action indices, length B
        Returns:
          - actions: torch.LongTensor of shape (B,)
        """
        B = len(states)
        if self.custom_flag:
            # graph version:
            # build a Batch from list of dicts
            batch = self._batch_dict_list(states)
            q = self.policy_net(batch)                      # shape [B, A]
        else:
            # simple grid version: stack into tensor [B, H, W]
            grids = np.stack([s for s in states], axis=0)   # here s should be the raw grid
            # if your state is a dict with multiple fields, adapt this
            x = torch.FloatTensor(grids).to(self.device)    # [B, H, W]
            q = self.policy_net(x)                          # [B, A]

        # mask invalid moves
        mask = torch.full_like(q, float('-inf'))
        for i, valid in enumerate(valid_actions_list):
            mask[i, valid] = 0.0
        q = q + mask

        # greedy selection
        return q.argmax(dim=1)  # tensor of shape (B,)

    
    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample from replay buffer.
        if self.use_priority:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None

        # Convert batch observations to network input.
        if self.custom_flag:
            # 'states' and 'next_states' are lists/dicts, so we batch them as graphs.
            # state_batch = self._dict_to_graph_batch(states).to(self.device)
            # next_state_batch = self._dict_to_graph_batch(next_states).to(self.device)
            state_batch = states
            next_state_batch = next_states
        else:
            state_batch = torch.FloatTensor(states).to(self.device)
            next_state_batch = torch.FloatTensor(next_states).to(self.device)
        
        # Compute current Q-values.
        current_q_values = self.policy_net(state_batch).gather(1, actions)
        
        with torch.no_grad():
            if self.use_double:
                next_q_values = self.policy_net(next_state_batch)
                next_actions = next_q_values.argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            else:
                next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        td_errors = torch.abs(target_q_values - current_q_values)
        elementwise_loss = F.smooth_l1_loss(current_q_values, target_q_values.detach(), reduction='none').squeeze(1)
        loss = (weights * elementwise_loss).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.use_priority and indices is not None:
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        if self.training_step % self.target_update == 0 and self.training_step > 0:
            print("$$$$$$ Updating target network $$$$$$")
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

        self.training_step += 1
        
        return loss.item(), target_q_values, current_q_values, next_q_values, dones, td_errors
    
    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.epsilon = checkpoint['epsilon']
        # self.training_step = checkpoint['training_step']
