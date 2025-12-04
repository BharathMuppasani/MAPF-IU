import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch_geometric.data import Data
from utils.grid_env import GridEnvironment, GridState  

# This function remains the same as it's independent of environment generation details
def grid_to_graph(grid, agent_pos, goal_pos, low):
    """
    Convert the entire grid to a graph.
    
    Every cell becomes a node with a 4-dimensional feature vector:
      [is_agent, is_goal, is_static_obstacle, is_dynamic_obstacle]
    
    Cell values are interpreted as:
      - -3: collison,
      - -2: dynamic obstacle,
      - -1: static obstacle,
      -  0: free cell,
      -  1: agent,
      -  2: goal,
      -  3: reached goal.
      
    Edges are added between 4-connected neighboring cells.
    
    Args:
        grid (np.array): 2D grid with cell type values.
        agent_pos (tuple): (row, col) position of the agent.
        goal_pos (tuple): (row, col) position of the goal.
        low (np.array): Low-dimensional features with shape (1,3).
        
    Returns:
        data (torch_geometric.data.Data): Graph data object.
    """
    H, W = grid.shape
    pos_to_node = {}
    node_features = []
    node_idx = 0

    # Create a node for every cell.
    for r in range(H):
        for c in range(W):
            pos_to_node[(r, c)] = node_idx
            cell_val = grid[r, c]
            is_agent = 1.0 if cell_val == 1 or cell_val == 3 else 0.0
            is_goal = 1.0 if cell_val == 2 or cell_val == 3 else 0.0
            is_static_obstacle = 1.0 if cell_val == -1 else 0.0
            is_dynamic_obstacle = 1.0 if cell_val == -2 else 0.0

            node_features.append([is_agent, is_goal, is_static_obstacle, is_dynamic_obstacle])
            node_idx += 1

    node_features = torch.tensor(node_features, dtype=torch.float)

    # Build edge_index using 4-connected neighbors.
    edge_index = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for (r, c), idx in pos_to_node.items():
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                neighbor_idx = pos_to_node[(nr, nc)]
                edge_index.append([idx, neighbor_idx])
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index)
    data.low = torch.tensor(low, dtype=torch.float) if not isinstance(low, torch.Tensor) else low

    # Ensure agent_pos and goal_pos are valid before accessing pos_to_node
    agent_node = pos_to_node.get(agent_pos, -1) # Default to -1 if None or not found
    goal_node  = pos_to_node.get(goal_pos, -1)  # Default to -1 if None or not found
    
    data.agent_idx = torch.tensor([agent_node], dtype=torch.long) # Ensure dtype is long for indices
    data.goal_idx  = torch.tensor([goal_node], dtype=torch.long)  # Ensure dtype is long for indices
    return data

class GridEnvWrapper(gym.Env):
    """
    A Gymnasium-compatible wrapper for the custom GridEnvironment.
    The observation is a dictionary containing:
      - 'grid': the grid state
      - 'direction': a 2D unit vector from the agent to the goal,
      - 'distance': a 1D float representing the Euclidean distance.
      
    Extra parameters:
      - generation_mode: "old", "maze", or "warehouse".
      - maze_density: float (0.0 to 1.0) for "maze" mode.
      - variable_grid (bool): if True, each reset chooses a random grid size.
      - grid_size_range (tuple): (min, max) grid sizes for variable_grid.
      - min_goal_dist (int): Minimum Manhattan-like distance between agent and goal.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(
            self, 
            grid_size=11, # Default grid size
            grid_size_range=None, 
            variable_grid=False,
            num_static_obstacles=5, 
            num_dynamic_obstacles=0,
            generation_mode="old", # Can be "old", "maze", "warehouse"
            maze_density=1.0, 
            max_steps=200,
            min_goal_dist=4 # Added min_goal_dist parameter
        ):
        super(GridEnvWrapper, self).__init__()
        self.variable_grid = variable_grid
        
        if self.variable_grid:
            if grid_size_range is None:
                raise ValueError("grid_size_range must be provided when variable_grid is True")
            self.grid_size_range = grid_size_range
            # Initial grid size will be selected in the first reset or here
            self.current_grid_size = self._select_random_grid_size(generation_mode)
        else:
            self.current_grid_size = grid_size
        
        # For maze generation, ensure grid_size is odd.
        # GridEnvironment itself also handles this, but good to be consistent.
        if generation_mode == "maze" and self.current_grid_size % 2 == 0:
            self.current_grid_size += 1
        
        self.maze_density = maze_density
        self.num_static_obstacles = num_static_obstacles
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.generation_mode = generation_mode
        self.max_steps = max_steps
        self.min_goal_dist = min_goal_dist # Store min_goal_dist

        # Initialize the underlying environment
        self.env = GridEnvironment(
            grid_size=self.current_grid_size, 
            num_static_obstacles=self.num_static_obstacles, 
            num_dynamic_obstacles=self.num_dynamic_obstacles,
            generation_mode=self.generation_mode, 
            maze_density=self.maze_density, 
            max_steps=self.max_steps,
            min_goal_dist=self.min_goal_dist # Pass min_goal_dist
        )
        
        # Define observation space based on current_grid_size.
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=-3, high=3, shape=(self.current_grid_size, self.current_grid_size), dtype=np.int32),
            'direction': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'distance': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(5)  # 0:Up, 1:Down, 2:Left, 3:Right, 4:Stay
    
    def _select_random_grid_size(self, generation_mode_for_select):
        """Selects a random grid size from the range."""
        min_size, max_size = self.grid_size_range
        # For maze mode, prefer odd sizes. For others, any size in range is fine.
        if generation_mode_for_select == "maze":
            possible_sizes = [s for s in range(min_size, max_size + 1) if s % 2 != 0]
            if not possible_sizes: # Fallback if no odd sizes in range (e.g. range is [6,6])
                 possible_sizes = list(range(min_size, max_size + 1))
                 if not possible_sizes: # Should not happen with valid range
                     return min_size # Default fallback
            return np.random.choice(possible_sizes)
        else: # For "old" or "warehouse"
            return np.random.randint(min_size, max_size + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Call super for seeding if necessary

        if self.variable_grid:
            self.current_grid_size = self._select_random_grid_size(self.generation_mode)
            # GridEnvironment also handles odd size for maze, but good to be explicit
            if self.generation_mode == "maze" and self.current_grid_size % 2 == 0:
                 self.current_grid_size +=1
            
            # Re-initialize the underlying environment with the new size and existing params
            self.env = GridEnvironment(
                grid_size=self.current_grid_size, 
                num_static_obstacles=self.num_static_obstacles,
                num_dynamic_obstacles=self.num_dynamic_obstacles,
                generation_mode=self.generation_mode, 
                maze_density=self.maze_density,
                max_steps=self.max_steps,
                min_goal_dist=self.min_goal_dist # Pass min_goal_dist
            )
            # Update observation_space shape.
            self.observation_space.spaces["grid"] = spaces.Box(
                low=-3, high=3, shape=(self.current_grid_size, self.current_grid_size), dtype=np.int32
            )
        else:
            # If not variable_grid, just regenerate the existing environment
            self.env.generate_grid() 
            
        return self._get_obs(), {}
    
    def _get_obs(self):
        grid_obs = self.env.grid.copy() # Current grid state
        
        # Ensure agent_pos and goal_pos are not None before calculations
        if self.env.agent_pos is None or self.env.goal_pos is None:
            # This might happen if grid generation failed to place agent/goal
            # Return a default observation or handle error appropriately
            agent_pos_arr = np.zeros(2, dtype=np.float32)
            goal_pos_arr = np.zeros(2, dtype=np.float32)
            print("Warning: Agent or Goal position is None in _get_obs.")
        else:
            agent_pos_arr = np.array(self.env.agent_pos, dtype=np.float32)
            goal_pos_arr = np.array(self.env.goal_pos, dtype=np.float32)

        vector = goal_pos_arr - agent_pos_arr
        norm = np.linalg.norm(vector)
        unit_vector = vector / norm if norm > 0 else np.zeros_like(vector, dtype=np.float32)
        distance = np.array([norm], dtype=np.float32)
        
        return {'grid': grid_obs, 'direction': unit_vector.astype(np.float32), 'distance': distance}
    
    def step(self, action):
        # Ensure agent_pos is valid before proceeding
        if self.env.agent_pos is None:
            # Handle cases where the environment might not be correctly initialized
            obs = self._get_obs()
            return obs, 0.0, True, False, {"error": "Agent position is None before step."}

        current_state_grid = self.env.grid.copy() # Grid before action
        current_state_obj = GridState(current_state_grid)
        
        # next_state now handles internal state updates of self.env
        new_grid_state_obj, done, goal_reached = self.env.next_state(current_state_obj, action)
        
        # The reward should be calculated based on the state *before* the action,
        # the action taken, and the outcome (done, goal_reached).
        # GridEnvironment's calculate_reward uses self.env.path_followed, which is updated in next_state.
        # So, it's implicitly using the outcome of the action.
        reward = self.env.calculate_reward(new_grid_state_obj, action, done, goal_reached)
        
        # self.env.grid is already updated by self.env.next_state()
        
        return self._get_obs(), reward, done, goal_reached, {} # Gymnasium expects 5 return values
    
    def render(self, mode='human', collision_message=None, goal_flag=False):
        # The render_manual in GridEnvironment now handles display
        self.env.render_manual(collision_message=collision_message, goal_flag=goal_flag)
    
    def get_actions(self):
        return self.env.get_actions()
    
    def close(self):
        # If GridEnvironment had any resources to clean up (like matplotlib figures in interactive mode)
        # they could be closed here. For now, __del__ in GridEnvironment handles its figure.
        if hasattr(self.env, '__del__'):
            self.env.__del__() # Explicitly call if needed, though Python's GC usually handles it.
        pass # No specific resources opened directly by the wrapper itself
    
    def get_graph_obs(self):
        obs = self._get_obs()
        grid = obs['grid']
        # Ensure low features are correctly shaped (1,3)
        low_features = np.concatenate([obs['direction'], obs['distance']]).reshape(1, -1)
        if low_features.shape[1] != 3: # Basic sanity check
            print(f"Warning: Low features shape is {low_features.shape}, expected (1,3). Adjusting.")
            # Attempt a reasonable default or padding/truncation if necessary
            # For now, if it's not (1,3), this might indicate an issue in _get_obs or concatenation
            # Fallback to zeros if shape is incorrect.
            low_features = np.zeros((1,3), dtype=np.float32)


        # Ensure agent_pos and goal_pos are tuples for grid_to_graph
        agent_pos_tuple = tuple(map(int, self.env.agent_pos)) if self.env.agent_pos else None
        goal_pos_tuple = tuple(map(int, self.env.goal_pos)) if self.env.goal_pos else None
        
        return grid_to_graph(grid, agent_pos_tuple, goal_pos_tuple, low_features)

