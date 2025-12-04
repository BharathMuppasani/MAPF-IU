import pickle

from utils.viz_utils import MultiAgentPathVisualizer
from utils.grid_env_wrapper import GridEnvWrapper

# 1. Load all 10 runs
with open('test_data/results_random/random_results_do_bfs_dqn_8.pkl', 'rb') as f:
    runs = pickle.load(f)   # runs is a list of 10 dicts

# 2. Pick which run to visualize (0‐9)
run_idx = 2
rec     = runs[run_idx]

# 3. Extract the pieces
#    Adjust these keys to match exactly what you saved!
trajectories = rec['final_trajectories']  # or 'agent_trajectories'
agent_envs    = rec['agent_envs']

num_agents = len(trajectories)

# --------- SCRIMP --------- #
# for visualizing SCRIMP plans, uncomment the following code:

# # # # 1. Load all 10 runs
# with open('simulation_data/results_paper/simulation_results_scrimp_32.pkl', 'rb') as f:
#     runs_temp = pickle.load(f)   # runs is a list of 10 dicts

# run_idx_temp = run_idx
# rec_temp     = runs_temp[run_idx_temp]

# trajectories_temp = rec_temp['trajectories']  # or 'agent_trajectories'
# trajectories = []


# for key, item in trajectories_temp.items():
#     trajectories.append(item['positions'])

# --------- SCRIMP --------- #


# 4. Derive starts & goals
agent_positions = [traj[0] for traj in trajectories]
goal_positions  = [env_i.env.goal_pos for env_i in agent_envs]

# 5. Create a “base” env just for drawing the grid
base_env = agent_envs[0].env

# 6. Launch the visualizer
app = MultiAgentPathVisualizer(
    base_env, 
    trajectories,
    agent_positions,
    goal_positions,
    cell_size=40,            # px per cell
    frames_per_transition=5, # animation smoothness
    delay=50                 # ms between frames
)
app.title(f"Simulation Run #{run_idx}")
app.mainloop()