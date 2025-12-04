
import numpy as np
from fix import fix_collisions
from utils.grid_env_wrapper import GridEnvWrapper
from dqn.dqn import ResNetDQN
import torch
import copy

def test_yield_strategy():
    print("Testing Yield Strategy...")
    
    # Scenario:
    # Agent 1: At goal (0,2) from t=0. (Finished)
    # Agent 2: Needs to pass through (0,2).
    # Path: (0,0)->(0,1)->(0,2)->(0,3).
    
    # Setup
    grid_size = 5
    env = GridEnvWrapper(grid_size=grid_size, generation_mode=None)
    env.env.grid = np.zeros((grid_size, grid_size), dtype=int)
    # Create a corridor: Walls at row 1, but leave (1,2) open for yielding
    for c in range(grid_size):
        if c != 2:
            env.env.grid[(1, c)] = -1 # Wall

    
    # Agent 1
    env1 = copy.deepcopy(env)
    env1.env.agent_pos = (0,2)
    env1.env.goal_pos = (0,2)
    env1.env.grid[(0,2)] = 1 # Agent 1 start/goal
    env1.current_observation = env1._get_obs()
    
    # Agent 2
    env2 = copy.deepcopy(env)
    env2.env.agent_pos = (0,0)
    env2.env.goal_pos = (0,3)
    env2.env.grid[(0,0)] = 1
    env2.env.grid[(0,3)] = 2
    env2.current_observation = env2._get_obs()
    
    agent_envs = [env1, env2]
    
    # Initial Plans
    # Agent 1: Stay (Wait)
    plan1 = [4] * 10 # Wait 10 steps
    traj1 = [(0,2)] * 11
    
    # Agent 2: Move Right
    # (0,0)->(0,1)->(0,2)->(0,3)
    plan2 = [3, 3, 3] 
    traj2 = [(0,0), (0,1), (0,2), (0,3)]
    
    initial_plans = [plan1, plan2]
    initial_trajs = [traj1, traj2]
    
    run_counters = {'collisions_total': 0}
    device = torch.device("cpu")
    model = None # Not needed for yield if hardcoded, but fix_collisions needs it.
    # We need a dummy model or ensure plan_with_search isn't called for Yield?
    # Yield uses simulate_plan, not plan_with_search.
    # But if Yield fails, it might try other strategies which use model.
    # Let's load a dummy model if possible or mock it.
    
    # Mock model
    class MockModel:
        def __call__(self, *args, **kwargs):
            # Return (logits, value, hidden, cell)
            return (torch.randn(1, 5), torch.randn(1, 1), None, None)
    model = MockModel()

    print("Running fix_collisions...")
    final_plans, final_trajs, _, _, log_data = fix_collisions(
        initial_plans, initial_trajs, agent_envs, model, run_counters, device,
        replan_strategy="best", info_setting="all", verbose=True,
        time_limit=10
    )
    
    # Check if Agent 1 yielded
    # Agent 1 original traj: [(0,2), (0,2), ...]
    # Expected: [(0,2), (neighbor), ..., (neighbor), (0,2)]
    
    traj1_final = final_trajs[0]
    print(f"Agent 1 Final Trajectory: {traj1_final}")
    
    has_moved = any(t != (0,2) for t in traj1_final)
    if has_moved:
        print("SUCCESS: Agent 1 yielded (moved from goal).")
    else:
        print("FAILURE: Agent 1 did not yield.")

if __name__ == "__main__":
    test_yield_strategy()
