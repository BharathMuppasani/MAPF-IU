
import numpy as np
import torch
from utils.env_utils import analyze_collisions
from fix import fix_collisions
from utils.grid_env_wrapper import GridEnvWrapper
from dqn.dqn import ResNetDQN
import copy

def test_stay_at_target_collision():
    print("Testing Stay-At-Target Collision Logic...")
    
    # Scenario:
    # Agent 1: Starts at (0,0), Goal at (0,2). Path: (0,0)->(0,1)->(0,2). Finishes at t=2.
    # Agent 2: Starts at (0,4), Goal at (0,0). Moves left.
    # Path: (0,4)->(0,3)->(0,2)->(0,1)->(0,0).
    # Collision should happen at t=2 (Agent 2 hits Agent 1's goal at (0,2) just as A1 arrives? No, A1 arrives at t=2)
    # Let's adjust.
    # A1: (0,0) [t=0], (0,1) [t=1], (0,2) [t=2] (Stay)
    # A2: (0,4) [t=0], (0,3) [t=1], (0,2) [t=2] ...
    # At t=2, both are at (0,2). Vertex collision.
    
    # Case 2: A2 arrives at (0,2) at t=3.
    # A1 is at (0,2) for t>=2.
    # A2: ... (0,3) [t=2], (0,2) [t=3].
    # Collision at t=3.
    
    traj1 = [(0,0), (0,1), (0,2)]
    traj2 = [(0,4), (0,3), (0,3), (0,2)] # Arrives at (0,2) at t=3
    
    goals = [(0,2), (0,0)]
    
    trajs = [traj1, traj2]
    
    collisions = analyze_collisions(trajs, goals)
    print(f"Collisions found: {len(collisions)}")
    for c in collisions:
        print(c)
        
    # Expectation: Collision at t=3 (Vertex)
    has_t3_collision = any(c['time'] == 3 and c['type'] == 'vertex' for c in collisions)
    
    if has_t3_collision:
        print("SUCCESS: Detected collision with finished agent at t=3.")
    else:
        print("FAILURE: Did not detect collision at t=3.")

    # Also check t=2 if they overlap exactly when one finishes
    traj3 = [(0,4), (0,3), (0,2)] # Arrives t=2
    trajs_b = [traj1, traj3]
    collisions_b = analyze_collisions(trajs_b, goals)
    has_t2_collision = any(c['time'] == 2 and c['type'] == 'vertex' for c in collisions_b)
    if has_t2_collision:
        print("SUCCESS: Detected collision at arrival time t=2.")
    else:
         print("FAILURE: Did not detect collision at t=2.")

if __name__ == "__main__":
    test_stay_at_target_collision()
