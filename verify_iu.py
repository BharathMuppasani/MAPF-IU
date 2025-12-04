
import numpy as np
import time
from fix import fix_collisions, InfoSharingTracker
from utils.env_utils import Environment, AgentEnv

class MockEnv:
    def __init__(self, grid, agent_pos, goal_pos):
        self.grid = grid
        self.agent_pos = agent_pos
        self.goal_pos = goal_pos

class MockAgentEnv:
    def __init__(self, grid, agent_pos, goal_pos):
        self.env = MockEnv(grid, agent_pos, goal_pos)

def verify_iu():
    print("Verifying IU Calculation...")
    
    # 10x10 grid
    grid = np.zeros((10, 10), dtype=int)
    
    # Agent 1: (0,0) -> (0,5)
    # Agent 2: (0,5) -> (0,0)
    # They will collide at (0,2)-(0,3)
    
    agent1_start = (0, 0)
    agent1_goal = (0, 5)
    agent2_start = (0, 5)
    agent2_goal = (0, 0)
    
    # Initial plans (naive)
    plan1 = [3, 3, 3, 3, 3] # RIGHT x 5
    traj1 = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5)]
    
    plan2 = [2, 2, 2, 2, 2] # LEFT x 5
    traj2 = [(0,5), (0,4), (0,3), (0,2), (0,1), (0,0)]
    
    initial_plans = [plan1, plan2]
    initial_trajs = [traj1, traj2]
    
    agent_envs = [
        MockAgentEnv(grid, agent1_start, agent1_goal),
        MockAgentEnv(grid, agent2_start, agent2_goal)
    ]
    
    # Run fix_collisions
    # Force Joint A* by disabling other strategies or setting them to fail?
    # Or just let it run default (Static -> Dynamic -> Joint)
    # Static might fail if window is small?
    
    print("\n--- Test 1: Default Strategy (Static/Dynamic/Joint) ---")
    run_counters = {}
    fix_collisions(
        initial_plans, initial_trajs, agent_envs,
        model=None, run_counters=run_counters, device="cpu",
        verbose=True,
        max_passes=5
    )
    
    # We expect some IU to be recorded.
    # Since we can't easily capture the internal tracker from here without modifying fix_collisions to return it,
    # we rely on stdout (verbose=True prints report).
    # OR we can monkeypatch InfoSharingTracker to capture the instance.
    
    print("\n--- Test 2: Force Yield (if possible) ---")
    # To force yield, we might need a specific scenario where static/dynamic fail.
    # Or we can just check if the code runs without error.

if __name__ == "__main__":
    # Monkeypatch InfoSharingTracker to inspect it
    original_init = InfoSharingTracker.__init__
    last_tracker = None
    
    def new_init(self):
        original_init(self)
        global last_tracker
        last_tracker = self
        
    InfoSharingTracker.__init__ = new_init
    
    verify_iu()
    
    if last_tracker:
        print("\nCaptured Tracker Metrics:")
        print(last_tracker.to_dict())
        
        # Assertions
        iu = last_tracker.to_dict()
        print(f"Total IU: {iu['totalInformationLoadIU']}")
        print(f"Joint A* IU: {iu['jointAStarIU']}")
        print(f"Parking Rejected IU: {iu['parkingRejectedIU']}")
        print(f"Revised Submission IU: {iu['revisedSubmissionIU']}")
        
        if iu['totalInformationLoadIU'] > 0:
            print("SUCCESS: IU recorded.")
        else:
            print("WARNING: No IU recorded (might be expected if no resolution found or no info shared).")
