import unittest
import cpp_collision
import sys

class TestCppCollision(unittest.TestCase):
    def setUp(self):
        self.starts = [(0, 0), (0, 4)]
        self.goals = [(0, 4), (0, 0)]
        self.grid = [
            [0, 0, 0, 0, 0],
            [0, -1, -1, -1, 0],
            [0, 0, 0, 0, 0]
        ]

    def test_yield_on_start(self):
        # Agent 1 at (0,0), Agent 2 moves from (0,4) to (0,0) AND THEN MOVES AWAY
        # Agent 1 should yield at start
        
        # Trajectories:
        # Agent 1: stays at (0,0)
        # Agent 2: (0,4)->(0,3)->(0,2)->(0,1)->(0,0)->(0,1)
        trajectories = [
            [(0,0)] * 6,
            [(0,4), (0,3), (0,2), (0,1), (0,0), (0,1)]
        ]
        
        # Collision at t=4 at (0,0)
        collision_time = 4
        
        # Agent 1 yields
        yield_plan = cpp_collision.resolve_yield_on_start_cpp(
            self.grid,
            collision_time,
            1, # Agent 1
            (0,0),
            trajectories,
            self.starts,
            4 # radius
        )
        
        self.assertTrue(yield_plan.success)
        print(f"Yield on Start Parking: {yield_plan.parking_cell.r}, {yield_plan.parking_cell.c}")

    def test_yield_on_goal(self):
        # Agent 1 at goal (0,4), Agent 2 needs to pass through (0,4) AND MOVE AWAY
        # Agent 1 should yield
        
        # Trajectories:
        # Agent 1: at goal (0,4)
        # Agent 2: (0,0)->(0,1)->(0,2)->(0,3)->(0,4)->(0,3)
        trajectories = [
            [(0,4)] * 6,
            [(0,0), (0,1), (0,2), (0,3), (0,4), (0,3)]
        ]
        
        collision_time = 4
        
        yield_plan = cpp_collision.resolve_yield_on_goal_cpp(
            self.grid,
            collision_time,
            1, # Agent 1
            (0,4),
            trajectories,
            self.starts,
            4
        )
        
        self.assertTrue(yield_plan.success)
        print(f"Yield on Goal Parking: {yield_plan.parking_cell.r}, {yield_plan.parking_cell.c}")

    def test_wait_yield(self):
        # Agent 1 moves (0,0)->(0,1)->(0,2)
        # Agent 2 moves (0,2)->(0,1)->(0,0)
        # To avoid swap, Agent 1 waits.
        # But if they are head-on in a corridor, waiting just delays the swap.
        # We need a scenario where waiting lets the other agent pass.
        # Crossing scenario:
        # A1: (0,0)->(0,1)->(0,2)
        # A2: (1,1)->(0,1)->(-1,1) (crosses at (0,1))
        
        # Grid is 3x5.
        # A1: (1,0)->(1,1)->(1,2)
        # A2: (0,1)->(1,1)->(2,1)
        # Collision at t=1 at (1,1).
        
        trajectories = [
            [(1,0), (1,1), (1,2)],
            [(0,1), (1,1), (2,1)]
        ]
        
        # If A1 waits 1 step: (1,0)->(1,0)->(1,1)->(1,2)
        # t=0: A1(1,0), A2(0,1)
        # t=1: A1(1,0), A2(1,1) (A2 takes the spot)
        # t=2: A1(1,1), A2(2,1) (A2 moves away, A1 takes spot)
        # Safe!
        
        current_traj = trajectories[0]
        
        yield_plan = cpp_collision.resolve_wait_yield_cpp(
            1, # collision time
            1, # Agent 1
            current_traj,
            trajectories,
            self.starts
        )
        
        self.assertTrue(yield_plan.success)
        self.assertEqual(len(yield_plan.path_to_goal), 4) # Inserted 1 wait
        print("Wait yield successful")

if __name__ == '__main__':
    unittest.main()
