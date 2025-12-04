
from utils.env_utils import analyze_collisions

def test_simple():
    print("Testing Simple Stay-At-Target...")
    traj1 = [(0,0), (0,1), (0,2)] # Ends at t=2
    traj2 = [(0,4), (0,3), (0,3), (0,2)] # Arrives at (0,2) at t=3
    goals = [(0,2), (0,0)]
    
    trajs = [traj1, traj2]
    
    collisions = analyze_collisions(trajs, goals)
    print(f"Collisions: {len(collisions)}")
    for c in collisions:
        print(c)

if __name__ == "__main__":
    test_simple()
