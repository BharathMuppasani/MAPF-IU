#!/usr/bin/env python3
"""
MAPF Visualization Tool
Visualize multi-agent pathfinding results from JSON log files.

Usage:
    python visualize_robo_lab.py <log_file> [options]
    
Examples:
    python visualize_robo_lab.py test_data/yield_verification_results/info_test_robo_lab_success.json
    python visualize_robo_lab.py logs/my_result.json --cell-size 40 --speed 5 --delay 80
"""

import json
import argparse
import sys
import os
import numpy as np
from utils.viz_utils import MultiAgentPathVisualizer

def load_and_visualize(log_file, cell_size=50, frames_per_transition=10, delay=100):
    """Load JSON log and launch visualization."""
    
    # Load the JSON log
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
        
    with open(log_file, 'r') as f:
        log_data = json.load(f)

    # Extract grid size
    grid_size = log_data['environment']['gridSize']
    rows, cols = grid_size

    # Create grid with obstacles
    grid = np.zeros((rows, cols), dtype=int)
    for obs in log_data['environment']['obstacles']:
        r, c = obs['cell']
        grid[r, c] = -1

    # Extract agent starts and goals
    agent_starts = []
    agent_goals = []
    for agent in log_data['agents']:
        start = tuple(agent['initialState']['cell'])
        goal = tuple(agent['goalState']['cell'])
        agent_starts.append(start)
        agent_goals.append(goal)

    # Extract final trajectories from the jointPlan
    final_subplan_ids = log_data['jointPlan']['subplans']

    # Find the corresponding subplans
    final_trajectories = []
    for subplan_id in final_subplan_ids:
        # Search in agentSubplans
        found = False
        for subplan in log_data.get('agentSubplans', []):
            if subplan['id'] == subplan_id:
                traj = [tuple(step['cell']) for step in subplan['steps']]
                final_trajectories.append(traj)
                found = True
                break
        
        # If not found, check agentPaths (for original plans)
        if not found:
            for path in log_data.get('agentPaths', []):
                if path['subplanId'] == subplan_id:
                    traj = [tuple(step['cell']) for step in path['steps']]
                    final_trajectories.append(traj)
                    break

    # Create a simple environment object
    class SimpleEnv:
        def __init__(self, grid):
            self.grid = grid

    base_env = SimpleEnv(grid)

    # Print summary
    print(f"📊 Visualizing MAPF Result")
    print(f"   Log file: {log_file}")
    print(f"   Grid size: {rows}x{cols}")
    print(f"   Agents: {len(final_trajectories)}")
    print(f"   Makespan: {log_data['jointPlan']['globalMakespan']} steps")
    print("\n🤖 Agent Details:")
    for i, traj in enumerate(final_trajectories):
        print(f"   Agent {i+1}: {len(traj)} steps | Start={traj[0]} → Goal={traj[-1]}")
    
    print(f"\n⚙️  Animation Settings:")
    print(f"   Cell size: {cell_size}px")
    print(f"   Smoothness: {frames_per_transition} frames/step")
    print(f"   Delay: {delay}ms")
    print("\n🎬 Launching visualization window...\n")

    # Launch visualizer
    title = f"MAPF Visualization - {os.path.basename(log_file)}"
    app = MultiAgentPathVisualizer(
        base_env,
        final_trajectories,
        agent_starts,
        agent_goals,
        cell_size=cell_size,
        frames_per_transition=frames_per_transition,
        delay=delay
    )
    app.title(title)
    app.mainloop()

def main():
    parser = argparse.ArgumentParser(
        description='Visualize MAPF experiment results from JSON logs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s test_data/results/robo_lab.json
  %(prog)s logs/experiment_001.json --cell-size 60 --speed 8
  %(prog)s my_result.json --delay 50
        """
    )
    
    parser.add_argument(
        'log_file',
        help='Path to the JSON log file from run_exp.py'
    )
    
    parser.add_argument(
        '--cell-size',
        type=int,
        default=50,
        metavar='SIZE',
        help='Size of each grid cell in pixels (default: 50)'
    )
    
    parser.add_argument(
        '--speed',
        type=int,
        default=10,
        metavar='FRAMES',
        help='Animation smoothness: frames per transition (default: 10, lower=faster)'
    )
    
    parser.add_argument(
        '--delay',
        type=int,
        default=100,
        metavar='MS',
        help='Delay between frames in milliseconds (default: 100, lower=faster)'
    )
    
    args = parser.parse_args()
    
    load_and_visualize(
        args.log_file,
        cell_size=args.cell_size,
        frames_per_transition=args.speed,
        delay=args.delay
    )

if __name__ == '__main__':
    main()
