#!/usr/bin/env python3
"""
Convert MAPF benchmark scenario files to custom .txt format

Usage:
    python convert_scen_to_txt.py --scen_file <path> --num_agents 10 --output <output_file>

Example:
    python convert_scen_to_txt.py --scen_file test_data/maps/scen-even/random-32-32-10-even-1.scen --num_agents 10 --output test_data/maps/benchmark_converted/scene_even_10agents.txt
    python scripts/convert_scen_to_txt.py --scen_file test_data/maps/random-32-32-20/scen-even/random-32-32-10-even-1.scen --num_agents 10 --output test_data/maps/random-32-32-20/benchmark_converted/scene_even1_10agents.txt
"""

import argparse
import os
import numpy as np
from pathlib import Path


def load_map_file(map_filepath):
    """
    Load a map from MAPF benchmark .map file.
    Returns: (grid, height, width)
    """
    with open(map_filepath, 'r') as f:
        lines = f.readlines()

    # Parse header
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])

    # Parse grid (starting from line 4, index 3)
    grid = []
    for line in lines[4:4+height]:
        row = []
        for char in line.strip():
            if char == '.':
                row.append(0)  # Free cell
            elif char == '@':
                row.append(-1)  # Obstacle
            elif char == 'T':
                row.append(-1)  # Obstacle
        grid.append(row)

    return grid, height, width


def load_scen_file(scen_filepath):
    """
    Load scenarios from .scen file.
    Returns: list of tuples (start_row, start_col, goal_row, goal_col)

    NOTE: .scen files use (x, y) format where x=column, y=row
    We convert to (row, col) format for the .txt output
    """
    scenarios = []
    with open(scen_filepath, 'r') as f:
        lines = f.readlines()

    # Skip header (version 1)
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 8:
            # bucket, map_file, height, width, start_x, start_y, goal_x, goal_y, optimal_distance
            start_x = int(parts[4])  # column
            start_y = int(parts[5])  # row
            goal_x = int(parts[6])   # column
            goal_y = int(parts[7])   # row

            # Convert to (row, col) format for .txt file
            scenarios.append((start_y, start_x, goal_y, goal_x))

    return scenarios


def convert_to_txt(grid, height, width, scenarios, num_agents, output_filepath):
    """
    Convert grid and scenarios to custom .txt format.
    Writes to output file.

    Note: scenarios are already in (row, col) format from load_scen_file()
    """
    with open(output_filepath, 'w') as f:
        # Write header
        f.write(f"{height} {width}\n")

        # Write grid
        for row in grid:
            row_str = ' '.join('@' if cell == -1 else '.' for cell in row)
            f.write(row_str + "\n")

        # Write number of agents
        f.write(f"{num_agents}\n")

        # Write first num_agents scenarios
        for i in range(min(num_agents, len(scenarios))):
            start_row, start_col, goal_row, goal_col = scenarios[i]
            f.write(f"{start_row} {start_col} {goal_row} {goal_col}\n")

    print(f"✓ Converted {num_agents} agents to: {output_filepath}")


def main():
    parser = argparse.ArgumentParser(description='Convert MAPF benchmark scenarios to .txt format')
    parser.add_argument('--scen_file', required=True, help='Path to .scen file')
    parser.add_argument('--map_file', help='Path to .map file (if different folder). Auto-detected if not provided.')
    parser.add_argument('--num_agents', type=int, default=10, help='Number of agents to extract (default: 10)')
    parser.add_argument('--output', required=True, help='Output .txt file path')

    args = parser.parse_args()

    # Find map file
    if args.map_file:
        map_filepath = args.map_file
    else:
        # Try to find map in same directory as scen file
        scen_dir = os.path.dirname(args.scen_file)
        # Extract map name from scen file (e.g., "random-32-32-10" from "random-32-32-10-even-1.scen")
        scen_basename = os.path.basename(args.scen_file)
        map_basename = scen_basename.split('-even-')[0] + '.map'
        map_filepath = os.path.join(scen_dir, map_basename)

    # Verify files exist
    if not os.path.exists(args.scen_file):
        print(f"ERROR: Scenario file not found: {args.scen_file}")
        return

    if not os.path.exists(map_filepath):
        print(f"ERROR: Map file not found: {map_filepath}")
        print(f"       Tried: {map_filepath}")
        return

    # Load data
    print(f"Loading map from: {map_filepath}")
    grid, height, width = load_map_file(map_filepath)

    print(f"Loading scenarios from: {args.scen_file}")
    scenarios = load_scen_file(args.scen_file)
    print(f"Found {len(scenarios)} scenarios in file")

    # Convert and save
    if args.num_agents > len(scenarios):
        print(f"WARNING: Requested {args.num_agents} agents but only {len(scenarios)} scenarios available")
        args.num_agents = len(scenarios)

    convert_to_txt(grid, height, width, scenarios, args.num_agents, args.output)

    # Print info
    print(f"\n📊 Conversion Summary:")
    print(f"   Grid size: {width}x{height}")
    print(f"   Agents: {args.num_agents}")
    print(f"   Output: {args.output}")
    print(f"\n✓ Ready to run:")
    print(f"   python run_exp.py --strategy best --info all --search_type astar --algo dqn --map_file {args.output} --timeout 300")


if __name__ == '__main__':
    main()
