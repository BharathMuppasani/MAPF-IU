#!/usr/bin/env python3
"""
Convert ICAPS test .txt files to .pth format for DCC/MAPF-LNS2 testing.

Input format (.txt):
    Line 1: height width
    Lines 2-(height+1): grid rows with '.' (free) and '@' (obstacle)
    Line (height+2): num_agents
    Next num_agents lines: start_row start_col goal_row goal_col

Output format (.pth):
    List of tuples: [(map, agents_pos, goals_pos), ...]
    - map: np.ndarray (H, W), dtype=int → 0=free, 1=obstacle
    - agents_pos: np.ndarray (N, 2), dtype=int → [x, y] per agent (col, row)
    - goals_pos: np.ndarray (N, 2), dtype=int → [x, y] per agent (col, row)

Usage:
    python scripts/convert_to_pth.py --input test_data/icaps_test --output test_set/icaps
    python scripts/convert_to_pth.py --input test_data/icaps_test/random-32-32-20/10_agents --output test_set/random32_10agents.pth
"""

import argparse
import pickle
import numpy as np
from pathlib import Path


def load_txt_file(filepath):
    """
    Load a single .txt problem file.

    Returns:
        map_array: np.ndarray (H, W), 0=free, 1=obstacle
        agents_pos: np.ndarray (N, 2), [x, y] format (col, row)
        goals_pos: np.ndarray (N, 2), [x, y] format (col, row)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse header
    height, width = map(int, lines[0].strip().split())

    # Parse grid
    map_array = np.zeros((height, width), dtype=int)
    for row_idx in range(height):
        row_data = lines[1 + row_idx].strip().split()
        for col_idx, cell in enumerate(row_data):
            if cell == '@':
                map_array[row_idx, col_idx] = 1  # obstacle
            # '.' remains 0 (free)

    # Parse agents
    num_agents = int(lines[1 + height].strip())
    agents_pos = []
    goals_pos = []

    for i in range(num_agents):
        parts = lines[2 + height + i].strip().split()
        start_row, start_col, goal_row, goal_col = map(int, parts)
        # Convert to [x, y] format (col, row)
        agents_pos.append([start_col, start_row])
        goals_pos.append([goal_col, goal_row])

    return map_array, np.array(agents_pos, dtype=int), np.array(goals_pos, dtype=int)


def convert_folder_to_pth(input_folder, output_path):
    """
    Convert all .txt files in a folder to a single .pth file.
    Each .txt file becomes one instance in the list.
    """
    input_folder = Path(input_folder)
    txt_files = sorted(input_folder.glob("*.txt"))

    if not txt_files:
        print(f"  No .txt files found in {input_folder}")
        return False

    instances = []
    for txt_file in txt_files:
        try:
            map_array, agents_pos, goals_pos = load_txt_file(txt_file)
            instances.append((map_array, agents_pos, goals_pos))
        except Exception as e:
            print(f"  Error loading {txt_file}: {e}")
            continue

    if not instances:
        return False

    # Save as .pth
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(instances, f)

    print(f"  Saved {len(instances)} instances to {output_path}")
    return True


def convert_icaps_dataset(input_base, output_base):
    """
    Convert entire ICAPS test dataset.

    Structure:
        input_base/
        ├── random-32-32-20/
        │   ├── 10_agents/
        │   │   ├── test_01.txt -> combined into one .pth
        │   │   └── ...

        output_base/
        ├── random-32-32-20/
        │   ├── 10_agents.pth  (contains all test instances)
    """
    input_base = Path(input_base)
    output_base = Path(output_base)

    print(f"Converting ICAPS dataset")
    print(f"Input: {input_base}")
    print(f"Output: {output_base}")
    print("=" * 50)

    # Find all map folders
    for map_folder in sorted(input_base.iterdir()):
        if not map_folder.is_dir():
            continue

        map_name = map_folder.name
        print(f"\nMap: {map_name}")

        # Find all agent count folders
        for agent_folder in sorted(map_folder.iterdir()):
            if not agent_folder.is_dir():
                continue

            agent_config = agent_folder.name  # e.g., "10_agents"
            output_file = output_base / map_name / f"{agent_config}.pth"

            print(f"  {agent_config}...")
            convert_folder_to_pth(agent_folder, output_file)

    print("\n" + "=" * 50)
    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description='Convert ICAPS .txt files to .pth format')
    parser.add_argument('--input', required=True, help='Input folder (icaps_test or specific agent folder)')
    parser.add_argument('--output', required=True, help='Output path (.pth file or folder)')
    parser.add_argument('--single', action='store_true', help='Convert single folder to single .pth')

    args = parser.parse_args()

    if args.single:
        # Convert single folder
        convert_folder_to_pth(args.input, args.output)
    else:
        # Convert entire dataset
        convert_icaps_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
