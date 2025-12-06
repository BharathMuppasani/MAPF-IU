#!/usr/bin/env python3
"""
Generate ICAPS Test Dataset

Creates test datasets for MAPF benchmarks with various agent configurations.

Usage:
    python scripts/generate_icaps_dataset.py

Output structure:
    test_data/icaps_test/
    ├── random-32-32-20/
    │   ├── 10_agents/
    │   │   ├── test_01.txt
    │   │   ├── test_02.txt
    │   │   └── ... (10 files)
    │   ├── 20_agents/
    │   └── ...
    ├── random-64-64-20/
    ├── den312d/
    └── warehouse/
"""

import os
import random
import argparse
from pathlib import Path


# Configuration
RANDOM_AGENT_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 128, 256]
COMPLEX_AGENT_COUNTS = [8, 16, 32, 64, 128, 256]

RANDOM_TEST_FILES = 20
COMPLEX_TEST_FILES = 20

# Map configurations: (folder_name, output_name, map_name_pattern, is_complex)
MAP_CONFIGS = [
    ("random-32-32-20", "random-32-32-20", "random-32-32-20", False),
    ("random-64-64-20", "random-64-64-20", "random-64-64-20", False),
    ("den312d", "den312d", "den312d", True),
    ("warehouse-10-20-10-2-2", "warehouse", "warehouse-10-20-10-2-2", True),
]


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
            start_x = int(parts[4])  # column
            start_y = int(parts[5])  # row
            goal_x = int(parts[6])   # column
            goal_y = int(parts[7])   # row

            # Convert to (row, col) format for .txt file
            scenarios.append((start_y, start_x, goal_y, goal_x))

    return scenarios


def load_all_scenarios(map_folder):
    """
    Load all scenarios from all .scen files in a map folder.
    Returns: list of all scenarios
    """
    scen_folder = map_folder / "scen-even"
    all_scenarios = []

    scen_files = sorted(scen_folder.glob("*.scen"))
    for scen_file in scen_files:
        scenarios = load_scen_file(scen_file)
        all_scenarios.extend(scenarios)

    return all_scenarios


def write_test_file(grid, height, width, scenarios, output_path):
    """
    Write a test file in custom .txt format.
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write(f"{height} {width}\n")

        # Write grid
        for row in grid:
            row_str = ' '.join('@' if cell == -1 else '.' for cell in row)
            f.write(row_str + "\n")

        # Write number of agents
        f.write(f"{len(scenarios)}\n")

        # Write scenarios
        for start_row, start_col, goal_row, goal_col in scenarios:
            f.write(f"{start_row} {start_col} {goal_row} {goal_col}\n")


def validate_scenarios(scenarios):
    """
    Validate that selected scenarios have no conflicts.

    Returns:
        True if valid, False if there are conflicts
    """
    start_positions = set()
    goal_positions = set()

    for start_row, start_col, goal_row, goal_col in scenarios:
        start_pos = (start_row, start_col)
        goal_pos = (goal_row, goal_col)

        # Check for duplicate start positions
        if start_pos in start_positions:
            return False
        start_positions.add(start_pos)

        # Check for duplicate goal positions
        if goal_pos in goal_positions:
            return False
        goal_positions.add(goal_pos)

        # Check if start equals goal (optional, usually valid but uncommon)
        # if start_pos == goal_pos:
        #     return False

    return True


def generate_random_test_file(all_scenarios, num_agents, used_combinations=None):
    """
    Generate a random test file by selecting random scenarios incrementally.
    Ensures no duplicate start or goal positions.

    Args:
        all_scenarios: List of all available scenarios
        num_agents: Number of agents to select
        used_combinations: Set of frozensets of scenario indices already used (to avoid duplicates)

    Returns:
        List of selected scenarios and the combination key
    """
    if used_combinations is None:
        used_combinations = set()

    max_attempts = 100  # Attempts to find a unique combination
    for _ in range(max_attempts):
        # Build selection incrementally to avoid conflicts
        selected_indices = []
        used_starts = set()
        used_goals = set()

        # Shuffle indices for random selection
        available_indices = list(range(len(all_scenarios)))
        random.shuffle(available_indices)

        for idx in available_indices:
            if len(selected_indices) >= num_agents:
                break

            start_row, start_col, goal_row, goal_col = all_scenarios[idx]
            start_pos = (start_row, start_col)
            goal_pos = (goal_row, goal_col)

            # Skip if start or goal already used
            if start_pos in used_starts or goal_pos in used_goals:
                continue

            # Add to selection
            selected_indices.append(idx)
            used_starts.add(start_pos)
            used_goals.add(goal_pos)

        # Check if we got enough agents
        if len(selected_indices) < num_agents:
            raise ValueError(f"Not enough non-conflicting scenarios: need {num_agents}, found {len(selected_indices)}")

        combination_key = frozenset(selected_indices)

        # Check if this exact combination was used before
        if combination_key not in used_combinations:
            used_combinations.add(combination_key)
            selected = [all_scenarios[i] for i in selected_indices]
            return selected, combination_key

    # If we couldn't find a unique combination, just return the last valid one
    selected = [all_scenarios[i] for i in selected_indices]
    return selected, frozenset(selected_indices)


def generate_dataset_for_map(base_path, map_folder_name, output_name, map_name_pattern, is_complex, output_base):
    """
    Generate test dataset for a single map type.
    """
    map_folder = base_path / map_folder_name
    scen_folder = map_folder / "scen-even"

    # Find map file
    map_file = scen_folder / f"{map_name_pattern}.map"
    if not map_file.exists():
        # Try parent folder
        map_file = map_folder / f"{map_name_pattern}.map"

    if not map_file.exists():
        print(f"  ⚠️  Map file not found: {map_file}")
        return False

    # Load map
    grid, height, width = load_map_file(map_file)
    print(f"  Map size: {width}x{height}")

    # Load all scenarios
    all_scenarios = load_all_scenarios(map_folder)
    print(f"  Total scenarios available: {len(all_scenarios)}")

    if len(all_scenarios) == 0:
        print(f"  ⚠️  No scenarios found for {map_folder_name}")
        return False

    # Create output folder for this map
    map_output_folder = output_base / output_name
    map_output_folder.mkdir(parents=True, exist_ok=True)

    # Select agent counts and number of test files based on map type
    agent_counts = COMPLEX_AGENT_COUNTS if is_complex else RANDOM_AGENT_COUNTS
    num_test_files = COMPLEX_TEST_FILES if is_complex else RANDOM_TEST_FILES

    # Generate test files for each agent count
    for num_agents in agent_counts:
        if num_agents > len(all_scenarios):
            print(f"  ⚠️  Skipping {num_agents} agents (only {len(all_scenarios)} scenarios available)")
            continue

        # Create folder for this agent count
        agent_folder = map_output_folder / f"{num_agents}_agents"
        agent_folder.mkdir(parents=True, exist_ok=True)

        # Track used combinations to avoid duplicates
        used_combinations = set()

        # Generate test files
        for file_idx in range(1, num_test_files + 1):
            selected_scenarios, _ = generate_random_test_file(
                all_scenarios, num_agents, used_combinations
            )

            output_file = agent_folder / f"test_{file_idx:02d}.txt"
            write_test_file(grid, height, width, selected_scenarios, output_file)

        print(f"    ✓ Generated {num_test_files} files for {num_agents} agents")

    return True


def main():
    parser = argparse.ArgumentParser(description='Generate ICAPS test dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='test_data/icaps_test',
                        help='Output directory (relative to project root)')
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Get project root (assuming script is in scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Set paths
    base_path = project_root / "test_data" / "maps"
    output_base = project_root / args.output

    print(f"ICAPS Test Dataset Generator")
    print(f"=" * 50)
    print(f"Output directory: {output_base}")
    print(f"Random maps agent counts: {RANDOM_AGENT_COUNTS}")
    print(f"Complex maps agent counts: {COMPLEX_AGENT_COUNTS}")
    print(f"Random maps test files: {RANDOM_TEST_FILES}")
    print(f"Complex maps test files: {COMPLEX_TEST_FILES}")
    print(f"Random seed: {args.seed}")
    print(f"=" * 50)

    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)

    # Generate datasets for each map
    success_count = 0
    for map_folder_name, output_name, map_name_pattern, is_complex in MAP_CONFIGS:
        print(f"\n📁 Processing: {output_name} ({'complex' if is_complex else 'random'})")

        if generate_dataset_for_map(base_path, map_folder_name, output_name,
                                    map_name_pattern, is_complex, output_base):
            success_count += 1

    # Summary
    print(f"\n{'=' * 50}")
    print(f"✅ Dataset generation complete!")
    print(f"   Maps processed: {success_count}/{len(MAP_CONFIGS)}")
    print(f"   Output location: {output_base}")

    # Print usage example
    print(f"\n📌 Example usage:")
    print(f"   python run_exp.py --map_file {output_base}/random-32-32-20/10_agents/test_01.txt --timeout 300")


if __name__ == '__main__':
    main()
