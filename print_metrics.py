#!/usr/bin/env python3
"""
Print metrics from a MAPF log file in a nicely formatted way.

Usage:
    python print_metrics.py <log_file.json>
"""

import json
import sys
import os


def print_separator(char='=', width=60):
    print(char * width)


def print_header(title, char='=', width=60):
    print()
    print_separator(char, width)
    print(f" {title}")
    print_separator(char, width)


def print_metrics(log_file_path):
    """Load and print metrics from a MAPF log file."""

    if not os.path.exists(log_file_path):
        print(f"Error: File not found: {log_file_path}")
        sys.exit(1)

    with open(log_file_path, 'r') as f:
        data = json.load(f)

    print_header("MAPF METRICS REPORT")
    print(f"Log file: {log_file_path}")

    # Basic metrics from top level
    print_header("EXECUTION SUMMARY", '-')
    print(f"  Total Passes:        {data.get('passes', 'N/A')}")
    print(f"  Total Time:          {data.get('time', 0):.3f} seconds")
    print(f"  Final Collisions:    {data.get('final_collisions', 'N/A')}")

    # Metrics section
    metrics = data.get('metrics', {})
    if metrics:
        print_header("MAPF METRICS", '-')
        print(f"  Makespan:                    {metrics.get('makespan', 'N/A')}")
        print(f"  Sum of Costs:                {metrics.get('sumOfCosts', 'N/A')}")
        print(f"  Sum of Costs (Trimmed):      {metrics.get('sumOfCostsTrimmed', 'N/A')}")
        print(f"  Total Time:                  {metrics.get('totalTime', 0):.3f} seconds")
        print(f"  Agents at Goal:              {metrics.get('agentsAtGoal', 'N/A')}")
        print(f"  Initial Conflicts:           {metrics.get('initialConflicts', 'N/A')}")
        print(f"  Deferred Agents Count:       {metrics.get('deferredAgentsCount', 'N/A')}")

        # Pass breakdown
        passes = metrics.get('passes', {})
        if passes:
            print_header("PASS BREAKDOWN", '-')
            print(f"  Total Passes:      {passes.get('total', 'N/A')}")
            print(f"  Phase 1 Passes:    {passes.get('phase1', 'N/A')}")
            print(f"  Phase 2 Passes:    {passes.get('phase2', 'N/A')}")
            print(f"  Post-Cleanup:      {passes.get('postCleanup', 'N/A')}")

        # Strategy attempts and successes
        strategies = metrics.get('strategiesTried', {})
        if strategies:
            print_header("STRATEGIES TRIED", '-')
            print(f"  {'Strategy':<20} {'Attempts':>10} {'Successes':>10} {'Success %':>10}")
            print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

            total_attempts = 0
            total_successes = 0

            for strategy_name, stats in strategies.items():
                attempts = stats.get('attempts', 0)
                successes = stats.get('successes', 0)
                success_rate = (successes / attempts * 100) if attempts > 0 else 0

                total_attempts += attempts
                total_successes += successes

                # Format strategy name nicely
                display_name = strategy_name.replace('_', ' ').title()
                print(f"  {display_name:<20} {attempts:>10} {successes:>10} {success_rate:>9.1f}%")

            print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
            overall_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
            print(f"  {'TOTAL':<20} {total_attempts:>10} {total_successes:>10} {overall_rate:>9.1f}%")

    # Strategy IU section
    strategy_iu = data.get('strategyIU', {})
    if strategy_iu:
        print_header("STRATEGY IU (Information Units)", '-')
        print(f"  Yield IU:                    {strategy_iu.get('yieldIU', 0)}")
        print(f"  Joint A* IU:                 {strategy_iu.get('jointAstarIU', 0)}")
        print(f"  Joint A* Cell Conflicts:     {strategy_iu.get('jointAstarCellConflicts', 0)}")
        print(f"  Static IU:                   {strategy_iu.get('staticIU', 0)}")
        print(f"    - Blocked Cells IU:        {strategy_iu.get('staticBlockedCellsIU', 0)}")
        print(f"    - Collision Cells IU:      {strategy_iu.get('staticCollisionCellsIU', 0)}")
        print(f"  Resubmission IU:             {strategy_iu.get('resubmissionIU', 0)}")

        # Calculate total strategy IU
        total_strategy_iu = (
            strategy_iu.get('yieldIU', 0) +
            strategy_iu.get('jointAstarIU', 0) +
            strategy_iu.get('staticIU', 0) +
            strategy_iu.get('resubmissionIU', 0)
        )
        print(f"  {'-'*40}")
        print(f"  Total Strategy IU:           {total_strategy_iu}")

    # Info sharing metrics (original)
    info_sharing = data.get('info_sharing', {})
    if info_sharing:
        print_header("INFO SHARING METRICS (Original)", '-')
        print(f"  Initial Submission IU:       {info_sharing.get('initialSubmissionIU', 0)}")
        print(f"  Revised Submission IU:       {info_sharing.get('revisedSubmissionIU', 0)}")
        print(f"  Conflict Alert IU:           {info_sharing.get('conflictAlertIU', 0)}")

        alert_details = info_sharing.get('alertDetailsIU', {})
        if alert_details:
            print(f"    - Static Alerts:           {alert_details.get('static', 0)}")
            print(f"    - Dynamic Alerts:          {alert_details.get('dynamic', 0)}")

        print(f"  Joint A* IU:                 {info_sharing.get('jointAStarIU', 0)}")
        print(f"  Parking Rejected IU:         {info_sharing.get('parkingRejectedIU', 0)}")
        print(f"  {'-'*40}")
        print(f"  Total Information Load IU:   {info_sharing.get('totalInformationLoadIU', 0)}")

    # Environment info
    env = data.get('environment', {})
    if env:
        print_header("ENVIRONMENT", '-')
        grid_size = env.get('gridSize', [])
        if grid_size:
            print(f"  Grid Size:           {grid_size[0]} x {grid_size[1]}")
        obstacles = env.get('obstacles', [])
        print(f"  Obstacles:           {len(obstacles)}")

    # Agents info
    agents = data.get('agents', [])
    if agents:
        print_header("AGENTS", '-')
        print(f"  Total Agents:        {len(agents)}")

    # Joint plan info
    joint_plan = data.get('jointPlan', {})
    if joint_plan:
        print_header("JOINT PLAN", '-')
        print(f"  Global Makespan:     {joint_plan.get('globalMakespan', 'N/A')}")

    print_separator()
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: Please provide a log file path.")
        sys.exit(1)

    log_file_path = sys.argv[1]
    print_metrics(log_file_path)


if __name__ == "__main__":
    main()
