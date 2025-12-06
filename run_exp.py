import numpy as np
import torch
import time
import copy
import os
import pickle
import datetime
import argparse
import json

from utils.grid_env_wrapper import GridEnvWrapper
from dqn.dqn import ResNetDQN
from ppo.ppo import PPOActorCritic
from utils.env_utils import analyze_collisions, simulate_plan
from utils.search_utils import plan_with_search, astar, astar_cpp
from fix import fix_collisions

# ==============================================================================
# SECTION 1: CUSTOM INSTANCE/MAP LOADERS
# ==============================================================================

def load_grid_from_map_file(filepath, default_fill_value=-1):
    """
    Loads a map from a .map file (e.g., from the MAPF benchmark) and ensures
    the output grid is square.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Map file not found: {filepath}")

    with open(filepath, 'r') as f: lines = f.readlines()

    declared_height = int(lines[1].split(' ')[1])
    declared_width = int(lines[2].split(' ')[1])
    
    original_grid_data_list = []
    for line in lines[4:]:
        row = [0 if char == '.' else -1 for char in line.strip()]
        original_grid_data_list.append(row)

    actual_height = len(original_grid_data_list)
    if actual_height != declared_height:
        raise ValueError(f"Declared height ({declared_height}) != actual height ({actual_height}).")

    side_length = max(declared_height, declared_width)
    square_grid_data = np.full((side_length, side_length), default_fill_value, dtype=int)
    square_grid_data[0:declared_height, 0:declared_width] = np.array(original_grid_data_list)

    return square_grid_data, side_length, side_length

def load_mapf_instance_from_txt(filepath):
    """
    Loads a MAPF instance from the user-specified custom .txt file format.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Map file not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 3:
        raise ValueError("File is too short to be a valid instance.")

    try:
        height, width = map(int, lines[0].split())
        grid_lines = lines[1 : 1 + height]
        if len(grid_lines) != height:
            raise ValueError(f"Declared height {height}, found {len(grid_lines)} grid lines.")
        
        grid = np.zeros((height, width), dtype=int)
        for r, row_str in enumerate(grid_lines):
            row_chars = row_str.split()
            if len(row_chars) != width:
                raise ValueError(f"Declared width {width}, but row {r} has {len(row_chars)} cells.")
            for c, char in enumerate(row_chars):
                if char == '@':
                    grid[r, c] = -1

        num_agents = int(lines[1 + height])
        agent_lines = lines[2 + height : 2 + height + num_agents]
        if len(agent_lines) != num_agents:
            raise ValueError(f"Declared {num_agents} agents, but found {len(agent_lines)} agent lines.")

        agents = [{'start': tuple(map(int, line.split()[:2])), 'goal': tuple(map(int, line.split()[2:]))} for line in agent_lines]
            
        return {'grid': grid, 'agents': agents, 'dims': (height, width)}
    except (ValueError, IndexError) as e:
        raise ValueError(f"Malformed instance file '{filepath}': {e}")


# ==============================================================================
# SECTION 2: EXAMPLE GENERATION
# ==============================================================================

def generate_new_agent_and_goal(env_instance, used_positions: set):
    """
    Randomly places a new agent and goal on the grid, avoiding used positions.
    """
    grid = env_instance.env.grid.copy()
    grid[grid == 1] = 0; grid[grid == 2] = 0
    
    free_cells = [tuple(map(int, idx)) for idx in np.argwhere(grid == 0) if tuple(map(int, idx)) not in used_positions]
    if len(free_cells) < 2:
        raise ValueError("Not enough free cells to place a new agent and goal.")
    
    agent_pos, goal_pos = np.random.choice(len(free_cells), 2, replace=False)
    agent_pos, goal_pos = free_cells[agent_pos], free_cells[goal_pos]
    
    env_instance.env.agent_pos = agent_pos
    env_instance.env.goal_pos = goal_pos
    grid[agent_pos] = 1; grid[goal_pos] = 2
    env_instance.env.grid = grid
    
    used_positions.add(agent_pos); used_positions.add(goal_pos)
    return env_instance._get_obs()

def make_example_from_txt(instance_path, model, device):
    """
    Creates a MAPF example dictionary from a custom .txt instance file.
    This version correctly initializes the environment and prepares it for planning.
    """
    try:
        instance = load_mapf_instance_from_txt(instance_path)
        print(f"Loaded instance from '{instance_path}': {len(instance['agents'])} agents, grid size {instance['dims']}.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading instance file: {e}")
        return None

    agent_envs, agent_plans, agent_trajectories = [], [], []
    height, width = instance['dims']
    
    # <<< FIX 1: Initialize the wrapper with generation_mode=None >>>
    # This creates a blank environment, preventing any random map generation.
    base_env = GridEnvWrapper(grid_size=max(height, width), generation_mode=None)
    
    # Now, manually set the grid from the loaded file
    base_env.env.grid = instance['grid']
    base_env.env.rows = height
    base_env.env.cols = width
    base_env.env.height = height
    base_env.env.width = width

    # print(base_env.env.grid)

    for agent_info in instance['agents']:
        env_inst = copy.deepcopy(base_env)
        start_pos = agent_info['start']
        goal_pos = agent_info['goal']

        env_inst.env.agent_pos = start_pos
        env_inst.env.goal_pos = goal_pos
        
        # <<< FIX 2: Manually mark the agent and goal on the grid before planning >>>
        # This ensures the A* planner has a correct view of the world state.
        if env_inst.env.grid[start_pos] == 0:
            env_inst.env.grid[start_pos] = 1 # Mark agent start
        if env_inst.env.grid[goal_pos] == 0:
            env_inst.env.grid[goal_pos] = 2 # Mark agent goal
        
        # Set the current observation after modifying the grid
        env_inst.current_observation = env_inst._get_obs()
        
        # Create a clean copy for the planner to use
        planning_env = copy.deepcopy(env_inst)
        plan = astar_cpp(planning_env, timeout=10, heuristic_weight=2)
        
        if plan is None:
            print(f"Warning: No initial plan found for agent start:{start_pos}->goal:{goal_pos}.")
            agent_plans.append([])
            agent_trajectories.append([])
        else:
            agent_plans.append(plan)
            # Simulate on a fresh environment to get the correct trajectory
            sim_env = copy.deepcopy(env_inst)
            agent_trajectories.append(simulate_plan(sim_env, plan))
            
        agent_envs.append(env_inst)

    return {"agent_envs": agent_envs, "agent_plans": agent_plans, "agent_trajectories": agent_trajectories}

def make_one_example(num_agents, model, device, map_filepath=None):
    """
    Generates one random MAPF instance.
    """
    base_env = GridEnvWrapper(grid_size=21, generation_mode='maze', maze_density=0.2)
    if map_filepath and map_filepath.endswith('.map'):
        grid_map, h, w = load_grid_from_map_file(map_filepath)
        base_env.env.grid = grid_map
        base_env.env.rows = h
        base_env.env.cols = w
        base_env.env.height = h
        base_env.env.width = w

    used_positions, agent_envs = set(), []
    for _ in range(num_agents):
        new_env = copy.deepcopy(base_env)
        generate_new_agent_and_goal(new_env, used_positions)
        agent_envs.append(new_env)

    agent_plans = [astar_cpp(copy.deepcopy(env), timeout=60, heuristic_weight=2) for env in agent_envs]
    agent_trajectories = [simulate_plan(env, p) if p else [] for env, p in zip(agent_envs, agent_plans)]

    return {"agent_envs": agent_envs, "agent_plans": agent_plans, "agent_trajectories": agent_trajectories}


# ==============================================================================
# SECTION 3: METRICS & ANALYSIS
# ==============================================================================

def calculate_trimmed_soc(trajectories, goal_positions):
    """
    Calculate sum of costs, trimming waits after reaching goal.

    For each agent, count only steps until the goal is first reached.
    """
    total = 0
    for i, traj in enumerate(trajectories):
        if not traj:
            continue
        goal = tuple(map(int, goal_positions[i]))
        # Find first time reaching goal
        for t, pos in enumerate(traj):
            if tuple(map(int, pos)) == goal:
                total += t + 1  # +1 because t is 0-indexed (position at t=0 is start)
                break
        else:
            # Never reached goal, count entire trajectory
            total += len(traj)
    return total


def count_agents_at_goal(trajectories, goal_positions):
    """Count agents whose final position is at their goal."""
    count = 0
    for i, traj in enumerate(trajectories):
        if traj and tuple(map(int, traj[-1])) == tuple(map(int, goal_positions[i])):
            count += 1
    return count


def compute_metrics(final_trajs, goal_positions, run_counters, start_time):
    """Computes and returns a dictionary of performance metrics."""
    num_agents = len(goal_positions)
    rem = analyze_collisions(final_trajs, goal_positions)
    final_collision_count = len(rem)
    collided_agents = set(agent_id for c in rem for agent_id in c['agents'])

    # Success rate calculation:
    # - If NO final collisions AND all agents at goal: success = 100%
    # - If there are final collisions: success = agents NOT involved in collisions / total agents
    if final_collision_count == 0:
        # No collisions - all agents at goal are successful
        success_rate = 1.0  # 100%
    else:
        # Collisions detected - count only agents NOT involved in collisions as successful
        success_rate = (num_agents - len(collided_agents)) / num_agents if num_agents > 0 else 0

    return {
        'total_collisions': run_counters.get('collisions_total', 0),
        'final_collisions': final_collision_count,
        'success_rate': success_rate,
        'total_time_sec': time.perf_counter() - start_time,
        'makespan': max((len(t) for t in final_trajs if t), default=0),
        'sum_of_costs': sum(len(t) for t in final_trajs if t)
    }

def run_and_analyze_mapf(initial_agent_plans, initial_agent_trajectories, agent_envs, model, run_counters, device, log_filepath=None, **kwargs):
    """
    Wrapper to call fix_collisions, finalize the log with the correct final
    plans, save it, and compute results.
    """
    start_time = time.perf_counter()

    # 1. Get the definitive final plans/trajectories and the historical log data
    # Note the new 'log_data' variable being captured from the return value.
    final_plans, final_trajs, unique_collisions, timed_out, log_data = fix_collisions(
        initial_agent_plans, initial_agent_trajectories, agent_envs, model, run_counters, device,
        **kwargs
    )
    
    # 2. Finalize and save the log file using the ground truth
    if log_filepath and log_data:
        # If the collision fixer returned a minimal log, synthesize the fields the visualizer expects.
        if "environment" not in log_data:
            grid = agent_envs[0].env.grid
            obstacles = [{"cell": [int(r), int(c)]} for r, c in zip(*((grid == -1).nonzero()))]
            log_data["environment"] = {
                "gridSize": [int(grid.shape[0]), int(grid.shape[1])],
                "obstacles": obstacles
            }
            log_data["agents"] = [
                {
                    "id": f"Robot-{i+1}",
                    "initialState": {"time": 0, "cell": [int(s[0]), int(s[1])]},
                    "goalState": {"cell": [int(g[0]), int(g[1])]}
                }
                for i, (s, g) in enumerate(
                    (env.env.agent_pos, env.env.goal_pos) for env in agent_envs
                )
            ]
            # Seed agentPaths with BOTH original and final trajectories
            log_data["agentPaths"] = [
                {
                    "agent": f"Robot-{i+1}",
                    "subplanId": f"Robot-{i+1}-Plan-Original",
                    "planCost": len(plan) if plan else 0,
                    "steps": [{"time": t, "cell": [int(pos[0]), int(pos[1])]} for t, pos in enumerate(traj)]
                }
                for i, (plan, traj) in enumerate(zip(initial_agent_plans, initial_agent_trajectories))
            ]
            # Add the final collision-free trajectories to agentPaths so they can be matched
            log_data["agentPaths"].extend([
                {
                    "agent": f"Robot-{i+1}",
                    "subplanId": f"Robot-{i+1}-Plan-Final",
                    "planCost": len(traj) if traj else 0,
                    "steps": [{"time": t, "cell": [int(pos[0]), int(pos[1])]} for t, pos in enumerate(traj)]
                }
                for i, traj in enumerate(final_trajs)
            ])
            log_data.setdefault("agentSubplans", [])
            log_data.setdefault("replanningStrategies", [])
            log_data.setdefault("conflictAlerts", [])
            log_data.setdefault("collisionEvents", [])
            # Normalize info sharing metrics naming if only minimal data is available
            if "informationSharingMetrics" not in log_data and "info_sharing" in log_data:
                log_data["informationSharingMetrics"] = log_data["info_sharing"]

        # Create a lookup table of all logged plans and their corresponding trajectories
        plan_id_to_traj = {}
        all_logged_plans = log_data.get("agentPaths", []) + log_data.get("agentSubplans", [])
        for p in all_logged_plans:
            plan_id = p.get("subplanId") or p.get("id")
            # Convert trajectory from a list of dicts to a list of tuples for easy comparison
            traj = [tuple(s['cell']) for s in p.get('steps', [])]
            plan_id_to_traj[plan_id] = traj

        # For each agent, find which logged plan matches its definitive final trajectory
        final_plans_map = {}
        for i, final_traj_list in enumerate(final_trajs):
            agent_id_str = f"Robot-{i+1}"
            # This is the ground-truth final trajectory for this agent
            final_traj_tuples = [tuple(map(int, pos)) for pos in final_traj_list]
            
            matched_plan_id = None
            # Search our "plan library" to find the ID of the plan that produced this exact trajectory
            for plan_id, logged_traj_tuples in plan_id_to_traj.items():
                # Check if this plan belongs to the current agent and if the trajectories match
                if f"Robot-{i+1}" in plan_id and logged_traj_tuples == final_traj_tuples:
                    matched_plan_id = plan_id
                    break 
            
            if matched_plan_id:
                final_plans_map[agent_id_str] = matched_plan_id
            else:
                # If no exact match (e.g., a deferred agent's trajectory is just its start pos),
                # fall back to the original plan ID as a placeholder. This handles deferred agents.
                original_plan = next((p for p in log_data.get("agentPaths", []) if p["agent"] == agent_id_str), None)
                if original_plan:
                    final_plans_map[agent_id_str] = original_plan["subplanId"]
                else:
                    # If for some reason even the original plan is missing, create a placeholder
                    final_plans_map[agent_id_str] = f"Robot-{i+1}-Plan-Deferred-Or-Missing"


        # 3. Populate the jointPlan with the now-correct subplan IDs
        log_data["jointPlan"] = {
            "id": "FinalJointPlan",
            "subplans": [final_plans_map.get(f"Robot-{i+1}") for i in range(len(agent_envs)) if f"Robot-{i+1}" in final_plans_map],
            "globalMakespan": max((len(t) for t in final_trajs if t), default=0)
        }

        # 4. Add comprehensive metrics section
        goal_positions = [env.env.goal_pos for env in agent_envs]
        metrics_raw = log_data.get('metrics_raw', {})
        strategy_iu_raw = log_data.get('strategy_iu_raw', {})

        log_data["metrics"] = {
            "makespan": max((len(t) for t in final_trajs if t), default=0),
            "sumOfCosts": sum(len(t) for t in final_trajs if t),
            "sumOfCostsTrimmed": calculate_trimmed_soc(final_trajs, goal_positions),
            "totalTime": log_data.get('time', 0),
            "agentsAtGoal": count_agents_at_goal(final_trajs, goal_positions),
            "initialConflicts": metrics_raw.get('initialConflicts', 0),
            "passes": {
                "total": log_data.get('passes', 0),
                "phase1": metrics_raw.get('phase1Passes', 0),
                "phase2": metrics_raw.get('phase2Passes', 0),
                "postCleanup": metrics_raw.get('postCleanupPasses', 0),
            },
            "strategiesTried": metrics_raw.get('strategies', {}),
            "deferredAgentsCount": metrics_raw.get('deferredAgentsCount', 0),
        }

        # 5. Add strategy IU section (only successful attempts)
        log_data["strategyIU"] = {
            "yieldIU": strategy_iu_raw.get('yieldIU', 0),
            "jointAstarIU": strategy_iu_raw.get('jointAstarIU', 0),
            "jointAstarCellConflicts": strategy_iu_raw.get('jointAstarCellConflicts', 0),
            "staticIU": strategy_iu_raw.get('staticIU', 0),
            "staticBlockedCellsIU": strategy_iu_raw.get('staticBlockedCellsIU', 0),
            "staticCollisionCellsIU": strategy_iu_raw.get('staticCollisionCellsIU', 0),
            "resubmissionIU": strategy_iu_raw.get('resubmissionIU', 0),
        }

        # Clean up raw metrics from log_data (they're now in the formatted sections)
        if 'metrics_raw' in log_data:
            del log_data['metrics_raw']
        if 'strategy_iu_raw' in log_data:
            del log_data['strategy_iu_raw']

        try:
            with open(log_filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            if kwargs.get('verbose', False): print(f"✔ Detailed execution log saved to '{log_filepath}'")
        except Exception as e:
            print(f"Error saving log file: {e}")

    # 4. Compute metrics and return
    goal_positions = [env.env.goal_pos for env in agent_envs]
    results = compute_metrics(final_trajs, goal_positions, run_counters, start_time)
    print(f"Analysis Complete. Success: {results['success_rate']:.2%}, Time: {results['total_time_sec']:.2f}s")
    
    # print(final_plans)
    for idx, item in enumerate(final_plans):
        print(f"Agent {idx+1} Final Plan: {item}")
    # print(final_trajs)
    return results, final_plans, final_trajs


# ==============================================================================
# SECTION 4: EXECUTION FLOW
# ==============================================================================

def run_single_instance(args, model, device):
    """Run a single MAPF instance from a specified .txt file."""
    print(f"→ Running single instance from: {args.map_file}")

    # Determine log file path: use --log_file if provided, otherwise use default
    if args.log_file:
        log_filepath = args.log_file
        # Extract directory and create if it doesn't exist
        log_dir = os.path.dirname(os.path.abspath(log_filepath))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            print(f"✓ Log directory created (if needed): {log_dir}")
    else:
        LOGS_DIR = "logs/info_test/"
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_filename = f"info_test_{os.path.basename(args.map_file).replace('.txt', '')}.json"
        log_filepath = os.path.join(LOGS_DIR, log_filename)

    example = make_example_from_txt(args.map_file, model, device)
    if not example: return

    print(f"Instance has {len(example['agent_envs'])} agents.")
    run_counters = {'collisions_total': 0}

    run_and_analyze_mapf(
        example["agent_plans"], example["agent_trajectories"], example["agent_envs"],
        model, run_counters, device, log_filepath=log_filepath,
        replan_strategy=args.strategy, info_setting=args.info, search_type=args.search_type,
        algo=args.algo, time_limit=args.timeout, heuristic_weight=args.heuristic_weight,
        max_expansions=args.max_expansions, verbose=args.verbose
    )

def run_batch_experiments(args, model, device):
    """Run a batch of experiments, generating data if needed."""
    DATA_DIR, RESULTS_DIR, LOGS_DIR = "simulation_data", "results", "logs"
    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(LOGS_DIR, exist_ok=True)

    MIN_AGENTS = args.num_agents if args.num_agents is not None else 3
    MAX_AGENTS = args.num_agents if args.num_agents is not None else 3
    NUM_DATAPOINTS = 1
    suffix = f"_{args.search_type}_{args.algo}_{args.strategy}"

    for num_agents in range(MIN_AGENTS, MAX_AGENTS + 1):
        data_file = os.path.join(DATA_DIR, f"random_{num_agents}.pkl")
        if not os.path.exists(data_file):
            print(f"Generating {NUM_DATAPOINTS} examples for {num_agents} agents...")
            examples = [make_one_example(num_agents, model, device, args.map_file) for _ in range(NUM_DATAPOINTS)]
            with open(data_file, "wb") as f: pickle.dump(examples, f)
        else:
            with open(data_file, "rb") as f: examples = pickle.load(f)

        print(f"\n→ Running batch for {num_agents} agents ({len(examples)} examples)")
        all_results = []
        for idx, ex in enumerate(examples, start=1):
            run_counters = {'collisions_total': 0}
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # log_filepath = os.path.join(LOGS_DIR, f"info_test/log_batch_{num_agents}agents_ex{idx}{suffix}_{timestamp}.json")
            log_filepath = os.path.join(LOGS_DIR, f"info_test/log_batch_{num_agents}agents_ex{idx}{suffix}.json")
            
            result, _, _ = run_and_analyze_mapf(
                ex["agent_plans"], ex["agent_trajectories"], ex["agent_envs"],
                model, run_counters, device, log_filepath=log_filepath,
                replan_strategy=args.strategy, info_setting=args.info, search_type=args.search_type,
                algo=args.algo, time_limit=args.timeout, heuristic_weight=args.heuristic_weight,
                max_expansions=args.max_expansions, verbose=False
            )
            all_results.append({"result": result})
            print(f"  [{num_agents} agents, ex {idx}/{len(examples)}] Total Collisions Logged: {run_counters['collisions_total']}")

        out_file = os.path.join(RESULTS_DIR, f"results{suffix}_{num_agents}_{args.info}.pkl")
        with open(out_file, "wb") as f: pickle.dump(all_results, f)
        print(f"✔ Saved batch results to '{out_file}'\n")

# ==============================================================================
# SECTION 5: MAIN
# ==============================================================================

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run MAPF analyses.")
    parser.add_argument("--strategy", choices=["best", "random", "farthest"], default="best", help="Agent selection policy.")
    parser.add_argument("--info", choices=["all", "no", "only_dyn"], default="all", help="Information sharing setting.")
    parser.add_argument("--search_type", choices=["greedy-bfs", "bfs", "astar", "astar-cpp"], default="astar", help="Planner for replanning.")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="ppo", help="Model type for RL-guided search.")
    parser.add_argument("--timeout", type=float, default=18000.0, help="Overall time limit (sec) for solving an instance.")
    parser.add_argument("--heuristic_weight", type=float, default=1.5, help="Heuristic weight for A*.")
    parser.add_argument("--max_expansions", type=int, default=10000, help="Max expansions for search.")
    parser.add_argument("--num_agents", type=int, default=None, help="Number of agents to simulate in batch mode.")
    parser.add_argument("--map_file", type=str, default=None, help="Path to a .txt or .map file to run a single instance.")
    parser.add_argument("--log_file", type=str, default=None, help="Path to save the output log JSON file.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for collision resolution.")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.algo == "dqn":
        model = ResNetDQN(num_actions=5).to(device)
        model.load_state_dict(torch.load("train_data/11_maze_dyn_new/final_model.pth", map_location=device)["policy_net_state_dict"], strict=False)
    else: # ppo
        model = PPOActorCritic(num_actions=5).to(device)
        model.load_state_dict(torch.load("train_data/11_maze_dyn_ppo/final_model.pth", map_location=device)["policy_state_dict"], strict=False)
    model.eval()

    if args.map_file and args.map_file.endswith('.txt'):
        run_single_instance(args, model, device)
    else:
        run_batch_experiments(args, model, device)

if __name__ == "__main__":
    main()
