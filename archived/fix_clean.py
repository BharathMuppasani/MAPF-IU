import numpy as np
import time
import copy
import random
import json

from collections import defaultdict, deque

from utils.env_utils import analyze_collisions, simulate_plan
from utils.search_utils import plan_with_search
from utils.yield_utils import find_deep_yield_cells, select_best_yield_cell
from utils.graph_utils import compute_cell_degrees, compute_flow_map, build_blocking_graph
from utils.corridor_utils import detect_corridors
from utils.reservation_utils import build_reservation_table, estimate_blocker_eta
from utils.central_coordinator import CentralCoordinator


class InfoSharingTracker:
    """A helper class to track and quantify information sharing events."""
    def __init__(self):
        self.initial_submission_iu = 0
        self.alert_iu = 0
        self.revised_submission_iu = 0
        self.alert_details_iu = {
            'static': 0,
            'dynamic': 0,
            'yield': 0
        }

    def record_initial_submission(self, initial_trajectories):
        """Records the IU from the initial submission of all agent paths."""
        self.initial_submission_iu = sum(len(t) for t in initial_trajectories if t)

    def record_static_alert(self, forbidden_cells):
        """Records IU from a static alert. IU = number of forbidden cells."""
        iu = len(forbidden_cells)
        self.alert_iu += iu
        self.alert_details_iu['static'] += iu

    def record_dynamic_alert(self, dynamic_path):
        """Records IU from a dynamic alert. IU = length of the provided sub-path."""
        iu = len(dynamic_path)
        self.alert_iu += iu
        self.alert_details_iu['dynamic'] += iu

    def record_yield_alert(self, yield_info_size=1):
        """Records IU from a yield command."""
        self.alert_iu += yield_info_size
        self.alert_details_iu['yield'] += yield_info_size

    def record_revised_submission(self, new_trajectory):
        """Records the IU from an agent submitting a revised path."""
        self.revised_submission_iu += len(new_trajectory)

    @property
    def total_iu(self):
        """Calculates the total information load."""
        return self.initial_submission_iu + self.alert_iu + self.revised_submission_iu

    def report(self):
        """Prints a formatted summary to the terminal."""
        print("\n--- Information Sharing Metrics ---")
        print(f"  - Initial Path Submission IU: {self.initial_submission_iu}")
        print(f"  - Revised Path Submission IU: {self.revised_submission_iu}")
        print(f"  - Conflict Alert IU:          {self.alert_iu}")
        print(f"    - Static Alerts:      {self.alert_details_iu['static']} IU")
        print(f"    - Dynamic Alerts:     {self.alert_details_iu['dynamic']} IU")
        print(f"    - Yield Alerts:       {self.alert_details_iu['yield']} IU")
        print("-----------------------------------")
        print(f"  Total Information Load (IU):  {self.total_iu}")
        print("-----------------------------------")

    def to_dict(self):
        """Converts the metrics to a dictionary for JSON serialization."""
        return {
            'initialSubmissionIU': self.initial_submission_iu,
            'revisedSubmissionIU': self.revised_submission_iu,
            'conflictAlertIU': self.alert_iu,
            'alertDetailsIU': self.alert_details_iu,
            'totalInformationLoadIU': self.total_iu
        }


def cell_key(cell):
    """Convert cell to hashable key."""
    if isinstance(cell, (list, tuple)):
        if len(cell) == 2 and not isinstance(cell[0], (list, tuple)):
            return tuple(map(int, cell))
        else:
            return tuple(tuple(map(int, c)) for c in cell)
    return cell


def select_replan_agents(collision, replan_strategy, agent_goals, current_trajectories, collision_time):
    """
    Select which agents should replan for a given collision.

    Returns: list of agent_ids (1-indexed) ordered by priority
    """
    agents_in_collision = list(collision['agents'])

    if replan_strategy == "random":
        return [random.choice(agents_in_collision)]

    elif replan_strategy == "farthest":
        # Agent farthest from goal replans
        def get_distance_to_goal(agent_id):
            idx = agent_id - 1
            if current_trajectories[idx] and collision_time < len(current_trajectories[idx]):
                pos = current_trajectories[idx][collision_time]
            else:
                pos = current_trajectories[idx][-1] if current_trajectories[idx] else agent_goals[idx]
            goal = agent_goals[idx]
            return np.linalg.norm(np.array(pos) - np.array(goal))

        agents_sorted = sorted(agents_in_collision, key=get_distance_to_goal, reverse=True)
        return [agents_sorted[0]]

    else:  # "best" - try all agents, prioritizing those at goal
        # Separate finished vs active agents
        finished = []
        active = []

        for agent_id in agents_in_collision:
            idx = agent_id - 1
            traj = current_trajectories[idx]
            goal = agent_goals[idx]

            is_finished = False
            if traj:
                if collision_time >= len(traj) - 1:
                    is_finished = True
                elif tuple(map(int, traj[collision_time])) == tuple(map(int, goal)):
                    # At goal, check if stays there
                    if all(tuple(map(int, p)) == tuple(map(int, goal)) for p in traj[collision_time:]):
                        is_finished = True

            if is_finished:
                finished.append(agent_id)
            else:
                active.append(agent_id)

        # Priority: finished agents first, then active agents
        return finished + active


def try_yield_strategy(
    agent_id, collision, current_trajectories, agent_goals, agent_starts,
    pristine_static_grid, cell_degrees, corridor_map, coordinator,
    agent_envs, model, device, search_type, algo, timeout, max_expansions,
    heuristic_weight, info_tracker, yield_contracts, overall_fix_attempt_count,
    verbose=False
):
    """
    Implement intelligent yield strategy with:
    - Heatmap-based yield cell selection
    - ETA-based wait duration
    - Yield contract creation

    Returns: (success, new_plan, new_trajectory, yield_cell, wait_duration) or (False, None, None, None, None)
    """
    idx = agent_id - 1
    goal = tuple(map(int, agent_goals[idx]))
    collision_time = collision['time']
    other_agents = [aid for aid in collision['agents'] if aid != agent_id]

    if verbose:
        print(f"  [YIELD Strategy] Agent {agent_id} yielding for agents {other_agents}")

    # Step 1: Build blocking graph and flow map for traffic awareness
    blocking_edges, blocked_cells_per_agent = build_blocking_graph(
        current_trajectories, agent_goals, pristine_static_grid,
        agent_envs, model, device, search_type, algo,
        horizon=10, max_expansions=500, timeout=2.0
    )
    flow_map = compute_flow_map(blocked_cells_per_agent, ignore_agent=idx)

    # Step 2: Update coordinator heatmap for global traffic awareness
    coordinator.update_heatmap(current_trajectories)

    # Step 3: Find yield cell candidates (2-6 steps from goal for better clearing)
    yield_candidates = find_deep_yield_cells(
        goal, pristine_static_grid,
        distance_range=(2, 6), max_candidates=30
    )

    if not yield_candidates:
        # Fallback to immediate neighbors
        rows, cols = pristine_static_grid.shape
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = goal[0] + dr, goal[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and pristine_static_grid[nr, nc] == 0:
                neighbors.append(((nr, nc), 1))
        yield_candidates = neighbors

    if not yield_candidates:
        if verbose:
            print(f"    No yield cells available for Agent {agent_id}")
        return False, None, None, None, None

    # Step 4: Select best yield cell using flow map and heatmap
    best_yield_cell = select_best_yield_cell(
        yield_candidates, pristine_static_grid,
        cell_degrees, flow_map, goal, corridor_map
    )

    if not best_yield_cell:
        if verbose:
            print(f"    No suitable yield cell found for Agent {agent_id}")
        return False, None, None, None, None

    # Step 5: Estimate wait duration based on blocker ETA
    collision_cell = collision['cell']
    if collision['type'] == 'vertex':
        blocking_region = {tuple(map(int, collision_cell))}
    else:  # edge collision
        blocking_region = {tuple(map(int, c)) for c in collision_cell}

    # Add nearby cells to blocking region for more conservative wait
    expanded_blocking_region = set(blocking_region)
    for cell in blocking_region:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (cell[0] + dr, cell[1] + dc)
            if 0 <= neighbor[0] < pristine_static_grid.shape[0] and \
               0 <= neighbor[1] < pristine_static_grid.shape[1]:
                expanded_blocking_region.add(neighbor)

    max_eta = 0
    for blocker_id in other_agents:
        eta = estimate_blocker_eta(
            blocker_id, expanded_blocking_region,
            current_trajectories, collision_time
        )
        max_eta = max(max_eta, eta)

    # ETA-based wait with bounds (min 3, max 25) and buffer
    wait_duration = max(3, min(max_eta + 3, 25))

    if verbose:
        print(f"    Yield cell: {best_yield_cell}, ETA: {max_eta}, Wait: {wait_duration} steps")

    # Step 6: Plan detour: goal -> yield_cell -> wait -> goal
    # 6a: Plan to yield cell
    env_to_yield = copy.deepcopy(agent_envs[idx])
    env_to_yield.env.grid = pristine_static_grid.copy()
    env_to_yield.env.agent_pos = goal
    env_to_yield.env.goal_pos = best_yield_cell

    plan_to_yield = plan_with_search(
        env_to_yield, model, device, search_type, algo,
        timeout=timeout, heuristic_weight=heuristic_weight, max_expansions=max_expansions
    )

    if not plan_to_yield:
        if verbose:
            print(f"    Failed to plan path to yield cell")
        return False, None, None, None, None

    # 6b: Wait actions
    wait_actions = [4] * wait_duration  # Action 4 = STAY

    # 6c: Plan return to goal
    env_return = copy.deepcopy(agent_envs[idx])
    env_return.env.grid = pristine_static_grid.copy()
    env_return.env.agent_pos = best_yield_cell
    env_return.env.goal_pos = goal

    plan_return = plan_with_search(
        env_return, model, device, search_type, algo,
        timeout=timeout, heuristic_weight=heuristic_weight, max_expansions=max_expansions
    )

    if not plan_return:
        if verbose:
            print(f"    Failed to plan return path from yield cell")
        return False, None, None, None, None

    # Step 7: Construct full plan
    current_plan = list(current_trajectories[idx])

    # Find insertion point (at collision time or before)
    insert_time = min(collision_time, len(current_plan) - 1) if current_plan else 0

    # Build new plan
    prefix = list(current_trajectories[idx][:insert_time]) if current_plan else []
    detour_actions = plan_to_yield + wait_actions + plan_return

    # Reconstruct from actions
    new_full_plan_actions = []

    # Convert prefix trajectory to actions (rough approximation, we'll simulate full plan anyway)
    # Just use the detour for the segment
    # We need to get actions from start to insert_time, then detour
    # Simpler: use original plan actions up to insert_time, then detour

    # Get original plan actions
    original_plan_actions = list(current_trajectories[idx]) if hasattr(current_trajectories[idx], '__iter__') else []

    # Actually, we need the PLAN (actions), not trajectory
    # The function signature suggests we might not have easy access to original plan actions
    # Let's reconstruct: simulate from start with detour

    # Build full action sequence
    # We need to map trajectory prefix back to actions - not trivial
    # Simpler approach: just use detour from current position

    # Get current position at insertion time
    if current_plan and insert_time < len(current_plan):
        current_pos = tuple(map(int, current_plan[insert_time]))
    else:
        current_pos = tuple(map(int, agent_starts[idx]))

    # Plan from current position to yield, wait, return
    env_full_replan = copy.deepcopy(agent_envs[idx])
    env_full_replan.env.grid = pristine_static_grid.copy()
    env_full_replan.env.agent_pos = current_pos
    env_full_replan.env.goal_pos = goal

    # Manually construct: current_pos -> goal -> yield -> wait -> goal
    # Since agent is at goal, we can use the detour directly

    # For simplicity: if agent is at goal, use detour actions directly
    # Otherwise, need to plan from current pos to goal first

    if current_pos == goal:
        # Agent is at goal, use detour directly
        new_plan_actions = plan_to_yield + wait_actions + plan_return
    else:
        # Agent not at goal yet, plan to goal first
        env_to_goal = copy.deepcopy(agent_envs[idx])
        env_to_goal.env.grid = pristine_static_grid.copy()
        env_to_goal.env.agent_pos = current_pos
        env_to_goal.env.goal_pos = goal

        plan_to_goal = plan_with_search(
            env_to_goal, model, device, search_type, algo,
            timeout=timeout, heuristic_weight=heuristic_weight, max_expansions=max_expansions
        )

        if not plan_to_goal:
            return False, None, None, None, None

        new_plan_actions = plan_to_goal + plan_to_yield + wait_actions + plan_return

    # Simulate full plan from start
    sim_env = copy.deepcopy(agent_envs[idx])
    sim_env.env.agent_pos = agent_starts[idx]
    new_trajectory = simulate_plan(sim_env, new_plan_actions)

    if not new_trajectory:
        if verbose:
            print(f"    Simulation failed for yield plan")
        return False, None, None, None, None

    # Step 8: Record information sharing
    if info_tracker:
        info_tracker.record_yield_alert(yield_info_size=1)  # Command to yield

    if verbose:
        print(f"    ✓ Yield plan created: {len(new_plan_actions)} actions, {len(new_trajectory)} steps")

    return True, new_plan_actions, new_trajectory, best_yield_cell, wait_duration


def fix_collisions(
    initial_agent_plans,
    initial_agent_trajectories,
    agent_envs,
    model,
    run_counters,
    device,
    log_filepath=None,
    replan_strategy="best",
    info_setting="all",
    search_type="astar",
    algo="ppo",
    timeout=10.0,
    heuristic_weight=1.0,
    max_expansions=200,
    time_limit=60,
    max_passes=50,
    verbose=True
):
    """
    Clean collision resolution with three strategies:
    1. YIELD (if agent at goal) - with ETA-based wait and contracts
    2. STATIC replanning - mark collision cells as obstacles
    3. DYNAMIC replanning - share blocker trajectory as dynamic obstacle
    """
    overall_start_time = time.perf_counter()

    # Initialize counters
    if run_counters is None:
        run_counters = {}
    for key in ['replan_attempts_static', 'replan_success_static',
                'replan_attempts_dynamic', 'replan_success_dynamic',
                'replan_attempts_yield', 'replan_success_yield',
                'collisions_total']:
        run_counters.setdefault(key, 0)

    # Copy initial plans and trajectories
    current_plans = [list(p) for p in initial_agent_plans]
    current_trajectories = [list(t) for t in initial_agent_trajectories]

    num_agents = len(current_plans)
    agent_goals = [tuple(map(int, env.env.goal_pos)) for env in agent_envs]
    agent_starts = [tuple(map(int, env.env.agent_pos)) for env in agent_envs]

    # Get pristine static grid
    pristine_static_grid = agent_envs[0].env.grid.copy()
    for r in range(pristine_static_grid.shape[0]):
        for c in range(pristine_static_grid.shape[1]):
            if pristine_static_grid[r, c] != -1:
                pristine_static_grid[r, c] = 0

    # Precompute structural information
    cell_degrees = compute_cell_degrees(pristine_static_grid)
    corridor_map = detect_corridors(cell_degrees, pristine_static_grid)

    if verbose:
        print(f"\n=== Collision Resolution Started ===")
        print(f"Grid: {pristine_static_grid.shape}, Agents: {num_agents}")
        print(f"Corridors detected: {len(corridor_map)}")
        print(f"Strategy: {replan_strategy}, Info: {info_setting}")

    # Initialize coordinator and tracking
    coordinator = CentralCoordinator(pristine_static_grid.shape)
    info_tracker = InfoSharingTracker()
    info_tracker.record_initial_submission(initial_agent_trajectories)

    # Yield contracts: agent_id -> contract
    yield_contracts = {}

    # History tracking to avoid repeated failures
    static_block_hist = defaultdict(set)  # agent_idx -> set of cells already blocked
    dyn_obs_hist = defaultdict(list)  # agent_idx -> list of frozensets of dyn obs cells
    collision_attempts = defaultdict(int)  # collision signature -> attempt count

    # Constants
    INIT_REWIND = 3
    MAX_REWIND_STATIC = 7
    MAX_REWIND_DYN = 7

    # Main resolution loop
    for pass_num in range(1, max_passes + 1):
        elapsed = time.perf_counter() - overall_start_time
        if elapsed > time_limit:
            if verbose:
                print(f"\n✗ Time limit exceeded ({elapsed:.2f}s)")
            break

        if verbose:
            print(f"\n--- Pass {pass_num}/{max_passes} (Elapsed: {elapsed:.2f}s) ---")

        # Update yield contracts
        expired_contracts = []
        for agent_id, contract in list(yield_contracts.items()):
            # Check if blocking agents have passed
            all_passed = True
            for blocker_id in contract['for_agents']:
                blocker_idx = blocker_id - 1
                if current_trajectories[blocker_idx]:
                    blocker_pos = tuple(map(int, current_trajectories[blocker_idx][-1]))
                    blocker_goal = agent_goals[blocker_idx]
                    if blocker_pos != blocker_goal:
                        all_passed = False
                        break

            if all_passed:
                expired_contracts.append(agent_id)
                if verbose:
                    print(f"  ✓ Contract expired for Agent {agent_id}")

        for agent_id in expired_contracts:
            del yield_contracts[agent_id]

        # Detect collisions
        collisions = analyze_collisions(current_trajectories, agent_goals, pristine_static_grid)
        run_counters['collisions_total'] += len(collisions)

        if not collisions:
            if verbose:
                print(f"✓ No collisions found! Resolution complete.")
            break

        if verbose:
            print(f"Detected {len(collisions)} collisions")

        # Sort collisions by time
        collisions_sorted = sorted(collisions, key=lambda c: (c['time'], min(c['agents'])))

        any_fix_this_pass = False

        for coll in collisions_sorted:
            coll_key = (coll['time'], coll['type'], cell_key(coll['cell']), frozenset(coll['agents']))
            collision_attempts[coll_key] += 1
            attempt_num = collision_attempts[coll_key]

            if verbose:
                print(f"\n  Collision: T={coll['time']}, Type={coll['type']}, "
                      f"Cell={coll['cell']}, Agents={list(coll['agents'])}, Attempt={attempt_num}")

            # Re-check if collision still exists
            current_colls = analyze_collisions(current_trajectories, agent_goals, pristine_static_grid)
            still_exists = any(
                (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key
                for c in current_colls
            )

            if not still_exists:
                if verbose:
                    print(f"  ✓ Collision already resolved")
                continue

            # Select agents to replan
            agents_to_try = select_replan_agents(
                coll, replan_strategy, agent_goals, current_trajectories, coll['time']
            )

            if verbose:
                print(f"  Agents selected for replanning: {agents_to_try}")

            # Try strategies in order: YIELD -> STATIC -> DYNAMIC
            fixed = False

            # STRATEGY 1: YIELD (if agent is at goal)
            if info_setting == "all":
                for agent_id in agents_to_try:
                    idx = agent_id - 1
                    traj = current_trajectories[idx]
                    goal = agent_goals[idx]

                    # Check if agent is at goal
                    is_at_goal = False
                    if traj:
                        coll_time = coll['time']
                        if coll_time < len(traj):
                            pos_at_coll = tuple(map(int, traj[coll_time]))
                            if pos_at_coll == goal:
                                # Check if stays at goal
                                if all(tuple(map(int, p)) == goal for p in traj[coll_time:]):
                                    is_at_goal = True

                    if is_at_goal:
                        if verbose:
                            print(f"  → Trying YIELD for Agent {agent_id} (at goal)")

                        run_counters['replan_attempts_yield'] += 1

                        success, new_plan, new_traj, yield_cell, wait_dur = try_yield_strategy(
                            agent_id, coll, current_trajectories, agent_goals, agent_starts,
                            pristine_static_grid, cell_degrees, corridor_map, coordinator,
                            agent_envs, model, device, search_type, algo, timeout, max_expansions,
                            heuristic_weight, info_tracker, yield_contracts, pass_num, verbose
                        )

                        if success:
                            # Verify the yield actually resolves the collision
                            temp_trajs = list(current_trajectories)
                            temp_trajs[idx] = new_traj
                            new_colls = analyze_collisions(temp_trajs, agent_goals, pristine_static_grid)

                            coll_resolved = not any(
                                (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key
                                for c in new_colls
                            )

                            if coll_resolved:
                                # Commit the yield
                                current_plans[idx] = new_plan
                                current_trajectories[idx] = new_traj

                                if info_tracker:
                                    info_tracker.record_revised_submission(new_traj)

                                # Create yield contract
                                other_agents = [aid for aid in coll['agents'] if aid != agent_id]
                                yield_contracts[agent_id] = {
                                    'for_agents': other_agents,
                                    'yield_cell': yield_cell,
                                    'wait_duration': wait_dur,
                                    'created_at_pass': pass_num
                                }

                                run_counters['replan_success_yield'] += 1
                                any_fix_this_pass = True
                                fixed = True

                                if verbose:
                                    print(f"  ✓ YIELD successful for Agent {agent_id}")
                                    print(f"    Contract: yield at {yield_cell}, wait {wait_dur} steps for {other_agents}")

                                break

            if fixed:
                continue

            # STRATEGY 2: STATIC replanning
            if info_setting in ["all", "no"]:
                if attempt_num <= 3:  # Try static for first 3 attempts
                    for agent_id in agents_to_try:
                        idx = agent_id - 1

                        if verbose:
                            print(f"  → Trying STATIC for Agent {agent_id}")

                        run_counters['replan_attempts_static'] += 1

                        # Determine cells to block
                        if coll['type'] == 'vertex':
                            cells_to_block = [tuple(map(int, coll['cell']))]
                        elif coll['type'] == 'edge':
                            cells_to_block = [tuple(map(int, c)) for c in coll['cell']]
                        else:  # obstacle collision
                            cells_to_block = [tuple(map(int, coll['cell']))]

                        # Filter out already blocked cells
                        new_cells_to_block = [c for c in cells_to_block if c not in static_block_hist[idx]]

                        if not new_cells_to_block:
                            if verbose:
                                print(f"    All collision cells already in block history")
                            continue

                        # Create environment with blocked cells
                        env_copy = copy.deepcopy(agent_envs[idx])
                        rewind = min(MAX_REWIND_STATIC, INIT_REWIND + attempt_num - 1)
                        replan_time = max(0, coll['time'] - rewind)

                        if current_trajectories[idx] and replan_time < len(current_trajectories[idx]):
                            replan_pos = tuple(map(int, current_trajectories[idx][replan_time]))
                        else:
                            replan_pos = agent_starts[idx]

                        planning_grid = pristine_static_grid.copy()

                        # Mark collision cells as obstacles
                        for cell in new_cells_to_block:
                            if 0 <= cell[0] < planning_grid.shape[0] and 0 <= cell[1] < planning_grid.shape[1]:
                                planning_grid[cell[0], cell[1]] = -1

                        env_copy.env.grid = planning_grid
                        env_copy.env.agent_pos = replan_pos
                        env_copy.env.goal_pos = agent_goals[idx]

                        # Plan
                        new_plan_segment = plan_with_search(
                            env_copy, model, device, search_type, algo,
                            timeout, heuristic_weight, max_expansions
                        )

                        if new_plan_segment:
                            # Build full plan
                            prefix_actions = current_plans[idx][:replan_time] if current_plans[idx] else []
                            new_full_plan = prefix_actions + new_plan_segment

                            # Simulate
                            sim_env = copy.deepcopy(agent_envs[idx])
                            sim_env.env.agent_pos = agent_starts[idx]
                            new_traj = simulate_plan(sim_env, new_full_plan)

                            if new_traj:
                                # Check if collision resolved
                                temp_trajs = list(current_trajectories)
                                temp_trajs[idx] = new_traj
                                new_colls = analyze_collisions(temp_trajs, agent_goals, pristine_static_grid)

                                coll_resolved = not any(
                                    (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key
                                    for c in new_colls
                                )

                                if coll_resolved:
                                    # Commit
                                    current_plans[idx] = new_full_plan
                                    current_trajectories[idx] = new_traj

                                    if info_tracker:
                                        info_tracker.record_static_alert(new_cells_to_block)
                                        info_tracker.record_revised_submission(new_traj)

                                    # Update history
                                    static_block_hist[idx].update(new_cells_to_block)

                                    run_counters['replan_success_static'] += 1
                                    any_fix_this_pass = True
                                    fixed = True

                                    if verbose:
                                        print(f"  ✓ STATIC successful for Agent {agent_id}")
                                        print(f"    Blocked cells: {new_cells_to_block}, new collisions: {len(new_colls)}")

                                    break

            if fixed:
                continue

            # STRATEGY 3: DYNAMIC replanning
            if info_setting in ["all", "only_dyn"]:
                if attempt_num <= 3:  # Try dynamic for first 3 attempts
                    for agent_id in agents_to_try:
                        idx = agent_id - 1
                        other_agents = [aid for aid in coll['agents'] if aid != agent_id]

                        if not other_agents:
                            continue

                        # Pick a random blocker
                        blocker_id = random.choice(other_agents)
                        blocker_idx = blocker_id - 1

                        if verbose:
                            print(f"  → Trying DYNAMIC for Agent {agent_id} (blocker: {blocker_id})")

                        run_counters['replan_attempts_dynamic'] += 1

                        # Setup environment
                        env_copy = copy.deepcopy(agent_envs[idx])
                        rewind = min(MAX_REWIND_DYN, INIT_REWIND + attempt_num - 1)
                        replan_time = max(0, coll['time'] - rewind)

                        if current_trajectories[idx] and replan_time < len(current_trajectories[idx]):
                            replan_pos = tuple(map(int, current_trajectories[idx][replan_time]))
                        else:
                            replan_pos = agent_starts[idx]

                        # Extract blocker trajectory segment
                        blocker_traj = current_trajectories[blocker_idx]
                        if not blocker_traj:
                            continue

                        obs_start_t = replan_time
                        obs_end_t = min(len(blocker_traj) - 1, replan_time + 2 * rewind)

                        if obs_start_t >= len(blocker_traj) or obs_start_t > obs_end_t:
                            continue

                        obs_path = [tuple(map(int, p)) for p in blocker_traj[obs_start_t:obs_end_t + 1]]

                        if not obs_path:
                            continue

                        # Setup environment with dynamic obstacle
                        planning_grid = pristine_static_grid.copy()
                        env_copy.env.grid = planning_grid
                        env_copy.env.agent_pos = replan_pos
                        env_copy.env.goal_pos = agent_goals[idx]
                        env_copy.env.dynamic_info = [{
                            'pos': obs_path[0],
                            'goal': obs_path[-1],
                            'path': deque(obs_path),
                            'stop_after_goal': True
                        }]
                        env_copy.env.num_dynamic_obstacles = 1

                        # Plan
                        new_plan_segment = plan_with_search(
                            env_copy, model, device, search_type, algo,
                            timeout, heuristic_weight, max_expansions
                        )

                        if new_plan_segment:
                            # Build full plan
                            prefix_actions = current_plans[idx][:replan_time] if current_plans[idx] else []
                            new_full_plan = prefix_actions + new_plan_segment

                            # Simulate
                            sim_env = copy.deepcopy(agent_envs[idx])
                            sim_env.env.agent_pos = agent_starts[idx]
                            new_traj = simulate_plan(sim_env, new_full_plan)

                            if new_traj:
                                # Check if collision resolved
                                temp_trajs = list(current_trajectories)
                                temp_trajs[idx] = new_traj
                                new_colls = analyze_collisions(temp_trajs, agent_goals, pristine_static_grid)

                                coll_resolved = not any(
                                    (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key
                                    for c in new_colls
                                )

                                if coll_resolved:
                                    # Commit
                                    current_plans[idx] = new_full_plan
                                    current_trajectories[idx] = new_traj

                                    if info_tracker:
                                        info_tracker.record_dynamic_alert(obs_path)
                                        info_tracker.record_revised_submission(new_traj)

                                    run_counters['replan_success_dynamic'] += 1
                                    any_fix_this_pass = True
                                    fixed = True

                                    if verbose:
                                        print(f"  ✓ DYNAMIC successful for Agent {agent_id}")
                                        print(f"    Blocker path: {len(obs_path)} steps, new collisions: {len(new_colls)}")

                                    break

            if not fixed and verbose:
                print(f"  ✗ Could not resolve collision")

        if not any_fix_this_pass:
            if verbose:
                print(f"\n✗ No progress in this pass, stopping")
            break

    # Final collision check
    final_collisions = analyze_collisions(current_trajectories, agent_goals, pristine_static_grid)

    if verbose:
        print(f"\n=== Resolution Complete ===")
        print(f"Time: {time.perf_counter() - overall_start_time:.2f}s")
        print(f"Passes: {pass_num}")
        print(f"Final collisions: {len(final_collisions)}")
        info_tracker.report()

    return current_plans, current_trajectories, info_tracker.to_dict()
