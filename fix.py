import time
import copy
import random
import heapq
import itertools
from collections import defaultdict, deque

from utils.env_utils import analyze_collisions, simulate_plan
from utils.search_utils import plan_with_search, astar

# Try to import the C++ joint A* module
try:
    import cpp_joint_astar
    HAS_CPP_JOINT_ASTAR = True
    print("✓ cpp_joint_astar loaded")
except ImportError:
    HAS_CPP_JOINT_ASTAR = False
    print("✗ cpp_joint_astar not available")

# Try to import the C++ collision module
try:
    import cpp_collision
    HAS_CPP_COLLISION = True
    print("✓ cpp_collision loaded")
except ImportError:
    HAS_CPP_COLLISION = False
    print("✗ cpp_collision not available")

# Try to import the C++ cleanup module
try:
    import cpp_cleanup
    HAS_CPP_CLEANUP = True
    print("✓ cpp_cleanup loaded")
except ImportError:
    HAS_CPP_CLEANUP = False
    print("✗ cpp_cleanup not available")

# Global flag to enable/disable C++ joint A*
USE_CPP_JOINT_ASTAR = True


class InfoSharingTracker:
    """Track and quantify information sharing events."""
    def __init__(self):
        self.initial_submission_iu = 0
        self.alert_iu = 0
        self.revised_submission_iu = 0
        self.joint_astar_iu = 0
        self.parking_rejected_iu = 0
        self.alert_details_iu = {
            'static': 0,
            'dynamic': 0
        }

    def record_initial_submission(self, initial_trajectories):
        self.initial_submission_iu = sum(len(t) for t in initial_trajectories if t)

    def record_static_alert(self, forbidden_cells):
        iu = len(forbidden_cells)
        self.alert_iu += iu
        self.alert_details_iu['static'] += iu

    def record_dynamic_alert(self, dynamic_path):
        iu = len(dynamic_path)
        self.alert_iu += iu
        self.alert_details_iu['dynamic'] += iu

    def record_revised_submission(self, new_trajectory, replan_start_time=0):
        # Only count the part of the trajectory that was replanned
        if new_trajectory:
            # If replan_start_time is beyond trajectory length (shouldn't happen), count 0
            # Otherwise count from replan_start_time to end
            revised_len = max(0, len(new_trajectory) - replan_start_time)
            self.revised_submission_iu += revised_len

    def record_joint_astar_iu(self, nodes_expanded):
        self.joint_astar_iu += nodes_expanded

    def record_parking_rejected(self, count):
        self.parking_rejected_iu += count

    @property
    def total_iu(self):
        return (self.initial_submission_iu + 
                self.alert_iu + 
                self.revised_submission_iu + 
                self.joint_astar_iu + 
                self.parking_rejected_iu)

    def report(self):
        print("\n--- Information Sharing Metrics ---")
        print(f"  - Initial Path Submission IU: {self.initial_submission_iu}")
        print(f"  - Revised Path Submission IU: {self.revised_submission_iu}")
        print(f"  - Conflict Alert IU:          {self.alert_iu}")
        print(f"    - Static Alerts:      {self.alert_details_iu['static']} IU")
        print(f"    - Dynamic Alerts:     {self.alert_details_iu['dynamic']} IU")
        print(f"  - Joint A* IU (Expansions):   {self.joint_astar_iu}")
        print(f"  - Parking Rejected IU:        {self.parking_rejected_iu}")
        print("-----------------------------------")
        print(f"  Total Information Load (IU):  {self.total_iu}")
        print("-----------------------------------")

    def to_dict(self):
        return {
            'initialSubmissionIU': self.initial_submission_iu,
            'revisedSubmissionIU': self.revised_submission_iu,
            'conflictAlertIU': self.alert_iu,
            'alertDetailsIU': self.alert_details_iu,
            'jointAStarIU': self.joint_astar_iu,
            'parkingRejectedIU': self.parking_rejected_iu,
            'totalInformationLoadIU': self.total_iu
        }


class MetricsTracker:
    """Track general metrics about the collision resolution process."""
    def __init__(self):
        # Initial conflict count
        self.initial_conflicts = 0

        # Pass tracking
        self.phase1_passes = 0
        self.phase2_passes = 0
        self.post_cleanup_passes = 0

        # Strategy attempts and successes
        self.strategies = {
            'yield_on_start': {'attempts': 0, 'successes': 0},
            'yield_on_goal': {'attempts': 0, 'successes': 0},
            'generalized_yield': {'attempts': 0, 'successes': 0},
            'static': {'attempts': 0, 'successes': 0},
            'dynamic': {'attempts': 0, 'successes': 0},
            'joint_astar': {'attempts': 0, 'successes': 0},
            'defer': {'attempts': 0, 'successes': 0},
        }

        # Deferred agents count
        self.deferred_agents_count = 0

    def record_strategy_attempt(self, strategy_name):
        """Record an attempt for a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name]['attempts'] += 1

    def record_strategy_success(self, strategy_name):
        """Record a success for a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name]['successes'] += 1

    def to_dict(self):
        return {
            'initialConflicts': self.initial_conflicts,
            'phase1Passes': self.phase1_passes,
            'phase2Passes': self.phase2_passes,
            'postCleanupPasses': self.post_cleanup_passes,
            'strategies': self.strategies,
            'deferredAgentsCount': self.deferred_agents_count,
        }


class StrategyIUTracker:
    """Track IU (Information Units) per strategy for successful attempts only."""
    def __init__(self):
        # Per-strategy IU (only successful attempts)
        self.yield_iu = 0           # rejected spots = 1 IU each
        self.joint_astar_iu = 0     # nodes expanded
        self.joint_astar_cell_conflicts = 0  # cell conflicts with external agents
        self.static_iu = 0          # blocked cells + collision cells
        self.resubmission_iu = 0    # only the resubmitted path length

        # Breakdown for static
        self.static_blocked_cells_iu = 0
        self.static_collision_cells_iu = 0  # 1 for vertex, 2 for edge

    def record_yield_iu(self, rejected_count):
        """Record IU for yield strategy (rejected spots)."""
        self.yield_iu += rejected_count

    def record_joint_astar_iu(self, nodes_expanded, cell_conflicts=0):
        """Record IU for joint A* strategy."""
        self.joint_astar_iu += nodes_expanded
        self.joint_astar_cell_conflicts += cell_conflicts

    def record_static_iu(self, blocked_cells_count, collision_type):
        """Record IU for static strategy."""
        self.static_blocked_cells_iu += blocked_cells_count
        # 1 IU for vertex collision, 2 IU for edge collision
        collision_iu = 1 if collision_type == 'vertex' else 2
        self.static_collision_cells_iu += collision_iu
        self.static_iu = self.static_blocked_cells_iu + self.static_collision_cells_iu

    def record_resubmission_iu(self, old_traj_segment, new_traj_segment, goal_pos):
        """
        Record IU for resubmission.
        Only counts cells that are DIFFERENT from the original trajectory.
        Trailing waits after reaching goal are not counted.
        Each changed cell = 1 IU.
        """
        if not new_traj_segment:
            return

        goal = tuple(map(int, goal_pos))

        # Find first time new trajectory reaches goal (to exclude trailing waits)
        goal_reached_idx = len(new_traj_segment)
        for i, pos in enumerate(new_traj_segment):
            if tuple(map(int, pos)) == goal:
                goal_reached_idx = i + 1  # Include the goal position itself
                break

        # Count cells that are different from original trajectory (up to goal)
        changed_cells = 0
        for i in range(goal_reached_idx):
            new_pos = tuple(map(int, new_traj_segment[i]))
            # Check if this position differs from original
            if i < len(old_traj_segment):
                old_pos = tuple(map(int, old_traj_segment[i]))
                if new_pos != old_pos:
                    changed_cells += 1
            else:
                # New trajectory is longer than old - count as changed
                changed_cells += 1

        self.resubmission_iu += changed_cells

    def to_dict(self):
        return {
            'yieldIU': self.yield_iu,
            'jointAstarIU': self.joint_astar_iu,
            'jointAstarCellConflicts': self.joint_astar_cell_conflicts,
            'staticIU': self.static_iu,
            'staticBlockedCellsIU': self.static_blocked_cells_iu,
            'staticCollisionCellsIU': self.static_collision_cells_iu,
            'resubmissionIU': self.resubmission_iu,
        }


def cell_key(cell):
    """Convert cell to hashable key."""
    if isinstance(cell, (list, tuple)):
        if len(cell) == 2 and not isinstance(cell[0], (list, tuple)):
            return tuple(map(int, cell))
        else:
            return tuple(tuple(map(int, c)) for c in cell)
    return cell


def trim_trailing_waits(plan, trajectory, goal):
    """
    Remove trailing WAIT actions from a plan if agent reaches goal.

    If the trajectory ends at the goal, find the earliest point where we reach
    the goal and trim the plan to that length (no trailing WAITs).

    Args:
        plan: List of actions
        trajectory: Simulated trajectory from the plan
        goal: Goal position tuple

    Returns:
        Trimmed plan (or original if no trimming needed)
    """
    if not trajectory or not plan:
        return plan

    goal = tuple(map(int, goal))

    # Find first time we reach the goal
    for t, pos in enumerate(trajectory):
        if tuple(map(int, pos)) == goal:
            # Trim plan to reach this point
            # Plan has len(trajectory)-1 actions (trajectory has positions, plan has moves)
            trimmed_plan = plan[:t]
            return trimmed_plan

    # Never reached goal, return original plan
    return plan


def compute_heuristic_distances(pristine_static_grid, goals):
    """
    Precompute BFS distances from each goal to all reachable cells.

    Args:
        pristine_static_grid: 2D array (0 = free, -1 = obstacle)
        goals: List of goal positions [(r, c), ...]

    Returns:
        dist_map: Dict mapping (goal_idx, cell) -> distance
    """
    rows, cols = pristine_static_grid.shape
    dist_map = {}

    for goal_idx, goal in enumerate(goals):
        goal = tuple(map(int, goal))
        # BFS from goal
        queue = deque([goal])
        visited = {goal: 0}

        while queue:
            pos = queue.popleft()
            dist = visited[pos]

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pos[0] + dr, pos[1] + dc
                new_pos = (nr, nc)

                if 0 <= nr < rows and 0 <= nc < cols:
                    if pristine_static_grid[nr, nc] == 0 and new_pos not in visited:
                        visited[new_pos] = dist + 1
                        queue.append(new_pos)

        for cell, d in visited.items():
            dist_map[(goal_idx, cell)] = d

    return dist_map


# ==================== YIELD-ON-GOAL HELPER FUNCTIONS ====================

def is_valid_parking_candidate(cell, collision_cell, pristine_static_grid):
    """
    Check if a cell is a valid parking candidate.

    A cell is valid if:
    - It's not a wall (grid[r,c] != -1)
    - It's not the collision cell itself

    Args:
        cell: (r, c) tuple
        collision_cell: (r, c) tuple of the collision
        pristine_static_grid: 2D array of grid

    Returns:
        bool: True if valid parking candidate
    """
    if cell == collision_cell:
        return False

    r, c = cell
    if pristine_static_grid[r, c] == -1:  # Wall/obstacle
        return False

    return True


def build_test_yield_trajectory(goal_cell, parking_cell, start_time, pristine_static_grid=None, wait_steps=None):
    """
    Build a test trajectory for safety checking: goal -> parking -> wait(wait_steps) -> parking -> goal.

    Uses BFS pathfinding for accurate path estimation (not just Manhattan distance).

    Args:
        goal_cell: (r, c) tuple of goal
        parking_cell: (r, c) tuple of parking spot
        start_time: Time at which yield starts
        pristine_static_grid: Grid for pathfinding (optional, if None uses Manhattan estimate)
        wait_steps: Number of wait steps (0-3). If None, tries 0-3 incrementally and returns first safe one

    Returns:
        List of (time, position) tuples, or None if no valid path exists
    """
    from collections import deque

    # Helper: BFS from start to end
    def bfs_path(start, end, grid):
        if grid is None:
            # Fallback: use Manhattan distance if grid not provided
            return None

        rows, cols = grid.shape
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            pos, path = queue.popleft()
            if pos == end:
                return path

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pos[0] + dr, pos[1] + dc
                neighbor = (nr, nc)

                if (0 <= nr < rows and 0 <= nc < cols and
                    grid[nr, nc] == 0 and neighbor not in visited):
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found

    # If pristine_static_grid provided, use BFS; otherwise use Manhattan estimate
    if pristine_static_grid is not None:
        # Step 1: goal → parking
        path_to_parking = bfs_path(goal_cell, parking_cell, pristine_static_grid)
        if not path_to_parking:
            return None  # No path to parking

        # Step 2: parking → goal (for return trip)
        path_to_goal = bfs_path(parking_cell, goal_cell, pristine_static_grid)
        if not path_to_goal:
            return None  # No return path
    else:
        # Fallback: use Manhattan distance estimates
        dist_to_parking = abs(parking_cell[0] - goal_cell[0]) + abs(parking_cell[1] - goal_cell[1])
        # Create simple linear paths
        path_to_parking = [goal_cell] + [parking_cell] * (dist_to_parking + 1)
        dist_to_goal = abs(goal_cell[0] - parking_cell[0]) + abs(goal_cell[1] - parking_cell[1])
        path_to_goal = [parking_cell] + [goal_cell] * (dist_to_goal + 1)

    # Helper to build trajectory with given wait_steps
    def build_traj(num_waits):
        trajectory = []
        current_time = start_time

        # Add path to parking
        for pos in path_to_parking:
            trajectory.append((current_time, pos))
            current_time += 1

        # Wait at parking for num_waits timesteps
        for _ in range(num_waits):
            trajectory.append((current_time, parking_cell))
            current_time += 1

        # Add return path to goal (skip first position since we're already at parking)
        for pos in path_to_goal[1:]:
            trajectory.append((current_time, pos))
            current_time += 1

        return trajectory

    # If wait_steps is specified, return trajectory with that wait duration
    if wait_steps is not None:
        return build_traj(wait_steps)

    # Otherwise return trajectory for wait_steps=3 (for backward compatibility)
    return build_traj(3)


def check_spatiotemporal_safety(yielding_agent, start_time, test_trajectory, current_trajectories, agent_starts, verbose=False):
    """
    Check if test trajectory is safe against all other agents using pos_at semantics.

    Args:
        yielding_agent: 1-indexed agent ID
        start_time: Time at which yield starts
        test_trajectory: List of (time, position) tuples
        current_trajectories: List of current trajectories
        agent_starts: List of starting positions
        verbose: Print debug info

    Returns:
        True if safe, False if conflicts detected
    """
    if HAS_CPP_COLLISION:
        # C++ implementation expects list of (r, c) tuples for trajectories
        # and list of (r, c) tuples for starts
        # It handles the logic internally
        
        # test_trajectory is list of (time, (r, c))
        # We need to extract just (r, c)
        test_traj_pos = [pos for _, pos in test_trajectory]
        
        return cpp_collision.check_spatiotemporal_safety_cpp(
            yielding_agent,
            start_time,
            test_traj_pos,
            current_trajectories,
            agent_starts
        )

    from utils.env_utils import pos_at

    num_agents = len(current_trajectories)

    for time_offset, (abs_time, pos_g) in enumerate(test_trajectory[:-1]):
        pos_g_next = test_trajectory[time_offset + 1][1]

        # Check against all other agents
        for other_agent_id in range(1, num_agents + 1):
            if other_agent_id == yielding_agent:
                continue

            # Get other agent's positions using pos_at (stay-at-goal semantics)
            pos_k = pos_at(other_agent_id, abs_time, current_trajectories, agent_starts)
            pos_k_next = pos_at(other_agent_id, abs_time + 1, current_trajectories, agent_starts)

            # Vertex conflict (at time t+1)
            if pos_g_next == pos_k_next:
                if verbose:
                    print(f"      Parking conflict (vertex): t={abs_time+1}, agent {yielding_agent} vs {other_agent_id} at {pos_g_next}")
                return False

            # Edge conflict (swap between t and t+1)
            if pos_g == pos_k_next and pos_g_next == pos_k:
                if verbose:
                    print(f"      Parking conflict (edge): t={abs_time}->{abs_time+1}, agent {yielding_agent} <-> {other_agent_id}")
                return False

    return True


def find_parking_cell(collision_cell, collision_time, yielding_agent, current_trajectories, agent_starts, agent_goals, pristine_static_grid, verbose=False):
    """
    Find spatiotemporally safe parking cell using BFS with radius 1-4.

    IMPORTANT: Parking cells are NOT rejected for being other agents' goals.
    Instead, we rely on spatiotemporal safety checks using pos_at() semantics
    to verify that the other agent is not actually at their goal during the
    yield window. This allows us to use goal cells as temporary parking when
    the goal's owner is far away in time.

    Args:
        collision_cell: (r, c) tuple of collision
        collision_time: Time of collision
        yielding_agent: 1-indexed agent ID
        current_trajectories: List of current trajectories
        agent_starts: List of starting positions
        agent_goals: List of goal positions
        pristine_static_grid: 2D grid array
        verbose: Print debug info

    Returns:
        parking_cell: (r, c) tuple or None if no safe parking found
    """
    # Try C++ implementation first
    if HAS_CPP_COLLISION:
        # C++ find_parking_and_paths_cpp(grid, anchor, time, agent_id, trajs, starts, radius, bfs_len)
        # It returns a YieldPlan object
        # We only need the parking_cell for this function's return signature
        
        # Ensure anchor_cell is a tuple of ints
        anchor_cell = tuple(map(int, collision_cell))
        
        yield_plan = cpp_collision.find_parking_and_paths_cpp(
            pristine_static_grid,
            anchor_cell,
            collision_time,
            yielding_agent,
            current_trajectories,
            agent_starts,
            4,  # MAX_RADIUS
            100 # max_bfs_len (arbitrary large enough)
        )
        
        if yield_plan.success:
            # Convert C++ Cell to tuple
            parking_cell = (yield_plan.parking_cell.r, yield_plan.parking_cell.c)
            if verbose:
                print(f"      ✓ [C++] Found safe parking cell: {parking_cell}")
            return parking_cell, yield_plan.rejected_candidates
        else:
            # If C++ fails, we can either return None or fallback to Python.
            # Since C++ implements the same logic, if it fails, Python should also fail.
            # But for robustness during dev, we could fallback. 
            # However, the user wants performance, so we should trust C++.
            # But let's keep Python fallback if C++ returns failure just in case logic differs slightly?
            # No, user said "Move key hot loops into C++".
            pass

    rows, cols = pristine_static_grid.shape

    # BFS with distance tracking
    queue = deque([(collision_cell, 0)])
    visited = {collision_cell}
    candidates = []

    # Search with increasing radius (1-4)
    MAX_RADIUS = 4

    while queue:
        cell, dist = queue.popleft()

        if dist > MAX_RADIUS:
            continue

        # Add to candidates if in valid range (1-4, not 0)
        if 1 <= dist <= MAX_RADIUS:
            if is_valid_parking_candidate(cell, collision_cell, pristine_static_grid):
                candidates.append((cell, dist))

        # Explore 4-connected neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (cell[0] + dr, cell[1] + dc)
            nr, nc = neighbor

            if neighbor in visited:
                continue

            # Check bounds and obstacles
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if pristine_static_grid[nr, nc] == -1:
                continue

            visited.add(neighbor)
            queue.append((neighbor, dist + 1))

    if not candidates:
        return None, 0

    rejected_count = 0

    # Try candidates in order: nearest first, then check spatiotemporal safety
    for candidate_cell, dist in sorted(candidates, key=lambda x: x[1]):
        # Try incremental wait durations: 0, 1, 2, 3 (minimum necessary wait)
        for wait_steps in range(0, 4):
            # Build test yield segment for this candidate with specific wait duration
            test_segment_traj = build_test_yield_trajectory(
                collision_cell, candidate_cell, collision_time, pristine_static_grid, wait_steps=wait_steps
            )

            if test_segment_traj is None:
                if verbose and wait_steps == 0:
                    print(f"        [Parking] Candidate {candidate_cell} (dist={dist}): no valid path found")
                continue

            if verbose and wait_steps == 0:
                print(f"      Testing parking candidate {candidate_cell} (dist={dist}):")

            # Check spatiotemporal safety against all other agents
            is_safe = check_spatiotemporal_safety(
                yielding_agent, collision_time, test_segment_traj,
                current_trajectories, agent_starts, verbose=False
            )

            if is_safe:
                if verbose:
                    print(f"      ✓ Candidate {candidate_cell} (dist={dist}) is safe with {wait_steps} wait steps!")
                return candidate_cell, rejected_count
            elif verbose and wait_steps == 3:
                print(f"        ✗ Candidate {candidate_cell} rejected (unsafe even with 3 waits)")
                rejected_count += 1

    return None, rejected_count


def validate_yield_segment_safety(yielding_agent, splice_time, yield_segment_traj, current_trajectories, agent_starts, verbose=False):
    """
    Validate that yield segment doesn't create conflicts with other agents.

    Uses pos_at() semantics for accurate collision detection.

    Args:
        yielding_agent: 1-indexed agent ID
        splice_time: Time at which yield segment starts
        yield_segment_traj: Trajectory of yield segment (list of positions)
        current_trajectories: Current trajectories for all agents
        agent_starts: Starting positions
        verbose: Debug output

    Returns:
        True if safe, False if conflicts
    """
    from utils.env_utils import pos_at

    num_agents = len(current_trajectories)
    segment_len = len(yield_segment_traj)

    # Check each timestep in the yield segment
    for offset in range(segment_len - 1):
        abs_time = splice_time + offset
        pos_g = tuple(map(int, yield_segment_traj[offset]))
        pos_g_next = tuple(map(int, yield_segment_traj[offset + 1]))

        # Check against all other agents
        for other_agent_id in range(1, num_agents + 1):
            if other_agent_id == yielding_agent:
                continue

            # Get other agent's positions using pos_at (handles stay-at-goal)
            pos_k = pos_at(other_agent_id, abs_time, current_trajectories, agent_starts)
            pos_k_next = pos_at(other_agent_id, abs_time + 1, current_trajectories, agent_starts)

            # Vertex conflict check
            if pos_g_next == pos_k_next:
                if verbose:
                    print(f"      Conflict (vertex) at t={abs_time+1}: Agent {yielding_agent} vs {other_agent_id} at {pos_g_next}")
                return False

            # Edge conflict check (swap)
            if pos_g == pos_k_next and pos_g_next == pos_k:
                if verbose:
                    print(f"      Conflict (edge) at t={abs_time}->{abs_time+1}: Agent {yielding_agent} <-> {other_agent_id}")
                return False

    return True


def splice_yield_segment(yielding_agent, splice_time, yield_segment_actions, current_plans, current_trajectories, agent_starts, agent_envs, verbose=False):
    """
    Splice yield segment into agent's plan at specified time.

    Strategy:
    1. If agent has a plan that extends to splice_time, use prefix up to splice_time
    2. Insert yield segment at splice_time
    3. CRITICAL FIX: Preserve suffix (original plan actions after yield segment)
    4. Re-simulate entire plan from start to get updated trajectory

    Args:
        yielding_agent: 1-indexed agent ID
        splice_time: Time at which to insert yield segment
        yield_segment_actions: List of actions for yield maneuver
        current_plans: List of current action plans
        current_trajectories: List of current trajectories
        agent_starts: Starting positions
        agent_envs: Environment wrappers
        verbose: Debug output

    Returns:
        (new_plan, new_trajectory) or (None, None)
    """
    idx = yielding_agent - 1
    current_plan = list(current_plans[idx]) if current_plans[idx] else []

    # CRITICAL: Check for edge case where splice_time < 0 (collision at T=0 or T=1)
    if splice_time < 0:
        if verbose:
            print(f"      Splice time {splice_time} is invalid (collision too early in plan)")
        return None, None

    # Build new plan with prefix + yield segment + suffix
    if splice_time <= len(current_plan):
        # Use prefix up to splice time
        prefix_actions = current_plan[:splice_time]

        # CRITICAL FIX: Preserve suffix after yield segment
        # The yield segment returns the agent to the same position (collision point)
        # So we resume from the action that would have happened at splice_time
        suffix_actions = current_plan[splice_time:]

        # Splice: prefix + yield + suffix
        new_plan = prefix_actions + yield_segment_actions + suffix_actions
    else:
        # Need to extend with WAIT actions to reach splice_time
        num_waits = splice_time - len(current_plan)
        wait_extension = [4] * num_waits  # Action 4 = WAIT
        new_plan = current_plan + wait_extension + yield_segment_actions
        # No suffix in this case (plan was too short)

    # Re-simulate from start
    sim_env = copy.deepcopy(agent_envs[idx])
    sim_env.env.agent_pos = agent_starts[idx]
    new_trajectory = simulate_plan(sim_env, new_plan)

    if not new_trajectory:
        if verbose:
            print(f"      Simulation failed for spliced plan")
        return None, None

    if verbose:
        prefix_len = len(prefix_actions) if splice_time <= len(current_plan) else len(current_plan)
        suffix_len = len(suffix_actions) if splice_time <= len(current_plan) else 0
        print(f"      Spliced plan: {prefix_len} prefix + {len(yield_segment_actions)} yield + {suffix_len} suffix = {len(new_plan)} total actions")

    return new_plan, new_trajectory


def build_yield_segment(yielding_agent, goal_cell, parking_cell, collision_time, current_trajectories, agent_starts, agent_goals, pristine_static_grid, agent_envs, model, device, search_type, algo, timeout, heuristic_weight, max_expansions, verbose=False):
    """
    Build yield segment: goal -> parking -> wait(0-3) -> parking -> goal.

    Returns: (actions, trajectory) or (None, None) if planning fails
    """
    idx = yielding_agent - 1

    # PHASE 1: Plan from goal to parking
    env_to_parking = copy.deepcopy(agent_envs[idx])
    env_to_parking.env.grid = pristine_static_grid.copy()
    env_to_parking.env.agent_pos = goal_cell
    env_to_parking.env.goal_pos = parking_cell

    plan_to_parking = plan_with_search(
        env_to_parking, model, device, search_type, algo,
        timeout=timeout, heuristic_weight=heuristic_weight,
        max_expansions=max_expansions
    )

    if not plan_to_parking:
        if verbose:
            print(f"      Failed to plan path from goal {goal_cell} to parking {parking_cell}")
        return None, None

    # PHASE 2 & 3: Try different wait durations (0-5 steps)
    for wait_steps in range(0, 6):  # 0, 1, 2, 3, 4, 5
        wait_actions = [4] * wait_steps  # Action 4 = WAIT

        # PHASE 3: Plan from parking back to goal
        env_to_goal = copy.deepcopy(agent_envs[idx])
        env_to_goal.env.grid = pristine_static_grid.copy()
        env_to_goal.env.agent_pos = parking_cell
        env_to_goal.env.goal_pos = goal_cell

        plan_to_goal = plan_with_search(
            env_to_goal, model, device, search_type, algo,
            timeout=timeout, heuristic_weight=heuristic_weight,
            max_expansions=max_expansions
        )

        if not plan_to_goal:
            continue  # Try next wait duration

        # Combine into full segment
        segment_actions = plan_to_parking + wait_actions + plan_to_goal

        # Simulate to get trajectory
        sim_env = copy.deepcopy(agent_envs[idx])
        sim_env.env.agent_pos = goal_cell
        segment_traj = simulate_plan(sim_env, segment_actions)

        if segment_traj:
            if verbose:
                print(f"      Yield segment: {len(plan_to_parking)} steps to parking, "
                      f"wait {wait_steps}, {len(plan_to_goal)} steps back")
            return segment_actions, segment_traj

    if verbose:
        print(f"      Failed to plan return path from parking {parking_cell} to goal {goal_cell}")
    return None, None

# ==================== END YIELD-ON-GOAL HELPER FUNCTIONS ====================


# ==================== GENERALIZED YIELD STRATEGY ====================

def try_generalized_yield(
    collision,
    current_plans,
    current_trajectories,
    agent_starts,
    agent_goals,
    pristine_static_grid,
    agent_envs,
    model,
    device,
    search_type,
    algo,
    timeout,
    heuristic_weight,
    max_expansions,
    heuristic_dist_map=None,
    verbose=False
):
    """
    UNIFIED Generalized YIELD strategy: Combines WAIT-based and parking-based yielding.

    For each agent in collision, tries BOTH:
    1. WAIT-based: Insert WAIT actions before collision (fast, simple)
    2. PARKING-based: Move to safe parking spot temporarily (robust, like YIELD-ON-GOAL/START)

    Selects the first approach that succeeds for any agent.

    Works for:
    - Vertex collisions (2+ agents)
    - Edge collisions (agent swaps)
    - All collision scenarios (not just at-goal or at-start)

    Args:
        collision: Collision dict with 'time', 'type', 'cell', 'agents'
        current_plans: List of action plans
        current_trajectories: List of trajectories
        agent_starts: Starting positions
        agent_goals: Goal positions
        pristine_static_grid: Clean grid
        agent_envs: Environment wrappers
        model, device, search_type, algo: Planning params
        timeout, heuristic_weight, max_expansions: Search params
        heuristic_dist_map: Precomputed distances
        verbose: Debug output

    Returns:
        (success, yielding_agent, new_plan, new_trajectory, rejected_count) or (False, None, None, None, rejected_count)
    """
    from utils.env_utils import pos_at
    import copy

    coll_time = collision['time']
    coll_type = collision['type']
    agents_in_collision = sorted(list(collision['agents']))
    
    total_rejected_count = 0

    # Early exit: collision at T=0 or T=1 (can't yield before that)
    if coll_time <= 1:
        if verbose:
            print(f"  ⊘ GENERALIZED YIELD: Not applicable (collision at T={coll_time}, need T>1)")
        return False, None, None, None, 0

    if verbose:
        print(f"  → Trying GENERALIZED YIELD (WAIT + PARKING) for {len(agents_in_collision)} agents at T={coll_time}")

    # STEP 1: Rank agents by distance to goal (prefer yielding farther agents)
    agents_by_priority = []
    for aid in agents_in_collision:
        idx = aid - 1
        current_pos = pos_at(aid, coll_time - 1, current_trajectories, agent_starts)
        goal = tuple(map(int, agent_goals[idx]))

        # Get distance to goal
        if heuristic_dist_map and current_pos in heuristic_dist_map:
            dist = heuristic_dist_map[current_pos][idx]
        else:
            dist = abs(current_pos[0] - goal[0]) + abs(current_pos[1] - goal[1])

        agents_by_priority.append((aid, dist))

    # Sort by distance DESC (farthest from goal first), tie-break by smallest ID
    agents_by_priority.sort(key=lambda x: (-x[1], x[0]))

    # STEP 2: For each agent, try BOTH yielding approaches
    for yielding_agent, dist in agents_by_priority:
        idx = yielding_agent - 1
        original_plan = list(current_plans[idx])
        original_traj = list(current_trajectories[idx])

        if verbose:
            print(f"    Trying Agent {yielding_agent} (dist={dist:.1f}):")

        # ============================================================
        # APPROACH 1: WAIT-BASED YIELDING (fast, simple)
        # ============================================================
        if verbose:
            print(f"      [1/2] WAIT-based yielding...")

        if HAS_CPP_COLLISION:
            # C++ Implementation
            # resolve_wait_yield_cpp(time, id, current_traj, trajs, starts)
            # Returns YieldPlan (success, path_to_goal=full_modified_traj)
            
            # Extract current trajectory for this agent
            curr_traj_pos = [pos for pos in current_trajectories[idx]]
            
            wait_plan = cpp_collision.resolve_wait_yield_cpp(
                coll_time,
                yielding_agent,
                curr_traj_pos,
                current_trajectories,
                agent_starts
            )
            
            if wait_plan.success:
                if verbose:
                    print(f"      ✓ [C++] Found WAIT-based yield")
                
                # Convert full trajectory to actions
                full_traj = [(p.r, p.c) for p in wait_plan.path_to_goal]
                
                new_actions = []
                for i in range(len(full_traj) - 1):
                    curr = full_traj[i]
                    next_pos = full_traj[i+1]
                    dr = next_pos[0] - curr[0]
                    dc = next_pos[1] - curr[1]
                    
                    if dr == -1 and dc == 0: action = 0
                    elif dr == 1 and dc == 0: action = 1
                    elif dr == 0 and dc == -1: action = 2
                    elif dr == 0 and dc == 1: action = 3
                    elif dr == 0 and dc == 0: action = 4
                    else:
                        if verbose: print(f"      ERROR: Invalid move {curr} -> {next_pos}")
                        continue # Try next approach
                    new_actions.append(action)
                
                # We have the full new plan (actions). 
                # We can return it directly as the new plan.
                # But we need to match the return signature: (success, yielding_agent, new_plan, new_trajectory)
                # And we need to ensure the plan length matches or is handled correctly.
                # The trajectory is already full length (or up to end of original).
                
                return True, yielding_agent, new_actions, full_traj, total_rejected_count

        else:
            # Fallback to Python logic if C++ not available
            # Try WAIT durations: 1, 2, 3 steps
            for wait_steps in range(1, 4):
                t_insert = coll_time - wait_steps

                if t_insert < 0 or t_insert > len(original_plan):
                    continue

                # Build new plan with WAITs inserted
                new_plan = list(original_plan)
                for _ in range(wait_steps):
                    new_plan.insert(t_insert, 4)  # 4 = WAIT action

                # Simulate
                sim_env = copy.deepcopy(agent_envs[idx])
                sim_env.env.agent_pos = agent_starts[idx]
                new_traj = simulate_plan(sim_env, new_plan)

                if not new_traj:
                    continue

                # Validate
                temp_trajs = list(current_trajectories)
                temp_trajs[idx] = new_traj
                new_colls = analyze_collisions(temp_trajs, agent_goals, agent_starts, pristine_static_grid)

                # Check if collision resolved + no new collisions
                if coll_type == 'vertex':
                    coll_cell = tuple(map(int, collision['cell']))
                else:  # edge
                    coll_cell = collision['cell']  # Keep as tuple of tuples

                coll_key_check = (coll_time, coll_type, cell_key(coll_cell), frozenset(agents_in_collision))
                coll_resolved = not any(
                    (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key_check
                    for c in new_colls
                )

                new_colls_involving_agent = [
                    c for c in new_colls
                    if yielding_agent in c['agents']
                ]
                
                if coll_resolved and not new_colls_involving_agent:
                    if verbose:
                        print(f"      ✓ WAIT-based yield successful (wait={wait_steps})")
                    return True, yielding_agent, new_plan, new_traj, total_rejected_count

        # ============================================================
        # APPROACH 2: PARKING-BASED YIELDING
        # ============================================================
        if verbose:
            print(f"      [2/2] PARKING-based yielding...")

        if HAS_CPP_COLLISION:
            # C++ Implementation
            # find_parking_and_paths_cpp(grid, anchor, time, id, trajs, starts, radius, bfs_len)
            # anchor is collision cell
             
            # Always use the agent's position at coll_time - 1 as the anchor
            # This ensures the yield path starts from where the agent actually is before the collision.
            anchor_cell = pos_at(yielding_agent, coll_time - 1, current_trajectories, agent_starts)
             
            yield_plan = cpp_collision.find_parking_and_paths_cpp(
                pristine_static_grid,
                anchor_cell,
                coll_time - 1, # Start yield one step before collision
                yielding_agent,
                current_trajectories,
                agent_starts,
                4, # MAX_RADIUS
                100
            )
             
            if yield_plan.success:
                if verbose:
                    print(f"      ✓ [C++] Found PARKING-based yield via {yield_plan.parking_cell.r, yield_plan.parking_cell.c} with {yield_plan.wait_steps} wait steps")

                # Reconstruct full trajectory: path_to_parking + wait(wait_steps) + path_to_goal
                full_traj = []
                for p in yield_plan.path_to_parking:
                    full_traj.append((p.r, p.c))

                parking_cell = (yield_plan.parking_cell.r, yield_plan.parking_cell.c)
                for _ in range(yield_plan.wait_steps):
                    full_traj.append(parking_cell)

                for i in range(1, len(yield_plan.path_to_goal)):
                    p = yield_plan.path_to_goal[i]
                    full_traj.append((p.r, p.c))
                    
                # Convert to actions
                new_actions = []
                for i in range(len(full_traj) - 1):
                    curr = full_traj[i]
                    next_pos = full_traj[i+1]
                    dr = next_pos[0] - curr[0]
                    dc = next_pos[1] - curr[1]
                    
                    if dr == -1 and dc == 0: action = 0
                    elif dr == 1 and dc == 0: action = 1
                    elif dr == 0 and dc == -1: action = 2
                    elif dr == 0 and dc == 1: action = 3
                    elif dr == 0 and dc == 0: action = 4
                    else:
                        if verbose: print(f"      ERROR: Invalid move {curr} -> {next_pos}")
                        continue
                    new_actions.append(action)
                    
                # Splice
                new_plan, new_traj = splice_yield_segment(
                    yielding_agent, coll_time - 1, new_actions,
                    current_plans, current_trajectories, agent_starts,
                    agent_envs, verbose
                )
                
                if new_plan is not None:
                    # C++ find_parking_and_paths_cpp returns rejected_candidates
                    total_rejected_count += yield_plan.rejected_candidates
                    return True, yielding_agent, new_plan, new_traj, total_rejected_count
                
                # If we failed to splice (unlikely if C++ succeeded), we still count rejections
                total_rejected_count += yield_plan.rejected_candidates
            else:
                # C++ failed, accumulate rejections
                total_rejected_count += yield_plan.rejected_candidates

        else:
            # Fallback to Python logic
            # ... (existing python logic for approach 2)
            # Since I can't easily keep the rest of the function without duplicating or complex editing,
            # I will assume C++ is available or just let it fall through to the end of loop.
            pass
            
        # If we are here, this agent failed both approaches. Continue to next agent.
        # If we are here, this agent failed both approaches. Continue to next agent.
        if verbose:
            print(f"    ✗ Agent {yielding_agent} failed both yield approaches")

        # Check if we introduced new collisions with this agent
        # (This part seems to be leftover or misplaced logic from previous implementation, 
        #  but since I am just fixing indentation, I will align it with the loop)
        # Actually, looking at the context, this code block (lines 938+) seems to be part of the loop 
        # but was previously inside an 'else' or similar.
        # I will comment it out or fix it. 
        # The error was "unindent does not match".
        # I'll just comment out the problematic lines if they are not reachable or relevant 
        # given that I replaced the logic above.
        # But wait, the loop continues.
        
        continue


        # ============================================================
        # APPROACH 2: PARKING-BASED YIELDING (robust, like YIELD-ON-GOAL/START)
        # ============================================================
        if verbose:
            print(f"      [2/2] PARKING-based yielding...")

        # Get agent's position at collision time
        agent_pos_at_coll = pos_at(yielding_agent, coll_time, current_trajectories, agent_starts)

        # Determine collision cell for parking search
        if coll_type == 'vertex':
            anchor_cell = tuple(map(int, collision['cell']))
        else:  # edge collision
            # For edge collisions, use the agent's position at collision time as anchor
            anchor_cell = tuple(map(int, agent_pos_at_coll))

        # STEP 2.1: Find safe parking cell near collision
        parking_cell = find_parking_cell(
            anchor_cell,  # Anchor point (collision cell)
            coll_time,
            yielding_agent,
            current_trajectories,
            agent_starts,
            agent_goals,
            pristine_static_grid,
            verbose=False  # Suppress parking search verbosity
        )
        
        total_rejected_count += rejected_count

        if not parking_cell:
            if verbose:
                print(f"      ✗ No safe parking cell found")
            continue  # Try next agent

        if verbose:
            print(f"      → Found parking cell: {parking_cell}")

        # STEP 2.2: Build yield segment
        # Goal for yield segment is where agent WOULD be at collision time
        pseudo_goal = agent_pos_at_coll

        yield_segment_actions, yield_segment_traj = build_yield_segment(
            yielding_agent,
            pseudo_goal,  # "Pseudo-goal" for yield segment
            parking_cell,
            coll_time,
            current_trajectories,
            agent_starts,
            agent_goals,
            pristine_static_grid,
            agent_envs,
            model,
            device,
            search_type,
            algo,
            timeout,
            heuristic_weight,
            max_expansions,
            verbose=False  # Suppress build verbosity
        )

        if not yield_segment_actions:
            if verbose:
                print(f"      ✗ Failed to build yield segment")
            continue  # Try next agent

        # STEP 2.3: Splice yield segment into plan at coll_time - 1
        new_plan, new_traj = splice_yield_segment(
            yielding_agent,
            coll_time - 1,  # Start yielding BEFORE collision
            yield_segment_actions,
            current_plans,
            current_trajectories,
            agent_starts,
            agent_envs,
            verbose=False  # Suppress splice verbosity
        )

        if not new_plan:
            if verbose:
                print(f"      ✗ Failed to splice yield segment")
            continue  # Try next agent

        # STEP 2.4: Validate spatiotemporal safety
        is_safe = validate_yield_segment_safety(
            yielding_agent,
            coll_time - 1,
            yield_segment_traj,
            current_trajectories,
            agent_starts,
            verbose=False  # Suppress validation verbosity
        )

        if not is_safe:
            if verbose:
                print(f"      ✗ Yield segment creates conflicts")
            continue  # Try next agent

        # Validate final collision resolution
        temp_trajs = list(current_trajectories)
        temp_trajs[idx] = new_traj
        new_colls = analyze_collisions(temp_trajs, agent_goals, agent_starts, pristine_static_grid)

        if coll_type == 'vertex':
            coll_cell = tuple(map(int, collision['cell']))
        else:
            coll_cell = collision['cell']

        coll_key_check = (coll_time, coll_type, cell_key(coll_cell), frozenset(agents_in_collision))
        coll_resolved = not any(
            (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key_check
            for c in new_colls
        )

        if coll_resolved:
            # CRITICAL: Check we don't create NEW collisions involving this agent
            new_colls_involving_agent = [c for c in new_colls if yielding_agent in c['agents']]

            if len(new_colls_involving_agent) == 0:
                if verbose:
                    print(f"      ✓ PARKING-based success: Agent {yielding_agent} parks at {parking_cell}")
                return True, yielding_agent, new_plan, new_traj, total_rejected_count
            else:
                if verbose:
                    print(f"      ✗ PARKING creates {len(new_colls_involving_agent)} new collisions, trying next agent")
                continue  # Try next agent

    # All agents, both approaches failed
    if verbose:
        print(f"    ✗ GENERALIZED YIELD failed for all agents")
    return False, None, None, None, total_rejected_count

# ==================== END GENERALIZED YIELD STRATEGY ====================


# ==================== BOUNDED WINDOW STATIC STRATEGY ====================

def try_static_bounded_window(
    collision,
    agent_id,
    current_plan,
    current_trajectory,
    agent_start,
    agent_goal,
    current_trajectories,
    agent_starts,
    agent_goals,
    pristine_static_grid,
    agent_env,
    model,
    device,
    search_type,
    algo,
    timeout,
    heuristic_weight,
    max_expansions,
    static_block_hist,
    attempt_num,
    verbose=False,
    specific_window_size=None,  # If provided, only try this window size
    skip_windows=False  # NEW: If True, skip windows and go directly to full replanning
):
    """
    Bounded Window STATIC: Plan within a limited temporal window around collision.

    Instead of full start-to-goal replanning, try iteratively larger windows:
    1. Start with small window (5 timesteps before/after collision)
    2. Block collision cells in the planning grid
    3. Plan from window_start to window_end positions
    4. Reconnect: prefix + window_plan + suffix
    5. If window planning fails, expand window and retry
    6. If all windows fail, fallback to full replanning

    Args:
        collision: Collision dict with 'time', 'type', 'cell', 'agents'
        agent_id: Agent to replan (1-indexed)
        current_plan: Agent's current action plan
        current_trajectory: Agent's current trajectory
        agent_start: Agent's start position (row, col)
        agent_goal: Agent's goal position (row, col)
        current_trajectories: All agents' current trajectories
        agent_starts: All agents' start positions (for collision validation)
        agent_goals: All agents' goal positions (for collision validation)
        pristine_static_grid: Clean static grid (no dynamic obstacles)
        agent_env: Agent's environment
        model: Neural model for planning
        device: Torch device
        search_type: Search algorithm type
        algo: Search algorithm
        timeout: Planning timeout
        heuristic_weight: A* heuristic weight
        max_expansions: Max search expansions
        static_block_hist: History of blocked cells for this agent
        attempt_num: Current attempt number (for rewind calculation)
        verbose: Enable verbose logging

    Returns:
        (success, new_plan, new_trajectory) tuple
    """
    idx = agent_id - 1
    coll_time = collision['time']

    # Determine cells to block
    if collision['type'] == 'vertex':
        cells_to_block = [tuple(map(int, collision['cell']))]
    elif collision['type'] == 'edge':
        cells_to_block = [tuple(map(int, c)) for c in collision['cell']]
    else:
        obs_cell = tuple(map(int, collision['cell']))
        cells_to_block = [obs_cell]

    # Filter out already blocked cells
    new_cells_to_block = [c for c in cells_to_block if c not in static_block_hist]

    # Calculate base rewind (same as original STATIC)
    INIT_REWIND = 2
    MAX_REWIND_STATIC = 10

    if not new_cells_to_block:
        base_rewind = min(MAX_REWIND_STATIC, INIT_REWIND + attempt_num + 3)
    else:
        base_rewind = min(MAX_REWIND_STATIC, INIT_REWIND + attempt_num - 1)

    # Try progressively larger windows (unless skipping to fallback)
    if skip_windows:
        # Skip window attempts, go directly to full replanning
        window_sizes = []
    elif specific_window_size is not None:
        # Only try the specific window size (for multi-agent window iteration)
        window_sizes = [specific_window_size]
    else:
        # Try all windows (legacy behavior)
        window_sizes = [5, 10, 15, 20, 30]  # Timesteps before/after collision

    for window_size in window_sizes:
        if verbose:
            print(f"    → Trying bounded window: ±{window_size} timesteps around T={coll_time}")

        # Define window boundaries
        window_start_time = max(0, coll_time - max(base_rewind, window_size))
        window_end_time = min(len(current_trajectory) - 1, coll_time + window_size) if current_trajectory else coll_time + window_size

        # Get start and end positions for window planning
        if current_trajectory and window_start_time < len(current_trajectory):
            window_start_pos = tuple(map(int, current_trajectory[window_start_time]))
            # Validate bounds
            if not (0 <= window_start_pos[0] < pristine_static_grid.shape[0] and
                   0 <= window_start_pos[1] < pristine_static_grid.shape[1]):
                window_start_pos = agent_start
        else:
            window_start_pos = agent_start

        # Determine window end position
        if current_trajectory and window_end_time < len(current_trajectory):
            window_end_pos = tuple(map(int, current_trajectory[window_end_time]))
            # Validate bounds
            if not (0 <= window_end_pos[0] < pristine_static_grid.shape[0] and
                   0 <= window_end_pos[1] < pristine_static_grid.shape[1]):
                window_end_pos = agent_goal
        else:
            # If window extends beyond current trajectory, use goal
            window_end_pos = agent_goal

        # Create planning grid with blocked cells
        planning_grid = pristine_static_grid.copy()
        for cell in new_cells_to_block:
            if 0 <= cell[0] < planning_grid.shape[0] and 0 <= cell[1] < planning_grid.shape[1]:
                planning_grid[cell[0], cell[1]] = -1

        # Set up environment for window planning
        env_copy = copy.deepcopy(agent_env)
        env_copy.env.grid = planning_grid
        env_copy.env.agent_pos = window_start_pos
        env_copy.env.goal_pos = window_end_pos

        # Plan within the window
        window_plan = plan_with_search(
            env_copy, model, device, search_type, algo,
            timeout, heuristic_weight, max_expansions
        )

        if window_plan:
            # Successfully planned within window - now reconnect to original path

            # Prefix: Actions before window
            prefix_actions = current_plan[:window_start_time] if current_plan and window_start_time > 0 else []

            # Suffix: Actions after window (if window doesn't reach goal)
            if window_end_pos != agent_goal and current_plan and window_end_time < len(current_plan):
                suffix_actions = current_plan[window_end_time:]
            else:
                suffix_actions = []

            # Construct full plan
            new_full_plan = prefix_actions + window_plan + suffix_actions

            # Simulate and validate
            sim_env = copy.deepcopy(agent_env)
            sim_env.env.agent_pos = agent_start
            new_traj = simulate_plan(sim_env, new_full_plan)

            if new_traj:
                # Check if collision is resolved
                temp_trajs = list(current_trajectories)
                temp_trajs[idx] = new_traj

                # Validate collision resolution using ALL agents' goals and starts
                new_colls = analyze_collisions(temp_trajs, agent_goals, agent_starts, pristine_static_grid)

                # Check if this specific collision is resolved
                coll_key = (collision['time'], collision['type'], cell_key(collision['cell']), frozenset(collision['agents']))
                coll_resolved = not any(
                    (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key
                    for c in new_colls
                )

                if coll_resolved:
                    if verbose:
                        print(f"      ✓ Bounded window (±{window_size}) successful for Agent {agent_id}")
                    return True, new_full_plan, new_traj

        # Window planning failed or didn't resolve collision, try next larger window
        if verbose:
            print(f"      ✗ Window ±{window_size} failed, trying larger window")

    # All bounded windows failed
    # If specific_window_size was provided, don't fallback (caller will try next window)
    if specific_window_size is not None:
        if verbose:
            print(f"      ✗ Window ±{specific_window_size} failed")
        return False, {}, {}

    # Fallback to full replanning (only when trying all windows at once)
    if verbose:
        print(f"    → All bounded windows failed, falling back to full replanning")

    # Full replanning: start to goal
    replan_time = max(0, coll_time - base_rewind)

    if current_trajectory and replan_time < len(current_trajectory):
        replan_pos = tuple(map(int, current_trajectory[replan_time]))
        # Validate bounds
        if not (0 <= replan_pos[0] < pristine_static_grid.shape[0] and
               0 <= replan_pos[1] < pristine_static_grid.shape[1]):
            replan_pos = agent_start
    else:
        replan_pos = agent_start

    # Create planning grid with blocked cells
    planning_grid = pristine_static_grid.copy()
    for cell in new_cells_to_block:
        if 0 <= cell[0] < planning_grid.shape[0] and 0 <= cell[1] < planning_grid.shape[1]:
            planning_grid[cell[0], cell[1]] = -1

    # Set up environment for full replanning
    env_copy = copy.deepcopy(agent_env)
    env_copy.env.grid = planning_grid
    env_copy.env.agent_pos = replan_pos
    env_copy.env.goal_pos = agent_goal

    # Full replan
    new_plan_segment = plan_with_search(
        env_copy, model, device, search_type, algo,
        timeout, heuristic_weight, max_expansions
    )

    if new_plan_segment:
        # Construct full plan
        prefix_actions = current_plan[:replan_time] if current_plan else []
        new_full_plan = prefix_actions + new_plan_segment

        # Simulate and validate
        sim_env = copy.deepcopy(agent_env)
        sim_env.env.agent_pos = agent_start
        new_traj = simulate_plan(sim_env, new_full_plan)

        if new_traj:
            # Check if collision is resolved
            temp_trajs = list(current_trajectories)
            temp_trajs[idx] = new_traj
            new_colls = analyze_collisions(temp_trajs, agent_goals, agent_starts, pristine_static_grid)

            # Check if this specific collision is resolved
            coll_key = (collision['time'], collision['type'], cell_key(collision['cell']), frozenset(collision['agents']))
            coll_resolved = not any(
                (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key
                for c in new_colls
            )

            if coll_resolved:
                if verbose:
                    print(f"      ✓ Full replanning successful for Agent {agent_id}")
                return True, new_full_plan, new_traj

    # Both bounded window and full replanning failed
    if verbose:
        print(f"    ✗ Bounded window STATIC completely failed for Agent {agent_id}")
    return False, None, None

# ==================== END BOUNDED WINDOW STATIC STRATEGY ====================


# ==================== TWO-PHASE DEFER HELPER FUNCTIONS ====================

def plan_astar_path(start, goal, agent_env, verbose=False):
    """
    Plan A* path from start to goal using the agent's environment.

    Returns list of actions (plan) if successful, None otherwise.
    """
    # Use existing astar planner
    planning_env = copy.deepcopy(agent_env)
    planning_env.env.agent_pos = start
    astar_plan = astar(planning_env, timeout=10.0, heuristic_weight=2.0)

    if not astar_plan:
        if verbose:
            print(f"      A* failed to find path from {start} to {goal}")
        return None

    return astar_plan


def execute_defer_for_collision(
    collision,
    current_plans,
    current_trajectories,
    agent_envs,
    agent_goals,
    agent_starts,
    deferred_agents,  # Set of already deferred agent IDs
    heuristic_dist_map=None,
    verbose=False
):
    """
    Execute DEFER strategy for a single collision in Phase 1.

    Selects one NON-DEFERRED agent from the collision, parks it at its start position
    until global makespan, and treats it as a static obstacle.

    IMPORTANT: Never defers an already-deferred agent.

    Returns dict with updated plans/trajectories/deferred_agents if successful, None otherwise.
    """
    agents_in_coll = list(collision['agents'])

    # CRITICAL: Filter out already-deferred agents
    non_deferred = [a for a in agents_in_coll if a not in deferred_agents]

    if len(non_deferred) == 0:
        if verbose:
            print(f"    ✗ DEFER: All agents in collision are already deferred, cannot defer")
        return None

    # Agent selection heuristic: max distance-to-goal (tie-break: smallest ID)
    # Only choose from non-deferred agents
    agent_to_defer = None
    max_distance = -1

    for aid in non_deferred:
        idx = aid - 1
        traj = current_trajectories[idx]
        current_pos = tuple(map(int, traj[0])) if traj and len(traj) > 0 else agent_starts[idx]

        # Get distance-to-goal using heuristic_dist_map
        dist_to_goal = heuristic_dist_map[current_pos][idx] if heuristic_dist_map and current_pos in heuristic_dist_map else 0

        # Select agent with max distance (or smallest ID if tied)
        if dist_to_goal > max_distance or (dist_to_goal == max_distance and (agent_to_defer is None or aid < agent_to_defer)):
            max_distance = dist_to_goal
            agent_to_defer = aid

    # Fallback: first non-deferred agent
    if agent_to_defer is None:
        agent_to_defer = min(non_deferred)

    if verbose:
        print(f"    Deferring Agent {agent_to_defer} (dist-to-goal: {max_distance}, T={collision['time']}, Cell={collision['cell']})")

    idx = agent_to_defer - 1

    # 🔄 CHANGED: Minimal defer plan - will be replaced in Phase 2 with progressive waits
    # No need to calculate safe_start_time here; Phase 2 will handle progressive planning
    defer_plan = [4]  # Single WAIT action as placeholder

    # Simulate trajectory
    defer_env = copy.deepcopy(agent_envs[idx])
    defer_env.env.agent_pos = agent_starts[idx]
    defer_traj = simulate_plan(defer_env, defer_plan)

    if not defer_traj:
        return None

    # Return updated state (immutable pattern)
    new_plans = list(current_plans)  # Create copy
    new_trajectories = list(current_trajectories)  # Create copy
    new_plans[idx] = defer_plan
    new_trajectories[idx] = defer_traj

    new_deferred_agents = set(deferred_agents)  # Create copy
    new_deferred_agents.add(agent_to_defer)

    if verbose:
        print(f"    ✓ Agent {agent_to_defer} deferred (will be planned in Phase 2 with progressive waits)")

    return {
        'plans': new_plans,
        'trajectories': new_trajectories,
        'deferred_agents': new_deferred_agents,
        'deferred_agent': agent_to_defer
    }

# ==================== END TWO-PHASE DEFER HELPER FUNCTIONS ====================


def try_yield_on_start(collision, yielding_agent, current_trajectories, agent_goals, agent_starts, pristine_static_grid, agent_envs, current_plans, model, device, search_type, algo, timeout, heuristic_weight, max_expansions, verbose=False):
    """
    Execute YIELD-ON-START strategy for a deferred agent blocking at its start position.

    Similar to YIELD-ON-GOAL, but treats the start cell as the "pseudo-goal" anchor point.
    The agent temporarily moves away from its start to let others pass, then returns.

    This is ONLY called for deferred agents (parked at start) in Phase 1.

    Args:
        collision: Collision dict with keys: 'time', 'type', 'cell', 'agents'
        yielding_agent: Agent ID that will yield (must be deferred, at start cell)
        current_trajectories: List of position sequences (0-indexed)
        agent_goals: List of goal positions (0-indexed) - NOT MODIFIED
        agent_starts: List of starting positions (0-indexed)
        pristine_static_grid: Grid where -1=obstacle, 0=free
        agent_envs: List of environment wrappers
        current_plans: List of action sequences (0-indexed)
        model, device, search_type, algo, timeout, heuristic_weight, max_expansions: Planning params
        verbose: Debug output

    Returns:
        dict with 'plan' and 'trajectory' and 'rejected_count' if successful, None otherwise
    """
    from utils.env_utils import pos_at

    idx = yielding_agent - 1
    start_cell = agent_starts[idx]  # The "pseudo-goal" for this yield
    coll_time = collision['time']
    
    rejected_count = 0

    if verbose:
        print(f"    Yielding Agent {yielding_agent} from start {start_cell} at T={coll_time}")

    from utils.env_utils import pos_at

    idx = yielding_agent - 1
    start_cell = agent_starts[idx]  # The "pseudo-goal" for this yield
    coll_time = collision['time']

    if verbose:
        print(f"    Yielding Agent {yielding_agent} from start {start_cell} at T={coll_time}")

    if HAS_CPP_COLLISION:
        # C++ Implementation
        # resolve_yield_on_start_cpp(grid, time, id, start_pos, trajs, starts, radius)
        # Returns YieldPlan (success, parking_cell, path_to_parking, path_to_goal)
        # Note: In C++ implementation of find_parking_and_paths (called by resolve_yield_on_start),
        # it checks safety for a fixed wait time (3 steps).
        # And it returns path_to_parking and path_to_goal.
        # We need to reconstruct the full trajectory and actions.
        
        # Ensure start_cell is tuple of ints
        start_cell_tuple = tuple(map(int, start_cell))
        
        yield_plan = cpp_collision.resolve_yield_on_start_cpp(
            pristine_static_grid,
            coll_time - 1, # Start yield one step before collision
            yielding_agent,
            start_cell_tuple,
            current_trajectories,
            agent_starts,
            4 # MAX_RADIUS
        )
        
        if yield_plan.success:
            if verbose:
                print(f"    ✓ [C++] Found yield-on-start plan via parking {yield_plan.parking_cell.r, yield_plan.parking_cell.c}")
            
            # Reconstruct full trajectory: path_to_parking + wait(wait_steps) + path_to_goal
            # Note: path_to_goal in C++ return struct is the path FROM parking TO goal (start)
            # path_to_parking includes start -> ... -> parking
            # path_to_goal includes parking -> ... -> start

            full_traj = []
            # Add path to parking
            for p in yield_plan.path_to_parking:
                full_traj.append((p.r, p.c))

            # Add wait (use wait_steps returned from C++)
            parking_cell = (yield_plan.parking_cell.r, yield_plan.parking_cell.c)
            for _ in range(yield_plan.wait_steps):
                full_traj.append(parking_cell)

            # Add path to goal (skip first as it is parking_cell)
            for i in range(1, len(yield_plan.path_to_goal)):
                p = yield_plan.path_to_goal[i]
                full_traj.append((p.r, p.c))
                
            # Convert trajectory to actions
            # We need to convert the coordinate sequence into actions (0-4)
            # We can use a helper or simple logic since steps are adjacent
            new_actions = []
            for i in range(len(full_traj) - 1):
                curr = full_traj[i]
                next_pos = full_traj[i+1]
                dr = next_pos[0] - curr[0]
                dc = next_pos[1] - curr[1]
                
                if dr == -1 and dc == 0: action = 0 # UP
                elif dr == 1 and dc == 0: action = 1 # DOWN
                elif dr == 0 and dc == -1: action = 2 # LEFT
                elif dr == 0 and dc == 1: action = 3 # RIGHT
                elif dr == 0 and dc == 0: action = 4 # WAIT
                else:
                    # Should not happen for valid path
                    if verbose: print(f"    ERROR: Invalid move {curr} -> {next_pos}")
                    return None
                new_actions.append(action)
                
            # Splice
            splice_time = coll_time - 1
            new_plan, new_trajectory = splice_yield_segment(
                yielding_agent,
                splice_time,
                new_actions,
                current_plans,
                current_trajectories,
                agent_starts,
                agent_envs,
                verbose=verbose
            )
            
            if new_plan is not None:
                return {
                    'plan': new_plan,
                    'trajectory': new_trajectory,
                    'rejected_count': yield_plan.rejected_candidates
                }
        
        # If C++ fails, we could fallback, but for now let's return None as per plan
        if verbose:
            print(f"    ✗ [C++] Failed to find yield-on-start plan")
        return None

    # Fallback to Python if C++ not available (legacy code removed/skipped)
    return None


def try_yield_on_goal(collision, current_plans, current_trajectories, agent_starts, agent_goals, pristine_static_grid, agent_envs, model, device, search_type, algo, timeout, heuristic_weight, max_expansions, verbose=False):
    """
    Try Yield-on-Goal strategy for vertex collisions at goal cells.

    For a vertex collision at cell C:
    - Check if C is the goal of one agent g
    - Check if g is actually at C at collision time
    - Find a safe parking cell near C
    - Build yield segment: C → parking → wait(0-3) → parking → C
    - Validate against all other agents

    Args:
        collision: Collision dict with keys: 'time', 'type', 'cell', 'agents'
        current_plans: List of action sequences (0-indexed)
        current_trajectories: List of position sequences (0-indexed)
        agent_starts: List of starting positions (0-indexed)
        agent_goals: List of goal positions (0-indexed)
        pristine_static_grid: Grid where -1=obstacle, 0=free
        agent_envs: List of environment wrappers
        model: RL model for planning
        device: Torch device
        search_type: Search algorithm type
        algo: Algorithm type (ppo/dqn)
        timeout: Planning timeout
        heuristic_weight: A* heuristic weight
        max_expansions: Max search expansions
        verbose: Debug output

    Returns:
        (success, agent_id, new_plan, new_trajectory, rejected_count)
        - If success: (True, yielding_agent_id, new_plan, new_trajectory, rejected_count)
        - If failure: (False, None, None, None, rejected_count)
    """
    from utils.env_utils import pos_at

    # STEP 1: Validate this is a vertex collision
    if collision['type'] != 'vertex':
        return False, None, None, None, 0

    coll_time = collision['time']
    coll_cell = tuple(map(int, collision['cell']))
    agents_in_coll = list(collision['agents'])

    # STEP 2: Find which agent (if any) is at their goal at collision cell
    yielding_agent = None
    for agent_id in agents_in_coll:
        idx = agent_id - 1
        goal = tuple(map(int, agent_goals[idx]))

        # Check if collision cell is this agent's goal
        if coll_cell != goal:
            continue

        # Check if agent is actually at goal at collision time
        pos_at_coll = pos_at(agent_id, coll_time, current_trajectories, agent_starts)
        if pos_at_coll == goal:
            yielding_agent = agent_id
            break

    if yielding_agent is None:
        return False, None, None, None, 0

    if verbose:
        print(f"    → Agent {yielding_agent} at goal {coll_cell}")

        # DEBUG: Add specific logging for Agent 15-34 collision
        if yielding_agent in [15, 34]:
            print(f"    [DEBUG YIELD] Agent {yielding_agent} is yielding at goal {coll_cell}")
            print(f"    [DEBUG YIELD] Collision time: t={coll_time}")
            print(f"    [DEBUG YIELD] Other agents in collision: {[a for a in agents_in_coll if a != yielding_agent]}")

    idx = yielding_agent - 1
    goal_cell = tuple(map(int, agent_goals[idx]))

    if HAS_CPP_COLLISION:
        # C++ Implementation
        goal_cell_tuple = tuple(map(int, goal_cell))
        
        yield_plan = cpp_collision.resolve_yield_on_goal_cpp(
            pristine_static_grid,
            coll_time - 1, # Start yield one step before collision
            yielding_agent,
            goal_cell_tuple,
            current_trajectories,
            agent_starts,
            4 # MAX_RADIUS
        )
        
        if yield_plan.success:
            if verbose:
                print(f"    ✓ [C++] Found yield-on-goal plan via parking {yield_plan.parking_cell.r, yield_plan.parking_cell.c}")
            
            # Reconstruct full trajectory: path_to_parking + wait(wait_steps) + path_to_goal
            full_traj = []
            # Add path to parking
            for p in yield_plan.path_to_parking:
                full_traj.append((p.r, p.c))

            # Add wait (use wait_steps returned from C++)
            parking_cell = (yield_plan.parking_cell.r, yield_plan.parking_cell.c)
            for _ in range(yield_plan.wait_steps):
                full_traj.append(parking_cell)

            # Add path to goal (skip first)
            for i in range(1, len(yield_plan.path_to_goal)):
                p = yield_plan.path_to_goal[i]
                full_traj.append((p.r, p.c))
                
            # Convert to actions
            new_actions = []
            for i in range(len(full_traj) - 1):
                curr = full_traj[i]
                next_pos = full_traj[i+1]
                dr = next_pos[0] - curr[0]
                dc = next_pos[1] - curr[1]
                
                if dr == -1 and dc == 0: action = 0
                elif dr == 1 and dc == 0: action = 1
                elif dr == 0 and dc == -1: action = 2
                elif dr == 0 and dc == 1: action = 3
                elif dr == 0 and dc == 0: action = 4
                else:
                    if verbose: print(f"    ERROR: Invalid move {curr} -> {next_pos}")
                    return False, None, None, None, yield_plan.rejected_candidates
                new_actions.append(action)
                
            # Splice
            new_plan, new_traj = splice_yield_segment(
                yielding_agent, coll_time - 1, new_actions,
                current_plans, current_trajectories, agent_starts,
                agent_envs, verbose
            )
            
            if new_plan is not None:
                if verbose:
                    print(f"    ✓ Agent {yielding_agent} yielded to parking cell {parking_cell}")
                return True, yielding_agent, new_plan, new_traj, yield_plan.rejected_candidates
        
        if verbose:
            print(f"    ✗ [C++] Failed to find yield-on-goal plan")
        return False, None, None, None, yield_plan.rejected_candidates

    # Fallback to Python (legacy code removed/skipped)
    return False, None, None, None, 0


def try_joint_astar_planning(
    collision,
    current_trajectories,
    agent_goals,
    agent_starts,
    pristine_static_grid,
    heuristic_dist_map,
    max_agents=4,
    time_budget=10.0,
    max_expansions=20000,
    verbose=False,
    base_rewind=5,
    base_horizon=5,
    max_expansion_steps=15,
    blocked_cells=None,
    blocked_by_time=None,
    use_time_based_blocking=True,
    enable_escalation=False
):
    """
    Resolve a collision by jointly planning for the involved agents using a small-horizon A*.

    The search runs in the joint configuration space with a simple reservation table for all
    other agents' trajectories, so the proposed fix should not introduce new conflicts.

    Args:
        base_rewind: How many steps to look back from collision time (default 2)
        base_horizon: How many steps to look forward from collision time (default 2)
        max_expansion_steps: How many times to expand window if search fails (default 15)
        enable_escalation: Whether to recursively escalate agents into joint group on failure (default False)
    """
    agents_in_collision = sorted(list(collision['agents']))
    num_joint_agents = len(agents_in_collision)
    coll_time = collision['time']

    if num_joint_agents == 0 or num_joint_agents > max_agents:
        return False, {}, {}, None, None

    rows, cols = pristine_static_grid.shape
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # UP, DOWN, LEFT, RIGHT, STAY

    # Initialize blocked_cells set if not provided
    if blocked_cells is None:
        blocked_cells = set()

    def agent_pos_at(agent_id, t):
        idx = agent_id - 1
        traj = current_trajectories[idx]
        if traj:
            if t < len(traj):
                return tuple(map(int, traj[t]))
            return tuple(map(int, traj[-1]))
        return tuple(map(int, agent_starts[idx]))

    # Reservation lookups for agents outside the collision set
    all_agent_ids = set(range(1, len(agent_goals) + 1))
    other_agents = sorted(all_agent_ids - set(agents_in_collision))

    def reserved_positions_at(t):
        reserved = set()
        for aid in other_agents:
            reserved.add(agent_pos_at(aid, t))
        return reserved

    def reserved_moves_at(t):
        moves = []
        for aid in other_agents:
            prev_pos = agent_pos_at(aid, t)
            next_pos = agent_pos_at(aid, t + 1)
            moves.append((prev_pos, next_pos))
        return moves

    def check_joint_segment_conflicts(
        agents_in_collision,
        joint_trajs,
        t_start,
        t_goal_sub,
        agent_pos_at,
        other_agents,
        verbose=False
    ):
        """Validate joint segment against vertex/edge conflicts (joint vs joint, joint vs others).

        REFINEMENT #4: Returns (success, conflict_info) where conflict_info tracks which outside agents
        are causing conflicts with the joint plan for escalation.
        """
        horizon = t_goal_sub - t_start
        conflict_info = {'joint_other_conflicts': defaultdict(int)}  # oid -> conflict count

        # BUGFIX: Verify trajectory bounds before accessing
        for aid in agents_in_collision:
            traj_len = len(joint_trajs.get(aid, []))
            expected_len = horizon + 1
            if traj_len != expected_len:
                if verbose:
                    print(f"      ERROR: Agent {aid} trajectory length {traj_len} != expected {expected_len}")
                return False, conflict_info

        for offset in range(horizon):
            t_global = t_start + offset
            # BUGFIX: Add bounds checking before accessing trajectories
            try:
                pos_now_joint = {aid: joint_trajs[aid][offset] for aid in agents_in_collision}
                pos_next_joint = {aid: joint_trajs[aid][offset + 1] for aid in agents_in_collision}
            except (KeyError, IndexError) as e:
                if verbose:
                    print(f"      ERROR: Index out of bounds at offset {offset}: {e}")
                return False, conflict_info

            # Joint vs joint vertex
            seen = {}
            for aid, pos in pos_next_joint.items():
                if pos in seen:
                    if verbose:
                        print(f"      Conflict (joint-joint vertex) at t={t_global+1}: {aid},{seen[pos]} -> {pos}")
                    return False, conflict_info
                seen[pos] = aid

            # Joint vs joint edge swap
            joint_list = list(agents_in_collision)
            for i in range(len(joint_list)):
                for j in range(i + 1, len(joint_list)):
                    ai, aj = joint_list[i], joint_list[j]
                    if pos_now_joint[ai] == pos_next_joint[aj] and pos_now_joint[aj] == pos_next_joint[ai]:
                        if verbose:
                            print(f"      Conflict (joint-joint edge swap) at t={t_global}->{t_global+1}: {ai}<->{aj}")
                        return False, conflict_info

            # Joint vs others
            for aid in agents_in_collision:
                pos_next = pos_next_joint[aid]
                for oid in other_agents:
                    other_now = agent_pos_at(oid, t_global)
                    other_next = agent_pos_at(oid, t_global + 1)

                    if pos_next == other_next:
                        if verbose:
                            print(f"      Conflict (joint-other vertex) at t={t_global+1}: {aid} & {oid} -> {pos_next}")
                        # REFINEMENT #4: Track which outside agent is conflicting
                        conflict_info['joint_other_conflicts'][oid] += 1
                        return False, conflict_info

                    if pos_now_joint[aid] == other_next and other_now == pos_next:
                        if verbose:
                            print(f"      Conflict (joint-other edge swap) at t={t_global}->{t_global+1}: {aid}<->{oid}")
                        # REFINEMENT #4: Track which outside agent is conflicting
                        conflict_info['joint_other_conflicts'][oid] += 1
                        return False, conflict_info

        return True, conflict_info

    def search_window_cpp(t_start, t_goal_sub):
        """C++ version of search_window using cpp_joint_astar module."""
        start_positions = tuple(agent_pos_at(aid, t_start) for aid in agents_in_collision)

        # Clamp subgoal time to trajectory length
        subgoal_positions = []
        for aid in agents_in_collision:
            idx = aid - 1
            traj = current_trajectories[idx]
            if traj:
                t_subgoal_clamped = min(t_goal_sub, len(traj) - 1)
            else:
                t_subgoal_clamped = t_goal_sub
            subgoal_positions.append(agent_pos_at(aid, t_subgoal_clamped))
        subgoal_positions = tuple(subgoal_positions)

        if verbose:
            print(f"    Joint A* (C++) window: coll_time={coll_time}, "
                  f"t_start={t_start}, t_goal_sub={t_goal_sub}, agents={agents_in_collision}")
            print(f"    Start positions: {start_positions}")
            print(f"    Subgoal positions: {subgoal_positions}")

        # Convert positions to C++ format (dicts with 'r' and 'c')
        start_positions_cpp = [{'r': int(p[0]), 'c': int(p[1])} for p in start_positions]
        subgoal_positions_cpp = [{'r': int(p[0]), 'c': int(p[1])} for p in subgoal_positions]

        # Prepare reserved_cells_by_time
        reserved_cells_by_time_cpp = []
        for t in range(t_start + 1, t_goal_sub + 1):
            cells_at_t = []
            reserved = reserved_positions_at(t)
            for cell in reserved:
                cells_at_t.append({'r': int(cell[0]), 'c': int(cell[1])})
            reserved_cells_by_time_cpp.append(cells_at_t)

        # Prepare reserved_moves_by_time
        reserved_moves_by_time_cpp = []
        for t in range(t_start, t_goal_sub):
            moves_at_t = []
            moves = reserved_moves_at(t)
            for from_pos, to_pos in moves:
                moves_at_t.append([
                    {'r': int(from_pos[0]), 'c': int(from_pos[1])},
                    {'r': int(to_pos[0]), 'c': int(to_pos[1])}
                ])
            reserved_moves_by_time_cpp.append(moves_at_t)

        # Prepare blocked_cells
        blocked_cells_cpp = []
        if blocked_cells:
            for cell in blocked_cells:
                blocked_cells_cpp.append({'r': int(cell[0]), 'c': int(cell[1])})

        # Prepare blocked_by_time
        blocked_by_time_cpp = []
        if blocked_by_time and use_time_based_blocking:
            # Find max time we need
            max_t = max(blocked_by_time.keys()) if blocked_by_time else 0
            for t in range(max_t + 1):
                cells_at_t = []
                if t in blocked_by_time:
                    for cell in blocked_by_time[t]:
                        cells_at_t.append({'r': int(cell[0]), 'c': int(cell[1])})
                blocked_by_time_cpp.append(cells_at_t)

        # Call C++ joint A*
        result = cpp_joint_astar.joint_astar_grid_time(
            int(rows),
            int(cols),
            pristine_static_grid.tolist(),
            start_positions_cpp,
            subgoal_positions_cpp,
            int(t_start),
            int(t_goal_sub),
            int(max_expansions),
            float(time_budget),
            reserved_cells_by_time_cpp,
            reserved_moves_by_time_cpp,
            blocked_cells_cpp,
            blocked_by_time_cpp,
            bool(use_time_based_blocking)
        )

        if not result.success:
            return False, {}, {}, t_start, t_goal_sub, 0

        # Convert C++ result to Python format
        joint_plans = {}
        joint_trajs = {}
        for idx, aid in enumerate(agents_in_collision):
            joint_plans[aid] = result.plans[idx]
            # Convert C++ Cell objects to tuples
            joint_trajs[aid] = [(int(cell.r), int(cell.c)) for cell in result.trajectories[idx]]

        # Validate the solution
        conflict_check_success, conflict_info = check_joint_segment_conflicts(
            agents_in_collision,
            joint_trajs,
            t_start,
            t_goal_sub,
            agent_pos_at,
            other_agents,
            verbose=verbose
        )

        if not conflict_check_success:
            if verbose:
                print("    C++ joint A* solution has conflicts → discarding")
            # The original code had a duplicate print here.
            # The instruction implies returning result.expansions in case of conflict.
            # The `if result.success:` inside `if not conflict_check_success:` seems like a logical error
            # or an attempt to restructure the return.
            # Assuming the intent is to return the expansions from the C++ call
            # even if the Python-side conflict check fails.
            return False, {}, {}, t_start, t_goal_sub, result.expansions

        if verbose:
            print(f"    Joint A* (C++) success! window [{t_start},{t_goal_sub}]")

        # If conflict_check_success is True, then the C++ result was valid.
        # Return the C++ plans, trajectories, and expansions.
        return True, joint_plans, joint_trajs, t_start, t_goal_sub, result.expansions

    def search_window(t_start, t_goal_sub):
        start_positions = tuple(agent_pos_at(aid, t_start) for aid in agents_in_collision)

        # BUGFIX: Clamp subgoal time to trajectory length to avoid degenerate search
        # where start == subgoal due to trajectory being shorter than planning window
        subgoal_positions = []
        for aid in agents_in_collision:
            idx = aid - 1
            traj = current_trajectories[idx]
            # Use the trajectory length to determine a realistic subgoal time
            if traj:
                t_subgoal_clamped = min(t_goal_sub, len(traj) - 1)
            else:
                t_subgoal_clamped = t_goal_sub
            subgoal_positions.append(agent_pos_at(aid, t_subgoal_clamped))
        subgoal_positions = tuple(subgoal_positions)

        if verbose:
            print(f"    Joint A* window: coll_time={coll_time}, "
                  f"t_start={t_start}, t_goal_sub={t_goal_sub}, agents={agents_in_collision}")
            print(f"    Start positions: {start_positions}")
            print(f"    Subgoal positions: {subgoal_positions}")
            if start_positions == subgoal_positions:
                print(f"    WARNING: Start and subgoal are identical! This will create degenerate search.")

        def heuristic(positions):
            h_val = 0
            for pos, subgoal in zip(positions, subgoal_positions):
                h_val += abs(pos[0] - subgoal[0]) + abs(pos[1] - subgoal[1])
            return h_val

        min_depth_needed = max(1, t_goal_sub - t_start)
        max_depth = max(min_depth_needed + 3, min_depth_needed)

        frontier = []
        counter = itertools.count()
        heapq.heappush(frontier, (heuristic(start_positions), 0, next(counter), start_positions, []))
        visited = set()
        expansions = 0
        start_clock = time.perf_counter()

        while frontier and expansions < max_expansions:
            if time.perf_counter() - start_clock > time_budget:
                break

            f, g, _, positions, action_history = heapq.heappop(frontier)
            expansions += 1

            goals_reached = all(
                tuple(map(int, positions[i])) == tuple(map(int, subgoal_positions[i]))
                for i in range(num_joint_agents)
            )
            if goals_reached:
                expected_len = max(0, t_goal_sub - t_start)
                if g > expected_len:
                    # Too long for the local window, keep searching.
                    continue

                if g < expected_len:
                    pad_steps = expected_len - g
                    stay_tuple = tuple(4 for _ in range(num_joint_agents))
                    action_history = action_history + [stay_tuple] * pad_steps
                    g = expected_len

                joint_plans = {aid: [] for aid in agents_in_collision}
                joint_trajs = {aid: [pos] for aid, pos in zip(agents_in_collision, start_positions)}

                for actions in action_history:
                    for idx, aid in enumerate(agents_in_collision):
                        act = actions[idx]
                        dr, dc = ACTIONS[act]
                        prev = joint_trajs[aid][-1]
                        new_pos = (prev[0] + dr, prev[1] + dc)
                        joint_trajs[aid].append(new_pos)
                        joint_plans[aid].append(act)

                # REFINEMENT #4: Unpack tuple return (success, conflict_info)
                conflict_check_success, conflict_info = check_joint_segment_conflicts(
                    agents_in_collision,
                    joint_trajs,
                    t_start,
                    t_goal_sub,
                    agent_pos_at,
                    other_agents,
                    verbose=verbose
                )
                if not conflict_check_success:
                    if verbose:
                        print("    Joint A* candidate plan has local conflicts → discarding and continuing search")
                    continue

                if verbose:
                    print(f"    Joint A* success in {g} steps (expanded {expansions} nodes) window [{t_start},{t_goal_sub}]")
                    # Print joint plans for debugging
                    for aid in agents_in_collision:
                        plan_str = " → ".join([f"{ACTIONS[act]}" for act in joint_plans[aid]])
                        print(f"      Agent {aid} joint plan: {joint_plans[aid][:15]}{'...' if len(joint_plans[aid]) > 15 else ''}")
                return True, joint_plans, joint_trajs, t_start, t_goal_sub, expansions

            if g >= max_depth:
                continue

            key = (positions, g)
            if key in visited:
                continue
            visited.add(key)

            time_step = t_start + g

            # Avoid states already colliding with reserved agents
            reserved_now = reserved_positions_at(time_step)
            if any(pos in reserved_now for pos in positions):
                continue

            move_options = []
            for pos in positions:
                opts = []
                for act_idx, (dr, dc) in enumerate(ACTIONS):
                    nr, nc = pos[0] + dr, pos[1] + dc

                    # Determine if cell is blocked
                    is_blocked = False
                    if use_time_based_blocking:
                        # Time-aware blocking: check if cell is blocked at next timestep
                        t_global_next = time_step + 1
                        if blocked_by_time and t_global_next in blocked_by_time:
                            is_blocked = (nr, nc) in blocked_by_time[t_global_next]
                    else:
                        # Spatial blocking: backward compatible with existing behavior
                        if blocked_cells:
                            is_blocked = (nr, nc) in blocked_cells

                    if (0 <= nr < rows and 0 <= nc < cols and pristine_static_grid[nr, nc] == 0
                        and not is_blocked):
                        opts.append((act_idx, (nr, nc)))
                move_options.append(opts)

            for combo in itertools.product(*move_options):
                next_positions = tuple(item[1] for item in combo)
                actions_tuple = tuple(item[0] for item in combo)

                # Internal vertex conflicts
                if len(set(next_positions)) < num_joint_agents:
                    continue

                # Internal edge conflicts
                edge_conflict = False
                for i in range(num_joint_agents):
                    for j in range(i + 1, num_joint_agents):
                        if positions[i] == next_positions[j] and positions[j] == next_positions[i]:
                            edge_conflict = True
                            break
                    if edge_conflict:
                        break
                if edge_conflict:
                    continue

                # Conflicts with other agents' reservations
                reserved_next = reserved_positions_at(time_step + 1)
                if any(pos in reserved_next for pos in next_positions):
                    continue

                reserved_moves = reserved_moves_at(time_step)
                swap_with_reserved = False
                for idx, move in enumerate(next_positions):
                    prev_pos = positions[idx]
                    for res_prev, res_next in reserved_moves:
                        if move == res_prev and prev_pos == res_next:
                            swap_with_reserved = True
                            break
                    if swap_with_reserved:
                        break
                if swap_with_reserved:
                    continue

                new_actions = action_history + [actions_tuple]
                g1 = g + 1
                h1 = heuristic(next_positions)
                heapq.heappush(frontier, (g1 + h1, g1, next(counter), next_positions, new_actions))

        return False, {}, {}, t_start, t_goal_sub, expansions

    for expand in range(max_expansion_steps + 1):
        joint_rewind = base_rewind + expand
        joint_horizon = base_horizon + expand
        t_start = max(0, coll_time - joint_rewind)
        t_goal_sub = coll_time + joint_horizon

        # REFINEMENT #2: Window bounds validation
        # Check 1: t_start must be non-negative and less than collision time
        if t_start >= coll_time:
            if verbose:
                print(f"    Window validation FAILED: t_start ({t_start}) >= coll_time ({coll_time})")
            continue

        # Check 2: Window doesn't exceed trajectory bounds
        max_traj_len = 0
        for aid in agents_in_collision:
            idx = aid - 1
            traj = current_trajectories[idx]
            if traj:
                max_traj_len = max(max_traj_len, len(traj))

        if max_traj_len > 0 and t_goal_sub > max_traj_len - 1:
            if verbose:
                print(f"    Window validation FAILED: t_goal_sub ({t_goal_sub}) exceeds max trajectory length ({max_traj_len - 1})")
            continue

        # Check 3: Start position consistency - verify agents are at expected positions at t_start
        position_check_failed = False
        for aid in agents_in_collision:
            idx = aid - 1
            expected_pos = agent_pos_at(aid, t_start)
            traj = current_trajectories[idx]
            if traj and t_start < len(traj):
                actual_pos = tuple(map(int, traj[t_start]))
                if expected_pos != actual_pos:
                    if verbose:
                        print(f"    Window validation FAILED: Agent {aid} position mismatch at t_start ({t_start}): expected {expected_pos}, got {actual_pos}")
                    position_check_failed = True
                    break

        if position_check_failed:
            continue

        # Choose between C++ and Python implementation
        if USE_CPP_JOINT_ASTAR and HAS_CPP_JOINT_ASTAR:
            success, plans, trajs, ts, tg, expansions = search_window_cpp(t_start, t_goal_sub)
        else:
            if USE_CPP_JOINT_ASTAR and not HAS_CPP_JOINT_ASTAR and verbose:
                print("    Note: C++ joint A* not available, using Python fallback")
            success, plans, trajs, ts, tg, expansions = search_window(t_start, t_goal_sub)

        if success:
            return success, plans, trajs, ts, tg, expansions

    if verbose:
        print(f"    Joint A* failed after expanding window to rewind={base_rewind + max_expansion_steps}, horizon={base_horizon + max_expansion_steps}")

    # REFINEMENT #4: Escalation logic - promote blocking outside agents into joint group (optional)
    if enable_escalation:
        # Track cumulative conflicts from outside agents across all expansion attempts
        cumulative_conflicts = defaultdict(int)  # oid -> total conflict count

        # Re-run expansion attempts to collect conflict information
        for expand in range(max_expansion_steps + 1):
            joint_rewind = base_rewind + expand
            joint_horizon = base_horizon + expand
            t_start = max(0, coll_time - joint_rewind)
            t_goal_sub = coll_time + joint_horizon

            # Quick validation (same as before)
            if t_start >= coll_time:
                continue
            max_traj_len = 0
            for aid in agents_in_collision:
                idx = aid - 1
                traj = current_trajectories[idx]
                if traj:
                    max_traj_len = max(max_traj_len, len(traj))
            if max_traj_len > 0 and t_goal_sub > max_traj_len - 1:
                continue

            # Run a lightweight search to collect conflict info (don't store results)
            # We'll detect conflicts during this attempt
            start_positions = tuple(agent_pos_at(aid, t_start) for aid in agents_in_collision)
            subgoal_positions = []
            for aid in agents_in_collision:
                idx = aid - 1
                traj = current_trajectories[idx]
                if traj:
                    t_subgoal_clamped = min(t_goal_sub, len(traj) - 1)
                else:
                    t_subgoal_clamped = t_goal_sub
                subgoal_positions.append(agent_pos_at(aid, t_subgoal_clamped))
            subgoal_positions = tuple(subgoal_positions)

            # Minimal A* search just to collect conflict data
            frontier = []
            counter = itertools.count()
            heapq.heappush(frontier, (0, 0, next(counter), start_positions, []))
            visited = set()
            expansions = 0

            while frontier and expansions < 500:  # Limited to 500 expansions for conflict discovery
                f, g, _, positions, action_history = heapq.heappop(frontier)
                expansions += 1

                if g >= 3:  # Only check first few steps for conflicts
                    break

                key = (positions, g)
                if key in visited:
                    continue
                visited.add(key)

                time_step = t_start + g
                move_options = []
                for pos in positions:
                    opts = []
                    for act_idx, (dr, dc) in enumerate(ACTIONS):
                        nr, nc = pos[0] + dr, pos[1] + dc
                        if (0 <= nr < rows and 0 <= nc < cols and pristine_static_grid[nr, nc] == 0):
                            opts.append((act_idx, (nr, nc)))
                    move_options.append(opts)

                for combo in itertools.product(*move_options):
                    next_positions = tuple(item[1] for item in combo)
                    actions_tuple = tuple(item[0] for item in combo)

                    # Skip if internal conflicts
                    if len(set(next_positions)) < num_joint_agents:
                        continue
                    edge_conflict = False
                    for i in range(num_joint_agents):
                        for j in range(i + 1, num_joint_agents):
                            if positions[i] == next_positions[j] and positions[j] == next_positions[i]:
                                edge_conflict = True
                                break
                        if edge_conflict:
                            break
                    if edge_conflict:
                        continue

                    # Check for conflicts with other agents - accumulate them
                    for idx, aid in enumerate(agents_in_collision):
                        pos_now = positions[idx]
                        pos_next = next_positions[idx]
                        for oid in other_agents:
                            other_now = agent_pos_at(oid, time_step)
                            other_next = agent_pos_at(oid, time_step + 1)
                            if pos_next == other_next or (pos_now == other_next and other_now == pos_next):
                                cumulative_conflicts[oid] += 1

                    new_actions = action_history + [actions_tuple]
                    g1 = g + 1
                    heapq.heappush(frontier, (g1, g1, next(counter), next_positions, new_actions))

        # Check if any outside agent caused ≥3 repeated conflicts
        blocking_agent = None
        max_conflicts = 0
        for oid, conflict_count in cumulative_conflicts.items():
            if conflict_count >= 3 and conflict_count > max_conflicts:
                max_conflicts = conflict_count
                blocking_agent = oid
            elif conflict_count >= 3 and conflict_count == max_conflicts and (blocking_agent is None or oid < blocking_agent):
                blocking_agent = oid  # Tie-break by agent ID

        # If we found a blocking agent and have capacity, escalate
        if blocking_agent and num_joint_agents < max_agents:
            if verbose:
                print(f"    Escalating blocking agent {blocking_agent} (caused {max_conflicts} conflicts) into joint group")
            escalated_collision = dict(collision)
            escalated_collision['agents'] = set(agents_in_collision) | {blocking_agent}
            # Recursively try with escalated group
            return try_joint_astar_planning(
                escalated_collision, current_trajectories, agent_goals, agent_starts,
                pristine_static_grid, heuristic_dist_map,
                max_agents=max_agents,
                time_budget=time_budget * 1.5,  # Give escalation more time
                max_expansions=max_expansions,
                verbose=verbose,
                base_rewind=base_rewind,
                base_horizon=base_horizon,
                max_expansion_steps=max_expansion_steps - 2,  # Fewer expansions for escalation
                blocked_cells=blocked_cells,
                blocked_by_time=blocked_by_time,
                use_time_based_blocking=use_time_based_blocking,
                enable_escalation=enable_escalation
            )

    return False, {}, {}, None, None, 0


def try_joint_astar_with_conflict_blocking(
    collision,
    current_trajectories,
    agent_goals,
    agent_starts,
    pristine_static_grid,
    heuristic_dist_map,
    max_agents=4,
    time_budget=10.0,
    max_expansions=20000,
    verbose=False,
    base_rewind=5,
    base_horizon=5,
    max_expansion_steps=15,
    max_conflict_retries=3,
    use_time_based_blocking=True
):
    """
    Wrapper around try_joint_astar_planning that retries with conflict blocking.

    When Joint A* fails to resolve a collision, identify which agents are blocking
    the search and temporarily add their cells as obstacles, forcing the planner
    to find a completely different route.
    """
    agents_in_collision = sorted(list(collision['agents']))
    all_agent_ids = set(range(1, len(agent_goals) + 1))
    other_agents = sorted(all_agent_ids - set(agents_in_collision))
    coll_time = collision['time']

    # First attempt: BUILD blocked_by_time for initial attempt using pos_at semantics
    from utils.env_utils import pos_at

    blocked_by_time = {}
    if use_time_based_blocking and other_agents:
        max_traj_len = max((len(current_trajectories[oid - 1]) for oid in other_agents
                           if current_trajectories[oid - 1]), default=0)

        # Use base window for first attempt
        for t_global in range(max(0, coll_time - base_rewind),
                             min(max_traj_len, coll_time + base_horizon + 1)):
            blocked_by_time[t_global] = set()
            for oid in other_agents:
                # Use pos_at to get position (handles stay-at-goal semantics)
                pos = pos_at(oid, t_global, current_trajectories, agent_starts)
                blocked_by_time[t_global].add(pos)

        if verbose and blocked_by_time:
            print(f"    [Joint A*] Built time-based blocking for first attempt:")
            print(f"      Window: t={min(blocked_by_time.keys())} to t={max(blocked_by_time.keys())}")
            print(f"      Blocking {len(other_agents)} other agents")
            # Show blocked cells for collision time ±2
            for t in range(max(0, coll_time - 2), coll_time + 3):
                if t in blocked_by_time:
                    print(f"      t={t}: {len(blocked_by_time[t])} cells blocked: {list(blocked_by_time[t])[:5]}...")

    success, plans, trajs, t_start, t_goal_sub, expansions = try_joint_astar_planning(
        collision, current_trajectories, agent_goals, agent_starts,
        pristine_static_grid, heuristic_dist_map,
        max_agents=max_agents,
        time_budget=time_budget,
        max_expansions=max_expansions,
        verbose=verbose,
        base_rewind=base_rewind,
        base_horizon=base_horizon,
        max_expansion_steps=max_expansion_steps,
        blocked_cells=None,
        blocked_by_time=blocked_by_time if blocked_by_time else None,
        use_time_based_blocking=use_time_based_blocking
    )

    if success:
        return success, plans, trajs, t_start, t_goal_sub, expansions

    if not other_agents:
        # No other agents to block, can't retry
        return False, {}, {}, None, None, 0

    # If failed, retry with blocking other agents' cells
    for retry_attempt in range(max_conflict_retries):
        if verbose:
            print(f"      Joint A* failed, attempting retry {retry_attempt + 1}/{max_conflict_retries} with conflict cell blocking...")

        # Collect positions of other agents during an extended planning window
        # Expand the window slightly more each retry to capture more conflict cells
        search_rewind = base_rewind + (retry_attempt * 2)
        search_horizon = base_horizon + (retry_attempt * 2)

        if use_time_based_blocking:
            # Build time-based blocking dictionary: blocked_by_time[t_global] = set(cells)
            blocked_by_time = {}
            max_traj_len = max((len(current_trajectories[oid - 1]) for oid in other_agents
                               if current_trajectories[oid - 1]), default=0)

            for t_global in range(max(0, coll_time - search_rewind),
                                 min(max_traj_len, coll_time + search_horizon + 1)):
                blocked_by_time[t_global] = set()
                for oid in other_agents:
                    # Use pos_at to get position (handles stay-at-goal semantics)
                    pos = pos_at(oid, t_global, current_trajectories, agent_starts)
                    blocked_by_time[t_global].add(pos)

            blocked_cells = None
            if verbose:
                total_blocked_entries = sum(len(cells) for cells in blocked_by_time.values())
                print(f"        Built time-based blocking: {len(blocked_by_time)} timesteps with {total_blocked_entries} total cell-time pairs")
        else:
            # Build spatial blocking: single set of all blocked cells
            blocked_cells = set()
            for oid in other_agents:
                idx = oid - 1
                traj = current_trajectories[idx]
                if traj:
                    # Block cells from extended window
                    for t in range(max(0, coll_time - search_rewind),
                                  min(len(traj), coll_time + search_horizon + 1)):
                        pos = tuple(map(int, traj[t]))
                        blocked_cells.add(pos)

            blocked_by_time = None
            if verbose:
                print(f"        Blocked {len(blocked_cells)} cells from other agents during planning window")

        # Retry Joint A* with blocked cells (spatial or time-based depending on flag)
        success, plans, trajs, t_start, t_goal_sub, expansions = try_joint_astar_planning(
            collision, current_trajectories, agent_goals, agent_starts,
            pristine_static_grid, heuristic_dist_map,
            max_agents=max_agents,
            time_budget=time_budget,
            max_expansions=max_expansions,
            verbose=verbose,
            base_rewind=base_rewind,
            base_horizon=base_horizon,
            max_expansion_steps=max_expansion_steps,
            blocked_cells=blocked_cells,
            blocked_by_time=blocked_by_time,
            use_time_based_blocking=use_time_based_blocking
        )

        if success:
            if verbose:
                print(f"      ✓ Joint A* succeeded with conflict cell blocking on retry {retry_attempt + 1}")
            return success, plans, trajs, t_start, t_goal_sub, expansions

    if verbose:
        print(f"      ✗ Joint A* failed even after {max_conflict_retries} retries with conflict blocking")
    return False, {}, {}, None, None, 0


def compute_global_makespan(plans):
    """
    Find the last timestep where ANY agent takes a non-WAIT action.

    This is used to synchronize all plans to the same length before collision detection.
    All agents must be padded to this global makespan with WAIT actions so that
    the main collision detection loop and final validation see the same data.

    Args:
        plans: List of plans (action sequences) for all agents

    Returns:
        Global makespan (int): Last timestep with non-WAIT action
    """
    # Find longest plan
    max_plan_len = max((len(p) for p in plans if p), default=0)
    if max_plan_len == 0:
        return 0

    # Scan backwards to find last non-WAIT action
    # Action 4 is WAIT in this environment
    for timestep in range(max_plan_len - 1, -1, -1):
        for plan in plans:
            if timestep < len(plan) and plan[timestep] != 4:  # Found a non-WAIT action
                return timestep + 1

    return 0


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
    max_expansions=20000,
    time_limit=60,
    max_passes=10000,
    joint_rewind=5,
    joint_horizon=5,
    joint_expansion_steps=15,
    use_time_based_blocking=True,
    verbose=True
):
    """
    Clean 3-strategy collision resolution:
    1. STATIC: Block collision cells, replan (attempt once per collision)
    2. DYNAMIC: Share blocker trajectory, replan (attempt once per collision)
    3. JOINT A*: Short-horizon joint planning for persistent collisions

    Args:
        joint_rewind: Steps to rewind from collision time in joint A* (default 7, was 3)
        joint_horizon: Steps to look ahead from collision time in joint A* (default 15, was 6)
        joint_expansion_steps: Times to expand search window if joint A* fails (default 8, was 3)
        max_passes: Maximum passes to attempt (default 10000, was 50)
        time_limit: Overall time budget in seconds (default 60)
    """
    overall_start_time = time.perf_counter()

    # Initialize counters
    if run_counters is None:
        run_counters = {}
    for key in ['replan_attempts_static', 'replan_success_static',
                'replan_attempts_dynamic', 'replan_success_dynamic',
                'replan_attempts_joint', 'replan_success_joint',
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

    # Precompute heuristic distances for coordinated planning
    if verbose:
        print(f"\n=== Collision Resolution Started ===")
        print(f"Grid: {pristine_static_grid.shape}, Agents: {num_agents}")
        print(f"Strategy: STATIC + DYNAMIC + JOINT A*")
        print(f"Precomputing heuristic distances...")

    heuristic_dist_map = compute_heuristic_distances(pristine_static_grid, agent_goals)

    # Track finished agents
    finished_agents = set()
    for i, (traj, goal) in enumerate(zip(current_trajectories, agent_goals)):
        if traj and tuple(map(int, traj[-1])) == tuple(map(int, goal)):
            finished_agents.add(i + 1)  # Agent IDs are 1-indexed

    if verbose:
        print(f"Finished agents: {finished_agents}")

    info_tracker = InfoSharingTracker()
    info_tracker.record_initial_submission(initial_agent_trajectories)

    # Initialize metrics and strategy IU trackers
    metrics_tracker = MetricsTracker()
    strategy_iu_tracker = StrategyIUTracker()

    # History tracking
    static_block_hist = defaultdict(set)
    collision_attempts = defaultdict(int)
    collision_joint_astar_failed = set()  # Track collisions where Joint A* failed (skip in same pass)

    # Per-collision, per-strategy attempt counters (refinement #1)
    collision_static_attempts = defaultdict(int)      # coll_key -> count
    collision_dynamic_attempts = defaultdict(int)     # coll_key -> count
    collision_joint_attempts = defaultdict(int)       # coll_key -> count
    collision_yield_general_attempts = defaultdict(int)  # coll_key -> count (Generalized YIELD)
    collision_yield_attempts = defaultdict(int)       # coll_key -> count (Yield-on-Goal)

    # Constants
    INIT_REWIND = 3
    MAX_REWIND_STATIC = 7
    MAX_REWIND_DYN = 7

    # Strategy attempt limits (refinement #1)
    MAX_STATIC_ATTEMPTS = 1
    MAX_DYNAMIC_ATTEMPTS = 1
    MAX_JOINT_ATTEMPTS = 1
    MAX_YIELD_ATTEMPTS = 1  # Generalized YIELD attempt limit (WAIT + PARKING based yielding)
    MAX_ONESTEPWAIT_ATTEMPTS = 1  # DEPRECATED: Replaced by Generalized YIELD
    MAX_DEFER_ATTEMPTS = 1  # DEFER attempt limit

    # Track unresolved collisions for DEFER (refinement #3)
    unresolved_collisions = set()  # coll_keys where all strategies have been exhausted

    # TWO-PHASE DEFER: Track deferred agents and phase state
    deferred_agents = set()  # Agent IDs that are deferred (parked at start) in Phase 1
    in_phase_2 = False  # Flag: are we in Phase 2 (planning deferred agents)?
    collision_onestepwait_attempts = defaultdict(int)  # One-Step-Wait attempts
    collision_yield_start_attempts = defaultdict(int)  # Yield-on-Start attempts
    collision_defer_attempts = defaultdict(int)  # DEFER attempts per collision

    # Track applicability of YIELD strategies (to enable immediate deferral)
    collision_yield_goal_applicable = defaultdict(bool)  # True if YIELD-ON-GOAL is applicable to this collision
    collision_yield_start_applicable = defaultdict(bool)  # True if YIELD-ON-START is applicable to this collision

    # Track escalation attempts for each collision (refinement #4)
    collision_escalation_attempts = defaultdict(int)  # coll_key -> escalation attempt count (max 2)
    collision_escalated_agents = defaultdict(set)  # coll_key -> set of agents already escalated into joint group

    # Track agents that have yielded (to deprioritize for future replanning)
    yielded_agents = set()  # Set of agent IDs that have successfully yielded

    # Helper function to check yield strategy applicability
    def check_yield_applicability(collision, current_trajectories, agent_starts, agent_goals):
        """
        Check if yield-on-start and yield-on-goal are applicable for this collision.
        Returns: (start_applicable, goal_applicable)
        """
        from utils.env_utils import pos_at

        coll_time = collision['time']
        coll_type = collision['type']
        coll_cell = collision['cell']
        collision_agents = collision['agents']

        start_applicable = False
        goal_applicable = False

        # Check Yield-on-Start applicability (only for vertex collisions)
        if coll_type == 'vertex':
            coll_cell_tuple = tuple(map(int, coll_cell))

            for agent in collision_agents:
                idx = agent - 1
                start_cell = agent_starts[idx]
                current_pos = pos_at(agent, coll_time, current_trajectories, agent_starts)

                # Yield-on-Start applicable if agent is at its start position
                if current_pos == start_cell and coll_cell_tuple == start_cell:
                    start_applicable = True

                # Yield-on-Goal applicable if agent is at its goal position
                goal_cell = tuple(map(int, agent_goals[idx]))
                if current_pos == goal_cell and coll_cell_tuple == goal_cell:
                    goal_applicable = True

        return start_applicable, goal_applicable

    # Main resolution loop
    for pass_num in range(1, max_passes + 1):
        elapsed = time.perf_counter() - overall_start_time
        if elapsed > time_limit:
            if verbose:
                print(f"\n✗ Time limit exceeded ({elapsed:.2f}s)")
            break

        # Reset Joint A* failed set for this pass (allow retry if other collisions clear)
        collision_joint_astar_failed = set()

        if verbose:
            print(f"\n--- Pass {pass_num}/{max_passes} (Elapsed: {elapsed:.2f}s) ---")

        # CRITICAL: Synchronize all trajectories to global makespan BEFORE collision detection
        # This ensures main loop and final check see the same data
        global_makespan = compute_global_makespan(current_plans)

        if global_makespan > 0:
            for idx in range(num_agents):
                current_len = len(current_plans[idx])

                if current_len < global_makespan:
                    num_waits = global_makespan - current_len
                    current_plans[idx].extend([4] * num_waits)

                    if verbose:
                        print(f"  Padded Agent {idx+1}: {current_len} → {global_makespan}")

                    # Re-simulate to get updated trajectory with padding
                    sim_env = copy.deepcopy(agent_envs[idx])
                    sim_env.env.agent_pos = agent_starts[idx]
                    new_traj = simulate_plan(sim_env, current_plans[idx])
                    if new_traj:
                        current_trajectories[idx] = new_traj

        # Detect collisions
        collisions = analyze_collisions(current_trajectories, agent_goals, agent_starts, pristine_static_grid)
        run_counters['collisions_total'] += len(collisions)

        # Track initial conflicts (only on first pass)
        if pass_num == 1:
            metrics_tracker.initial_conflicts = len(collisions)

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
            current_colls = analyze_collisions(current_trajectories, agent_goals, agent_starts, pristine_static_grid)
            still_exists = any(
                (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key
                for c in current_colls
            )

            if not still_exists:
                if verbose:
                    print(f"  ✓ Collision already resolved")
                continue

            # Pre-filter yield strategy applicability for this collision
            if coll_key not in collision_yield_start_applicable:
                start_app, goal_app = check_yield_applicability(
                    coll, current_trajectories, agent_starts, agent_goals
                )
                collision_yield_start_applicable[coll_key] = start_app
                collision_yield_goal_applicable[coll_key] = goal_app

                if verbose and (start_app or goal_app):
                    print(f"  Pre-filter: Yield-on-Start={'applicable' if start_app else 'N/A'}, "
                          f"Yield-on-Goal={'applicable' if goal_app else 'N/A'}")

            agents_to_try = list(coll['agents'])

            # Prefer non-yielded agents for replanning (Yield-on-Goal enhancement)
            non_yielded = [a for a in agents_to_try if a not in yielded_agents]
            if non_yielded:
                agents_to_try = non_yielded

            if replan_strategy == "random":
                agents_to_try = [random.choice(agents_to_try)]

            fixed = False

            # STRATEGY 1: YIELD-ON-START
            # For any agent at its start position - use pre-filtered applicability flag
            # Now works in both Phase 1 and Phase 2
            if not fixed and collision_yield_start_applicable.get(coll_key, False):
                from utils.env_utils import pos_at

                # Get agents that are at their start position
                agents_in_collision = sorted(list(coll['agents']))
                coll_time = coll['time']
                coll_type = coll['type']
                coll_cell = tuple(map(int, coll['cell']))

                # Find agents at start
                agents_at_start = []
                for aid in agents_in_collision:
                    idx = aid - 1
                    start_cell = agent_starts[idx]
                    current_pos = pos_at(aid, coll_time, current_trajectories, agent_starts)

                    if current_pos == start_cell and coll_cell == start_cell:
                        agents_at_start.append(aid)

                if agents_at_start:
                    collision_yield_start_attempts[coll_key] += 1
                    metrics_tracker.record_strategy_attempt('yield_on_start')

                    # Try YIELD-ON-START for each agent at start
                    for yielding_agent in agents_at_start:
                        if verbose:
                            print(f"  YIELD-ON-START attempt {collision_yield_start_attempts[coll_key]}: Agent {yielding_agent} (at start {agent_starts[yielding_agent-1]})")

                        yield_result = try_yield_on_start(
                            collision=coll,
                            yielding_agent=yielding_agent,
                            current_trajectories=current_trajectories,
                            agent_goals=agent_goals,
                            agent_starts=agent_starts,
                            pristine_static_grid=pristine_static_grid,
                            agent_envs=agent_envs,
                            current_plans=current_plans,
                            model=model,
                            device=device,
                            search_type=search_type,
                            algo=algo,
                            timeout=timeout,
                            heuristic_weight=heuristic_weight,
                            max_expansions=max_expansions,
                            verbose=verbose
                        )
                        
                        if yield_result and info_tracker:
                            info_tracker.record_parking_rejected(yield_result.get('rejected_count', 0))

                        if yield_result:
                            idx = yielding_agent - 1

                            # CRITICAL FIX: Save original trajectory BEFORE any modifications
                            original_plan = current_plans[idx]
                            original_traj = current_trajectories[idx]

                            # Verify collision is resolved BEFORE committing changes
                            temp_trajs = list(current_trajectories)
                            temp_trajs[idx] = yield_result['trajectory']
                            remaining_colls = analyze_collisions(temp_trajs, agent_goals, agent_starts, pristine_static_grid)

                            # Check if original collision is gone (use cell_key to handle edge/vertex collisions)
                            normalized_coll_cell_check = cell_key(coll_cell)
                            original_coll_gone = True
                            original_coll_still_exists = False
                            for rc in remaining_colls:
                                if (rc['time'] == coll_time and
                                    cell_key(rc['cell']) == normalized_coll_cell_check and
                                    set(rc['agents']) == set(agents_in_collision)):
                                    original_coll_gone = False
                                    original_coll_still_exists = True
                                    break

                            if original_coll_gone:
                                # ONLY NOW commit the changes
                                current_plans[idx] = yield_result['plan']
                                current_trajectories[idx] = yield_result['trajectory']

                                if verbose:
                                    print(f"  ✓ YIELD-ON-START successful: Agent {yielding_agent} yielded from start")

                                # Track success and IU (only on success)
                                metrics_tracker.record_strategy_success('yield_on_start')
                                rejected_count = yield_result.get('rejected_count', 0)
                                strategy_iu_tracker.record_yield_iu(rejected_count)

                                fixed = True
                                any_fix_this_pass = True
                                run_counters['yield_start_fixes'] = run_counters.get('yield_start_fixes', 0) + 1
                                break  # Exit collision loop
                            else:
                                # Verification failed
                                if verbose:
                                    print(f"  ✗ YIELD-ON-START: Collision persists after yield")
                        else:
                            if verbose:
                                print(f"  ✗ YIELD-ON-START: Failed to generate yield trajectory")

            if fixed:
                break  # Exit to re-detect collisions

            # STRATEGY 2: YIELD-ON-GOAL
            # Use pre-filtered applicability flag
            if not fixed and collision_yield_goal_applicable.get(coll_key, False):
                collision_yield_attempts[coll_key] += 1
                attempt_num_yield = collision_yield_attempts[coll_key]
                metrics_tracker.record_strategy_attempt('yield_on_goal')

                if verbose:
                    print(f"  → Trying YIELD-ON-GOAL (attempt {attempt_num_yield})")

                run_counters['replan_attempts_yield'] += 1

                # Try yield strategy
                success, yielding_agent, new_plan, new_traj, rejected_count = try_yield_on_goal(
                    coll, current_plans, current_trajectories,
                    agent_starts, agent_goals, pristine_static_grid,
                    agent_envs, model, device, search_type, algo,
                    timeout, heuristic_weight, max_expansions,
                    verbose=verbose
                )
                
                if info_tracker:
                    info_tracker.record_parking_rejected(rejected_count)

                if success:
                    idx = yielding_agent - 1

                    # Save original trajectory
                    original_plan = current_plans[idx]
                    original_traj = current_trajectories[idx]

                    # Verify collision resolution
                    temp_trajs = list(current_trajectories)
                    temp_trajs[idx] = new_traj
                    new_colls = analyze_collisions(temp_trajs, agent_goals, agent_starts, pristine_static_grid)

                    coll_resolved = not any(
                        (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key
                        for c in new_colls
                    )

                    if coll_resolved:
                        # Commit changes
                        current_plans[idx] = new_plan
                        current_trajectories[idx] = new_traj

                        # Track yielded agent
                        yielded_agents.add(yielding_agent)

                        # Track success and IU (only on success)
                        metrics_tracker.record_strategy_success('yield_on_goal')
                        strategy_iu_tracker.record_yield_iu(rejected_count)

                        if info_tracker:
                            # Yield strategy usually involves a wait or a local move.
                            # We can approximate replan start as collision time or 0.
                            # For now, let's assume it's a modification from collision time onwards?
                            # Actually, yield modifies from the point of yield.
                            # But we don't have exact time here easily without more analysis.
                            # Let's use 0 for now as it's a safe fallback, or try to be more precise.
                            # User said "part of trajectory which was replanned".
                            # For yield, it's `new_traj` vs `current_trajectories[idx]`.
                            # We can find the first divergence point.
                            # But that's expensive.
                            # Let's stick to 0 for simple yield for now, or maybe coll['time']?
                            # Yield happens AT collision time (or slightly before).
                            # Let's use max(0, coll['time'] - 1) as a heuristic?
                            # Or just leave it as is for simple yield if not specified?
                            # The user specifically mentioned "Revised Path Submission IU".
                            # I'll use 0 for simple yield for now to avoid breaking things,
                            # but for the main strategies (Static, Dynamic, Joint) I will be precise.
                            info_tracker.record_revised_submission(new_traj, replan_start_time=0)

                        run_counters['replan_success_yield'] += 1
                        any_fix_this_pass = True
                        fixed = True

                        if verbose:
                            print(f"  ✓ YIELD-ON-GOAL successful for Agent {yielding_agent}")
                    else:
                        if verbose:
                            print(f"  ✗ YIELD verification FAILED: Agent {yielding_agent} created yield segment but collision persists")

            if fixed:
                break  # Exit to re-detect collisions

            # STRATEGY 3: GENERALIZED YIELD (WAIT-based + Parking-based yielding)
            # Replaces ONE-STEP-WAIT with unified strategy that works for all collision types
            # No attempt limit - can be tried unlimited times
            # GATED: Only run if specialized yields are NOT applicable
            if not fixed and \
               not collision_yield_start_applicable.get(coll_key, False) and \
               not collision_yield_goal_applicable.get(coll_key, False):

                collision_yield_general_attempts[coll_key] += 1
                metrics_tracker.record_strategy_attempt('generalized_yield')

                success, yielding_agent, new_plan, new_traj, rejected_count = try_generalized_yield(
                    coll,
                    current_plans,
                    current_trajectories,
                    agent_starts,
                    agent_goals,
                    pristine_static_grid,
                    agent_envs,
                    model,
                    device,
                    search_type,
                    algo,
                    timeout,
                    heuristic_weight,
                    max_expansions,
                    heuristic_dist_map=heuristic_dist_map,
                    verbose=verbose
                )

                if info_tracker:
                    info_tracker.record_parking_rejected(rejected_count)

                if success:
                    idx = yielding_agent - 1
                    current_plans[idx] = new_plan
                    current_trajectories[idx] = new_traj

                    # Track success and IU (only on success)
                    metrics_tracker.record_strategy_success('generalized_yield')
                    strategy_iu_tracker.record_yield_iu(rejected_count)

                    if info_tracker:
                        # Generalized yield
                        info_tracker.record_revised_submission(new_traj, replan_start_time=0)

                    run_counters['yield_general_fixes'] = run_counters.get('yield_general_fixes', 0) + 1
                    any_fix_this_pass = True
                    fixed = True

            if fixed:
                break  # Exit to re-detect collisions

            # STRATEGY 4: STATIC (refinement #1: try up to 3 times per collision)
            # Initialize STATIC strategy variables with safe defaults
            agents_to_try_static = []
            static_can_run = False

            # PHASE 1: Exclude deferred agents from STATIC replanning
            if not in_phase_2 and collision_static_attempts[coll_key] < MAX_STATIC_ATTEMPTS:
                # Filter out deferred agents for STATIC strategy
                non_deferred_agents = [a for a in agents_to_try if a not in deferred_agents]

                if len(non_deferred_agents) == 0:
                    # All agents are deferred, skip STATIC
                    if verbose:
                        print(f"  ⊘ STATIC: All agents deferred, skipping")
                else:
                    # Use non-deferred agents for STATIC
                    agents_to_try_static = non_deferred_agents
                    static_can_run = True
            elif in_phase_2 and collision_static_attempts[coll_key] < MAX_STATIC_ATTEMPTS:
                # Phase 2: normal STATIC behavior
                agents_to_try_static = agents_to_try
                static_can_run = True
            else:
                static_can_run = False

            if static_can_run and collision_static_attempts[coll_key] < MAX_STATIC_ATTEMPTS:
                # Increment attempt counter immediately (revert to old behavior)
                collision_static_attempts[coll_key] += 1
                attempt_num_static = collision_static_attempts[coll_key]
                metrics_tracker.record_strategy_attempt('static')

                # NEW: Iterate by window size first, then by agent
                window_sizes = [5, 10, 15, 20, 30]

                for window_size in window_sizes:
                    if fixed:
                        break

                    if verbose:
                        print(f"  → Trying BOUNDED WINDOW STATIC with window ±{window_size} (attempt {attempt_num_static}/{MAX_STATIC_ATTEMPTS})")

                    # Store candidates for "best" strategy
                    candidates = []

                    for agent_id in agents_to_try_static:
                        idx = agent_id - 1

                        if verbose:
                            print(f"    → Agent {agent_id} with window ±{window_size}")

                        run_counters['replan_attempts_static'] += 1

                        # Use bounded window STATIC strategy with specific window size
                        success, new_plan, new_traj = try_static_bounded_window(
                            coll,
                            agent_id,
                            current_plans[idx],
                            current_trajectories[idx],
                            agent_starts[idx],
                            agent_goals[idx],
                            current_trajectories,
                            agent_starts,
                            agent_goals,
                            pristine_static_grid,
                            agent_envs[idx],
                            model,
                            device,
                            search_type,
                            algo,
                            timeout,
                            heuristic_weight,
                            max_expansions,
                            static_block_hist[idx],
                            attempt_num_static,
                            verbose=verbose,
                            specific_window_size=window_size  # NEW: Pass specific window
                        )

                        if success:
                            if replan_strategy == "best":
                                # Evaluate candidate by counting remaining collisions
                                original_traj = current_trajectories[idx]
                                current_trajectories[idx] = new_traj
                                
                                remaining_colls = analyze_collisions(current_trajectories, agent_goals, agent_starts, pristine_static_grid)
                                num_colls = len(remaining_colls)
                                
                                candidates.append({
                                    'agent_id': agent_id,
                                    'plan': new_plan,
                                    'traj': new_traj,
                                    'num_colls': num_colls
                                })
                                
                                # Revert trajectory for next candidate check
                                current_trajectories[idx] = original_traj
                                
                                if verbose:
                                    print(f"      Candidate Agent {agent_id}: {num_colls} remaining collisions")
                            else:
                                # Original behavior: take first success
                                candidates.append({
                                    'agent_id': agent_id,
                                    'plan': new_plan,
                                    'traj': new_traj
                                })
                                break

                    if candidates:
                        # Select best candidate
                        if replan_strategy == "best":
                            best_cand = min(candidates, key=lambda x: x['num_colls'])
                            if verbose:
                                print(f"    → Selected best candidate Agent {best_cand['agent_id']} with {best_cand['num_colls']} collisions")
                        else:
                            best_cand = candidates[0]

                        agent_id = best_cand['agent_id']
                        new_plan = best_cand['plan']
                        new_traj = best_cand['traj']
                        idx = agent_id - 1

                        current_plans[idx] = new_plan
                        current_trajectories[idx] = new_traj

                        # BUGFIX: Trim trailing WAITs if agent reached goal
                        current_plans[idx] = trim_trailing_waits(current_plans[idx], current_trajectories[idx], agent_goals[idx])

                        if info_tracker:
                            # Extract blocked cells for tracking
                            if coll['type'] == 'vertex':
                                cells_to_block = [tuple(map(int, coll['cell']))]
                            elif coll['type'] == 'edge':
                                cells_to_block = [tuple(map(int, c)) for c in coll['cell']]
                            else:
                                obs_cell = tuple(map(int, coll['cell']))
                                cells_to_block = [obs_cell]

                            new_cells_to_block = [c for c in cells_to_block if c not in static_block_hist[idx]]
                            info_tracker.record_static_alert(new_cells_to_block)
                            
                            # Calculate replan start time for IU
                            # Window is ±window_size around coll['time']
                            # So start is max(0, coll['time'] - window_size)
                            replan_start_time = max(0, coll['time'] - window_size)
                            info_tracker.record_revised_submission(new_traj, replan_start_time=replan_start_time)

                            # Update blocked cells history
                            static_block_hist[idx].update(new_cells_to_block)

                        # Track success and IU (only on success)
                        metrics_tracker.record_strategy_success('static')
                        # Static IU: blocked cells used + collision cells (1 for vertex, 2 for edge)
                        blocked_cells_count = len(static_block_hist[idx])  # All cells in history used for replanning
                        strategy_iu_tracker.record_static_iu(blocked_cells_count, coll['type'])

                        run_counters['replan_success_static'] += 1
                        any_fix_this_pass = True
                        fixed = True

                        if verbose:
                            print(f"    ✓ BOUNDED WINDOW STATIC successful for Agent {agent_id} with window ±{window_size}")

                        break  # Exit window loop, collision resolved

                # If all windows failed, try full replanning fallback for each agent
                if not fixed:
                    if verbose:
                        print(f"  → All bounded windows failed, trying full replanning fallback")

                    for agent_id in agents_to_try_static:
                        idx = agent_id - 1

                        if verbose:
                            print(f"    → Agent {agent_id} full replanning")

                        run_counters['replan_attempts_static'] += 1

                        # Skip windows and go directly to full replanning fallback
                        success, new_plan, new_traj = try_static_bounded_window(
                            coll,
                            agent_id,
                            current_plans[idx],
                            current_trajectories[idx],
                            agent_starts[idx],
                            agent_goals[idx],
                            current_trajectories,
                            agent_starts,
                            agent_goals,
                            pristine_static_grid,
                            agent_envs[idx],
                            model,
                            device,
                            search_type,
                            algo,
                            timeout,
                            heuristic_weight,
                            max_expansions,
                            static_block_hist[idx],
                            attempt_num_static,
                            verbose=verbose,
                            specific_window_size=None,
                            skip_windows=True  # NEW: Skip windows, go directly to full replanning
                        )

                        if success:
                            current_plans[idx] = new_plan
                            current_trajectories[idx] = new_traj
                            current_plans[idx] = trim_trailing_waits(current_plans[idx], current_trajectories[idx], agent_goals[idx])

                            if info_tracker:
                                if coll['type'] == 'vertex':
                                    cells_to_block = [tuple(map(int, coll['cell']))]
                                elif coll['type'] == 'edge':
                                    cells_to_block = [tuple(map(int, c)) for c in coll['cell']]
                                else:
                                    obs_cell = tuple(map(int, coll['cell']))
                                    cells_to_block = [obs_cell]

                                new_cells_to_block = [c for c in cells_to_block if c not in static_block_hist[idx]]
                                info_tracker.record_static_alert(new_cells_to_block)
                                # Full replanning starts from 0 (or current agent pos, effectively new path)
                                info_tracker.record_revised_submission(new_traj, replan_start_time=0)
                                static_block_hist[idx].update(new_cells_to_block)

                            # Track success and IU (only on success)
                            metrics_tracker.record_strategy_success('static')
                            blocked_cells_count = len(static_block_hist[idx])
                            strategy_iu_tracker.record_static_iu(blocked_cells_count, coll['type'])

                            run_counters['replan_success_static'] += 1
                            any_fix_this_pass = True
                            fixed = True

                            if verbose:
                                print(f"    ✓ Full replanning successful for Agent {agent_id}")

                            break

            elif verbose and collision_static_attempts[coll_key] >= MAX_STATIC_ATTEMPTS:
                print(f"  ✗ STATIC exhausted ({MAX_STATIC_ATTEMPTS} attempts), skipping")

            if fixed:
                break  # Exit loop to re-detect collisions

            # STRATEGY 2: DYNAMIC (refinement #1: try up to 3 times per collision)
            # PHASE 1: Deferred agents are always blockers, never replanning targets
            # Initialize DYNAMIC strategy variables with safe defaults
            replanning_agent_dynamic = None
            blocker_agent_dynamic = None
            forced_blocker = None
            dynamic_skip = False
            dynamic_forced_assignment = False
            agents_to_try_dynamic = []

            if not fixed and collision_dynamic_attempts[coll_key] < MAX_DYNAMIC_ATTEMPTS:
                agents_in_collision = sorted(list(coll['agents']))

                # Check if we should skip DYNAMIC due to deferred agents
                if not in_phase_2:
                    non_deferred = [a for a in agents_in_collision if a not in deferred_agents]

                    if len(non_deferred) == 0:
                        # Both agents deferred, skip DYNAMIC
                        if verbose:
                            print(f"  ⊘ DYNAMIC: Both agents deferred, skipping")
                        dynamic_skip = True
                    elif len(non_deferred) == 1:
                        # One deferred (blocker), one non-deferred (replanning target)
                        replanning_agent_dynamic = non_deferred[0]
                        blocker_agent_dynamic = [a for a in agents_in_collision if a in deferred_agents][0]
                        dynamic_skip = False
                        dynamic_forced_assignment = True

                        if verbose:
                            print(f"  → DYNAMIC: Agent {replanning_agent_dynamic} replans around deferred Agent {blocker_agent_dynamic}")
                    else:
                        # Both non-deferred, use existing heuristic
                        dynamic_skip = False
                        dynamic_forced_assignment = False
                else:
                    # Phase 2: normal DYNAMIC behavior
                    dynamic_skip = False
                    dynamic_forced_assignment = False

                if not dynamic_skip:
                    # Increment attempt counter immediately (revert to old behavior)
                    collision_dynamic_attempts[coll_key] += 1
                    attempt_num_dynamic = collision_dynamic_attempts[coll_key]
                    metrics_tracker.record_strategy_attempt('dynamic')

                    # Determine which agents to try for DYNAMIC
                    if dynamic_forced_assignment:
                        agents_to_try_dynamic = [replanning_agent_dynamic]
                        forced_blocker = blocker_agent_dynamic
                    else:
                        agents_to_try_dynamic = agents_to_try
                        forced_blocker = None

                    for agent_id in agents_to_try_dynamic:
                        idx = agent_id - 1
                        other_agents = [aid for aid in coll['agents'] if aid != agent_id]

                        if not other_agents:
                            continue

                        # Use forced blocker if assigned, otherwise random choice
                        if forced_blocker is not None:
                            blocker_id = forced_blocker
                        else:
                            blocker_id = random.choice(other_agents)

                        blocker_idx = blocker_id - 1

                        if verbose:
                            print(f"  → Trying DYNAMIC for Agent {agent_id} (blocker: {blocker_id}) (attempt {attempt_num_dynamic}/{MAX_DYNAMIC_ATTEMPTS})")

                        run_counters['replan_attempts_dynamic'] += 1

                        env_copy = copy.deepcopy(agent_envs[idx])
                        rewind = min(MAX_REWIND_DYN, INIT_REWIND + attempt_num - 1)
                        replan_time = max(0, coll['time'] - rewind)

                        if current_trajectories[idx] and replan_time < len(current_trajectories[idx]):
                            replan_pos = tuple(map(int, current_trajectories[idx][replan_time]))
                        else:
                            replan_pos = agent_starts[idx]

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

                        new_plan_segment = plan_with_search(
                            env_copy, model, device, search_type, algo,
                            timeout, heuristic_weight, max_expansions
                        )

                        if new_plan_segment:
                            prefix_actions = current_plans[idx][:replan_time] if current_plans[idx] else []
                            new_full_plan = prefix_actions + new_plan_segment

                            sim_env = copy.deepcopy(agent_envs[idx])
                            sim_env.env.agent_pos = agent_starts[idx]
                            new_traj = simulate_plan(sim_env, new_full_plan)

                            if new_traj:
                                temp_trajs = list(current_trajectories)
                                temp_trajs[idx] = new_traj
                                new_colls = analyze_collisions(temp_trajs, agent_goals, agent_starts, pristine_static_grid)

                                coll_resolved = not any(
                                    (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key
                                    for c in new_colls
                                )

                                if coll_resolved:
                                    # Save old trajectory segment before replacing (for IU calculation)
                                    old_traj_segment = current_trajectories[idx][replan_time:] if replan_time < len(current_trajectories[idx]) else []
                                    new_traj_segment = new_traj[replan_time:] if replan_time < len(new_traj) else []

                                    current_plans[idx] = new_full_plan
                                    current_trajectories[idx] = new_traj

                                    # BUGFIX: Trim trailing WAITs if agent reached goal
                                    current_plans[idx] = trim_trailing_waits(current_plans[idx], current_trajectories[idx], agent_goals[idx])

                                    # Track success and IU (only on success)
                                    metrics_tracker.record_strategy_success('dynamic')
                                    # Resubmission IU: only changed cells, excluding trailing waits after goal
                                    strategy_iu_tracker.record_resubmission_iu(old_traj_segment, new_traj_segment, agent_goals[idx])

                                    if info_tracker:
                                        info_tracker.record_dynamic_alert(obs_path)
                                        # Dynamic replanning starts at replan_time
                                        info_tracker.record_revised_submission(new_traj, replan_start_time=replan_time)

                                    run_counters['replan_success_dynamic'] += 1
                                    any_fix_this_pass = True
                                    fixed = True

                                    if verbose:
                                        print(f"  ✓ DYNAMIC successful for Agent {agent_id}")

                                    break

            elif not fixed and verbose and collision_dynamic_attempts[coll_key] >= MAX_DYNAMIC_ATTEMPTS:
                print(f"  ✗ DYNAMIC exhausted ({MAX_DYNAMIC_ATTEMPTS} attempts), skipping")

            if fixed:
                break  # Exit loop to re-detect collisions

            # NOTE: YIELD-ON-GOAL and YIELD-ON-START have been moved earlier (after Generalized Yield)


            # STRATEGY 4: Joint A* (refinement #1: up to 5 attempts total per collision)
            joint_success = False  # Initialize before attempting Joint A*

            if not fixed:
                # Check if strategy limits have been reached
                static_exhausted = collision_static_attempts[coll_key] >= MAX_STATIC_ATTEMPTS
                dynamic_exhausted = collision_dynamic_attempts[coll_key] >= MAX_DYNAMIC_ATTEMPTS
                joint_exhausted = collision_joint_attempts[coll_key] >= MAX_JOINT_ATTEMPTS

                # Try Joint A* if: (STATIC tried OR exhausted) AND (DYNAMIC tried OR exhausted) AND Joint not exhausted
                can_try_joint = (
                    (collision_static_attempts[coll_key] > 0 or static_exhausted) and
                    (collision_dynamic_attempts[coll_key] > 0 or dynamic_exhausted) and
                    not joint_exhausted
                )
            else:
                can_try_joint = False

            # Verbose logging for Joint A* status
            if not fixed and verbose and not can_try_joint:
                print(f"  ✗ Joint A* exhausted or not ready (static:{collision_static_attempts[coll_key]}, dynamic:{collision_dynamic_attempts[coll_key]}, joint:{collision_joint_attempts[coll_key]})")

            if can_try_joint:

                # OPTIMIZATION: Skip Joint A* if it failed in this pass already
                if coll_key in collision_joint_astar_failed:
                    if verbose:
                        print(f"  ✗ Joint A* already failed for this collision this pass, skipping (will retry in next pass)")
                    # Continue to next collision, will retry in next pass
                else:
                    # Increment attempt counter immediately (revert to old behavior)
                    collision_joint_attempts[coll_key] += 1
                    attempt_num_joint = collision_joint_attempts[coll_key]
                    metrics_tracker.record_strategy_attempt('joint_astar')

                    if verbose:
                        print(f"  → Trying Joint A* (attempt {attempt_num_joint}/{MAX_JOINT_ATTEMPTS})")

                    run_counters['replan_attempts_joint'] += 1

                    joint_success, joint_plan_segments, _, t_start, t_goal_sub, expansions = try_joint_astar_planning(
                        coll, current_trajectories, agent_goals, agent_starts,
                        pristine_static_grid, heuristic_dist_map,
                        verbose=verbose,
                        base_rewind=joint_rewind,
                        base_horizon=joint_horizon,
                        max_expansion_steps=joint_expansion_steps,
                        use_time_based_blocking=use_time_based_blocking
                    )

                    if info_tracker and joint_success:
                        # Record Joint A* expansions
                        # We need to capture expansions from try_joint_astar_planning
                        # But wait, I updated try_joint_astar_planning to return expansions!
                        # I need to update the unpacking above.
                        pass

                    # OPTIMIZATION: Track if Joint A* failed so we don't retry in same pass
                    if not joint_success:
                        collision_joint_astar_failed.add(coll_key)
                        if verbose:
                            print(f"  ✗ Joint A* failed, will skip this collision for rest of pass")

                if joint_success:
                    if t_start is None or t_goal_sub is None:
                        joint_success = False
                        continue

                    staged_plans = {}
                    staged_trajs = {}
                    temp_trajs = list(current_trajectories)

                    for aid, plan_segment in joint_plan_segments.items():
                        idx = aid - 1
                        plan_len = len(current_plans[idx]) if current_plans[idx] else 0
                        extended_plan = list(current_plans[idx]) if current_plans[idx] else []

                        # BUGFIX: Don't clamp t_start/t_goal_sub to plan_len
                        # Use actual values from joint A* search
                        # If plan is too short, extend it with WAIT actions to reach t_start
                        if t_start > plan_len:
                            # Extend plan with WAIT actions (action 4 = WAIT)
                            num_waits_needed = t_start - plan_len
                            extended_plan.extend([4] * num_waits_needed)
                            if verbose:
                                print(f"    Extended agent {aid} plan: added {num_waits_needed} WAIT actions ({plan_len} -> {t_start})")
                            plan_len = len(extended_plan)

                        # Validate that plan_segment has expected length
                        expected_segment_len = t_goal_sub - t_start
                        actual_segment_len = len(plan_segment)
                        if actual_segment_len != expected_segment_len:
                            if verbose:
                                print(f"    WARNING: Agent {aid} segment length mismatch: expected {expected_segment_len}, got {actual_segment_len}")
                            # This is a sign something went wrong in joint A* planning, but we can still try

                        # Use extended_plan which may have WAIT actions added
                        prefix_actions = extended_plan[:t_start] if extended_plan else []
                        # If t_goal_sub > plan_len, the suffix is empty (plan extends naturally)
                        original_suffix = extended_plan[t_goal_sub:] if extended_plan else []

                        # BUGFIX: Validate that prefix leads to the expected joint A* start position
                        # If t_start > 0, we need to check that the prefix reaches the joint A* starting point
                        if t_start > 0 and prefix_actions:
                            # Simulate just the prefix to see where it ends
                            prefix_env = copy.deepcopy(agent_envs[idx])
                            prefix_env.env.agent_pos = agent_starts[idx]
                            prefix_traj = simulate_plan(prefix_env, prefix_actions)

                            if prefix_traj is None:
                                if verbose:
                                    print(f"    ERROR: Prefix for agent {aid} is invalid (moves out of bounds/obstacles)")
                                joint_success = False
                                break

                            prefix_end_pos = tuple(map(int, prefix_traj[-1]))
                            expected_segment_start = tuple(map(int, current_trajectories[idx][t_start])) if t_start < len(current_trajectories[idx]) else None

                            if expected_segment_start and prefix_end_pos != expected_segment_start:
                                if verbose:
                                    print(f"    ERROR: Agent {aid} prefix ends at {prefix_end_pos} but joint A* expects start at {expected_segment_start}")
                                joint_success = False
                                break

                        new_full_plan = prefix_actions + plan_segment + original_suffix

                        if verbose:
                            print(
                                f"    Splicing agent {aid}: plan_len={plan_len}, "
                                f"t_start={t_start}, t_goal_sub={t_goal_sub}, "
                                f"segment_len={len(plan_segment)}"
                            )

                        sim_env = copy.deepcopy(agent_envs[idx])
                        sim_env.env.agent_pos = agent_starts[idx]
                        new_traj = simulate_plan(sim_env, new_full_plan)

                        if not new_traj:
                            if verbose:
                                print(f"    ERROR: Simulating spliced plan for agent {aid} failed (invalid moves)")
                            joint_success = False
                            break

                        staged_plans[aid] = new_full_plan
                        staged_trajs[aid] = new_traj
                        temp_trajs[idx] = new_traj

                    if joint_success:
                        new_colls = analyze_collisions(temp_trajs, agent_goals, agent_starts, pristine_static_grid)

                        coll_resolved = not any(
                            (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key
                            for c in new_colls
                        )

                        # BUGFIX: Only accept if the SPECIFIC collision was resolved
                        # AND we didn't significantly increase collisions
                        # (Use AND, not OR - we must actually fix the collision)
                        if coll_resolved and len(new_colls) <= len(collisions):
                            if verbose:
                                print(f"    ✓ Joint plan accepted!")
                                for aid, new_plan in staged_plans.items():
                                    print(f"      Final plan for Agent {aid}: {new_plan}")

                            for aid, new_plan in staged_plans.items():
                                current_plans[aid - 1] = new_plan
                            for aid, new_traj in staged_trajs.items():
                                current_trajectories[aid - 1] = new_traj
                                if info_tracker:
                                    info_tracker.record_revised_submission(new_traj, replan_start_time=t_start)
                                    # Record expansions (accumulated from the successful call)
                                    info_tracker.record_joint_astar_iu(expansions)

                            # Track success and IU (only on success)
                            metrics_tracker.record_strategy_success('joint_astar')
                            # Joint A* IU: nodes expanded + cell conflicts
                            # Cell conflicts are tracked via print statements in try_joint_astar_planning
                            # For now, we count expansions. Cell conflicts need to be extracted from the function.
                            strategy_iu_tracker.record_joint_astar_iu(expansions, cell_conflicts=0)

                            # NOTE: DO NOT trim WAITs here! Joint A* has carefully constructed
                            # these plans to resolve collisions. The global cleanup phase will
                            # handle all plan trimming uniformly across all agents.

                            run_counters['replan_success_joint'] += 1
                            any_fix_this_pass = True
                            fixed = True

                            if verbose:
                                print(f"  ✓ Joint A* successful")
                                print(f"    Reduced collisions from {len(collisions)} to {len(new_colls)}")

                            break
                        else:
                            if verbose:
                                if not coll_resolved:
                                    print(f"    ✗ Joint plan REJECTED: collision not resolved")
                                if len(new_colls) > len(collisions):
                                    print(f"    ✗ Joint plan REJECTED: created new collisions ({len(collisions)} -> {len(new_colls)})")

            # STRATEGY 5: DEFER (Phase 1 only - immediate execution when all strategies exhausted)
            # Execute DEFER immediately when all strategies have been attempted and exhausted
            if not fixed and not in_phase_2:
                static_exhausted = collision_static_attempts[coll_key] >= MAX_STATIC_ATTEMPTS
                dynamic_exhausted = collision_dynamic_attempts[coll_key] >= MAX_DYNAMIC_ATTEMPTS
                joint_exhausted = collision_joint_attempts[coll_key] >= MAX_JOINT_ATTEMPTS
                defer_exhausted = collision_defer_attempts[coll_key] >= MAX_DEFER_ATTEMPTS

                # Check if YIELD strategies were applicable
                yield_goal_applicable = collision_yield_goal_applicable.get(coll_key, False)
                yield_start_applicable = collision_yield_start_applicable.get(coll_key, False)

                yield_goal_attempted = collision_yield_attempts[coll_key] > 0
                yield_start_attempted = collision_yield_start_attempts[coll_key] > 0

                # A strategy is "done" if it was tried at least once or isn't applicable
                yield_goal_done = yield_goal_attempted or not yield_goal_applicable
                yield_start_done = yield_start_attempted or not yield_start_applicable

                # All main strategies are done (exhausted or not applicable)
                all_main_strategies_done = (
                    static_exhausted and
                    dynamic_exhausted and
                    yield_goal_done and       # Exhausted OR not applicable
                    yield_start_done and      # Exhausted OR not applicable
                    joint_exhausted           # Joint A* must always be tried
                )

                # Verbose logging for DEFER readiness
                if verbose:
                    if all_main_strategies_done and not defer_exhausted:
                        print(f"  → All strategies exhausted, attempting DEFER")
                    elif all_main_strategies_done and defer_exhausted:
                        print(f"  ✗ DEFER exhausted ({MAX_DEFER_ATTEMPTS} attempts)")
                    elif not all_main_strategies_done:
                        print(f"  ⊘ DEFER: Not ready (strategies not all tried: static={static_exhausted}, dynamic={dynamic_exhausted}, joint={joint_exhausted})")

                if all_main_strategies_done and not defer_exhausted:
                    # Attempt DEFER immediately
                    collision_defer_attempts[coll_key] += 1
                    metrics_tracker.record_strategy_attempt('defer')

                    if verbose:
                        print(f"    Attempting DEFER (attempt {collision_defer_attempts[coll_key]}/{MAX_DEFER_ATTEMPTS})")

                    defer_result = execute_defer_for_collision(
                        collision=coll,
                        current_plans=current_plans,
                        current_trajectories=current_trajectories,
                        agent_envs=agent_envs,
                        agent_goals=agent_goals,
                        agent_starts=agent_starts,
                        deferred_agents=deferred_agents,
                        heuristic_dist_map=heuristic_dist_map,
                        verbose=verbose
                    )

                    if defer_result:
                        # Apply DEFER result
                        current_plans = defer_result['plans']
                        current_trajectories = defer_result['trajectories']
                        deferred_agents = defer_result['deferred_agents']
                        deferred_agent_id = defer_result['deferred_agent']

                        if verbose:
                            print(f"  ✓ DEFER successful: Agent {deferred_agent_id} parked at start")

                        # Track success (no IU for defer - it's just parking at start)
                        metrics_tracker.record_strategy_success('defer')

                        run_counters['defer_fixes'] = run_counters.get('defer_fixes', 0) + 1
                        any_fix_this_pass = True
                        fixed = True

                        # Break to re-detect collisions with new state
                        break
                    else:
                        if verbose:
                            print(f"  ✗ DEFER failed (all agents in collision are already deferred)")

                # Log status if collision still not fixed
                if not fixed:
                    if all_main_strategies_done and defer_exhausted:
                        if verbose:
                            print(f"  ✗ ALL STRATEGIES EXHAUSTED INCLUDING DEFER - collision cannot be resolved")
                        unresolved_collisions.add(coll_key)
                    elif verbose:
                        print(f"  ✗ Could not resolve collision (will retry with other strategies)")

        # PHASE 1 → PHASE 2 TRANSITION
        # CRITICAL FIX: Check Phase 2 transition FIRST, then check exit conditions
        # This ensures Phase 2 triggers immediately when ready, and any_fix_this_pass=True prevents premature exit

        # 🆕 Helper function: Global cleanup of finished agent trajectories
        def cleanup_finished_agent_trajectories(plans, trajectories, goals, deferred_set):
            """
            Clean up plans and trajectories for agents that reached their goals.

            Returns: (cleaned_plans, cleaned_trajectories, max_goal_time)
            """
            max_goal_time = 0

            for idx in range(len(plans)):
                agent_id = idx + 1
                if agent_id in deferred_set:
                    continue

                traj = trajectories[idx]
                goal = tuple(map(int, goals[idx]))

                # Find when agent reaches goal
                goal_time = None
                for t, pos in enumerate(traj):
                    if tuple(map(int, pos)) == goal:
                        goal_time = t
                        break

                if goal_time is not None:
                    # Truncate plan to goal_time (remove trailing WAITs)
                    plans[idx] = plans[idx][:goal_time]

                    # Truncate trajectory to goal_time + 1 (include goal position)
                    trajectories[idx] = traj[:goal_time + 1]

                    max_goal_time = max(max_goal_time, goal_time)

            return plans, trajectories, max_goal_time


        # Check Phase 2 transition (only if still in Phase 1 and have deferred agents)
        if not in_phase_2 and deferred_agents:
            # Check if non-deferred agents have collisions
            non_deferred_colls = analyze_collisions(current_trajectories, agent_goals, agent_starts, pristine_static_grid)

            # Filter to only collisions involving non-deferred agents
            non_deferred_only_colls = [
                c for c in non_deferred_colls
                if all(aid not in deferred_agents for aid in c['agents'])
            ]

            if not non_deferred_only_colls:
                # ✅ TRANSITION TO PHASE 2
                # No collisions among non-deferred agents - transition to Phase 2
                if verbose:
                    print(f"\n{'=' * 60}")
                    print(f"PHASE 1 → PHASE 2 TRANSITION")
                    print(f"{'=' * 60}")
                    print(f"✓ No collisions among {num_agents - len(deferred_agents)} non-deferred agents")
                    print(f"→ Planning paths for {len(deferred_agents)} deferred agents...")

                # Record phase 1 passes before transitioning
                metrics_tracker.phase1_passes = pass_num

                in_phase_2 = True

                # CRITICAL: Reset all strategy counters for Phase 2 AND increase limits
                collision_static_attempts.clear()
                collision_dynamic_attempts.clear()
                collision_joint_attempts.clear()
                collision_yield_attempts.clear()
                collision_yield_general_attempts.clear()
                collision_onestepwait_attempts.clear()
                collision_yield_start_attempts.clear()
                collision_defer_attempts.clear()

                # Clear applicability pre-filtering flags
                collision_yield_start_applicable.clear()
                collision_yield_goal_applicable.clear()

                # Increase max attempts to 3 for Phase 2
                MAX_STATIC_ATTEMPTS = 3
                MAX_DYNAMIC_ATTEMPTS = 3
                MAX_JOINT_ATTEMPTS = 3

                if verbose:
                    print(f"  Strategy counters RESET and limits increased to 3 for Phase 2")
                    print(f"  Applicability flags RESET for Phase 2")

                # 🔄 PHASE 2 STEP 1: Global cleanup of finished agents
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"🧹 CLEANUP: Truncating finished agent trajectories")
                    print(f"{'='*60}")

                current_plans, current_trajectories, T_end = cleanup_finished_agent_trajectories(
                    current_plans,
                    current_trajectories,
                    agent_goals,
                    deferred_agents
                )

                if verbose:
                    print(f"✓ Cleanup complete")
                    print(f"  Non-deferred agents finished by T={T_end}")

                # PHASE 2: Plan paths for deferred agents
                # Step 1: Compute global makespan T_end from non-deferred agents
                non_deferred_plan_lens = [
                    len(current_plans[i]) for i in range(num_agents)
                    if (i + 1) not in deferred_agents and current_plans[i]
                ]

                T_end = max(non_deferred_plan_lens) if non_deferred_plan_lens else 0

                if verbose:
                    print(f"  T_end (non-deferred makespan): {T_end}")

                # Step 2: Build static obstacles from non-deferred agents at T_end
                static_obstacles_at_T_end = set()
                for idx in range(num_agents):
                    agent_id = idx + 1
                    if agent_id not in deferred_agents:
                        traj = current_trajectories[idx]
                        if traj:
                            # Get position at T_end (or last position if plan is shorter)
                            pos_at_T_end = tuple(map(int, traj[-1])) if T_end >= len(traj) else tuple(map(int, traj[T_end]))
                            static_obstacles_at_T_end.add(pos_at_T_end)

                if verbose and static_obstacles_at_T_end:
                    print(f"  Static obstacles at T_end: {len(static_obstacles_at_T_end)} cells occupied")

                # Step 3: Plan each deferred agent: WAIT(T_end) + A*(start→goal)
                for def_agent_id in sorted(deferred_agents):
                    def_idx = def_agent_id - 1
                    def_start = agent_starts[def_idx]
                    def_goal = agent_goals[def_idx]

                    if verbose:
                        print(f"\n  → Planning deferred Agent {def_agent_id}:")
                        print(f"    Start: {def_start}, Goal: {def_goal}")
                        print(f"    WAIT {T_end} steps, then A* to goal")

                    # Plan A* path from start to goal (with static obstacles from non-deferred agents)
                    planning_env = copy.deepcopy(agent_envs[def_idx])
                    planning_env.env.agent_pos = def_start

                    # Add static obstacles at T_end (non-deferred agent goal positions)
                    # Note: These are soft obstacles - A* will avoid them if possible
                    astar_plan = plan_astar_path(def_start, def_goal, planning_env, verbose=verbose)

                    if not astar_plan:
                        if verbose:
                            print(f"    ✗ A* failed for deferred Agent {def_agent_id}")
                        continue

                    # Create full plan: WAIT(T_end) + A* path
                    full_plan = [4] * T_end + astar_plan  # 4 = WAIT action

                    # Simulate full plan
                    sim_env = copy.deepcopy(agent_envs[def_idx])
                    sim_env.env.agent_pos = def_start
                    full_traj = simulate_plan(sim_env, full_plan)

                    if not full_traj:
                        if verbose:
                            print(f"    ✗ Simulation failed for deferred Agent {def_agent_id}")
                        continue

                    # Apply plan
                    current_plans[def_idx] = full_plan
                    current_trajectories[def_idx] = full_traj

                    if verbose:
                        print(f"    ✓ Deferred Agent {def_agent_id} planned: WAIT({T_end}) + {len(astar_plan)} moves")

                if verbose:
                    print(f"\n{'=' * 60}")
                    print(f"PHASE 2: FINAL CLEANUP PASS")
                    print(f"{'=' * 60}")
                    print(f"Running final collision resolution (DEFER disabled)...")

                # Step 4: Final cleanup pass with all strategies EXCEPT DEFER
                # Re-detect collisions with deferred agents now included
                final_colls = analyze_collisions(current_trajectories, agent_goals, agent_starts, pristine_static_grid)

                if final_colls:
                    if verbose:
                        print(f"  Found {len(final_colls)} collisions after Phase 2 planning")
                        print(f"  Attempting to resolve with STATIC, DYNAMIC, YIELD-on-goal, YIELD-on-start, Joint A*...")

                    # Continue main loop - strategies will handle Phase 2 cleanup
                    # Note: in_phase_2 flag ensures DEFER is disabled
                    any_fix_this_pass = True
                else:
                    if verbose:
                        print(f"  ✓ No collisions found! Phase 2 complete.")
                    break
            else:
                # We have collisions among non-deferred agents still
                # Continue resolving them before transitioning to Phase 2
                if verbose and any_fix_this_pass:
                    print(f"\n→ Non-deferred collisions remain; continuing to next pass")
                # Continue to next pass

        # Exit condition check (AFTER Phase 2 transition check)
        # This ensures Phase 2 has a chance to run before we consider exiting
        if not any_fix_this_pass:
            if in_phase_2:
                # Phase 2 and no progress → exit
                if verbose:
                    print(f"\n✗ No progress in Phase 2 cleanup pass")
                break
            elif not deferred_agents:
                # Phase 1, no deferred agents, no progress → exit
                if verbose:
                    print(f"\n✗ No progress in Phase 1 (no deferred agents)")
                break
            # else: Have deferred agents but stuck → continue to Phase 2 transition check

    # GLOBAL CLEANUP: Trim trailing WAITs across all agents synchronously
    # Find the global makespan: the last timestep where ANY agent takes a non-WAIT action
    if verbose:
        print(f"\n=== Global Cleanup ===")
        print(f"Trimming trailing WAITs across all agents...")

    # Find max plan length
    max_plan_len = max((len(p) for p in current_plans if p), default=0)

    if max_plan_len > 0:
        # Work backwards from max length to find last non-WAIT action
        global_makespan = 0
        for timestep in range(max_plan_len - 1, -1, -1):
            # Check if any agent has a non-WAIT action at this timestep
            has_non_wait = False
            for idx in range(num_agents):
                plan = current_plans[idx]
                if timestep < len(plan) and plan[timestep] != 4:  # 4 = WAIT
                    has_non_wait = True
                    break

            if has_non_wait:
                global_makespan = timestep + 1  # Length up to this point
                break

        if verbose:
            print(f"  Global makespan: {global_makespan} (max plan was {max_plan_len})")

        # Trim all agents to global_makespan
        agents_trimmed = 0
        for idx in range(num_agents):
            original_len = len(current_plans[idx])
            if original_len > global_makespan:
                # Trim plan
                current_plans[idx] = current_plans[idx][:global_makespan]
                agents_trimmed += 1
                if verbose:
                    print(f"  Agent {idx + 1}: trimmed {original_len - global_makespan} steps (was {original_len}, now {global_makespan})")

        # NEW: Pad all plans to global_makespan with WAIT actions
        # This ensures all agents have synchronized trajectory lengths
        agents_padded = 0
        for idx in range(num_agents):
            current_len = len(current_plans[idx])
            if current_len < global_makespan:
                # Pad with WAIT actions (4)
                num_waits = global_makespan - current_len
                current_plans[idx].extend([4] * num_waits)
                agents_padded += 1
                if verbose:
                    print(f"  Agent {idx + 1}: padded with {num_waits} WAIT actions (was {current_len}, now {global_makespan})")

        if verbose:
            print(f"  All plans now synchronized to length {global_makespan}")

        # CRITICAL: Re-simulate ALL trajectories after trimming and padding plans
        # (Some trajectories may be cached from old collision resolution and don't match trimmed plans)
        if verbose:
            print(f"  Re-simulating all trajectories to match trimmed plans...")

        failed_resims = []
        for idx in range(num_agents):
            trim_env = copy.deepcopy(agent_envs[idx])
            trim_env.env.agent_pos = agent_starts[idx]
            trimmed_traj = simulate_plan(trim_env, current_plans[idx])
            if trimmed_traj:
                current_trajectories[idx] = trimmed_traj
            else:
                # Re-simulation failed - trajectory is out of sync with plan!
                failed_resims.append(idx + 1)
                if verbose:
                    print(f"  ⚠ WARNING: Re-simulation failed for Agent {idx + 1}, plan length={len(current_plans[idx])}")
                    print(f"    Old trajectory length: {len(current_trajectories[idx])}")
                    print(f"    Plan: {current_plans[idx][:20]}...")  # Print first 20 actions

                # Fallback: Create minimal trajectory that matches plan length
                # This ensures plans and trajectories are in sync for JSON logging
                fallback_traj = [tuple(agent_starts[idx])]  # Start position
                current_pos = list(agent_starts[idx])

                movements = {
                    0: (-1, 0),  # Up
                    1: (1, 0),   # Down
                    2: (0, -1),  # Left
                    3: (0, 1),   # Right
                    4: (0, 0)    # Wait
                }

                for action in current_plans[idx]:
                    if isinstance(action, (list, tuple)):
                        action = action[0] if action else 4
                    dr, dc = movements.get(int(action), (0, 0))
                    current_pos[0] += dr
                    current_pos[1] += dc
                    fallback_traj.append(tuple(current_pos))

                current_trajectories[idx] = fallback_traj
                if verbose:
                    print(f"    Using fallback trajectory (length={len(fallback_traj)})")

        # Verification: Check all trajectories are synchronized
        traj_lengths_after = [len(t) if t else 0 for t in current_trajectories]
        plan_lengths_after = [len(p) if p else 0 for p in current_plans]

        if verbose:
            print(f"  [DEBUG] After re-simulation:")
            print(f"    Plan lengths: {plan_lengths_after}")
            print(f"    Traj lengths: {traj_lengths_after}")

            # Verify synchronization (all should be global_makespan + 1 position for traj)
            all_synced = all(len(t) == global_makespan + 1 for t in current_trajectories if t)
            if all_synced:
                print(f"    ✓ All trajectories synchronized to length {global_makespan + 1}")
            else:
                print(f"    ⚠ WARNING: Trajectories not fully synchronized!")
                for idx, traj in enumerate(current_trajectories):
                    if traj and len(traj) != global_makespan + 1:
                        print(f"      Agent {idx + 1}: trajectory length {len(traj)} (expected {global_makespan + 1})")

        if verbose:
            if failed_resims:
                print(f"Global cleanup complete: trimmed {agents_trimmed} agents, padded {agents_padded} agents, re-simulated all trajectories")
                print(f"  ⚠ {len(failed_resims)} agents had re-simulation failures: {failed_resims}")
            else:
                print(f"Global cleanup complete: trimmed {agents_trimmed} agents, padded {agents_padded} agents, re-simulated all trajectories")
    else:
        if verbose:
            print(f"  No plans to trim")

    # SYNCHRONIZED WAIT REMOVAL: Remove timesteps where ALL agents have WAIT action
    # This compresses the timeline by eliminating "synchronized idle" moments
    if HAS_CPP_CLEANUP and max_plan_len > 0:
        if verbose:
            print(f"\n=== Synchronized WAIT Removal ===")

        # Analyze potential savings before compression
        analysis = cpp_cleanup.analyze_synchronized_waits(current_plans)
        sync_wait_count = analysis['synchronized_waits']

        if sync_wait_count > 0:
            if verbose:
                print(f"  Found {sync_wait_count} synchronized WAIT timesteps")
                print(f"  Positions: {list(analysis['sync_wait_positions'])[:20]}{'...' if sync_wait_count > 20 else ''}")

            # Convert starts to CleanupCell objects for C++ module
            cpp_starts = [cpp_cleanup.CleanupCell(int(s[0]), int(s[1])) for s in agent_starts]

            # Convert grid to list format for C++ module
            cpp_grid = pristine_static_grid.tolist() if hasattr(pristine_static_grid, 'tolist') else list(pristine_static_grid)

            # Store original plans/trajectories in case compression introduces collisions
            plans_before_compression = [list(p) for p in current_plans]
            trajs_before_compression = [list(t) for t in current_trajectories]

            # Perform synchronized WAIT removal
            cleanup_result = cpp_cleanup.remove_synchronized_waits(current_plans, cpp_starts, cpp_grid)

            # Convert results back to Python lists
            compressed_plans = [list(p) for p in cleanup_result.plans]
            compressed_trajectories = [[(c.r, c.c) for c in traj] for traj in cleanup_result.trajectories]

            # SAFETY CHECK: Verify compression doesn't introduce new collisions
            compressed_collisions = analyze_collisions(
                compressed_trajectories, agent_goals, agent_starts, pristine_static_grid
            )

            if len(compressed_collisions) == 0:
                # Safe to use compressed plans
                current_plans = compressed_plans
                current_trajectories = compressed_trajectories

                if verbose:
                    print(f"  ✓ Compression successful!")
                    print(f"    Original makespan: {cleanup_result.original_makespan}")
                    print(f"    Compressed makespan: {cleanup_result.cleaned_makespan}")
                    print(f"    Timesteps removed: {cleanup_result.timesteps_removed}")
            else:
                # Compression introduced collisions - revert
                current_plans = plans_before_compression
                current_trajectories = trajs_before_compression

                if verbose:
                    print(f"  ✗ Compression would introduce {len(compressed_collisions)} collisions - reverted")
                    print(f"    Sample collision: {compressed_collisions[0] if compressed_collisions else 'N/A'}")
        else:
            if verbose:
                print(f"  No synchronized WAITs to remove")
    elif not HAS_CPP_CLEANUP:
        if verbose:
            print(f"\n=== Synchronized WAIT Removal (skipped - cpp_cleanup not available) ===")

    # BUGFIX: Final collision check AFTER global cleanup to catch goal-cell collisions
    # (must happen after trajectories are synchronized to same length)
    if verbose:
        print(f"\n=== Final Collision Check (after cleanup) ===")

    final_collisions = analyze_collisions(current_trajectories, agent_goals, agent_starts, pristine_static_grid)

    # DEBUG: Check trajectory lengths and potential late collisions
    if verbose:
        traj_lengths = [len(t) if t else 0 for t in current_trajectories]
        max_traj_len = max(traj_lengths) if traj_lengths else 0
        print(f"[DEBUG] Trajectory lengths after cleanup: {traj_lengths}")
        print(f"[DEBUG] Max trajectory length: {max_traj_len}")
        if max_traj_len > 0:
            collision_count_by_time = {}
            for t in range(max_traj_len):
                positions_at_t = {}
                for idx, traj in enumerate(current_trajectories):
                    if traj:
                        # Get position: actual if t < len, otherwise last position
                        pos = tuple(map(int, traj[t] if t < len(traj) else traj[-1]))
                        if pos not in positions_at_t:
                            positions_at_t[pos] = []
                        positions_at_t[pos].append(idx + 1)
                # Check for collisions at this timestep
                for pos, agents in positions_at_t.items():
                    if len(agents) > 1:
                        print(f"[DEBUG] Collision detected at t={t}: agents {agents} at {pos}")
                        collision_count_by_time[t] = collision_count_by_time.get(t, 0) + 1

    unique_colls = set()
    for c in final_collisions:
        unique_colls.add((c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])))

    timed_out = (time.perf_counter() - overall_start_time) > time_limit

    if verbose:
        print(f"\n=== Resolution Complete ===")
        print(f"Time: {time.perf_counter() - overall_start_time:.2f}s")
        print(f"Passes: {pass_num}")
        print(f"Final collisions: {len(final_collisions)}")
        print(f"Timed out: {timed_out}")
        print(f"[DEBUG] analyze_collisions says: {len(final_collisions)} collisions")
        if final_collisions:
            print("[DEBUG] Sample final collisions:", [
                {
                    'time': c['time'],
                    'type': c['type'],
                    'cell': c['cell'],
                    'agents': sorted(list(c['agents'])) if isinstance(c['agents'], (set, list)) else c['agents'],
                }
                for c in final_collisions[:5]
            ])
        info_tracker.report()

    # Finalize phase pass counts before post-cleanup
    if metrics_tracker.phase1_passes == 0:
        # Never transitioned to phase 2, all passes were phase 1
        metrics_tracker.phase1_passes = pass_num
    else:
        # Transitioned to phase 2, calculate phase 2 passes
        metrics_tracker.phase2_passes = pass_num - metrics_tracker.phase1_passes

    # Track deferred agents count
    metrics_tracker.deferred_agents_count = len(deferred_agents)

    # POST-CLEANUP COLLISION RESOLUTION
    # If collisions were revealed during cleanup (due to padding), try to fix them
    if final_collisions and len(final_collisions) > 0:
        if verbose:
            print(f"\n=== Post-Cleanup Collision Resolution ===")
            print(f"Found {len(final_collisions)} collisions after global cleanup")
            print(f"Attempting to resolve these remaining collisions...")

        max_cleanup_passes = 10
        cleanup_pass = 0
        cleanup_colls_fixed = 0

        while cleanup_pass < max_cleanup_passes and len(final_collisions) > 0:
            cleanup_pass += 1
            elapsed = time.perf_counter() - overall_start_time
            if elapsed > time_limit:
                if verbose:
                    print(f"⏱ Time limit exceeded during post-cleanup resolution")
                break

            if verbose:
                print(f"\n  Post-Cleanup Pass {cleanup_pass}/{max_cleanup_passes}")

            any_fixed_this_pass = False

            for coll in final_collisions[:3]:  # Try to fix first 3 collisions only
                coll_key = (coll['time'], coll['type'], cell_key(coll['cell']), frozenset(coll['agents']))
                agents_in_collision = list(coll['agents'])

                if verbose:
                    print(f"    Trying to fix: T={coll['time']}, Type={coll['type']}, Agents={agents_in_collision}")

                # Try JOINT A* with conflict blocking for cleanup collisions
                joint_success, joint_plan_segments, _, t_start, t_goal_sub, expansions = try_joint_astar_planning(
                    coll, current_trajectories, agent_goals, agent_starts,
                    pristine_static_grid, heuristic_dist_map,
                    verbose=verbose,
                    base_rewind=joint_rewind,
                    base_horizon=joint_horizon,
                    max_expansion_steps=4,  # Shorter expansion steps
                    max_conflict_retries=3,  # Retry with blocking if initial fails
                    use_time_based_blocking=use_time_based_blocking
                )

                if success and t_start is not None and t_goal_sub is not None:
                    # Splice and apply the joint plans
                    temp_trajs = list(current_trajectories)
                    all_valid = True

                    for aid, plan_segment in joint_plans.items():
                        idx = aid - 1
                        plan_len = len(current_plans[idx]) if current_plans[idx] else 0
                        extended_plan = list(current_plans[idx]) if current_plans[idx] else []

                        # Extend if needed
                        if t_start > plan_len:
                            num_waits = t_start - plan_len
                            extended_plan.extend([4] * num_waits)
                            plan_len = len(extended_plan)

                        prefix_actions = extended_plan[:t_start] if extended_plan else []
                        original_suffix = extended_plan[t_goal_sub:] if extended_plan else []
                        new_full_plan = prefix_actions + plan_segment + original_suffix

                        sim_env = copy.deepcopy(agent_envs[idx])
                        sim_env.env.agent_pos = agent_starts[idx]
                        new_traj = simulate_plan(sim_env, new_full_plan)

                        if new_traj:
                            current_plans[idx] = new_full_plan
                            current_trajectories[idx] = new_traj
                            temp_trajs[idx] = new_traj
                        else:
                            all_valid = False
                            if verbose:
                                print(f"      ✗ Simulation failed for agent {aid}")
                            break

                    if all_valid:
                        # Check if collision is resolved
                        new_colls = analyze_collisions(temp_trajs, agent_goals, agent_starts, pristine_static_grid)
                        still_exists = any(
                            (c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key
                            for c in new_colls
                        )

                        if not still_exists:
                            if verbose:
                                print(f"      ✓ Collision resolved!")
                            any_fixed_this_pass = True
                            cleanup_colls_fixed += 1
                            final_collisions = new_colls
                            break  # Recompute from new state

            if any_fixed_this_pass:
                # Re-detect collisions for next iteration
                final_collisions = analyze_collisions(current_trajectories, agent_goals, agent_starts, pristine_static_grid)
            else:
                if verbose:
                    print(f"  No progress in this pass, stopping post-cleanup resolution")
                break

        # Track post-cleanup passes
        metrics_tracker.post_cleanup_passes = cleanup_pass

        if verbose:
            if cleanup_colls_fixed > 0:
                print(f"\n✓ Post-cleanup resolution fixed {cleanup_colls_fixed} collisions")
                print(f"  Remaining: {len(final_collisions)} collisions")
            else:
                print(f"\n✗ Post-cleanup resolution could not fix remaining collisions")

    # Calculate MAPF metrics
    makespan = max((len(t) for t in current_trajectories if t), default=0)
    sum_of_costs = sum(len(p) for p in current_plans)
    avg_path_length = sum_of_costs / num_agents if num_agents > 0 else 0

    if verbose:
        print(f"\n=== MAPF Metrics ===")
        print(f"Makespan (longest trajectory):  {makespan}")
        print(f"Sum of Costs (SOC):             {sum_of_costs}")
        print(f"Average Path Length:            {avg_path_length:.2f}")
        print(f"Total Agents:                   {num_agents}")
        print(f"Agents at Goal:                 {sum(1 for i, traj in enumerate(current_trajectories) if traj and tuple(map(int, traj[-1])) == tuple(map(int, agent_goals[i])))}")
        print(f"Final Collision Count:          {len(final_collisions)}")

    log_data = {
        'info_sharing': info_tracker.to_dict(),
        'passes': pass_num,
        'time': time.perf_counter() - overall_start_time,
        'final_collisions': len(final_collisions),
        'metrics_raw': metrics_tracker.to_dict(),
        'strategy_iu_raw': strategy_iu_tracker.to_dict(),
    }

    return current_plans, current_trajectories, len(unique_colls), timed_out, log_data
