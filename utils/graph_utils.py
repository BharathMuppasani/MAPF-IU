"""
Graph-based utilities for traffic-aware coordination and deadlock detection.

This module provides:
- Static structural analysis (cell degrees)
- Blocking graph construction
- Cycle detection
- Pivot agent selection
- Traffic flow maps
"""

import numpy as np
import copy
from typing import List, Tuple, Set, Dict, Optional
from collections import defaultdict, deque


def compute_cell_degrees(static_grid: np.ndarray) -> np.ndarray:
    """
    Precompute the degree (number of free neighbors) for each cell.
    
    Args:
        static_grid: 2D array where 0 = free, -1 = obstacle
    
    Returns:
        cell_degrees: 2D array where:
            - 0 = obstacle
            - 1 = dead end (good for yielding)
            - 2 = corridor (still blocks traffic)
            - 3+ = junction/open area (more flexible)
    """
    rows, cols = static_grid.shape
    cell_degrees = np.zeros_like(static_grid, dtype=np.int32)
    
    for r in range(rows):
        for c in range(cols):
            if static_grid[r, c] == -1:
                cell_degrees[r, c] = 0  # Obstacle
                continue
            
            # Count free 4-connected neighbors
            degree = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if static_grid[nr, nc] == 0:  # Free cell
                        degree += 1
            
            cell_degrees[r, c] = degree
    
    return cell_degrees


def build_blocking_graph(
    current_trajectories: List[List[Tuple[int, int]]],
    agent_goals: List[Tuple[int, int]],
    static_grid: np.ndarray,
    agent_envs: List,
    model,
    device,
    search_type: str,
    algo: str,
    horizon: int = 10,
    max_expansions: int = 500,
    timeout: float = 2.0
) -> Tuple[List[Tuple[int, int]], Dict[int, Set[Tuple[int, int]]]]:
    """
    Build blocking graph showing which agents block which others' optimal paths.
    
    Args:
        current_trajectories: List of trajectories (list of (r,c) tuples) per agent
        agent_goals: List of goal positions per agent
        static_grid: Base static grid (0 = free, -1 = obstacle)
        agent_envs: List of per-agent environments
        model, device, search_type, algo: Planning infrastructure
        horizon: Time window to check for blocking
        max_expansions, timeout: Planning constraints
    
    Returns:
        blocking_edges: List of (blocker_id, blocked_id) pairs
        blocked_cells_per_agent: Dict mapping agent_id -> set of desired cells
    """
    from utils.search_utils import plan_with_search
    from utils.env_utils import simulate_plan
    
    num_agents = len(current_trajectories)
    blocked_cells_per_agent = {}
    
    # Step 1: Compute optimal static path for each agent
    for aid in range(num_agents):
        # Get current position (last position in trajectory)
        if not current_trajectories[aid]:
            blocked_cells_per_agent[aid] = set()
            continue
        
        current_pos = tuple(map(int, current_trajectories[aid][-1]))
        goal_pos = tuple(map(int, agent_goals[aid]))
        
        # Skip if already at goal
        if current_pos == goal_pos:
            blocked_cells_per_agent[aid] = {goal_pos}
            continue
        
        # Plan optimal path to goal
        env_copy = copy.deepcopy(agent_envs[aid])
        env_copy.env.grid = static_grid.copy()
        env_copy.env.agent_pos = current_pos
        env_copy.env.goal_pos = goal_pos
        
        optimal_plan = plan_with_search(
            env_copy, model, device, search_type, algo,
            timeout=timeout, heuristic_weight=1.0, max_expansions=max_expansions
        )
        
        if optimal_plan:
            # Simulate to get trajectory
            sim_env = copy.deepcopy(env_copy)
            optimal_traj = simulate_plan(sim_env, optimal_plan)
            # Convert to set of cells
            blocked_cells_per_agent[aid] = set(tuple(map(int, pos)) for pos in optimal_traj)
        else:
            # No path found, just mark goal
            blocked_cells_per_agent[aid] = {goal_pos}
    
    # Step 2: Build blocking edges
    blocking_edges = []
    
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            
            # Check if agent i blocks agent j
            # Look at agent i's trajectory for up to horizon steps
            traj_i = current_trajectories[i]
            if not traj_i:
                continue
            
            # Check cells in trajectory
            for t in range(min(horizon, len(traj_i))):
                cell_i = tuple(map(int, traj_i[t]))
                
                # Does this cell appear in j's desired path?
                if cell_i in blocked_cells_per_agent.get(j, set()):
                    # Agent i blocks agent j
                    blocking_edges.append((i, j))
                    break  # One blocking instance is enough
    
    return blocking_edges, blocked_cells_per_agent


def find_cycles(blocking_edges: List[Tuple[int, int]], num_agents: int) -> List[List[int]]:
    """
    Find all cycles in the blocking graph using DFS.
    
    Args:
        blocking_edges: List of (blocker_id, blocked_id) tuples
        num_agents: Total number of agents
    
    Returns:
        cycles: List of cycles, each cycle is a list of agent IDs
    """
    # Build adjacency list
    graph = defaultdict(list)
    for i, j in blocking_edges:
        graph[i].append(j)
    
    # DFS-based cycle detection
    visited = set()
    rec_stack = set()  # Recursion stack for current DFS path
    path = []  # Current path
    cycles = []
    
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found a cycle! Extract it
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:]
                cycles.append(cycle[:])  # Copy the cycle
        
        path.pop()
        rec_stack.remove(node)
    
    # Run DFS from each unvisited node
    for agent_id in range(num_agents):
        if agent_id not in visited and agent_id in graph:
            dfs(agent_id)
    
    return cycles


def choose_pivot_agent(
    cycle: List[int],
    finished_agents: Set[int],
    cell_degrees: np.ndarray,
    current_trajectories: List[List[Tuple[int, int]]]
) -> int:
    """
    Select the best pivot agent from a cycle to perform deep yield.
    
    Scoring heuristic (lower = better pivot):
    - Finished agents: -5 (prefer moving them)
    - In corridor (degree 2): -2 (easier to clear)
    - In junction (degree 3+): +1 (harder to clear)
    
    Args:
        cycle: List of agent IDs in the cycle
        finished_agents: Set of agent IDs already at goal
        cell_degrees: 2D array from compute_cell_degrees
        current_trajectories: List of trajectories per agent
    
    Returns:
        pivot_agent_id: Agent ID that should yield
    """
    best_agent = cycle[0]
    best_score = float('inf')
    
    for agent_id in cycle:
        score = 0.0
        
        # Prefer finished agents
        if agent_id in finished_agents:
            score -= 5
        
        # Get agent's current position
        if current_trajectories[agent_id]:
            pos = tuple(map(int, current_trajectories[agent_id][-1]))
            r, c = pos
            
            # Check cell degree
            if 0 <= r < cell_degrees.shape[0] and 0 <= c < cell_degrees.shape[1]:
                deg = cell_degrees[r, c]
                
                if deg == 2:  # Corridor
                    score -= 2  # Easier to yield from corridor
                elif deg >= 3:  # Junction/open
                    score += 1  # Harder to yield from junction
        
        if score < best_score:
            best_score = score
            best_agent = agent_id
    
    return best_agent


def compute_flow_map(
    blocked_cells_per_agent: Dict[int, Set[Tuple[int, int]]],
    ignore_agent: Optional[int] = None
) -> Dict[Tuple[int, int], int]:
    """
    Compute traffic flow map: how many agents want each cell.
    
    Args:
        blocked_cells_per_agent: Dict mapping agent_id -> set of desired cells
        ignore_agent: Optional agent ID to exclude from flow computation
    
    Returns:
        flow_map: Dict mapping (r, c) -> count of agents wanting that cell
    """
    flow_map = defaultdict(int)
    
    for agent_id, cells in blocked_cells_per_agent.items():
        if ignore_agent is not None and agent_id == ignore_agent:
            continue
        
        for cell in cells:
            flow_map[cell] += 1
    
    return dict(flow_map)
