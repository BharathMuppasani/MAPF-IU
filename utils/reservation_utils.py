"""
Reservation table utilities for time-aware path planning and yield decisions.

This module provides:
- Reservation table construction from trajectories
- Temporal occupancy analysis for cells
- Blocker ETA estimation
- Yield cell scoring with time-awareness
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict


def build_reservation_table(
    trajectories: List[List[Tuple[int, int]]],
    horizon: int = 50
) -> Dict[int, Dict[Tuple[int, int], Set[int]]]:
    """
    Build reservation table: time -> cell -> set of agent IDs.
    
    Args:
        trajectories: List of trajectories per agent
        horizon: Maximum time steps to consider
    
    Returns:
        reservation_table: Dict[time] -> Dict[cell] -> Set[agent_ids]
    """
    reservation_table = defaultdict(lambda: defaultdict(set))
    
    for agent_idx, traj in enumerate(trajectories):
        agent_id = agent_idx + 1
        
        for t in range(min(horizon, len(traj))):
            cell = tuple(map(int, traj[t]))
            reservation_table[t][cell].add(agent_id)
        
        # If trajectory ends before horizon, agent stays at last position
        if len(traj) < horizon:
            last_cell = tuple(map(int, traj[-1]))
            for t in range(len(traj), horizon):
                reservation_table[t][last_cell].add(agent_id)
    
    return dict(reservation_table)


def compute_cell_occupancy_score(
    cell: Tuple[int, int],
    reservation_table: Dict[int, Dict[Tuple[int, int], Set[int]]],
    ignore_agent: Optional[int] = None,
    time_window: Tuple[int, int] = (0, 50)
) -> float:
    """
    Score cell based on how often it's occupied over time window.
    Lower score = less occupied = better for yielding.
    
    Args:
        cell: Position to score
        reservation_table: Time -> cell -> agents mapping
        ignore_agent: Agent ID to ignore (the yielding agent)
        time_window: (start_time, end_time) to analyze
    
    Returns:
        Occupancy score (lower = better)
    """
    total_occupancy = 0
    time_steps = 0
    
    for t in range(time_window[0], time_window[1]):
        if t in reservation_table and cell in reservation_table[t]:
            agents_here = reservation_table[t][cell]
            
            # Filter out ignored agent
            if ignore_agent:
                agents_here = {a for a in agents_here if a != ignore_agent}
            
            total_occupancy += len(agents_here)
        time_steps += 1
    
    # Average occupancy over time window
    return total_occupancy / max(time_steps, 1)


def find_best_time_to_occupy(
    cell: Tuple[int, int],
    reservation_table: Dict[int, Dict[Tuple[int, int], Set[int]]],
    ignore_agent: Optional[int] = None,
    horizon: int = 50
) -> Tuple[int, int]:
    """
    Find time interval when cell is least occupied.
    
    Returns:
        (best_start_time, best_duration) - when cell is free longest
    """
    best_start = 0
    best_duration = 0
    current_start = None
    current_duration = 0
    
    for t in range(horizon):
        occupied = False
        
        if t in reservation_table and cell in reservation_table[t]:
            agents_here = reservation_table[t][cell]
            if ignore_agent:
                agents_here = {a for a in agents_here if a != ignore_agent}
            
            if agents_here:
                occupied = True
        
        if not occupied:
            if current_start is None:
                current_start = t
            current_duration += 1
        else:
            # End of free period
            if current_duration > best_duration:
                best_duration = current_duration
                best_start = current_start if current_start is not None else 0
            current_start = None
            current_duration = 0
    
    # Check final period
    if current_duration > best_duration:
        best_duration = current_duration
        best_start = current_start if current_start is not None else 0
    
    return best_start, best_duration


def estimate_blocker_eta(
    blocker_id: int,
    blocking_region: Set[Tuple[int, int]],
    trajectories: List[List[Tuple[int, int]]],
    current_time: int = 0
) -> int:
    """
    Estimate when blocker will clear the blocking region.
    
    Args:
        blocker_id: Agent ID (1-indexed)
        blocking_region: Set of cells that need to be clear
        trajectories: Current trajectories
        current_time: Current time step
    
    Returns:
        Number of time steps until region is clear (0 if already clear)
    """
    blocker_idx = blocker_id - 1
    
    if blocker_idx >= len(trajectories) or not trajectories[blocker_idx]:
        return 0
    
    blocker_traj = trajectories[blocker_idx]
    
    # Find when blocker leaves all blocking cells
    for t in range(current_time, len(blocker_traj)):
        cell = tuple(map(int, blocker_traj[t]))
        if cell not in blocking_region:
            return t - current_time
    
    # Check if blocker's final position is in region
    final_cell = tuple(map(int, blocker_traj[-1]))
    if final_cell in blocking_region:
        # Blocker will stay in region indefinitely
        return 999  # Large value
    
    return 0


def select_yield_cell_with_reservation(
    candidates: List[Tuple[Tuple[int, int], int]],
    reservation_table: Dict[int, Dict[Tuple[int, int], Set[int]]],
    yielding_agent_id: int,
    cell_degrees: np.ndarray,
    corridor_map: Dict,
    goal_pos: Tuple[int, int],
    current_time: int = 0,
    lookahead: int = 20
) -> Optional[Tuple[int, int]]:
    """
    Select best yield cell using reservation table for time-aware scoring.
    
    Scoring (lower = better):
    - Structure penalties (dead end bonus, corridor/junction penalties)
    - Temporal occupancy: heavily penalize if occupied in next 20 steps
    - Spatial: prefer further from high-traffic areas
    
    Returns:
        Best cell, or None if no good candidates
    """
    if not candidates:
        return None
    
    best_cell = None
    best_score = float('inf')
    
    for cell, dist in candidates:
        r, c = cell
        
        # Compute occupancy in lookahead window
        occupancy_score = compute_cell_occupancy_score(
            cell, reservation_table, ignore_agent=yielding_agent_id,
            time_window=(current_time, current_time + lookahead)
        )
        
        # HARD FILTER: Skip if heavily occupied (avg > 0.5 agents per timestep)
        if occupancy_score > 0.5:
            continue
        
        score = 0.0
        
        # Distance component (small weight)
        manhattan_dist = abs(cell[0] - goal_pos[0]) + abs(cell[1] - goal_pos[1])
        score += 0.2 * manhattan_dist
        
        # Structure component
        if 0 <= r < cell_degrees.shape[0] and 0 <= c < cell_degrees.shape[1]:
            deg = cell_degrees[r, c]
            
            if deg == 1:  # Dead end
                score -= 8  # Strong bonus
            elif deg == 2:  # Corridor
                score += 3
            elif deg >= 3:  # Junction
                score += 6
        
        # Temporal occupancy (MOST IMPORTANT)
        # Exponential penalty based on occupancy
        score += 50 * (2 ** (occupancy_score * 10) - 1)
        
        if score < best_score:
            best_score = score
            best_cell = cell
    
    return best_cell
