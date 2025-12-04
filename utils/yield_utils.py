"""
Helper utilities for smart deep yield strategy.

This module provides functions to find better yield positions for finished agents
by moving them further away from corridors and bottlenecks.
"""

import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from collections import deque


def find_deep_yield_cells(
    goal_pos: Tuple[int, int],
    static_grid: np.ndarray,
    distance_range: Tuple[int, int] = (2, 4),
    max_candidates: int = 20
) -> List[Tuple[Tuple[int, int], int]]:
    """
    Find cells at specified distance from goal using BFS.
    
    Args:
        goal_pos: Starting position (agent's goal)
        static_grid: Grid where -1 = obstacle, 0 = free
        distance_range: (min_dist, max_dist) for candidate cells
        max_candidates: Maximum number of cells to return
    
    Returns:
        List of (position, distance) tuples
    """
    candidates = []
    visited = set()
    queue = deque([(goal_pos, 0)])
    rows, cols = static_grid.shape
    
    while queue and len(candidates) < max_candidates:
        pos, dist = queue.popleft()
        
        if pos in visited or dist > distance_range[1]:
            continue
        visited.add(pos)
        
        # Add to candidates if in range
        if distance_range[0] <= dist <= distance_range[1]:
            candidates.append((pos, dist))
        
        # Explore neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if static_grid[nr, nc] == 0 and (nr, nc) not in visited:
                    queue.append(((nr, nc), dist + 1))
    
    return candidates


def count_free_neighbors(pos: Tuple[int, int], static_grid: np.ndarray) -> int:
    """Count number of free (non-obstacle) neighbors."""
    count = 0
    rows, cols = static_grid.shape
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols and static_grid[nr, nc] == 0:
            count += 1
    return count


def score_yield_cell(
    cell: Tuple[int, int],
    static_grid: np.ndarray,
    optimal_paths: Optional[List[Set[Tuple[int, int]]]] = None
) -> float:
    """
    Score a yield cell. Lower is better.
    
    Scoring criteria:
    - Dead ends (1 neighbor): -10 (excellent)
    - Corridors (2 neighbors): +5 (poor, blocks traffic)
    - Open areas (3+ neighbors): 0 (acceptable)
    - Path interference: +20 per agent path going through cell
    
    Args:
        cell: Position to score
        static_grid: Grid
        optimal_paths: List of sets containing optimal path cells for each agent
    
    Returns:
        Score (lower = better yield location)
    """
    score = 0.0
    
    # Connectivity-based scoring
    num_neighbors = count_free_neighbors(cell, static_grid)
    if num_neighbors == 1:
        score -= 10  # Dead end - ideal!
    elif num_neighbors == 2:
        score += 5  # Corridor - still blocks
    else:  # 3 or 4
        score += 0  # Open area - acceptable
    
    # Path interference penalty
    if optimal_paths:
        for path_set in optimal_paths:
            if cell in path_set:
                score += 20  # Heavy penalty for blocking optimalpaths
    
    return score


def select_best_yield_cell(
    candidates: List[Tuple[Tuple[int, int], int]],
    static_grid: np.ndarray,
    cell_degrees: Optional[np.ndarray] = None,
    flow_map: Optional[Dict[Tuple[int, int], int]] = None,
    goal_pos: Optional[Tuple[int, int]] = None,
    corridor_map: Optional[Dict] = None
) -> Optional[Tuple[int, int]]:
    """
    Select the best yield cell - AGGRESSIVE anti-choke-point scoring!
    
    CRITICAL: Must avoid choke points (high-traffic corridors/junctions)!
    
    Scoring (lower = better):
    - Distance: +0.3 * manhattan_dist
    - Dead end (deg 1): -5 (EXCELLENT)
    - Corridor (deg 2): +5 (BAD - bottleneck!)
    - Corridor endpoint: +8 additional
    - Junction (deg 3+): +8 (VERY BAD)
    - Traffic: +6 * (2^flow - 1) EXPONENTIAL!
    - Hard filter: flow >= 3 rejected
    
    Returns:
        Best cell position, or None if no candidates
    """
    
    def is_corridor_endpoint(cell):
        """Check if cell at corridor entrance/exit"""
        if not corridor_map:
            return False
        for cid, cinfo in corridor_map.items():
            if not cinfo.get("cells"):
                continue
            if cell == cinfo["cells"][0] or cell == cinfo["cells"][-1]:
                return True
        return False
    
    best_cell = None
    best_score = float('inf')
    
    for cell, dist in candidates:
        r, c = cell
        
        # Get traffic flow
        flow = flow_map.get(cell, 0) if flow_map else 0
        
        # HARD FILTER: Reject if >= 3 agents want this cell (moderated from 4)
        if flow >= 3:
            continue
        
        score = 0.0
        
        # Distance penalty (reduced weight)
        if goal_pos is not None:
            manhattan_dist = abs(cell[0] - goal_pos[0]) + abs(cell[1] - goal_pos[1])
            score += 0.3 * manhattan_dist  # Reduced from 0.5
        
        # Structure-based scoring (MUCH more aggressive)
        if cell_degrees is not None:
            if 0 <= r < cell_degrees.shape[0] and 0 <= c < cell_degrees.shape[1]:
                deg = cell_degrees[r, c]
                
                if deg == 1:  # Dead end - EXCELLENT!
                    score -= 5
                    
                elif deg == 2:  # Corridor - BAD!
                    score += 5  # Moderated from +8
                    
                    # Extra penalty for corridor endpoints
                    if is_corridor_endpoint(cell):
                        score += 8  # Moderated from +10
                        
                elif deg >= 3:  # Junction - VERY BAD!
                    score += 8  # Moderated from +15
        
        # Traffic-based scoring (EXPONENTIAL penalty - moderated)
        if flow > 0:
            score += 6 * (2 ** flow - 1)  # 1→6, 2→18, 3→42 (moderated from 10x)
        
        if score < best_score:
            best_score = score
            best_cell = cell
    
    return best_cell

