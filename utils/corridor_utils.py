"""
Corridor detection and management utilities for cooperative path planning.

This module provides:
- Corridor detection (maximal degree-2 chains)
- Cell classification (corridor vs junction vs dead-end vs open)
- Corridor ordering for cooperative behavior
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import deque, defaultdict


def detect_corridors(cell_degrees: np.ndarray, static_grid: np.ndarray) -> Dict[int, Dict]:
    """
    Detect corridor segments as maximal connected components of degree-2 cells.
    
    Algorithm:
    1. Find all degree-2 cells (corridor candidates)
    2. Use flood-fill to group connected degree-2 cells
    3. For each segment, identify entrances/exits and order cells
    
    Args:
        cell_degrees: 2D array from compute_cell_degrees
        static_grid: Base static grid (0=free, -1=obstacle)
    
    Returns:
        corridor_map: Dict[corridor_id] -> {
            "cells": [(r,c), ...],  # Ordered list along corridor
            "entrances": [(r,c), ...],  # Degree-1 or degree-3+ neighbors
            "exits": [(r,c), ...],
            "length": int
        }
    """
    rows, cols = cell_degrees.shape
    corridor_map = {}
    corridor_id = 0
    visited = np.zeros_like(cell_degrees, dtype=bool)
    
    # Find all degree-2 cells and group into corridors
    for r in range(rows):
        for c in range(cols):
            if cell_degrees[r, c] == 2 and not visited[r, c]:
                # Start new corridor segment
                corridor_cells = []
                queue = deque([(r, c)])
                visited[r, c] = True
                
                # Flood-fill to find all connected degree-2 cells
                while queue:
                    curr_r, curr_c = queue.popleft()
                    corridor_cells.append((curr_r, curr_c))
                    
                    # Check 4-connected neighbors
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if cell_degrees[nr, nc] == 2 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                
                # Only consider corridors with at least 2 cells
                if len(corridor_cells) >= 2:
                    # Order cells along corridor
                    ordered_cells = _order_corridor_cells(corridor_cells, cell_degrees)
                    
                    # Find entrances/exits
                    entrances, exits = _find_corridor_endpoints(
                        ordered_cells, cell_degrees, rows, cols
                    )
                    
                    corridor_map[corridor_id] = {
                        "cells": ordered_cells,
                        "entrances": entrances,
                        "exits": exits,
                        "length": len(ordered_cells)
                    }
                    corridor_id += 1
    
    return corridor_map


def _order_corridor_cells(cells: List[Tuple[int, int]], cell_degrees: np.ndarray) -> List[Tuple[int, int]]:
    """
    Order corridor cells sequentially from one end to the other.
    
    Args:
        cells: Unord unordered list of corridor cells
        cell_degrees: Cell degree grid
    
    Returns:
        Ordered list of cells
    """
    if len(cells) <= 1:
        return cells
    
    # Build adjacency for corridor cells
    cell_set = set(cells)
    adjacency = defaultdict(list)
    
    for r, c in cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_set:
                adjacency[(r, c)].append((nr, nc))
    
    # Find an endpoint (cell with only 1 neighbor in corridor)
    start = None
    for cell in cells:
        if len(adjacency[cell]) == 1:
            start = cell
            break
    
    # If no clear endpoint (loop?), just start from first cell
    if start is None:
        start = cells[0]
    
    # Traverse from start to build ordered list
    ordered = [start]
    visited = {start}
    current = start
    
    while len(ordered) < len(cells):
        # Find next unvisited neighbor
        next_cell = None
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                next_cell = neighbor
                break
        
        if next_cell is None:
            break  # Dead end or loop
        
        ordered.append(next_cell)
        visited.add(next_cell)
        current = next_cell
    
    return ordered


def _find_corridor_endpoints(
    ordered_cells: List[Tuple[int, int]],
    cell_degrees: np.ndarray,
    rows: int,
    cols: int
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Find entrance and exit cells of a corridor.
    
    Endpoints are cells with degree 1 (dead ends) or degree 3+ (junctions)
    that are neighbors of the corridor.
    
    Returns:
        (entrances, exits) - both are lists of (r,c) tuples
    """
    entrances = []
    exits = []
    
    # Check neighbors of first and last cells
    first_cell = ordered_cells[0]
    last_cell = ordered_cells[-1]
    
    corridor_set = set(ordered_cells)
    
    def get_endpoint_neighbors(cell):
        neighbors = []
        r, c = cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) not in corridor_set and cell_degrees[nr, nc] != 0:
                    neighbors.append((nr, nc))
        return neighbors
    
    # First cell's external neighbors are entrances
    entrances = get_endpoint_neighbors(first_cell)
    
    # Last cell's external neighbors are exits
    exits = get_endpoint_neighbors(last_cell)
    
    return entrances, exits


def classify_cell(
    cell: Tuple[int, int],
    cell_degrees: np.ndarray,
    corridor_map: Dict[int, Dict]
) -> Dict[str, any]:
    """
    Classify a cell as corridor/junction/dead_end/open.
    
    Args:
        cell: (r, c) position
        cell_degrees: Cell degree grid
        corridor_map: Result from detect_corridors
    
    Returns:
        {
            "type": "corridor" | "junction" | "dead_end" | "open",
            "corridor_id": int (if type=="corridor"),
            "index_in_corridor": int (if type=="corridor")
        }
    """
    r, c = cell
    degree = cell_degrees[r, c]
    
    # Check if cell is in any corridor
    for corridor_id, corridor_info in corridor_map.items():
        if cell in corridor_info["cells"]:
            index = corridor_info["cells"].index(cell)
            return {
                "type": "corridor",
                "corridor_id": corridor_id,
                "index_in_corridor": index
            }
    
    # Not in corridor, classify by degree
    if degree == 0:
        return {"type": "obstacle"}
    elif degree == 1:
        return {"type": "dead_end"}
    elif degree >= 3:
        return {"type": "junction"}
    else:  # degree == 2 but not in detected corridor (isolated)
        return {"type": "open"}


def assign_corridor_order(
    corridor_id: int,
    corridor_info: Dict,
    blocked_cells_per_agent: Dict[int, Set[Tuple[int, int]]],
    current_trajectories: List[List[Tuple[int, int]]],
    agent_goals: List[Tuple[int, int]]
) -> List[int]:
    """
    Assign priority ordering for agents wanting to use a corridor.
    
    Ordering rule:
    1. Group by direction (which entrance→exit they use)
    2. Within group, sort by distance to entrance or agent ID
    
    Args:
        corridor_id: ID of corridor
        corridor_info: Corridor metadata
        blocked_cells_per_agent: Dict[agent_id] -> desired cells
        current_trajectories: Current trajectories per agent
        agent_goals: Goal positions per agent
    
    Returns:
        Ordered list of agent IDs [aid1, aid2, ...]
    """
    corridor_cells = set(corridor_info["cells"])
    ordered_cells = corridor_info["cells"]
    
    # Find agents whose desired paths use this corridor
    agents_wanting_corridor = []
    for aid, desired_cells in blocked_cells_per_agent.items():
        if corridor_cells & desired_cells:  # Intersection
            agents_wanting_corridor.append(aid)
    
    if not agents_wanting_corridor:
        return []
    
    # Classify agents by direction
    direction_groups = {"forward": [], "backward": []}
    
    for aid in agents_wanting_corridor:
        # Get agent's corridor cells in order of their desired path
        agent_corridor_cells = [
            cell for cell in blocked_cells_per_agent[aid]
            if cell in corridor_cells
        ]
        
        if not agent_corridor_cells:
            continue
        
        # Determine direction by comparing first/last corridor cell indices
        first_cell = agent_corridor_cells[0]
        last_cell = agent_corridor_cells[-1]
        
        try:
            first_idx = ordered_cells.index(first_cell)
            last_idx = ordered_cells.index(last_cell)
            
            if first_idx < last_idx:
                direction_groups["forward"].append((aid, first_idx))
            else:
                direction_groups["backward"].append((aid, first_idx))
        except ValueError:
            # Cell not in ordered list, skip
            continue
    
    # Sort within groups by entrance distance
    direction_groups["forward"].sort(key=lambda x: x[1])  # Sort by first_idx
    direction_groups["backward"].sort(key=lambda x: -x[1])  # Sort by first_idx descending
    
    # Combine: forward first, then backward
    ordered_agents = [aid for aid, _ in direction_groups["forward"]]
    ordered_agents += [aid for aid, _ in direction_groups["backward"]]
    
    return ordered_agents
