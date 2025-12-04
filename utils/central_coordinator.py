"""
Central Coordinator - Global information manager for MAPF collision resolution.

Maintains:
- Cell occupancy heatmap (frequency of agent visits)
- Safe corner identification
- Agent priority calculation
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict


class CentralCoordinator:
    """
    Central coordinator with global knowledge for intelligent decision-making.
    """
    
    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape
        self.cell_heatmap = np.zeros(grid_shape, dtype=np.int32)
        self.safe_corners = None
        
    def update_heatmap(self, trajectories: List[List[Tuple[int, int]]]):
        """Update cell occupancy heatmap from current trajectories."""
        self.cell_heatmap.fill(0)
        
        for traj in trajectories:
            for pos in traj:
                r, c = int(pos[0]), int(pos[1])
                if 0 <= r < self.grid_shape[0] and 0 <= c < self.grid_shape[1]:
                    self.cell_heatmap[r, c] += 1
    
    def identify_safe_corners(
        self, 
        static_grid: np.ndarray, 
        cell_degrees: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Identify safe corner cells for pivoting.
        
        Safe corners are:
        - Dead ends (degree 1) or low-traffic corners
        - Far from high-traffic areas
        - Not goals
        """
        rows, cols = static_grid.shape
        safe_corners = []
        
        # Find dead ends
        for r in range(rows):
            for c in range(cols):
                if static_grid[r, c] != 0:
                    continue
                
                deg = cell_degrees[r, c]
                traffic = self.cell_heatmap[r, c]
                
                # Dead end with low traffic
                if deg == 1 and traffic < 5:
                    safe_corners.append((r, c))
                # Corner cells (edges of grid) with low traffic
                elif (r == 0 or r == rows-1 or c == 0 or c == cols-1) and traffic < 3:
                    safe_corners.append((r, c))
        
        # Sort by traffic (lowest first)
        safe_corners.sort(key=lambda cell: self.cell_heatmap[cell[0], cell[1]])
        
        self.safe_corners = safe_corners
        return safe_corners
    
    def select_safe_pivot(
        self,
        agent_goal: Tuple[int, int],
        blocked_cells: Set[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """
        Select best safe pivot cell for an agent at goal.
        
        Criteria:
        - Not in blocked_cells (other agents' paths)
        - Low traffic (heatmap)
        - Reasonable distance from goal (2-6 cells)
        """
        if not self.safe_corners:
            return None
        
        best_pivot = None
        best_score = float('inf')
        
        for pivot in self.safe_corners:
            if pivot in blocked_cells:
                continue
            
            # Distance from goal
            dist = abs(pivot[0] - agent_goal[0]) + abs(pivot[1] - agent_goal[1])
            if dist < 2 or dist > 8:
                continue
            
            # Score: traffic + distance penalty
            traffic = self.cell_heatmap[pivot[0], pivot[1]]
            score = traffic * 10 + dist * 0.5
            
            if score < best_score:
                best_score = score
                best_pivot = pivot
        
        return best_pivot
    
    def get_agent_priority(
        self,
        agent_id: int,
        is_at_goal: bool,
        collision_count: int,
        path_length: int
    ) -> float:
        """
        Calculate priority score for collision resolution.
        
        Lower score = higher priority.
        
        Factors:
        - Finished agents have lower priority (should yield)
        - Agents with more collisions have higher priority
        - Shorter paths have higher priority (close to goal)
        """
        score = 0.0
        
        # Finished agents should yield
        if is_at_goal:
            score += 100
        
        # Fewer collisions = higher priority
        score -= collision_count * 5
        
        # Longer path = lower priority
        score += path_length * 0.1
        
        return score
    
    def recommend_yield_cell(
        self,
        candidates: List[Tuple[Tuple[int, int], int]],
        cell_degrees: np.ndarray,
        flow_map: Dict[Tuple[int, int], int]
    ) -> Tuple[int, int]:
        """
        Recommend best yield cell using global heatmap + local flow.
        
        Combines:
        - Cell degree (structure)
        - Local flow (blocking graph)
        - Global heatmap (historical traffic)
        """
        if not candidates:
            return None
        
        best_cell = None
        best_score = float('inf')
        
        for cell, dist in candidates:
            r, c = cell
            
            score = 0.0
            
            # Structure (dead end bonus)
            if 0 <= r < cell_degrees.shape[0] and 0 <= c < cell_degrees.shape[1]:
                deg = cell_degrees[r, c]
                if deg == 1:
                    score -= 8
                elif deg == 2:
                    score += 4
                else:
                    score += 7
            
            # Local flow (immediate conflicts)
            local_flow = flow_map.get(cell, 0)
            score += 8 * (2 ** local_flow - 1)
            
            # Global heatmap (historical traffic)
            if 0 <= r < self.cell_heatmap.shape[0] and 0 <= c < self.cell_heatmap.shape[1]:
                traffic = self.cell_heatmap[r, c]
                score += traffic * 2
            
            # Distance (small penalty)
            score += 0.3 * dist
            
            if score < best_score:
                best_score = score
                best_cell = cell
        
        return best_cell
