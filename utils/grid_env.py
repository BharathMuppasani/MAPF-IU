from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

#####################################
# Helper Functions for Maze Generation
#####################################

def bfs_reachable(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    dynamic_obstacles: bool = False
) -> bool:
    """
    Returns True if there is a path from 'start' to 'goal' in the given 'grid'.

    Args:
        grid: 2D numpy array with:
            - -1 for static walls
            -  0 for free cells
            -  1 for agent
            -  2 for goal
            - -2 for dynamic obstacles
        start: (row, col) of the agent
        goal:  (row, col) of the goal
        dynamic_obstacles: if True, treat -2 cells as blocked; 
                           if False, treat -2 as free.

    Returns:
        True if a 4-neighbour path exists, False otherwise.
    """
    rows, cols = grid.shape
    if start == goal:
        return True

    # decide which cell-values to block
    blocked = {-1}
    if dynamic_obstacles:
        blocked.add(-2)

    visited = set([start])
    queue = deque([start])

    # 4-connected moves
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        pass  # just documenting

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if (nx, ny) not in visited and grid[nx, ny] not in blocked:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return False

def shortest_path_func(self,
                  grid:  np.ndarray,
                  start: Tuple[int, int],
                  goal:  Tuple[int, int]
                 ) -> Optional[List[Tuple[int, int]]]:
    """
    Breadth-First Search on the CURRENT grid.

    Returns a list of (row, col) cells that forms the shortest 4-neighbour
    path from `start` to `goal`. Walls (-1) and dynamic obstacles (-2)
    are treated as blocked.  If no path exists, returns None.
    """
    if start == goal:
        return [start]

    H, W = grid.shape
    sr, sc       = start
    gr, gc       = goal
    visited      = np.zeros_like(grid, dtype=bool)
    parent       = np.full((H, W, 2), -1, dtype=int)   # store predecessors
    q = deque([(sr, sc)])
    visited[sr, sc] = True

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]         # 4-connected

    # --- BFS -------------------------------------------------------------
    while q:
        r, c = q.popleft()
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W \
               and not visited[nr, nc]     \
               and grid[nr, nc] >= 0:      # passable
                visited[nr, nc] = True
                parent[nr, nc]  = [r, c]   # remember where we came from
                if (nr, nc) == (gr, gc):   # reached goal → reconstruct
                    path = [(gr, gc)]
                    while (r, c) != (sr, sc):
                        path.append((r, c))
                        r, c = parent[r, c]
                    path.append((sr, sc))
                    path.reverse()
                    return path
                q.append((nr, nc))

    # --------------------------------------------------------------------
    return None   # goal unreachable


def carve_passages(grid: np.ndarray, r: int, c: int):
    """
    Carves passages into the maze grid using recursive backtracking.
    The grid is assumed to be filled with obstacles (-1). 
    Passage cells are set to 0.
    Only odd indices are passages.
    """
    grid[r, c] = 0  # Mark as free passage
    directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
    random.shuffle(directions)
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 < nr < grid.shape[0] and 0 < nc < grid.shape[1] and grid[nr, nc] == -1:
            # Remove wall between (r, c) and (nr, nc)
            grid[r + dr//2, c + dc//2] = 0
            grid[nr, nc] = 0
            carve_passages(grid, nr, nc)

def get_candidate_walls(grid: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return a list of wall cell coordinates that are between two passages.
    """
    rows, cols = grid.shape
    candidates = []
    # Horizontal candidates
    for r in range(rows):
        for c in range(1, cols - 1):
            if grid[r, c] == -1 and grid[r, c - 1] == 0 and grid[r, c + 1] == 0:
                candidates.append((r, c))
    # Vertical candidates
    for r in range(1, rows - 1):
        for c in range(cols):
            if grid[r, c] == -1 and grid[r - 1, c] == 0 and grid[r + 1, c] == 0:
                candidates.append((r, c))
    return candidates

def neighbors_4(r: int, c: int, rows: int, cols: int):
    """Yield valid 4-directional neighbors of (r,c)."""
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

def remove_walls_in_clusters(grid: np.ndarray, candidate_walls: List[Tuple[int, int]], cluster_fraction: float):
    """
    Remove a fraction (cluster_fraction) of candidate walls in clusters.
    This tends to leave behind bigger blocks of obstacles (often T or L shaped).
    """
    total_to_remove = int(cluster_fraction * len(candidate_walls))
    if total_to_remove <= 0:
        return
    wall_set = set(candidate_walls)
    removed_count = 0
    random.shuffle(candidate_walls)
    for (r, c) in candidate_walls:
        if removed_count >= total_to_remove:
            break
        if (r, c) not in wall_set:
            continue
        # Start a cluster
        cluster = [(r, c)]
        frontier = deque([(r, c)])
        while frontier and len(cluster) < 5:  # Cluster size limit; tweak as needed.
            rr, cc = frontier.popleft()
            for nr, nc in neighbors_4(rr, cc, grid.shape[0], grid.shape[1]):
                if (nr, nc) in wall_set and random.random() < 0.75:
                    wall_set.remove((nr, nc))
                    cluster.append((nr, nc))
                    frontier.append((nr, nc))
                    if len(cluster) >= 5:
                        break
        for (rr, cc) in cluster:
            grid[rr, cc] = 0
        removed_count += len(cluster)

def generate_maze_grid(grid_size: int, maze_density: float = 1.0) -> np.ndarray:
    """
    Generates a maze using recursive backtracking on an odd-sized grid.
    grid_size must be odd.
    - The maze is initially perfect (all passages connected) with obstacles (-1) filling non-passage cells.
    - Then candidate walls (those between passages) are selectively removed in clusters.
    
    maze_density=1.0 leaves the perfect maze (more obstacles),
    while lower values (e.g. 0.8 or 0.5) remove more candidate walls.
    """
    # Create grid and initialize with obstacles
    grid = np.full((grid_size, grid_size), -1, dtype=int)
    # Start carving from cell (1,1)
    carve_passages(grid, 1, 1)
    # Get candidate walls (walls between passages)
    candidate_walls = get_candidate_walls(grid)
    # Remove a fraction of candidate walls (1 - maze_density)
    remove_fraction = 1.0 - maze_density
    remove_walls_in_clusters(grid, candidate_walls, remove_fraction)
    return grid

#####################################
# Grid Environment Classes
#####################################

class GridState:
    """
    Represents a single state of the grid environment.
    """
    __slots__ = ['grid', 'hash']

    def __init__(self, grid: np.ndarray):
        self.grid: np.ndarray = grid
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.grid.tobytes())
        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.grid, other.grid)

class GridEnvironment:
    """
    Represents the grid environment with static obstacles, an agent, and a goal.
    
    Two generation modes are supported:
      - "old": Uses the original random placement of static obstacles.
      - "maze": Uses a maze-generation method based on recursive backtracking.
      
    In maze mode, grid_size should be odd; if an even value is provided it is incremented by 1.
    """
    moves: List[str] = ['U', 'D', 'L', 'R']

    def __init__(
            self, 
            grid_size: int, 
            num_static_obstacles: int, 
            num_dynamic_obstacles: int = 0,
            generation_mode: str = None, # "maze", "warehouse"
            maze_density: float = 1.0, 
            max_steps: Optional[int] = 200,
            min_goal_dist: int = 4 # Added for warehouse and potentially other modes
        ):
        
        self.generation_mode = generation_mode
        if self.generation_mode == "maze" and grid_size % 2 == 0:
            grid_size += 1
        self.grid_size = grid_size # Used by place_randomly and other methods
        self.rows = grid_size # Explicit rows for clarity
        self.cols = grid_size # Explicit cols for clarity

        self.num_static_obstacles = num_static_obstacles # Used by _generate_old_grid
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.maze_density = maze_density
        self.min_goal_dist = min_goal_dist # Used for agent-goal separation

        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.last_agent_pos: Optional[Tuple[int, int]] = None
        self.oscillation: bool = False
        self.step: int = 0
        self.max_steps: Optional[int] = max_steps
        self.shortest_path: Optional[List[Tuple[int, int]]] = None
        self.path_followed: bool = True
        self.visited_states: set = set()
        self.dynamic_info: List[dict] = []

        if self.generation_mode is not None:
            self.generate_grid()

    def _initialize_dynamic_obstacles(self):
        """
        After grid generation, locate each dynamic obstacle (-2), sample
        a random goal, and compute an initial (possibly non-optimal) path.
        """
        self.dynamic_info.clear()
        # find all positions of -2
        obstacle_positions = list(map(tuple, np.argwhere(self.grid == -2)))
        free_cells = list(map(tuple, np.argwhere(self.grid == 0)))
        for pos in obstacle_positions:
            goal = random.choice(free_cells)
            goal = tuple(int(p) for p in goal)
            path = [tuple(int(p) for p in x) for x in self._compute_nonoptimal_path(pos, goal)]            
            pos = tuple(int(p) for p in pos)
            self.dynamic_info.append({
                'pos': pos,
                'goal': goal,
                'path': deque(path),
                'stop_after_goal': True,
            })
            # print(f"Dynamic obstacle at {pos} with goal {goal} and path {path}")

            # print(f"Dynamic obstacle at {pos} with goal {goal} and path {path}")

    def shortest_path_func(self,
                    grid:  np.ndarray,
                    start: Tuple[int, int],
                    goal:  Tuple[int, int]
                    ) -> Optional[List[Tuple[int, int]]]:
        """
        Breadth-First Search on the CURRENT grid.

        Returns a list of (row, col) cells that forms the shortest 4-neighbour
        path from `start` to `goal`. Walls (-1) and dynamic obstacles (-2)
        are treated as blocked.  If no path exists, returns None.
        """
        if start == goal:
            return [start]

        H, W = grid.shape
        sr, sc       = start
        gr, gc       = goal
        visited      = np.zeros_like(grid, dtype=bool)
        parent       = np.full((H, W, 2), -1, dtype=int)   # store predecessors
        q = deque([(sr, sc)])
        visited[sr, sc] = True

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]         # 4-connected

        # --- BFS -------------------------------------------------------------
        while q:
            r, c = q.popleft()
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W \
                and not visited[nr, nc]     \
                and grid[nr, nc] >= 0:      # passable
                    visited[nr, nc] = True
                    parent[nr, nc]  = [r, c]   # remember where we came from
                    if (nr, nc) == (gr, gc):   # reached goal → reconstruct
                        path = [(gr, gc)]
                        while (r, c) != (sr, sc):
                            path.append((r, c))
                            r, c = parent[r, c]
                        path.append((sr, sc))
                        path.reverse()
                        return path
                    q.append((nr, nc))

        # --------------------------------------------------------------------
        return None   # goal unreachable

    def _compute_nonoptimal_path(self,
                                start: Tuple[int, int],
                                goal: Tuple[int, int]
                                ) -> List[Tuple[int, int]]:
        """
        Generate a path from start to goal that is not necessarily shortest.
        We optionally insert a random pivot to lengthen the route.
        Passable cells are *only* those with grid value == 0 (empty),
        except that the very last step into `goal` (grid==2) is allowed.
        Walls (-1) or dynamic obstacles (-2) are blocked.
        """
        H, W = self.grid.shape
        free_cells = [(r, c) for r in range(H) for c in range(W) if self.grid[r, c] == 0]

        def bfs_path(s: Tuple[int, int], g: Tuple[int, int]) -> List[Tuple[int, int]]:
            """Return the shortest path s→g using 4-neighbours, or [s] if unreachable."""
            queue = deque([[s]])
            visited = {s}
            while queue:
                path = queue.popleft()
                x, y = path[-1]
                if (x, y) == g:
                    return path
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited:
                        # allow only empty cells (grid==0), except permit stepping into the goal cell
                        if self.grid[nx, ny] == 0 or (nx, ny) == g:
                            visited.add((nx, ny))
                            queue.append(path + [(nx, ny)])
            return [s]

        # 50% chance to pick a random pivot (detour)
        if free_cells and random.random() < 0.5:
            pivot = random.choice(free_cells)
            path1 = bfs_path(start, pivot)
            path2 = bfs_path(pivot, goal)
            # if both segments succeeded (length>1), stitch them
            if len(path1) > 1 and len(path2) > 1:
                # drop duplicate pivot from second segment
                return path1 + path2[1:]

        # fallback to the direct shortest path
        return bfs_path(start, goal)

    def generate_batch_states(self, batch_size: int, min_obstacles: int = 0) -> List[GridState]:
        """
        Generate a batch of states with varying number of static obstacles.
        """
        states = []
        original_num = self.num_static_obstacles
        for _ in range(batch_size):
            if self.generation_mode == "old" and self.num_static_obstacles > 0:
                self.num_static_obstacles = random.randint(min_obstacles, original_num)
            self.generate_grid()
            states.append(GridState(self.grid.copy()))
        self.num_static_obstacles = original_num
        return states

    def generate_grid(self):
        if self.generation_mode == "maze":
            self._generate_maze_grid()
        elif self.generation_mode == "warehouse": 
            self._generate_warehouse_grid()
        else: # Default to "old"
            self._generate_old_grid()
        self.step = 0

        self._initialize_dynamic_obstacles()
        # Ensure agent_pos and goal_pos are tuples of ints after generation
        if self.agent_pos: self.agent_pos = tuple(map(int, self.agent_pos))
        if self.goal_pos: self.goal_pos = tuple(map(int, self.goal_pos))

    def _generate_old_grid(self):
        """
        Original grid generation:
          - Fills grid with zeros.
          - Randomly places agent (1), goal (2) and static obstacles (-1).
          - Repeats until BFS confirms connectivity between agent and goal.
        """
        while True:
            self.grid.fill(0)

            # Place agent
            self.agent_pos = self.place_randomly(value=1)

            # Place goal ensuring separation
            self.goal_pos = self.place_randomly(value=2)
            while (abs(self.agent_pos[0] - self.goal_pos[0]) < 4) or (abs(self.agent_pos[1] - self.goal_pos[1]) < 4):
                self.grid[self.goal_pos] = 0
                self.goal_pos = self.place_randomly(value=2)

            # Place static obstacles; number chosen randomly between 10 and num_static_obstacles
            # num_obs = random.randint(10, self.num_static_obstacles)
            num_obs = self.num_static_obstacles
            for _ in range(num_obs):
                self.place_randomly(value=-1)

            for _ in range(self.num_dynamic_obstacles):
                self.place_randomly(value=-2)

            if bfs_reachable(self.grid, tuple(self.agent_pos), tuple(self.goal_pos)):
                break

        self.shortest_path = shortest_path_func(self, self.grid, tuple(self.agent_pos), tuple(self.goal_pos))
        # self._initialize_dynamic_obstacles()

    def _generate_maze_grid(self):
        """
        Maze-based grid generation:
          - Uses the maze generator to create a maze (with obstacles as full cells).
          - Then randomly places an agent (1) and goal (2) in free cells,
            ensuring a minimum separation and connectivity.
        """
        # Use the maze generator (grid_size must be odd)
        while True:
            new_grid = generate_maze_grid(self.grid_size, self.maze_density)
            
            # Local helper to place a value in a random free cell of new_grid
            def place_randomly_in(g: np.ndarray, value: int) -> Tuple[int, int]:
                free_cells = list(zip(*np.where(g == 0)))
                pos = random.choice(free_cells)
                g[pos] = value
                return pos

            agent_pos = place_randomly_in(new_grid, 1)
            goal_pos = place_randomly_in(new_grid, 2)
            while (abs(agent_pos[0] - goal_pos[0]) < 4) or (abs(agent_pos[1] - goal_pos[1]) < 4):
                new_grid[goal_pos] = 0
                goal_pos = place_randomly_in(new_grid, 2)
            
            if bfs_reachable(new_grid, tuple(agent_pos), tuple(goal_pos)):
                # Place dynamic obstacles
                for _ in range(self.num_dynamic_obstacles):
                    place_randomly_in(new_grid, -2)
                self.grid = new_grid
                self.agent_pos = agent_pos
                self.goal_pos = goal_pos
                break

        self.shortest_path = shortest_path_func(self, self.grid, tuple(self.agent_pos), tuple(self.goal_pos))
        # self._initialize_dynamic_obstacles()

    def _generate_warehouse_grid(self):
        """
        Generates a structured warehouse grid with 3-cell borders.
        Shelves are -1, Aisles are 0.
        Aisles between shelf blocks are 2 cells wide.
        """
        border_size = 3
        if self.rows <= 2 * border_size or self.cols <= 2 * border_size:
            print(f"Warning: Grid size {self.grid_size}x{self.grid_size} is too small for {border_size}-cell border and shelves. Using 'old' generation.")
            self._generate_old_grid()
            return

        # Warehouse structure parameters
        shelf_unit_depth = 2  # How many rows a single shelf unit occupies
        shelf_unit_width = 3  # How many columns a single shelf unit occupies
        main_aisle_width = 2 # Horizontal aisles BETWEEN shelf rows
        intra_shelf_aisle_width = 1 # Vertical gap BETWEEN shelf units IN THE SAME shelf row
        
        self.grid.fill(0) # Start with all paths (borders will remain 0)

        # Define the inner area for shelf placement
        shelf_area_start_r = border_size
        shelf_area_end_r = self.rows - border_size
        shelf_area_start_c = border_size
        shelf_area_end_c = self.cols - border_size

        # Place horizontal shelf blocks within the defined middle area
        current_r = shelf_area_start_r
        while current_r < shelf_area_end_r:
            # Try to place a shelf block
            if current_r + shelf_unit_depth <= shelf_area_end_r:
                for r_offset in range(shelf_unit_depth):
                    r_shelf = current_r + r_offset
                    current_c = shelf_area_start_c
                    while current_c < shelf_area_end_c:
                        if current_c + shelf_unit_width <= shelf_area_end_c:
                            self.grid[r_shelf, current_c : current_c + shelf_unit_width] = -1
                            current_c += shelf_unit_width
                        else: break 
                        
                        if current_c < shelf_area_end_c : 
                            aisle_end_c = min(shelf_area_end_c, current_c + intra_shelf_aisle_width)
                            # self.grid[r_shelf, current_c : aisle_end_c] = 0 # Already 0
                            current_c += intra_shelf_aisle_width
                        else: break
                current_r += shelf_unit_depth
            else: break

            # Add main horizontal aisle within the shelf area
            if current_r < shelf_area_end_r:
                aisle_end_r = min(shelf_area_end_r, current_r + main_aisle_width)
                # self.grid[current_r : aisle_end_r, shelf_area_start_c:shelf_area_end_c] = 0 # Already 0
                current_r += main_aisle_width
            else: break
        
        # --- Placement and Connectivity (Retry loop) ---
        max_overall_retries = 30 
        for attempt in range(max_overall_retries):
            current_attempt_grid = self.grid.copy() 

            def place_on_attempt_grid(g: np.ndarray, val: int) -> Optional[Tuple[int,int]]:
                empty_cells = list(zip(*np.where(g == 0)))
                if not empty_cells: return None
                pos = random.choice(empty_cells)
                g[pos] = val
                return pos

            temp_agent_pos = place_on_attempt_grid(current_attempt_grid, 1)
            temp_goal_pos = place_on_attempt_grid(current_attempt_grid, 2)

            if temp_agent_pos is None or temp_goal_pos is None:
                if attempt < max_overall_retries -1 : continue 
                else: break 

            dist_ok = False
            for _ in range(self.rows * self.cols): 
                if not ((abs(temp_agent_pos[0] - temp_goal_pos[0]) < self.min_goal_dist) and \
                        (abs(temp_agent_pos[1] - temp_goal_pos[1]) < self.min_goal_dist)) and \
                   (temp_agent_pos != temp_goal_pos):
                    dist_ok = True
                    break
                if temp_goal_pos: current_attempt_grid[temp_goal_pos] = 0 
                temp_goal_pos = place_on_attempt_grid(current_attempt_grid, 2)
                if temp_goal_pos is None: break
            
            if not dist_ok or temp_goal_pos is None:
                if attempt < max_overall_retries -1 : continue
                else: break

            for _ in range(self.num_dynamic_obstacles):
                place_on_attempt_grid(current_attempt_grid, -2)

            if bfs_reachable(current_attempt_grid, temp_agent_pos, temp_goal_pos):
                self.grid = current_attempt_grid 
                self.agent_pos = temp_agent_pos
                self.goal_pos = temp_goal_pos
                return 
        
        print(f"Warning: _generate_warehouse_grid failed after {max_overall_retries} attempts. Using fallback 'old' generation.")


    def _place_randomly_on_grid(self, value: int) -> Optional[Tuple[int, int]]:
        """
        Helper to place a value in a random free cell (value 0) of self.grid.
        Returns position if successful, None otherwise.
        """
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if not empty_cells:
            return None
        pos = random.choice(empty_cells)
        self.grid[pos] = value
        return pos

    def place_randomly(self, value: int) -> Tuple[int, int]:
        """
        Place a value (agent, goal, or obstacle) in a random free cell (value 0) of self.grid.
        """
        while True:
            x = np.random.randint(0, self.rows)
            y = np.random.randint(0, self.cols)
            if self.grid[x, y] == 0:
                self.grid[x, y] = value
                return (x, y)


    # def update_dynamic_obstacles(self):
    #     """
    #     Move each dynamic obstacle along its path; when its own path is done or it
    #     reaches its goal:
    #     - if stop_after_goal == True → it stays at the goal indefinitely
    #     - else (stop_after_goal == False) → pick a new random goal & path (old behavior)
    #     """
    #     if self.num_dynamic_obstacles == 0:
    #         return

    #     new_grid      = self.grid.copy()
    #     agent_cell    = tuple(self.agent_pos)
    #     old_positions = [info['pos'] for info in self.dynamic_info]
    #     new_positions = set()
    #     static_walls  = {tuple(rc) for rc in np.argwhere(self.grid == -1)}
    #     free_cells    = [tuple(rc) for rc in np.argwhere(self.grid == 0)]

    #     for info in self.dynamic_info:
    #         curr      = info['pos']
    #         goal      = info.get('goal', None)
    #         stop_flag = info.get('stop_after_goal', False)

    #         # 1) Path exhausted or reached its goal
    #         if not info['path'] or curr == goal:
    #             if stop_flag:
    #                 # Stay at goal: mark it as a dynamic obstacle there
    #                 new_grid[curr] = -2
    #                 new_positions.add(curr)
    #             else:
    #                 # Respawn with a fresh random goal & path
    #                 info['goal'] = random.choice(free_cells)
    #                 info['path'] = deque(self._compute_nonoptimal_path(
    #                     curr, info['goal']
    #                 ))
    #                 new_grid[curr]    = -2
    #                 new_positions.add(curr)
    #             continue

    #         # 2) Normal “advance along path” logic
    #         next_pos = info['path'][0]

    #         # Clear old position
    #         if curr == tuple(self.goal_pos):
    #             new_grid[curr] = 2
    #         else:
    #             new_grid[curr] = 0

    #         # Mark next position
    #         if next_pos == agent_cell:
    #             new_grid[next_pos] = -3  # collision
    #         else:
    #             new_grid[next_pos] = -2  # dynamic obstacle

    #         # Advance
    #         info['path'].popleft()
    #         info['pos'] = next_pos
    #         new_positions.add(next_pos)

    #     # Commit updated grid; dynamic_info entries remain (no removals)
    #     self.grid = new_grid



    def update_dynamic_obstacles(self):
        """
        Move each dynamic obstacle along its path; when its own path is done:
        - if stop_after_goal == True → remove it outright
        - else (stop_after_goal == False) → pick a new random goal & path (old behavior)
        """

        if self.num_dynamic_obstacles == 0:
            return

        new_grid        = self.grid.copy()
        agent_cell      = tuple(self.agent_pos)
        old_positions   = [info['pos'] for info in self.dynamic_info]
        new_positions   = set()
        static_walls    = {tuple(rc) for rc in np.argwhere(self.grid == -1)}

        free_cells = [tuple(rc) for rc in np.argwhere(self.grid == 0)]
        to_remove  = []

        for info in self.dynamic_info:
            curr      = info['pos']
            goal      = info.get('goal', None)
            stop_flag = info.get('stop_after_goal', False)

            # 1) If its own path is exhausted or it just reached its goal:
            if not info['path'] or curr == goal:
                if stop_flag:
                    # delete it completely
                    new_grid[curr] = 0
                    to_remove.append(info)
                else:
                    # old behavior: respawn with a fresh random goal+path
                    info['goal'] = random.choice(free_cells)
                    info['path'] = deque(self._compute_nonoptimal_path(
                        curr, info['goal']
                    ))
                    new_grid[curr]    = -2
                    new_positions.add(curr)
                continue

            # 2) Normal “advance along path” logic
            next_pos = info['path'][0]
            # reserved = (
            #     set(old_positions) - {curr}
            #     | static_walls
            #     | new_positions
            #     | {agent_cell}
            # )
            # if next_pos in reserved:
            #     # collision → replan & stay
            #     info['path'] = deque(self._compute_nonoptimal_path(
            #         curr, info['goal']
            #     ))
            #     new_grid[curr]    = -2
            #     new_positions.add(curr)
            #     continue

            # 3) Step forward
            if curr == tuple(self.goal_pos):
                new_grid[curr] = 2
            else:
                new_grid[curr] = 0

            if next_pos == agent_cell:
                new_grid[next_pos] = -3  # collision
            else:
                new_grid[next_pos] = -2


            info['path'].popleft()
            # new_grid[next_pos] = -2
            info['pos']        = next_pos
            new_positions.add(next_pos)

        # 4) Commit and purge
        self.grid = new_grid
        for info in to_remove:
            self.dynamic_info.remove(info)


    def get_actions(self) -> List[int]:
        valid_actions = []
        ax, ay = self.agent_pos
        last = self.last_agent_pos

        # Standard action-to-movement mapping
        movements = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

        # Determine forbidden cells based on dynamic obstacles' next step
        forbidden: set[Tuple[int, int]] = set()
        for obs in getattr(self, 'dynamic_info', []):
            path = obs.get('path')
            # If obstacle has a next step, mark it forbidden
            if path and len(path) > 1:
                next_cell = path[1]
                forbidden.add(tuple(next_cell))

        for action, (dx, dy) in movements.items():
            nx, ny = ax + dx, ay + dy
            # Out of bounds
            if nx < 0 or nx >= self.rows or ny < 0 or ny >= self.cols:
                continue
            # Walls or static obstacles
            if action != 4 and self.grid[nx, ny] in [-1, -2]:
                continue
            # Avoid stepping into where a dynamic obstacle will move
            if (nx, ny) in forbidden:
                continue
            # Optional: avoid going back to last position
            # if action != 4 and last is not None and (nx, ny) == last:
            #     continue

            valid_actions.append(action)

        return valid_actions

    # def get_actions(self) -> List[int]:
    #     valid_actions = []
    #     ax, ay = self.agent_pos
    #     last = self.last_agent_pos

    #     movements = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

    #     for action, (dx, dy) in movements.items():
    #         nx, ny = ax + dx, ay + dy
    #         # Check for out-of-bounds
    #         if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
    #             continue
    #         # Check for walls and obstacles
    #         if action != 4 and self.grid[nx, ny] in [-1, -2]:
    #             continue
    #         # Check for oscillation
    #         # if action != 4 and last is not None and (nx, ny) == last: 
    #         #     continue

    #         valid_actions.append(action)

    #     # if len(valid_actions) == 1 and valid_actions[0] == 4:
    #     #     valid_actions = []
    #     #     for action, (dx, dy) in movements.items():
    #     #         nx, ny = ax + dx, ay + dy
    #     #         # Check for out-of-bounds
    #     #         if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
    #     #             continue
    #     #         # Check for walls and obstacles
    #     #         if action != 4 and self.grid[nx, ny] in [-1, -2]:
    #     #             continue
    #     #         valid_actions.append(action)


    #     return valid_actions


    def next_state(self, state: GridState, action_index: int) -> Tuple[GridState, bool, bool]:
        """
        Apply an action to the current state and return the next state along with flags.
        
        Args:
            state (GridState): Current state.
            action_index (int): Action (0: Up, 1: Down, 2: Left, 3: Right).
            
        Returns:
            Tuple[GridState, bool, bool]: (next_state, done, goal_reached)
        """
        
        self.update_dynamic_obstacles()  # Update dynamic obstacles first.
        if self.grid[self.agent_pos] == -3:
            return state, True, False
        
        new_grid = np.array(self.grid, copy=True)  # Use the updated self.grid.
        temp_copy = np.array(new_grid, copy=True)
        # remove all dynamic obstacles in temp_copy -2 values
        temp_copy[temp_copy == -2] = 0
        self.visited_states.add(GridState(temp_copy))
        self.step += 1

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        
        movements = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),    # Right
            4: (0, 0)    # Stay
        }
        dx, dy = movements[action_index]
        nx, ny = ax + dx, ay + dy # new agent position

        if nx < 0 or nx >= self.rows or ny < 0 or ny >= self.cols:
            self.agent_pos = (ax, ay)
            return state, True, False
        
        if new_grid[nx, ny] in [-1,-2]:
            new_grid[nx, ny] = -3  # Mark as collision
            self.agent_pos = (nx, ny)
            self.last_agent_pos = (ax, ay)
            return state, True, False
        
        if action_index != 4:
            new_grid[ax, ay] = 0
            new_grid[nx, ny] = 1
            self.agent_pos = (nx, ny)
            self.grid = new_grid
            self.last_agent_pos = (ax, ay)
        else:
            self.last_agent_pos = (ax, ay)

        # ─── NEW: keep shortest_path aligned & clipped ──────────────────────
        if not self.shortest_path or self.shortest_path[0] != (ax, ay):
            self.shortest_path = shortest_path_func(
                self, new_grid, (ax, ay), (gx, gy)
            ) or []

        # did we follow the path?
        if len(self.shortest_path) >= 2 and (nx, ny) == self.shortest_path[1]:
            # clip the head (agent progressed)
            self.shortest_path = self.shortest_path[1:]
            self.path_followed = True
        else:
            # recompute if deviated
            self.shortest_path = shortest_path_func(
                self, new_grid, (nx, ny), (gx, gy)
            ) or []
            self.path_followed = False
        # ────────────────────────────────────────────────────────────────────

        if (nx, ny) == (gx, gy):
            new_grid[gx, gy] = 3  # Mark goal reached
            self.grid = new_grid
            self.agent_pos = (gx, gy)
            self.goal_pos = (gx, gy)
            return GridState(new_grid), True, True
        
        if self.step >= self.max_steps:
            return GridState(new_grid), True, False

        return GridState(new_grid), False, False

    
    def calculate_reward(self,
                        state: GridState,
                        action: int,
                        done: bool,
                        goal_flag: bool) -> float:
        """
        • +20   goal reached
        • -5    collision
        • -3    timeout

        Non-terminal:
        • +0.20 if path_followed else -0.10
        • -0.05 for 'stay' (action==4)
        • -0.10 if any neighbor is a dynamic obstacle
        • +a*(old_dist - new_dist)  distance-shaping
        • -0.10 if immediately backtrack
        """
        # ─── TERMINAL ───────────────────────────────────
        if done:
            if goal_flag:
                return +20.0
            return -3.0 if self.step < self.max_steps else -2.0

        if len(np.argwhere(state.grid == 1)) == 0:
            return -3.0

        # ─── LOCATE CURRENT & NEXT POS ─────────────────
        moves = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1), 4:(0,0)}
        
        # print(state.grid)
        # print(self.agent_pos)
        # print(action)
        # print(np.argwhere(state.grid == 1))
        cur_r, cur_c = np.argwhere(state.grid == 1)[0]
        dx, dy = moves[action]
        nxt_r, nxt_c = cur_r + dx, cur_c + dy

        if nxt_r >= self.rows or nxt_c >= self.cols:
            return -3.0
        if nxt_r < 0 or nxt_c < 0:
            return -3.0
        
        # ─── PATH FOLLOWED REWARD ──────────────────────

        # check if the new state is in visited states then give negative reward
        new_grid = state.grid.copy()
        new_grid[cur_r, cur_c] = 0
        new_grid[nxt_r, nxt_c] = 1

        temp_copy = np.array(new_grid, copy=True)
        temp_copy[temp_copy == -2] = 0
        if GridState(temp_copy) in self.visited_states:
            step_r = -0.15
        else:
            if action == 4:
                step_r = -0.1
            else:
                step_r = +0.20 if self.path_followed else -0.02

        # ─── NEAR-DYNAMIC PENALTY ──────────────────────
        near_pen = 0.0
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            rr, cc = nxt_r+dr, nxt_c+dc
            if 0<=rr<state.grid.shape[0] and 0<=cc<state.grid.shape[1] \
            and state.grid[rr,cc]==-2:
                near_pen = -0.05
                break

        # ─── TOTAL ─────────────────────────────────────
        return step_r + near_pen

    

    def render_manual(self, collision_message=None, goal_flag=False):
        rows, cols = self.grid.shape
        fig, ax = plt.subplots(figsize=(5, 5))
        for i in range(rows):
            for j in range(cols):
                value = self.grid[i, j]
                if value == -1:
                    ax.fill([j, j+1, j+1, j], [i, i, i+1, i+1], color='black', alpha=0.3)
                elif value == 0:
                    ax.fill([j, j+1, j+1, j], [i, i, i+1, i+1], color='white', alpha=0.0)
                elif value == 1:
                    if (i, j) == self.goal_pos:
                        ax.fill([j, j+1, j+1, j], [i, i, i+1, i+1], color='mediumseagreen', alpha=0.5)
                        ax.text(j+0.5, i+0.5, 'A/G', ha='center', va='center', 
                                color='darkgreen', fontweight='bold')
                    else:
                        ax.fill([j, j+1, j+1, j], [i, i, i+1, i+1], color='lightblue', alpha=0.3)
                        ax.text(j+0.5, i+0.5, 'A', ha='center', va='center', 
                                color='blue', fontweight='bold')
                elif value == 2:
                    ax.fill([j, j+1, j+1, j], [i, i, i+1, i+1], color='lightgreen', alpha=0.3)
                    ax.text(j+0.5, i+0.5, 'G', ha='center', va='center', 
                            color='green', fontweight='bold')
                elif value == -2:
                    ax.fill([j, j+1, j+1, j], [i, i, i+1, i+1], color='red', alpha=0.3)
                    ax.text(j+0.5, i+0.5, 'D', ha='center', va='center', 
                            color='red', fontweight='bold')
        for i in range(rows + 1):
            ax.axhline(y=i, color='gray', linewidth=1)
        for j in range(cols + 1):
            ax.axvline(x=j, color='gray', linewidth=1)
        if collision_message and not goal_flag:
            ax.add_patch(plt.Rectangle((0, 0), cols, rows, color='red', alpha=0.3))
            ax.text(cols/2, rows/2, "Collision", ha='center', va='center',
                    fontsize=16, fontweight='bold', color='red',
                    bbox=dict(facecolor='white', edgecolor='red', pad=10))
        elif collision_message and goal_flag:
            ax.add_patch(plt.Rectangle((0, 0), cols, rows, color='green', alpha=0.3))
            ax.text(cols/2, rows/2, "Goal reached!", ha='center', va='center',
                    fontsize=16, fontweight='bold', color='green',
                    bbox=dict(facecolor='white', edgecolor='green', pad=10))
        ax.set_title("Grid Environment")
        ax.set_xlim(-0.05, cols + 0.05)
        ax.set_ylim(rows + 0.05, -0.05)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    def render(self, collision_message=None, goal_flag=False):
        """
        Render the grid with real-time visualization updates.
        """
        if not hasattr(self, 'fig'):
            self.fig = plt.figure(figsize=(8, 8))
            plt.ion()
        plt.clf()
        ax = plt.gca()
        rows, cols = self.grid.shape
        for i in range(rows):
            for j in range(cols):
                value = self.grid[i, j]
                if value == -1:
                    plt.fill([j, j+1, j+1, j], [i, i, i+1, i+1], 'black', alpha=0.3)
                elif value == 0:
                    pass
                elif value == 1:
                    if (i, j) == self.goal_pos:
                        plt.fill([j, j+1, j+1, j], [i, i, i+1, i+1], 'mediumseagreen', alpha=0.5)
                        plt.text(j+0.5, i+0.5, 'A/G', ha='center', va='center', color='darkgreen', fontweight='bold')
                    else:
                        plt.fill([j, j+1, j+1, j], [i, i, i+1, i+1], 'lightblue', alpha=0.3)
                        plt.text(j+0.5, i+0.5, 'A', ha='center', va='center', color='blue', fontweight='bold')
                elif value == 2:
                    plt.fill([j, j+1, j+1, j], [i, i, i+1, i+1], 'lightgreen', alpha=0.3)
                    plt.text(j+0.5, i+0.5, 'G', ha='center', va='center', color='green', fontweight='bold')
        for i in range(rows + 1):
            plt.axhline(y=i, color='gray', linewidth=1)
        for j in range(cols + 1):
            plt.axvline(x=j, color='gray', linewidth=1)
        plt.grid(False)
        plt.axis('equal')
        plt.xlim(-0.05, cols + 0.05)
        plt.ylim(rows + 0.05, -0.05)
        plt.xticks([])
        plt.yticks([])
        if collision_message and not goal_flag:
            plt.fill([0, cols, cols, 0], [0, 0, rows, rows], 'red', alpha=0.1)
            plt.text(cols/2, rows/2, collision_message,
                     ha='center', va='center', color='red', fontsize=12,
                     fontweight='bold', bbox=dict(facecolor='white', edgecolor='red', pad=10))
        elif collision_message and goal_flag:
            plt.fill([0, cols, cols, 0], [0, 0, rows, rows], 'green', alpha=0.1)
            plt.text(cols/2, rows/2, "Goal reached!",
                     ha='center', va='center', color='green', fontsize=12,
                     fontweight='bold', bbox=dict(facecolor='white', edgecolor='green', pad=10))
        plt.title("Grid Environment")
        plt.tight_layout()
        if collision_message:
            plt.ioff()
            plt.show()
        else:
            plt.pause(0.3)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def __del__(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            plt.ioff()
