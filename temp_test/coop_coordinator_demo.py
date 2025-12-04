
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional, Set

Grid = List[List[int]]
Pos = Tuple[int, int]
DIRS = [(1,0),(-1,0),(0,1),(0,-1)]

def in_bounds(g: Grid, p: Pos) -> bool:
    h, w = len(g), len(g[0])
    return 0 <= p[0] < h and 0 <= p[1] < w

def neighbors(g: Grid, p: Pos) -> List[Pos]:
    return [(p[0]+dx, p[1]+dy) for dx,dy in DIRS if in_bounds(g, (p[0]+dx, p[1]+dy)) and g[p[0]+dx][p[1]+dy] == 0]

def bfs_path(g: Grid, start: Pos, goal: Pos, blocked: Set[Pos]=set()) -> Optional[List[Pos]]:
    if start in blocked or goal in blocked: return None
    q = deque([start]); prev = {start: None}; seen = {start}
    while q:
        u = q.popleft()
        if u == goal:
            path = []
            cur = u
            while cur is not None:
                path.append(cur); cur = prev[cur]
            return path[::-1]
        for v in neighbors(g, u):
            if v in seen or v in blocked: continue
            seen.add(v); prev[v] = u; q.append(v)
    return None

def parse_ascii_zero_index(text: str) -> Tuple[Grid, List[Dict]]:
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    # Find N line
    n_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().isdigit():
            n_idx = i; break
    if n_idx is None: raise ValueError("No agent count line found.")
    grid_lines = lines[:n_idx]; n = int(lines[n_idx]); agent_lines = lines[n_idx+1:n_idx+1+n]
    # Parse grid: '@' -> 1, '.' -> 0 (ignore spaces)
    grid: Grid = []
    for ln in grid_lines:
        row = []
        for ch in ln:
            if ch == '@': row.append(1)
            elif ch == '.': row.append(0)
            else: continue
        if row: grid.append(row)
    w = len(grid[0])
    for r in grid:
        if len(r) != w: raise ValueError("Non-rectangular grid.")
    # Agents
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    agents = []
    for i, ln in enumerate(agent_lines):
        sr, sc, gr, gc = map(int, ln.split())
        agents.append({"id": i, "label": labels[i] if i < len(labels) else f"X{i}", "start": (sr, sc), "goal": (gr, gc)})
    return grid, agents

def degrees(g: Grid) -> Dict[Pos, int]:
    deg = {}
    h, w = len(g), len(g[0])
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0:
                deg[(r,c)] = len(neighbors(g, (r,c)))
    return deg

class Coordinator:
    """
    Centralized step-by-step simulator with:
      - static BFS paths per agent (can be replaced by your planner),
      - vertex and edge-swap conflict avoidance,
      - junction pullout yielding (one agent tucks into a side cell W),
      - minimal reservation hints for the next tick.
    """
    def __init__(self, grid: Grid, agents: List[Dict], max_steps: int=200):
        self.g = grid
        self.agents = agents
        self.N = len(agents)
        self.max_steps = max_steps
        self.paths: Dict[int, List[Pos]] = {}
        for a in agents:
            self.paths[a["id"]] = bfs_path(grid, a["start"], a["goal"], blocked=set())
        self.pos: Dict[int, Pos] = {a["id"]: a["start"] for a in agents}
        self.ptr: Dict[int, int] = {a["id"]: 0 for a in agents}
        self.at_goal: Set[int] = set()
        self.timeline: List[Dict[int, Pos]] = []
        self.reservations: Dict[int, Dict[Pos, int]] = defaultdict(dict)
        self.events: List[str] = []
        self.deg = degrees(grid)

    def priority(self, i: int) -> Tuple[int, int]:
        # higher priority = fewer remaining steps (let the nearly-done agent pass)
        p = self.paths[i]
        rem = 10**6
        if p is not None:
            rem = len(p) - 1 - self.ptr[i]
        return (-rem, -i)

    def propose(self, t: int) -> Dict[int, Pos]:
        prop = {}
        for a in self.agents:
            i = a["id"]; cur = self.pos[i]; p = self.paths[i]
            if cur == a["goal"]:
                prop[i] = cur; self.at_goal.add(i); continue
            if p is None or self.ptr[i]+1 >= len(p):
                prop[i] = cur; continue
            prop[i] = p[self.ptr[i]+1]
        return prop

    def resolve(self, t: int, prop: Dict[int, Pos]) -> Dict[int, Pos]:
        cur = dict(self.pos)
        final = dict(prop)
        target_to_agents = defaultdict(list)
        for i, cell in prop.items():
            target_to_agents[cell].append(i)

        # (A) Junction-with-pullout rule:
        # If two agents propose the same cell J AND J has a neighbor W with deg != 2 (a bay),
        # lower-priority agent moves into J now, higher-priority waits; next tick, the yielder moves to W and the other moves to J.
        for J, ids in list(target_to_agents.items()):
            if len(ids) == 2:
                pullouts = [nb for nb in neighbors(self.g, J) if self.deg.get(nb, 0) != 2]
                if pullouts and all(cur[i] != J for i in ids):
                    i, j = ids[0], ids[1]
                    if self.priority(i) >= self.priority(j):
                        yielder, other = j, i
                    else:
                        yielder, other = i, j
                    final[other] = cur[other]
                    final[yielder] = J
                    # choose a free pullout for next tick
                    W = None
                    for cand in pullouts:
                        if cand not in cur.values() and cand not in self.reservations.get(t+1, {}):
                            W = cand; break
                    if W:
                        self.reservations[t+1][W] = yielder
                        self.reservations[t+1][J] = other
                        self.events.append(f"t={t}: pullout at J{J}, yielder {yielder} -> J then W{W}, other {other} waits then -> J.")

        # Rebuild target map after special handling
        target_to_agents.clear()
        for i, cell in final.items():
            target_to_agents[cell].append(i)

        # (B) Generic vertex conflicts: let higher priority go
        for cell, ids in list(target_to_agents.items()):
            if len(ids) > 1:
                winner = max(ids, key=self.priority)
                for j in ids:
                    if j != winner:
                        final[j] = cur[j]

        # (C) Edge-swap conflicts: block the lower-priority agent
        for i in range(self.N):
            for j in range(i+1, self.N):
                if final[i] == cur[j] and final[j] == cur[i] and final[i] != cur[i]:
                    if self.priority(i) >= self.priority(j):
                        final[j] = cur[j]
                    else:
                        final[i] = cur[i]

        # (D) Enforce reservations at tick t
        for cell, owner in self.reservations.get(t, {}).items():
            for i, nxt in list(final.items()):
                if nxt == cell and i != owner:
                    final[i] = cur[i]

        return final

    def step(self, t: int):
        prop = self.propose(t)
        final = self.resolve(t, prop)
        # Force owners into reserved cells (if adjacent)
        for cell, owner in self.reservations.get(t, {}).items():
            cur = self.pos[owner]
            if cell in neighbors(self.g, cur):
                final[owner] = cell
        # Commit
        for a in self.agents:
            i = a["id"]; curp = self.pos[i]; nxt = final[i]
            if nxt != curp:
                self.pos[i] = nxt
                p = self.paths[i]
                if p is not None and self.ptr[i]+1 < len(p) and p[self.ptr[i]+1] == nxt:
                    self.ptr[i] += 1
            if self.pos[i] == a["goal"]:
                self.at_goal.add(i)
        self.timeline.append(dict(self.pos))

    def run(self):
        for t in range(self.max_steps):
            if all(self.pos[a["id"]] == a["goal"] for a in self.agents):
                break
            self.step(t)
        return self.timeline, self.events

def print_positions_only(timeline: List[Dict[int, Pos]], agents: List[Dict], limit=20):
    labels = {a["id"]: a["label"] for a in agents}
    for t, pos in enumerate(timeline[:limit]):
        coords = {labels[i]: pos[i] for i in pos}
        collision = len(set(coords.values())) < len(coords.values())
        print(f"t={t}: {coords} {'<-- COLLISION' if collision else ''}")
