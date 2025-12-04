
import numpy as np
import torch
import time
import copy

import torch, torch.nn.functional as F
from ppo.ppo import PPOActorCritic

import heapq
import itertools


from utils.env_utils import process_state, state_to_key

# Try to import the C++ A* module
try:
    import cpp_astar
    HAS_CPP_ASTAR = True
except ImportError:
    HAS_CPP_ASTAR = False

def manhattan_distance(pos1, pos2) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def plan_with_search(env, model, device,
                     search_type: str = "astar",
                     algo: str       = "ppo",
                     timeout: float  = 30.0,
                     heuristic_weight: float = 1.0,
                     max_expansions: int     = 5000):
    """
    Dispatch to different planners:
      - "rollout"    → greedy_policy_rollout
      - "beam"       → beam_search
      - "greedy-bfs" → greedy_best_first
      - "astar"      → astar_dqn(_ppo)
      - "astar-cpp"  → astar_cpp (C++ geometric A*)
    """
    st = search_type.lower()

    if st in ("greedy-bfs", "bfs"):
        if algo.lower() == "dqn":
            return bfs_dqn(env, model, device, max_expansions=max_expansions)
        else:
            return bfs_ppo(env, model, device, max_expansions=max_expansions)

    # NEW: pure geometric A* via C++ (no model needed)
    if st in ("astar-cpp", "astar_grid", "astar-grid"):
        # We ignore 'model' and 'algo' here; this is a pure grid search.
        return astar_cpp(
            env,
            timeout=timeout,
            heuristic_weight=heuristic_weight,
            max_expansions=max_expansions if max_expansions is not None else 500000,
        )

    if st in ("astar",):
        if algo.lower() == "dqn":
            return astar_dqn(
                env, model,
                timeout=timeout,
                heuristic_weight=heuristic_weight
            )
        else:
            return astar_ppo(
                env, model, device,
                timeout=timeout,
                heuristic_weight=heuristic_weight
            )
    raise ValueError(f"Unknown search_type {search_type}")


def astar(initial_env, timeout= 30.0, heuristic_weight= 2.0 ):

    start_time = time.time()
    
    env = initial_env.env 
    
    if env.agent_pos is None or env.goal_pos is None:
        return None

    start_node_pos = tuple(map(int, env.agent_pos))
    goal_node_pos = tuple(map(int, env.goal_pos))

    if start_node_pos == goal_node_pos:
        return [] 

    h_initial = manhattan_distance(start_node_pos, goal_node_pos)
    unique_counter = itertools.count() 
    
    frontier = []
    heapq.heappush(frontier, (0 + heuristic_weight * h_initial, 0, next(unique_counter), start_node_pos, []))
    
    visited_positions = {start_node_pos}
    g_scores = {start_node_pos: 0}

    while frontier:
        if time.time() - start_time > timeout:
            return None

        f, g, _, current_pos, current_actions_path = heapq.heappop(frontier)

        if current_pos == goal_node_pos:
            return current_actions_path 
        
        for action_idx in range(4): 
            movements = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)} 
            dr, dc = movements[action_idx]
            next_pos_r, next_pos_c = current_pos[0] + dr, current_pos[1] + dc
            next_pos = (next_pos_r, next_pos_c)

            if not (0 <= next_pos_r < env.rows and 0 <= next_pos_c < env.cols):
                continue
            if env.grid[next_pos_r, next_pos_c] == -1:
                continue

            new_g_score = g + 1 

            if next_pos not in visited_positions or new_g_score < g_scores.get(next_pos, float('inf')):
                visited_positions.add(next_pos)
                g_scores[next_pos] = new_g_score
                h_score = manhattan_distance(next_pos, goal_node_pos)
                f_score = new_g_score + heuristic_weight * h_score
                
                new_actions_path = current_actions_path + [action_idx]
                heapq.heappush(frontier, (f_score, new_g_score, next(unique_counter), next_pos, new_actions_path))
                
    return None


def astar_cpp(initial_env,
              timeout: float = 30.0,
              heuristic_weight: float = 1.0,
              max_expansions: int = 500000):
    """
    Call the C++ A* (cpp_astar.astar_grid) on our GridEnvWrapper.

    initial_env: GridEnvWrapper instance (same as plan_with_search expects).
    Returns:
        List[int] of actions (0..3) or None if no path / module missing.
    """
    if not HAS_CPP_ASTAR:
        # Fall back to Python A* if C++ module is not available
        print("Warning: cpp_astar module not available. Falling back to Python A*.")
        return astar(initial_env, timeout=timeout, heuristic_weight=heuristic_weight)

    env = initial_env.env

    if env.agent_pos is None or env.goal_pos is None:
        return None

    start_r, start_c = map(int, env.agent_pos)
    goal_r, goal_c = map(int, env.goal_pos)

    if (start_r, start_c) == (goal_r, goal_c):
        return []

    # Convert grid to a nested Python list of ints for cpp_astar
    # Assumes obstacles are marked with -1 (adjust if your convention differs)
    grid = env.grid.astype(np.int32).tolist()

    # Map timeout to max_expansions conservatively, if you like:
    # e.g., max_expansions ~= rows * cols * 10
    H, W = env.grid.shape
    if max_expansions is None:
        max_expansions = H * W * 10

    actions = cpp_astar.astar_grid(
        grid,
        int(start_r),
        int(start_c),
        int(goal_r),
        int(goal_c),
        int(max_expansions),
        float(heuristic_weight),
    )

    if actions is None:
        return None
    if len(actions) == 0 and (start_r, start_c) != (goal_r, goal_c):
        # C++ returned empty path, treat as failure
        return None

    return actions


def astar_dqn(initial_env, model, device='mps', timeout=30.0, heuristic_weight=2.0):
    """
    Performs an A* search guided by the RL policy to find a plan (sequence of actions)
    that leads to a goal state (where done and goal_flag are True).

    Each step has a cost of 1. The heuristic h(n) is computed as:
         h(n) = heuristic_weight * ( - max(Q(s,a)) )
    so that states with higher predicted returns (higher Q-values) have a lower (better) heuristic.

    Parameters:
        initial_env: a deepcopy of the initial environment with its current_observation set.
        model: the trained DQN model.
        timeout: maximum time (in seconds) to search before giving up.
        heuristic_weight: weight for the heuristic value relative to the step cost.
        
    Returns:
        A list of actions (the plan) if a goal state is reached; otherwise, None.
    """
    start_time = time.time()

    # Compute the heuristic for the initial state.
    obs_tensor = process_state(initial_env.current_observation, device)
    with torch.no_grad():
        q_values = model(obs_tensor)
        # Here higher Q is better so we take the negative. Multiply by heuristic_weight.
        h = heuristic_weight * (-max(q_values[0].tolist()))
    
    counter = itertools.count()  # Unique counter for tie-breaking.

    # Priority queue (min-heap): each element is a tuple (f, g, counter, env, plan)
    # where f = g + h.
    frontier = []
    heapq.heappush(frontier, (0 + h, 0, next(counter), copy.deepcopy(initial_env), []))
    
    visited = set()
    
    while frontier:
        # Check timeout.
        if time.time() - start_time > timeout:
            break
        
        f, g, _, current_env, current_path = heapq.heappop(frontier)
        key = state_to_key(current_env.current_observation)
        if key in visited:
            continue
        visited.add(key)
        
        if len(np.argwhere(current_env.current_observation['grid'] == 1)) == 0:
            continue

        obs_tensor = process_state(current_env.current_observation, device)
        with torch.no_grad():
            q_values = model(obs_tensor)
        
        actions = current_env.get_actions()
        # Sort actions solely by the Q-value (highest Q first). No special bonus now.
        actions_sorted = sorted(actions, key=lambda a: q_values[0, a].item(), reverse=True)
        
        for action in actions_sorted:
            sim_env = copy.deepcopy(current_env)
            next_obs, reward, done, goal_flag, _ = sim_env.step(action)
            sim_env.current_observation = next_obs
            
            new_path = current_path + [action]
            if done and goal_flag:
                return new_path
            
            g_new = g + 1  # Each step has a cost of 1.
            
            obs_tensor_new = process_state(sim_env.current_observation, device)
            with torch.no_grad():
                q_new = model(obs_tensor_new)
            h_new = heuristic_weight * (-max(q_new[0].tolist()))
            
            f_new = g_new + h_new
            heapq.heappush(frontier, (f_new, g_new, next(counter), sim_env, new_path))
    
    return None


def astar_ppo(initial_env,
                               model: PPOActorCritic,
                               device,
                               timeout: float = 30.0,
                               heuristic_weight: float = 1.0):
    """
    A* search where the heuristic is −V(s) from PPO's critic, and expansion
    order is by π(a|s) from PPO's actor.
    """
    start = time.time()
    counter = itertools.count()
    
    # initial h
    obs0 = process_state(initial_env.current_observation, device)
    with torch.no_grad():
        out = model(obs0, None, None)
        # out = (logits, value, _, _)
        value0 = out[1].item()
    h0 = heuristic_weight * (- value0)
    
    # frontier: (f = g+h, g, tie, env_copy, path)
    frontier = [(0 + h0, 0, next(counter), copy.deepcopy(initial_env), [])]
    visited = set()
    
    while frontier:
        if time.time() - start > timeout: # write in seconds equivalent
            break
        
        f, g, _, env_cur, path = heapq.heappop(frontier)

        key = state_to_key(env_cur.current_observation)
        if key in visited:
            continue
        visited.add(key)

        if len(np.argwhere(env_cur.current_observation['grid'] == 1)) == 0:
            continue
        # print(env_cur.current_observation['grid'])
        # evaluate policy & value
        obs_t = process_state(env_cur.current_observation, device)
        # print('-', obs_t['grid'])
        with torch.no_grad():
            logits, value, _, _ = model(obs_t, None, None)
            logits = logits.squeeze(0)
            value  = value.item()
        # policy probabilities
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        
        # valid actions
        valid = env_cur.get_actions()
        # sort descending by policy prob
        actions_sorted = sorted(valid, key=lambda a: probs[a], reverse=True)

        for a in actions_sorted:
            env_next = copy.deepcopy(env_cur)
            obs1, reward, done, goal_flag, _ = env_next.step(a)
            env_next.current_observation = obs1
            new_path = path + [a]
            
            if done and goal_flag:
                return new_path
            
            g1 = g + 1
            # compute heuristic at new state
            obs1_t = process_state(obs1, device)
            with torch.no_grad():
                _, v1, _, _ = model(obs1_t, None, None)
                v1 = v1.item()
            h1 = heuristic_weight * (- v1)
            
            f1 = g1 + h1
            heapq.heappush(frontier, (f1, g1, next(counter), env_next, new_path))
    
    # failed to find within timeout
    return None


def bfs_ppo(env, model, device, max_expansions=5000):
    counter = itertools.count()
    expansions = 0

    def h(e):
        obs = process_state(e.current_observation, device)
        with torch.no_grad():
            _, v, _, _ = model(obs, None, None)
        return -v.item()

    frontier = [(h(env), next(counter), copy.deepcopy(env), [])]
    visited = set()

    while frontier and expansions < max_expansions:
        _, _, cur_env, path = heapq.heappop(frontier)
        key = state_to_key(cur_env.current_observation)
        if key in visited:
            continue
        visited.add(key)
        expansions += 1

        # Goal check
        if hasattr(cur_env.env, 'agent_pos') and cur_env.env.agent_pos == cur_env.env.goal_pos:
            return path

        obs = process_state(cur_env.current_observation, device)
        with torch.no_grad():
            logits, _, _, _ = model(obs, None, None)
        probs = F.softmax(logits.squeeze(0), dim=-1).cpu().numpy()

        for a in cur_env.get_actions():
            nxt = copy.deepcopy(cur_env)
            obs1, _, done, goal_flag, _ = nxt.step(a)
            nxt.current_observation = obs1
            new_path = path + [a]
            if done and goal_flag:
                return new_path
            heapq.heappush(frontier, (h(nxt), next(counter), nxt, new_path))

    return None

def bfs_dqn(env, model, device, max_expansions=5000):
    """
    Greedy best‐first search guided by DQN Q‐values:
      - Heuristic h(s) = - max_a Q(s,a)
      - Expands up to max_expansions nodes.
    """
    def h(e):
        obs = process_state(e.current_observation, device)
        with torch.no_grad():
            q_vals = model(obs).squeeze(0)
        return -q_vals.max().item()

    counter = itertools.count()
    frontier = [(h(env), next(counter), copy.deepcopy(env), [])]
    visited = set()
    expansions = 0

    while frontier and expansions < max_expansions:
        _, _, cur_env, path = heapq.heappop(frontier)
        key = state_to_key(cur_env.current_observation)
        if key in visited:
            continue
        visited.add(key)
        expansions += 1

        # Goal check (assumes done & goal_flag in observation tuple or attributes)
        obs = cur_env.current_observation
        if obs.get('done', False) and obs.get('goal_flag', False):
            return path

        for a in cur_env.get_actions():
            nxt = copy.deepcopy(cur_env)
            obs1, _, done, goal_flag, _ = nxt.step(a)
            nxt.current_observation = obs1
            new_path = path + [a]
            if done and goal_flag:
                return new_path
            heapq.heappush(frontier, (h(nxt), next(counter), nxt, new_path))

    return None



def rl_guided_astar_with_reservations(initial_env,
                                      model,
                                      device,
                                      reservation_table: set,
                                      timeout: float = 10.0,
                                      heuristic_weight: float = 1.0):
    """
    Reservation‐table A* guided by PPO: critic for heuristic, actor for expansion order.
    reservation_table: set of ((row,col), t) pairs that cannot be occupied at time t.
    """
    start_time = time.time()
    counter = itertools.count()

    # initial heuristic
    obs0 = process_state(initial_env.current_observation, device)
    with torch.no_grad():
        _, v0, _, _ = model(obs0, None, None)
    h0 = heuristic_weight * (-v0.item())

    frontier = [(h0, 0, next(counter), copy.deepcopy(initial_env), [])]
    visited = set()

    while frontier:
        if time.time() - start_time > timeout:
            break

        f, g, _, env_cur, path = heapq.heappop(frontier)

        cur_pos = env_cur.env.agent_pos
        # forbid reserved occupancy
        if (cur_pos, g) in reservation_table:
            continue

        # avoid revisiting same state at same time
        key = (state_to_key(env_cur.current_observation), g)
        if key in visited:
            continue
        visited.add(key)

        # goal check: compare agent_pos to env.goal_pos
        if cur_pos == tuple(env_cur.env.goal_pos):
            return path

        # evaluate policy and value
        obs_t = process_state(env_cur.current_observation, device)
        with torch.no_grad():
            logits, v_t, _, _ = model(obs_t, None, None)
        logits = logits.squeeze(0)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        valid_actions = env_cur.get_actions()
        for a in sorted(valid_actions, key=lambda a: probs[a], reverse=True):
            env_next = copy.deepcopy(env_cur)
            obs1, reward, done, goal_flag, _ = env_next.step(a)
            env_next.current_observation = obs1
            new_pos = env_next.env.agent_pos
            g1 = g + 1

            # reservation at next time
            if (new_pos, g1) in reservation_table:
                continue

            new_path = path + [a]
            # immediate check using env.goal_pos
            if new_pos == tuple(env_next.env.goal_pos):
                return new_path

            # compute heuristic
            obs1_t = process_state(obs1, device)
            with torch.no_grad():
                _, v1, _, _ = model(obs1_t, None, None)
            h1 = heuristic_weight * (-v1.item())
            f1 = g1 + h1
            heapq.heappush(frontier, (f1, g1, next(counter), env_next, new_path))

    return None

