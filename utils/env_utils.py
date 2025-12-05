
import numpy as np
import torch

def process_state(obs, device):
    """
    Convert an observation (with keys 'grid', 'direction', 'distance')
    into a dictionary of batched tensors.
    """
    grid = torch.FloatTensor(np.array(obs['grid'])).unsqueeze(0).to(device)
    direction = torch.FloatTensor(np.array(obs['direction'])).unsqueeze(0).to(device)
    distance = torch.FloatTensor(np.array(obs['distance'])).unsqueeze(0).to(device)
    return {'grid': grid, 'direction': direction, 'distance': distance}

def state_to_key(state):
    """
    Convert the observation dictionary into a hashable key.
    This simple implementation flattens the 'grid', 'direction', and 'distance'
    into tuples. Adjust if your state structure changes.
    """
    grid_key = tuple(np.array(state['grid']).flatten())
    direction_key = tuple(np.array(state['direction']).flatten())
    distance_key = tuple(np.array(state['distance']).flatten())
    return (grid_key, direction_key, distance_key)

def simulate_plan(env_instance, plan):
    """
    Simulate the plan on a copy of the environment and record the trajectory (agent positions).
    A wait action (4) causes no movement (i.e. the agent remains in its current cell).

    Returns the trajectory if valid, or None if the plan is invalid (moves out of bounds or into obstacles).
    """

    movements = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
        4: (0, 0)    # Wait
    }

    # Get grid for validation
    grid = env_instance.env.grid
    rows, cols = grid.shape

    trajectory = []
    current_pos = tuple(map(int, env_instance.env.agent_pos))
    trajectory.append(current_pos)

    for action_idx, action in enumerate(plan):
        # Validate action is in range
        if action not in movements:
            print(f"  [simulate_plan] Invalid action {action} at step {action_idx}")
            return None

        # Get the movement
        move = movements[action]
        new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

        # BUGFIX: Validate new position
        r, c = new_pos

        # Check bounds
        if not (0 <= r < rows and 0 <= c < cols):
            print(f"  [simulate_plan] Move out of bounds at step {action_idx}: {current_pos} + {move} = {new_pos}")
            return None

        # Check obstacle
        if grid[r, c] == -1:
            print(f"  [simulate_plan] Move into obstacle at step {action_idx}: {new_pos}")
            return None

        current_pos = new_pos
        trajectory.append(current_pos)

    return trajectory


def pos_at(agent_id, t, trajectories, agent_starts):
    """
    Get agent position at timestep t with 'stay-at-last-position' semantics.
    This is the unified helper function used by all collision detection functions.

    Args:
        agent_id: 1-indexed agent ID
        t: timestep
        trajectories: list of trajectories (indexed by agent_id - 1)
        agent_starts: list of starting positions (indexed by agent_id - 1)

    Returns:
        Position tuple (row, col) as integers
        - If agent has a trajectory, returns position at t (or last position if t >= trajectory length)
        - If agent has no trajectory, returns starting position
    """
    idx = agent_id - 1
    traj = trajectories[idx]
    if traj:
        if t < len(traj):
            return tuple(map(int, traj[t]))
        return tuple(map(int, traj[-1]))  # Stay at last position
    return tuple(map(int, agent_starts[idx]))  # No trajectory -> use start


# --- Multi-Agent Planning ---

# Try to import C++ collision module
try:
    import cpp_collision
    HAS_CPP_COLLISION = True
except ImportError:
    HAS_CPP_COLLISION = False

def analyze_collisions(trajectories, goal_positions, agent_starts=None, static_grid=None, verbose=False):
    """
    Detect vertex and edge (swap) collisions across all agents.

    - Trajectories are lists of positions; after the final position the agent is assumed
      to remain stationary.
    - Vertex collisions: any time t where two agents occupy the same cell.
    - Edge collisions (swaps): any time interval t→t+1 where two agents swap cells.
    - If static_grid is provided, also reports collisions with static obstacles (-1).
    - agent_starts: starting positions for agents (required for collision detection of agents
      without trajectories). If None, uses empty check fallback behavior (legacy mode).
    """
    # Use C++ implementation if available and compatible
    # C++ implementation currently supports vertex and edge collisions
    # It does NOT support static obstacle collisions yet (though we could add it)
    # So if static_grid is provided, we must use Python fallback or extend C++
    # The C++ implementation signature is: analyze_collisions_grid_time(trajectories, goals, starts)
    if HAS_CPP_COLLISION and static_grid is None and agent_starts is not None:
        # Convert goals to list of tuples if needed (usually they are)
        # C++ expects list of (r, c) tuples
        return cpp_collision.analyze_collisions_grid_time(trajectories, goal_positions, agent_starts)

    collisions = []

    if not trajectories:
        return collisions

    max_steps = max((len(traj) for traj in trajectories if traj), default=0)
    if max_steps == 0:
        return collisions

    # Static obstacle collisions
    if static_grid is not None:
        for agent_idx, traj in enumerate(trajectories):
            for t, pos in enumerate(traj):
                r, c = int(pos[0]), int(pos[1])
                if 0 <= r < static_grid.shape[0] and 0 <= c < static_grid.shape[1]:
                    if static_grid[r, c] == -1:
                        col = {
                            'time': t,
                            'cell': (int(pos[0]), int(pos[1])),
                            'agents': [agent_idx + 1],
                            'type': 'obstacle',
                            'obstacle': True
                        }
                        collisions.append(col)
                        if verbose:
                            print(f"[analyze_collisions] Obstacle collision at t={t}: agent={agent_idx+1}, cell={col['cell']}")

    # Vertex collisions
    for t in range(max_steps):
        for i in range(len(trajectories)):
            # Use shared pos_at() helper if agent_starts provided, else fallback to old behavior
            if agent_starts is not None:
                pos_i = pos_at(i + 1, t, trajectories, agent_starts)
            else:
                # Legacy mode: skip agents without trajectories
                if not trajectories[i]:
                    continue
                pos_i = tuple(map(int, trajectories[i][t] if t < len(trajectories[i]) else trajectories[i][-1]))

            for j in range(i + 1, len(trajectories)):
                # Use shared pos_at() helper if agent_starts provided, else fallback to old behavior
                if agent_starts is not None:
                    pos_j = pos_at(j + 1, t, trajectories, agent_starts)
                else:
                    # Legacy mode: skip agents without trajectories
                    if not trajectories[j]:
                        continue
                    pos_j = tuple(map(int, trajectories[j][t] if t < len(trajectories[j]) else trajectories[j][-1]))

                if pos_i == pos_j:
                    col = {
                        'time': t,
                        'cell': pos_i,
                        'agents': [i + 1, j + 1],
                        'type': 'vertex'
                    }
                    collisions.append(col)
                    if verbose:
                        print(f"[analyze_collisions] Vertex collision at t={t}: agents={col['agents']}, cell={pos_i}")

    # Edge collisions (swaps)
    for t in range(max_steps - 1):
        for i in range(len(trajectories)):
            # Use shared pos_at() helper if agent_starts provided, else fallback to old behavior
            if agent_starts is not None:
                pos_i_t = pos_at(i + 1, t, trajectories, agent_starts)
                pos_i_tp1 = pos_at(i + 1, t + 1, trajectories, agent_starts)
            else:
                # Legacy mode: skip agents without trajectories
                if not trajectories[i]:
                    continue
                pos_i_t = tuple(map(int, trajectories[i][t] if t < len(trajectories[i]) else trajectories[i][-1]))
                pos_i_tp1 = tuple(map(int, trajectories[i][t + 1] if t + 1 < len(trajectories[i]) else trajectories[i][-1]))

            for j in range(i + 1, len(trajectories)):
                # Use shared pos_at() helper if agent_starts provided, else fallback to old behavior
                if agent_starts is not None:
                    pos_j_t = pos_at(j + 1, t, trajectories, agent_starts)
                    pos_j_tp1 = pos_at(j + 1, t + 1, trajectories, agent_starts)
                else:
                    # Legacy mode: skip agents without trajectories
                    if not trajectories[j]:
                        continue
                    pos_j_t = tuple(map(int, trajectories[j][t] if t < len(trajectories[j]) else trajectories[j][-1]))
                    pos_j_tp1 = tuple(map(int, trajectories[j][t + 1] if t + 1 < len(trajectories[j]) else trajectories[j][-1]))

                if pos_i_t == pos_j_tp1 and pos_i_tp1 == pos_j_t:
                    col = {
                        'time': t + 1,
                        'cell': (pos_i_t, pos_i_tp1),
                        'agents': [i + 1, j + 1],
                        'type': 'edge'
                    }
                    collisions.append(col)
                    if verbose:
                        print(f"[analyze_collisions] Edge swap at t={t}->{t+1}: agents={[i+1,j+1]}, cells={col['cell']}")

    return collisions
