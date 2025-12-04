import numpy as np
import time
import copy
import random
import json

from collections import defaultdict, deque

from utils.env_utils import analyze_collisions, simulate_plan
from utils.search_utils import plan_with_search
from utils.yield_utils import find_deep_yield_cells, select_best_yield_cell
from utils.graph_utils import (compute_cell_degrees, build_blocking_graph, 
                                find_cycles, choose_pivot_agent, compute_flow_map)
from utils.corridor_utils import detect_corridors, classify_cell, assign_corridor_order
from utils.reservation_utils import (build_reservation_table, compute_cell_occupancy_score,
                                      estimate_blocker_eta, select_yield_cell_with_reservation)
from utils.central_coordinator import CentralCoordinator

class InfoSharingTracker:
    """A helper class to track and quantify information sharing events."""
    def __init__(self):
        # IU count for the initial S1 -> S2 path submissions
        self.initial_submission_iu = 0
        
        # IU count for all S3 -> S4 alerts
        self.alert_iu = 0
        
        # IU count for all S4 -> S2 revised path submissions
        self.revised_submission_iu = 0
        
        # More granular details for the alerts
        self.alert_details_iu = {
            'static': 0,
            'dynamic': 0,
            'pivot': 0,
            'defer': 0
        }

    def record_initial_submission(self, initial_trajectories):
        """Records the IU from the initial submission of all agent paths."""
        self.initial_submission_iu = sum(len(t) for t in initial_trajectories if t)

    def record_static_alert(self, forbidden_cells):
        """Records IU from a static alert. IU = number of forbidden cells."""
        iu = len(forbidden_cells)
        self.alert_iu += iu
        self.alert_details_iu['static'] += iu

    def record_dynamic_alert(self, dynamic_path):
        """Records IU from a dynamic alert. IU = length of the provided sub-path."""
        iu = len(dynamic_path)
        self.alert_iu += iu
        self.alert_details_iu['dynamic'] += iu
    
    def record_pivot_alert(self):
        """Records IU from a pivot alert. IU = 1 (for the pivot cell)."""
        self.alert_iu += 1
        self.alert_details_iu['pivot'] += 1

    def record_defer_alert(self):
        """Records IU from a deferral command. IU = 1 (for the command)."""
        self.alert_iu += 1
        self.alert_details_iu['defer'] += 1

    def record_revised_submission(self, new_trajectory):
        """Records the IU from an agent submitting a revised path."""
        self.revised_submission_iu += len(new_trajectory)

    @property
    def total_iu(self):
        """Calculates the total information load."""
        return self.initial_submission_iu + self.alert_iu + self.revised_submission_iu

    def report(self):
        """Prints a formatted summary to the terminal."""
        print("\n--- Information Sharing Metrics ---")
        print(f"  - Initial Path Submission IU: {self.initial_submission_iu}")
        print(f"  - Revised Path Submission IU: {self.revised_submission_iu}")
        print(f"  - Conflict Alert IU:          {self.alert_iu}")
        print(f"    - Static Alerts:      {self.alert_details_iu['static']} IU")
        print(f"    - Dynamic Alerts:     {self.alert_details_iu['dynamic']} IU")
        print(f"    - Pivot Alerts:       {self.alert_details_iu['pivot']} IU")
        print(f"    - Defer Alerts:       {self.alert_details_iu['defer']} IU")
        print("-----------------------------------")
        print(f"  Total Information Load (IU):  {self.total_iu}")
        print("-----------------------------------")

    def to_dict(self):
        """Converts the metrics to a dictionary for JSON serialization."""
        return {
            'initialSubmissionIU': self.initial_submission_iu,
            'revisedSubmissionIU': self.revised_submission_iu,
            'conflictAlertIU': self.alert_iu,
            'alertDetailsIU': self.alert_details_iu,
            'totalInformationLoadIU': self.total_iu
        }

def _handle_active_collision_passes(
    current_agent_plans, current_trajectories, agent_envs, model, run_counters, device,
    agent_goals, agent_starts, pristine_static_grid, cell_degrees, corridor_map, coordinator,
    yield_contracts, deep_yield_count, forbidden_regions,
    static_block_hist, dyn_obs_hist, pivot_attempts_hist, deferred_agents,
    coll_specific_attempts, unique_colls_ever,
    replan_strategy, search_type, algo, timeout, heuristic_weight, max_expansions,
    max_passes_inner, time_limit, overall_start_time, overall_fix_attempt_count, verbose,
    # --- LOGGING ARGUMENTS ---
    log_data, log_context,
    # Constants passed down
    init_rewind_val, max_rewind_static_val, max_rewind_dyn_val, info_setting="all", info_tracker=None
):
    """
    Handles multiple passes to resolve collisions among non-deferred agents.
    Returns True if a timeout occurred within these passes, False otherwise.
    Modifies plans, trajectories, histories, and deferred_agents set in place.
    info_setting: "all" for all strategies, 
                  "no" for only static obstacle strategy (no deferral),
                  "only_dyn" for only dynamic obstacle strategy (no deferral).
                  Deferral is only active with "all".
    """
    num_agents = len(current_agent_plans)

    def cell_key(c): # Local helper
        if isinstance(c, (list, np.ndarray)):
            try: c = tuple(map(int,c))
            except (ValueError, TypeError): return tuple(cell_key(e) for e in c) if hasattr(c, '__iter__') else c
        if isinstance(c, tuple): return tuple(cell_key(e) for e in c)
        try: return int(c)
        except (TypeError, ValueError): return c

    for pass_num in range(1, max_passes_inner + 1):
        current_elapsed_time = time.perf_counter() - overall_start_time
        if current_elapsed_time > time_limit:
            if verbose: print(f"Inner Passes: Overall time limit {time_limit}s reached during pass {pass_num}.")
            return True # Timeout occurred

        # --- LOGGING: Start of a pass ---
        pass_log = {"pass_number": pass_num, "events": []}
        log_data["execution_trace"]["attempts"][-1]["passes"].append(pass_log)
        # --------------------------------

        if verbose: print(f"\n--- Inner Pass {pass_num} of Overall Attempt #{overall_fix_attempt_count} (Elapsed {current_elapsed_time:.2f}s) ---")
        any_fix_this_pass_iteration = False

        # COOPERATIVE PROTOCOLS: Manage Contracts and Forbidden Regions
        # 1. Clear forbidden regions for this pass (rebuilt from active contracts)
        forbidden_regions.clear()
        
        # 2. Check and update yield contracts
        expired_contracts = []
        for aid, contract in yield_contracts.items():
            is_expired = False
            
            # Check expiration conditions
            if contract["valid_until"]["type"] == "pass_count":
                # Expire if current pass exceeds creation + duration
                # We use overall_fix_attempt_count as a proxy for "time" or just pass_num if local
                # Let's use pass_num for local duration within this fix attempt
                # But contracts might span multiple fix attempts? 
                # For now, assume contracts are valid within a fix_collisions call.
                if pass_num >= contract["created_at_pass"] + contract["valid_until"]["value"]:
                    is_expired = True
            
            elif contract["valid_until"]["type"] == "agent_passed":
                # Expire if all target agents are at goal
                all_passed = True
                for target_aid in contract["valid_until"]["value"]:
                    target_idx = target_aid - 1
                    # Check if at goal
                    if not (tuple(map(int, current_trajectories[target_idx][-1])) == tuple(map(int, agent_goals[target_idx]))):
                        all_passed = False; break
                if all_passed:
                    is_expired = True
            
            if is_expired:
                expired_contracts.append(aid)
                if verbose: print(f"  Contract for Agent {aid} expired.")
            else:
                # Contract active: Enforce region
                # Add to forbidden_regions
                if contract["region"]["type"] == "cells":
                    forbidden_regions[aid-1].update(contract["region"]["cells"])
                elif contract["region"]["type"] == "corridor":
                    cid = contract["region"]["corridor_id"]
                    if cid in corridor_map:
                        forbidden_regions[aid-1].update(corridor_map[cid]["cells"])
        
        # Remove expired contracts
        for aid in expired_contracts:
            del yield_contracts[aid]
            # Also decrement deep_yield_count? Maybe not, that's a rate limiter.


        all_detected_collisions_this_pass = analyze_collisions(current_trajectories, agent_goals, pristine_static_grid)
        run_counters['collisions_total'] += len(all_detected_collisions_this_pass)

        active_colls_this_pass = []
        for coll_item in all_detected_collisions_this_pass:

            # --- LOGGING: Collision Detection (S2) ---
            coll_id = f"Collision-{coll_item['time']}-{coll_item['cell']}".replace(" ", "")
            if not any(c['id'] == coll_id for c in log_data['collisionEvents']):
                    # Handle different collision types
                    if coll_item['type'] == 'obstacle':
                        location = [int(coll_item['cell'][0]), int(coll_item['cell'][1])]
                    elif coll_item['type'] == 'vertex':
                        location = [int(x) for x in coll_item['cell']]
                    else:  # edge
                        location = [[int(c[0]), int(c[1])] for c in coll_item['cell']]
                    
                    log_data['collisionEvents'].append({
                    "id": coll_id,
                    "time": coll_item['time'],
                    "type": coll_item['type'],
                    "location": location,
                    "agents": [f"Robot-{aid}" for aid in coll_item['agents']]
                })

            is_only_between_deferred_agents_at_start = True
            for agent_id_plus_1 in coll_item['agents']:
                agent_idx = agent_id_plus_1 - 1

                if agent_idx not in deferred_agents:
                    is_only_between_deferred_agents_at_start = False; break

                # Check if agent is at its start and the collision involves that start position at that time
                # Ensure trajectory exists and collision time is valid
                if not (current_trajectories[agent_idx] and \
                        0 <= coll_item['time'] < len(current_trajectories[agent_idx]) and \
                        tuple(map(int,current_trajectories[agent_idx][coll_item['time']])) == agent_starts[agent_idx]):
                    is_only_between_deferred_agents_at_start = False; break
            
            if not is_only_between_deferred_agents_at_start:
                active_colls_this_pass.append(coll_item)
        current_colls_to_process = active_colls_this_pass

        if not current_colls_to_process:
            if verbose: print("No active collisions to process in this inner pass.")
            return False # No timeout, but no active collisions to process in this inner loop

        if verbose: print(f"Detected {len(current_colls_to_process)} active collisions to process in this inner pass.")
        for c_outer in current_colls_to_process:
            unique_colls_ever.add((c_outer['time'], c_outer['type'], cell_key(c_outer['cell']), frozenset(c_outer['agents'])))

        sorted_colls_to_process = sorted(current_colls_to_process, key=lambda c: (c['time'], min(c['agents'])))

        for coll in sorted_colls_to_process:
            static_success, dyn_success, pivot_success = False, False, False
            coll_key_tuple = (coll['time'], coll['type'], cell_key(coll['cell']), frozenset(coll['agents']))

            active_colls_recheck_inner = analyze_collisions(current_trajectories, agent_goals, pristine_static_grid)
            is_coll_still_present_inner = any(
                (item['time'], item['type'], cell_key(item['cell']), frozenset(item['agents'])) == coll_key_tuple
                for item in active_colls_recheck_inner )
            if not is_coll_still_present_inner:
                 if verbose: print(f"  Skipping {coll_key_tuple}: resolved by prior action in this inner pass.")
                 continue

            coll_specific_attempts[coll_key_tuple] += 1
            coll_attempt_num = coll_specific_attempts[coll_key_tuple]
            coll_t, coll_type_str, coll_cell_orig = coll['time'], coll['type'], coll['cell']
            current_coll_active_agents_list = [aid for aid in sorted(list(coll['agents'])) if (aid - 1) not in deferred_agents]

            if not current_coll_active_agents_list: continue
            if verbose: print(f"\n-- Handling Collision (Inner Pass): T={coll_t},Type={coll_type_str},Cell={coll_cell_orig},Agents={current_coll_active_agents_list},Attempt={coll_attempt_num}")

            # Initialize reference variables to avoid UnboundLocalError if strategies are skipped
            cell_str_ref = str(coll['cell']).replace(" ", "")
            coll_id_ref = f"Collision-{coll['time']}-{cell_str_ref}"
            log_context["alert_id_counter"] += 1
            alert_id = f"Alert-{log_context['alert_id_counter']}"
            log_context["strategy_id_counter"] += 1
            strategy_id = f"Strategy-{log_context['strategy_id_counter']}"

            # ==================================================================================
            # TIER 0: CORRIDOR ORDERING ENFORCEMENT
            # ==================================================================================
            # Check if collision is in a corridor and enforce ordering
            # Handle vertex vs edge collisions
            target_corridor_id = None
            
            if coll_type_str == 'vertex':
                coll_cell_tuple = tuple(map(int, coll_cell_orig)) if isinstance(coll_cell_orig, (list, tuple, np.ndarray)) else coll_cell_orig
                cell_info = classify_cell(coll_cell_tuple, cell_degrees, corridor_map)
                if cell_info["type"] == "corridor":
                    target_corridor_id = cell_info["corridor_id"]
            elif coll_type_str == 'edge':
                # coll_cell_orig is ((r1,c1), (r2,c2))
                c1 = tuple(map(int, coll_cell_orig[0]))
                c2 = tuple(map(int, coll_cell_orig[1]))
                info1 = classify_cell(c1, cell_degrees, corridor_map)
                info2 = classify_cell(c2, cell_degrees, corridor_map)
                
                # If both in same corridor, use it
                if info1["type"] == "corridor" and info2["type"] == "corridor" and info1["corridor_id"] == info2["corridor_id"]:
                    target_corridor_id = info1["corridor_id"]
                # If one is corridor and other is junction/open, maybe still enforce?
                # For now, strict same-corridor check
            
            if target_corridor_id is not None:
                corridor_id = target_corridor_id
                corridor_info = corridor_map[corridor_id]
                
                # Assign order for this corridor
                # We need blocked_cells_per_agent for this, which we might not have computed yet
                # Compute it locally if needed or reuse from previous steps if available
                # For now, let's compute a lightweight version or assume we can access it
                # To avoid recomputing full blocking graph, we'll just check current trajectories + goals
                
                # Quick check: who is in the corridor or wants to be?
                # For simplicity in this phase, we'll use the agents involved in collision + any in corridor
                
                # Get all agents currently in this corridor
                agents_in_corridor = []
                for aid in range(1, len(agent_goals) + 1):
                    idx = aid - 1
                    if current_trajectories[idx]:
                        # Check current position
                        curr_pos = current_trajectories[idx][min(coll_t, len(current_trajectories[idx])-1)]
                        if tuple(map(int, curr_pos)) in corridor_info["cells"]:
                            agents_in_corridor.append(aid)
                
                # Combine with collision agents
                relevant_agents = list(set(current_coll_active_agents_list + agents_in_corridor))
                
                # Simple ordering: Direction based
                # 1. Determine direction for each agent (start -> goal relative to corridor)
                # 2. Group and sort
                
                # Simplified ordering for now: Just pick one to own it
                # Prefer agent closest to exiting the corridor? Or agent with lower ID?
                # Let's use assign_corridor_order from utils if we can build the inputs
                # But building blocked_cells_per_agent is expensive.
                # Let's use a heuristic:
                # If one agent is moving in consistent direction and others are opposing, give priority to consistent one.
                
                # For this implementation, let's pick the agent with the lowest ID as the "owner" for now
                # and force others to yield. This is a basic protocol.
                # Better: Use the assign_corridor_order if we can.
                
                # Let's try to use the proper function by building a mini blocked_cells map
                mini_blocked_cells = {}
                for aid in relevant_agents:
                    idx = aid - 1
                    # Estimate desired path: simple A* on static grid
                    # This is expensive to do every time.
                    # Fallback: Use current trajectory as proxy for desired path
                    mini_blocked_cells[aid] = set(map(tuple, current_trajectories[idx]))
                
                ordered_agents = assign_corridor_order(
                    corridor_id, corridor_info, mini_blocked_cells, current_trajectories, agent_goals
                )
                
                if ordered_agents:
                    owner_agent = ordered_agents[0]
                    if verbose: print(f"    Corridor {corridor_id} owner: Agent {owner_agent}. Others must yield.")
                    
                    # Enforce: Mark corridor as forbidden for others
                    for aid in ordered_agents[1:]:
                        forbidden_regions[aid-1].update(corridor_info["cells"])
                        
                        # If non-owner is currently IN corridor, force them to leave
                        idx = aid - 1
                        curr_pos = current_trajectories[idx][min(coll_t, len(current_trajectories[idx])-1)]
                        if tuple(map(int, curr_pos)) in corridor_info["cells"]:
                            if verbose: print(f"    Agent {aid} is in corridor owned by {owner_agent}. Forcing exit.")
                            # We will let the subsequent logic handle the move, but we've set the forbidden region
                            # So static/dynamic replanning should naturally move them out.
                            # But to be sure, we can trigger a deep yield here if they are in collision
                            if aid in current_coll_active_agents_list:
                                # Force deep yield logic to run for this agent
                                # We can do this by setting a flag or prioritizing it
                                pass 
            
            # ==================================================================================
            # TIER 1: CHECK FOR FINISHED AGENTS (Goal Collisions)
            # ==================================================================================
            # If any agent in the collision is already at its goal, prioritize moving them aside.
            
            yield_success = False
            finished_agents_in_coll = []
            
            if info_setting == "all":
                for aid in current_coll_active_agents_list:
                    idx = aid - 1
                    traj = current_trajectories[idx]
                    if not traj: continue
                    
                    # Check if agent is effectively finished (at goal and staying)
                    is_finished = False
                    if coll_t >= len(traj) - 1:
                        is_finished = True
                    elif tuple(map(int, traj[coll_t])) == tuple(map(int, agent_goals[idx])):
                        # It's at goal, check if it stays there till end
                        if all(tuple(map(int, p)) == tuple(map(int, agent_goals[idx])) for p in traj[coll_t:]):
                            is_finished = True
                    
                    if is_finished:
                        finished_agents_in_coll.append(aid)

            # --- Strategy 2.5: Yield (for finished agents) ---
            # Only try this if we haven't failed repeatedly (attempt < 3).
            # If attempt >= 3, we assume simple yield failed and fall through to Tier 2 (Coordinated Yield).
            if finished_agents_in_coll and coll_attempt_num < 3:
                if verbose: print(f"  [Tier 1] Found finished agents {finished_agents_in_coll} in collision. Trying Yield Strategy 2.5.")
                # Try to yield with one of the finished agents
                random.shuffle(finished_agents_in_coll)
                
                for yield_agent_id in finished_agents_in_coll:
                    yield_idx = yield_agent_id - 1
                    
                    # Find a free neighbor to step aside to
                    goal_pos = tuple(map(int, agent_goals[yield_idx]))
                    current_grid = pristine_static_grid
                    rows, cols = current_grid.shape
                    
                    # NO YIELD CAP: Agents must always cooperate for global objective
                    # Oscillations prevented by:
                    # - Better yield cell selection (flow-based)
                    # - Dynamic wait duration (ETA-based)
                    # - Yield contracts (prevent premature return)
                    # - Rewind window (backtrack to find better positions)
                    
                    # Build blocking graph and flow map for traffic-aware selection
                    blocking_edges, blocked_cells_per_agent = build_blocking_graph(
                        current_trajectories, agent_goals, pristine_static_grid,
                        agent_envs, model, device, search_type, algo,
                        horizon=10, max_expansions=500, timeout=2.0
                    )
                    
                    flow_map = compute_flow_map(blocked_cells_per_agent, ignore_agent=yield_idx)
                    
                    # Find yield cell candidates
                    deep_yield_candidates = find_deep_yield_cells(
                        goal_pos, pristine_static_grid, 
                        distance_range=(2, 6), max_candidates=25
                    )
                    
                    # Use flow-based selection (softer than reservation filtering)
                    if deep_yield_candidates:
                        best_yield_cell = select_best_yield_cell(
                            deep_yield_candidates, pristine_static_grid,
                            cell_degrees, flow_map, goal_pos, corridor_map
                        )
                        use_astar = True
                    else:
                        best_yield_cell = None
                        use_astar = False
                    
                    # Fallback to immediate neighbors if no good deep cells
                    if not best_yield_cell:
                        neighbors = []
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = goal_pos[0]+dr, goal_pos[1]+dc
                            if 0 <= nr < rows and 0 <= nc < cols and current_grid[nr, nc] == 0:
                                neighbors.append(((nr, nc), 1))
                        
                        if neighbors:
                            best_yield_cell = select_best_yield_cell(
                                neighbors, pristine_static_grid, cell_degrees, 
                                flow_map, goal_pos, corridor_map
                            )
                            use_astar = False
                        else:
                            continue
                    
                    if not best_yield_cell: continue
                    
                    yield_cell = best_yield_cell
                    
                    # B) Dynamic wait duration based on blocker ETA
                    # Estimate how long blockers need to clear the area
                    blocker_ids = [aid for aid in current_coll_active_agents_list if aid != yield_agent_id]
                    max_eta = 5  # Default minimum
                    for blocker_id in blocker_ids:
                        blocking_region = {goal_pos, yield_cell}  # Region to clear
                        eta = estimate_blocker_eta(
                            blocker_id, blocking_region, current_trajectories, coll_t
                        )
                        max_eta = max(max_eta, eta)
                    
                    # Cap wait duration between 5 and 15 steps
                    wait_steps = min(max(max_eta, 5), 15)
                    if verbose: print(f"    Yield wait duration: {wait_steps} steps (ETA-based)")
                    
                    # Construct Yield Detour
                    # Construct Yield Detour
                    if use_astar:
                        # Deep yield: use A* to plan path to/from yield cell
                        env_yield_out = copy.deepcopy(agent_envs[yield_idx])
                        env_yield_out.env.grid = pristine_static_grid.copy()
                        env_yield_out.env.agent_pos = goal_pos
                        env_yield_out.env.goal_pos = yield_cell
                        
                        path_out = plan_with_search(
                            env_yield_out, model, device, search_type, algo,
                            timeout=1.0, heuristic_weight=1.0, max_expansions=200
                        )
                        
                        if not path_out:
                            continue  # Couldn't reach yield cell, try next
                        
                        # Path back from yield to goal
                        env_yield_back = copy.deepcopy(agent_envs[yield_idx])
                        env_yield_back.env.grid = pristine_static_grid.copy()
                        env_yield_back.env.agent_pos = yield_cell
                        env_yield_back.env.goal_pos = goal_pos
                        
                        path_back = plan_with_search(
                            env_yield_back, model, device, search_type, algo,
                            timeout=1.0, heuristic_weight=1.0, max_expansions=200
                        )
                        if not path_back:
                            continue  # Couldn't return, try next
                        
                        detour_actions = path_out + [4] * wait_steps + path_back
                        
                        # D) CONTRACT: Create yield contract for deep yield
                        yield_contracts[yield_idx+1] = {
                            "for_agents": blocker_ids,
                            "region": {"type": "cells", "cells": {yield_cell}},
                            "valid_until": {"type": "agent_passed", "value": blocker_ids},
                            "yield_position": yield_cell,
                            "created_at_pass": pass_num
                        }
                        deep_yield_count[yield_idx] += 1
                            
                    else:
                        # Shallow yield: simple one-step move
                        def get_action(p1, p2):
                            d = (p2[0]-p1[0], p2[1]-p1[1])
                            if d == (-1, 0): return 0 # Up
                            if d == (1, 0): return 1 # Down
                            if d == (0, -1): return 2 # Left
                            if d == (0, 1): return 3 # Right
                            return 4 # Wait
                                
                        a_out = get_action(goal_pos, yield_cell)
                        a_wait = [4] * (wait_steps - 1)
                        a_back = get_action(yield_cell, goal_pos)
                        detour_actions = [a_out] + a_wait + [a_back]
                        
                        # Splice into plan
                        curr_plan = list(current_agent_plans[yield_idx])
                        # We need to insert this detour at coll_t (or slightly before)
                        # Since agent is finished, we can append to the end or insert at coll_t
                        # Inserting at coll_t is safer to resolve immediate collision
                        
                        insert_t = coll_t
                        if insert_t > len(curr_plan): insert_t = len(curr_plan)
                        
                        base_plan = curr_plan[:insert_t]
                        remaining_plan = curr_plan[insert_t:] # Should be mostly waits if finished
                        
                        new_full_plan = base_plan + detour_actions + remaining_plan
                        
                        # Simulate
                        sim_env = copy.deepcopy(agent_envs[yield_idx])
                        sim_env.env.agent_pos = agent_starts[yield_idx]
                        new_full_traj_sim = simulate_plan(sim_env, new_full_plan)
                        
                        # VALIDATE: Check if simulated trajectory hits any obstacles
                        invalid_trajectory = False
                        for traj_pos in new_full_traj_sim:
                            r, c = int(traj_pos[0]), int(traj_pos[1])
                            if 0 <= r < pristine_static_grid.shape[0] and 0 <= c < pristine_static_grid.shape[1]:
                                if pristine_static_grid[r, c] == -1:
                                    if verbose: print(f"      Yield detour for Agent {yield_idx+1} would hit obstacle at {(r,c)}. Trying next neighbor.")
                                    invalid_trajectory = True
                                    break
                        if invalid_trajectory: continue  # Try next yield_cell
                        
                        # Check collisions
                        temp_trajs = list(current_trajectories)
                        temp_trajs[yield_idx] = new_full_traj_sim
                        
                        new_colls_list = analyze_collisions(temp_trajs, agent_goals, pristine_static_grid)
                        orig_coll_ok = not any((c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key_tuple for c in new_colls_list)
                        
                        if orig_coll_ok:
                            # <<< LOGGING >>>
                            log_context["alert_id_counter"] += 1
                            alert_id_y = f"Alert-{log_context['alert_id_counter']}"
                            log_context["strategy_id_counter"] += 1
                            strategy_id_y = f"Strategy-{log_context['strategy_id_counter']}"
                            cell_str_ref_y = str(coll['cell']).replace(" ", "")
                            coll_id_ref_y = f"Collision-{coll['time']}-{cell_str_ref_y}"
                            
                            alert_log_y = {
                                "id": alert_id_y, "alertsConflict": coll_id_ref_y, "targetAgent": f"Robot-{yield_agent_id}",
                                "alertType": "yieldAtGoal", "rationale": "Finished agent yielding to clear goal cell."
                            }
                            log_data["conflictAlerts"].append(alert_log_y)
                            strategy_log_y = {"id": strategy_id_y, "triggeredBy": alert_id_y, "type": "YieldAtGoal"}
                            log_data["replanningStrategies"].append(strategy_log_y)
                            
                            log_context["subplan_id_counter"] += 1
                            new_subplan_id_y = f"Robot-{yield_idx+1}-Plan-Resolved-{log_context['subplan_id_counter']}"
                            plan_log_y = {
                                "id": new_subplan_id_y, "belongsToAgent": f"Robot-{yield_idx+1}",
                                "derivedFromConflict": coll_id_ref_y, "generatedBy": strategy_id_y,
                                "planCost": len(new_full_plan),
                                "steps": [{"time": t, "cell": [int(p[0]), int(p[1])]} for t, p in enumerate(new_full_traj_sim)]
                            }
                            log_data["agentSubplans"].append(plan_log_y)
                            pass_log["events"].append(f"Yield success for Agent {yield_idx+1}")
                            # <<< END LOGGING >>>

                            current_agent_plans[yield_idx] = new_full_plan
                            current_trajectories[yield_idx] = new_full_traj_sim
                            if info_tracker: info_tracker.record_revised_submission(new_full_traj_sim)
                            
                            yield_success = True; any_fix_this_pass_iteration = run_counters['replan_success_yield'] = run_counters.get('replan_success_yield', 0) + 1
                            if verbose: print(f"    Committed YIELD fix for Agent {yield_idx+1}.")
                            break
                    if yield_success: break
            
            if yield_success: 
                continue # Collision resolved by Tier 1, move to next collision


            # ==================================================================================
            # PHASE 4: PIVOT PARKING (Cycle Resolution)
            # ==================================================================================
            # Trigger: Persistent collisions (pass >= 5) or oscillations (deep_yield_count high)
            # Logic: Build blocking graph, detect cycles, pick pivot, park them with LONG-TERM contract.
            
            pivot_parking_success = False
            if info_setting == "all" and pass_num >= 5 and not static_success:
                # Check if we should try pivot parking
                # Only try if we haven't recently
                
                # Build blocking graph (expensive, so only if needed)
                # We can reuse the one from Tier 1 if available? 
                # Tier 1 builds it locally. Let's rebuild or cache.
                # For now, rebuild.
                
                if verbose: print(f"  Checking for Cycles/Deadlocks (Pivot Parking)...")
                
                blocking_edges, blocked_cells_per_agent = build_blocking_graph(
                    current_trajectories, agent_goals, pristine_static_grid,
                    agent_envs, model, device, search_type, algo,
                    horizon=15, max_expansions=500, timeout=2.0
                )
                
                cycles = find_cycles(blocking_edges, len(agent_goals))
                
                if cycles:
                    if verbose: print(f"    Found {len(cycles)} cycles: {cycles}")
                    
                    for cycle in cycles:
                        # Check if this collision is part of the cycle
                        # Intersection of cycle agents and collision agents
                        common_agents = set(cycle) & set(current_coll_active_agents_list)
                        if not common_agents: continue
                        
                        # Choose pivot
                        pivot_id = choose_pivot_agent(cycle, [], cell_degrees, current_trajectories) # finished_agents not easily avail here, pass empty
                        pivot_idx = pivot_id - 1
                        
                        if verbose: print(f"    Selected Pivot Agent {pivot_id} for cycle {cycle}")
                        
                        # Plan parking for pivot
                        # 1. Compute flow map (ignore pivot)
                        flow_map = compute_flow_map(blocked_cells_per_agent, ignore_agent=pivot_id)
                        
                        # 2. Find parking spot (deep yield)
                        # Prefer further away (3-6 steps)
                        pivot_pos = tuple(map(int, current_trajectories[pivot_idx][min(coll_t, len(current_trajectories[pivot_idx])-1)]))
                        # Or start from goal if finished?
                        # Use current pos
                        
                        parking_candidates = find_deep_yield_cells(
                            pivot_pos, pristine_static_grid,
                            distance_range=(3, 6), max_candidates=30
                        )
                        
                        if not parking_candidates: continue
                        
                        best_parking_cell = select_best_yield_cell(
                            parking_candidates, pristine_static_grid,
                            cell_degrees, flow_map, pivot_pos, corridor_map
                        )
                        
                        if not best_parking_cell: continue
                        
                        # 3. Plan detour to parking and WAIT
                        # Use A*
                        env_park = copy.deepcopy(agent_envs[pivot_idx])
                        env_park.env.grid = pristine_static_grid.copy()
                        env_park.env.agent_pos = pivot_pos
                        env_park.env.goal_pos = best_parking_cell
                        
                        # Mark forbidden regions (respect others' contracts)
                        if pivot_idx in forbidden_regions:
                            for r, c in forbidden_regions[pivot_idx]:
                                if 0 <= r < env_park.env.grid.shape[0] and 0 <= c < env_park.env.grid.shape[1]:
                                    env_park.env.grid[r, c] = -1
                        
                        path_to_park = plan_with_search(
                            env_park, model, device, search_type, algo,
                            timeout=2.0, heuristic_weight=1.0, max_expansions=500
                        )
                        
                        if not path_to_park: continue
                        
                        # 4. Create LONG-TERM contract
                        # Wait until other agents in cycle have passed
                        others_in_cycle = [aid for aid in cycle if aid != pivot_id]
                        
                        # Construct plan: Move to park -> Wait (LONG) -> Return (maybe later)
                        # For now, just wait a long time (e.g. 20 steps) or until contract expires?
                        # We can't easily "wait until event" in a static plan.
                        # So we'll insert a long wait.
                        wait_steps = 20
                        
                        # Path back? Maybe not needed immediately.
                        # Just park and stay.
                        # But we need a valid full plan.
                        # Let's plan back to goal after wait.
                        
                        env_return = copy.deepcopy(agent_envs[pivot_idx])
                        env_return.env.grid = pristine_static_grid.copy()
                        env_return.env.agent_pos = best_parking_cell
                        env_return.env.goal_pos = agent_goals[pivot_idx]
                        
                        path_return = plan_with_search(
                            env_return, model, device, search_type, algo,
                            timeout=2.0, heuristic_weight=1.0, max_expansions=500
                        )
                        
                        if not path_return: 
                            # If can't return, just stay?
                            path_return = [] # Will stay at parking spot
                        
                        detour_actions = path_to_park + [4] * wait_steps + path_return
                        
                        # Splice
                        curr_plan = list(current_agent_plans[pivot_idx])
                        # Insert at current time
                        insert_t = min(coll_t, len(curr_plan))
                        base_plan = curr_plan[:insert_t]
                        
                        new_full_plan = base_plan + detour_actions
                        # Note: we discard the rest of the old plan as we re-planned return
                        
                        # Simulate
                        sim_env = copy.deepcopy(agent_envs[pivot_idx])
                        sim_env.env.agent_pos = agent_starts[pivot_idx]
                        new_full_traj_sim = simulate_plan(sim_env, new_full_plan)
                        
                        # Update
                        current_agent_plans[pivot_idx] = new_full_plan
                        current_trajectories[pivot_idx] = new_full_traj_sim
                        
                        # Create Contract
                        yield_contracts[pivot_idx] = {
                            "for_agents": others_in_cycle,
                            "region": {
                                "type": "cells",
                                "cells": set(blocked_cells_per_agent.get(pivot_id, [])) # Stay out of own desired path (conflict zone)
                            },
                            "valid_until": {
                                "type": "agent_passed",
                                "value": others_in_cycle
                            },
                            "yield_position": best_parking_cell,
                            "created_at_pass": overall_fix_attempt_count
                        }
                        
                        # Add to forbidden for others IMMEDIATELY?
                        # Yes, pivot is moving out. Others should treat pivot's *original* path as free?
                        # Actually, pivot is moving to parking.
                        # We want others to know pivot is yielding.
                        
                        if verbose: print(f"    Committed PIVOT PARKING for Agent {pivot_id}. Park at {best_parking_cell} for {others_in_cycle}")
                        
                        pivot_parking_success = True
                        any_fix_this_pass_iteration = True
                        break
            
            if pivot_parking_success: continue


            # ==================================================================================
            # TIER 2: ACTIVE STRATEGIES (Static -> General Yield -> Dynamic)
            # ==================================================================================
            # If no finished agents involved, or Yield failed, try standard strategies.
            
            # --- Strategy 1: Static Replanning ---
            static_success = False
            if (info_setting == "all" or info_setting == "no") and coll_attempt_num <= 3: 
                agents_to_try_static = []
                if replan_strategy == "best": 
                    agents_to_try_static = list(current_coll_active_agents_list)
                elif replan_strategy == "random": 
                    agents_to_try_static = [random.choice(list(current_coll_active_agents_list))]
                elif replan_strategy == "farthest": # Heuristic: Agent farthest from goal replans
                     dists = []
                     for aid in current_coll_active_agents_list:
                         idx = aid - 1
                         curr_pos = current_trajectories[idx][coll_t] if current_trajectories[idx] and coll_t < len(current_trajectories[idx]) else agent_starts[idx]
                         d = abs(curr_pos[0] - agent_goals[idx][0]) + abs(curr_pos[1] - agent_goals[idx][1])
                         dists.append((d, aid))
                     dists.sort(key=lambda x: x[0], reverse=True)
                     agents_to_try_static = [dists[0][1]]
                
                best_static_outcome = {'agent_idx': -1, 'plan': None, 'traj': None, 'num_new_colls': float('inf')}

                for replan_agent_id in agents_to_try_static:
                    replan_idx = replan_agent_id - 1; run_counters['replan_attempts_static'] += 1
                    if verbose: print(f"  Static Replan for Agent {replan_agent_id}")

                    env_copy = copy.deepcopy(agent_envs[replan_idx])
                    rewind = min(max_rewind_static_val, init_rewind_val + coll_attempt_num -1)
                    replan_t_step = max(0, coll_t - rewind)
                    # Ensure replan_t_step is valid index for trajectory
                    replan_t_step = min(replan_t_step, len(current_trajectories[replan_idx]) -1 if current_trajectories[replan_idx] else 0)
                    current_pos_for_replan = agent_starts[replan_idx]

                    if current_trajectories[replan_idx] and replan_t_step < len(current_trajectories[replan_idx]):
                        current_pos_for_replan = current_trajectories[replan_idx][replan_t_step]

                    planning_grid_static = pristine_static_grid.copy()
                    env_copy.env.grid = planning_grid_static
                    env_copy.env.agent_pos = tuple(map(int,current_pos_for_replan))
                    if 0 <= current_pos_for_replan[0] < env_copy.env.grid.shape[0] and \
                       0 <= current_pos_for_replan[1] < env_copy.env.grid.shape[1]:
                        env_copy.env.grid[tuple(map(int,current_pos_for_replan))] = 1 # Agent's current pos
                    
                    env_copy.env.goal_pos = agent_goals[replan_idx]
                    goal_pos_tuple = tuple(map(int, agent_goals[replan_idx]))
                    if goal_pos_tuple != tuple(map(int,current_pos_for_replan)) and \
                       0 <= goal_pos_tuple[0] < env_copy.env.grid.shape[0] and \
                       0 <= goal_pos_tuple[1] < env_copy.env.grid.shape[1] and \
                       env_copy.env.grid[goal_pos_tuple] != -1: # Not an obstacle
                        env_copy.env.grid[goal_pos_tuple] = 2 # Goal marker

                    blocks_for_this_agent = set(static_block_hist[replan_idx])
                    # Handle different collision types
                    if coll_type_str == 'obstacle':
                        # Obstacle collisions indicate invalid plans - skip resolution
                        if verbose: print(f"    WARNING: Agent has invalid plan hitting obstacle. Skipping.")
                        continue
                    elif coll_type_str == 'vertex':
                        blocks_for_this_agent.add(cell_key(coll_cell_orig))
                    else: # edge collision
                        blocks_for_this_agent.add(cell_key(coll_cell_orig[0]))
                        blocks_for_this_agent.add(cell_key(coll_cell_orig[1]))

                    for block_k in blocks_for_this_agent:
                        block_tup = tuple(map(int,block_k))
                        if block_tup != tuple(map(int,env_copy.env.agent_pos)) and block_tup != goal_pos_tuple:
                            if 0 <= block_tup[0] < env_copy.env.grid.shape[0] and 0 <= block_tup[1] < env_copy.env.grid.shape[1]:
                                env_copy.env.grid[block_tup] = -1 # Obstacle marker

                    if info_tracker:
                        info_tracker.record_static_alert(blocks_for_this_agent)

                    # --- LOGGING: Alert and Strategy (S3) ---
                    log_context["alert_id_counter"] += 1
                    alert_id = f"Alert-{log_context['alert_id_counter']}"
                    log_context["strategy_id_counter"] += 1
                    strategy_id = f"Strategy-{log_context['strategy_id_counter']}"
                    
                    coll_id_ref = f"Collision-{coll['time']}-{coll['cell']}".replace(" ", "")
                    
                    alert_log = {
                        "id": alert_id,
                        "alertsConflict": coll_id_ref,
                        "targetAgent": f"Robot-{replan_agent_id}",
                        "rewindWindow": rewind,
                        "alertType": "staticReplan",
                        "rationale": f"Agent selected via '{replan_strategy}' strategy during attempt #{coll_attempt_num} for this collision.",
                        "staticForbiddenCells": [list(map(int, c)) for c in blocks_for_this_agent]
                    }
                    log_data["conflictAlerts"].append(alert_log)
                    
                    strategy_log = {
                        "id": strategy_id,
                        "triggeredBy": alert_id,
                        "type": "Static Obstacle Avoidance"
                    }
                    log_data["replanningStrategies"].append(strategy_log)
                    # ------------------------------------------

                    # CONTRACT ENFORCEMENT: Mark forbidden regions as obstacles
                    if replan_idx in forbidden_regions:
                        for r, c in forbidden_regions[replan_idx]:
                            if 0 <= r < env_copy.env.grid.shape[0] and 0 <= c < env_copy.env.grid.shape[1]:
                                env_copy.env.grid[r, c] = -1
                    
                    # Also treat other agents' current positions as obstacles if we want strict avoidance
                    # But static replanning usually ignores dynamic agents.
                    # However, if we are in a corridor owned by someone else, we MUST treat it as obstacle.
                    # This is handled by forbidden_regions.
                    
                    new_plan_seg = plan_with_search(env_copy, model, device, search_type, algo, timeout, heuristic_weight, max_expansions)
                    if verbose: print(f"    New plan segment for Agent {replan_agent_id}: {new_plan_seg}")
                    if new_plan_seg is not None:
                        prefix_plan_actions = current_agent_plans[replan_idx][:replan_t_step] if current_trajectories[replan_idx] else []
                        new_full_plan = prefix_plan_actions + new_plan_seg

                        sim_env_for_full_plan = copy.deepcopy(agent_envs[replan_idx])
                        sim_env_for_full_plan.env.agent_pos = agent_starts[replan_idx] # Simulate from actual start

                        new_full_traj = simulate_plan(sim_env_for_full_plan, new_full_plan)
                        temp_trajs = list(current_trajectories)
                        temp_trajs[replan_idx] = new_full_traj

                        new_colls_list = analyze_collisions(temp_trajs, agent_goals, pristine_static_grid)
                        num_new_colls = len(new_colls_list)
                        orig_coll_ok = not any((c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key_tuple for c in new_colls_list)

                        if verbose: print(f"    Agent {replan_agent_id} static: OrigOK={orig_coll_ok}, NewTotal={num_new_colls}")

                        if orig_coll_ok:
                            # <<< LOGGING: New Resolved Plan (S4) >>>
                            log_context["subplan_id_counter"] += 1
                            new_subplan_id = f"Robot-{replan_agent_id}-Plan-Resolved-{log_context['subplan_id_counter']}"
                            
                            plan_log = {
                                "id": new_subplan_id, "belongsToAgent": f"Robot-{replan_agent_id}",
                                "derivedFromConflict": coll_id_ref, "generatedBy": strategy_id,
                                "planCost": len(new_full_plan),
                                "steps": [{"time": t, "cell": [int(p[0]), int(p[1])]} for t, p in enumerate(new_full_traj)]
                            }
                            pass_log["events"].append(f"Static replan success for Agent {replan_agent_id}, created plan {new_subplan_id}")
                            # <<< END LOGGING >>>

                            if replan_strategy == "best":
                                if num_new_colls == 0: # Perfect fix
                                    log_data["agentSubplans"].append(plan_log) # Log only on commit

                                    current_agent_plans[replan_idx] = new_full_plan
                                    current_trajectories[replan_idx] = new_full_traj

                                    if info_tracker:
                                        info_tracker.record_revised_submission(new_full_traj)

                                    # Add to static_block_hist
                                    if coll_type_str == 'vertex': static_block_hist[replan_idx].add(cell_key(coll_cell_orig))
                                    else: static_block_hist[replan_idx].add(cell_key(coll_cell_orig[0])); static_block_hist[replan_idx].add(cell_key(coll_cell_orig[1]))
                                    
                                    static_success = True; any_fix_this_pass_iteration = True; run_counters['replan_success_static'] += 1
                                    if verbose: print(f"    Committed PERFECT static fix for Agent {replan_idx+1}.")
                                    break # Break from agents_to_try_static loop (found perfect fix)
                                elif num_new_colls < best_static_outcome['num_new_colls']:
                                    best_static_outcome = {'agent_idx': replan_idx, 'plan': new_full_plan, 'traj': new_full_traj, 'num_new_colls': num_new_colls}
                            else: # "random" or "farthest" - commit first success
                                log_data["agentSubplans"].append(plan_log)
                                
                                current_agent_plans[replan_idx] = new_full_plan
                                current_trajectories[replan_idx] = new_full_traj

                                if info_tracker:
                                    info_tracker.record_revised_submission(new_full_traj)

                                # Add to static_block_hist
                                if coll_type_str == 'vertex': static_block_hist[replan_idx].add(cell_key(coll_cell_orig))
                                else: static_block_hist[replan_idx].add(cell_key(coll_cell_orig[0])); static_block_hist[replan_idx].add(cell_key(coll_cell_orig[1]))

                                static_success = True; any_fix_this_pass_iteration = True; run_counters['replan_success_static'] += 1
                                if verbose: print(f"    Committed static fix for Agent {replan_idx+1}.")
                                break # Break from agents_to_try_static loop

                if replan_strategy == "best" and not static_success and best_static_outcome['agent_idx'] != -1:
                    commit_idx = best_static_outcome['agent_idx']
                    current_agent_plans[commit_idx] = best_static_outcome['plan']
                    current_trajectories[commit_idx] = best_static_outcome['traj']
                    if info_tracker:
                        info_tracker.record_revised_submission(best_static_outcome['traj'])
                    # --- LOGGING: New Resolved Plan (S4) ---
                    log_context["subplan_id_counter"] += 1
                    new_subplan_id = f"Robot-{commit_idx+1}-Plan-Resolved-{log_context['subplan_id_counter']}"
                    
                    plan_log = {
                        "id": new_subplan_id,
                        "belongsToAgent": f"Robot-{commit_idx+1}",
                        "derivedFromConflict": coll_id_ref,
                        "generatedBy": strategy_id,
                        "planCost": len(best_static_outcome['plan']),
                        "steps": [{"time": t, "cell": [int(p[0]), int(p[1])]} for t, p in enumerate(best_static_outcome['traj'])]
                    }
                    log_data["agentSubplans"].append(plan_log)
                    pass_log["events"].append(f"Static replan success for Agent {commit_idx+1}, created plan {new_subplan_id}")
                    # -----------------------------------------

                    # Add to static_block_hist for the committed best outcome
                    if coll_type_str == 'vertex': static_block_hist[commit_idx].add(cell_key(coll_cell_orig))
                    else: static_block_hist[commit_idx].add(cell_key(coll_cell_orig[0])); static_block_hist[commit_idx].add(cell_key(coll_cell_orig[1]))
                    
                    static_success = True; any_fix_this_pass_iteration = True; run_counters['replan_success_static'] += 1
                    if verbose: print(f"    Committed BEST static fix for Agent {commit_idx+1} (New Colls: {best_static_outcome['num_new_colls']}).")
            
            if static_success: break # Break from trying other strategies for THIS collision
            if any_fix_this_pass_iteration: continue # If static fixed it, move to next collision (this line might be redundant due to break above)


            if any_fix_this_pass_iteration: continue 

            # --- Strategy 2: General Yield (Active Agents) ---
            # Try this BEFORE Dynamic since it's cheaper and fixes deadlocks
            # Skip if we are in a deep pass cycle (pass_num >= 10), implying this strategy isn't working.
            general_yield_success = False
            if info_setting == "all" and not static_success and pass_num < 10:
                yield_candidates = list(current_coll_active_agents_list)
                random.shuffle(yield_candidates)
                
                for yield_agent_id in yield_candidates:
                    yield_idx = yield_agent_id - 1
                    if verbose: print(f"  General Yield Strategy for Agent {yield_agent_id}")
                    
                    curr_traj = list(current_trajectories[yield_idx])
                    curr_plan = list(current_agent_plans[yield_idx])
                    
                    start_yield_t = max(0, coll_t - 1)
                    if start_yield_t >= len(curr_traj): continue
                        
                    start_yield_pos = tuple(map(int, curr_traj[start_yield_t]))
                    
                    # Find free neighbors
                    current_grid = pristine_static_grid
                    rows, cols = current_grid.shape
                    neighbors = []
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = start_yield_pos[0]+dr, start_yield_pos[1]+dc
                        if 0 <= nr < rows and 0 <= nc < cols and current_grid[nr, nc] == 0:
                            neighbors.append((nr, nc))
                            
                    if not neighbors: continue
                        
                    random.shuffle(neighbors)
                    for yield_cell in neighbors:
                        wait_steps = 5
                        
                        def get_action(p1, p2):
                            d = (p2[0]-p1[0], p2[1]-p1[1])
                            if d == (-1, 0): return 0 
                            if d == (1, 0): return 1 
                            if d == (0, -1): return 2 
                            if d == (0, 1): return 3 
                            return 4 
                            
                        a_out = get_action(start_yield_pos, yield_cell)
                        a_wait = [4] * (wait_steps - 1)
                        a_back = get_action(yield_cell, start_yield_pos)
                        
                        detour_actions = [a_out] + a_wait + [a_back]
                        
                        base_plan = curr_plan[:start_yield_t]
                        remaining_plan = curr_plan[start_yield_t:]
                        
                        new_full_plan = base_plan + detour_actions + remaining_plan
                        
                        sim_env = copy.deepcopy(agent_envs[yield_idx])
                        sim_env.env.agent_pos = agent_starts[yield_idx]
                        new_full_traj_sim = simulate_plan(sim_env, new_full_plan)
                        
                        temp_trajs = list(current_trajectories)
                        temp_trajs[yield_idx] = new_full_traj_sim
                        
                        new_colls_list = analyze_collisions(temp_trajs, agent_goals, pristine_static_grid)
                        orig_coll_ok = not any((c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key_tuple for c in new_colls_list)
                        
                        if orig_coll_ok:
                            # LOGGING
                            log_context["alert_id_counter"] += 1
                            alert_id_gy = f"Alert-{log_context['alert_id_counter']}"
                            log_context["strategy_id_counter"] += 1
                            strategy_id_gy = f"Strategy-{log_context['strategy_id_counter']}"
                            cell_str_ref_gy = str(coll['cell']).replace(" ", "")
                            coll_id_ref_gy = f"Collision-{coll['time']}-{cell_str_ref_gy}"
                            
                            alert_log_gy = {
                                "id": alert_id_gy, "alertsConflict": coll_id_ref_gy, "targetAgent": f"Robot-{yield_agent_id}",
                                "alertType": "generalYield", "rationale": "Active agent yielding to resolve deadlock."
                            }
                            log_data["conflictAlerts"].append(alert_log_gy)
                            strategy_log_gy = {"id": strategy_id_gy, "triggeredBy": alert_id_gy, "type": "GeneralYield"}
                            log_data["replanningStrategies"].append(strategy_log_gy)
                            
                            log_context["subplan_id_counter"] += 1
                            new_subplan_id_gy = f"Robot-{yield_idx+1}-Plan-Resolved-{log_context['subplan_id_counter']}"
                            plan_log_gy = {
                                "id": new_subplan_id_gy, "belongsToAgent": f"Robot-{yield_idx+1}",
                                "derivedFromConflict": coll_id_ref_gy, "generatedBy": strategy_id_gy,
                                "planCost": len(new_full_plan),
                                "steps": [{"time": t, "cell": [int(p[0]), int(p[1])]} for t, p in enumerate(new_full_traj_sim)]
                            }
                            log_data["agentSubplans"].append(plan_log_gy)
                            pass_log["events"].append(f"General Yield success for Agent {yield_idx+1}")
                            
                            current_agent_plans[yield_idx] = new_full_plan
                            current_trajectories[yield_idx] = new_full_traj_sim
                            if info_tracker: info_tracker.record_revised_submission(new_full_traj_sim)
                            
                            general_yield_success = True; any_fix_this_pass_iteration = True; run_counters['replan_success_general_yield'] = run_counters.get('replan_success_general_yield', 0) + 1
                            if verbose: print(f"    Committed GENERAL YIELD fix for Agent {yield_idx+1}.")
                            break
                    if general_yield_success: break
            
            if general_yield_success: break
            if any_fix_this_pass_iteration: continue

            # --- Strategy 3: Dynamic Replanning ---
            dyn_success = False
            if (info_setting == "all" or info_setting == "only_dyn"): 
                if not static_success and coll_attempt_num <= 2:
                    dyn_agents_to_evaluate = []
                    if replan_strategy == "best":
                        dyn_agents_to_evaluate = list(current_coll_active_agents_list)
                        # For "best", order of attempts doesn't strictly matter unless a perfect one is found early.
                        # Could sort by agent ID for determinism or shuffle. Keeping it simple.
                    elif replan_strategy == "random":
                        dyn_agents_to_evaluate = list(current_coll_active_agents_list)
                        random.shuffle(dyn_agents_to_evaluate)
                    elif replan_strategy == "farthest":
                        def get_dist_to_goal_dyn(agent_id_plus_1):
                            agent_idx = agent_id_plus_1 - 1
                            pos_at_t_minus_1 = agent_starts[agent_idx]
                            if current_trajectories[agent_idx] and coll_t > 0 and coll_t < len(current_trajectories[agent_idx]):
                                pos_at_t_minus_1 = current_trajectories[agent_idx][coll_t - 1]
                            elif current_trajectories[agent_idx] and coll_t == 0:
                                pos_at_t_minus_1 = current_trajectories[agent_idx][0]
                            # Ensure agent_goals[agent_idx] is valid before np.linalg.norm
                            if agent_idx < len(agent_goals) and agent_goals[agent_idx] is not None:
                                return np.linalg.norm(np.array(pos_at_t_minus_1) - np.array(agent_goals[agent_idx]))
                            return float('inf') # Should ideally not happen with consistent data
                        dyn_agents_to_evaluate = sorted(list(current_coll_active_agents_list), key=get_dist_to_goal_dyn, reverse=True)

                    best_dyn_outcome = {'agent_idx': -1, 'plan': None, 'traj': None, 'num_new_colls': float('inf'), 'dyn_obs_fset': None}

                    for replan_agent_id in dyn_agents_to_evaluate:
                        replan_idx = replan_agent_id - 1
                        others_for_dyn = [aid for aid in current_coll_active_agents_list if aid != replan_agent_id]

                        if not others_for_dyn: continue

                        dyn_obs_id = random.choice(others_for_dyn) # Still choosing one dynamic obstacle randomly
                        dyn_obs_idx = dyn_obs_id - 1
                        run_counters['replan_attempts_dynamic'] +=1

                        if verbose: print(f"  Dynamic Replan for Agent {replan_agent_id} (DynObs: {dyn_obs_id})")

                        env_copy = copy.deepcopy(agent_envs[replan_idx])
                        rewind = min(max_rewind_dyn_val, init_rewind_val + coll_attempt_num -1)
                        replan_t_agent = max(0, coll_t - rewind)
                        replan_t_agent = min(replan_t_agent, len(current_trajectories[replan_idx]) -1 if current_trajectories[replan_idx] else 0)
                        current_pos_on_grid_dyn = agent_starts[replan_idx]
                        if current_trajectories[replan_idx] and replan_t_agent < len(current_trajectories[replan_idx]):
                            current_pos_on_grid_dyn = current_trajectories[replan_idx][replan_t_agent]

                        planning_grid_dyn = pristine_static_grid.copy()
                        env_copy.env.grid = planning_grid_dyn
                        env_copy.env.agent_pos = tuple(map(int,current_pos_on_grid_dyn))
                        if 0 <= current_pos_on_grid_dyn[0] < env_copy.env.grid.shape[0] and \
                           0 <= current_pos_on_grid_dyn[1] < env_copy.env.grid.shape[1]:
                             env_copy.env.grid[tuple(map(int,current_pos_on_grid_dyn))] = 1
                        
                        env_copy.env.goal_pos = agent_goals[replan_idx]
                        goal_pos_tuple_dyn = tuple(map(int, agent_goals[replan_idx]))
                        if goal_pos_tuple_dyn != tuple(map(int,current_pos_on_grid_dyn)) and \
                           0 <= goal_pos_tuple_dyn[0] < env_copy.env.grid.shape[0] and \
                           0 <= goal_pos_tuple_dyn[1] < env_copy.env.grid.shape[1] and \
                           env_copy.env.grid[goal_pos_tuple_dyn] != -1:
                            env_copy.env.grid[goal_pos_tuple_dyn] = 2
                        
                        obs_traj = current_trajectories[dyn_obs_idx]
                        if not obs_traj: continue

                        obs_start_t = replan_t_agent 
                        obs_end_t = min(len(obs_traj) -1, replan_t_agent + (2*rewind) ) # Consider a window around replan_t_agent
                        if obs_start_t >= len(obs_traj) or obs_start_t > obs_end_t : continue
                        
                        obs_path_tups = [tuple(map(int,p)) for p in obs_traj[obs_start_t : obs_end_t + 1]]
                        if not obs_path_tups: continue
                        
                        if info_tracker:
                            info_tracker.record_dynamic_alert(obs_path_tups)

                        new_dyn_fset = frozenset(cell_key(p) for p in obs_path_tups)
                        # Optional: Check against dyn_obs_hist if needed, original code had it commented
                        # if any(not old_fset.isdisjoint(new_dyn_fset) for old_fset in dyn_obs_hist[replan_idx]) and dyn_obs_hist[replan_idx]:
                        # if verbose: print(f"    New dynamic obs overlaps history for Agent {replan_agent_id}. Skipping."); continue

                        env_copy.env.dynamic_info = [{'pos': obs_path_tups[0], 'goal': obs_path_tups[-1], 'path': deque(obs_path_tups), 'stop_after_goal': True }]
                        env_copy.env.num_dynamic_obstacles = 1
                        
                        # <<< LOGGING: Dynamic Alert and Strategy (S3) >>>
                        log_context["alert_id_counter"] += 1
                        alert_id_dyn = f"Alert-{log_context['alert_id_counter']}"
                        log_context["strategy_id_counter"] += 1
                        strategy_id_dyn = f"Strategy-{log_context['strategy_id_counter']}"
                        cell_str_ref_dyn = str(coll['cell']).replace(" ", "")
                        coll_id_ref_dyn = f"Collision-{coll['time']}-{cell_str_ref_dyn}"
                        
                        alert_log_dyn = {
                            "id": alert_id_dyn, "alertsConflict": coll_id_ref_dyn, "targetAgent": f"Robot-{replan_agent_id}",
                            "rewindWindow": rewind, "alertType": "dynamicReplan",
                            "rationale": f"Agent selected via '{replan_strategy}' strategy, avoiding Agent {dyn_obs_id}'s path.",
                            "dynamicForbiddenPath": [{"time": t + obs_start_t, "cell": [int(p[0]), int(p[1])]} for t, p in enumerate(obs_path_tups)]
                        }
                        log_data["conflictAlerts"].append(alert_log_dyn)
                        
                        strategy_log_dyn = {"id": strategy_id_dyn, "triggeredBy": alert_id_dyn, "type": "Dynamic Obstacle Avoidance"}
                        log_data["replanningStrategies"].append(strategy_log_dyn)
                        # <<< END LOGGING >>>

                        new_plan_seg = plan_with_search(env_copy, model, device, search_type, algo, timeout, heuristic_weight, max_expansions)
                        if new_plan_seg is not None:
                            prefix = current_agent_plans[replan_idx][:replan_t_agent] if current_trajectories[replan_idx] else []
                            new_full_plan = prefix + new_plan_seg
                            
                            sim_env_for_full_dyn_plan = copy.deepcopy(agent_envs[replan_idx])
                            sim_env_for_full_dyn_plan.env.agent_pos = agent_starts[replan_idx]
                            new_full_traj = simulate_plan(sim_env_for_full_dyn_plan, new_full_plan)
                            
                            temp_trajs = list(current_trajectories)
                            temp_trajs[replan_idx] = new_full_traj

                            new_colls_list = analyze_collisions(temp_trajs, agent_goals, pristine_static_grid)
                            num_new_colls = len(new_colls_list)
                            orig_coll_ok = not any((c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key_tuple for c in new_colls_list)
                            
                            if verbose: print(f"    Agent {replan_agent_id} dynamic: OrigOK={orig_coll_ok}, NewTotal={num_new_colls}")
                            
                            if orig_coll_ok:
                                # <<< LOGGING: New Resolved Plan (S4) >>>
                                log_context["subplan_id_counter"] += 1
                                new_subplan_id_dyn = f"Robot-{replan_agent_id}-Plan-Resolved-{log_context['subplan_id_counter']}"
                                
                                plan_log_dyn = {
                                    "id": new_subplan_id_dyn, "belongsToAgent": f"Robot-{replan_agent_id}",
                                    "derivedFromConflict": coll_id_ref_dyn, "generatedBy": strategy_id_dyn,
                                    "planCost": len(new_full_plan),
                                    "steps": [{"time": t, "cell": [int(p[0]), int(p[1])]} for t, p in enumerate(new_full_traj)]
                                }
                                pass_log["events"].append(f"Dynamic replan success for Agent {replan_agent_id}, created plan {new_subplan_id_dyn}")
                                # <<< END LOGGING >>>

                                if replan_strategy == "best":
                                    if num_new_colls == 0: # Perfect fix
                                        log_data["agentSubplans"].append(plan_log_dyn)
                                        current_agent_plans[replan_idx] = new_full_plan
                                        current_trajectories[replan_idx] = new_full_traj

                                        if info_tracker: info_tracker.record_revised_submission(new_full_traj)

                                        dyn_obs_hist[replan_idx].append(new_dyn_fset)

                                        dyn_success = True; any_fix_this_pass_iteration = True; run_counters['replan_success_dynamic'] += 1
                                        if verbose: print(f"    Committed PERFECT dynamic fix for Agent {replan_idx+1}.")
                                        break # Break from dyn_agents_to_evaluate loop
                                    elif num_new_colls < best_dyn_outcome['num_new_colls']:
                                        best_dyn_outcome = {'agent_idx': replan_idx, 'plan': new_full_plan, 'traj': new_full_traj, 'num_new_colls': num_new_colls, 'dyn_obs_fset': new_dyn_fset}
                                else: # "random" or "farthest"
                                    log_data["agentSubplans"].append(plan_log_dyn)
                                    current_agent_plans[replan_idx] = new_full_plan
                                    current_trajectories[replan_idx] = new_full_traj

                                    if info_tracker: info_tracker.record_revised_submission(new_full_traj)

                                    dyn_obs_hist[replan_idx].append(new_dyn_fset)

                                    dyn_success = True; any_fix_this_pass_iteration = True; run_counters['replan_success_dynamic'] += 1
                                    if verbose: print(f"    Committed dynamic fix for Agent {replan_idx+1}.")
                                    break # Break from dyn_agents_to_evaluate loop
                    
                    # After the loop, commit the best outcome if "best" strategy and no perfect fix was found and a better outcome exists
                    if replan_strategy == "best" and not dyn_success and best_dyn_outcome['agent_idx'] != -1:
                        commit_idx = best_dyn_outcome['agent_idx']
                        current_agent_plans[commit_idx] = best_dyn_outcome['plan']
                        current_trajectories[commit_idx] = best_dyn_outcome['traj']

                        if info_tracker: info_tracker.record_revised_submission(best_dyn_outcome['traj'])

                        dyn_obs_hist[commit_idx].append(best_dyn_outcome['dyn_obs_fset'])

                        # --- LOGGING: New Resolved Plan (S4) ---
                        log_context["subplan_id_counter"] += 1
                        new_subplan_id = f"Robot-{commit_idx+1}-Plan-Resolved-{log_context['subplan_id_counter']}"
                        
                        plan_log = {
                            "id": new_subplan_id,
                            "belongsToAgent": f"Robot-{commit_idx+1}",
                            "derivedFromConflict": coll_id_ref,
                            "generatedBy": strategy_id,
                            "planCost": len(best_dyn_outcome['plan']),
                            "steps": [{"time": t, "cell": [int(p[0]), int(p[1])]} for t, p in enumerate(best_dyn_outcome['traj'])]
                        }
                        log_data["agentSubplans"].append(plan_log)
                        pass_log["events"].append(f"Dynamic replan success for Agent {commit_idx+1}, created plan {new_subplan_id}")
                        # -----------------------------------------

                        dyn_success = True; any_fix_this_pass_iteration = True; run_counters['replan_success_dynamic'] += 1
                        if verbose: print(f"    Committed BEST dynamic fix for Agent {commit_idx+1} (New Colls: {best_dyn_outcome['num_new_colls']}).")

                if dyn_success: break # Break from strategy attempts for THIS collision
            
            if any_fix_this_pass_iteration and not static_success and not general_yield_success: continue

            # --- Strategy 3.5: Smart Coordinated Yield (Path Clearing) ---
            # Trigger: Earlier activation (coll_attempt_num >= 2) to resolve deadlocks quickly.
            # Logic: Identify blocked agent, find its optimal path, and move the blocker out of that specific path.
            coord_yield_success = False
            if info_setting == "all" and not static_success and not dyn_success and not general_yield_success:
                # AGGRESSIVE TRIGGER: attempt >= 1 OR pass >= 3 OR ANY finished agent involved
                if coll_attempt_num >= 1 or pass_num >= 3 or finished_agents_in_coll:
                    if verbose: print(f"  Smart Coordinated Yield Strategy for Agents {current_coll_active_agents_list}")
                    
                    agents_in_coll = list(current_coll_active_agents_list)
                    if len(agents_in_coll) < 2: pass
                    else:
                        # 1. Identify Blocked vs Blocker
                        # Heuristic: Agent with longer remaining path is "Blocked" (needs to go through), 
                        # Agent with shorter path (or at goal) is "Blocker" (can step aside).
                        # Or just try both combinations.
                        
                        # Let's try to find a valid clearing move for ANY agent that clears the path for ANOTHER agent.
                        
                        best_coord_fix = None
                        
                        # Try each agent as the "Blocked" one (who gets priority)
                        for blocked_aid in agents_in_coll:
                            blocker_aid = [a for a in agents_in_coll if a != blocked_aid][0] # Assume 2 agents for now
                            
                            blocked_idx = blocked_aid - 1
                            blocker_idx = blocker_aid - 1
                            
                            # 2. Compute Optimal Static Path for Blocked Agent
                            # We need a path from current pos to goal, ignoring dynamic obstacles (but respecting static)
                            env_blocked = copy.deepcopy(agent_envs[blocked_idx])
                            env_blocked.env.grid = pristine_static_grid.copy()
                            
                            # CONTRACT ENFORCEMENT: Mark forbidden regions as obstacles
                            if blocked_idx in forbidden_regions:
                                for r, c in forbidden_regions[blocked_idx]:
                                    if 0 <= r < env_blocked.env.grid.shape[0] and 0 <= c < env_blocked.env.grid.shape[1]:
                                        env_blocked.env.grid[r, c] = -1
                            
                            # Also treat other agents' current positions as obstacles if we want strict avoidance
                            # But static replanning usually ignores dynamic agents.
                            # However, if we are in a corridor owned by someone else, we MUST treat it as obstacle.
                            # This is handled by forbidden_regions.
                            
                            # Start from slightly before collision to ensure continuity
                            start_t = max(0, coll_t - 1)
                            if not current_trajectories[blocked_idx] or start_t >= len(current_trajectories[blocked_idx]):
                                continue
                                
                            start_pos_blocked = tuple(map(int, current_trajectories[blocked_idx][start_t]))
                            env_blocked.env.agent_pos = start_pos_blocked
                            
                            # Use A* to find optimal path
                            # We can use plan_with_search but need to ensure it returns a full path to goal
                            # Temporarily increase timeout/expansions for this critical pathfinding
                            optimal_path_segment = plan_with_search(
                                env_blocked, model, device, "astar", algo, 
                                timeout=2.0, heuristic_weight=1.0, max_expansions=500
                            )
                            
                            if not optimal_path_segment: continue
                            
                            # Simulate to get cells
                            sim_env_blocked = copy.deepcopy(env_blocked)
                            optimal_traj_blocked = simulate_plan(sim_env_blocked, optimal_path_segment)
                            if not optimal_traj_blocked: continue
                            
                            optimal_path_cells = set(tuple(map(int, p)) for p in optimal_traj_blocked)
                            
                            # 3. Find Blocker's Position and Valid Neighbors
                            if not current_trajectories[blocker_idx]: continue
                            
                            # Blocker's pos at start_t
                            if start_t >= len(current_trajectories[blocker_idx]):
                                start_pos_blocker = tuple(map(int, current_trajectories[blocker_idx][-1]))
                            else:
                                start_pos_blocker = tuple(map(int, current_trajectories[blocker_idx][start_t]))
                                
                            # Find neighbors for Blocker that are NOT on Blocked Agent's optimal path
                            current_grid = pristine_static_grid
                            rows, cols = current_grid.shape
                            
                            valid_neighbors = []
                            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                                nr, nc = start_pos_blocker[0]+dr, start_pos_blocker[1]+dc
                                if 0 <= nr < rows and 0 <= nc < cols and current_grid[nr, nc] == 0:
                                    if (nr, nc) not in optimal_path_cells and (nr, nc) != start_pos_blocked:
                                        valid_neighbors.append((nr, nc))
                            
                            if not valid_neighbors: continue
                            
                            # Found a valid clearing move!
                            # Plan: Blocker moves to Neighbor -> Wait -> Return/Resume
                            # Blocked Agent follows optimal path (or keeps current if valid, but optimal is better)
                            
                            clearing_neighbor = random.choice(valid_neighbors)
                            
                            # Construct Blocker's Detour
                            wait_steps = 5
                            def get_action(p1, p2):
                                d = (p2[0]-p1[0], p2[1]-p1[1])
                                if d == (-1, 0): return 0 
                                if d == (1, 0): return 1 
                                if d == (0, -1): return 2 
                                if d == (0, 1): return 3 
                                return 4 
                                
                            a_out = get_action(start_pos_blocker, clearing_neighbor)
                            a_wait = [4] * wait_steps
                            a_back = get_action(clearing_neighbor, start_pos_blocker)
                            
                            blocker_detour = [a_out] + a_wait + [a_back]
                            
                            curr_plan_blocker = current_agent_plans[blocker_idx]
                            base_plan_blocker = curr_plan_blocker[:start_t]
                            rem_plan_blocker = curr_plan_blocker[start_t:]
                            
                            new_plan_blocker = base_plan_blocker + blocker_detour + rem_plan_blocker
                            
                            # Construct Blocked Agent's Plan (Follow Optimal)
                            # We splice the optimal segment into the current plan
                            curr_plan_blocked = current_agent_plans[blocked_idx]
                            base_plan_blocked = curr_plan_blocked[:start_t]
                            # We replace the rest with the optimal path
                            new_plan_blocked = base_plan_blocked + optimal_path_segment
                            
                            # Verify
                            temp_trajs = list(current_trajectories)
                            
                            sim_env_blocker = copy.deepcopy(agent_envs[blocker_idx])
                            sim_env_blocker.env.agent_pos = agent_starts[blocker_idx]
                            temp_trajs[blocker_idx] = simulate_plan(sim_env_blocker, new_plan_blocker)
                            
                            sim_env_blocked_verify = copy.deepcopy(agent_envs[blocked_idx])
                            sim_env_blocked_verify.env.agent_pos = agent_starts[blocked_idx]
                            temp_trajs[blocked_idx] = simulate_plan(sim_env_blocked_verify, new_plan_blocked)
                            
                            new_colls_list = analyze_collisions(temp_trajs, agent_goals, pristine_static_grid)
                            orig_coll_ok = not any((c['time'], c['type'], cell_key(c['cell']), frozenset(c['agents'])) == coll_key_tuple for c in new_colls_list)
                            
                            if orig_coll_ok:
                                best_coord_fix = (blocked_idx, new_plan_blocked, temp_trajs[blocked_idx], 
                                                  blocker_idx, new_plan_blocker, temp_trajs[blocker_idx])
                                break # Found a fix!
                        
                        if best_coord_fix:
                            b_idx, b_plan, b_traj, bl_idx, bl_plan, bl_traj = best_coord_fix
                            
                            # Commit
                            log_context["alert_id_counter"] += 1
                            alert_id_cy = f"Alert-{log_context['alert_id_counter']}"
                            log_context["strategy_id_counter"] += 1
                            strategy_id_cy = f"Strategy-{log_context['strategy_id_counter']}"
                            cell_str_ref_cy = str(coll['cell']).replace(" ", "")
                            coll_id_ref_cy = f"Collision-{coll['time']}-{cell_str_ref_cy}"
                            
                            alert_log_cy = {
                                "id": alert_id_cy, "alertsConflict": coll_id_ref_cy, 
                                "targetAgent": f"Agents-[{b_idx+1}, {bl_idx+1}]",
                                "alertType": "smartCoordinatedYield", 
                                "rationale": f"Agent {bl_idx+1} clears path for Agent {b_idx+1}."
                            }
                            log_data["conflictAlerts"].append(alert_log_cy)
                            strategy_log_cy = {"id": strategy_id_cy, "triggeredBy": alert_id_cy, "type": "SmartCoordinatedYield"}
                            log_data["replanningStrategies"].append(strategy_log_cy)
                            
                            # Update Blocked
                            current_agent_plans[b_idx] = b_plan
                            current_trajectories[b_idx] = b_traj
                            if info_tracker: info_tracker.record_revised_submission(b_traj)
                            
                            # Update Blocker
                            current_agent_plans[bl_idx] = bl_plan
                            current_trajectories[bl_idx] = bl_traj
                            if info_tracker: info_tracker.record_revised_submission(bl_traj)
                            
                            pass_log["events"].append(f"Smart Coordinated Yield: Agent {bl_idx+1} yields for {b_idx+1}")
                            
                            coord_yield_success = True; any_fix_this_pass_iteration = True; run_counters['replan_success_coord_yield'] = run_counters.get('replan_success_coord_yield', 0) + 1
                            if verbose: print(f"    Committed SMART COORDINATED YIELD: Agent {bl_idx+1} yields for {b_idx+1}.")
            
            if coord_yield_success: break
            if any_fix_this_pass_iteration: continue


            # --- Strategy 4: Defer ---
            if info_setting == "all":
                if not static_success and not dyn_success and not general_yield_success:
                    # Defer if multiple attempts for this specific collision failed, or if it's a persistent issue.
                    # Original logic used coll_attempt_num >= 2.
                    if coll_attempt_num >= 2 : 
                        if verbose: print(f"  All strategies failed for {coll_key_tuple}. Deferring agents: {current_coll_active_agents_list}")
                        deferred_this_coll_count = 0
                        for agent_id_defer in current_coll_active_agents_list:
                            idx_defer = agent_id_defer - 1
                            if idx_defer not in deferred_agents: # Defer only if not already deferred in this overall attempt
                                
                                if info_tracker: info_tracker.record_defer_alert()
                                # <<< LOGGING: Deferral Alert >>>
                                log_context["alert_id_counter"] += 1
                                alert_id_def = f"Alert-{log_context['alert_id_counter']}"
                                cell_str_ref_def = str(coll['cell']).replace(" ", "")
                                coll_id_ref_def = f"Collision-{coll['time']}-{cell_str_ref_def}"
                                
                                defer_alert_log = {
                                    "id": alert_id_def, "alertsConflict": coll_id_ref_def,
                                    "targetAgent": f"Robot-{agent_id_defer}", "alertType": "defer",
                                    "rationale": f"Agent deferred after {coll_attempt_num} failed attempts to resolve collision."
                                }
                                log_data["conflictAlerts"].append(defer_alert_log)
                                pass_log["events"].append(f"Agent {agent_id_defer} was deferred.")
                                # <<< END LOGGING >>>                                
                                
                                deferred_agents.add(idx_defer)
                                current_agent_plans[idx_defer] = [] # Clear plan
                                current_trajectories[idx_defer] = [agent_starts[idx_defer]] # Agent waits at start
                                run_counters['agents_deferred'] += 1
                                deferred_this_coll_count +=1
                                if verbose: print(f"    Agent {agent_id_defer} DEFERRED at {agent_starts[idx_defer]}.")
                        
                        if deferred_this_coll_count > 0:
                            any_fix_this_pass_iteration = True 
                            # If deferral happened, it's a significant change, so break from processing more collisions in this pass
                            # to re-evaluate the situation in the next pass or iteration.
                            break # Break from THIS collision's strategies, and also implies breaking from sorted_colls_to_process loop due to next check
            
            # If a deferral happened (which sets any_fix_this_pass_iteration),
            # and it was due to all prior strategies failing for *this* collision,
            # we should break from the loop over `sorted_colls_to_process` to restart the pass
            # because the set of active agents and their trajectories has changed significantly.
            if any_fix_this_pass_iteration and not static_success and not dyn_success and not pivot_success:
                if info_setting == "all" and coll_attempt_num >=2 : # This implies deferral might have happened
                     if verbose: print(f"Deferral occurred for collision involving agents {current_coll_active_agents_list}. Breaking inner pass to re-evaluate.")
                     break # Break from the sorted_colls_to_process loop


        if not any_fix_this_pass_iteration and pass_num > 1 : # If no fix in the entire pass (and not the first pass)
            if verbose: print(f"No fixes made for any collision in this inner pass {pass_num}. Proceeding to deferred or next overall attempt.")
            break # Break from inner passes loop
        
        if current_elapsed_time > time_limit : break # Check time limit after each pass's collision processing

    return current_elapsed_time > time_limit

def _plan_for_deferred_agents(
    current_agent_plans, current_trajectories, agent_envs, model, run_counters, device,
    agent_goals, agent_starts, pristine_static_grid, deferred_agents,
    search_type, algo, timeout, heuristic_weight, max_expansions,
    overall_fix_attempt_count, verbose,
    # <<< LOGGING ARGUMENTS >>>
    log_data, log_context
    # <<< END LOGGING >>>
):
    """
    Plans paths for deferred agents sequentially.
    Modifies current_agent_plans and current_trajectories in place.
    """
    import random
    import copy

    num_agents = len(current_agent_plans)
    if not deferred_agents:
        return

    # <<< LOGGING: Deferred Planning Strategy >>>
    log_context["strategy_id_counter"] += 1
    deferred_strategy_id = f"Strategy-{log_context['strategy_id_counter']}"
    deferred_strategy_log = {"id": deferred_strategy_id, "type": "Sequential Deferred Planning"}
    log_data["replanningStrategies"].append(deferred_strategy_log)
    # <<< END LOGGING >>>

    if verbose:
        print(f"\n--- Handling Deferred Agents Sequentially "
              f"({len(deferred_agents)} agents) for Overall Attempt #{overall_fix_attempt_count} ---")

    # Compute when each non-deferred agent first hits its goal
    active_non_deferred_trajs = [
        current_trajectories[i]
        for i in range(num_agents)
        if i not in deferred_agents and current_trajectories[i]
    ]

    first_goal_times = []
    for i, traj in enumerate(current_trajectories):
        if i in deferred_agents or not traj:
            continue
        for t, pos in enumerate(traj):
            if tuple(map(int, pos)) == agent_goals[i]:
                first_goal_times.append(t)
                break

    if first_goal_times:
        base_wait = max(first_goal_times)
    elif active_non_deferred_trajs:
        # fallback: longest raw trajectory if nobody reached goal
        base_wait = max(len(traj) - 1 for traj in active_non_deferred_trajs)
    else:
        base_wait = 0

    if verbose:
        print(f"  Non-deferred agents' effective makespan (wait steps for deferred): {base_wait}")

    # Plan each deferred agent with the same fixed wait
    sorted_deferred_agent_indices = sorted(deferred_agents)
    random.shuffle(sorted_deferred_agent_indices)

    for def_idx in sorted_deferred_agent_indices:
        start_pos = agent_starts[def_idx]
        goal_pos = agent_goals[def_idx]
        if verbose:
            print(f"  Planning for deferred Agent {def_idx + 1} "
                  f"from {start_pos} to {goal_pos} (after {base_wait} wait steps).")

        # Build a fresh copy of the environment
        env_def = copy.deepcopy(agent_envs[def_idx])
        env_def.env.grid = pristine_static_grid.copy()
        env_def.env.agent_pos = start_pos
        env_def.env.goal_pos = goal_pos

        # Mark start & goal on the grid
        if env_def.env.grid[start_pos] != -1:
            env_def.env.grid[start_pos] = 1
        else:
            if verbose:
                print(f"    Deferred Agent {def_idx+1} starts on a static obstacle. Cannot plan.")
            current_agent_plans[def_idx] = [4] * (base_wait + 1)
            current_trajectories[def_idx] = [start_pos] * (base_wait + 1)
            continue

        if goal_pos != start_pos and env_def.env.grid[goal_pos] != -1:
            env_def.env.grid[goal_pos] = 2
        elif goal_pos == start_pos and env_def.env.grid[start_pos] == 1:
            env_def.env.grid[start_pos] = 3

        # Run A* search for this deferred agent
        astar_plan_segment = plan_with_search(
            env_def, model, device,
            search_type, algo,
            timeout * 2, heuristic_weight,
            max_expansions * 2
        )

        # Fixed wait before movement
        wait_plan_actions = [4] * base_wait
        wait_trajectory_positions = [start_pos] * base_wait

        final_plan = wait_plan_actions.copy()
        final_traj = wait_trajectory_positions.copy()

        if astar_plan_segment is not None:
            final_plan.extend(astar_plan_segment)

            # Simulate trajectory of the A* segment
            sim_env = copy.deepcopy(agent_envs[def_idx])
            sim_env.env.grid = pristine_static_grid.copy()
            sim_env.env.agent_pos = start_pos
            astar_traj = simulate_plan(sim_env, astar_plan_segment)

            # --- Check for blocking finished agents ---
            blocking_finished_agent = None
            if astar_traj:
                full_traj_check = wait_trajectory_positions + astar_traj
                for t, pos in enumerate(full_traj_check):
                    if t < base_wait: continue # Ignore wait period (assumed safe-ish or handled by base_wait)
                    pos_tuple = tuple(map(int, pos))
                    
                    for other_idx, other_traj in enumerate(current_trajectories):
                        if other_idx == def_idx or not other_traj: continue
                        
                        # Is other agent finished?
                        other_goal = agent_goals[other_idx]
                        if len(other_traj) > 0 and tuple(map(int, other_traj[-1])) == other_goal:
                             # Check if it stays at goal/current pos
                             if t < len(other_traj):
                                 other_pos = tuple(map(int, other_traj[t]))
                             else:
                                 other_pos = tuple(map(int, other_traj[-1]))
                             
                             if other_pos == pos_tuple:
                                 blocking_finished_agent = other_idx
                                 break
                    if blocking_finished_agent is not None: break
            
            if blocking_finished_agent is not None:
                if verbose: print(f"    Agent {def_idx+1} path blocked by finished Agent {blocking_finished_agent+1}. Forcing yield.")
                
                yield_idx = blocking_finished_agent
                goal_pos_yield = agent_goals[yield_idx]
                
                # Find neighbor
                current_grid = pristine_static_grid
                rows, cols = current_grid.shape
                neighbors = []
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = goal_pos_yield[0]+dr, goal_pos_yield[1]+dc
                    if 0 <= nr < rows and 0 <= nc < cols and current_grid[nr, nc] == 0:
                        neighbors.append((nr, nc))
                
                if neighbors:
                    yield_cell = random.choice(neighbors)
                    wait_steps_yield = 5
                    
                    def get_action(p1, p2):
                        d = (p2[0]-p1[0], p2[1]-p1[1])
                        if d == (-1, 0): return 0 
                        if d == (1, 0): return 1 
                        if d == (0, -1): return 2 
                        if d == (0, 1): return 3 
                        return 4 
                        
                    a_out = get_action(goal_pos_yield, yield_cell)
                    a_wait = [4] * wait_steps_yield
                    a_back = get_action(yield_cell, goal_pos_yield)
                    
                    detour = [a_out] + a_wait + [a_back]
                    
                    # Apply yield to blocking agent
                    current_agent_plans[yield_idx].extend(detour)
                    sim_env_yield = copy.deepcopy(agent_envs[yield_idx])
                    sim_env_yield.env.agent_pos = agent_starts[yield_idx]
                    new_traj_yield = simulate_plan(sim_env_yield, current_agent_plans[yield_idx])
                    
                    # VALIDATE: Check if yield detour would hit obstacles
                    invalid_yield = False
                    for traj_pos in new_traj_yield:
                        r, c = int(traj_pos[0]), int(traj_pos[1])
                        if 0 <= r < pristine_static_grid.shape[0] and 0 <= c < pristine_static_grid.shape[1]:
                            if pristine_static_grid[r, c] == -1:
                                if verbose: print(f"      Deferred yield for Agent {yield_idx+1} would hit obstacle at {(r,c)}. Skipping this yield.")
                                invalid_yield = True
                                # Roll back the plan extension
                                current_agent_plans[yield_idx] = current_agent_plans[yield_idx][:-len(detour)]
                                break
                    
                    if not invalid_yield:
                        current_trajectories[yield_idx] = new_traj_yield
                        if verbose: print(f"    Agent {yield_idx+1} yields to clear path.")

            if astar_traj:
                # Append without duplicating the first position
                if final_traj and tuple(map(int, astar_traj[0])) == final_traj[-1]:
                    final_traj.extend(tuple(map(int, p)) for p in astar_traj[1:])
                else:
                    final_traj.extend(tuple(map(int, p)) for p in astar_traj)
            else:
                # If planning succeeded but simulation failed, just wait one step at goal
                final_traj.append(start_pos)

            # <<< LOGGING: New plan for deferred agent >>>
            log_context["subplan_id_counter"] += 1
            deferred_plan_id = f"Robot-{def_idx+1}-Plan-Resolved-{log_context['subplan_id_counter']}"
            plan_log_def = {
                "id": deferred_plan_id, "belongsToAgent": f"Robot-{def_idx+1}",
                "generatedBy": deferred_strategy_id, # Linked to the deferred strategy
                "planCost": len(final_plan),
                "steps": [{"time": t, "cell": [int(p[0]), int(p[1])]} for t, p in enumerate(final_traj)]
            }
            log_data["agentSubplans"].append(plan_log_def)
            # <<< END LOGGING >>>
        else:
            if verbose:
                print(f"    Failed to find A* plan for deferred Agent {def_idx + 1}. Will only wait.")
            final_plan.append(4)
            final_traj.append(start_pos)

        # Commit
        current_agent_plans[def_idx] = final_plan
        current_trajectories[def_idx] = [tuple(map(int, p)) for p in final_traj]

        reached = final_traj and tuple(map(int, final_traj[-1])) == goal_pos
        if verbose:
            print(f"    Planned for deferred Agent {def_idx + 1}. "
                  f"Plan length: {len(final_plan)}. Reached goal: {reached}")


def fix_collisions(initial_agent_plans,
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
                   max_expansions=200,
                   time_limit=60,
                   max_passes=50, 
                   verbose=True):
    """
    Main orchestrator for collision resolution.
    Includes an outer loop to re-run the entire process if collisions persist.
    """
    overall_start_time = time.perf_counter()
    
    if run_counters is None: run_counters = {}
    for key in ['replan_attempts_static', 'replan_success_static', 
                'replan_attempts_dynamic', 'replan_success_dynamic', 
                'replan_attempts_pivot', 'replan_success_pivot', 'agents_deferred',
                'collisions_total']: 
        run_counters.setdefault(key, 0)

    current_agent_plans_main = [list(p) for p in initial_agent_plans] 
    current_trajectories_main = [list(t) for t in initial_agent_trajectories] 
    
    num_agents = len(current_agent_plans_main)
    agent_goals = [tuple(map(int,env.env.goal_pos)) for env in agent_envs]
    agent_starts = [tuple(map(int,env.env.agent_pos)) for env in agent_envs]

    pristine_static_grid = agent_envs[0].env.grid.copy()
    for r_idx in range(pristine_static_grid.shape[0]):
        for c_idx in range(pristine_static_grid.shape[1]):
            if pristine_static_grid[r_idx,c_idx] != -1: 
                pristine_static_grid[r_idx,c_idx] = 0
    
    # TRAFFIC-AWARE: Precompute cell degrees for structure-aware yield
    cell_degrees = compute_cell_degrees(pristine_static_grid)
    if verbose: print(f"  Computed cell degrees for {pristine_static_grid.shape} grid")
    
    # COOPERATIVE PROTOCOLS: Detect corridor segments
    corridor_map = detect_corridors(cell_degrees, pristine_static_grid)
    if verbose: print(f"  Detected {len(corridor_map)} corridor segments")
    for cid, cinfo in corridor_map.items():
        if verbose: print(f"    Corridor {cid}: {cinfo['length']} cells, entrances={cinfo['entrances']}, exits={cinfo['exits']}")
    
    # COOPERATIVE PROTOCOLS: Initialize yield contracts and tracking
    yield_contracts = {}  # agent_id -> contract dict
    deep_yield_count = defaultdict(int)  # agent_id -> count of deep yields
    forbidden_regions = defaultdict(set)  # agent_id -> set of forbidden cells
    
    # CENTRAL COORDINATOR: Initialize for global decision-making
    coordinator = CentralCoordinator(pristine_static_grid.shape)
    if verbose: print(f"  Initialized Central Coordinator for global coordination")

    info_tracker = InfoSharingTracker()
    # <<< Record the first information sharing event >>>
    info_tracker.record_initial_submission(initial_agent_trajectories)

    # <<< LOGGING: Initialize the main log object >>>
    log_data = {
        "environment": {
            "id": "SimulatedWarehouseEnv", "gridSize": [pristine_static_grid.shape[1], pristine_static_grid.shape[0]],
            "obstacles": [{"id": f"Obs-{r}-{c}", "cell": [int(r), int(c)]} for r, c in np.argwhere(pristine_static_grid == -1)]
        },
        "agents": [
            {"id": f"Robot-{i+1}", "initialState": {"time": 0, "cell": [int(p[0]), int(p[1])]}, "goalState": {"cell": [int(g[0]), int(g[1])]}}
            for i, (p, g) in enumerate(zip(agent_starts, agent_goals))
        ],
        "agentPaths": [
            { "agent": f"Robot-{i+1}", "subplanId": f"Robot-{i+1}-Plan-Original", "planCost": len(p) if len(p)>0 else 0,
              "steps": [{"time": t, "cell": [int(pos[0]), int(pos[1])]} for t, pos in enumerate(traj)]}
            for i, (p, traj) in enumerate(zip(initial_agent_plans, initial_agent_trajectories)) if p
        ],
        "execution_trace": {"attempts": []}, "collisionEvents": [], "conflictAlerts": [],
        "replanningStrategies": [], "agentSubplans": [], "jointPlan": {}
    }
    log_context = {"alert_id_counter": 0, "strategy_id_counter": 0, "subplan_id_counter": 0}
    # <<< END LOGGING >>>

    unique_colls_ever = set() 
    
    # Define constants for rewind logic to be passed to helper
    INIT_REWIND_CONST = 3 
    MAX_REWIND_STATIC_CONST = 7
    MAX_REWIND_DYN_CONST = 7
    
    max_overall_fix_attempts = 5
    overall_fix_attempt_count = 0
    timed_out_overall = False

    while overall_fix_attempt_count < max_overall_fix_attempts:
        overall_fix_attempt_count += 1

        # <<< LOGGING: Start of an overall attempt >>>
        log_data["execution_trace"]["attempts"].append({"attempt_number": overall_fix_attempt_count, "passes": []})
        # <<< END LOGGING >>>

        if verbose: print(f"\n<<<<< Overall Fix Attempt #{overall_fix_attempt_count} >>>>>")

        static_block_hist = {i: set() for i in range(num_agents)} 
        dyn_obs_hist = {i: [] for i in range(num_agents)} 
        pivot_attempts_hist = {i: 0 for i in range(num_agents)} 
        deferred_agents_this_attempt = set() 
        coll_specific_attempts = defaultdict(int)
        
        # COORDINATOR: Update global state for this attempt
        coordinator.update_heatmap(current_trajectories_main)
        coordinator.identify_safe_corners(pristine_static_grid, cell_degrees)
        if verbose: print(f"  Coordinator: Updated heatmap, identified {len(coordinator.safe_corners) if coordinator.safe_corners else 0} safe corners")

        timeout_occurred_in_passes = _handle_active_collision_passes(
            current_agent_plans_main, current_trajectories_main, agent_envs, model, run_counters, device,
            agent_goals, agent_starts, pristine_static_grid, cell_degrees, corridor_map, coordinator,
            yield_contracts, deep_yield_count, forbidden_regions,
            static_block_hist, dyn_obs_hist, pivot_attempts_hist, deferred_agents_this_attempt,
            coll_specific_attempts, unique_colls_ever,
            replan_strategy, search_type, algo, timeout, heuristic_weight, max_expansions,
            max_passes, time_limit, overall_start_time, overall_fix_attempt_count, verbose,
            log_data, log_context,
            INIT_REWIND_CONST, MAX_REWIND_STATIC_CONST, MAX_REWIND_DYN_CONST , info_setting=info_setting,
            info_tracker=info_tracker
        )
        if timeout_occurred_in_passes: # Check the return value from the helper
            timed_out_overall = True; break 

        if not timed_out_overall and deferred_agents_this_attempt: # Only plan if not timed out
            _plan_for_deferred_agents(
                current_agent_plans_main, current_trajectories_main, agent_envs, model, run_counters, device,
                agent_goals, agent_starts, pristine_static_grid, deferred_agents_this_attempt,
                search_type, algo, timeout, heuristic_weight, max_expansions,
                overall_fix_attempt_count, verbose, 
                log_data, log_context
            )
        
        final_colls_after_this_major_attempt = analyze_collisions(current_trajectories_main, agent_goals, pristine_static_grid)
        if verbose:
            print(f"--- Collisions after Overall Attempt #{overall_fix_attempt_count} (including deferred): {len(final_colls_after_this_major_attempt)} ---")
            if final_colls_after_this_major_attempt: print(f"    Details: {final_colls_after_this_major_attempt[:min(3, len(final_colls_after_this_major_attempt))]} ...")

        if not final_colls_after_this_major_attempt:
            if verbose: print("All collisions resolved in this overall attempt.")
            break 
        
        current_elapsed_time_after_major_attempt = time.perf_counter() - overall_start_time
        if current_elapsed_time_after_major_attempt > time_limit:
            if verbose: print("Overall time limit reached after this major attempt."); timed_out_overall = True
            break 
        if overall_fix_attempt_count >= max_overall_fix_attempts:
            if verbose: print("Max overall fix attempts reached. Collisions may persist.")
            break 
        if verbose: print("Collisions persist. Will attempt another overall fix cycle if limits allow.")
    
    total_overall_time = time.perf_counter() - overall_start_time
    final_check_collisions = analyze_collisions(current_trajectories_main, agent_goals, pristine_static_grid) 

    if verbose:
        info_tracker.report()

    info_tracker.report()
    # <<< Add the metrics to the final log file >>>
    log_data["informationSharingMetrics"] = info_tracker.to_dict()
    
    if verbose:
        print(f"\nfix_collisions finished. Total time: {total_overall_time:.2f}s. Overall Attempts: {overall_fix_attempt_count}")
        print(f"Final collisions reported by fix_collisions: {len(final_check_collisions)}")

        if final_check_collisions: print(f"    Details: {final_check_collisions[:min(3, len(final_check_collisions))]} ...")

        print(f"Unique collisions ever encountered: {len(unique_colls_ever)}")
        print(f"CollisionsTotalDetected (sum over passes): {run_counters['collisions_total']}")
        print(f"Static S/A: {run_counters['replan_success_static']}/{run_counters['replan_attempts_static']}. Dynamic S/A: {run_counters['replan_success_dynamic']}/{run_counters['replan_attempts_dynamic']}. Pivot S/A: {run_counters['replan_success_pivot']}/{run_counters['replan_attempts_pivot']}. Deferred: {run_counters['agents_deferred']}")

    # # <<< LOGGING: Finalize and save the log file >>>
    # final_plans_map = {}
    # for path in log_data["agentPaths"]: final_plans_map[path["agent"]] = path["subplanId"]
    # for subplan in log_data["agentSubplans"]: final_plans_map[subplan["belongsToAgent"]] = subplan["id"]

    # log_data["jointPlan"] = {
    #     "id": "FinalJointPlan",
    #     "subplans": [final_plans_map.get(f"Robot-{i+1}", f"Robot-{i+1}-Plan-Original") for i in range(num_agents)],
    #     "globalMakespan": max((len(t) for t in current_trajectories_main if t), default=0)
    # }

    # if log_filepath:
    #     try:
    #         with open(log_filepath, 'w') as f:
    #             json.dump(log_data, f, indent=2)
    #         if verbose: print(f"✔ Detailed execution log saved to '{log_filepath}'")
    #     except Exception as e:
    #         if verbose: print(f"Error saving log file: {e}")
    # # <<< END LOGGING >>>

    return current_agent_plans_main, current_trajectories_main, len(unique_colls_ever), timed_out_overall, log_data
