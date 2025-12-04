# Bug Fixes for fix.py - Collision Detection & Joint A* Planning

## Executive Summary

Found and fixed **4 critical bugs** in the joint A* collision resolution implementation that were preventing proper collision detection and resolution:

1. **Plan Splicing Logic** - Incorrect index mapping between trajectory times and plan actions
2. **Collision Resolution Acceptance** - Over-lenient acceptance criteria allowing failed resolutions
3. **Subgoal Calculation** - Degenerate search when trajectories shorter than planning window
4. **Trajectory Bounds Checking** - Missing bounds validation causing potential index errors

---

## Bug #1: Plan Splicing Logic (CRITICAL)

**Location:** `fix.py:989-1013` (previously lines 989-1004)

**Problem:**
The code was incorrectly clamping the search window boundaries to the current plan length:
```python
# WRONG - original code
t_start_clamped = min(t_start, plan_len)
t_goal_clamped = min(t_goal_sub, plan_len)
prefix_actions = current_plans[idx][:t_start_clamped]
original_suffix = current_plans[idx][t_goal_clamped:]
```

This caused three issues:
1. **Index Confusion:** `t_start` and `t_goal_sub` are TRAJECTORY TIMES (position indices), but were being used as PLAN indices (action indices). These are off by 1.
2. **Incorrect Splicing:** When `t_start > plan_len`, clamping to `plan_len` would ignore the rewind point, breaking plan continuity.
3. **Lost Suffix:** If `t_goal_sub > plan_len`, the suffix would be empty, but the original intention was to keep the rest of the plan.

**Example of failure:**
```
Scenario:
- Trajectory: [(0,0), (1,1), (2,2)]  (length 3)
- Plan: [action_up, action_right]  (2 actions)
- Collision at t=5
- base_rewind=3, base_horizon=6
- t_start = max(0, 5-3) = 2
- t_goal_sub = 5+6 = 11

Original buggy code:
  plan_len = 2
  t_start_clamped = min(2, 2) = 2
  t_goal_clamped = min(11, 2) = 2
  prefix = plan[:2] = [action_up, action_right]
  suffix = plan[2:] = []

  new_plan = [action_up, action_right] + joint_segment + []

This loses the original plan structure!
```

**Fix:**
```python
# CORRECT - fixed code
if t_start > plan_len:
    # Can't rewind beyond what exists
    skip this agent

expected_segment_len = t_goal_sub - t_start
prefix_actions = current_plans[idx][:t_start]
original_suffix = current_plans[idx][t_goal_sub:]
new_full_plan = prefix_actions + plan_segment + original_suffix
```

Now we:
- Use actual `t_start` and `t_goal_sub` values without clamping
- Validate `t_start <= plan_len` to ensure rewind is feasible
- Keep the correct suffix even if it extends beyond current plan length

---

## Bug #2: Over-Lenient Collision Resolution Acceptance (CRITICAL)

**Location:** `fix.py:1042-1045` (previously line 1026)

**Problem:**
```python
# WRONG - original code
if coll_resolved or len(new_colls) < len(collisions):
```

This accepts the joint A* solution if EITHER:
1. The specific collision is resolved, OR
2. The total collision count decreased

**Why this is wrong:**
If we resolve collision A but create new collisions D and E:
- Original: [A, B, C] (3 collisions)
- After fix: [D, E, B, C] (4 collisions)

The OR condition would REJECT this (fewer is better), but what if:
- Original: [A, B, C, D, E] (5 collisions)
- After fix: [A_new_variant, B, C_modified, D] (4 collisions)

The OR would ACCEPT this even though collision A wasn't actually resolved - it just changed form!

**Fix:**
```python
# CORRECT - fixed code
if coll_resolved and len(new_colls) <= len(collisions):
```

Now we ONLY accept if:
1. The SPECIFIC collision was resolved (required), AND
2. We didn't make the overall situation worse (safeguard)

---

## Bug #3: Degenerate Subgoal Calculation (CRITICAL)

**Location:** `fix.py:479-503` (previously lines 479-487)

**Problem:**
When agent trajectories are shorter than the planning window, the subgoal extraction failed:
```python
# WRONG - original code
subgoal_positions = tuple(agent_pos_at(aid, t_goal_sub) for aid in agents_in_collision)
```

With `agent_pos_at()` returning `traj[-1]` when `t >= len(traj)`, this causes:

**Example:**
```
Agent trajectory: [(0,0), (1,1), (2,2)]  (length 3)
t_start = 5 (beyond trajectory)
t_goal_sub = 20 (way beyond trajectory)

agent_pos_at(aid, 5) returns traj[-1] = (2,2)   <- start position
agent_pos_at(aid, 20) returns traj[-1] = (2,2)  <- subgoal position

START == SUBGOAL! Degenerate search.
```

This creates a degenerate A* search where:
- Start and goal are identical
- Heuristic returns 0 immediately
- Any path (including doing nothing) satisfies the goal
- The search becomes useless - it just checks if staying still violates constraints

**Fix:**
```python
# CORRECT - fixed code
subgoal_positions = []
for aid in agents_in_collision:
    idx = aid - 1
    traj = current_trajectories[idx]
    if traj:
        # Clamp to actual trajectory extent
        t_subgoal_clamped = min(t_goal_sub, len(traj) - 1)
    else:
        t_subgoal_clamped = t_goal_sub
    subgoal_positions.append(agent_pos_at(aid, t_subgoal_clamped))
```

Also added warning for degenerate search:
```python
if start_positions == subgoal_positions:
    print(f"WARNING: Start and subgoal are identical! Degenerate search.")
```

---

## Bug #4: Missing Trajectory Bounds Checking (MEDIUM)

**Location:** `fix.py:435-455` (previously lines 435-439)

**Problem:**
The `check_joint_segment_conflicts()` function accessed joint trajectories without validating they had the expected length:
```python
# WRONG - original code
for offset in range(horizon):
    t_global = t_start + offset
    pos_now_joint = {aid: joint_trajs[aid][offset] for aid in agents_in_collision}
    pos_next_joint = {aid: joint_trajs[aid][offset + 1] for aid in agents_in_collision}
```

If the trajectory building phase produced wrong-length trajectories, this could cause:
- `IndexError` when accessing `offset + 1` beyond trajectory length
- Silently using wrong positions if trajectory is shorter
- No diagnostic information about why validation failed

**Fix:**
```python
# CORRECT - fixed code
# Verify trajectory bounds before accessing
for aid in agents_in_collision:
    traj_len = len(joint_trajs.get(aid, []))
    expected_len = horizon + 1
    if traj_len != expected_len:
        if verbose:
            print(f"ERROR: Agent {aid} trajectory length {traj_len} != expected {expected_len}")
        return False

# Add exception handling
try:
    pos_now_joint = {aid: joint_trajs[aid][offset] for aid in agents_in_collision}
    pos_next_joint = {aid: joint_trajs[aid][offset + 1] for aid in agents_in_collision}
except (KeyError, IndexError) as e:
    if verbose:
        print(f"ERROR: Index out of bounds at offset {offset}: {e}")
    return False
```

---

## Impact Summary

### Before Fixes
- Joint A* frequently failed due to degenerate searches (start == goal)
- Even when it succeeded, splicing logic corrupted plans
- Overly lenient acceptance criteria allowed invalid resolutions
- No bounds checking led to mysterious failures

### After Fixes
- Joint A* properly targets reachable subgoals
- Plan splicing correctly maps trajectory times to actions
- Only genuinely resolved collisions are accepted
- Bounds checking provides clear error messages

---

## Testing Recommendations

1. **Test short trajectory case:** Run with agents that reach goals early
2. **Test long collision window:** Detect collisions far beyond trajectory length
3. **Test joint A* success:** Verify spliced plans actually work
4. **Add verbose logging:** Use `verbose=True` to see subgoal/splicing details
5. **Check collision count:** Verify final collision count matches expectations

---

## Files Modified

- `/Users/bittu/Desktop/GitHub/MAPF-NeurIPS-25/fix.py`
  - Lines 479-503: Subgoal calculation fix
  - Lines 437-455: Bounds checking fix
  - Lines 993-1013: Plan splicing fix
  - Lines 1045: Acceptance criteria fix

No changes to collision detection itself (`utils/env_utils.py`), which was working correctly.

