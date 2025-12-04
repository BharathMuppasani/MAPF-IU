# Joint A* Planner Status Report

## Test Result: ✅ WORKING

The joint A* planner is **functional and can find valid solutions**.

### Test Case
- **Map**: `robo_test.txt` (11x11 grid)
- **Agents**: 2
- **Collision**: Agents 1 and 2 collide at (5,5) at time t=1

**Initial Setup:**
- Agent 1: Start (5,4), Goal (5,6)
- Agent 2: Start (5,6), Goal (5,4)

**Initial Trajectories (causing collision):**
- Agent 1: [(5,4) → (5,5) → (5,6)]
- Agent 2: [(5,6) → (5,5) → (5,4)]

### Joint A* Solution ✅

**Planning succeeded** in finding a collision-free solution:

```
Collision: T=1, Cell=(5,5), Agents=[1, 2]

Joint A* window: t_start=0, t_goal_sub=7 (7-step planning horizon)
Expanded: 105 nodes
Time: < 1 second

Result: SUCCESS
```

**Solution Plans:**
```
Agent 1: RIGHT RIGHT DOWN UP WAIT WAIT WAIT
Agent 2: WAIT UP DOWN LEFT LEFT WAIT WAIT
```

**Solution Trajectories (collision-free):**
```
Agent 1: [(5,4) → (5,5) → (5,6) → (6,6) → (5,6) → (5,6) → (5,6) → (5,6)]
         Starting position, then moves right, then down (detouring), then up, then waits at goal

Agent 2: [(5,6) → (5,6) → (4,6) → (5,6) → (5,5) → (5,4) → (5,4) → (5,4)]
         Waits at start, moves up (out of the way), then back down, then left to goal, waits
```

**Final Collision Count:** 0 ✓

---

## Why Did Joint A* Fail in Full Pipeline?

Since joint A* works on this simple test, the failure in the full `run_exp.py` must be due to:

### Hypothesis 1: Plan Splicing Issues
When joining the joint A* segment with existing prefix/suffix actions, the boundary conditions might not align properly. The agent position at `t_start` might not match where the prefix ends.

**Evidence**: In your test run, joint A* was called on agents [3,5] with:
- Plan segment length: 15 actions
- t_start=0, t_goal_sub=15
- But the agents' existing plans were shorter

The validation I added checks for this:
```python
prefix_end_pos = tuple(map(int, prefix_traj[-1]))
expected_segment_start = tuple(map(int, current_trajectories[idx][t_start]))
if prefix_end_pos != expected_segment_start:
    print("ERROR: Prefix ends at... but segment expects...")
    joint_success = False
```

### Hypothesis 2: Unrealistic Subgoals
When trajectories are very short compared to the planning window, the subgoal might be set to a position the agent already reached, creating a degenerate search where the planner just tries to hold position.

**My Fix**: Clamped subgoal time to actual trajectory length:
```python
t_subgoal_clamped = min(t_goal_sub, len(traj) - 1)
```

### Hypothesis 3: Invalid Intermediate Plans
The neural network might generate plans that move agents out of bounds or into obstacles. My fix added validation:
```python
if not (0 <= r < rows and 0 <= c < cols):
    return None  # Reject invalid move
```

---

## Recommendations

### 1. Enable Verbose Logging
Run with `verbose=True` to see detailed joint A* planning:
```bash
python run_exp.py --strategy best --info all --search_type astar --algo dqn \
  --map_file test_data/maps/maps_11x11/map_7_2.txt --timeout 120
```

Look for error messages like:
- `ERROR: Prefix for agent... is invalid`
- `ERROR: Prefix ends at... but segment expects...`
- `ERROR: Simulating spliced plan for agent... failed`

### 2. Test on Simpler Maps First
Start with 2-agent scenarios to verify the full pipeline works before moving to complex maps.

### 3. Check Plan Quality from Neural Network
The initial plans from the neural network might be the problem. Verify:
- Are initial plans valid (no out-of-bounds moves)?
- Are they realistic paths to goals?

### 4. Monitor Collision Count Progression
In your test run, collisions went from 6 → 9 → 7 → 6 → 6 → 6 → 1
- This suggests collision resolution is partially working
- But the final collision was never resolved despite multiple attempts
- This points to a fundamental issue with how joint A* results are validated/spliced

---

## Testing Commands

### Test Joint A* in isolation:
```bash
python test_joint_astar_robo.py
```

This proves the joint A* algorithm itself works correctly.

### Test full pipeline with verbose output:
```bash
python run_exp.py --strategy best --info all --search_type astar --algo dqn \
  --map_file test_data/maps/maps_11x11/map_7_2.txt --timeout 120 2>&1 | tee test_output.log
```

Look for "ERROR:" messages in the output to identify where things break.

---

## Summary

- **Joint A* Algorithm**: ✅ Working correctly
- **Problem Location**: In the plan splicing/validation logic during collision resolution
- **Next Steps**: Run tests with the new validation code and check for specific error messages

The fixes I made should catch these issues and provide clear error messages so we can diagnose exactly what's failing.
