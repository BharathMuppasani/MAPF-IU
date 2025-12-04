# Multi-Agent Pathfinding (MAPF) Collision Resolution Report

## Overview

This document describes the three-tier collision resolution strategy implemented in `fix.py` for resolving path conflicts in multi-agent pathfinding. The system uses a progressive escalation approach: attempting simple, local solutions first, then escalating to complex joint planning when necessary.

---

## 1. Three-Tier Collision Resolution Strategy

### Tier Architecture

```
Collision Detected
       ↓
┌─────────────────────────────────────┐
│ TIER 1: STATIC BLOCKING             │
│ • Block collision cells as obstacles│
│ • Single agent replans              │
│ • Minimal communication overhead    │
│ • Attempt: ONCE per collision       │
└─────────────────────────────────────┘
       ↓ (if fails)
┌─────────────────────────────────────┐
│ TIER 2: DYNAMIC OBSTACLE SHARING    │
│ • Share blocker's trajectory        │
│ • Single agent replans around it    │
│ • Moderate communication overhead   │
│ • Attempt: ONCE per collision       │
└─────────────────────────────────────┘
       ↓ (if fails)
┌─────────────────────────────────────┐
│ TIER 3: JOINT A* PLANNING           │
│ • Jointly plan all involved agents  │
│ • Configuration space search        │
│ • High communication overhead       │
│ • Attempt: MULTIPLE window expansions│
└─────────────────────────────────────┘
       ↓ (if fails)
┌─────────────────────────────────────┐
│ TIER 4: DEFER STRATEGY              │
│ • Defer one agent to wait           │
│ • Replan after others clear         │
│ • Last resort coordination          │
└─────────────────────────────────────┘
```

---

## 2. Detailed Strategy Descriptions

### TIER 1: Static Blocking Strategy

**Concept:** Treat collision cell(s) as temporary obstacles and replan.

**Algorithm:**
```python
1. Identify collision cells from detected collisions
2. Create new grid with collision cells marked as -1 (obstacle)
3. Rewind from collision time: t_rewind = max(0, collision_time - INIT_REWIND)
4. Get agent's position at t_rewind
5. Run A* search from this position to goal with blocked cells
6. Splice plan: prefix (before rewind) + new_segment + suffix (after collision time)
7. Simulate and verify collision is resolved
```

**Parameters:**
- `INIT_REWIND = 3`: Steps to look back from collision
- `MAX_REWIND_STATIC = 7`: Maximum rewind allowed
- Attempted once per unique collision

**Communication Overhead:**
- IU = number of blocked cells
- Example: Blocking (5,5) = 1 IU

**When it works:** Agent can bypass collision cell by taking a nearby alternative path

**When it fails:**
- Collision cell is critical (no detours available)
- Blocking the cell creates worse collisions elsewhere

---

### TIER 2: Dynamic Obstacle Sharing Strategy

**Concept:** Share the blocker agent's future trajectory, then replan around it.

**Algorithm:**
```python
1. Identify blocking agent's trajectory at collision time
2. Extract blocker's path segment: [t_rewind, t_rewind + 2*rewind]
3. Set this trajectory as a moving obstacle in the environment
4. Run A* for replan agent with dynamic obstacle avoidance
5. Splice and verify collision is resolved
```

**Parameters:**
- `MAX_REWIND_DYN = 7`: Maximum rewind for dynamic
- Rewind increases with each failed attempt
- Blocker trajectory shared as moving obstacle

**Communication Overhead:**
- IU = length of blocker's trajectory segment
- Example: Sharing 10-step path = 10 IU

**When it works:** Agent can navigate around the blocker's dynamic trajectory

**When it fails:**
- Blocker's path blocks all escape routes
- Both agents are in tight corridors
- Timing conflicts (agent and blocker at same cell at different times)

---

### TIER 3: Joint A* Planning Strategy

**Concept:** Jointly plan for all colliding agents in the configuration space.

#### 3.1 Key Innovation: Adaptive Window Search

The search doesn't plan the entire remaining trajectory. Instead, it:
1. **Focuses on collision neighborhood** (rewind + horizon window)
2. **Expands window adaptively** if search fails
3. **Splices results** back into full plans

```
Collision at t=10
Window expansion:
  Attempt 1: [t=7-16]   (rewind=3, horizon=6)
  Attempt 2: [t=6-17]   (rewind=4, horizon=7)
  Attempt 3: [t=5-18]   (rewind=5, horizon=8)
  ...up to 8 expansion steps
```

#### 3.2 Configuration Space Search

The joint A* search operates in the **configuration space**: a tuple of all agent positions.

```python
State = (agent_1_pos, agent_2_pos, ..., agent_N_pos)
# For 3 agents: ((5,5), (6,6), (7,7))
```

**Search frontier:** Min-heap ordered by `f = g + h`
- `g`: number of steps taken from start
- `h`: sum of Manhattan distances to subgoals

#### 3.3 Conflict Detection in Joint Search

The search prunes invalid moves:

```python
# Vertex conflicts (two agents at same cell)
if len(set(next_positions)) < num_agents:
    continue  # Invalid: agents collide

# Edge conflicts (two agents swap positions)
for i, j in agent_pairs:
    if positions[i] == next_positions[j] and positions[j] == next_positions[i]:
        continue  # Invalid: agents swap

# Conflicts with reserved agents (non-colliding agents)
if any(pos in reserved_positions_at(time_step) for pos in next_positions):
    continue  # Invalid: hits external agent
```

#### 3.4 Subgoal Selection

Critical to prevent degenerate search (start == goal):

```python
for aid in agents_in_collision:
    traj = current_trajectories[aid-1]
    if traj:
        # Use position at clamped time from trajectory
        t_subgoal_clamped = min(t_goal_sub, len(traj) - 1)
    else:
        t_subgoal_clamped = t_goal_sub
    subgoal_positions.append(agent_pos_at(aid, t_subgoal_clamped))
```

This ensures subgoals are realistic (from existing trajectories) not arbitrary.

#### 3.5 Plan Splicing

Once joint plan segment is found, splice it carefully:

```python
# Extend plan if t_start > current_plan_length
if t_start > plan_len:
    extended_plan = original_plan + [4]*(t_start - plan_len)  # Add WAITs

# Validate prefix reaches expected start position
prefix_traj = simulate_plan(agent_env, extended_plan[:t_start])
if prefix_traj[-1] != expected_start_pos:
    reject plan  # Prefix doesn't match joint start

# Build final plan
final_plan = prefix + joint_segment + suffix
```

#### 3.6 Default Parameters

```python
base_rewind = 7         # Default 7 (was 3)
base_horizon = 15       # Default 15 (was 6)
max_expansion_steps = 8 # Default 8 (was 3)
max_agents = 4          # Max agents to jointly plan
time_budget = 2.0       # Per attempt timeout (seconds)
max_expansions = 20000  # Max A* nodes to expand
```

**Why these values work:**
- Larger window (7+15=22 steps) covers wider collision neighborhood
- 8 expansion steps allows significant growth (up to rewind=15, horizon=23)
- Still bounded: worst case is O(4^22) states, pruned aggressively
- Time budget prevents runaway searches

---

## 3. Information Units (IU) Calculation

### 3.1 Definition

**Information Units quantify communication overhead** when agents share data to resolve collisions:

| Strategy | What's Shared | IU Calculation |
|----------|--------------|---|
| **STATIC** | Blocked cells | # of cells = 1 IU per cell |
| **DYNAMIC** | Blocker trajectory | # of positions = 1 IU per timestep |
| **JOINT A*** | Joint plan length | # of steps in joint segment = IU per agent |

### 3.2 IU Tracking in Code

```python
class InfoSharingTracker:
    def __init__(self):
        self.initial_submission_iu = 0      # Initial plan submission
        self.alert_iu = 0                   # Collision alerts
        self.revised_submission_iu = 0      # Revised plan submission
        self.alert_details_iu = {
            'static': 0,      # Static blocking
            'dynamic': 0,     # Dynamic obstacles
            'pibt': 0         # Joint A* / PIBT
        }
```

### 3.3 IU Accumulation Process

**Initial Submission:**
```python
info_tracker.record_initial_submission(initial_trajectories)
# IU = sum(len(traj) for all agents)
# Example: 7 agents with avg path length 10 = 70 IU
```

**Static Alert (per collision fixed):**
```python
info_tracker.record_static_alert(forbidden_cells)
# IU += number of cells blocked
# Example: Blocking (5,5), (6,5) = 2 IU
```

**Dynamic Alert (per collision fixed):**
```python
info_tracker.record_dynamic_alert(dynamic_path)
# IU += length of shared trajectory
# Example: Sharing 12-step path = 12 IU
```

**Joint A* Alert (per collision fixed):**
```python
info_tracker.record_pibt_alert(num_pibt_steps * num_agents)
# IU += (horizon) * (num_agents_in_collision)
# Example: 15 steps for 2 agents = 30 IU
```

**Revised Submission (whenever plan changes):**
```python
info_tracker.record_revised_submission(new_trajectory)
# IU += length of new trajectory
# Accumulated with every replanning
```

### 3.4 Total IU Calculation

```python
total_iu = initial_submission_iu + alert_iu + revised_submission_iu

# Breakdown:
alert_iu = static_iu + dynamic_iu + joint_iu
```

### 3.5 Example Walkthrough

```
Scenario: 3 agents, 2 collisions

1. Initial Submission:
   Agent 1: 10 steps, Agent 2: 12 steps, Agent 3: 8 steps
   initial_submission_iu = 30

2. Collision 1 detected (Agents 1-2)

   Tier 1 (STATIC):
     Block cells (5,5), (6,5)
     alert_iu += 2  (static: 2)

   Tier 2 (DYNAMIC):
     Share Agent 2's path [8 positions]
     alert_iu += 8  (dynamic: 8)

   Revised submission of Agent 1:
     New trajectory: 12 steps
     revised_submission_iu += 12

3. Collision 2 detected (Agents 2-3)

   Tier 3 (JOINT A*):
     Joint planning window: 15 steps, 2 agents
     alert_iu += 30  (pibt: 30)

   Revised submission of Agent 2:
     New trajectory: 14 steps
     revised_submission_iu += 14

Total IU = 30 + (2+8+30) + (12+14) = 30 + 40 + 26 = 96
```

---

## 4. Global Cleanup: Synchronous WAIT Trimming

### Problem
When Joint A* splices plans, it may preserve suffix WAITs from the original plan, even after agents reach their goals:

```
Agent 1 original: [move, move, move, WAIT, WAIT]  len=5
Agent 1 after joint splice: [move, move, move, WAIT, WAIT, WAIT, WAIT]  len=7
Goal reached at step 2, but plan has 5 more steps
```

### Solution: Global Synchronization

Work **backwards across all agents together** to find the last meaningful timestep:

```python
1. Find max plan length (e.g., 22)
2. Start from timestep 21 and work backwards
3. At each timestep, check ALL agents
4. If ANY agent has action ≠ 4 (WAIT), stop
5. That's the global_makespan
6. Trim ALL agents to that length
```

**Example:**
```
Agent 1: [actions up to step 10][WAITs until step 21]
Agent 2: [actions up to step 4]
Agent 3: [actions up to step 18][WAITs until step 21]
Agent 4: [actions up to step 16][WAITs until step 21]

Step 21: all are WAIT → skip
Step 20: all are WAIT → skip
...
Step 18: Agent 3 is WAIT, Agent 4 is WAIT, others undefined → check
         But actually Agent 3 reaches goal at 18, only waits after
         So this is still WAIT
...
Step 17: Agent 3 has actual move (step 17 of its path)
         → global_makespan = 18 (steps 0-17)

All agents trimmed to length 18
```

### Benefits
- **Single global makespan** instead of ragged endings
- **Cleaner trajectories** for visualization/analysis
- **Accurate cost metrics** (SOC, makespan)
- **No spurious collisions** at the end

---

## 5. MAPF Metrics

### 5.1 Key Metrics Computed

**Makespan (MS):**
```python
makespan = max(len(trajectory) for all agents)
# Definition: Time when LAST agent reaches goal
# Goal: Minimize (faster completion)
```

**Sum of Costs (SOC):**
```python
sum_of_costs = sum(len(plan) for all agents)
# Definition: Total steps summed across all agents
# Goal: Minimize (efficiency)
```

**Average Path Length:**
```python
avg_path_length = sum_of_costs / num_agents
# Definition: Average steps per agent
# Indicator: Relative balance
```

**Agents at Goal:**
```python
agents_at_goal = count(trajectory[-1] == goal for all agents)
# Definition: How many agents successfully reached goal
# Goal: Maximize (all agents should reach goal)
```

### 5.2 Output Format

```
=== MAPF Metrics ===
Makespan (longest trajectory):  18
Sum of Costs (SOC):             95
Average Path Length:            13.57
Total Agents:                   7
Agents at Goal:                 7
Final Collision Count:          0
```

### 5.3 Optimality Analysis

Compare against:
- **Optimal lower bound**: Manhattan distance sum (if no agents)
- **Initial solution**: Original plan SOC/makespan
- **Final solution**: After collision resolution

**Optimality gap = (Final SOC - Optimal lower bound) / Optimal lower bound**

---

## 6. Resolution Loop and Convergence

### 6.1 Pass Structure

Each pass:
```python
1. Detect all collisions
2. Sort by (time, agent_id)
3. For each collision:
     - Try Tier 1 (STATIC)
       - If fails, Try Tier 2 (DYNAMIC)
         - If fails, Try Tier 3 (JOINT A*)
     - If fixes: break and re-detect
4. If any fixes this pass: continue
   Else: try DEFER strategy
```

### 6.2 Convergence Guarantees

**Upper bounds:**
- `max_passes = 10000` (was 50, now unlimited by passes, limited by time)
- `time_limit = 60` seconds (global budget)
- Each pass has time limit for Joint A*: 2 seconds

**Termination conditions:**
1. No collisions remain → **Success**
2. Time budget exceeded → **Timeout**
3. No progress for one pass → **Attempt DEFER**
4. DEFER successful → **Continue resolution**
5. DEFER fails → **Stuck (partial success)**

---

## 7. Summary of Configuration

### Current Parameters (Optimized for Larger Instances)

```python
# Joint A* Search
base_rewind = 7              # (was 3)   Look back 7 steps
base_horizon = 15            # (was 6)   Look ahead 15 steps
max_expansion_steps = 8      # (was 3)   Expand window up to 8 times
max_agents = 4               # Max agents to jointly plan

# Timeouts and Limits
time_limit = 60              # Overall resolution budget (seconds)
max_passes = 10000           # Maximum passes before giving up
max_expansions = 20000       # A* nodes per search

# Static/Dynamic Strategies
INIT_REWIND = 3
MAX_REWIND_STATIC = 7
MAX_REWIND_DYN = 7
```

### Tuning Guidance

**To improve success rate (higher chance of collision-free):**
- Increase `base_rewind`, `base_horizon`, `max_expansion_steps`
- Increase `time_limit`
- But: slower execution

**To reduce runtime:**
- Decrease search windows
- Reduce `time_limit`
- But: lower success rate

---

## 8. Conclusion

The three-tier strategy balances **optimality, communication efficiency, and runtime:**

1. **TIER 1** is cheap (O(1) agents, minimal IU)
2. **TIER 2** is moderate (O(1) agents, shared trajectory IU)
3. **TIER 3** is expensive (O(4^window) complexity, high IU) but most powerful
4. **TIER 4** (DEFER) is insurance policy for persistent conflicts

The IU metric quantifies the **information sharing cost** - directly measuring communication overhead in distributed MAPF systems. This is critical for real-world robot teams where communication bandwidth is limited.

The global cleanup ensures **clean, analyzable results** with clear metrics for benchmarking and comparison.

---

## References

- **Configuration Space Search**: Surynek et al., 2015
- **PIBT (Priority Inheritance with Backtracking)**: Okumura et al., 2019
- **IU (Information Units)**: Custom metric for communication overhead quantification
