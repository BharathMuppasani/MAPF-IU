# ICAPS Test Dataset Format

## Overview

The `test_data/icaps_test/` folder contains MAPF (Multi-Agent Path Finding) benchmark problems derived from standard MAPF benchmark scenarios.

## Directory Structure

```
test_data/icaps_test/
├── random-32-32-20/          # 32x32 random map, 20% obstacles
│   ├── 10_agents/            # 10 agent configurations
│   │   ├── test_01.txt       # Test instance 1
│   │   ├── test_02.txt       # Test instance 2
│   │   └── ... (10 files)
│   ├── 20_agents/
│   └── ... (up to 128_agents)
├── random-64-64-20/          # 64x64 random map
├── den312d/                  # Game map (Dragon Age)
└── warehouse/                # Warehouse layout
```

## .txt File Format

Each `.txt` file contains one MAPF problem instance:

```
<height> <width>
<grid row 1>
<grid row 2>
...
<grid row H>
<num_agents>
<start_row> <start_col> <goal_row> <goal_col>  # Agent 1
<start_row> <start_col> <goal_row> <goal_col>  # Agent 2
...
```

### Example (10 agents on 32x32):

```
32 32
. . . @ . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. @ . . . . . . . . . @ . . . . . . . . . . . . . . . . . . . .
...
10
5 12 28 3
14 7 20 25
...
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| height, width | Grid dimensions |
| `.` | Free cell (traversable) |
| `@` | Obstacle (blocked) |
| num_agents | Number of agents in this instance |
| start_row, start_col | Agent's starting position (0-indexed) |
| goal_row, goal_col | Agent's goal position (0-indexed) |

## Map Statistics

| Map | Grid Size | Free Cells | Obstacle % |
|-----|-----------|------------|------------|
| random-32-32-20 | 32×32 | 819 | ~20% |
| random-64-64-20 | 64×64 | 3,270 | ~20% |
| den312d | 65×81 | 2,445 | ~54% |
| warehouse | 170×84 | 9,776 | ~32% |

## Agent Configurations

| Map Type | Agent Counts | Test Files |
|----------|--------------|------------|
| Random maps | 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 128 | 10 each |
| Complex maps | 8, 16, 32, 64, 128 | 5 each |

## Converting to .pth Format

For use with DCC/MAPF-LNS2 or similar frameworks, convert using:

```bash
# Convert entire dataset
python scripts/convert_to_pth.py \
    --input test_data/icaps_test \
    --output test_set/icaps

# Convert single configuration (all tests combined)
python scripts/convert_to_pth.py \
    --input test_data/icaps_test/random-32-32-20/10_agents \
    --output test_set/random32_10agents.pth \
    --single
```

### .pth Format

```python
# List of tuples, one per test instance
[
    (map, agents_pos, goals_pos),  # Instance 1
    (map, agents_pos, goals_pos),  # Instance 2
    ...
]

# Where:
# - map: np.ndarray (H, W), dtype=int → 0=free, 1=obstacle
# - agents_pos: np.ndarray (N, 2), dtype=int → [x, y] (col, row) per agent
# - goals_pos: np.ndarray (N, 2), dtype=int → [x, y] (col, row) per agent
```

### Coordinate Convention

| Format | Convention | Example |
|--------|------------|---------|
| .txt | (row, col) | `5 12` = row 5, col 12 |
| .pth | [x, y] = [col, row] | `[12, 5]` = col 12, row 5 |

## Loading Examples

### Load .txt directly
```python
def load_txt(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    h, w = map(int, lines[0].split())
    grid = [[c for c in line.split()] for line in lines[1:h+1]]
    n = int(lines[h+1])
    agents = [list(map(int, lines[h+2+i].split())) for i in range(n)]
    return h, w, grid, agents
```

### Load .pth
```python
import pickle

with open("test_set/random32_10agents.pth", "rb") as f:
    instances = pickle.load(f)

# Each instance
for map_, agents_pos, goals_pos in instances:
    print(f"Map shape: {map_.shape}")
    print(f"Agents: {len(agents_pos)}")
```

## Source

Generated from [Moving AI MAPF Benchmarks](https://movingai.com/benchmarks/mapf.html) using `scripts/generate_icaps_dataset.py`.
