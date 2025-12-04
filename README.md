# Scalable Multi-Agent Path Finding using Collision-Aware Dynamic Alert Mask and a Hybrid  Execution Strategy

This repository contains the Python implementation and experimental setup for "Scalable Multi-Agent Path Finding using Collision-Aware Dynamic Alert Mask and a Hybrid Execution Strategy." Our framework integrates decentralized Reinforcement Learning (RL) agents with a lightweight centralized coordinator to achieve efficient and scalable solutions for Multi-Agent Path Finding (MAPF).

---

## 📖 Introduction

Multi-Agent Path Finding (MAPF) is a fundamental problem in robotics and autonomous systems. While centralized methods offer optimality, they struggle with scalability. Distributed learning approaches improve scalability but can compromise solution quality. This project introduces a novel hybrid MAPF framework designed to balance these trade-offs.

Our system employs RL agents (DDQN and PPO) for decentralized path planning. A central module detects potential collisions and issues minimal, targeted alerts to specific agents. These alerts, incorporated into the agent's observation via a "Collision-Aware Dynamic Alert Mask," trigger a tiered replanning process. This approach strategically limits inter-agent information sharing while maintaining high solution feasibility, even in large-scale scenarios.

---

## ✨ Core Features

* **Hybrid MAPF Framework:** Combines decentralized RL-based planning (DDQN/PPO) with centralized collision detection and alert-based coordination.
* **Collision-Aware Dynamic Alert Mask:** An integral part of the RL agent's observation space, enabling adaptive responses to potential conflicts.
* **Tiered Replanning Strategy:** Agents first attempt to resolve conflicts using static obstacle avoidance, escalating to dynamic obstacle avoidance (using partial trajectory information of conflicting agents) if necessary.
* **Implemented RL Agents:** Includes Deep Q-Network (DDQN) and Proximal Policy Optimization (PPO) agents for single agent path finding in presence of dynamic obstacles.
* **Experimentation Suite:** Scripts for training RL models (`train.py`) and running comprehensive benchmark evaluations (`run_exp.py`) across various map types (mazes, warehouses, random) and agent densities.
* **Analysis & Visualization:** Tools (`test_data/result_analysis.ipynb`, `result_viz.py`, `utils/plot_metrics.py`) for processing experimental data and visualizing results.

---

## ⚙️ Installation

1.  **Clone the Repository:**
    ```bash
    # Clone this repository to your local machine
    git clone https://github.com/BharathMuppasani/MAPF-NeurIPS-25
    cd MAPF-NeurIPS-25
    ```

2.  **Create and Activate Conda Environment:**
    Use the provided `mapf_env.yml` file to create the Conda environment.
    ```bash
    conda env create -f mapf_env.yml
    conda activate mapf_env 
    ```
---

## 🚀 Usage

**Training an RL Agent (DDQN/PPO):**

To train a new agent model using `train.py`. The following examples are based on configurations used in the accompanying paper.

* **DDQN Training Example (Paper Config):** 
    ```bash
    python train.py \
      --env CustomGrid-v0 \
      --algo dqn \
      --episodes 30000 \
      --batch-size 128 \
      --buffer-size 1000000 \
      --lr 0.0003 \
      --gamma 0.97 \
      --target-update 300 \
      --grid-size 11 \
      --num-dynamic-obstacles 4 \
      --max-steps 50 \
      --double \
      --priority \
      --epsilon-start 1.0 \
      --epsilon-end 0.01 \
      --epsilon-decay 0.999 \
      --save-dir train_data/ddqn_11x11_paper_config/
    ```

* **PPO Training Example (Paper Config):** 
    ```bash
    python train.py \
      --env CustomGrid-v0 \
      --algo ppo \
      --episodes 30000 \
      --batch-size 128 \
      --lr 0.0003 \
      --gamma 0.95 \
      --grid-size 11 \
      --num-dynamic-obstacles 4 \
      --max-steps 50 \
      --eps-clip 0.2 \
      --k-epochs 4 \
      --ent-high 0.05 --ent-low 0.01 --ent-anneal-episodes 15000 \
      --vf-coef 0.5 \
      --save-dir train_data/ppo_11x11_paper_config/
    ```

**Running MAPF Experiments:**

To evaluate the hybrid framework using `run_exp.py`. This uses the replanning strategies and information settings evaluated in the paper.

```bash
python run_exp.py --strategy [best|random|farthest] \
                  --info [all|no|only_dyn] \
                  --search_type [bfs|astar] \
                  --algo [dqn|ppo] \
                  --num_agents [count] \
                  --timeout [seconds] \
                  --heuristic_weight [value] \
                  --map_path [path_to_map_or_scenario_file] \
                  --model_path [path_to_trained_rl_model.pth] \
                  --output_dir [path_to_save_experiment_results]
```
* For `--search_type astar` or other RL-guided search types, ensure the `--model_path` argument points to a relevant trained RL model (`.pth` file).
* The `--map_path` argument should point to the specific map file (e.g., `.txt`) or a scenario file (e.g., `.pkl`).

**Analyzing Results:**
Experimental results are saved as `.pkl` files and can be processed using the examples shown in `result_analysis.ipynb` Jupyter notebook.

---

## 📁 Project Structure

```
.
├── dqn/               # Deep Q-Network (DDQN) agent implementation
├── ppo/               # Proximal Policy Optimization (PPO) agent implementation
├── test_data/         # Benchmark maps, scenarios, and experiment results
├── train_data/        # Saved RL models, training logs, and plots
├── utils/             # Environment definitions and utility scripts
├── replay_buffer.py   # Experience Replay Buffer for DRL agents
├── run_exp.py         # Script to run MAPF experiments
├── train.py           # Script to train RL agents
└── mapf_env.yml       # Conda environment specification
```



## 📜 License

This project is licensed under the **MIT License**.

