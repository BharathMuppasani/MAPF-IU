import gymnasium as gym
from gymnasium.wrappers import NormalizeReward
import torch
import numpy as np
import argparse
from collections import deque
import json
import os
import time

from utils.grid_env_wrapper import GridEnvWrapper
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

torch.set_num_threads(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Training for CustomGrid')
    parser.add_argument('--env', type=str, default='CustomGrid-v0',
                        help='Gymnasium environment name')
    parser.add_argument('--norm', action='store_true',
                        help='Normalize rewards')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                        help='Size of replay buffer')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--target-update', type=int, default=10,
                        help='Steps between target updates (DQN only)')
    parser.add_argument('--grid-size', type=int, default=10,
                        help='Size of the grid environment')
    parser.add_argument('--num-static-obstacles', type=int, default=15,
                        help='Number of static obstacles')
    parser.add_argument('--num-dynamic-obstacles', type=int, default=0,
                        help='Number of dynamic obstacles')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Max steps per episode')
    # DQN flags
    parser.add_argument('--double', action='store_true',
                        help='Use Double DQN')
    parser.add_argument('--priority', action='store_true',
                        help='Use Prioritized Experience Replay')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Starting epsilon (DQN)')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help='Final epsilon (DQN)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate (DQN)')
    # PPO flags
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy coefficient (PPO)')
    parser.add_argument('--eps-clip', type=float, default=0.2,
                        help='PPO clipping epsilon')
    parser.add_argument('--k-epochs', type=int, default=4,
                        help='PPO epochs per update')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='Value‐loss coefficient (PPO)')
    
    parser.add_argument('--ent-high', type=float, default=0.05,
                    help='initial entropy coefficient')
    parser.add_argument('--ent-low',  type=float, default=0.005,
                        help='final entropy coefficient')
    parser.add_argument('--ent-anneal-episodes', type=int, default=5000,
                        help='how many episodes to decay ent_coef')

    # common save/load
    parser.add_argument('--save-dir', type=str,
                        default='train_data/checkpoints',
                        help='Directory to save models')
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path to load a pretrained model')
    # choose algorithm
    parser.add_argument('--algo', type=str, choices=['dqn', 'ppo'],
                        default='dqn',
                        help="Which algorithm to use: 'dqn' or 'ppo'")
    return parser.parse_args()


def curriculum_density(ep: int) -> float:

    if   ep < 500:  return [0.10, 0]
    elif ep < 3000: return [0.10, 1]
    elif ep < 6000:  return [0.20, 2]
    else:            return [0.30, 3]


def create_env(env_name, grid_size, static_obstacles, dynamic_obstacles, max_steps):
    assert env_name == "CustomGrid-v0"
    generation_mode = "maze"
    maze_density = 0.3
    env = GridEnvWrapper(
        grid_size=grid_size,
        variable_grid=False,
        num_dynamic_obstacles=dynamic_obstacles,
        generation_mode=generation_mode,
        maze_density=maze_density,
        max_steps=max_steps,
    )
    print("Custom Grid Environment:")
    print(f"  Generation Mode      : {generation_mode}")
    print(f"  Maze Density         : {maze_density}")
    print(f"  Dynamic Obstacles    : {dynamic_obstacles}")
    print(f"  Fixed Grid Size      : {grid_size}")
    env.reset()
    obs_shape  = env.observation_space.spaces["grid"].shape
    action_dim = env.action_space.n
    return env, obs_shape, action_dim

def save_training_config(args, save_dir):
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

def aggregate_stats(arr):
    if not arr:
        return None
    a = np.array(arr, dtype=float)
    return dict(mean=a.mean(), max=a.max(), min=a.min(), std=a.std())

def main():
    args = parse_args()

    # ─── Dynamically choose DQN vs PPO ───
    if args.algo == 'dqn':
        from dqn_agent import DQNAgent as Agent
    else:
        from ppo_agent import PPOAgent as Agent

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ─── Create and wrap env ───
    env, state_dim, action_dim = create_env(
        args.env,
        args.grid_size,
        args.num_static_obstacles,
        args.num_dynamic_obstacles,
        args.max_steps
    )
    if args.norm:
        env = NormalizeReward(env, gamma=args.gamma, epsilon=1e-8)
    print(f"State dimension: {state_dim},  Action dimension: {action_dim}")

    # ─── Instantiate agent ───
    if args.algo == 'dqn':
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            learning_rate=args.lr,
            gamma=args.gamma,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            target_update=args.target_update,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            use_double=args.double,
            use_priority=args.priority,
            custom_flag=True
        )
        print(f"Using DQN: ε_start={args.epsilon_start:.3f}, ε_end={args.epsilon_end:.3f}, "
              f"ε_decay={args.epsilon_decay:.3f}, double={args.double}, priority={args.priority}")
    else:
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            lr=args.lr,
            gamma=args.gamma,
            eps_clip=args.eps_clip,
            k_epochs=args.k_epochs,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            valid_coef=0.1,
            custom_flag=True
        )
        print(f"Using PPO: lr={args.lr:.6f}, γ={args.gamma:.3f}, clip={args.eps_clip:.3f}, "
              f"k_epochs={args.k_epochs}, ent_coef={args.ent_coef:.3f}, vf_coef={args.vf_coef:.3f}")

    # ─── Optionally load weights ───
    if args.load_path and os.path.exists(args.load_path):
        agent.load(args.load_path)
        print(f"Loaded model from {args.load_path}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_training_config(args, args.save_dir)

    # ─── Prepare logging containers ───
    training_logs = {
        'rewards': [], 'avg_rewards': [], 'losses': [], 'epsilons': [],
        'episode_lengths': [], 'successes': [], 'total_frames': 0,
        'frames_per_episode': [], 'fps': [], 'config': vars(args)
    }
    q_values_log = {'episodes': []}
    reward_window = deque(maxlen=100)
    best_avg_reward = -np.inf
    total_frames = 0

    console = Console()
    with Progress(
        TextColumn("[bold yellow]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[green]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Training ", total=args.episodes)

        for episode in range(args.episodes):

            env.env.maze_density = curriculum_density(episode)[0]
            env.env.num_dynamic_obstacles = curriculum_density(episode)[1]

            if args.algo == 'ppo':
                agent.reset_memory()
            # ─── Reset at start of each episode ───
            state, _ = env.reset()
            episode_reward = 0
            episode_loss = []
            done = False
            truncated = False
            steps = 0
            episode_start_time = time.time()
            episode_success = 0

            # buffer per-episode Q-stats
            episode_target_q = []
            episode_current_q = []
            episode_next_q = []
            episode_td_errors = []

            # ─── Rollout loop ───
            while not (done or truncated):
                steps += 1
                total_frames += 1

                if args.norm:
                    valid_actions = env.unwrapped.get_actions()
                else:
                    valid_actions = env.get_actions()

                if args.algo == 'dqn':
                    action = agent.select_action(state, valid_actions=valid_actions)
                else:
                    action = agent.select_action(state, valid_actions)

                next_state, reward, done, goal_flag, _ = env.step(action)
                if goal_flag:
                    episode_success = 1

                if args.algo == 'dqn':
                    agent.memory.push(state, action, reward, next_state, done)
                else:
                    agent.push_reward(reward, done)

                if args.algo == 'dqn' and len(agent.memory) > args.batch_size:
                    loss, t_q, c_q, n_q, dones, td_errs = agent.update()
                    if loss is not None:
                        episode_loss.append(loss)
                        episode_target_q.append(t_q.mean().item())
                        episode_current_q.append(c_q.mean().item())
                        episode_next_q.append(n_q.mean().item())
                        episode_td_errors.append(td_errs.mean().item())

                episode_reward += reward
                state = next_state
                if done:
                    break

            if args.algo == 'ppo':
                # Linear decay from ent_high → ent_low over ent_anneal_episodes
                frac = min(1.0, episode / args.ent_anneal_episodes)
                agent.ent_coef = args.ent_high + frac * (args.ent_low - args.ent_high)


            # ─── End-of-episode update for PPO ───
            if args.algo == 'ppo':
                ppo_loss = agent.update()
                episode_loss.append(ppo_loss)

            # ─── Epsilon decay (DQN only) ───
            if args.algo == 'dqn':
                agent.epsilon = max(agent.epsilon_end,
                                    agent.epsilon * agent.epsilon_decay)

            # ─── Collect stats & logs ───
            duration = time.time() - episode_start_time
            fps = steps / duration if duration > 0 else 0
            avg_loss = float(np.mean(episode_loss)) if episode_loss else 0.0

            reward_window.append(episode_reward)
            avg_reward = np.mean(reward_window)

            training_logs['rewards'].append(episode_reward)
            training_logs['avg_rewards'].append(avg_reward)
            training_logs['losses'].append(avg_loss)
            training_logs['epsilons'].append(agent.epsilon if args.algo=='dqn' else None)
            training_logs['episode_lengths'].append(steps)
            training_logs['successes'].append(episode_success)
            training_logs['total_frames'] = total_frames
            training_logs['frames_per_episode'].append(steps)
            training_logs['fps'].append(fps)

            q_stats = {
                'target_q': aggregate_stats(episode_target_q),
                'current_q': aggregate_stats(episode_current_q),
                'next_q': aggregate_stats(episode_next_q),
                'td_errors': aggregate_stats(episode_td_errors)
            }
            q_values_log['episodes'].append({'episode': episode, 'q_values': q_stats})

            # ─── Save best model so far ───
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(os.path.join(args.save_dir, 'best_model.pth'))

            progress.update(task, advance=1)

            # ─── Per-episode console log ───
            success_str = "[bold green]Yes[/bold green]" if episode_success else "[bold red]No[/bold red]"
            if args.algo == 'dqn':
                console.print(
                    f"[bold yellow]Episode[/bold yellow] "
                    f"[white]{episode+1:4}/{args.episodes:4}[/white]"
                    f" | [bold yellow]Eps:[/bold yellow] [white]{agent.epsilon:7.3f}[/white]"
                    f" | [bold yellow]Loss:[/bold yellow] [white]{avg_loss:7.4f}[/white]"
                    f" | [bold yellow]Rew:[/bold yellow] [white]{episode_reward:7.3f}[/white]"
                    f" | [bold yellow]AvgRew:[/bold yellow] [white]{avg_reward:7.3f}[/white]"
                    f" | [bold yellow]Len:[/bold yellow] [white]{steps:4}[/white]"
                    f" | [bold cyan]Success:[/bold cyan] {success_str}"
                )
            else:
                console.print(
                    f"[bold yellow]Episode[/bold yellow] "
                    f"[white]{episode+1:4}/{args.episodes:4}[/white]"
                    f" | [bold yellow]Loss:[/bold yellow] [white]{avg_loss:7.4f}[/white]"
                    f" | [bold yellow]Rew:[/bold yellow] [white]{episode_reward:7.3f}[/white]"
                    f" | [bold yellow]AvgRew:[/bold yellow] [white]{avg_reward:7.3f}[/white]"
                    f" | [bold yellow]Len:[/bold yellow] [white]{steps:4}[/white]"
                    f" | [bold cyan]Success:[/bold cyan] {success_str}"
                )

            # ─── Persist logs each episode ───
            with open(os.path.join(args.save_dir, 'training_logs.json'), 'w') as f:
                json.dump(training_logs, f, indent=4)

    # ─── Final save & summary ───
    agent.save(os.path.join(args.save_dir, 'final_model.pth'))
    with open(os.path.join(args.save_dir, 'training_logs.json'), 'w') as f:
        json.dump(training_logs, f, indent=4)
    with open(os.path.join(args.save_dir, 'q_values_log.json'), 'w') as f:
        json.dump(q_values_log, f, indent=4)

    console.print("\n[bold green]Training completed![/bold green]")
    console.print(f"[bold yellow]Best average reward:[/bold yellow] [white]{best_avg_reward:.2f}[/white]")
    console.print(f"[bold yellow]Total frames seen:[/bold yellow] [white]{total_frames}[/white]")
    console.print(f"[bold yellow]Models and logs saved in:[/bold yellow] [white]{args.save_dir}[/white]")

    env.close()

if __name__ == "__main__":
    main()
