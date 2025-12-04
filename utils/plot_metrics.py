import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from typing import Dict, List
import argparse
import glob
import os
import pandas as pd

def setup_style():
    """Set up clean and website-like plotting style with extended colors."""
    # Extended harmonious color palette
    colors = [
        '#F28B82',  # Soft Red
        '#9FA8DA',  # Muted Blue
        '#CE93D8',  # Soft Purple
        '#F6BF72',  # Muted Orange
        '#80CBC4',  # Soft Teal
        '#B0BEC5'   # Warm Gray
    ]
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.serif': ['Source Sans Pro'],
        'axes.titlesize': 16,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.labelcolor': '#2D2D2D',
        'text.color': '#2D2D2D',
        'axes.edgecolor': '#DADADA',
        'grid.color': '#f2dad8',
        'grid.alpha': 0.4,
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#fffdfc',
        'lines.linewidth': 2.5
    })
    return colors

def smooth_data(data: list, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing to data."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def calculate_frames_seen(episode_lengths: List[int]) -> List[int]:
    """Calculate cumulative frames seen by the agent."""
    return np.cumsum(episode_lengths).tolist()

def plot_q_values(q_values_log: Dict, save_dir: str, colors, start_episode: int = 1):
    """Create plots for Q-value metrics."""
    episodes = []
    target_q_data = {'mean': [], 'std': [], 'max': [], 'min': []}
    current_q_data = {'mean': [], 'std': [], 'max': [], 'min': []}
    next_q_data = {'mean': [], 'std': [], 'max': [], 'min': []}
    td_error_data = {'mean': [], 'std': [], 'max': [], 'min': []}
    
    # Extract data from the log, skipping null values and early episodes
    for episode_data in q_values_log['episodes']:
        episode_num = episode_data['episode']
        if episode_num < start_episode:
            continue
            
        q_values = episode_data['q_values']
        
        if all(q_values[key] is not None for key in ['target_q', 'current_q', 'next_q', 'td_errors']):
            episodes.append(episode_num)
            
            for metric, data in [('target_q', target_q_data), 
                               ('current_q', current_q_data),
                               ('next_q', next_q_data),
                               ('td_errors', td_error_data)]:
                stats = q_values[metric]
                data['mean'].append(stats['mean'])
                data['std'].append(stats['std'])
                data['max'].append(stats['max'])
                data['min'].append(stats['min'])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Q-Values Analysis', fontsize=16, y=0.95, color='#2D2D2D')

    def plot_metric(ax, data, title, color):
        means = np.array(data['mean'])
        stds = np.array(data['std'])
        maxs = np.array(data['max'])
        mins = np.array(data['min'])
        
        ax.plot(episodes, means, label='Mean', color=color, linewidth=2)
        ax.fill_between(episodes, means - stds, means + stds, 
                       alpha=0.2, color=color, label='±1 std')
        ax.fill_between(episodes, mins, maxs, 
                       alpha=0.1, color=color, label='Min-Max')
        
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.grid(True, alpha=0.2)

    plot_metric(ax1, target_q_data, 'Target Q-Values', colors[0])
    plot_metric(ax2, current_q_data, 'Current Q-Values', colors[1])
    plot_metric(ax3, next_q_data, 'Next Q-Values', colors[2])
    plot_metric(ax4, td_error_data, 'TD Errors', colors[3])

    plt.tight_layout()
    plt.savefig(f'{save_dir}/q_values_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_single_run(metrics: dict, q_values_log: dict, save_dir: str, colors,
                    start_episode: int = 1,
                    reward_window: int = 100,
                    loss_window: int = 100,
                    length_window: int = 20):
    """
    Create plots for a single training run.
    
    Plots:
      - ax1: Average Episode Reward (with moving average and std)
      - ax2: Episode Success (Goal Reached) as a scatter plot and moving average success rate.
         (A commented alternative for sample efficiency is provided.)
      - ax3: Training Loss (with moving average and std)
      - ax4: Episode Lengths (with moving average and std)
    """
    
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    
    fig.suptitle('Training Metrics Overview', fontsize=16, y=0.95, color='#2D2D2D')
    
    # Slice data starting from start_episode
    episodes = range(start_episode, len(metrics['rewards']) + 1)
    rewards = metrics['rewards'][start_episode - 1:]
    losses = metrics['losses'][start_episode - 1:]
    episode_lengths = metrics['episode_lengths'][start_episode - 1:]
    
    # For plotting, convert episodes to numpy array.
    episodes_array = np.array(episodes)
    
    ### Plot Average Reward (with Success Markers)
    reward_color = colors[0]
    rewards_array = np.array(rewards)
    smoothed_rewards = smooth_data(rewards, window=reward_window)
    if len(rewards_array) >= reward_window:
        rolling_std = np.array([
            np.std(rewards_array[i:i+reward_window])
            for i in range(len(rewards_array) - reward_window + 1)
        ])
    else:
        rolling_std = np.zeros_like(rewards_array)
    valid_episodes = episodes_array[reward_window - 1:]
    ax1.plot(episodes, rewards, alpha=0.2, label='Raw', linewidth=1, color=reward_color)
    ax1.fill_between(valid_episodes,
                     smoothed_rewards - rolling_std,
                     smoothed_rewards + rolling_std,
                     alpha=0.1, color=reward_color, label='±1 std')
    ax1.plot(valid_episodes, smoothed_rewards,
             label='Moving Average', linewidth=2, color=reward_color, zorder=5)
    
    # # Overlay success markers on the rewards plot.
    # # Assume metrics['successes'] is aligned with episodes (0/1 for each episode).
    # successes = np.array(metrics['successes'][start_episode - 1:])
    # # Use the same episodes_array as for rewards.
    # # Determine a y-value near the top of the current y-axis.
    # y_top = ax1.get_ylim()[1]
    # # We place markers slightly below the top so they don't get clipped.
    # marker_y = y_top - 0.05 * y_top  
    # # Find episodes with success (value 1)
    # success_indices = episodes_array[successes == 1]
    # # Plot these as small upward-pointing triangles.
    # ax1.scatter(success_indices, np.full(success_indices.shape, marker_y),
    #             marker='*', s=2, color=colors[2], label='Success')
    
    ax1.set_title('Average Episode Reward')
    ax1.set_ylabel('Reward')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.7)
    # fix legend at bottom left
    ax1.legend(loc='lower left', bbox_to_anchor=(0, 0), ncol=1, )
    ax1.set_ylim(bottom=-5, top=10)

    
    """
    ### Plot Episode Success (Goal Reached)
    # Assume metrics['successes'] is a list of 0s and 1s per episode.
    success_color = colors[2]
    successes = metrics['successes'][start_episode - 1:]
    successes_array = np.array(successes)
    # Create an x-axis array for episodes corresponding to successes.
    success_episodes = np.array(range(start_episode, start_episode + len(successes_array)))
    
    # Plot raw successes as a scatter plot (only episodes with success, i.e. value 1).
    success_episode_indices = success_episodes[successes_array == 1]
    ax2.scatter(success_episode_indices, np.ones_like(success_episode_indices), s=10,
                color=success_color, label='Goal Reached (raw)')
    
    # Compute a moving average for success rate.
    if len(successes_array) >= success_window:
        smoothed_success = np.array([
            np.mean(successes_array[i:i+success_window])
            for i in range(len(successes_array) - success_window + 1)
        ])
    else:
        smoothed_success = successes_array
    valid_episodes_success = success_episodes[success_window - 1:]
    
    ax2.plot(valid_episodes_success, smoothed_success, linewidth=2, color=success_color,
             label='Moving Average Success Rate')
    ax2.set_title('Episode Success (Goal Reached)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
    """

    # ----- Alternative: Sample Efficiency Plot (Commented Out) -----
    
    # If you prefer to plot sample efficiency (total frames vs. average reward) on ax2 instead, use:
    eff_color = colors[2]
    frames_seen = calculate_frames_seen(episode_lengths)
    smoothed_eff_rewards = smooth_data(rewards, window=100)
    rewards_array = np.array(rewards)
    if len(rewards_array) >= 100:
        rolling_std_eff = np.array([
            np.std(rewards_array[i:i+100])
            for i in range(len(rewards_array) - 100 + 1)
        ])
    else:
        rolling_std_eff = np.zeros_like(rewards_array)
    valid_frames = np.array(frames_seen)[100 - 1:]
    ax2.plot(frames_seen, rewards, alpha=0.2, linewidth=1, color=eff_color, label='Raw')
    ax2.fill_between(valid_frames,
                     smoothed_eff_rewards - rolling_std_eff,
                     smoothed_eff_rewards + rolling_std_eff,
                     alpha=0.1, color=eff_color, label='±1 std')
    ax2.plot(valid_frames, smoothed_eff_rewards, linewidth=2, color=eff_color, zorder=5, label='Moving Average')
    ax2.set_xlabel('Total Frames')
    ax2.set_ylabel('Average Reward')
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax2.set_title('Sample Efficiency')
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax2.set_ylim(bottom=-5, top=5)

    
    ### Plot Training Loss
    loss_color = colors[1]
    losses_array = np.array(losses)
    smoothed_losses = smooth_data(losses, window=loss_window)
    if len(losses_array) >= loss_window:
        rolling_std_loss = np.array([
            np.std(losses_array[i:i+loss_window])
            for i in range(len(losses_array) - loss_window + 1)
        ])
    else:
        rolling_std_loss = np.zeros_like(losses_array)
    valid_episodes_loss = episodes_array[loss_window - 1:]
    ax3.plot(episodes, losses, alpha=0.2, label='Raw', linewidth=1, color=loss_color)
    ax3.fill_between(valid_episodes_loss,
                     smoothed_losses - rolling_std_loss,
                     smoothed_losses + rolling_std_loss,
                     alpha=0.1, color=loss_color, label='±1 std')
    ax3.plot(valid_episodes_loss, smoothed_losses,
             label='Moving Average', linewidth=2, color=loss_color, zorder=5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    ### Plot Episode Lengths
    length_color = colors[3]
    lengths_array = np.array(episode_lengths)
    smoothed_lengths = smooth_data(episode_lengths, window=length_window)
    if len(lengths_array) >= length_window:
        rolling_std_length = np.array([
            np.std(lengths_array[i:i+length_window])
            for i in range(len(lengths_array) - length_window + 1)
        ])
    else:
        rolling_std_length = np.zeros_like(lengths_array)
    valid_episodes_length = episodes_array[length_window - 1:]
    ax4.plot(episodes, episode_lengths, alpha=0.2, label='Raw', linewidth=1, color=length_color)
    ax4.fill_between(valid_episodes_length,
                     smoothed_lengths - rolling_std_length,
                     smoothed_lengths + rolling_std_length,
                     alpha=0.1, color=length_color, label='±1 std')
    ax4.plot(valid_episodes_length, smoothed_lengths,
             label='Moving Average', linewidth=2, color=length_color, zorder=5)
    ax4.set_title('Episode Lengths')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Steps')
    ax4.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if q_values_log:
        plot_q_values(q_values_log, save_dir, colors, start_episode)

def load_metrics(log_path):
    with open(log_path, 'r') as f:
        data = json.load(f)
    return (
        np.array(data['rewards']),
        np.array(data['losses']),
        np.array(data['episode_lengths'])
    )

def smooth(x, window=100):
    if len(x) < window:
        return x, np.arange(len(x))
    kernel = np.ones(window) / window
    y = np.convolve(x, kernel, mode='valid')
    # align: y[i] = mean(x[i:i+window])
    return y, np.arange(window - 1, len(x))

def professional_training_plot(
        log_dir: str,
        save_path: str = "training_overview.png",
        dpi: int = 100
    ):
    """
    Reads training_logs.json in `log_dir`, plots:
      • Episode Reward vs Episodes
      • Sample Efficiency (Reward vs Frames)
      • Training Loss vs Episodes
      • Episode Length vs Episodes
    """

    sns.set_theme(style='whitegrid')
    colors = {
      'reward': '#F28B82',
      'eff':    '#CE93D8',
      'loss':   '#9FA8DA',
      'len':    '#F6BF72'
    }

    # ─── LOAD & PROCESS ───────────────────────────────────────────
    rewards, losses, lengths = load_metrics(Path(log_dir) / "training_logs.json")
    episodes = np.arange(1, len(rewards) + 1)
    frames   = np.cumsum(lengths)

    # ─── SMOOTH ────────────────────────────────────────────────────
    r_smooth, r_idx     = smooth(rewards, window=200)
    l_smooth, l_idx     = smooth(losses,  window=200)
    len_smooth, len_idx = smooth(lengths, window=300)

    # ─── FIGURE SETUP ─────────────────────────────────────────────
    fig, axes = plt.subplots(2,2, figsize=(8,6), dpi=dpi)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # --- Episode Reward ---
    ax1.plot(episodes,    rewards, alpha=0.2, lw=1, color=colors['reward'], label='Raw')
    ax1.plot(r_idx+1,     r_smooth,      lw=2, color=colors['reward'], label='Smoothed')
    ax1.set_title("Episode Reward")
    ax1.set_ylabel("Reward")
    ax1.xaxis.set_major_locator(MaxNLocator(6))
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1.legend(fontsize=8, loc='upper right')

    # --- Sample Efficiency ---
    ax2.plot(frames,      rewards, alpha=0.2, lw=1, color=colors['eff'], label='Raw')
    # plot smoothed reward against the *frames* at the same episode indices
    ax2.plot(frames[r_idx], r_smooth,      lw=2, color=colors['eff'], label='Smoothed')
    ax2.set_title("Sample Efficiency")
    ax2.set_xlabel("Total Frames")
    ax2.set_ylabel("Reward")
    ax2.xaxis.set_major_locator(MaxNLocator(6))
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax2.legend(fontsize=8, loc='upper right')

    # --- Training Loss ---
    ax3.plot(episodes,    losses, alpha=0.2, lw=1, color=colors['loss'], label='Raw')
    ax3.plot(l_idx+1,     l_smooth,      lw=2, color=colors['loss'], label='Smoothed')
    ax3.set_title("Training Loss")
    ax3.set_ylabel("Loss")
    ax3.xaxis.set_major_locator(MaxNLocator(6))
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax3.legend(fontsize=8, loc='upper right')

    # --- Episode Length ---
    ax4.plot(episodes,    lengths, alpha=0.2, lw=1, color=colors['len'], label='Raw')
    ax4.plot(len_idx+1,   len_smooth,     lw=2, color=colors['len'], label='Smoothed')
    ax4.set_title("Episode Length")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Steps")
    ax4.xaxis.set_major_locator(MaxNLocator(6))
    ax4.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax4.legend(fontsize=8, loc='upper right')

    for ax in axes.flatten():
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_comparison_q_values(run_dirs: List[str], save_dir: str, colors, start_episode: int = 1):
    """Create comprehensive Q-value comparison plots for multiple runs."""
    metrics_list = []
    q_values_list = []
    labels = []
    
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, 'training_logs.json'), 'r') as f:
            metrics = json.load(f)
            metrics_list.append(metrics)
        
        q_values_path = os.path.join(run_dir, 'q_values_log.json')
        if os.path.exists(q_values_path):
            with open(q_values_path, 'r') as f:
                q_values = json.load(f)
                q_values_list.append(q_values)
        
        variant = []
        config = metrics['config']
        if config.get('double', False): variant.append('Double')
        if config.get('dueling', False): variant.append('Dueling')
        if config.get('priority', False): variant.append('PER')
        labels.append(' + '.join(variant) if variant else 'DQN')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Q-Values Comparison Across Variants', fontsize=16, y=0.95, color='#2D2D2D')

    def plot_metric_comparison(ax, metric_name, title, base_color_idx):
        base_alpha = 0.6
        alpha_increment = 0.1
        std_alpha_base = 0.1
        minmax_alpha_base = 0.05
        alpha_reduction_factor = 0.8
        base_zorder = 2  # Starting z-order for line plots
        
        num_variants = len(q_values_list)
        if num_variants > 2:
            std_alpha_base *= alpha_reduction_factor
            minmax_alpha_base *= alpha_reduction_factor
        
        for idx, (q_values, label, color) in enumerate(zip(q_values_list, labels, colors)):
            episodes = []
            means = []
            stds = []
            mins = []
            maxs = []
            
            for episode_data in q_values['episodes']:
                episode_num = episode_data['episode']
                if episode_num < start_episode:
                    continue
                    
                q_vals = episode_data['q_values']
                if q_vals[metric_name] is not None:
                    episodes.append(episode_num)
                    means.append(q_vals[metric_name]['mean'])
                    stds.append(q_vals[metric_name]['std'])
                    mins.append(q_vals[metric_name]['min'])
                    maxs.append(q_vals[metric_name]['max'])
            
            if episodes:
                episodes = np.array(episodes)
                means = np.array(means)
                stds = np.array(stds)
                mins = np.array(mins)
                maxs = np.array(maxs)
                
                current_alpha = base_alpha + (idx * alpha_increment)
                current_std_alpha = std_alpha_base * (1 + idx * 0.2)
                current_minmax_alpha = minmax_alpha_base * (1 + idx * 0.2)
                current_zorder = base_zorder + idx  # Increment z-order

                # Add shaded regions and lines with updated z-order
                ax.fill_between(episodes, mins, maxs, alpha=current_minmax_alpha, color=color, zorder=current_zorder - 1)
                ax.fill_between(episodes, means - stds, means + stds, alpha=current_std_alpha, color=color, zorder=current_zorder - 1)
                ax.plot(episodes, means, label=label, color=color, linewidth=2, alpha=current_alpha, zorder=current_zorder)
        
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.2)
    
    plot_metric_comparison(ax1, 'target_q', 'Target Q-Values', 0)
    plot_metric_comparison(ax2, 'current_q', 'Current Q-Values', 1)
    plot_metric_comparison(ax3, 'next_q', 'Next Q-Values', 2)
    plot_metric_comparison(ax4, 'td_errors', 'TD Errors', 3)
    
    fig.text(0.02, 0.02, 'Shaded areas represent ±1 std dev (darker) and min-max range (lighter)', 
             fontsize=8, color='#666666')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/q_values_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison(run_dirs: List[str], save_dir: str, colors, start_episode: int = 1):
    """Create comprehensive comparison plots for multiple training runs."""
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    metrics_list = []
    labels = []
    
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, 'training_logs.json'), 'r') as f:
            metrics = json.load(f)
            metrics_list.append(metrics)
            variant = []
            config = metrics['config']
            if config.get('double', False): variant.append('Double')
            if config.get('dueling', False): variant.append('Dueling')
            if config.get('priority', False): variant.append('PER')
            labels.append(' + '.join(variant) if variant else 'DQN')
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    
    fig.suptitle('DQN Variants Comparison', fontsize=16, y=0.95, color='#2D2D2D')
    
    base_alpha = 0.70
    increment = 0.05
    base_zorder = 2  # Starting z-order for line plots
    
    for idx, ((metrics, label), color) in enumerate(zip(zip(metrics_list, labels), colors)):
        # Slice data starting from start_episode
        episodes = range(start_episode, len(metrics['rewards']) + 1)
        rewards = metrics['rewards'][start_episode-1:]
        losses = metrics['losses'][start_episode-1:]
        episode_lengths = metrics['episode_lengths'][start_episode-1:]
        frames_seen = calculate_frames_seen(episode_lengths)
        
        z_order = base_zorder + idx  # Increment z-order for each variant
        
        # Plot rewards
        smoothed_rewards = smooth_data(rewards, window=20)
        ax1.plot(episodes[len(episodes)-len(smoothed_rewards):], 
                 smoothed_rewards, label=label, linewidth=2, color=color, alpha=base_alpha, zorder=z_order)
        ax1.plot(episodes, rewards, alpha=0.2, linewidth=1, color=color, zorder=z_order)
        
        # Plot sample efficiency
        smoothed_eff_rewards = smooth_data(rewards, window=50)
        ax2.plot(frames_seen[len(frames_seen)-len(smoothed_eff_rewards):], 
                 smoothed_eff_rewards, label=label, linewidth=2, color=color, alpha=base_alpha, zorder=z_order)
        
        # Plot losses
        smoothed_losses = smooth_data(losses, window=10)
        ax3.plot(episodes[len(episodes)-len(smoothed_losses):], 
                 smoothed_losses, label=label, linewidth=2, color=color, alpha=base_alpha, zorder=z_order)
        ax3.plot(episodes, losses, alpha=0.2, linewidth=1, color=color, zorder=z_order)
        
        # Plot episode lengths
        smoothed_lengths = smooth_data(episode_lengths, window=20)
        ax4.plot(episodes[len(episodes)-len(smoothed_lengths):], 
                 smoothed_lengths, label=label, linewidth=2, color=color, alpha=base_alpha, zorder=z_order)
        ax4.plot(episodes, episode_lengths, alpha=0.2, linewidth=1, color=color, zorder=z_order)
        
        base_alpha += increment
    
    ax1.set_title('Average Episode Reward')
    ax1.set_ylabel('Reward')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize='x-small')
    
    ax2.set_title('Sample Efficiency')
    ax2.set_xlabel('Total Frames')
    ax2.set_ylabel('Average Reward')
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize='x-small')
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize='x-small')
    
    ax4.set_title('Episode Lengths')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Steps')
    ax4.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize='x-small')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/variants_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additionally plot Q-values comparison if available
    has_q_values = all(os.path.exists(os.path.join(d, 'q_values_log.json')) for d in run_dirs)
    if has_q_values:
        plot_comparison_q_values(run_dirs, save_dir, colors, start_episode)

def main():
    parser = argparse.ArgumentParser(description='Plot DQN training metrics')
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='Directory or directories containing training logs')
    parser.add_argument('--save-dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--start-episode', type=int, default=10,
                        help='Episode number to start plotting from (default: 5)')
    
    args = parser.parse_args()
    colors = setup_style()
    
    if len(args.dirs) == 1:
        # Single run
        with open(os.path.join(args.dirs[0], 'training_logs.json'), 'r') as f:
            metrics = json.load(f)
        
        q_values_log = None
        q_values_path = os.path.join(args.dirs[0], 'q_values_log.json')
        if os.path.exists(q_values_path):
            with open(q_values_path, 'r') as f:
                q_values_log = json.load(f)
        
        plot_single_run(metrics, q_values_log, args.save_dir, colors, args.start_episode)
        professional_training_plot(args.dirs[0], os.path.join(args.save_dir, 'training_overview.png'))
    else:
        # Multiple runs comparison
        plot_comparison(args.dirs, args.save_dir, colors, args.start_episode)

if __name__ == "__main__":
    main()