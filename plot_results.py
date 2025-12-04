#!/usr/bin/env python3
"""
MAPF Results Plotting Script

This script generates research-style plots for MAPF benchmark results,
showing success rate, makespan, computation time, and Information Units (IU)
across different maps and agent counts.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set publication-quality plotting style
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.markersize'] = 8


class MAPFResultsAnalyzer:
    """Analyzes and plots MAPF benchmark results."""
    
    def __init__(self, base_dir: str):
        """Initialize the analyzer with the base directory."""
        self.base_dir = Path(base_dir)
        self.results = {
            'random-32x32-10': {},
            'random-32x32-20': {},
            'warehouse': {}
        }
        
        # Map type configurations
        self.map_configs = {
            'random-32x32-10': {
                'path': 'logs/benchmark_results_32x32-10/astar-cpp',
                'pattern': 'scene_even1_{n}agents.json',
                'label': 'Random 32×32 (10% obstacles)',
                'marker': 'o',
                'color': '#2E86AB',
                'linestyle': '-'
            },
            'random-32x32-20': {
                'path': 'logs/benchmark_results_32x32-02',
                'pattern': 'scene_even1_{n}agents.json',
                'label': 'Random 32×32 (20% obstacles)',
                'marker': 's',
                'color': '#A23B72',
                'linestyle': '--'
            },
            'warehouse': {
                'path': 'logs/benchmark_warehouse',
                'pattern': 'info_test_scene1_even_{n}agents.json',
                'label': 'Warehouse',
                'marker': '^',
                'color': '#F18F01',
                'linestyle': '-.'
            }
        }
        
    def load_data(self):
        """Load all JSON result files."""
        print("Loading data...")
        
        for map_type, config in self.map_configs.items():
            data_path = self.base_dir / config['path']
            
            if not data_path.exists():
                print(f"Warning: Directory not found: {data_path}")
                continue
                
            # Check for agent counts from 10 to 100
            for n_agents in range(10, 101, 10):
                filename = config['pattern'].format(n=n_agents)
                filepath = data_path / filename
                
                if filepath.exists():
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            self.results[map_type][n_agents] = data
                            print(f"  Loaded {map_type}: {n_agents} agents")
                    except Exception as e:
                        print(f"  Error loading {filepath}: {e}")
                else:
                    print(f"  File not found: {filepath}")
        
        print(f"\nData loading complete.")
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of loaded data."""
        print("\n=== Data Summary ===")
        for map_type in self.results:
            agent_counts = sorted(self.results[map_type].keys())
            if agent_counts:
                print(f"{map_type}: {len(agent_counts)} scenarios "
                      f"(agents: {min(agent_counts)}-{max(agent_counts)})")
            else:
                print(f"{map_type}: No data loaded")
    
    def extract_metrics(self) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
        """
        Extract metrics from loaded data.
        
        Returns:
            Dictionary with structure:
            {
                'success_rate': {'map_type': [(n_agents, value), ...]},
                'makespan': {...},
                'time': {...},
                'iu': {...}
            }
        """
        metrics = {
            'success_rate': {},
            'makespan': {},
            'time': {},
            'iu': {}
        }
        
        for map_type in self.results:
            metrics['success_rate'][map_type] = []
            metrics['makespan'][map_type] = []
            metrics['time'][map_type] = []
            metrics['iu'][map_type] = []
            
            for n_agents in sorted(self.results[map_type].keys()):
                data = self.results[map_type][n_agents]
                
                # Success rate: 100% if no final collisions, 0% otherwise
                final_collisions = data.get('final_collisions', 1)
                success_rate = 100.0 if final_collisions == 0 else 0.0
                metrics['success_rate'][map_type].append((n_agents, success_rate))
                
                # Makespan
                if 'jointPlan' in data and 'globalMakespan' in data['jointPlan']:
                    makespan = data['jointPlan']['globalMakespan']
                    metrics['makespan'][map_type].append((n_agents, makespan))
                
                # Computation time
                if 'time' in data:
                    time = data['time']
                    metrics['time'][map_type].append((n_agents, time))
                
                # Information Units
                if 'info_sharing' in data and 'totalInformationLoadIU' in data['info_sharing']:
                    iu = data['info_sharing']['totalInformationLoadIU']
                    metrics['iu'][map_type].append((n_agents, iu))
        
        return metrics
    
    def plot_metric(self, metric_name: str, metric_data: Dict[str, List[Tuple[int, float]]],
                   ylabel: str, title: str, filename: str, log_scale: bool = False):
        """
        Plot a single metric comparison across all map types.
        
        Args:
            metric_name: Name of the metric
            metric_data: Dictionary mapping map types to [(n_agents, value), ...]
            ylabel: Y-axis label
            title: Plot title
            filename: Output filename
            log_scale: Whether to use logarithmic scale for y-axis
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for map_type in ['random-32x32-10', 'random-32x32-20', 'warehouse']:
            if map_type not in metric_data or not metric_data[map_type]:
                continue
                
            config = self.map_configs[map_type]
            data_points = metric_data[map_type]
            
            if not data_points:
                continue
            
            # Sort by number of agents
            data_points = sorted(data_points, key=lambda x: x[0])
            n_agents = [x[0] for x in data_points]
            values = [x[1] for x in data_points]
            
            ax.plot(n_agents, values,
                   marker=config['marker'],
                   linestyle=config['linestyle'],
                   color=config['color'],
                   label=config['label'],
                   linewidth=2.5,
                   markersize=8,
                   alpha=0.85)
        
        ax.set_xlabel('Number of Agents', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        ax.legend(loc='best', framealpha=0.95, edgecolor='black')
        
        if log_scale:
            ax.set_yscale('log')
        
        # Set x-axis to show all agent counts
        ax.set_xticks(range(10, 101, 10))
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.base_dir / 'logs' / 'plots' / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all four metric plots."""
        print("\n=== Generating Plots ===")
        
        metrics = self.extract_metrics()
        
        # Plot 1: Success Rate
        self.plot_metric(
            'success_rate',
            metrics['success_rate'],
            'Success Rate (%)',
            'MAPF Success Rate vs Number of Agents',
            'success_rate_comparison.png'
        )
        
        # Plot 2: Makespan
        self.plot_metric(
            'makespan',
            metrics['makespan'],
            'Makespan (timesteps)',
            'MAPF Makespan vs Number of Agents',
            'makespan_comparison.png'
        )
        
        # Plot 3: Computation Time
        self.plot_metric(
            'time',
            metrics['time'],
            'Computation Time (seconds)',
            'MAPF Computation Time vs Number of Agents',
            'computation_time_comparison.png',
            log_scale=True
        )
        
        # Plot 4: Information Units
        self.plot_metric(
            'iu',
            metrics['iu'],
            'Information Units (IU)',
            'MAPF Information Load vs Number of Agents',
            'information_units_comparison.png'
        )
        
        print("\n=== Plot Generation Complete ===")
        print(f"All plots saved to: {self.base_dir / 'logs' / 'plots'}")
    
    def print_statistics(self):
        """Print statistical summary of the metrics."""
        print("\n=== Statistical Summary ===")
        metrics = self.extract_metrics()
        
        for metric_name, metric_data in metrics.items():
            print(f"\n{metric_name.upper()}:")
            for map_type in ['random-32x32-10', 'random-32x32-20', 'warehouse']:
                if map_type in metric_data and metric_data[map_type]:
                    values = [v for _, v in metric_data[map_type]]
                    print(f"  {map_type}:")
                    print(f"    Mean: {np.mean(values):.2f}")
                    print(f"    Std:  {np.std(values):.2f}")
                    print(f"    Min:  {np.min(values):.2f}")
                    print(f"    Max:  {np.max(values):.2f}")


def main():
    """Main function to run the analysis and plotting."""
    # Get the base directory (project root)
    base_dir = Path(__file__).parent
    
    # Create analyzer
    analyzer = MAPFResultsAnalyzer(base_dir)
    
    # Load data
    analyzer.load_data()
    
    # Generate plots
    analyzer.generate_all_plots()
    
    # Print statistics
    analyzer.print_statistics()
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
