#!/usr/bin/env python3
"""
Generate LaTeX tables from MAPF benchmark results.

This script creates publication-ready LaTeX tables showing performance metrics
(Success Rate, Makespan, Time, Information Units) across different agent counts
for multiple map types.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class LaTeXTableGenerator:
    """Generates LaTeX tables from MAPF benchmark results."""
    
    def __init__(self, base_dir: str):
        """Initialize the generator with the base directory."""
        self.base_dir = Path(base_dir)
        
        # Map configurations
        self.map_configs = [
            {
                'name': '32x32-10',
                'path': 'logs/benchmark_results_32x32-10/astar-cpp',
                'pattern': 'scene_even1_{n}agents.json',
                'label': '32×32 random (10\\% obstacles) with A* CPP',
                'agent_counts': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            },
            {
                'name': '32x32-20',
                'path': 'logs/benchmark_results_32x32-20/astar-cpp',
                'pattern': 'scene_even1_{n}agents.json',
                'label': '32×32 random (20\\% obstacles) with A* CPP',
                'agent_counts': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            },
            {
                'name': 'warehouse',
                'path': 'logs/benchmark_warehouse',
                'pattern': 'info_test_scene1_even_{n}agents.json',
                'label': 'Warehouse',
                'agent_counts': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            }
        ]
        
        self.results = {}
    
    def load_data(self):
        """Load all JSON result files."""
        print("Loading data...")
        
        for config in self.map_configs:
            map_name = config['name']
            self.results[map_name] = {}
            data_path = self.base_dir / config['path']
            
            if not data_path.exists():
                print(f"Warning: Directory not found: {data_path}")
                continue
            
            for n_agents in config['agent_counts']:
                filename = config['pattern'].format(n=n_agents)
                filepath = data_path / filename
                
                if filepath.exists():
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            self.results[map_name][n_agents] = data
                            print(f"  Loaded {map_name}: {n_agents} agents")
                    except Exception as e:
                        print(f"  Error loading {filepath}: {e}")
        
        print("Data loading complete.\n")
    
    def extract_metrics(self, map_name: str, n_agents: int) -> Dict[str, float]:
        """
        Extract metrics for a specific map and agent count.
        
        Returns:
            Dictionary with keys: 'sr', 'ms', 'time', 'iu'
        """
        if map_name not in self.results or n_agents not in self.results[map_name]:
            return None
        
        data = self.results[map_name][n_agents]
        metrics = {}
        
        # Success Rate
        final_collisions = data.get('final_collisions', 1)
        metrics['sr'] = 100.0 if final_collisions == 0 else 0.0
        
        # Makespan
        if 'jointPlan' in data and 'globalMakespan' in data['jointPlan']:
            metrics['ms'] = data['jointPlan']['globalMakespan']
        else:
            metrics['ms'] = None
        
        # Time (in seconds)
        if 'time' in data:
            metrics['time'] = data['time']  # Keep in seconds
        else:
            metrics['time'] = None
        
        # Information Units
        if 'info_sharing' in data and 'totalInformationLoadIU' in data['info_sharing']:
            metrics['iu'] = data['info_sharing']['totalInformationLoadIU']
        else:
            metrics['iu'] = None
        
        # Conflicts Resolved (static conflicts from alertDetailsIU)
        if 'info_sharing' in data and 'alertDetailsIU' in data['info_sharing']:
            alert_details = data['info_sharing']['alertDetailsIU']
            # Static conflicts are the ones detected and resolved during planning
            metrics['conflicts'] = alert_details.get('static', 0)
        else:
            metrics['conflicts'] = None
        
        return metrics
    
    def format_value(self, value, metric_type='number', bold=False):
        """Format a value for LaTeX table."""
        if value is None:
            return '-'
        
        if metric_type == 'sr':
            formatted = f"{value:.2f}\\%"
        elif metric_type == 'time':
            formatted = f"{value:.2f}"
        elif metric_type == 'number':
            if isinstance(value, float):
                formatted = f"{value:.1f}"
            else:
                formatted = f"{value}"
        else:
            formatted = f"{value}"
        
        if bold:
            formatted = f"\\textbf{{{formatted}}}"
        
        return formatted
    
    def find_best_values(self, map_name: str, agent_counts: List[int], metric: str, 
                        higher_is_better: bool = True) -> set:
        """Find agent counts with best values for a given metric."""
        best_agents = set()
        
        for n_agents in agent_counts:
            metrics = self.extract_metrics(map_name, n_agents)
            if metrics and metrics.get(metric) is not None:
                # For now, don't bold any values - user can customize later
                pass
        
        return best_agents
    
    def generate_subtable(self, config: dict) -> str:
        """Generate LaTeX code for a single subtable."""
        map_name = config['name']
        label = config['label']
        agent_counts = [a for a in config['agent_counts'] 
                       if a in self.results.get(map_name, {})]
        
        if not agent_counts:
            return f"% No data available for {map_name}\n"
        
        # Determine column groupings based on agent counts
        # Group into sets of 3 or 4 agents
        num_agents = len(agent_counts)
        
        latex = []
        latex.append("  \\begin{subtable}{\\textwidth}")
        latex.append("    \\centering")
        latex.append("    \\scriptsize")
        latex.append("    \\setlength{\\tabcolsep}{4pt}")
        latex.append(f"    \\caption{{{label}}}")
        latex.append(f"    \\label{{tab:perf_{map_name}}}")
        
        # Create column specification
        col_spec = "@{}>{{\\raggedright\\arraybackslash}}p{5.0em}"
        for _ in range(num_agents):
            col_spec += ">{{\\centering\\arraybackslash}}p{3.8em}"
        col_spec += "@{}"
        
        latex.append(f"    \\begin{{tabular}}{{{col_spec}}}")
        latex.append("      \\toprule")
        
        # Header row with agent counts
        agent_cols = " & ".join([str(a) for a in agent_counts])
        latex.append(f"      Metrics & {agent_cols} \\\\")
        latex.append("      \\midrule")
        
        # Success Rate row
        sr_values = []
        for n_agents in agent_counts:
            metrics = self.extract_metrics(map_name, n_agents)
            if metrics and metrics['sr'] is not None:
                sr_values.append(self.format_value(metrics['sr'], 'sr'))
            else:
                sr_values.append('-')
        latex.append(f"      SR (\\%) ($\\uparrow$) & {' & '.join(sr_values)} \\\\")
        
        # Makespan row
        ms_values = []
        for n_agents in agent_counts:
            metrics = self.extract_metrics(map_name, n_agents)
            if metrics and metrics['ms'] is not None:
                ms_values.append(self.format_value(metrics['ms'], 'number'))
            else:
                ms_values.append('-')
        latex.append(f"      MS ($\\downarrow$) & {' & '.join(ms_values)} \\\\")
        
        # Time row (in minutes)
        time_values = []
        for n_agents in agent_counts:
            metrics = self.extract_metrics(map_name, n_agents)
            if metrics and metrics['time'] is not None:
                time_values.append(self.format_value(metrics['time'], 'time'))
            else:
                time_values.append('-')
        latex.append(f"      Time (s) ($\\downarrow$) & {' & '.join(time_values)} \\\\")
        
        # Information Units row
        iu_values = []
        for n_agents in agent_counts:
            metrics = self.extract_metrics(map_name, n_agents)
            if metrics and metrics['iu'] is not None:
                iu_values.append(self.format_value(metrics['iu'], 'number'))
            else:
                iu_values.append('-')
        latex.append(f"      IU ($\\downarrow$) & {' & '.join(iu_values)} \\\\")
        
        # Conflicts Resolved row
        conflict_values = []
        for n_agents in agent_counts:
            metrics = self.extract_metrics(map_name, n_agents)
            if metrics and metrics['conflicts'] is not None:
                conflict_values.append(self.format_value(metrics['conflicts'], 'number'))
            else:
                conflict_values.append('-')
        latex.append(f"      Conflicts ($\\downarrow$) & {' & '.join(conflict_values)} \\\\")
        
        latex.append("      \\bottomrule")
        latex.append("    \\end{tabular}")
        latex.append("  \\end{subtable}")
        latex.append("")
        
        return "\n".join(latex)
    
    def generate_full_table(self) -> str:
        """Generate the complete LaTeX table with all subtables."""
        latex = []
        
        latex.append("\\begin{table*}[!t]")
        latex.append("  \\centering")
        latex.append("  \\caption{\\footnotesize")
        latex.append("  Comparison of performance metrics for MAPF problems across different map types and agent counts.")
        latex.append("  Results show Success Rate (SR), Makespan (MS), Computation Time (T), and Information Units (IU).")
        latex.append("  Higher values are better for SR ($\\uparrow$), while lower values are better for MS, Time, and IU ($\\downarrow$).")
        latex.append("  }")
        latex.append("  \\label{tab:mapf_performance}")
        latex.append("")
        
        # Generate subtables for each map
        for config in self.map_configs:
            if config['name'] in self.results and self.results[config['name']]:
                latex.append(self.generate_subtable(config))
        
        latex.append("\\end{table*}")
        
        return "\n".join(latex)
    
    def save_table(self, output_file: str):
        """Save the LaTeX table to a file."""
        output_path = self.base_dir / output_file
        table_content = self.generate_full_table()
        
        with open(output_path, 'w') as f:
            f.write(table_content)
        
        print(f"LaTeX table saved to: {output_path}")
        print(f"\nYou can include this in your LaTeX document with:")
        print(f"  \\input{{{output_file}}}")
        
        # Also print to console
        print("\n" + "="*80)
        print("GENERATED LATEX TABLE:")
        print("="*80)
        print(table_content)
        print("="*80)
    
    def print_summary(self):
        """Print a summary of available data."""
        print("\n=== Data Summary ===")
        for config in self.map_configs:
            map_name = config['name']
            if map_name in self.results:
                agent_counts = sorted(self.results[map_name].keys())
                if agent_counts:
                    print(f"{map_name}: {len(agent_counts)} scenarios "
                          f"(agents: {min(agent_counts)}-{max(agent_counts)})")
                else:
                    print(f"{map_name}: No data")
            else:
                print(f"{map_name}: No data")


def main():
    """Main function to generate LaTeX tables."""
    # Get the base directory (project root)
    base_dir = Path(__file__).parent
    
    # Create generator
    generator = LaTeXTableGenerator(base_dir)
    
    # Load data
    generator.load_data()
    
    # Print summary
    generator.print_summary()
    
    # Generate and save table
    generator.save_table('results_table.tex')
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
