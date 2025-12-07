#!/usr/bin/env python3
"""
ICAPS Analysis Script

Analyzes and presents MAPF experiment results from EPH, SCRIMP, and IO-MAPF algorithms.
Generates LaTeX tables and comparison plots.

Usage:
    python scripts/icaps_analysis.py [OPTIONS]

Options:
    --plot                  Generate comparison plots
    --compact               Use compact table format (grouped agent counts)
    --include-path-sub      Include path submission IU in IO-MAPF IU calculation
    --output FILE           Output LaTeX file (default: icaps_results_table.tex)
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class ICAPSAnalyzer:
    """Analyzes ICAPS benchmark results from multiple MAPF solvers."""

    def __init__(self, base_dir: str, include_path_submission: bool = False, iu_percent: bool = False):
        """
        Initialize the analyzer.

        Args:
            base_dir: Base directory of the project
            include_path_submission: If True, include path submission IU in IO-MAPF IU calculation
            iu_percent: If True, show IU as percentage increase relative to IO-MAPF baseline
        """
        self.base_dir = Path(base_dir)
        self.include_path_submission = include_path_submission
        self.iu_percent = iu_percent

        # Map configurations
        self.maps = {
            'random-32-32-20': {
                'label': 'random-32-32-20',
                'image': 'content/images/random-32-32-20.png',
                'display': r'\shortstack{\texttt{random}\\\texttt{32$\times$32-20}}',
                'agent_counts': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 128]
            },
            'random-64-64-20': {
                'label': 'random-64-64-20',
                'image': 'content/images/random-64-64-20.png',
                'display': r'\shortstack{\texttt{random}\\\texttt{64$\times$64-20}}',
                'agent_counts': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 128]
            },
            'den312d': {
                'label': 'den312d',
                'image': 'content/images/den312d',
                'display': r'\shortstack{\texttt{den312d}\\\texttt{65$\times$81}}',
                'agent_counts': [8, 16, 32, 64, 128]
            },
            'warehouse': {
                'label': 'warehouse',
                'image': 'content/images/warehouse',
                'display': r'\shortstack{\texttt{warehouse}\\\texttt{161$\times$63}}',
                'agent_counts': [8, 16, 32, 64, 128]
            }
        }

        # Solver configurations
        self.solvers = ['EECBS', 'IO-MAPF', 'DCC', 'SCRIMP', 'EPH']

        # Results storage
        self.results = {}

    def load_all_results(self):
        """Load results from all solvers and maps."""
        print("Loading results...")

        for map_name, map_config in self.maps.items():
            self.results[map_name] = {}

            for agent_count in map_config['agent_counts']:
                self.results[map_name][agent_count] = {
                    'EECBS': None,  # Placeholder for future
                    'IO-MAPF': self._load_iomapf_results(map_name, agent_count),
                    'DCC': None,    # Placeholder for future
                    'SCRIMP': self._load_scrimp_results(map_name, agent_count),
                    'EPH': self._load_eph_results(map_name, agent_count)
                }

        print("Results loading complete.")
        self._print_summary()

    def _load_eph_results(self, map_name: str, agent_count: int) -> Optional[Dict]:
        """
        Load EPH results for a given map and agent count.

        Returns:
            Dict with keys: 'el' (makespan), 'sr' (success rate), 'iu' (info units)
            or None if not found
        """
        results_path = self.base_dir / 'logs' / 'icaps_results_eph' / map_name / f'{agent_count}_agents' / 'results.json'

        if not results_path.exists():
            return None

        try:
            with open(results_path, 'r') as f:
                data = json.load(f)

            aggregate = data.get('aggregate', {})

            return {
                'el': aggregate.get('avg_makespan'),
                'sr': aggregate.get('success_rate'),
                'iu': aggregate.get('avg_info_units')
            }
        except Exception as e:
            print(f"  Error loading EPH {map_name}/{agent_count}: {e}")
            return None

    def _load_scrimp_results(self, map_name: str, agent_count: int) -> Optional[Dict]:
        """
        Load SCRIMP results for a given map and agent count.

        Returns:
            Dict with keys: 'el' (makespan), 'sr' (success rate), 'iu' (info units)
            or None if not found
        """
        results_path = self.base_dir / 'logs' / 'icaps_results_scrimp' / map_name / f'{agent_count}_agents' / f'results_{agent_count}_agents.pkl'

        if not results_path.exists():
            return None

        try:
            with open(results_path, 'rb') as f:
                data = pickle.load(f)

            if not isinstance(data, list) or len(data) == 0:
                return None

            # Aggregate across all instances
            makespans = [d.get('makespan', d.get('episode_len')) for d in data if d.get('makespan') or d.get('episode_len')]
            success_rates = [d.get('success_rate', 0) for d in data]
            iu_values = [d.get('information_sharing_iu', 0) for d in data]

            return {
                'el': np.mean(makespans) if makespans else None,
                'sr': np.mean(success_rates) * 100 if success_rates else None,  # Convert to percentage
                'iu': np.mean(iu_values) if iu_values else None
            }
        except Exception as e:
            print(f"  Error loading SCRIMP {map_name}/{agent_count}: {e}")
            return None

    def _load_iomapf_results(self, map_name: str, agent_count: int) -> Optional[Dict]:
        """
        Load IO-MAPF results for a given map and agent count.
        Aggregates across all test instances (test_01.json to test_20.json).

        Returns:
            Dict with keys: 'el' (makespan), 'sr' (success rate), 'iu' (info units)
            or None if not found
        """
        results_dir = self.base_dir / 'logs' / 'icaps_test' / map_name / f'{agent_count}_agents'

        if not results_dir.exists():
            return None

        makespans = []
        success_count = 0
        iu_values = []
        total_instances = 0

        # Load all test files
        for test_file in sorted(results_dir.glob('test_*.json')):
            try:
                with open(test_file, 'r') as f:
                    data = json.load(f)

                total_instances += 1

                # Success: final_collisions == 0
                if data.get('final_collisions', 1) == 0:
                    success_count += 1

                    # Only count makespan for successful instances
                    if 'jointPlan' in data and 'globalMakespan' in data['jointPlan']:
                        makespans.append(data['jointPlan']['globalMakespan'])
                    elif 'metrics' in data and 'makespan' in data['metrics']:
                        makespans.append(data['metrics']['makespan'])

                # IU calculation
                if 'info_sharing' in data:
                    info = data['info_sharing']
                    total_iu = info.get('totalInformationLoadIU', 0)

                    if not self.include_path_submission:
                        # Subtract path submission IU
                        initial_sub = info.get('initialSubmissionIU', 0)
                        revised_sub = info.get('revisedSubmissionIU', 0)
                        iu = total_iu - initial_sub - revised_sub
                    else:
                        iu = total_iu

                    iu_values.append(iu)

            except Exception as e:
                print(f"  Error loading {test_file}: {e}")
                continue

        if total_instances == 0:
            return None

        return {
            'el': np.mean(makespans) if makespans else None,
            'sr': (success_count / total_instances) * 100,
            'iu': np.mean(iu_values) if iu_values else None
        }

    def _print_summary(self):
        """Print summary of loaded data."""
        print("\n=== Data Summary ===")
        for map_name in self.maps:
            print(f"\n{map_name}:")
            for solver in self.solvers:
                available = []
                for agent_count in self.maps[map_name]['agent_counts']:
                    if self.results[map_name][agent_count][solver] is not None:
                        available.append(agent_count)
                if available:
                    print(f"  {solver}: {available}")
                else:
                    print(f"  {solver}: No data")

    def generate_latex_table(self, compact: bool = False) -> str:
        """
        Generate LaTeX table code.

        Args:
            compact: If True, use grouped agent counts for random maps

        Returns:
            LaTeX table code as string
        """
        if compact:
            return self._generate_compact_table()
        else:
            return self._generate_full_table()

    def _generate_full_table(self) -> str:
        """Generate full LaTeX table with all agent counts."""
        lines = []

        # Table header
        lines.append(r"\begin{table*}[t]")
        lines.append(r"  \centering")
        lines.append(r"  \scriptsize")
        lines.append(r"  \setlength{\tabcolsep}{3pt}")
        lines.append(r"  \renewcommand{\arraystretch}{1.1}")
        lines.append("")
        lines.append(r"  % 3 non-numeric cols + 15 numeric (5 solvers x 3 metrics)")
        lines.append(r"  \begin{tabular}{c c c | *{15}{r}}")
        lines.append(r"    \toprule")
        lines.append(r"    & & & \multicolumn{6}{c}{Search-based Solvers} & \multicolumn{9}{c}{Learning-based Solvers} \\")
        lines.append(r"    \cmidrule(lr){4-9} \cmidrule(lr){10-18}")
        lines.append(r"    Map & & $m$ &")
        lines.append(r"    \multicolumn{3}{c}{EECBS} &")
        lines.append(r"    \multicolumn{3}{c}{IO-MAPF} &")
        lines.append(r"    \multicolumn{3}{c}{DCC} &")
        lines.append(r"    \multicolumn{3}{c}{SCRIMP} &")
        lines.append(r"    \multicolumn{3}{c}{EPH} \\")
        lines.append(r"    & & &")
        lines.append(r"    EL & SR & IU &")
        lines.append(r"    EL & SR & IU &")
        lines.append(r"    EL & SR & IU &")
        lines.append(r"    EL & SR & IU &")
        lines.append(r"    EL & SR & IU \\")
        lines.append(r"    \midrule")
        lines.append("")

        # Generate rows for each map
        for map_name, map_config in self.maps.items():
            lines.extend(self._generate_map_rows(map_name, map_config))
            lines.append(r"    \midrule")
            lines.append("")

        # Remove last midrule
        lines = lines[:-2]

        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")

        # Generate caption based on iu_percent flag
        if self.iu_percent:
            caption = (r"  \caption{Performance of search-based (EECBS, IO-MAPF) and learning-based (DCC, SCRIMP, EPH) MAPF solvers. "
                      r"EL represents makespan (maximum timesteps). "
                      r"IU shows information units with percentage increase relative to IO-MAPF baseline in subscript (IO-MAPF = 0\%). "
                      r"Lower EL$\downarrow$ and IU$\downarrow$ are better, higher SR$\uparrow$ is better.}")
        else:
            caption = (r"  \caption{Performance of search-based (EECBS, IO-MAPF) and learning-based (DCC, SCRIMP, EPH) MAPF solvers. "
                      r"EL represents makespan (maximum timesteps). "
                      r"Lower EL$\downarrow$ and IU$\downarrow$ are better, higher SR$\uparrow$ is better.}")

        lines.append(caption)
        lines.append(r"  \label{tab:search-vs-learning}")
        lines.append(r"\end{table*}")

        return "\n".join(lines)

    def _generate_map_rows(self, map_name: str, map_config: Dict) -> List[str]:
        """Generate LaTeX rows for a single map."""
        lines = []
        agent_counts = map_config['agent_counts']
        num_rows = len(agent_counts)

        # Comment for map section
        lines.append(f"    % ----------------- {map_name} -----------------")

        for i, agent_count in enumerate(agent_counts):
            row_parts = []

            if i == 0:
                # First row includes map image and label
                row_parts.append(f"    \\multirow{{{num_rows}}}{{*}}{{\\includegraphics[height=1.5cm]{{content/images/{map_name}}}}}")
                row_parts.append(f" &\n    \\multirow{{{num_rows}}}{{*}}{{\\rotatebox{{90}}{{{map_config['display']}}}}}")
            else:
                row_parts.append("    ")
                row_parts.append(" &")

            row_parts.append(f" &\n    {agent_count}")

            # Get IO-MAPF IU as baseline for percentage calculation
            iomapf_result = self.results[map_name][agent_count].get('IO-MAPF')
            baseline_iu = iomapf_result.get('iu') if iomapf_result else None

            # Add metrics for each solver
            for solver in self.solvers:
                result = self.results[map_name][agent_count].get(solver)
                row_parts.append(self._format_metrics(result, solver=solver, baseline_iu=baseline_iu))

            lines.append("".join(row_parts) + r" \\")

        return lines

    def _format_metrics(self, result: Optional[Dict], solver: str = None, baseline_iu: float = None) -> str:
        """
        Format metrics for a single solver result.

        Args:
            result: Result dictionary with 'el', 'sr', 'iu' keys
            solver: Solver name (for IU percentage calculation)
            baseline_iu: IO-MAPF IU value to use as baseline for percentage
        """
        if result is None:
            return " & --- & --- & ---"

        el = self._format_value(result.get('el'), decimals=1)
        sr = self._format_value(result.get('sr'), decimals=0)

        # Format IU with optional percentage
        iu_val = result.get('iu')
        if iu_val is None:
            iu = "---"
        elif self.iu_percent and baseline_iu is not None and baseline_iu > 0:
            if solver == 'IO-MAPF':
                # IO-MAPF is baseline, show (0%)
                iu = f"{iu_val:.1f} \\textsubscript{{(0\\%)}}"
            else:
                # Calculate percentage increase relative to IO-MAPF
                pct_increase = ((iu_val - baseline_iu) / baseline_iu) * 100
                if pct_increase >= 0:
                    iu = f"{iu_val:.1f} \\textsubscript{{(+{pct_increase:.0f}\\%)}}"
                else:
                    iu = f"{iu_val:.1f} \\textsubscript{{({pct_increase:.0f}\\%)}}"
        else:
            iu = self._format_value(iu_val, decimals=1)

        return f" & {el} & {sr} & {iu}"

    def _format_value(self, value, decimals: int = 1) -> str:
        """Format a numeric value or return --- if None."""
        if value is None:
            return "---"
        if decimals == 0:
            return f"{value:.0f}"
        return f"{value:.{decimals}f}"

    def _generate_compact_table(self) -> str:
        """Generate compact LaTeX table with grouped agent counts."""
        lines = []

        # Groupings for random maps
        random_groups = {
            '10-30': [10, 20, 30],
            '40-60': [40, 50, 60],
            '70-90': [70, 80, 90],
            '100': [100],
            '128': [128]
        }

        # Groupings for benchmark maps (den312d, warehouse)
        benchmark_groups = {
            '8-16': [8, 16],
            '32-64': [32, 64],
            '128': [128]
        }

        # Table header
        lines.append(r"\begin{table*}[t]")
        lines.append(r"  \centering")
        lines.append(r"  \scriptsize")
        lines.append(r"  \setlength{\tabcolsep}{3pt}")
        lines.append(r"  \renewcommand{\arraystretch}{1.1}")
        lines.append("")
        lines.append(r"  \begin{tabular}{c c c | *{15}{r}}")
        lines.append(r"    \toprule")
        lines.append(r"    & & & \multicolumn{6}{c}{Search-based Solvers} & \multicolumn{9}{c}{Learning-based Solvers} \\")
        lines.append(r"    \cmidrule(lr){4-9} \cmidrule(lr){10-18}")
        lines.append(r"    Map & & $m$ &")
        lines.append(r"    \multicolumn{3}{c}{EECBS} &")
        lines.append(r"    \multicolumn{3}{c}{IO-MAPF} &")
        lines.append(r"    \multicolumn{3}{c}{DCC} &")
        lines.append(r"    \multicolumn{3}{c}{SCRIMP} &")
        lines.append(r"    \multicolumn{3}{c}{EPH} \\")
        lines.append(r"    & & &")
        lines.append(r"    EL & SR & IU &")
        lines.append(r"    EL & SR & IU &")
        lines.append(r"    EL & SR & IU &")
        lines.append(r"    EL & SR & IU &")
        lines.append(r"    EL & SR & IU \\")
        lines.append(r"    \midrule")
        lines.append("")

        # Generate rows for each map
        for map_name, map_config in self.maps.items():
            if map_name in ['random-32-32-20', 'random-64-64-20']:
                groups = random_groups
            else:
                groups = benchmark_groups

            lines.extend(self._generate_compact_map_rows(map_name, map_config, groups))
            lines.append(r"    \midrule")
            lines.append("")

        # Remove last midrule
        lines = lines[:-2]

        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")

        # Generate caption based on iu_percent flag
        if self.iu_percent:
            caption = (r"  \caption{Performance of search-based (EECBS, IO-MAPF) and learning-based (DCC, SCRIMP, EPH) MAPF solvers (compact view). "
                      r"EL represents makespan (maximum timesteps). Values are averaged within agent count groups. "
                      r"IU shows information units with percentage increase relative to IO-MAPF baseline in subscript (IO-MAPF = 0\%). "
                      r"Lower EL$\downarrow$ and IU$\downarrow$ are better, higher SR$\uparrow$ is better.}")
        else:
            caption = (r"  \caption{Performance of search-based (EECBS, IO-MAPF) and learning-based (DCC, SCRIMP, EPH) MAPF solvers (compact view). "
                      r"EL represents makespan (maximum timesteps). Values are averaged within agent count groups. "
                      r"Lower EL$\downarrow$ and IU$\downarrow$ are better, higher SR$\uparrow$ is better.}")

        lines.append(caption)
        lines.append(r"  \label{tab:search-vs-learning-compact}")
        lines.append(r"\end{table*}")

        return "\n".join(lines)

    def _generate_compact_map_rows(self, map_name: str, map_config: Dict, groups: Dict) -> List[str]:
        """Generate compact LaTeX rows for a single map with grouped agent counts."""
        lines = []
        group_names = list(groups.keys())
        num_rows = len(group_names)

        # Comment for map section
        lines.append(f"    % ----------------- {map_name} (compact) -----------------")

        for i, (group_name, agent_counts) in enumerate(groups.items()):
            row_parts = []

            if i == 0:
                # First row includes map image and label
                row_parts.append(f"    \\multirow{{{num_rows}}}{{*}}{{\\includegraphics[height=1.5cm]{{content/images/{map_name}}}}}")
                row_parts.append(f" &\n    \\multirow{{{num_rows}}}{{*}}{{\\rotatebox{{90}}{{{map_config['display']}}}}}")
            else:
                row_parts.append("    ")
                row_parts.append(" &")

            row_parts.append(f" &\n    {group_name}")

            # Get IO-MAPF averaged IU as baseline for percentage calculation
            iomapf_avg = self._average_results(map_name, agent_counts, 'IO-MAPF')
            baseline_iu = iomapf_avg.get('iu') if iomapf_avg else None

            # Add averaged metrics for each solver
            for solver in self.solvers:
                avg_result = self._average_results(map_name, agent_counts, solver)
                row_parts.append(self._format_metrics(avg_result, solver=solver, baseline_iu=baseline_iu))

            lines.append("".join(row_parts) + r" \\")

        return lines

    def _average_results(self, map_name: str, agent_counts: List[int], solver: str) -> Optional[Dict]:
        """Average results across multiple agent counts."""
        els = []
        srs = []
        ius = []

        for agent_count in agent_counts:
            if agent_count not in self.results[map_name]:
                continue
            result = self.results[map_name][agent_count].get(solver)
            if result is not None:
                if result.get('el') is not None:
                    els.append(result['el'])
                if result.get('sr') is not None:
                    srs.append(result['sr'])
                if result.get('iu') is not None:
                    ius.append(result['iu'])

        if not els and not srs and not ius:
            return None

        return {
            'el': np.mean(els) if els else None,
            'sr': np.mean(srs) if srs else None,
            'iu': np.mean(ius) if ius else None
        }

    def generate_plots(self, output_dir: Optional[str] = None):
        """
        Generate comparison plots.

        Args:
            output_dir: Output directory for plots (default: logs/plots)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
        except ImportError:
            print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
            return

        # Set publication-quality plotting style
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.labelsize'] = 11
        mpl.rcParams['axes.titlesize'] = 12
        mpl.rcParams['xtick.labelsize'] = 9
        mpl.rcParams['ytick.labelsize'] = 9
        mpl.rcParams['legend.fontsize'] = 9
        mpl.rcParams['figure.titlesize'] = 14
        mpl.rcParams['lines.linewidth'] = 2
        mpl.rcParams['lines.markersize'] = 6

        # Solver styles
        solver_styles = {
            'IO-MAPF': {'color': '#2E86AB', 'marker': 'o', 'linestyle': '-'},
            'SCRIMP': {'color': '#A23B72', 'marker': 's', 'linestyle': '--'},
            'EPH': {'color': '#F18F01', 'marker': '^', 'linestyle': '-.'}
        }

        # Create figure with subplots: 4 maps x 3 metrics
        fig, axes = plt.subplots(4, 3, figsize=(14, 16))
        fig.suptitle('MAPF Solver Comparison: IO-MAPF vs SCRIMP vs EPH', fontsize=14, fontweight='bold')

        metrics = [
            ('sr', 'Success Rate (%)', False),
            ('el', 'Makespan (EL)', False),
            ('iu', 'Information Units (IU)', False)
        ]

        map_names = list(self.maps.keys())

        for row, map_name in enumerate(map_names):
            agent_counts = self.maps[map_name]['agent_counts']

            for col, (metric_key, metric_label, log_scale) in enumerate(metrics):
                ax = axes[row, col]

                for solver in ['IO-MAPF', 'SCRIMP', 'EPH']:
                    x_vals = []
                    y_vals = []

                    for agent_count in agent_counts:
                        result = self.results[map_name][agent_count].get(solver)
                        if result is not None and result.get(metric_key) is not None:
                            x_vals.append(agent_count)
                            y_vals.append(result[metric_key])

                    if x_vals:
                        style = solver_styles[solver]
                        ax.plot(x_vals, y_vals,
                               marker=style['marker'],
                               linestyle=style['linestyle'],
                               color=style['color'],
                               label=solver,
                               alpha=0.85)

                # Formatting
                if col == 0:
                    ax.set_ylabel(f'{map_name}\n{metric_label}', fontsize=10)
                else:
                    ax.set_ylabel(metric_label, fontsize=10)

                if row == 0:
                    ax.set_title(metric_label, fontsize=11, fontweight='bold')

                if row == len(map_names) - 1:
                    ax.set_xlabel('Number of Agents', fontsize=10)

                ax.grid(True, alpha=0.3, linestyle=':')
                ax.set_xticks(agent_counts)
                ax.tick_params(axis='x', rotation=45)

                if log_scale:
                    ax.set_yscale('log')

                if row == 0 and col == 0:
                    ax.legend(loc='best', framealpha=0.95)

        plt.tight_layout()

        # Save figure
        if output_dir is None:
            output_dir = self.base_dir / 'logs' / 'plots'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'icaps_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.close()

    def print_results_table(self):
        """Print a simple text table of results to console."""
        print("\n" + "=" * 100)
        print("RESULTS SUMMARY")
        print("=" * 100)

        for map_name, map_config in self.maps.items():
            print(f"\n{map_name}:")
            print("-" * 90)
            print(f"{'Agents':<10} | {'IO-MAPF':<25} | {'SCRIMP':<25} | {'EPH':<25}")
            print(f"{'':10} | {'EL':>7} {'SR':>7} {'IU':>7} | {'EL':>7} {'SR':>7} {'IU':>7} | {'EL':>7} {'SR':>7} {'IU':>7}")
            print("-" * 90)

            for agent_count in map_config['agent_counts']:
                row = f"{agent_count:<10} |"

                for solver in ['IO-MAPF', 'SCRIMP', 'EPH']:
                    result = self.results[map_name][agent_count].get(solver)
                    if result is None:
                        row += f" {'---':>7} {'---':>7} {'---':>7} |"
                    else:
                        el = f"{result['el']:.1f}" if result.get('el') is not None else '---'
                        sr = f"{result['sr']:.0f}" if result.get('sr') is not None else '---'
                        iu = f"{result['iu']:.1f}" if result.get('iu') is not None else '---'
                        row += f" {el:>7} {sr:>7} {iu:>7} |"

                print(row)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='ICAPS Analysis - Generate tables and plots from MAPF experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/icaps_analysis.py                    # Generate full table
  python scripts/icaps_analysis.py --compact          # Generate compact table
  python scripts/icaps_analysis.py --plot             # Generate plots
  python scripts/icaps_analysis.py --include-path-sub # Include path submission in IU
        """
    )

    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--compact', action='store_true',
                       help='Use compact table format (grouped agent counts)')
    parser.add_argument('--include-path-sub', action='store_true',
                       help='Include path submission IU in IO-MAPF IU calculation')
    parser.add_argument('--iu-percent', action='store_true',
                       help='Show IU as percentage increase relative to IO-MAPF baseline')
    parser.add_argument('--output', type=str, default='icaps_results_table.tex',
                       help='Output LaTeX file (default: icaps_results_table.tex)')
    parser.add_argument('--no-print', action='store_true',
                       help='Do not print results to console')

    args = parser.parse_args()

    # Get base directory (project root)
    base_dir = Path(__file__).parent.parent

    # Create analyzer
    analyzer = ICAPSAnalyzer(base_dir,
                             include_path_submission=args.include_path_sub,
                             iu_percent=args.iu_percent)

    # Load results
    analyzer.load_all_results()

    # Print results to console
    if not args.no_print:
        analyzer.print_results_table()

    # Generate LaTeX table
    latex_table = analyzer.generate_latex_table(compact=args.compact)

    # Save table
    output_path = base_dir / args.output
    with open(output_path, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to: {output_path}")

    # Print table to console
    print("\n" + "=" * 80)
    print("GENERATED LATEX TABLE:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    # Generate plots if requested
    if args.plot:
        analyzer.generate_plots()

    print("\nDone!")


if __name__ == '__main__':
    main()
