import os
import json
import pickle
import pandas as pd
import numpy as np
import re

def load_json_data(logs_dir, algorithm_name):
    """Loads and parses all JSON log files for Alert-BFS and Alert-A*."""
    all_data = []
    if not os.path.exists(logs_dir):
        print(f"Warning: Logs directory not found at '{logs_dir}' for {algorithm_name}")
        return pd.DataFrame()

    filename_pattern = re.compile(r"info_test_map_(\d+)_(\d+)\.json")
    for filename in sorted(os.listdir(logs_dir)):
        match = filename_pattern.match(filename)
        if match:
            num_agents, problem_id = int(match.group(1)), int(match.group(2))
            try:
                with open(os.path.join(logs_dir, filename), "r") as f:
                    log_data = json.load(f)
                
                all_agents_reached_goal = True
                if not log_data.get("agents"): all_agents_reached_goal = False
                plan_map = {p.get("subplanId") or p.get("id"): p for p in log_data.get("agentPaths", []) + log_data.get("agentSubplans", [])}
                for agent_info in log_data.get("agents", []):
                    agent_id = agent_info["id"]
                    goal_pos = tuple(agent_info["goalState"]["cell"])
                    agent_index = int(agent_id.split('-')[1]) - 1
                    if agent_index < len(log_data["jointPlan"]["subplans"]):
                        final_plan_id = log_data["jointPlan"]["subplans"][agent_index]
                        if final_plan_id in plan_map:
                            final_plan = plan_map[final_plan_id]
                            if not final_plan.get("steps") or tuple(final_plan["steps"][-1]["cell"]) != goal_pos:
                                all_agents_reached_goal = False; break
                        else: all_agents_reached_goal = False; break
                    else: all_agents_reached_goal = False; break

                all_data.append({
                    'num_agents': num_agents, 'problem_id': problem_id, 'algorithm': algorithm_name,
                    'status': 'Success' if all_agents_reached_goal else 'Failure',
                    'makespan': log_data["jointPlan"].get("globalMakespan") if all_agents_reached_goal else np.nan,
                    'iu': log_data.get("informationSharingMetrics", {}).get("totalInformationLoadIU", 0)
                })
            except Exception as e:
                print(f"  Warning: Could not process {filename} for {algorithm_name}. Error: {e}")
    
    return pd.DataFrame(all_data)

def load_pkl_data(logs_dir, algorithm_name, filename_pattern_str):
    """Generic loader for PKL result files (PRIMAL, SCRIMP)."""
    all_data = []
    if not os.path.exists(logs_dir):
        print(f"Warning: Logs directory not found at '{logs_dir}' for {algorithm_name}")
        return pd.DataFrame()

    filename_pattern = re.compile(filename_pattern_str)
    for filename in sorted(os.listdir(logs_dir)):
        match = filename_pattern.match(filename)
        if match:
            num_agents = int(match.group(1))
            try:
                with open(os.path.join(logs_dir, filename), "rb") as f:
                    results = pickle.load(f)
                for i, run in enumerate(results):
                    is_success = run.get('success_rate') == 1.0
                    all_data.append({
                        'num_agents': num_agents, 'problem_id': i, 'algorithm': algorithm_name,
                        'status': 'Success' if is_success else 'Failure',
                        'makespan': run.get('makespan') if is_success else np.nan,
                        'iu': run.get('information_sharing_iu', 0)
                    })
            except Exception as e:
                print(f"  Warning: Could not process {filename} for {algorithm_name}. Error: {e}")

    return pd.DataFrame(all_data)


def generate_comparative_report():
    """Generates a single HTML report comparing the four algorithms."""
    bfs_dir = os.path.join("logs", "info_test_bfs")
    astar_dir = os.path.join("logs", "info_test_astar")
    primal_dir = os.path.join("logs", "primal_info_test")
    scrimp_dir = os.path.join("logs", "scrimp_info_test")

    print("Loading data for Alert-BFS...")
    df_bfs = load_json_data(bfs_dir, 'Alert-BFS')
    
    print("\nLoading data for Alert-A*...")
    df_astar = load_json_data(astar_dir, 'Alert-A*')

    print("\nLoading data for PRIMAL...")
    df_primal = load_pkl_data(primal_dir, 'PRIMAL', r"info_test_primal_(\d+)_instrumented\.pkl")

    print("\nLoading data for SCRIMP...")
    df_scrimp = load_pkl_data(scrimp_dir, 'SCRIMP', r"random_scrimp_(\d+)_instrumented\.pkl")

    # Combine data from all algorithms into a single DataFrame
    df_combined = pd.concat([df_bfs, df_astar, df_primal, df_scrimp], ignore_index=True)
    
    if df_combined.empty:
        print("\nNo data found for any algorithm. Aborting.")
        return

    # --- Generate Summary Data for Text ---
    df_success = df_combined[df_combined['status'] == 'Success'].copy()
    summary_text = "<p>Could not generate summary text due to missing data for one or more algorithms.</p>"
    try:
        summary_pivot_text = df_success.groupby('algorithm')['iu'].mean().reset_index()
        iu_primal_avg = summary_pivot_text[summary_pivot_text['algorithm'] == 'PRIMAL']['iu'].iloc[0]
        iu_scrimp_avg = summary_pivot_text[summary_pivot_text['algorithm'] == 'SCRIMP']['iu'].iloc[0]
        iu_bfs_avg = summary_pivot_text[summary_pivot_text['algorithm'] == 'Alert-BFS']['iu'].iloc[0]
        
        bfs_reduction_vs_primal = ((iu_primal_avg - iu_bfs_avg) / iu_primal_avg) * 100 if iu_primal_avg > 0 else 0
        bfs_reduction_vs_scrimp = ((iu_scrimp_avg - iu_bfs_avg) / iu_scrimp_avg) * 100 if iu_scrimp_avg > 0 else 0

        summary_text = f"""
        <p style="font-size: 16px; line-height: 1.6;">
            This report presents a comparative analysis of four multi-agent pathfinding algorithms.
            The alert-based methods (Alert-BFS, Alert-A*) are benchmarked against learning-based approaches (PRIMAL, SCRIMP).
            On average, across all successfully solved instances, Alert-BFS demonstrated significant information efficiency, requiring
            <strong>{bfs_reduction_vs_primal:.1f}% less information</strong> than PRIMAL and
            <strong>{bfs_reduction_vs_scrimp:.1f}% less information</strong> than SCRIMP.
        </p>
        """
    except (IndexError, KeyError):
        pass # Keep default summary text if data is missing


    # --- HTML Report Generation ---
    html = f"""
    <html><head><title>4-Way Comparative Performance Report</title><style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }} h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0 40px; font-size: 12px; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: center; }}
        th.main-header {{ background-color: #e9ecef; }}
        th.sub-header {{ background-color: #f8f9fa; }}
        td.problem-id {{ text-align: left; font-weight: bold; }}
        .success {{ color: green; font-weight: bold; }}
        .failure {{ color: red; font-weight: bold; }}
        .best {{ font-weight: bold; background-color: #f0fff0; }}
        .summary-table th.main-header-bfs {{ background-color: #d4edda; }}
        .summary-table th.main-header-astar {{ background-color: #d1ecf1; }}
        .summary-table th.main-header-primal {{ background-color: #e6f7ff; }}
        .summary-table th.main-header-scrimp {{ background-color: #fbeaff; }}
    </style></head><body>
        <h1>Comparative Report: Alert-BFS vs. Alert-A* vs. PRIMAL vs. SCRIMP</h1>
        {summary_text}
    """

    # --- Generate Summary Table ---
    if not df_success.empty:
        summary = df_success.groupby(['num_agents', 'algorithm']).agg(
            Success_Count=('status', 'count'), Avg_Makespan=('makespan', 'mean'), Avg_IU=('iu', 'mean')
        ).reset_index()
        summary_pivot = summary.pivot_table(index='num_agents', columns='algorithm', values=['Success_Count', 'Avg_Makespan', 'Avg_IU'])
        summary_pivot.columns = [f'{val} ({col})' for val, col in summary_pivot.columns]
        
        # Calculate IU Reduction vs Baselines
        for algo in ['Alert-BFS', 'Alert-A*']:
            iu_algo_col = f'Avg_IU ({algo})'
            for baseline in ['PRIMAL', 'SCRIMP']:
                iu_baseline_col = f'Avg_IU ({baseline})'
                if iu_algo_col in summary_pivot and iu_baseline_col in summary_pivot:
                    reduction_col_name = f'IU Reduction ({algo} vs {baseline})'
                    summary_pivot[reduction_col_name] = ((summary_pivot[iu_baseline_col] - summary_pivot[iu_algo_col]) / summary_pivot[iu_baseline_col]) * 100
        
        summary_pivot.reset_index(inplace=True)

        html += """
        <h2>Comparative Summary (Averages of Successful Runs)</h2>
        <table class="summary-table"><thead><tr>
            <th rowspan="2">Num Agents</th>
            <th colspan="3" class="main-header-bfs">Alert-BFS</th>
            <th colspan="3" class="main-header-astar">Alert-A*</th>
            <th colspan="3" class="main-header-primal">PRIMAL</th>
            <th colspan="3" class="main-header-scrimp">SCRIMP</th>
            <th rowspan="2" class="main-header">IU Reduction (vs PRIMAL)</th>
            <th rowspan="2" class="main-header">IU Reduction (vs SCRIMP)</th>
        </tr><tr>
            <th class="sub-header">Success</th><th class="sub-header">Avg Makespan</th><th class="sub-header">Avg IU</th>
            <th class="sub-header">Success</th><th class="sub-header">Avg Makespan</th><th class="sub-header">Avg IU</th>
            <th class="sub-header">Success</th><th class="sub-header">Avg Makespan</th><th class="sub-header">Avg IU</th>
            <th class="sub-header">Success</th><th class="sub-header">Avg Makespan</th><th class="sub-header">Avg IU</th>
        </tr></thead><tbody>
        """
        for _, row in summary_pivot.sort_values(by='num_agents').iterrows():
            def f_int(val): return f"{int(val)}" if pd.notna(val) else "N/A"
            def f_float(val): return f"{val:.1f}" if pd.notna(val) else "N/A"
            html += f"""
                <tr>
                    <td class="problem-id">{f_int(row.get('num_agents'))}</td>
                    <td>{f_int(row.get('Success_Count (Alert-BFS)'))}</td><td>{f_float(row.get('Avg_Makespan (Alert-BFS)'))}</td><td>{f_int(row.get('Avg_IU (Alert-BFS)'))}</td>
                    <td>{f_int(row.get('Success_Count (Alert-A*)'))}</td><td>{f_float(row.get('Avg_Makespan (Alert-A*)'))}</td><td>{f_int(row.get('Avg_IU (Alert-A*)'))}</td>
                    <td>{f_int(row.get('Success_Count (PRIMAL)'))}</td><td>{f_float(row.get('Avg_Makespan (PRIMAL)'))}</td><td>{f_int(row.get('Avg_IU (PRIMAL)'))}</td>
                    <td>{f_int(row.get('Success_Count (SCRIMP)'))}</td><td>{f_float(row.get('Avg_Makespan (SCRIMP)'))}</td><td>{f_int(row.get('Avg_IU (SCRIMP)'))}</td>
                    <td>
                        <b>BFS:</b> {f_float(row.get('IU Reduction (Alert-BFS vs PRIMAL)'))}%<br>
                        <b>A*:</b> {f_float(row.get('IU Reduction (Alert-A* vs PRIMAL)'))}%
                    </td>
                    <td>
                        <b>BFS:</b> {f_float(row.get('IU Reduction (Alert-BFS vs SCRIMP)'))}%<br>
                        <b>A*:</b> {f_float(row.get('IU Reduction (Alert-A* vs SCRIMP)'))}%
                    </td>
                </tr>"""
        html += "</tbody></table>"

    # --- Generate Detailed Tables ---
    df_pivot_detailed = df_combined.pivot_table(index=['num_agents', 'problem_id'], columns='algorithm', values=['status', 'makespan', 'iu'], aggfunc='first').reset_index()
    df_pivot_detailed.columns = [f'{val[0]}_{val[1]}' if val[1] else val[0] for val in df_pivot_detailed.columns]
    
    for n_agents in sorted(df_pivot_detailed['num_agents'].unique()):
        html += f"<h2>Comparative Detailed Report for {int(n_agents)} Agents</h2>"
        html += """
        <table><thead><tr>
            <th rowspan="2">Problem ID</th>
            <th colspan="3" class="main-header-bfs">Alert-BFS</th>
            <th colspan="3" class="main-header-astar">Alert-A*</th>
            <th colspan="3" class="main-header-primal">PRIMAL</th>
            <th colspan="3" class="main-header-scrimp">SCRIMP</th>
        </tr><tr>
            <th class="sub-header">Status</th><th class="sub-header">Makespan</th><th class="sub-header">Info IU</th>
            <th class="sub-header">Status</th><th class="sub-header">Makespan</th><th class="sub-header">Info IU</th>
            <th class="sub-header">Status</th><th class="sub-header">Makespan</th><th class="sub-header">Info IU</th>
            <th class="sub-header">Status</th><th class="sub-header">Makespan</th><th class="sub-header">Info IU</th>
        </tr></thead><tbody>
        """
        group_df = df_pivot_detailed[df_pivot_detailed['num_agents'] == n_agents].sort_values(by='problem_id')
        for _, row in group_df.iterrows():
            metrics = {
                'BFS': {'status': row.get('status_Alert-BFS'), 'makespan': row.get('makespan_Alert-BFS'), 'iu': row.get('iu_Alert-BFS')},
                'A*': {'status': row.get('status_Alert-A*'), 'makespan': row.get('makespan_Alert-A*'), 'iu': row.get('iu_Alert-A*')},
                'PRIMAL': {'status': row.get('status_PRIMAL'), 'makespan': row.get('makespan_PRIMAL'), 'iu': row.get('iu_PRIMAL')},
                'SCRIMP': {'status': row.get('status_SCRIMP'), 'makespan': row.get('makespan_SCRIMP'), 'iu': row.get('iu_SCRIMP')}
            }
            
            valid_makespans = [m['makespan'] for m in metrics.values() if m['status'] == 'Success' and pd.notna(m['makespan'])]
            best_makespan = min(valid_makespans) if valid_makespans else None
            valid_ius = [m['iu'] for m in metrics.values() if pd.notna(m['iu'])]
            best_iu = min(valid_ius) if valid_ius else None

            html += f"<tr><td class='problem-id'>{int(row['problem_id'])}</td>"
            for algo_key, algo_name in [('BFS', 'Alert-BFS'), ('A*', 'Alert-A*'), ('PRIMAL', 'PRIMAL'), ('SCRIMP', 'SCRIMP')]:
                m = metrics[algo_key]
                status_html = f"<td class='{'success' if m['status'] == 'Success' else 'failure'}'>{'✔' if m['status'] == 'Success' else '✖'}</td>"
                makespan_html = f"<td class='{'best' if m['makespan'] == best_makespan else ''}'>{m['makespan'] if pd.notna(m['makespan']) else 'N/A'}</td>"
                iu_html = f"<td class='{'best' if m['iu'] == best_iu else ''}'>{int(m['iu']) if pd.notna(m['iu']) else 'N/A'}</td>"
                html += status_html + makespan_html + iu_html
            html += "</tr>"
        html += "</tbody></table>"

    html += "</body></html>"

    report_filename = "final_4way_comparative_report.html"
    with open(report_filename, "w") as f:
        f.write(html)
    
    print(f"\n✔ Final 4-way comparative report generated. Open '{report_filename}' in your browser.")

if __name__ == "__main__":
    generate_comparative_report()