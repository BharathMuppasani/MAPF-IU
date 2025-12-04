import os
import json
import pandas as pd
import numpy as np
import re

def generate_html_report_for_custom_algo():
    """
    Analyzes performance data from custom algorithm's JSON log files
    and generates a comprehensive HTML report.
    """
    logs_dir = os.path.join("logs", "info_test")

    all_data = []
    print("Searching for JSON log files...")

    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory not found at '{logs_dir}'")
        return

    filename_pattern = re.compile(r"info_test_map_(\d+)_(\d+)\.json")

    for filename in os.listdir(logs_dir):
        match = filename_pattern.match(filename)
        if match:
            file_path = os.path.join(logs_dir, filename)
            num_agents = int(match.group(1))
            problem_id = int(match.group(2))

            try:
                with open(file_path, "r") as f:
                    log_data = json.load(f)

                # --- Correctly build a map of all available plans ---
                plan_map = {}
                # Handle original plans which use 'subplanId'
                for p in log_data.get("agentPaths", []):
                    if "subplanId" in p:
                        plan_map[p["subplanId"]] = p
                # Handle replanned paths which use 'id'
                for p in log_data.get("agentSubplans", []):
                    if "id" in p:
                        plan_map[p["id"]] = p

                # --- Robust Success Check: Did every agent reach its goal? ---
                all_agents_reached_goal = True
                if not log_data.get("agents"): # If there are no agents, it can't be a success
                    all_agents_reached_goal = False
                
                for agent_info in log_data.get("agents", []):
                    agent_id = agent_info["id"]
                    goal_pos = tuple(agent_info["goalState"]["cell"])
                    
                    # Find the agent's final plan ID from the jointPlan
                    agent_index = int(agent_id.split('-')[1]) - 1
                    final_plan_id = log_data["jointPlan"]["subplans"][agent_index]

                    if final_plan_id in plan_map:
                        final_plan = plan_map[final_plan_id]
                        if not final_plan["steps"]: # Empty plan means failure
                            all_agents_reached_goal = False
                            break
                        
                        last_step = tuple(final_plan["steps"][-1]["cell"])
                        if last_step != goal_pos:
                            all_agents_reached_goal = False
                            break
                    else:
                        # If the agent's final plan isn't in the map, it's a failure
                        all_agents_reached_goal = False
                        break

                success = 1.0 if all_agents_reached_goal else 0.0

                run_metrics = {
                    'num_agents': num_agents,
                    'problem_id': problem_id,
                    'success_rate': success,
                    'makespan': log_data["jointPlan"].get("globalMakespan", 0),
                    'information_sharing_iu': log_data.get("informationSharingMetrics", {}).get("totalInformationLoadIU", 0)
                }
                all_data.append(run_metrics)

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"  Warning: Could not process file {filename}. Error: {e}")

    if not all_data:
        print("\nNo valid log data found. Aborting report generation.")
        return

    print(f"\nSuccessfully loaded and processed {len(all_data)} log files.")
    print("Generating the report...")

    df = pd.DataFrame(all_data)

    # --- Data Transformation ---
    df['Status'] = df['success_rate'].apply(lambda x: 'Success' if x == 1.0 else 'Failure')
    df.loc[df['Status'] == 'Failure', 'makespan'] = np.nan
    df.rename(columns={
        'problem_id': 'Problem ID',
        'makespan': 'Makespan',
        'information_sharing_iu': 'Info Sharing IU'
    }, inplace=True)

    # --- HTML Report Generation ---
    html_content = """
    <html>
    <head>
        <title>Custom MAPF Algorithm Performance Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 80%; margin-top: 20px; margin-bottom: 40px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .summary-table th { background-color: #d4edda; }
        </style>
    </head>
    <body>
        <h1>Custom MAPF Algorithm Performance Report</h1>
    """

    for n_agents in sorted(df['num_agents'].unique(), reverse=True):
        html_content += f"<h2>Detailed Report for {n_agents} Agents</h2>"
        group_df = df[df['num_agents'] == n_agents].sort_values(by='Problem ID')
        detailed_table_df = group_df[['Problem ID', 'Status', 'Makespan', 'Info Sharing IU']]
        html_content += detailed_table_df.to_html(index=False, na_rep='N/A', justify='left')

    html_content += "<h2>Summary of Successful Runs (Averages)</h2>"
    successful_runs_df = df[df['Status'] == 'Success'].copy()

    if not successful_runs_df.empty:
        summary = successful_runs_df.groupby('num_agents').agg(
            Success_Count=('Status', 'count'),
            Avg_Makespan=('Makespan', 'mean'),
            Avg_Info_Sharing_IU=('Info Sharing IU', 'mean')
        ).reset_index()
        summary.rename(columns={'num_agents': 'Num Agents'}, inplace=True)
        summary['Avg_Makespan'] = summary['Avg_Makespan'].round(1)
        summary['Avg_Info_Sharing_IU'] = summary['Avg_Info_Sharing_IU'].astype(int)
        html_content += summary.sort_values(by='Num Agents', ascending=False).to_html(
            index=False, justify='left', classes='summary-table'
        )
    else:
        html_content += "<p>No successful runs were found to generate a summary.</p>"

    html_content += "</body></html>"

    # --- Save the HTML Report ---
    report_filename = "custom_algorithm_report.html"
    # <<< THIS BLOCK IS NOW CORRECTED >>>
    with open(report_filename, "w") as f:
        f.write(html_content)
    
    print(f"\n✔ Report generation complete. Open '{report_filename}' in your browser.")

if __name__ == "__main__":
    generate_html_report_for_custom_algo()