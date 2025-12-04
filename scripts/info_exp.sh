#!/bin/bash

# This script automates running the MAPF experiments for your custom algorithm.
# It iterates through agent counts (5 to 20) and problem indices (0 to 9),
# constructing the map file path and executing the run_exp.py script for each.

# Stop the script if any command fails
set -e

# --- Configuration ---
# The directory where your map files are located.
MAP_DIR="test_data/maps/maps_11x11"
# The directory where the python script saves the logs.
LOGS_DIR="logs"

# --- Script Start ---
echo "Starting batch run for custom MAPF algorithm..."
echo "Maps will be sourced from: ${MAP_DIR}"
echo "JSON logs will be saved to: ${LOGS_DIR}"
echo "----------------------------------------------------"

# Check if the map directory exists before starting
if [ ! -d "$MAP_DIR" ]; then
    echo "Error: Map directory not found at '${MAP_DIR}'"
    echo "Please ensure your test data is in the correct location."
    exit 1
fi

# Outer loop: Iterates through the number of agents from 5 to 20.
for num_agents in $(seq 5 20); do
    echo ""
    echo "===== Processing Agent Configuration: ${num_agents} ====="

    # Inner loop: Iterates through the problem index from 0 to 9.
    for problem_index in $(seq 0 9); do
        MAP_FILE="${MAP_DIR}/map_${num_agents}_${problem_index}.txt"

        # Check if the map file actually exists before trying to run the experiment.
        if [ -f "$MAP_FILE" ]; then
            echo "--> Running: ${MAP_FILE}"
            
            # Execute the python script with the specified parameters and the constructed map file path.
            python run_exp.py \
                --search_type astar \
                --algo dqn \
                --strategy best \
                --timeout 30 \
                --map_file "$MAP_FILE" \
                --info all
            
            echo "--> Finished: ${MAP_FILE}"
        else
            echo "--> Skipping: ${MAP_FILE} (File not found)"
        fi
    done
done

echo ""
echo "----------------------------------------------------"
echo "All experiments completed successfully."
echo "All JSON log files have been saved in the '${LOGS_DIR}' directory."