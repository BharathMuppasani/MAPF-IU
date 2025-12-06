#!/bin/bash

# Benchmark script for random-32-32-20 map
# Map size: 32x32, Free cells: 819
# Agent counts: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 128
# Test files per config: 10

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Configuration
MAP_NAME="random-32-32-20"
STRATEGY="best"
INFO_SETTING="all"
SEARCH_TYPE="astar-cpp"
ALGO="dqn"
TIMEOUT=300  # 5 minutes
MAX_EXPANSIONS=10000  # 819 free cells × ~12

# Directories
INPUT_DIR="test_data/icaps_test/$MAP_NAME"
OUTPUT_DIR="logs/icaps_test/$MAP_NAME"

# Agent counts for random maps
# AGENT_COUNTS="10 20 30 40 50 60 70 80 90 100 128"
AGENT_COUNTS="90 100 128"


echo "=============================================="
echo "ICAPS Benchmark: $MAP_NAME"
echo "=============================================="
echo "Strategy: $STRATEGY"
echo "Info Setting: $INFO_SETTING"
echo "Search Type: $SEARCH_TYPE"
echo "Algorithm: $ALGO"
echo "Timeout: ${TIMEOUT}s (5 min)"
echo "Max Expansions: $MAX_EXPANSIONS"
echo "=============================================="
echo ""

# Track results
total_tests=0
passed_tests=0
failed_tests=0

# Run tests for each agent count
for num_agents in $AGENT_COUNTS; do
    agent_input_dir="$INPUT_DIR/${num_agents}_agents"
    agent_output_dir="$OUTPUT_DIR/${num_agents}_agents"

    # Create output directory
    mkdir -p "$agent_output_dir"

    echo ""
    echo "=============================================="
    echo "Processing: $num_agents agents"
    echo "=============================================="

    # Check if input directory exists
    if [ ! -d "$agent_input_dir" ]; then
        echo "WARNING: Input directory not found: $agent_input_dir"
        continue
    fi

    # Run all test files
    for test_file in "$agent_input_dir"/test_*.txt; do
        if [ ! -f "$test_file" ]; then
            continue
        fi

        test_name=$(basename "$test_file" .txt)
        log_file="$agent_output_dir/${test_name}.json"
        stdout_file="$agent_output_dir/${test_name}.log"

        echo -n "  Running $test_name... "

        # Run experiment (timeout handled by run_exp.py)
        python run_exp.py \
            --strategy "$STRATEGY" \
            --info "$INFO_SETTING" \
            --search_type "$SEARCH_TYPE" \
            --algo "$ALGO" \
            --timeout "$TIMEOUT" \
            --max_expansions "$MAX_EXPANSIONS" \
            --map_file "$test_file" \
            --log_file "$log_file" \
            --verbose \
            > "$stdout_file" 2>&1

        exit_code=$?
        total_tests=$((total_tests + 1))

        if [ $exit_code -eq 0 ]; then
            echo "OK"
            passed_tests=$((passed_tests + 1))
        else
            echo "FAILED (exit: $exit_code)"
            failed_tests=$((failed_tests + 1))
        fi
    done
done

# Summary
echo ""
echo "=============================================="
echo "BENCHMARK SUMMARY: $MAP_NAME"
echo "=============================================="
echo "Total Tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"
echo "Success Rate: $(echo "scale=1; $passed_tests * 100 / $total_tests" | bc)%"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo "=============================================="
