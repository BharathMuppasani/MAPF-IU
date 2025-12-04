#!/bin/bash

# Benchmark script to run collision resolution for different agent counts
# Usage: ./run_all_benchmarks.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
STRATEGY="best"
INFO_SETTING="all"
SEARCH_TYPE="astar-cpp"
ALGO="dqn"
AGENT_COUNTS="100"
MAP_DIR="test_data/maps/random-32-32-10/benchmark_converted"

# Create results directory
RESULTS_DIR="logs/benchmark_results_32x32-10/astar-cpp"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "MAPF Collision Resolution Benchmark"
echo "=========================================="
echo "Strategy: $STRATEGY"
echo "Info Setting: $INFO_SETTING"
echo "Search Type: $SEARCH_TYPE"
echo "Algorithm: $ALGO"
echo "Results Directory: $RESULTS_DIR"
echo "=========================================="
echo ""

# Track results
total_tests=0
passed_tests=0

# Run tests for each agent count
for num_agents in $AGENT_COUNTS; do
    map_file="$MAP_DIR/scene_even2_${num_agents}agents.txt"

    echo ""
    echo "=========================================="
    echo "Running: $num_agents agents"
    echo "Map: $map_file"
    echo "=========================================="

    # Check if map file exists
    if [ ! -f "$map_file" ]; then
        echo "❌ ERROR: Map file not found: $map_file"
        total_tests=$((total_tests + 1))
        continue
    fi

    # Run the experiment
    if python run_exp.py \
        --strategy "$STRATEGY" \
        --info "$INFO_SETTING" \
        --search_type "$SEARCH_TYPE" \
        --algo "$ALGO" \
        --map_file "$map_file" \
        --log_file "$RESULTS_DIR/scene_even1_${num_agents}agents.json" \
        2>&1 | tee "$RESULTS_DIR/${num_agents}agents.log"
    then
        echo "✓ Completed: $num_agents agents"
        passed_tests=$((passed_tests + 1))
    else
        echo "❌ FAILED: $num_agents agents"
    fi

    total_tests=$((total_tests + 1))
    echo ""
done

# Summary
echo ""
echo "=========================================="
echo "BENCHMARK SUMMARY"
echo "=========================================="
echo "Total Tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $((total_tests - passed_tests))"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Output log files (JSON):"
ls -lh "$RESULTS_DIR"/log_*.json 2>/dev/null || echo "  No log files generated"
echo ""
echo "Stdout logs:"
ls -lh "$RESULTS_DIR"/stdout_*.log
echo "=========================================="
echo ""

# Check if all passed
if [ $passed_tests -eq $total_tests ] && [ $total_tests -gt 0 ]; then
    echo "✓ All benchmarks passed!"
    exit 0
else
    echo "⚠ Some tests completed. Check logs for details."
    exit 0
fi
