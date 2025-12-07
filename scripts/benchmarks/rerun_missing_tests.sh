#!/bin/bash

# Re-run script for missing/corrupted test files
# These tests were deleted due to git merge conflicts
#
# Missing tests:
# - den312d/8_agents: test_10, test_15
# - den312d/64_agents: test_06, test_12, test_13
# - random-32-32-20/20_agents: test_14
# - random-32-32-20/30_agents: test_08
# - random-32-32-20/40_agents: test_10
# - random-32-32-20/60_agents: test_06
# - random-32-32-20/80_agents: test_16
# - random-32-32-20/90_agents: test_07, test_09

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Common configuration
STRATEGY="best"
INFO_SETTING="all"
SEARCH_TYPE="astar-cpp"
ALGO="dqn"
TIMEOUT=600  # 10 minutes

echo "=============================================="
echo "Re-running Missing ICAPS Tests"
echo "=============================================="
echo "Strategy: $STRATEGY"
echo "Info Setting: $INFO_SETTING"
echo "Search Type: $SEARCH_TYPE"
echo "Algorithm: $ALGO"
echo "Timeout: ${TIMEOUT}s"
echo "=============================================="
echo ""

# Track results
total_tests=0
passed_tests=0
failed_tests=0

# Function to run a single test
run_test() {
    local map_name="$1"
    local num_agents="$2"
    local test_num="$3"
    local max_expansions="$4"

    local input_dir="test_data/icaps_test/$map_name/${num_agents}_agents"
    local output_dir="logs/icaps_test/$map_name/${num_agents}_agents"
    local test_file="$input_dir/test_${test_num}.txt"
    local log_file="$output_dir/test_${test_num}.json"
    local stdout_file="$output_dir/test_${test_num}.log"

    # Create output directory
    mkdir -p "$output_dir"

    # Check if input file exists
    if [ ! -f "$test_file" ]; then
        echo "  ERROR: Input file not found: $test_file"
        return 1
    fi

    echo -n "  Running $map_name/${num_agents}_agents/test_${test_num}... "

    # Run experiment
    python run_exp.py \
        --strategy "$STRATEGY" \
        --info "$INFO_SETTING" \
        --search_type "$SEARCH_TYPE" \
        --algo "$ALGO" \
        --timeout "$TIMEOUT" \
        --max_expansions "$max_expansions" \
        --map_file "$test_file" \
        --log_file "$log_file" \
        --verbose \
        > "$stdout_file" 2>&1

    local exit_code=$?
    total_tests=$((total_tests + 1))

    if [ $exit_code -eq 0 ]; then
        echo "OK"
        passed_tests=$((passed_tests + 1))
        return 0
    else
        echo "FAILED (exit: $exit_code)"
        failed_tests=$((failed_tests + 1))
        return 1
    fi
}

# ============================================
# den312d tests (max_expansions=10000)
# ============================================
echo ""
echo "=============================================="
echo "Map: den312d"
echo "=============================================="

# den312d/8_agents: test_10, test_15
run_test "den312d" "8" "10" 10000
run_test "den312d" "8" "15" 10000

# den312d/64_agents: test_06, test_12, test_13
run_test "den312d" "64" "06" 10000
run_test "den312d" "64" "12" 10000
run_test "den312d" "64" "13" 10000

# ============================================
# random-32-32-20 tests (max_expansions=10000)
# ============================================
echo ""
echo "=============================================="
echo "Map: random-32-32-20"
echo "=============================================="

# random-32-32-20/20_agents: test_14
run_test "random-32-32-20" "20" "14" 10000

# random-32-32-20/30_agents: test_08
run_test "random-32-32-20" "30" "08" 10000

# random-32-32-20/40_agents: test_10
run_test "random-32-32-20" "40" "10" 10000

# random-32-32-20/60_agents: test_06
run_test "random-32-32-20" "60" "06" 10000

# random-32-32-20/80_agents: test_16
run_test "random-32-32-20" "80" "16" 10000

# random-32-32-20/90_agents: test_07, test_09
run_test "random-32-32-20" "90" "07" 10000
run_test "random-32-32-20" "90" "09" 10000

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "RE-RUN SUMMARY"
echo "=============================================="
echo "Total Tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"
if [ $total_tests -gt 0 ]; then
    echo "Success Rate: $(echo "scale=1; $passed_tests * 100 / $total_tests" | bc)%"
fi
echo "=============================================="
echo ""
echo "After running, verify with:"
echo "  python scripts/icaps_analysis.py"
echo "=============================================="
