#!/bin/bash

# Master script to run all ICAPS benchmarks
# Runs benchmarks for all 4 map types sequentially
#
# Usage:
#   ./run_all_icaps.sh           # Run all maps
#   ./run_all_icaps.sh random    # Run only random maps
#   ./run_all_icaps.sh complex   # Run only complex maps (den312d, warehouse)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Parse arguments
RUN_MODE="${1:-all}"

echo "=============================================="
echo "ICAPS Benchmark Suite"
echo "=============================================="
echo "Mode: $RUN_MODE"
echo "Start time: $(date)"
echo "=============================================="
echo ""

# Create main log directory
MASTER_LOG="logs/icaps_test/benchmark_run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$MASTER_LOG")"

run_benchmark() {
    local script=$1
    local name=$2

    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "Starting: $name"
    echo "Time: $(date)"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ""

    bash "$SCRIPT_DIR/$script" 2>&1 | tee -a "$MASTER_LOG"

    echo ""
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "Completed: $name"
    echo "Time: $(date)"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
}

# Run benchmarks based on mode
case $RUN_MODE in
    "random")
        run_benchmark "run_random_32x32.sh" "random-32-32-20"
        run_benchmark "run_random_64x64.sh" "random-64-64-20"
        ;;
    "complex")
        run_benchmark "run_den312d.sh" "den312d"
        run_benchmark "run_warehouse.sh" "warehouse"
        ;;
    "all"|*)
        run_benchmark "run_random_32x32.sh" "random-32-32-20"
        run_benchmark "run_random_64x64.sh" "random-64-64-20"
        run_benchmark "run_den312d.sh" "den312d"
        run_benchmark "run_warehouse.sh" "warehouse"
        ;;
esac

echo ""
echo "=============================================="
echo "ALL BENCHMARKS COMPLETED"
echo "=============================================="
echo "End time: $(date)"
echo "Master log: $MASTER_LOG"
echo ""
echo "Results structure:"
echo "  logs/icaps_test/"
echo "  ├── random-32-32-20/"
echo "  │   ├── 10_agents/"
echo "  │   │   ├── test_01.json"
echo "  │   │   └── test_01.log"
echo "  │   └── ..."
echo "  ├── random-64-64-20/"
echo "  ├── den312d/"
echo "  └── warehouse/"
echo "=============================================="
