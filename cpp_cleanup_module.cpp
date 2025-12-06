// cpp_cleanup_module.cpp
//
// C++ implementation of global cleanup operations for MAPF plans,
// exposed to Python via pybind11 as cpp_cleanup.
//
// Action encoding (MUST match Python and other modules):
// 0: up    (-1,  0)
// 1: down  ( 1,  0)
// 2: left  ( 0, -1)
// 3: right ( 0,  1)
// 4: wait  ( 0,  0)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <algorithm>
#include <tuple>

namespace py = pybind11;

// ==================== Constants ====================

static const int ACTION_WAIT = 4;

// Actions: 0=up, 1=down, 2=left, 3=right, 4=wait
static const int dr[5] = {-1, 1,  0, 0, 0};
static const int dc[5] = { 0, 0, -1, 1, 0};

// ==================== Data Structures ====================

struct CleanupCell {
    int r;
    int c;

    CleanupCell() : r(0), c(0) {}
    CleanupCell(int row, int col) : r(row), c(col) {}

    bool operator==(const CleanupCell& other) const {
        return r == other.r && c == other.c;
    }
};

struct CleanupResult {
    std::vector<std::vector<int>> plans;           // Cleaned plans
    std::vector<std::vector<CleanupCell>> trajectories;   // Re-simulated trajectories
    int original_makespan;                         // Original max plan length
    int cleaned_makespan;                          // After cleanup
    int timesteps_removed;                         // Number of synchronized WAITs removed
};

// ==================== Helper Functions ====================

// Simulate a plan from a starting position, returning trajectory
std::vector<CleanupCell> simulate_plan_cpp(
    const CleanupCell& start,
    const std::vector<int>& plan,
    const std::vector<std::vector<int>>& grid
) {
    std::vector<CleanupCell> trajectory;
    trajectory.reserve(plan.size() + 1);

    CleanupCell pos = start;
    trajectory.push_back(pos);

    int H = grid.size();
    int W = (H > 0) ? grid[0].size() : 0;

    for (int action : plan) {
        if (action < 0 || action > 4) {
            action = ACTION_WAIT;  // Invalid action → treat as wait
        }

        int nr = pos.r + dr[action];
        int nc = pos.c + dc[action];

        // Bounds check and obstacle check
        if (nr >= 0 && nr < H && nc >= 0 && nc < W && grid[nr][nc] != -1) {
            pos.r = nr;
            pos.c = nc;
        }
        // If invalid move, stay in place (like a wait)

        trajectory.push_back(pos);
    }

    return trajectory;
}

// ==================== 1. Compute Global Makespan ====================
// Find the last timestep where ANY agent takes a non-WAIT action

int compute_global_makespan_cpp(
    const std::vector<std::vector<int>>& plans
) {
    if (plans.empty()) {
        return 0;
    }

    // Find max plan length
    int max_len = 0;
    for (const auto& plan : plans) {
        max_len = std::max(max_len, static_cast<int>(plan.size()));
    }

    if (max_len == 0) {
        return 0;
    }

    // Scan backwards to find last non-WAIT action
    for (int t = max_len - 1; t >= 0; --t) {
        for (const auto& plan : plans) {
            if (t < static_cast<int>(plan.size()) && plan[t] != ACTION_WAIT) {
                return t + 1;  // Makespan is length, not index
            }
        }
    }

    return 0;  // All WAITs
}

// ==================== 2. Trim Trailing WAITs ====================
// Remove trailing WAITs from plans (standard cleanup)

CleanupResult trim_trailing_waits_global_cpp(
    const std::vector<std::vector<int>>& plans,
    const std::vector<CleanupCell>& starts,
    const std::vector<std::vector<int>>& grid
) {
    CleanupResult result;
    int num_agents = plans.size();

    // Find original max length
    int max_len = 0;
    for (const auto& plan : plans) {
        max_len = std::max(max_len, static_cast<int>(plan.size()));
    }
    result.original_makespan = max_len;

    // Compute global makespan
    int global_makespan = compute_global_makespan_cpp(plans);
    result.cleaned_makespan = global_makespan;
    result.timesteps_removed = max_len - global_makespan;

    // Trim and pad all plans to global_makespan
    result.plans.resize(num_agents);
    result.trajectories.resize(num_agents);

    for (int i = 0; i < num_agents; ++i) {
        // Copy and trim/pad
        result.plans[i].clear();
        result.plans[i].reserve(global_makespan);

        for (int t = 0; t < global_makespan; ++t) {
            if (t < static_cast<int>(plans[i].size())) {
                result.plans[i].push_back(plans[i][t]);
            } else {
                result.plans[i].push_back(ACTION_WAIT);  // Pad with WAIT
            }
        }

        // Re-simulate trajectory
        result.trajectories[i] = simulate_plan_cpp(starts[i], result.plans[i], grid);
    }

    return result;
}

// ==================== 3. Remove Synchronized WAITs ====================
// NEW: Remove timesteps where ALL agents have WAIT action
// This compresses the timeline by removing "idle" moments

CleanupResult remove_synchronized_waits_cpp(
    const std::vector<std::vector<int>>& plans,
    const std::vector<CleanupCell>& starts,
    const std::vector<std::vector<int>>& grid
) {
    CleanupResult result;
    int num_agents = plans.size();

    if (num_agents == 0) {
        result.original_makespan = 0;
        result.cleaned_makespan = 0;
        result.timesteps_removed = 0;
        return result;
    }

    // Find max plan length
    int max_len = 0;
    for (const auto& plan : plans) {
        max_len = std::max(max_len, static_cast<int>(plan.size()));
    }
    result.original_makespan = max_len;

    if (max_len == 0) {
        result.cleaned_makespan = 0;
        result.timesteps_removed = 0;
        result.plans.resize(num_agents);
        result.trajectories.resize(num_agents);
        for (int i = 0; i < num_agents; ++i) {
            result.trajectories[i].push_back(starts[i]);
        }
        return result;
    }

    // Step 1: Identify timesteps where ALL agents have WAIT (or no action)
    std::vector<bool> is_synchronized_wait(max_len, true);

    for (int t = 0; t < max_len; ++t) {
        for (int i = 0; i < num_agents; ++i) {
            int action = ACTION_WAIT;  // Default for agents with shorter plans
            if (t < static_cast<int>(plans[i].size())) {
                action = plans[i][t];
            }

            if (action != ACTION_WAIT) {
                is_synchronized_wait[t] = false;
                break;
            }
        }
    }

    // Step 2: Build compressed plans by skipping synchronized WAITs
    result.plans.resize(num_agents);

    for (int i = 0; i < num_agents; ++i) {
        result.plans[i].clear();
        result.plans[i].reserve(max_len);  // Upper bound

        for (int t = 0; t < max_len; ++t) {
            if (!is_synchronized_wait[t]) {
                // Keep this timestep
                int action = ACTION_WAIT;
                if (t < static_cast<int>(plans[i].size())) {
                    action = plans[i][t];
                }
                result.plans[i].push_back(action);
            }
            // Skip synchronized WAIT timesteps
        }
    }

    // Count removed timesteps
    int removed = 0;
    for (int t = 0; t < max_len; ++t) {
        if (is_synchronized_wait[t]) {
            ++removed;
        }
    }
    result.timesteps_removed = removed;

    // Compute new makespan
    int new_makespan = 0;
    for (const auto& plan : result.plans) {
        new_makespan = std::max(new_makespan, static_cast<int>(plan.size()));
    }
    result.cleaned_makespan = new_makespan;

    // Pad all plans to same length with WAIT
    for (int i = 0; i < num_agents; ++i) {
        while (static_cast<int>(result.plans[i].size()) < new_makespan) {
            result.plans[i].push_back(ACTION_WAIT);
        }
    }

    // Step 3: Re-simulate trajectories
    result.trajectories.resize(num_agents);
    for (int i = 0; i < num_agents; ++i) {
        result.trajectories[i] = simulate_plan_cpp(starts[i], result.plans[i], grid);
    }

    return result;
}

// ==================== 4. Full Cleanup Pipeline ====================
// Combines trailing WAIT removal AND synchronized WAIT compression

CleanupResult full_cleanup_cpp(
    const std::vector<std::vector<int>>& plans,
    const std::vector<CleanupCell>& starts,
    const std::vector<std::vector<int>>& grid,
    bool remove_synchronized  // Whether to also remove mid-plan synchronized WAITs
) {
    if (!remove_synchronized) {
        // Just do trailing WAIT removal
        return trim_trailing_waits_global_cpp(plans, starts, grid);
    }

    // Step 1: First trim trailing WAITs
    CleanupResult intermediate = trim_trailing_waits_global_cpp(plans, starts, grid);

    // Step 2: Then remove synchronized WAITs (mid-plan compression)
    CleanupResult result = remove_synchronized_waits_cpp(intermediate.plans, starts, grid);

    // Combine stats
    result.original_makespan = intermediate.original_makespan;
    result.timesteps_removed = intermediate.timesteps_removed + result.timesteps_removed;

    return result;
}

// ==================== 5. Trim Single Agent (for individual cleanup) ====================

std::vector<int> trim_trailing_waits_single_cpp(
    const std::vector<int>& plan,
    const std::vector<CleanupCell>& trajectory,
    const CleanupCell& goal
) {
    if (plan.empty() || trajectory.empty()) {
        return plan;
    }

    // Find first time we reach the goal
    for (size_t t = 0; t < trajectory.size(); ++t) {
        if (trajectory[t] == goal) {
            // Trim plan to this point
            // Plan has len(trajectory)-1 actions
            if (t < plan.size()) {
                return std::vector<int>(plan.begin(), plan.begin() + t);
            } else {
                return plan;  // Already at goal from start
            }
        }
    }

    // Never reached goal
    return plan;
}

// ==================== 6. Analyze Synchronized WAITs (diagnostic) ====================

py::dict analyze_synchronized_waits_cpp(
    const std::vector<std::vector<int>>& plans
) {
    py::dict result;
    int num_agents = plans.size();

    if (num_agents == 0) {
        result["total_timesteps"] = 0;
        result["synchronized_waits"] = 0;
        result["sync_wait_positions"] = py::list();
        return result;
    }

    int max_len = 0;
    for (const auto& plan : plans) {
        max_len = std::max(max_len, static_cast<int>(plan.size()));
    }

    py::list sync_positions;
    int sync_count = 0;

    for (int t = 0; t < max_len; ++t) {
        bool all_wait = true;

        for (int i = 0; i < num_agents; ++i) {
            int action = ACTION_WAIT;
            if (t < static_cast<int>(plans[i].size())) {
                action = plans[i][t];
            }

            if (action != ACTION_WAIT) {
                all_wait = false;
                break;
            }
        }

        if (all_wait) {
            sync_positions.append(t);
            ++sync_count;
        }
    }

    result["total_timesteps"] = max_len;
    result["synchronized_waits"] = sync_count;
    result["sync_wait_positions"] = sync_positions;
    result["potential_savings"] = sync_count;

    return result;
}

// ==================== Python Bindings ====================

PYBIND11_MODULE(cpp_cleanup, m) {
    m.doc() = "Fast C++ cleanup operations for MAPF plans";

    // Bind CleanupCell struct
    py::class_<CleanupCell>(m, "CleanupCell")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def_readwrite("r", &CleanupCell::r)
        .def_readwrite("c", &CleanupCell::c);

    // Bind CleanupResult struct
    py::class_<CleanupResult>(m, "CleanupResult")
        .def(py::init<>())
        .def_readwrite("plans", &CleanupResult::plans)
        .def_readwrite("trajectories", &CleanupResult::trajectories)
        .def_readwrite("original_makespan", &CleanupResult::original_makespan)
        .def_readwrite("cleaned_makespan", &CleanupResult::cleaned_makespan)
        .def_readwrite("timesteps_removed", &CleanupResult::timesteps_removed);

    // Function: Compute global makespan
    m.def(
        "compute_global_makespan",
        &compute_global_makespan_cpp,
        py::arg("plans"),
        R"pbdoc(
            Compute the global makespan (last timestep with non-WAIT action).

            Args:
                plans: List of action lists for each agent

            Returns:
                int: The global makespan (plan length needed)
        )pbdoc"
    );

    // Function: Trim trailing WAITs globally
    m.def(
        "trim_trailing_waits_global",
        &trim_trailing_waits_global_cpp,
        py::arg("plans"),
        py::arg("starts"),
        py::arg("grid"),
        R"pbdoc(
            Trim trailing synchronized WAITs from all plans.

            Finds the last timestep where any agent has a non-WAIT action,
            then trims/pads all plans to that length.

            Args:
                plans: List of action lists for each agent
                starts: List of CleanupCell start positions
                grid: 2D grid (0=free, -1=obstacle)

            Returns:
                CleanupResult with cleaned plans and trajectories
        )pbdoc"
    );

    // Function: Remove synchronized WAITs (compression)
    m.def(
        "remove_synchronized_waits",
        &remove_synchronized_waits_cpp,
        py::arg("plans"),
        py::arg("starts"),
        py::arg("grid"),
        R"pbdoc(
            Remove timesteps where ALL agents have WAIT action.

            This compresses the timeline by removing "synchronized idle" moments.

            Args:
                plans: List of action lists for each agent
                starts: List of CleanupCell start positions
                grid: 2D grid (0=free, -1=obstacle)

            Returns:
                CleanupResult with compressed plans and trajectories
        )pbdoc"
    );

    // Function: Full cleanup pipeline
    m.def(
        "full_cleanup",
        &full_cleanup_cpp,
        py::arg("plans"),
        py::arg("starts"),
        py::arg("grid"),
        py::arg("remove_synchronized") = true,
        R"pbdoc(
            Full cleanup pipeline: trailing WAITs + synchronized WAIT compression.

            Args:
                plans: List of action lists for each agent
                starts: List of CleanupCell start positions
                grid: 2D grid (0=free, -1=obstacle)
                remove_synchronized: If true, also remove mid-plan synchronized WAITs

            Returns:
                CleanupResult with fully cleaned plans and trajectories
        )pbdoc"
    );

    // Function: Trim single agent
    m.def(
        "trim_trailing_waits_single",
        &trim_trailing_waits_single_cpp,
        py::arg("plan"),
        py::arg("trajectory"),
        py::arg("goal"),
        R"pbdoc(
            Trim trailing WAITs from a single agent's plan.

            Args:
                plan: Action list for the agent
                trajectory: Simulated trajectory (positions)
                goal: Goal position CleanupCell

            Returns:
                Trimmed plan
        )pbdoc"
    );

    // Function: Analyze synchronized WAITs (diagnostic)
    m.def(
        "analyze_synchronized_waits",
        &analyze_synchronized_waits_cpp,
        py::arg("plans"),
        R"pbdoc(
            Analyze plans for synchronized WAIT patterns.

            Returns diagnostic info about which timesteps have all agents waiting.

            Args:
                plans: List of action lists for each agent

            Returns:
                Dict with total_timesteps, synchronized_waits, sync_wait_positions, potential_savings
        )pbdoc"
    );
}
