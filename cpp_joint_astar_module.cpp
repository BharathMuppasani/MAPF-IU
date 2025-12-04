// cpp_joint_astar_module.cpp
//
// C++ implementation of joint A* for multi-agent pathfinding with time-based constraints,
// exposed to Python via pybind11 as cpp_joint_astar.joint_astar_grid_time().
//
// Action encoding (MUST match Python):
// 0: up    (-1,  0)
// 1: down  ( 1,  0)
// 2: left  ( 0, -1)
// 3: right ( 0,  1)
// 4: wait  ( 0,  0)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>

namespace py = pybind11;

// ==================== Data Structures ====================

struct Cell {
    int r;
    int c;

    Cell() : r(0), c(0) {}
    Cell(int row, int col) : r(row), c(col) {}

    bool operator==(const Cell& other) const {
        return r == other.r && c == other.c;
    }

    bool operator!=(const Cell& other) const {
        return !(*this == other);
    }
};

// Hash function for Cell
namespace std {
    template <>
    struct hash<Cell> {
        size_t operator()(const Cell& c) const {
            return hash<int>()(c.r) ^ (hash<int>()(c.c) << 1);
        }
    };
}

// Represents a joint state for multiple agents
struct JointState {
    std::vector<Cell> positions;  // One position per agent
    int g;                         // Cost so far (steps)
    int f;                         // g + heuristic

    JointState() : g(0), f(0) {}
};

// Comparator for priority queue (min-heap by f value)
struct JointStateCompare {
    bool operator()(const JointState& a, const JointState& b) const {
        return a.f > b.f;  // Invert for min-heap
    }
};

// Hash function for joint state (positions + depth)
struct JointStateHash {
    size_t operator()(const std::pair<std::vector<Cell>, int>& key) const {
        size_t h = std::hash<int>()(key.second);  // Include depth/g
        for (const auto& cell : key.first) {
            h ^= std::hash<Cell>()(cell) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

// Result structure
struct JointAStarResult {
    bool success;
    // Per-agent action sequences
    std::vector<std::vector<int>> plans;
    // Per-agent trajectories (sequence of cells)
    std::vector<std::vector<Cell>> trajectories;
    // Number of expanded nodes
    int expansions;
};

// ==================== Helper Functions ====================

// Actions: 0=up, 1=down, 2=left, 3=right, 4=wait
static const int NUM_ACTIONS = 5;
static const int dr[NUM_ACTIONS] = {-1, 1,  0, 0, 0};
static const int dc[NUM_ACTIONS] = { 0, 0, -1, 1, 0};

inline int manhattan(const Cell& a, const Cell& b) {
    return std::abs(a.r - b.r) + std::abs(a.c - b.c);
}

// ==================== Main Joint A* Function ====================

JointAStarResult joint_astar_grid_time(
    int H, int W,
    const std::vector<std::vector<int>>& static_grid,
    const std::vector<Cell>& start_positions,
    const std::vector<Cell>& subgoal_positions,
    int t_start,
    int t_goal_sub,
    int max_expansions,
    double time_budget_seconds,
    const std::vector<std::unordered_set<Cell>>& reserved_cells_by_time,
    const std::vector<std::vector<std::pair<Cell, Cell>>>& reserved_moves_by_time,
    const std::unordered_set<Cell>& blocked_cells,
    const std::vector<std::unordered_set<Cell>>& blocked_by_time,
    bool use_time_based_blocking
) {
    JointAStarResult result;
    result.success = false;
    result.expansions = 0;

    const int num_agents = start_positions.size();
    if (num_agents == 0 || num_agents != (int)subgoal_positions.size()) {
        return result;
    }

    auto start_clock = std::chrono::steady_clock::now();

    // Lambda: Check if cell is in bounds and not a wall
    auto in_bounds_free = [&](const Cell& c) {
        return c.r >= 0 && c.r < H && c.c >= 0 && c.c < W && static_grid[c.r][c.c] == 0;
    };

    // Lambda: Check if cell is blocked at time t
    auto is_blocked = [&](const Cell& c, int t) {
        if (use_time_based_blocking) {
            // Time-based blocking
            if (t >= 0 && t < (int)blocked_by_time.size()) {
                return blocked_by_time[t].count(c) > 0;
            }
            return false;
        } else {
            // Spatial blocking
            return blocked_cells.count(c) > 0;
        }
    };

    // Lambda: Compute heuristic for joint state
    auto heuristic = [&](const std::vector<Cell>& positions) {
        int h = 0;
        for (int i = 0; i < num_agents; ++i) {
            h += manhattan(positions[i], subgoal_positions[i]);
        }
        return h;
    };

    // Lambda: Check if all agents reached their subgoals
    auto all_at_subgoals = [&](const std::vector<Cell>& positions) {
        for (int i = 0; i < num_agents; ++i) {
            if (positions[i] != subgoal_positions[i]) {
                return false;
            }
        }
        return true;
    };

    // Lambda: Get reserved positions at time t
    auto get_reserved_at = [&](int t) -> const std::unordered_set<Cell>& {
        static const std::unordered_set<Cell> empty_set;
        int idx = t - t_start - 1;
        if (idx >= 0 && idx < (int)reserved_cells_by_time.size()) {
            return reserved_cells_by_time[idx];
        }
        return empty_set;
    };

    // Lambda: Get reserved moves at time t
    auto get_reserved_moves_at = [&](int t) -> const std::vector<std::pair<Cell, Cell>>& {
        static const std::vector<std::pair<Cell, Cell>> empty_vec;
        int idx = t - t_start;
        if (idx >= 0 && idx < (int)reserved_moves_by_time.size()) {
            return reserved_moves_by_time[idx];
        }
        return empty_vec;
    };

    // Compute min depth and max depth
    int min_depth_needed = std::max(1, t_goal_sub - t_start);
    int max_depth = std::max(min_depth_needed + 3, min_depth_needed);

    // Priority queue and visited set
    std::priority_queue<JointState, std::vector<JointState>, JointStateCompare> frontier;
    std::unordered_set<std::pair<std::vector<Cell>, int>, JointStateHash> visited;

    // Initialize with start state
    JointState initial;
    initial.positions = start_positions;
    initial.g = 0;
    initial.f = heuristic(start_positions);
    frontier.push(initial);

    int expansions = 0;

    // Store parent information for path reconstruction
    std::unordered_map<std::pair<std::vector<Cell>, int>,
                       std::pair<std::vector<Cell>, std::vector<int>>,
                       JointStateHash> parent_map;

    while (!frontier.empty() && expansions < max_expansions) {
        // Check time budget
        auto elapsed = std::chrono::steady_clock::now() - start_clock;
        double elapsed_sec = std::chrono::duration<double>(elapsed).count();
        if (elapsed_sec > time_budget_seconds) {
            break;
        }

        JointState current = frontier.top();
        frontier.pop();
        expansions++;

        // Check if all agents reached their subgoals
        if (all_at_subgoals(current.positions)) {
            int expected_len = std::max(0, t_goal_sub - t_start);

            // If too long, continue searching
            if (current.g > expected_len) {
                continue;
            }

            // Reconstruct path
            std::vector<std::vector<int>> action_sequence;
            auto key = std::make_pair(current.positions, current.g);

            // Backtrack to get action sequence
            while (parent_map.count(key)) {
                auto parent_info = parent_map[key];
                action_sequence.push_back(parent_info.second);
                int prev_g = key.second - 1;
                key = std::make_pair(parent_info.first, prev_g);
            }
            std::reverse(action_sequence.begin(), action_sequence.end());

            // Pad with WAIT actions if needed
            if (current.g < expected_len) {
                int pad_steps = expected_len - current.g;
                std::vector<int> stay_actions(num_agents, 4);  // 4 = WAIT
                for (int i = 0; i < pad_steps; ++i) {
                    action_sequence.push_back(stay_actions);
                }
            }

            // Convert action sequence to per-agent plans and trajectories
            result.plans.resize(num_agents);
            result.trajectories.resize(num_agents);

            for (int i = 0; i < num_agents; ++i) {
                result.trajectories[i].push_back(start_positions[i]);
            }

            for (const auto& joint_action : action_sequence) {
                for (int i = 0; i < num_agents; ++i) {
                    int act = joint_action[i];
                    result.plans[i].push_back(act);

                    Cell prev = result.trajectories[i].back();
                    Cell next(prev.r + dr[act], prev.c + dc[act]);
                    result.trajectories[i].push_back(next);
                }
            }

            result.success = true;
            result.expansions = expansions;
            return result;
        }

        // Check depth limit
        if (current.g >= max_depth) {
            continue;
        }

        // Check if already visited
        auto state_key = std::make_pair(current.positions, current.g);
        if (visited.count(state_key)) {
            continue;
        }
        visited.insert(state_key);

        int time_step = t_start + current.g;

        // Check if any current position conflicts with reserved positions
        const auto& reserved_now = get_reserved_at(time_step);
        bool conflict_now = false;
        for (const auto& pos : current.positions) {
            if (reserved_now.count(pos)) {
                conflict_now = true;
                break;
            }
        }
        if (conflict_now) {
            continue;
        }

        // Generate successors: for each agent, enumerate valid actions
        std::vector<std::vector<std::pair<int, Cell>>> move_options(num_agents);

        for (int i = 0; i < num_agents; ++i) {
            const Cell& pos = current.positions[i];
            for (int act = 0; act < NUM_ACTIONS; ++act) {
                Cell next_cell(pos.r + dr[act], pos.c + dc[act]);

                // Check bounds and static obstacles
                if (!in_bounds_free(next_cell)) {
                    continue;
                }

                // Check blocking
                int t_next = time_step + 1;
                if (is_blocked(next_cell, t_next)) {
                    continue;
                }

                move_options[i].push_back({act, next_cell});
            }
        }

        // Generate all combinations using recursive enumeration (avoiding itertools.product)
        std::function<void(int, std::vector<int>&, std::vector<Cell>&)> enumerate_combos;
        enumerate_combos = [&](int agent_idx, std::vector<int>& actions, std::vector<Cell>& next_pos) {
            if (agent_idx == num_agents) {
                // Have a complete joint action, validate and add to frontier

                // Check internal vertex conflicts
                std::unordered_set<Cell> pos_set(next_pos.begin(), next_pos.end());
                if ((int)pos_set.size() < num_agents) {
                    return;  // Vertex conflict
                }

                // Check internal edge conflicts (swaps)
                bool edge_conflict = false;
                for (int i = 0; i < num_agents && !edge_conflict; ++i) {
                    for (int j = i + 1; j < num_agents; ++j) {
                        if (current.positions[i] == next_pos[j] &&
                            current.positions[j] == next_pos[i]) {
                            edge_conflict = true;
                            break;
                        }
                    }
                }
                if (edge_conflict) {
                    return;
                }

                // Check conflicts with reserved positions at next time
                const auto& reserved_next = get_reserved_at(time_step + 1);
                for (const auto& p : next_pos) {
                    if (reserved_next.count(p)) {
                        return;  // Conflict with reservation
                    }
                }

                // Check conflicts with reserved moves (edge conflicts with other agents)
                const auto& reserved_moves = get_reserved_moves_at(time_step);
                for (int i = 0; i < num_agents; ++i) {
                    const Cell& prev = current.positions[i];
                    const Cell& nxt = next_pos[i];
                    for (const auto& move : reserved_moves) {
                        if (nxt == move.first && prev == move.second) {
                            return;  // Swap conflict with reserved agent
                        }
                    }
                }

                // Valid successor, add to frontier
                JointState successor;
                successor.positions = next_pos;
                successor.g = current.g + 1;
                successor.f = successor.g + heuristic(successor.positions);

                // Store parent for path reconstruction
                auto succ_key = std::make_pair(successor.positions, successor.g);
                parent_map[succ_key] = {current.positions, actions};

                frontier.push(successor);
                return;
            }

            // Enumerate options for current agent
            for (const auto& opt : move_options[agent_idx]) {
                actions.push_back(opt.first);
                next_pos.push_back(opt.second);
                enumerate_combos(agent_idx + 1, actions, next_pos);
                actions.pop_back();
                next_pos.pop_back();
            }
        };

        std::vector<int> actions;
        std::vector<Cell> next_positions;
        enumerate_combos(0, actions, next_positions);
    }

    // Search failed
    result.expansions = expansions;
    return result;
}

// ==================== pybind11 Bindings ====================

PYBIND11_MODULE(cpp_joint_astar, m) {
    m.doc() = "C++ joint A* for multi-agent pathfinding with time-based constraints";

    py::class_<Cell>(m, "Cell")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def_readwrite("r", &Cell::r)
        .def_readwrite("c", &Cell::c)
        .def("__repr__", [](const Cell& c) {
            return "Cell(" + std::to_string(c.r) + ", " + std::to_string(c.c) + ")";
        });

    py::class_<JointAStarResult>(m, "JointAStarResult")
        .def(py::init<>())
        .def_readwrite("success", &JointAStarResult::success)
        .def_readwrite("plans", &JointAStarResult::plans)
        .def_readwrite("trajectories", &JointAStarResult::trajectories)
        .def_readwrite("expansions", &JointAStarResult::expansions);

    m.def(
        "joint_astar_grid_time",
        [](int H, int W,
           const std::vector<std::vector<int>>& static_grid,
           const py::list& start_positions_py,
           const py::list& subgoal_positions_py,
           int t_start,
           int t_goal_sub,
           int max_expansions,
           double time_budget_seconds,
           const py::list& reserved_cells_by_time_py,
           const py::list& reserved_moves_by_time_py,
           const py::list& blocked_cells_py,
           const py::list& blocked_by_time_py,
           bool use_time_based_blocking) -> JointAStarResult {

            // Convert start positions
            std::vector<Cell> start_positions;
            for (auto item : start_positions_py) {
                auto dict = item.cast<py::dict>();
                Cell c(dict["r"].cast<int>(), dict["c"].cast<int>());
                start_positions.push_back(c);
            }

            // Convert subgoal positions
            std::vector<Cell> subgoal_positions;
            for (auto item : subgoal_positions_py) {
                auto dict = item.cast<py::dict>();
                Cell c(dict["r"].cast<int>(), dict["c"].cast<int>());
                subgoal_positions.push_back(c);
            }

            // Convert reserved_cells_by_time
            std::vector<std::unordered_set<Cell>> reserved_cells_by_time;
            for (auto time_list : reserved_cells_by_time_py) {
                std::unordered_set<Cell> cells_at_t;
                for (auto cell : time_list.cast<py::list>()) {
                    auto dict = cell.cast<py::dict>();
                    cells_at_t.insert(Cell(dict["r"].cast<int>(), dict["c"].cast<int>()));
                }
                reserved_cells_by_time.push_back(cells_at_t);
            }

            // Convert reserved_moves_by_time
            std::vector<std::vector<std::pair<Cell, Cell>>> reserved_moves_by_time;
            for (auto time_list : reserved_moves_by_time_py) {
                std::vector<std::pair<Cell, Cell>> moves_at_t;
                for (auto move : time_list.cast<py::list>()) {
                    auto move_pair = move.cast<py::list>();
                    auto from_dict = move_pair[0].cast<py::dict>();
                    auto to_dict = move_pair[1].cast<py::dict>();
                    Cell from(from_dict["r"].cast<int>(), from_dict["c"].cast<int>());
                    Cell to(to_dict["r"].cast<int>(), to_dict["c"].cast<int>());
                    moves_at_t.push_back({from, to});
                }
                reserved_moves_by_time.push_back(moves_at_t);
            }

            // Convert blocked_cells
            std::unordered_set<Cell> blocked_cells;
            for (auto cell : blocked_cells_py) {
                auto dict = cell.cast<py::dict>();
                blocked_cells.insert(Cell(dict["r"].cast<int>(), dict["c"].cast<int>()));
            }

            // Convert blocked_by_time
            std::vector<std::unordered_set<Cell>> blocked_by_time;
            for (auto time_list : blocked_by_time_py) {
                std::unordered_set<Cell> cells_at_t;
                for (auto cell : time_list.cast<py::list>()) {
                    auto dict = cell.cast<py::dict>();
                    cells_at_t.insert(Cell(dict["r"].cast<int>(), dict["c"].cast<int>()));
                }
                blocked_by_time.push_back(cells_at_t);
            }

            return joint_astar_grid_time(
                H, W, static_grid,
                start_positions, subgoal_positions,
                t_start, t_goal_sub,
                max_expansions, time_budget_seconds,
                reserved_cells_by_time, reserved_moves_by_time,
                blocked_cells, blocked_by_time,
                use_time_based_blocking
            );
        },
        py::arg("H"),
        py::arg("W"),
        py::arg("static_grid"),
        py::arg("start_positions"),
        py::arg("subgoal_positions"),
        py::arg("t_start"),
        py::arg("t_goal_sub"),
        py::arg("max_expansions"),
        py::arg("time_budget_seconds"),
        py::arg("reserved_cells_by_time"),
        py::arg("reserved_moves_by_time"),
        py::arg("blocked_cells"),
        py::arg("blocked_by_time"),
        py::arg("use_time_based_blocking") = true,
        R"pbdoc(
            Run joint A* for multi-agent pathfinding with time-based constraints.

            Args:
                H, W: Grid dimensions
                static_grid: 2D grid (0=free, -1=obstacle)
                start_positions: List of {'r': row, 'c': col} dicts
                subgoal_positions: List of {'r': row, 'c': col} dicts
                t_start: Start time
                t_goal_sub: Goal time
                max_expansions: Maximum node expansions
                time_budget_seconds: Time limit in seconds
                reserved_cells_by_time: List of lists of reserved cells per time
                reserved_moves_by_time: List of lists of reserved moves per time
                blocked_cells: List of permanently blocked cells
                blocked_by_time: List of lists of blocked cells per time
                use_time_based_blocking: Use time-based blocking (default: True)

            Returns:
                JointAStarResult with success flag, plans, trajectories, and expansions
        )pbdoc"
    );
}
