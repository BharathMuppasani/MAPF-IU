// cpp_collision_module.cpp
#include <algorithm>
#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

// ==================== Data Structures ====================

struct CollisionCell {
  int r;
  int c;

  CollisionCell() : r(0), c(0) {}
  CollisionCell(int row, int col) : r(row), c(col) {}

  bool operator==(const CollisionCell &other) const {
    return r == other.r && c == other.c;
  }
  bool operator!=(const CollisionCell &other) const {
    return !(*this == other);
  }
};

// Hash function for CollisionCell
namespace std {
template <> struct hash<CollisionCell> {
  size_t operator()(const CollisionCell &c) const {
    return hash<int>()(c.r) ^ (hash<int>()(c.c) << 1);
  }
};
} // namespace std

struct YieldPlan {
  bool success;
  CollisionCell parking_cell;
  std::vector<CollisionCell> path_to_parking;
  std::vector<CollisionCell> path_to_goal;
  int wait_steps;          // Number of wait steps that actually worked (0-3)
  int rejected_candidates; // Number of candidates rejected due to conflict
};

// ==================== Helper Functions ====================

// Get position at time t with stay-at-last-position semantics
CollisionCell
pos_at(int agent_idx, int t,
       const std::vector<std::vector<CollisionCell>> &trajectories,
       const std::vector<CollisionCell> &starts) {
  if (agent_idx < 0 || agent_idx >= (int)trajectories.size()) {
    // Should not happen if indices are correct
    return CollisionCell(-1, -1);
  }
  const auto &traj = trajectories[agent_idx];
  if (traj.empty()) {
    if (agent_idx < (int)starts.size()) {
      return starts[agent_idx];
    }
    return CollisionCell(-1, -1);
  }
  if (t < 0)
    return starts[agent_idx]; // Assuming start at t=0
  if (t >= (int)traj.size()) {
    return traj.back();
  }
  return traj[t];
}

// ==================== 1. Collision Analyzer ====================

py::list analyze_collisions_grid_time(
    const std::vector<std::vector<CollisionCell>> &trajectories,
    const std::vector<CollisionCell> &goals,
    const std::vector<CollisionCell> &starts) {
  py::list collisions;

  if (trajectories.empty()) {
    return collisions;
  }

  int max_time = 0;
  for (const auto &traj : trajectories) {
    if (!traj.empty()) {
      max_time = std::max(max_time, (int)traj.size());
    }
  }

  if (max_time == 0)
    return collisions;

  int num_agents = trajectories.size();

  // Vertex collisions
  for (int t = 0; t < max_time; ++t) {
    for (int i = 0; i < num_agents; ++i) {
      CollisionCell pos_i = pos_at(i, t, trajectories, starts);

      // Skip if agent i has no valid position (e.g. empty traj and no start?)
      // Logic in python fallback was: if not trajectories[i] continue (legacy
      // mode) But here we use pos_at which handles empty traj by using start.
      // We should match Python's "Legacy mode: skip agents without
      // trajectories" if starts is None. But here starts is required. So we
      // assume we check all agents. Wait, Python implementation says: "Use
      // shared pos_at() helper if agent_starts provided, else fallback to old
      // behavior" We are implementing the version where agent_starts IS
      // provided.

      for (int j = i + 1; j < num_agents; ++j) {
        CollisionCell pos_j = pos_at(j, t, trajectories, starts);

        if (pos_i == pos_j) {
          py::dict col;
          col["time"] = t;
          col["type"] = "vertex";
          col["cell"] = py::make_tuple(pos_i.r, pos_i.c);
          py::list agents;
          agents.append(i + 1);
          agents.append(j + 1);
          col["agents"] = agents;
          collisions.append(col);
        }
      }
    }
  }

  // Edge collisions
  for (int t = 0; t < max_time - 1; ++t) {
    for (int i = 0; i < num_agents; ++i) {
      CollisionCell pos_i_t = pos_at(i, t, trajectories, starts);
      CollisionCell pos_i_tp1 = pos_at(i, t + 1, trajectories, starts);

      for (int j = i + 1; j < num_agents; ++j) {
        CollisionCell pos_j_t = pos_at(j, t, trajectories, starts);
        CollisionCell pos_j_tp1 = pos_at(j, t + 1, trajectories, starts);

        if (pos_i_t == pos_j_tp1 && pos_i_tp1 == pos_j_t) {
          py::dict col;
          col["time"] = t + 1;
          col["type"] = "edge";
          col["cell"] =
              py::make_tuple(py::make_tuple(pos_i_t.r, pos_i_t.c),
                             py::make_tuple(pos_i_tp1.r, pos_i_tp1.c));
          py::list agents;
          agents.append(i + 1);
          agents.append(j + 1);
          col["agents"] = agents;
          collisions.append(col);
        }
      }
    }
  }

  return collisions;
}

// ==================== 2. Spatio-Temporal Safety Checker ====================

bool check_spatiotemporal_safety_cpp(
    int yielding_agent, // 1-indexed
    int start_time,     // absolute time when test trajectory starts
    const std::vector<CollisionCell> &test_traj, // positions per local step
    const std::vector<std::vector<CollisionCell>> &trajectories,
    const std::vector<CollisionCell> &starts) {
  int num_agents = trajectories.size();
  int yielding_idx = yielding_agent - 1;

  // std::cout << "Check Safety: Agent " << yielding_agent
  //           << " StartT=" << start_time << " TrajLen=" << test_traj.size()
  //           << std::endl;

  for (size_t time_offset = 0; time_offset < test_traj.size() - 1;
       ++time_offset) {
    int abs_time = start_time + time_offset;
    CollisionCell pos_g = test_traj[time_offset];
    CollisionCell pos_g_next = test_traj[time_offset + 1];

    // std::cout << "  T=" << abs_time << " Move " << pos_g.r << "," << pos_g.c
    //           << " -> " << pos_g_next.r << "," << pos_g_next.c << std::endl;

    for (int other_idx = 0; other_idx < num_agents; ++other_idx) {
      if (other_idx == yielding_idx)
        continue;

      CollisionCell pos_k = pos_at(other_idx, abs_time, trajectories, starts);
      CollisionCell pos_k_next =
          pos_at(other_idx, abs_time + 1, trajectories, starts);

      // std::cout << "    Vs Agent " << (other_idx + 1) << " at " << pos_k.r
      //           << "," << pos_k.c << " -> " << pos_k_next.r << ","
      //           << pos_k_next.c << std::endl;

      // Vertex conflict at t+1
      if (pos_g_next == pos_k_next) {
        // std::cout << "    FAIL: Vertex conflict with Agent " << (other_idx +
        // 1)
        //           << " at " << pos_g_next.r << "," << pos_g_next.c <<
        //           std::endl;
        return false;
      }

      // Edge conflict (swap)
      if (pos_g == pos_k_next && pos_g_next == pos_k) {
        // std::cout << "    FAIL: Edge conflict with Agent " << (other_idx + 1)
        //           << std::endl;
        return false;
      }
    }
  }
  return true;
}

// ==================== 3. Yield / Parking Helpers ====================

// BFS to find path on static grid
std::vector<CollisionCell> bfs_path(const CollisionCell &start,
                                    const CollisionCell &end,
                                    const std::vector<std::vector<int>> &grid,
                                    int H, int W) {
  if (start == end)
    return {start};

  std::queue<std::pair<CollisionCell, std::vector<CollisionCell>>> q;
  q.push({start, {start}});

  std::unordered_set<CollisionCell> visited;
  visited.insert(start);

  int dr[] = {-1, 1, 0, 0};
  int dc[] = {0, 0, -1, 1};

  while (!q.empty()) {
    auto [curr, path] = q.front();
    q.pop();

    if (curr == end)
      return path;

    for (int i = 0; i < 4; ++i) {
      CollisionCell next = {curr.r + dr[i], curr.c + dc[i]};

      if (next.r >= 0 && next.r < H && next.c >= 0 && next.c < W &&
          grid[next.r][next.c] == 0 && visited.find(next) == visited.end()) {
        visited.insert(next);
        auto new_path = path;
        new_path.push_back(next);
        q.push({next, new_path});
      }
    }
  }
  return {};
}

YieldPlan find_parking_and_paths_cpp(
    const std::vector<std::vector<int>> &grid, // -1 obstacles, 0 free
    CollisionCell anchor_cell,                 // goal or start cell
    int collision_time, int agent_id,
    const std::vector<std::vector<CollisionCell>> &trajectories,
    const std::vector<CollisionCell> &starts,
    int max_radius, // search radius around anchor
    int max_bfs_len // Not strictly used in simple BFS radius search but good
                    // for limits
) {
  YieldPlan result;
  result.success = false;
  result.wait_steps = 0; // Default to 0 waits
  result.rejected_candidates = 0;

  int H = grid.size();
  int W = (H > 0) ? grid[0].size() : 0;

  // BFS for candidates
  std::queue<std::pair<CollisionCell, int>> q;
  q.push({anchor_cell, 0});

  std::unordered_set<CollisionCell> visited;
  visited.insert(anchor_cell);

  std::vector<std::pair<CollisionCell, int>> candidates;

  int dr[] = {-1, 1, 0, 0};
  int dc[] = {0, 0, -1, 1};

  while (!q.empty()) {
    auto [curr, dist] = q.front();
    q.pop();

    if (dist > max_radius)
      continue;

    // Check if valid parking candidate (1 <= dist <= max_radius)
    if (dist >= 1 && dist <= max_radius) {
      // Must not be obstacle (already checked in BFS expansion)
      // Must not be collision cell (anchor_cell) - implicitly handled by dist
      // >= 1
      candidates.push_back({curr, dist});
    }

    for (int i = 0; i < 4; ++i) {
      CollisionCell next = {curr.r + dr[i], curr.c + dc[i]};

      if (next.r >= 0 && next.r < H && next.c >= 0 && next.c < W &&
          grid[next.r][next.c] != -1 && visited.find(next) == visited.end()) {
        visited.insert(next);
        q.push({next, dist + 1});
      }
    }
  }

  // Sort candidates by distance
  std::sort(candidates.begin(), candidates.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });

  for (const auto &[parking_cell, dist] : candidates) {
    // Build paths
    // 1. anchor -> parking
    std::vector<CollisionCell> path_to_parking =
        bfs_path(anchor_cell, parking_cell, grid, H, W);
    if (path_to_parking.empty())
      continue;

    // 2. parking -> anchor
    std::vector<CollisionCell> path_to_goal =
        bfs_path(parking_cell, anchor_cell, grid, H, W);
    if (path_to_goal.empty())
      continue;

    // Try incremental wait durations from 0 to 3 steps
    bool cell_success = false;
    for (int wait_steps = 0; wait_steps <= 3; ++wait_steps) {
      // Construct test trajectory: path_to_parking -> wait(wait_steps) ->
      // path_to_goal
      std::vector<CollisionCell> test_traj;

      // Add path to parking
      for (const auto &p : path_to_parking) {
        test_traj.push_back(p);
      }

      // Wait for wait_steps (try 0, 1, 2, 3 in order)
      for (int w = 0; w < wait_steps; ++w) {
        test_traj.push_back(parking_cell);
      }

      // Add path to goal (skip first as it is parking_cell)
      for (size_t k = 1; k < path_to_goal.size(); ++k) {
        test_traj.push_back(path_to_goal[k]);
      }

      // Check safety
      if (check_spatiotemporal_safety_cpp(agent_id, collision_time, test_traj,
                                          trajectories, starts)) {
        result.success = true;
        result.parking_cell = parking_cell;
        result.path_to_parking = path_to_parking;
        result.path_to_goal = path_to_goal;
        result.wait_steps = wait_steps; // Store the wait_steps that worked
        // result.rejected_candidates is already populated with previous
        // failures
        return result;
      }
    }

    if (!cell_success) {
      result.rejected_candidates++;
    }
  }

  return result;
}

// 4. Resolve Yield on Start
YieldPlan resolve_yield_on_start_cpp(
    const std::vector<std::vector<int>> &grid, int collision_time, int agent_id,
    CollisionCell start_pos,
    const std::vector<std::vector<CollisionCell>> &trajectories,
    const std::vector<CollisionCell> &starts, int max_radius) {
  // Logic: Find parking near start, build path (start->parking->wait->start),
  // check safety. This is essentially find_parking_and_paths_cpp with
  // anchor=start_pos
  return find_parking_and_paths_cpp(grid, start_pos, collision_time, agent_id,
                                    trajectories, starts, max_radius, 100);
}

// 5. Resolve Yield on Goal
YieldPlan resolve_yield_on_goal_cpp(
    const std::vector<std::vector<int>> &grid, int collision_time, int agent_id,
    CollisionCell goal_pos,
    const std::vector<std::vector<CollisionCell>> &trajectories,
    const std::vector<CollisionCell> &starts, int max_radius) {
  // Logic: Find parking near goal, build path (goal->parking->wait->goal),
  // check safety. This is essentially find_parking_and_paths_cpp with
  // anchor=goal_pos
  return find_parking_and_paths_cpp(grid, goal_pos, collision_time, agent_id,
                                    trajectories, starts, max_radius, 100);
}

// 6. Resolve Wait Yield
YieldPlan resolve_wait_yield_cpp(
    int collision_time, int agent_id,
    const std::vector<CollisionCell> &current_traj,
    const std::vector<std::vector<CollisionCell>> &trajectories,
    const std::vector<CollisionCell> &starts) {
  YieldPlan result;
  result.success = false;

  // Try inserting 1, 2, 3 wait steps
  for (int wait_steps = 1; wait_steps <= 3; ++wait_steps) {
    int t_insert = collision_time - wait_steps;
    if (t_insert < 0)
      continue;

    // Construct new trajectory with wait
    std::vector<CollisionCell> test_traj;

    // Copy up to t_insert
    for (int t = 0; t < t_insert && t < (int)current_traj.size(); ++t) {
      test_traj.push_back(current_traj[t]);
    }

    // Insert waits (repeat position at t_insert-1)
    if (!test_traj.empty()) {
      CollisionCell wait_pos = test_traj.back();
      for (int w = 0; w < wait_steps; ++w) {
        test_traj.push_back(wait_pos);
      }
    } else {
      // If inserting at 0, use start pos
      int idx = agent_id - 1;
      if (idx >= 0 && idx < (int)starts.size()) {
        CollisionCell start = starts[idx];
        for (int w = 0; w < wait_steps; ++w) {
          test_traj.push_back(start);
        }
      }
    }

    // Copy remaining trajectory
    for (int t = t_insert; t < (int)current_traj.size(); ++t) {
      test_traj.push_back(current_traj[t]);
    }

    // Check safety
    // We need to check safety for the modified part and onwards?
    // check_spatiotemporal_safety_cpp takes start_time. If we pass
    // start_time=0, it checks the whole thing.
    if (check_spatiotemporal_safety_cpp(agent_id, 0, test_traj, trajectories,
                                        starts)) {
      result.success = true;
      // For wait yield, we return the FULL modified trajectory as path_to_goal
      // (hacky reuse of struct)
      result.path_to_goal = test_traj;
      return result;
    }
  }

  return result;
}

// ==================== Pybind11 Module Definition ====================

// ==================== Pybind11 Module Definition ====================

PYBIND11_MODULE(cpp_collision, m) {
  m.doc() = "C++ collision analysis and yield helpers";

  py::class_<CollisionCell>(m, "CollisionCell")
      .def(py::init<int, int>())
      .def_readwrite("r", &CollisionCell::r)
      .def_readwrite("c", &CollisionCell::c)
      .def("__repr__", [](const CollisionCell &c) {
        return "(" + std::to_string(c.r) + ", " + std::to_string(c.c) + ")";
      });

  py::class_<YieldPlan>(m, "YieldPlan")
      .def(py::init<>())
      .def_readwrite("success", &YieldPlan::success)
      .def_readwrite("parking_cell", &YieldPlan::parking_cell)
      .def_readwrite("path_to_parking", &YieldPlan::path_to_parking)
      .def_readwrite("path_to_goal", &YieldPlan::path_to_goal)
      .def_readwrite("wait_steps", &YieldPlan::wait_steps)
      .def_readwrite("rejected_candidates", &YieldPlan::rejected_candidates);

  m.def(
      "analyze_collisions_grid_time",
      [](const std::vector<std::vector<std::pair<int, int>>> &trajectories_py,
         const std::vector<std::pair<int, int>> &goals_py,
         const std::vector<std::pair<int, int>> &starts_py) {
        // Convert inputs
        std::vector<std::vector<CollisionCell>> trajectories;
        for (const auto &t : trajectories_py) {
          std::vector<CollisionCell> traj;
          for (const auto &p : t)
            traj.push_back({p.first, p.second});
          trajectories.push_back(traj);
        }

        std::vector<CollisionCell> goals;
        for (const auto &p : goals_py)
          goals.push_back({p.first, p.second});

        std::vector<CollisionCell> starts;
        for (const auto &p : starts_py)
          starts.push_back({p.first, p.second});

        return analyze_collisions_grid_time(trajectories, goals, starts);
      },
      "Analyze collisions in grid-time space");

  m.def(
      "check_spatiotemporal_safety_cpp",
      [](int yielding_agent, int start_time,
         const std::vector<std::pair<int, int>> &test_traj_py,
         const std::vector<std::vector<std::pair<int, int>>> &trajectories_py,
         const std::vector<std::pair<int, int>> &starts_py) {
        std::vector<CollisionCell> test_traj;
        for (const auto &p : test_traj_py)
          test_traj.push_back({p.first, p.second});

        std::vector<std::vector<CollisionCell>> trajectories;
        for (const auto &t : trajectories_py) {
          std::vector<CollisionCell> traj;
          for (const auto &p : t)
            traj.push_back({p.first, p.second});
          trajectories.push_back(traj);
        }

        std::vector<CollisionCell> starts;
        for (const auto &p : starts_py)
          starts.push_back({p.first, p.second});

        return check_spatiotemporal_safety_cpp(yielding_agent, start_time,
                                               test_traj, trajectories, starts);
      },
      "Check spatio-temporal safety");

  m.def(
      "find_parking_and_paths_cpp",
      [](const std::vector<std::vector<int>> &grid,
         std::pair<int, int> anchor_cell_py, int collision_time, int agent_id,
         const std::vector<std::vector<std::pair<int, int>>> &trajectories_py,
         const std::vector<std::pair<int, int>> &starts_py, int max_radius,
         int max_bfs_len) {
        CollisionCell anchor_cell = {anchor_cell_py.first,
                                     anchor_cell_py.second};

        std::vector<std::vector<CollisionCell>> trajectories;
        for (const auto &t : trajectories_py) {
          std::vector<CollisionCell> traj;
          for (const auto &p : t)
            traj.push_back({p.first, p.second});
          trajectories.push_back(traj);
        }

        std::vector<CollisionCell> starts;
        for (const auto &p : starts_py)
          starts.push_back({p.first, p.second});

        return find_parking_and_paths_cpp(grid, anchor_cell, collision_time,
                                          agent_id, trajectories, starts,
                                          max_radius, max_bfs_len);
      },
      "Find parking cell and paths");

  m.def(
      "resolve_yield_on_start_cpp",
      [](const std::vector<std::vector<int>> &grid, int collision_time,
         int agent_id, std::pair<int, int> start_pos_py,
         const std::vector<std::vector<std::pair<int, int>>> &trajectories_py,
         const std::vector<std::pair<int, int>> &starts_py, int max_radius) {
        CollisionCell start_pos = {start_pos_py.first, start_pos_py.second};

        std::vector<std::vector<CollisionCell>> trajectories;
        for (const auto &t : trajectories_py) {
          std::vector<CollisionCell> traj;
          for (const auto &p : t)
            traj.push_back({p.first, p.second});
          trajectories.push_back(traj);
        }

        std::vector<CollisionCell> starts;
        for (const auto &p : starts_py)
          starts.push_back({p.first, p.second});

        return resolve_yield_on_start_cpp(grid, collision_time, agent_id,
                                          start_pos, trajectories, starts,
                                          max_radius);
      },
      "Resolve Yield on Start");

  m.def(
      "resolve_yield_on_goal_cpp",
      [](const std::vector<std::vector<int>> &grid, int collision_time,
         int agent_id, std::pair<int, int> goal_pos_py,
         const std::vector<std::vector<std::pair<int, int>>> &trajectories_py,
         const std::vector<std::pair<int, int>> &starts_py, int max_radius) {
        CollisionCell goal_pos = {goal_pos_py.first, goal_pos_py.second};

        std::vector<std::vector<CollisionCell>> trajectories;
        for (const auto &t : trajectories_py) {
          std::vector<CollisionCell> traj;
          for (const auto &p : t)
            traj.push_back({p.first, p.second});
          trajectories.push_back(traj);
        }

        std::vector<CollisionCell> starts;
        for (const auto &p : starts_py)
          starts.push_back({p.first, p.second});

        return resolve_yield_on_goal_cpp(grid, collision_time, agent_id,
                                         goal_pos, trajectories, starts,
                                         max_radius);
      },
      "Resolve Yield on Goal");

  m.def(
      "resolve_wait_yield_cpp",
      [](int collision_time, int agent_id,
         const std::vector<std::pair<int, int>> &current_traj_py,
         const std::vector<std::vector<std::pair<int, int>>> &trajectories_py,
         const std::vector<std::pair<int, int>> &starts_py) {
        std::vector<CollisionCell> current_traj;
        for (const auto &p : current_traj_py)
          current_traj.push_back({p.first, p.second});

        std::vector<std::vector<CollisionCell>> trajectories;
        for (const auto &t : trajectories_py) {
          std::vector<CollisionCell> traj;
          for (const auto &p : t)
            traj.push_back({p.first, p.second});
          trajectories.push_back(traj);
        }

        std::vector<CollisionCell> starts;
        for (const auto &p : starts_py)
          starts.push_back({p.first, p.second});

        return resolve_wait_yield_cpp(collision_time, agent_id, current_traj,
                                      trajectories, starts);
      },
      "Resolve Wait Yield");
}
