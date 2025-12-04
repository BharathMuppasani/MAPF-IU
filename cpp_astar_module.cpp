// cpp_astar_module.cpp
//
// C++ implementation of a simple 4-connected A* on a grid,
// exposed to Python via pybind11 as cpp_astar.astar_grid().

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

struct Node {
    int r;
    int c;
    int g;  // cost so far
    int f;  // g + w*h
};

struct NodeCompare {
    bool operator()(const Node &a, const Node &b) const {
        // priority_queue is max-heap by default, so we invert
        return a.f > b.f;
    }
};

inline int manhattan(int r1, int c1, int r2, int c2) {
    return std::abs(r1 - r2) + std::abs(c1 - c2);
}

// grid: 2D int grid. -1 = obstacle, everything else traversable.
// returns: vector<int> of action indices (0: up, 1: down, 2: left, 3: right)
std::vector<int> astar_grid(
    const std::vector<std::vector<int>> &grid,
    int start_r,
    int start_c,
    int goal_r,
    int goal_c,
    int max_expansions,
    double heuristic_weight
) {
    const int H = static_cast<int>(grid.size());
    if (H == 0) {
        return {};  // empty grid
    }
    const int W = static_cast<int>(grid[0].size());

    auto in_bounds = [&](int r, int c) {
        return (r >= 0 && r < H && c >= 0 && c < W);
    };

    auto idx = [&](int r, int c) {
        return r * W + c;
    };

    const int start_idx = idx(start_r, start_c);
    const int goal_idx  = idx(goal_r, goal_c);

    const int N = H * W;
    std::vector<int> g_score(N, std::numeric_limits<int>::max());
    std::vector<int> came_from(N, -1);
    std::vector<int> came_action(N, -1);

    std::priority_queue<Node, std::vector<Node>, NodeCompare> open;

    int h0 = manhattan(start_r, start_c, goal_r, goal_c);
    g_score[start_idx] = 0;
    open.push(Node{start_r, start_c, 0, 0 + static_cast<int>(heuristic_weight * h0)});

    // 0: up, 1: down, 2: left, 3: right
    const int dr[4] = {-1, 1,  0, 0};
    const int dc[4] = { 0, 0, -1, 1};

    int expansions = 0;

    while (!open.empty() && expansions < max_expansions) {
        Node cur = open.top();
        open.pop();

        int cur_idx = idx(cur.r, cur.c);

        // If we've already found a better path to this cell, skip
        if (cur.g > g_score[cur_idx]) {
            continue;
        }

        // Goal reached: reconstruct path
        if (cur.r == goal_r && cur.c == goal_c) {
            std::vector<int> actions;
            int i = goal_idx;
            while (i != start_idx) {
                int a = came_action[i];
                if (a < 0) break;
                actions.push_back(a);
                i = came_from[i];
            }
            std::reverse(actions.begin(), actions.end());
            return actions;
        }

        expansions++;

        for (int a = 0; a < 4; ++a) {
            int nr = cur.r + dr[a];
            int nc = cur.c + dc[a];

            if (!in_bounds(nr, nc)) {
                continue;
            }
            if (grid[nr][nc] == -1) {
                continue;  // obstacle
            }

            int n_idx = idx(nr, nc);
            int tentative_g = cur.g + 1;  // uniform step cost

            if (tentative_g < g_score[n_idx]) {
                g_score[n_idx] = tentative_g;
                came_from[n_idx] = cur_idx;
                came_action[n_idx] = a;

                int h = manhattan(nr, nc, goal_r, goal_c);
                int f = tentative_g + static_cast<int>(heuristic_weight * h);

                open.push(Node{nr, nc, tentative_g, f});
            }
        }
    }

    // No path found (or max_expansions exceeded)
    return {};
}

PYBIND11_MODULE(cpp_astar, m) {
    m.doc() = "Fast C++ A* for grid environments";

    m.def(
        "astar_grid",
        &astar_grid,
        py::arg("grid"),
        py::arg("start_r"),
        py::arg("start_c"),
        py::arg("goal_r"),
        py::arg("goal_c"),
        py::arg("max_expansions") = 500000,
        py::arg("heuristic_weight") = 1.0,
        R"pbdoc(
            Run A* on a 2D grid.

            grid: 2D list of ints (0 free, -1 obstacle at minimum)
            start_r, start_c: start cell (row, col)
            goal_r, goal_c: goal cell (row, col)
            max_expansions: limit on expanded nodes
            heuristic_weight: weight on Manhattan heuristic

            Returns: list of action indices (0=up,1=down,2=left,3=right).
            Empty list means "no path found".
        )pbdoc"
    );
}
