#include "betweenness.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>

const std::vector<double>& betweenness(const Graph& g, double θ, BCMWorkspace& ws) {
    int nv = static_cast<int>(boost::num_vertices(g));
    int ne = static_cast<int>(boost::num_edges(g));

    ws.degree.assign(nv, 0);
    auto [ei, ei_end] = boost::edges(g);
    for (auto it = ei; it != ei_end; ++it) {
        int u = static_cast<int>(boost::source(*it, g));
        int v = static_cast<int>(boost::target(*it, g));
        ++ws.degree[u];
        ++ws.degree[v];
    }

    ws.offset.assign(nv + 1, 0);
    for (int i = 0; i < nv; ++i) {
        ws.offset[i + 1] = ws.offset[i] + ws.degree[i];
    }

    int m2 = ws.offset[nv];
    ws.neighbor.resize(m2);
    ws.weight.resize(m2);
    ws.pos.assign(ws.offset.begin(), ws.offset.end() - 1);

    for (auto it = ei; it != ei_end; ++it) {
        int u = static_cast<int>(boost::source(*it, g));
        int v = static_cast<int>(boost::target(*it, g));
        double prod = static_cast<double>(ws.degree[u]) * ws.degree[v];
        double w = std::pow(std::max(1e-15, prod), θ);

        int pu = ws.pos[u]++;
        ws.neighbor[pu] = v;
        ws.weight[pu] = w;

        int pv = ws.pos[v]++;
        ws.neighbor[pv] = u;
        ws.weight[pv] = w;
    }

    const double inf = std::numeric_limits<double>::infinity();
    ws.dist.assign(nv, inf);
    ws.sigma.assign(nv, 0.0);
    ws.delta.assign(nv, 0.0);
    ws.bc.assign(nv, 0.0);
    ws.pred_head.assign(nv, -1);
    ws.pred_vertex.resize(m2 > 0 ? m2 : 1);
    ws.pred_next.resize(m2 > 0 ? m2 : 1);
    ws.order.reserve(nv);
    ws.heap.reserve(ne > 0 ? 2 * ne : 1);

    for (int s = 0; s < nv; ++s) {
        ws.dist[s] = 0.0;
        ws.sigma[s] = 1.0;
        ws.heap.clear();
        ws.order.clear();
        int pred_count = 0;

        ws.heap.emplace_back(0.0, s);
        std::push_heap(ws.heap.begin(), ws.heap.end(), std::greater<>());

        while (!ws.heap.empty()) {
            std::pop_heap(ws.heap.begin(), ws.heap.end(), std::greater<>());
            auto [d, u] = ws.heap.back();
            ws.heap.pop_back();

            if (d > ws.dist[u]) {
                continue;
            }
            ws.order.push_back(u);

            for (int k = ws.offset[u]; k < ws.offset[u + 1]; ++k) {
                int v = ws.neighbor[k];
                double nd = d + ws.weight[k];
                if (nd < ws.dist[v]) {
                    ws.dist[v] = nd;
                    ws.sigma[v] = ws.sigma[u];
                    ws.pred_head[v] = pred_count;
                    ws.pred_vertex[pred_count] = u;
                    ws.pred_next[pred_count] = -1;
                    ++pred_count;
                    ws.heap.emplace_back(nd, v);
                    std::push_heap(ws.heap.begin(), ws.heap.end(), std::greater<>());
                } else if (nd == ws.dist[v]) {
                    ws.sigma[v] += ws.sigma[u];
                    ws.pred_next[pred_count] = ws.pred_head[v];
                    ws.pred_head[v] = pred_count;
                    ws.pred_vertex[pred_count] = u;
                    ++pred_count;
                }
            }
        }

        for (int idx = static_cast<int>(ws.order.size()) - 1; idx >= 0; --idx) {
            int u = ws.order[idx];
            for (int e = ws.pred_head[u]; e != -1; e = ws.pred_next[e]) {
                int p = ws.pred_vertex[e];
                ws.delta[p] += (ws.sigma[p] / ws.sigma[u]) * (1.0 + ws.delta[u]);
            }
            if (u != s) {
                ws.bc[u] += ws.delta[u];
            }
        }

        for (int u : ws.order) {
            ws.dist[u] = inf;
            ws.sigma[u] = 0.0;
            ws.delta[u] = 0.0;
            ws.pred_head[u] = -1;
        }
    }

    return ws.bc;
}
