#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory_resource>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <set>
#include <vector>

// Settings ----------------------------------------------------------------------
constexpr char CSV_DELIMITER = ';';
constexpr const char *RESULTS_DIR = "5_deltas/results";

// Timing ------------------------------------------------------------------------
class Stopwatch {
public:
    Stopwatch() : start_(std::chrono::steady_clock::now()) {}

    void restart() { start_ = std::chrono::steady_clock::now(); }

    double elapsedSeconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

// Data types --------------------------------------------------------------------
class Point {
public:
    Point(int x, int y, int cost) : x_(x), y_(y), cost_(cost) {}

    int x() const { return x_; }
    int y() const { return y_; }
    int cost() const { return cost_; }

private:
    int x_, y_, cost_;
};

using Tour = std::vector<int>;  // indices of the selected points, in visiting order
using DistanceMatrix = std::vector<std::vector<int>>;  // Euclidean distances, rounded

// Input -------------------------------------------------------------------------
std::vector<Point> readPoints(const std::string &path) {
    std::vector<Point> points;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Could not open " << path << "\n";
        return points;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream fields(line);
        std::string x, y, cost;
        std::getline(fields, x, CSV_DELIMITER);
        std::getline(fields, y, CSV_DELIMITER);
        std::getline(fields, cost, CSV_DELIMITER);
        try {
            points.emplace_back(std::stoi(x), std::stoi(y), std::stoi(cost));
        } catch (const std::exception &) {
            std::cerr << "Skipping malformed line: " << line << "\n";
        }
    }
    return points;
}

// Problem model -----------------------------------------------------------------
// Euclidean distance, rounded mathematically (round-half-away-from-zero) to an integer,
// as required by the problem description.
int euclideanDistanceRounded(const Point &a, const Point &b) {
    const double dx = a.x() - b.x();
    const double dy = a.y() - b.y();
    return static_cast<int>(std::lround(std::sqrt(dx * dx + dy * dy)));
}

DistanceMatrix buildDistanceMatrix(const std::vector<Point> &points) {
    const size_t n = points.size();
    DistanceMatrix matrix(n, std::vector<int>(n, 0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            matrix[i][j] = matrix[j][i] = euclideanDistanceRounded(points[i], points[j]);
        }
    }
    return matrix;
}

std::vector<int> extractCosts(const std::vector<Point> &points) {
    std::vector<int> costs;
    costs.reserve(points.size());
    for (const Point &p : points) costs.push_back(p.cost());
    return costs;
}

// Number of points a tour must contain: half of all points, rounded up.
size_t tourSize(size_t numPoints) { return (numPoints + 1) / 2; }

// Objective: length of the closed tour plus the cost of every selected point.
// From this point on, algorithms only ever touch `distances` and `costs`, never
// point coordinates directly, so the same code works for instances given only as a
// distance matrix.
long long evaluateTour(const Tour &tour, const DistanceMatrix &distances, const std::vector<int> &costs) {
    if (tour.empty()) return 0;

    long long total = 0;
    for (size_t i = 0; i < tour.size(); ++i) {
        const int from = tour[i];
        const int to = tour[(i + 1) % tour.size()];  // last point wraps back to the first
        total += distances[from][to] + costs[from];
    }
    return total;
}

// Random starting solutions --------------------------------------------------------
// Random solution: pick `size` distinct random nodes, in random order.
Tour randomSolution(size_t numPoints, std::mt19937 &rng) {
    const size_t size = tourSize(numPoints);
    std::vector<int> candidates(numPoints);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::shuffle(candidates.begin(), candidates.end(), rng);
    return Tour(candidates.begin(), candidates.begin() + size);
}

// Baseline: local search from exercise 3 --------------------------------------------------------------------------
// The neighborhood of a solution is the union of
//   * inter-route moves: replace a selected node (at tour position i) with an
//     unselected node (at position u of the `unselected` list),
//   * intra-route moves on tour positions i < j, either
//       - two-nodes exchange: swap tour[i] and tour[j], or
//       - two-edges exchange (2-opt): remove edges (tour[i], tour[i+1]) and
//         (tour[j], tour[j+1]), reconnect by reversing tour[i+1..j].
// Every move is identified by positions only, so the list of all moves never changes
// while the search runs. Moves are evaluated by the delta of the objective function
// computed from the few edges/costs they touch; the neighbor is never built explicitly.
enum class IntraMove { NodesExchange, EdgesExchange };
enum class SearchType { Steepest, Greedy };

struct Move {
    enum Kind : unsigned char { Inter, Intra } kind;
    int a;  // tour position i
    int b;  // Inter: position in `unselected`; Intra: tour position j (> i)
};

struct SearchCounters {
    long long evaluated = 0;  // moves whose delta was computed
    long long applied = 0;    // improving moves applied
};

class LocalSearch {
public:
    LocalSearch(const DistanceMatrix &distances, const std::vector<int> &costs, IntraMove intra, SearchType type)
        : d_(distances), costs_(costs), intra_(intra), type_(type) {
        const int n = static_cast<int>(distances.size());
        const int k = static_cast<int>(tourSize(distances.size()));
        for (int i = 0; i < k; ++i) {
            for (int j = i + 1; j < k; ++j) {
                // 2-opt on two adjacent edges is the identity move.
                const bool adjacentEdges = (j == i + 1) || (i == 0 && j == k - 1);
                if (intra == IntraMove::EdgesExchange && adjacentEdges) continue;
                moves_.push_back({Move::Intra, i, j});
            }
        }
        for (int i = 0; i < k; ++i) {
            for (int u = 0; u < n - k; ++u) moves_.push_back({Move::Inter, i, u});
        }
        order_.resize(moves_.size());
        std::iota(order_.begin(), order_.end(), 0);
    }

    Tour run(Tour tour, std::mt19937 &rng, SearchCounters &counters) {
        tour_ = std::move(tour);
        std::vector<bool> selected(d_.size(), false);
        for (int v : tour_) selected[v] = true;
        unselected_.clear();
        for (size_t v = 0; v < d_.size(); ++v) {
            if (!selected[v]) unselected_.push_back(static_cast<int>(v));
        }

        while (type_ == SearchType::Steepest ? steepestStep(counters) : greedyStep(rng, counters)) {
            ++counters.applied;
        }
        return tour_;
    }

    size_t neighborhoodSize() const { return moves_.size(); }

private:
    int at(int pos) const {
        const int k = static_cast<int>(tour_.size());
        return tour_[(pos + k) % k];
    }

    long long delta(const Move &m) const {
        if (m.kind == Move::Inter) return interDelta(m.a, m.b);
        return intra_ == IntraMove::NodesExchange ? nodesExchangeDelta(m.a, m.b) : edgesExchangeDelta(m.a, m.b);
    }

    long long interDelta(int i, int u) const {
        const int prev = at(i - 1), v = tour_[i], next = at(i + 1), w = unselected_[u];
        return d_[prev][w] + d_[w][next] - d_[prev][v] - d_[v][next] + costs_[w] - costs_[v];
    }

    long long nodesExchangeDelta(int i, int j) const {
        const int k = static_cast<int>(tour_.size());
        const int a = tour_[i], b = tour_[j];
        if (j == i + 1) {  // ... prev, a, b, next ...  ->  ... prev, b, a, next ...
            const int prev = at(i - 1), next = at(j + 1);
            return d_[prev][b] + d_[a][next] - d_[prev][a] - d_[b][next];
        }
        if (i == 0 && j == k - 1) {  // wrap-around: ... prev, b | a, next ...
            const int prev = at(j - 1), next = at(i + 1);
            return d_[prev][a] + d_[b][next] - d_[prev][b] - d_[a][next];
        }
        const int pa = at(i - 1), na = at(i + 1), pb = at(j - 1), nb = at(j + 1);
        return d_[pa][b] + d_[b][na] + d_[pb][a] + d_[a][nb] - d_[pa][a] - d_[a][na] - d_[pb][b] - d_[b][nb];
    }

    long long edgesExchangeDelta(int i, int j) const {
        const int a = tour_[i], an = at(i + 1), b = tour_[j], bn = at(j + 1);
        return d_[a][b] + d_[an][bn] - d_[a][an] - d_[b][bn];
    }

    void apply(const Move &m) {
        if (m.kind == Move::Inter) {
            std::swap(tour_[m.a], unselected_[m.b]);
        } else if (intra_ == IntraMove::NodesExchange) {
            std::swap(tour_[m.a], tour_[m.b]);
        } else {
            std::reverse(tour_.begin() + m.a + 1, tour_.begin() + m.b + 1);
        }
    }

    // Steepest: evaluate the whole neighborhood, apply the best move if it improves.
    bool steepestStep(SearchCounters &counters) {
        long long bestDelta = 0;
        const Move *best = nullptr;
        for (const Move &m : moves_) {
            const long long dlt = delta(m);
            if (dlt < bestDelta) {
                bestDelta = dlt;
                best = &m;
            }
        }
        counters.evaluated += static_cast<long long>(moves_.size());
        if (best == nullptr) return false;
        apply(*best);
        return true;
    }

    // Greedy: browse the neighborhood (both move kinds mixed together) in uniformly
    // random order, generated lazily with an incremental Fisher-Yates shuffle: the next
    // move is drawn at random from those not yet tried in this step, and only that one
    // is evaluated. The first improving move is applied right away.
    bool greedyStep(std::mt19937 &rng, SearchCounters &counters) {
        const size_t total = order_.size();
        for (size_t tried = 0; tried < total; ++tried) {
            std::uniform_int_distribution<size_t> pick(tried, total - 1);
            std::swap(order_[tried], order_[pick(rng)]);
            const Move &m = moves_[order_[tried]];
            ++counters.evaluated;
            if (delta(m) < 0) {
                apply(m);
                return true;
            }
        }
        return false;
    }

    const DistanceMatrix &d_;
    const std::vector<int> &costs_;
    IntraMove intra_;
    SearchType type_;
    std::vector<Move> moves_;
    std::vector<size_t> order_;
    Tour tour_;
    std::vector<int> unselected_;
};

// Steepest local search with a list of improving moves (LM) ------------------------
// Same neighborhood as the baseline (two-edges exchange + inter-route exchange), but
// move evaluations are kept between iterations. A move is stored by the nodes that
// define it, so its delta stays exact for as long as the move is applicable:
//   * TwoOpt (a, an, b, bn): remove edges (a,an) and (b,bn), add (a,b) and (an,bn).
//     Applicable iff both edges are in the tour with the same relative direction
//     (a->an and b->bn, or an->a and bn->b). With opposite directions the same
//     reconnection would split the tour into two cycles: the move stays in LM, since a
//     later reversal can make it applicable again. If an edge is gone, it is dropped.
//   * Exchange (p, v, n, w): remove v (between neighbors p and n), insert unselected w
//     in its place. Applicable iff w is unselected and v still sits between p and n.
// After a move is applied only the moves that touch the new edges / changed nodes are
// evaluated. When no stored move is applicable, all moves are evaluated once more; the
// search stops when that adds nothing (the solution is then a steepest local optimum).
struct ListMove {
    long long delta;
    enum Kind : unsigned char { TwoOpt, Exchange } kind;
    int n1, n2, n3, n4;

    bool operator<(const ListMove &o) const {
        return std::tie(delta, kind, n1, n2, n3, n4) < std::tie(o.delta, o.kind, o.n1, o.n2, o.n3, o.n4);
    }
};

struct ListCounters {
    long long evaluated = 0;
    long long applied = 0;
    long long fullEvaluations = 0;  // full-neighborhood evaluations (initial one included)
    long long peakListSize = 0;
};

class ListLocalSearch {
public:
    ListLocalSearch(const DistanceMatrix &distances, const std::vector<int> &costs)
        : d_(distances), costs_(costs) {}

    Tour run(Tour tour, ListCounters &counters) {
        tour_ = std::move(tour);
        const int n = static_cast<int>(d_.size());
        pos_.assign(n, -1);
        for (int i = 0; i < static_cast<int>(tour_.size()); ++i) pos_[tour_[i]] = i;
        unselected_.clear();
        for (int v = 0; v < n; ++v) {
            if (pos_[v] < 0) unselected_.push_back(v);
        }
        list_.clear();
        counters_ = &counters;

        // LM may keep moves that are not applicable now (opposite edge directions), so
        // it never has to become empty: the search stops when no stored move is
        // applicable right after a full evaluation of the neighborhood.
        evaluateAll();
        bool justEvaluatedAll = true;
        while (true) {
            if (applyBestApplicable()) {
                ++counters.applied;
                justEvaluatedAll = false;
            } else if (justEvaluatedAll) {
                break;
            } else {
                evaluateAll();
                justEvaluatedAll = true;
            }
        }
        return tour_;
    }

private:
    int k() const { return static_cast<int>(tour_.size()); }
    int succ(int v) const { return tour_[(pos_[v] + 1) % k()]; }
    int pred(int v) const { return tour_[(pos_[v] + k() - 1) % k()]; }

    // +1 if the tour goes x -> y, -1 if it goes y -> x, 0 if (x, y) is not a tour edge.
    int edgeDirection(int x, int y) const {
        if (pos_[x] < 0 || pos_[y] < 0) return 0;
        if (succ(x) == y) return 1;
        if (pred(x) == y) return -1;
        return 0;
    }

    void push(const ListMove &m) {
        ++counters_->evaluated;
        if (m.delta >= 0) return;
        list_.insert(m);
        counters_->peakListSize = std::max(counters_->peakListSize, static_cast<long long>(list_.size()));
    }

    // Both reconnections of edges a->an and b->bn: the applicable one, (a,b)+(an,bn),
    // and the one with the second edge relatively inverted, (a,bn)+(an,b), stored as
    // the move on edges (a,an) and (bn,b).
    void evaluateEdgePair(int a, int an, int b, int bn) {
        if (a == b || a == bn || an == b || an == bn) return;  // edges share a node
        const long long removed = d_[a][an] + d_[b][bn];
        push({d_[a][b] + d_[an][bn] - removed, ListMove::TwoOpt, a, an, b, bn});
        push({d_[a][bn] + d_[an][b] - removed, ListMove::TwoOpt, a, an, bn, b});
    }

    void evaluateExchange(int v, int w) {
        const int p = pred(v), nx = succ(v);
        push({d_[p][w] + d_[w][nx] - d_[p][v] - d_[v][nx] + costs_[w] - costs_[v], ListMove::Exchange, p, v, nx, w});
    }

    void evaluateAll() {
        ++counters_->fullEvaluations;
        for (int i = 0; i < k(); ++i) {
            for (int j = i + 1; j < k(); ++j) {
                evaluateEdgePair(tour_[i], tour_[(i + 1) % k()], tour_[j], tour_[(j + 1) % k()]);
            }
        }
        for (int v : tour_) {
            for (int w : unselected_) evaluateExchange(v, w);
        }
    }

    // New tour edge x -> y: pair it with every other tour edge.
    void evaluateNewEdge(int x, int y) {
        for (int i = 0; i < k(); ++i) evaluateEdgePair(x, y, tour_[i], tour_[(i + 1) % k()]);
    }

    // Node v got a new neighbor: re-evaluate replacing it with every unselected node.
    void evaluateNodeExchanges(int v) {
        for (int w : unselected_) evaluateExchange(v, w);
    }

    // Browse LM from the best move: drop moves that no longer exist, skip moves whose
    // edges have opposite relative directions, apply the first applicable one.
    bool applyBestApplicable() {
        for (auto it = list_.begin(); it != list_.end();) {
            const ListMove m = *it;
            if (m.kind == ListMove::TwoOpt) {
                const int dir1 = edgeDirection(m.n1, m.n2), dir2 = edgeDirection(m.n3, m.n4);
                if (dir1 == 0 || dir2 == 0) {
                    it = list_.erase(it);
                } else if (dir1 != dir2) {
                    ++it;
                } else {
                    list_.erase(it);
                    applyTwoOpt(m, dir1);
                    return true;
                }
            } else {
                const bool applicable = pos_[m.n4] < 0 && pos_[m.n2] >= 0 &&
                                        ((pred(m.n2) == m.n1 && succ(m.n2) == m.n3) ||
                                         (pred(m.n2) == m.n3 && succ(m.n2) == m.n1));
                if (!applicable) {
                    it = list_.erase(it);
                } else {
                    list_.erase(it);
                    applyExchange(m);
                    return true;
                }
            }
        }
        return false;
    }

    void applyTwoOpt(const ListMove &m, int direction) {
        const int a = m.n1, an = m.n2, b = m.n3, bn = m.n4;
        // Tour positions where the two removed edges start (in tour order).
        int i = direction == 1 ? pos_[a] : pos_[an];
        int j = direction == 1 ? pos_[b] : pos_[bn];
        if (i > j) std::swap(i, j);
        std::reverse(tour_.begin() + i + 1, tour_.begin() + j + 1);
        for (int p = i + 1; p <= j; ++p) pos_[tour_[p]] = p;

        // New edges (a,b) and (an,bn), taken in their current tour direction.
        const int x1 = edgeDirection(a, b) == 1 ? a : b, y1 = x1 == a ? b : a;
        const int x2 = edgeDirection(an, bn) == 1 ? an : bn, y2 = x2 == an ? bn : an;
        evaluateNewEdge(x1, y1);
        evaluateNewEdge(x2, y2);
        for (int v : {a, an, b, bn}) evaluateNodeExchanges(v);
    }

    void applyExchange(const ListMove &m) {
        const int v = m.n2, w = m.n4;
        const int at = pos_[v];
        tour_[at] = w;
        pos_[w] = at;
        pos_[v] = -1;
        *std::find(unselected_.begin(), unselected_.end(), w) = v;

        const int p = pred(w), nx = succ(w);
        evaluateNewEdge(p, w);
        evaluateNewEdge(w, nx);
        for (int u : {p, w, nx}) evaluateNodeExchanges(u);
        // v is a new candidate for every position of the tour.
        for (int u : tour_) {
            if (u != p && u != w && u != nx) evaluateExchange(u, v);
        }
    }

    const DistanceMatrix &d_;
    const std::vector<int> &costs_;
    Tour tour_;
    std::vector<int> pos_;
    std::vector<int> unselected_;
    // LM nodes come from a pool: thousands of insertions/erasures per run would
    // otherwise each hit the general-purpose allocator.
    std::pmr::unsynchronized_pool_resource pool_;
    std::pmr::set<ListMove> list_{&pool_};
    ListCounters *counters_ = nullptr;
};

// Experiment driver ---------------------------------------------------------------
struct Stats {
    long long min = std::numeric_limits<long long>::max();
    long long max = std::numeric_limits<long long>::min();
    double avg = 0.0;
    double timeMinMs = std::numeric_limits<double>::max();
    double timeMaxMs = 0.0;
    double timeAvgMs = 0.0;
    double evaluatedAvg = 0.0;
    double appliedAvg = 0.0;
    double fullEvaluationsAvg = 0.0;
    double peakListAvg = 0.0;
    Tour best;
};

void writeTour(const std::string &path, const Tour &tour) {
    std::ofstream out(path);
    for (int idx : tour) out << idx << "\n";
}

// Runs `method(run, counters)` for every starting solution; times only the search.
template <typename Method>
Stats runMany(int runs, const DistanceMatrix &distances, const std::vector<int> &costs, Method method) {
    Stats stats;
    double sum = 0.0, timeSum = 0.0, evaluatedSum = 0.0, appliedSum = 0.0, fullSum = 0.0, peakSum = 0.0;
    for (int run = 0; run < runs; ++run) {
        ListCounters counters;
        Stopwatch stopwatch;
        const Tour tour = method(run, counters);
        const double ms = stopwatch.elapsedSeconds() * 1000.0;

        const long long score = evaluateTour(tour, distances, costs);
        sum += static_cast<double>(score);
        timeSum += ms;
        evaluatedSum += static_cast<double>(counters.evaluated);
        appliedSum += static_cast<double>(counters.applied);
        fullSum += static_cast<double>(counters.fullEvaluations);
        peakSum += static_cast<double>(counters.peakListSize);
        stats.timeMinMs = std::min(stats.timeMinMs, ms);
        stats.timeMaxMs = std::max(stats.timeMaxMs, ms);
        if (score < stats.min) {
            stats.min = score;
            stats.best = tour;
        }
        if (score > stats.max) stats.max = score;
    }
    stats.avg = sum / runs;
    stats.timeAvgMs = timeSum / runs;
    stats.evaluatedAvg = evaluatedSum / runs;
    stats.appliedAvg = appliedSum / runs;
    stats.fullEvaluationsAvg = fullSum / runs;
    stats.peakListAvg = peakSum / runs;
    return stats;
}

void runInstance(const std::string &instanceName, std::mt19937 &rng) {
    const std::string dataPath = "0_setup/" + instanceName + "_data.csv";

    const std::vector<Point> points = readPoints(dataPath);
    if (points.empty()) {
        std::cerr << "No points read from " << dataPath << "\n";
        return;
    }

    const DistanceMatrix distances = buildDistanceMatrix(points);
    const std::vector<int> costs = extractCosts(points);
    const size_t n = points.size();
    const int runs = 200;
    std::cout << "Instance " << instanceName << ": " << n << " points, tour size " << tourSize(n) << "\n";

    // Both methods start from the same 200 random solutions.
    std::vector<Tour> starts;
    for (int r = 0; r < runs; ++r) starts.push_back(randomSolution(n, rng));

    struct Method {
        std::string key;
        std::string label;
        Stats stats;
    };
    std::vector<Method> methods;

    LocalSearch baseline(distances, costs, IntraMove::EdgesExchange, SearchType::Steepest);
    methods.push_back({"ls_steepest_edges", "Steepest LS, edges exchange (baseline)",
                       runMany(runs, distances, costs, [&](int r, ListCounters &counters) {
                           SearchCounters c;
                           Tour result = baseline.run(starts[r], rng, c);
                           counters.evaluated = c.evaluated;
                           counters.applied = c.applied;
                           return result;
                       })});

    ListLocalSearch withList(distances, costs);
    methods.push_back({"ls_steepest_edges_lm", "Steepest LS, edges exchange, list of improving moves",
                       runMany(runs, distances, costs, [&](int r, ListCounters &counters) {
                           return withList.run(starts[r], counters);
                       })});

    std::ofstream summary(std::string(RESULTS_DIR) + "/" + instanceName + "_summary.csv");
    summary << "instance,method,label,min,max,avg,time_min_ms,time_max_ms,time_avg_ms,evaluated_avg,applied_avg,"
               "full_evaluations_avg,peak_list_avg\n";
    summary << std::fixed;
    for (const auto &m : methods) {
        const Stats &s = m.stats;
        std::cout << "  " << m.label << ": min=" << s.min << " max=" << s.max << " avg=" << s.avg << " (avg "
                  << s.timeAvgMs << " ms, " << s.evaluatedAvg << " evaluated, " << s.appliedAvg << " applied, "
                  << s.fullEvaluationsAvg << " full evaluations)\n";
        summary << std::setprecision(1) << instanceName << "," << m.key << ",\"" << m.label << "\"," << s.min << ","
                << s.max << "," << s.avg << "," << std::setprecision(4) << s.timeMinMs << "," << s.timeMaxMs << ","
                << s.timeAvgMs << "," << std::setprecision(1) << s.evaluatedAvg << "," << s.appliedAvg << ","
                << s.fullEvaluationsAvg << "," << s.peakListAvg << "\n";
        writeTour(std::string(RESULTS_DIR) + "/" + instanceName + "_" + m.key + "_best.txt", s.best);
    }
}

int main() {
    std::filesystem::create_directories(RESULTS_DIR);
    // Fixed seed: reproducible runs for the report.
    std::mt19937 rng(42);
    for (const std::string &instance : {"A", "B"}) {
        runInstance(instance, rng);
    }
    return 0;
}
