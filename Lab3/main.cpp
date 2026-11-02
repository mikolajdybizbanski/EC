#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// Settings ----------------------------------------------------------------------
constexpr char CSV_DELIMITER = ';';
constexpr const char *RESULTS_DIR = "3_local/results";

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

// Constructive heuristics (exercise 1) ------------------------------------------
// Random solution: pick `size` distinct random nodes, in random order.
Tour randomSolution(size_t numPoints, std::mt19937 &rng) {
    const size_t size = tourSize(numPoints);
    std::vector<int> candidates(numPoints);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::shuffle(candidates.begin(), candidates.end(), rng);
    return Tour(candidates.begin(), candidates.begin() + size);
}

// Nearest neighbor, end only: always append the unvisited node that minimizes the
// increase of the objective function (distance from the last node + cost of the
// candidate) to the end of the path.
Tour nearestNeighborEnd(const DistanceMatrix &distances, const std::vector<int> &costs, int start) {
    const size_t numPoints = distances.size();
    const size_t size = tourSize(numPoints);

    Tour tour;
    tour.reserve(size);
    std::vector<bool> used(numPoints, false);

    tour.push_back(start);
    used[start] = true;

    while (tour.size() < size) {
        const int last = tour.back();
        int best = -1;
        long long bestDelta = std::numeric_limits<long long>::max();
        for (size_t c = 0; c < numPoints; ++c) {
            if (used[c]) continue;
            const long long delta = distances[last][c] + costs[c];
            if (delta < bestDelta) {
                bestDelta = delta;
                best = static_cast<int>(c);
            }
        }
        tour.push_back(best);
        used[best] = true;
    }
    return tour;
}

// Nearest neighbor, any position: grow an open path, at every step inserting the
// unvisited node at whichever position (before the first node, after the last node,
// or between any two consecutive nodes) minimizes the increase of the objective
// function. The path is only closed into a cycle once it reaches full size.
Tour nearestNeighborAnyPosition(const DistanceMatrix &distances, const std::vector<int> &costs, int start) {
    const size_t numPoints = distances.size();
    const size_t size = tourSize(numPoints);

    Tour tour;
    tour.reserve(size);
    std::vector<bool> used(numPoints, false);

    tour.push_back(start);
    used[start] = true;

    while (tour.size() < size) {
        int bestNode = -1;
        size_t bestPos = 0;
        long long bestDelta = std::numeric_limits<long long>::max();

        for (size_t c = 0; c < numPoints; ++c) {
            if (used[c]) continue;

            // Prepend (no edge removed).
            {
                const long long delta = distances[c][tour.front()] + costs[c];
                if (delta < bestDelta) {
                    bestDelta = delta;
                    bestNode = static_cast<int>(c);
                    bestPos = 0;
                }
            }
            // Append (no edge removed).
            {
                const long long delta = distances[tour.back()][c] + costs[c];
                if (delta < bestDelta) {
                    bestDelta = delta;
                    bestNode = static_cast<int>(c);
                    bestPos = tour.size();
                }
            }
            // Insert strictly inside the path (removes one edge, adds two).
            for (size_t p = 1; p < tour.size(); ++p) {
                const int prev = tour[p - 1];
                const int next = tour[p];
                const long long delta = distances[prev][c] + distances[c][next] - distances[prev][next] + costs[c];
                if (delta < bestDelta) {
                    bestDelta = delta;
                    bestNode = static_cast<int>(c);
                    bestPos = p;
                }
            }
        }

        tour.insert(tour.begin() + bestPos, bestNode);
        used[bestNode] = true;
    }
    return tour;
}

// Greedy cycle: grow a closed cycle (starting from two nodes), at every step
// inserting the unvisited node, between whichever pair of adjacent cycle nodes,
// that minimizes the increase of the objective function.
Tour greedyCycle(const DistanceMatrix &distances, const std::vector<int> &costs, int start) {
    const size_t numPoints = distances.size();
    const size_t size = tourSize(numPoints);

    std::vector<bool> used(numPoints, false);

    // Seed the cycle with `start` and its cheapest companion.
    int second = -1;
    long long bestSeedDelta = std::numeric_limits<long long>::max();
    for (size_t c = 0; c < numPoints; ++c) {
        if (static_cast<int>(c) == start) continue;
        const long long delta = distances[start][c] + costs[c];
        if (delta < bestSeedDelta) {
            bestSeedDelta = delta;
            second = static_cast<int>(c);
        }
    }

    Tour tour = {start, second};
    used[start] = true;
    used[second] = true;

    while (tour.size() < size) {
        int bestNode = -1;
        size_t bestPos = 0;
        long long bestDelta = std::numeric_limits<long long>::max();

        for (size_t c = 0; c < numPoints; ++c) {
            if (used[c]) continue;
            for (size_t i = 0; i < tour.size(); ++i) {
                const int prev = tour[i];
                const int next = tour[(i + 1) % tour.size()];
                const long long delta = distances[prev][c] + distances[c][next] - distances[prev][next] + costs[c];
                if (delta < bestDelta) {
                    bestDelta = delta;
                    bestNode = static_cast<int>(c);
                    bestPos = i + 1;
                }
            }
        }

        tour.insert(tour.begin() + bestPos, bestNode);
        used[bestNode] = true;
    }
    return tour;
}

// Regret heuristics (exercise 2) ------------------------------------------------
// Both regret heuristics share one selection rule. For every unvisited candidate c we
// find the cheapest and the second-cheapest way to insert it into the current partial
// solution (delta of the objective function: added edges - removed edge + cost of c).
//   regret(c) = secondBest(c) - best(c)
// The chosen candidate maximizes
//   score(c) = regretWeight * regret(c) - deltaWeight * best(c)
// and is inserted at its best position.
//   * pure 2-regret:   regretWeight = 1, deltaWeight = 0
//   * weighted sum:    regretWeight = 1, deltaWeight = 1 (equal weights)
struct RegretWeights {
    double regret;
    double delta;
};

constexpr RegretWeights PURE_REGRET{1.0, 0.0};
constexpr RegretWeights WEIGHTED_REGRET{1.0, 1.0};

// Tracks the best and second-best insertion of a single candidate.
struct InsertionOptions {
    long long best = std::numeric_limits<long long>::max();
    long long second = std::numeric_limits<long long>::max();
    size_t bestPos = 0;

    void offer(long long delta, size_t pos) {
        if (delta < best) {
            second = best;
            best = delta;
            bestPos = pos;
        } else if (delta < second) {
            second = delta;
        }
    }

    // With a single possible insertion there is nothing to regret.
    long long regret() const {
        return second == std::numeric_limits<long long>::max() ? 0 : second - best;
    }
};

// Picks the candidate with the highest weighted score, returns {node, position}.
template <typename EnumerateInsertions>
std::pair<int, size_t> pickByRegret(const std::vector<bool> &used, RegretWeights weights,
                                    EnumerateInsertions enumerate) {
    int bestNode = -1;
    size_t bestPos = 0;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (size_t c = 0; c < used.size(); ++c) {
        if (used[c]) continue;
        InsertionOptions options;
        enumerate(static_cast<int>(c), options);
        const double score = weights.regret * static_cast<double>(options.regret()) -
                             weights.delta * static_cast<double>(options.best);
        if (score > bestScore) {
            bestScore = score;
            bestNode = static_cast<int>(c);
            bestPos = options.bestPos;
        }
    }
    return {bestNode, bestPos};
}

// Regret heuristic built on nearest neighbor, any position: the partial solution is an
// open path; a candidate can be prepended, appended, or inserted between two
// consecutive path nodes. Position p means "insert before tour[p]".
Tour regretNearestNeighbor(const DistanceMatrix &distances, const std::vector<int> &costs, int start,
                           RegretWeights weights) {
    const size_t numPoints = distances.size();
    const size_t size = tourSize(numPoints);

    Tour tour;
    tour.reserve(size);
    std::vector<bool> used(numPoints, false);

    tour.push_back(start);
    used[start] = true;

    while (tour.size() < size) {
        const auto [node, pos] = pickByRegret(used, weights, [&](int c, InsertionOptions &options) {
            options.offer(distances[c][tour.front()] + costs[c], 0);
            options.offer(distances[tour.back()][c] + costs[c], tour.size());
            for (size_t p = 1; p < tour.size(); ++p) {
                const int prev = tour[p - 1];
                const int next = tour[p];
                options.offer(distances[prev][c] + distances[c][next] - distances[prev][next] + costs[c], p);
            }
        });
        tour.insert(tour.begin() + static_cast<std::ptrdiff_t>(pos), node);
        used[node] = true;
    }
    return tour;
}

// Regret heuristic built on greedy cycle: the partial solution is a closed cycle
// (seeded, as in greedy cycle, with `start` and its cheapest companion); a candidate
// can be inserted into any cycle edge. Position i means "insert after tour[i]".
Tour regretGreedyCycle(const DistanceMatrix &distances, const std::vector<int> &costs, int start,
                       RegretWeights weights) {
    const size_t numPoints = distances.size();
    const size_t size = tourSize(numPoints);

    std::vector<bool> used(numPoints, false);

    int second = -1;
    long long bestSeedDelta = std::numeric_limits<long long>::max();
    for (size_t c = 0; c < numPoints; ++c) {
        if (static_cast<int>(c) == start) continue;
        const long long delta = distances[start][c] + costs[c];
        if (delta < bestSeedDelta) {
            bestSeedDelta = delta;
            second = static_cast<int>(c);
        }
    }

    Tour tour = {start, second};
    tour.reserve(size);
    used[start] = true;
    used[second] = true;

    while (tour.size() < size) {
        const auto [node, pos] = pickByRegret(used, weights, [&](int c, InsertionOptions &options) {
            for (size_t i = 0; i < tour.size(); ++i) {
                const int prev = tour[i];
                const int next = tour[(i + 1) % tour.size()];
                options.offer(distances[prev][c] + distances[c][next] - distances[prev][next] + costs[c], i);
            }
        });
        tour.insert(tour.begin() + static_cast<std::ptrdiff_t>(pos) + 1, node);
        used[node] = true;
    }
    return tour;
}

// Local search --------------------------------------------------------------------
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
    Tour best;
};

void writeTour(const std::string &path, const Tour &tour) {
    std::ofstream out(path);
    for (int idx : tour) out << idx << "\n";
}

// Runs `method(run)` for run = 0..runs-1; each call returns one solution and may
// fill `counters`. The time of each run (construction + local search) is measured.
template <typename Method>
Stats runMany(int runs, const DistanceMatrix &distances, const std::vector<int> &costs, Method method) {
    Stats stats;
    double sum = 0.0, timeSum = 0.0, evaluatedSum = 0.0, appliedSum = 0.0;
    for (int run = 0; run < runs; ++run) {
        SearchCounters counters;
        Stopwatch stopwatch;
        const Tour tour = method(run, counters);
        const double ms = stopwatch.elapsedSeconds() * 1000.0;

        const long long score = evaluateTour(tour, distances, costs);
        sum += static_cast<double>(score);
        timeSum += ms;
        evaluatedSum += static_cast<double>(counters.evaluated);
        appliedSum += static_cast<double>(counters.applied);
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
    const int runs = static_cast<int>(n);  // one run per starting node; 200 random runs
    std::cout << "Instance " << instanceName << ": " << n << " points, tour size " << tourSize(n) << "\n";

    struct Method {
        std::string key;
        std::string label;
        Stats stats;
    };
    std::vector<Method> methods;

    // Previous (constructive) methods.
    using Construct = std::function<Tour(int)>;
    const std::vector<std::tuple<std::string, std::string, Construct>> constructive = {
        {"nn_end", "Nearest neighbor (end only)", [&](int s) { return nearestNeighborEnd(distances, costs, s); }},
        {"nn_any", "Nearest neighbor (any position)",
         [&](int s) { return nearestNeighborAnyPosition(distances, costs, s); }},
        {"greedy_cycle", "Greedy cycle", [&](int s) { return greedyCycle(distances, costs, s); }},
        {"regret_nn", "2-regret (nearest neighbor, any position)",
         [&](int s) { return regretNearestNeighbor(distances, costs, s, PURE_REGRET); }},
        {"weighted_nn", "Weighted 2-regret + delta (nearest neighbor, any position)",
         [&](int s) { return regretNearestNeighbor(distances, costs, s, WEIGHTED_REGRET); }},
        {"regret_cycle", "2-regret (greedy cycle)",
         [&](int s) { return regretGreedyCycle(distances, costs, s, PURE_REGRET); }},
        {"weighted_cycle", "Weighted 2-regret + delta (greedy cycle)",
         [&](int s) { return regretGreedyCycle(distances, costs, s, WEIGHTED_REGRET); }},
    };

    methods.push_back({"random", "Random solution",
                       runMany(runs, distances, costs, [&](int, SearchCounters &) { return randomSolution(n, rng); })});
    size_t bestConstructive = 0;
    for (size_t c = 0; c < constructive.size(); ++c) {
        const auto &[key, label, construct] = constructive[c];
        methods.push_back({key, label, runMany(runs, distances, costs, [&construct = construct](int s, SearchCounters &) {
                               return construct(s);
                           })});
        if (methods.back().stats.avg < methods[1 + bestConstructive].stats.avg) bestConstructive = c;
    }
    const auto &[greedyKey, greedyLabel, greedyConstruct] = constructive[bestConstructive];
    const Construct &greedyStart = greedyConstruct;
    std::cout << "  Greedy starting solutions: " << greedyLabel << "\n";

    // The same 200 random starting solutions are used by every random-start variant.
    std::vector<Tour> randomStarts;
    for (int r = 0; r < runs; ++r) randomStarts.push_back(randomSolution(n, rng));

    for (SearchType type : {SearchType::Steepest, SearchType::Greedy}) {
        for (IntraMove intra : {IntraMove::NodesExchange, IntraMove::EdgesExchange}) {
            for (bool fromGreedy : {false, true}) {
                LocalSearch search(distances, costs, intra, type);
                const std::string key = std::string(type == SearchType::Steepest ? "ls_steepest" : "ls_greedy") +
                                        (intra == IntraMove::NodesExchange ? "_nodes" : "_edges") +
                                        (fromGreedy ? "_greedy" : "_random");
                const std::string label =
                    std::string(type == SearchType::Steepest ? "Steepest LS" : "Greedy LS") +
                    (intra == IntraMove::NodesExchange ? ", nodes exchange" : ", edges exchange") +
                    (fromGreedy ? ", greedy start" : ", random start");
                methods.push_back({key, label, runMany(runs, distances, costs, [&](int r, SearchCounters &counters) {
                                       Tour start = fromGreedy ? greedyStart(r) : randomStarts[r];
                                       return search.run(std::move(start), rng, counters);
                                   })});
            }
        }
    }

    std::ofstream summary(std::string(RESULTS_DIR) + "/" + instanceName + "_summary.csv");
    summary << "instance,method,label,min,max,avg,time_min_ms,time_max_ms,time_avg_ms,evaluated_avg,applied_avg\n";
    summary << std::fixed;
    for (const auto &m : methods) {
        const Stats &s = m.stats;
        std::cout << "  " << m.label << ": min=" << s.min << " max=" << s.max << " avg=" << s.avg
                  << " (avg " << s.timeAvgMs << " ms, " << s.evaluatedAvg << " evaluated moves)\n";
        summary << std::setprecision(1) << instanceName << "," << m.key << ",\"" << m.label << "\"," << s.min << ","
                << s.max << "," << s.avg << "," << std::setprecision(4) << s.timeMinMs << "," << s.timeMaxMs << ","
                << s.timeAvgMs << "," << std::setprecision(1) << s.evaluatedAvg << "," << s.appliedAvg << "\n";
        writeTour(std::string(RESULTS_DIR) + "/" + instanceName + "_" + m.key + "_best.txt", s.best);
    }
    std::ofstream(std::string(RESULTS_DIR) + "/" + instanceName + "_greedy_start.txt") << greedyKey << "\n";
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
