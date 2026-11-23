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
constexpr const char *RESULTS_DIR = "7_large/results";

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

// Steepest local search with a list of improving moves (LM), from exercise 5 ---------
// Neighborhood of exercise 3 (two-edges exchange + inter-route exchange), but
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

// Local search used everywhere in this exercise -------------------------------------
// The best steepest LS of the previous exercises: two-edges exchange + inter-route
// exchange, with the list of improving moves (exercise 5).
class SteepestLS {
public:
    SteepestLS(const DistanceMatrix &distances, const std::vector<int> &costs) : ls_(distances, costs) {}

    Tour operator()(Tour tour) {
        ListCounters counters;
        return ls_.run(std::move(tour), counters);
    }

private:
    ListLocalSearch ls_;
};

// Repair operator: weighted 2-regret greedy cycle (exercise 2) ------------------------
// The best constructive heuristic of exercise 2 (lowest average over both instances).
// The partial solution is a closed cycle; a candidate can be inserted into any of its
// edges. For every unselected node c:
//   best(c), second(c) = cheapest / second-cheapest insertion delta
//                        (d(prev,c) + d(c,next) - d(prev,next) + cost(c))
//   score(c) = (second(c) - best(c)) - best(c)     (regret and delta, equal weights)
// The node with the highest score is inserted at its best edge, until the cycle has
// ceil(n/2) nodes. Every unselected node is a candidate (the removed ones included).
// Because the partial solution is a cycle, no extra cycle breaking is needed.
Tour repairWeightedRegret(Tour tour, const DistanceMatrix &d, const std::vector<int> &costs) {
    const int n = static_cast<int>(d.size());
    const size_t target = tourSize(d.size());
    std::vector<bool> used(n, false);
    for (int v : tour) used[v] = true;
    std::vector<int> candidates;
    for (int v = 0; v < n; ++v) {
        if (!used[v]) candidates.push_back(v);
    }
    tour.reserve(target);

    while (tour.size() < target) {
        const size_t k = tour.size();
        double bestScore = -std::numeric_limits<double>::infinity();
        size_t bestCand = 0, bestPos = 0;
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            const int c = candidates[ci];
            long long best = std::numeric_limits<long long>::max(), second = best;
            size_t pos = 0;
            for (size_t i = 0; i < k; ++i) {
                const int prev = tour[i], next = tour[(i + 1) % k];
                const long long delta = d[prev][c] + d[c][next] - d[prev][next];
                if (delta < best) {
                    second = best;
                    best = delta;
                    pos = i;
                } else if (delta < second) {
                    second = delta;
                }
            }
            best += costs[c];
            const long long regret = second == std::numeric_limits<long long>::max() ? 0 : second + costs[c] - best;
            const double score = static_cast<double>(regret) - static_cast<double>(best);
            if (score > bestScore) {
                bestScore = score;
                bestCand = ci;
                bestPos = pos;
            }
        }
        tour.insert(tour.begin() + static_cast<std::ptrdiff_t>(bestPos) + 1, candidates[bestCand]);
        candidates[bestCand] = candidates.back();
        candidates.pop_back();
    }
    return tour;
}

// Destroy operators ----------------------------------------------------------------------
// All three remove `count` nodes (30% of the tour = 30 nodes) and keep the order of the
// remaining ones, so the result is still a (shorter) closed cycle.
enum class Destroy { RandomNodes, SingleSubpath, MultiSubpath };

Tour keepUnmarked(const Tour &tour, const std::vector<bool> &removed) {
    Tour kept;
    kept.reserve(tour.size());
    for (size_t i = 0; i < tour.size(); ++i) {
        if (!removed[i]) kept.push_back(tour[i]);
    }
    return kept;
}

// Random composition of `total` into `parts` positive integers.
std::vector<int> randomComposition(int total, int parts, std::mt19937 &rng) {
    std::vector<int> cuts(total - 1);
    std::iota(cuts.begin(), cuts.end(), 1);
    std::shuffle(cuts.begin(), cuts.end(), rng);
    cuts.resize(parts - 1);
    cuts.push_back(0);
    cuts.push_back(total);
    std::sort(cuts.begin(), cuts.end());
    std::vector<int> sizes;
    for (int i = 1; i <= parts; ++i) sizes.push_back(cuts[i] - cuts[i - 1]);
    return sizes;
}

Tour destroy(const Tour &tour, Destroy kind, int count, std::mt19937 &rng) {
    const int k = static_cast<int>(tour.size());
    std::vector<bool> removed(k, false);
    if (kind == Destroy::RandomNodes) {
        // `count` tour positions drawn uniformly without replacement.
        std::vector<int> positions(k);
        std::iota(positions.begin(), positions.end(), 0);
        std::shuffle(positions.begin(), positions.end(), rng);
        for (int i = 0; i < count; ++i) removed[positions[i]] = true;
    } else {
        // Subpaths: `parts` removed segments separated by `parts` kept segments, all of
        // random positive lengths, laid around the cycle from a random start position.
        // The segments never overlap or touch, so exactly `count` nodes are removed.
        const int parts = kind == Destroy::SingleSubpath ? 1 : std::uniform_int_distribution<int>(3, 5)(rng);
        const std::vector<int> cut = randomComposition(count, parts, rng);
        const std::vector<int> keep = randomComposition(k - count, parts, rng);
        int p = std::uniform_int_distribution<int>(0, k - 1)(rng);
        for (int s = 0; s < parts; ++s) {
            for (int i = 0; i < cut[s]; ++i) removed[(p + i) % k] = true;
            p += cut[s] + keep[s];
        }
    }
    return keepUnmarked(tour, removed);
}

// Perturbation for ILS ---------------------------------------------------------------------
// `moves` random moves of the LS neighborhood, applied blindly: each one is, with equal
// probability, a two-edges exchange (reverse a random segment) or an inter-route
// exchange (replace a random selected node with a random unselected node).
Tour perturb(Tour tour, size_t numPoints, int moves, std::mt19937 &rng) {
    const int k = static_cast<int>(tour.size());
    std::vector<bool> selected(numPoints, false);
    for (int v : tour) selected[v] = true;
    std::uniform_int_distribution<int> pos(0, k - 1), node(0, static_cast<int>(numPoints) - 1), coin(0, 1);
    for (int m = 0; m < moves; ++m) {
        if (coin(rng)) {
            int i = pos(rng), j = pos(rng);
            while (std::abs(i - j) < 2) j = pos(rng);
            if (i > j) std::swap(i, j);
            std::reverse(tour.begin() + i + 1, tour.begin() + j + 1);
        } else {
            int w = node(rng);
            while (selected[w]) w = node(rng);
            const int i = pos(rng);
            selected[tour[i]] = false;
            selected[w] = true;
            tour[i] = w;
        }
    }
    return tour;
}

// Methods --------------------------------------------------------------------------------
struct RunResult {
    Tour tour;
    long long iterations = 0;  // main-loop iterations (MSLS: local searches)
};

// MSLS: `starts` steepest local searches from random solutions, keep the best.
RunResult msls(int starts, const DistanceMatrix &d, const std::vector<int> &costs, SteepestLS &ls,
               std::mt19937 &rng) {
    RunResult r;
    long long best = std::numeric_limits<long long>::max();
    for (int s = 0; s < starts; ++s) {
        Tour t = ls(randomSolution(d.size(), rng));
        const long long f = evaluateTour(t, d, costs);
        if (f < best) {
            best = f;
            r.tour = std::move(t);
        }
        ++r.iterations;
    }
    return r;
}

constexpr int ILS_PERTURBATION_MOVES = 10;

// ILS: perturb the current solution, apply LS, accept if strictly better.
RunResult ils(double budgetSec, const DistanceMatrix &d, const std::vector<int> &costs, SteepestLS &ls,
              std::mt19937 &rng) {
    Stopwatch clock;
    RunResult r;
    r.tour = ls(randomSolution(d.size(), rng));
    long long fx = evaluateTour(r.tour, d, costs);
    while (clock.elapsedSeconds() < budgetSec) {
        Tour y = ls(perturb(r.tour, d.size(), ILS_PERTURBATION_MOVES, rng));
        const long long fy = evaluateTour(y, d, costs);
        if (fy < fx) {
            fx = fy;
            r.tour = std::move(y);
        }
        ++r.iterations;
    }
    return r;
}

constexpr double DESTROY_FRACTION = 0.3;

// LNS: destroy + repair (+ optional LS) the current solution, accept if strictly better.
// The random starting solution is always improved by LS first.
RunResult lns(double budgetSec, Destroy kind, bool useLS, const DistanceMatrix &d, const std::vector<int> &costs,
              SteepestLS &ls, std::mt19937 &rng) {
    Stopwatch clock;
    RunResult r;
    r.tour = ls(randomSolution(d.size(), rng));
    long long fx = evaluateTour(r.tour, d, costs);
    const int removeCount = static_cast<int>(std::lround(DESTROY_FRACTION * static_cast<double>(r.tour.size())));
    while (clock.elapsedSeconds() < budgetSec) {
        Tour y = repairWeightedRegret(destroy(r.tour, kind, removeCount, rng), d, costs);
        if (useLS) y = ls(std::move(y));
        const long long fy = evaluateTour(y, d, costs);
        if (fy < fx) {
            fx = fy;
            r.tour = std::move(y);
        }
        ++r.iterations;
    }
    return r;
}

// Experiment driver ---------------------------------------------------------------
struct Stats {
    long long min = std::numeric_limits<long long>::max();
    long long max = std::numeric_limits<long long>::min();
    double avg = 0.0;
    double timeMinMs = std::numeric_limits<double>::max();
    double timeMaxMs = 0.0;
    double timeAvgMs = 0.0;
    long long itMin = std::numeric_limits<long long>::max();
    long long itMax = 0;
    double itAvg = 0.0;
    Tour best;
};

void writeTour(const std::string &path, const Tour &tour) {
    std::ofstream out(path);
    for (int idx : tour) out << idx << "\n";
}

constexpr int RUNS = 20;
constexpr int MSLS_STARTS = 200;

template <typename Method>
Stats runMany(const DistanceMatrix &d, const std::vector<int> &costs, Method method) {
    Stats s;
    double sum = 0.0, timeSum = 0.0, itSum = 0.0;
    for (int run = 0; run < RUNS; ++run) {
        Stopwatch sw;
        const RunResult r = method(run);
        const double ms = sw.elapsedSeconds() * 1000.0;
        const long long score = evaluateTour(r.tour, d, costs);
        if (r.tour.size() != tourSize(d.size())) std::cerr << "INVALID TOUR SIZE\n";
        sum += static_cast<double>(score);
        timeSum += ms;
        itSum += static_cast<double>(r.iterations);
        s.timeMinMs = std::min(s.timeMinMs, ms);
        s.timeMaxMs = std::max(s.timeMaxMs, ms);
        s.itMin = std::min(s.itMin, r.iterations);
        s.itMax = std::max(s.itMax, r.iterations);
        if (score < s.min) {
            s.min = score;
            s.best = r.tour;
        }
        s.max = std::max(s.max, score);
    }
    s.avg = sum / RUNS;
    s.timeAvgMs = timeSum / RUNS;
    s.itAvg = itSum / RUNS;
    return s;
}

void runInstance(const std::string &instanceName) {
    const std::string dataPath = "0_setup/" + instanceName + "_data.csv";
    const std::vector<Point> points = readPoints(dataPath);
    if (points.empty()) {
        std::cerr << "No points read from " << dataPath << "\n";
        return;
    }
    const DistanceMatrix distances = buildDistanceMatrix(points);
    const std::vector<int> costs = extractCosts(points);
    std::cout << "Instance " << instanceName << ": " << points.size() << " points, tour size "
              << tourSize(points.size()) << std::endl;

    SteepestLS ls(distances, costs);
    // Every method gets its own reproducible random stream per run.
    auto seeded = [&](int methodIndex, int run) {
        return std::mt19937(static_cast<unsigned>(1000003 * (instanceName[0] - 'A' + 1) + 1009 * methodIndex + run));
    };

    struct Method {
        std::string key, label;
        Stats stats;
    };
    std::vector<Method> methods;
    auto report = [&](const Method &m) {
        const Stats &s = m.stats;
        std::cout << "  " << m.label << ": avg=" << s.avg << " (" << s.min << " - " << s.max << "), "
                  << s.timeAvgMs << " ms, " << s.itAvg << " iterations" << std::endl;
    };

    methods.push_back({"msls", "MSLS (200 local searches)", runMany(distances, costs, [&](int run) {
                           std::mt19937 rng = seeded(0, run);
                           return msls(MSLS_STARTS, distances, costs, ls, rng);
                       })});
    report(methods.back());
    // Time budget of ILS and LNS: the average running time of MSLS on this instance.
    const double budget = methods.back().stats.timeAvgMs / 1000.0;

    methods.push_back({"ils", "ILS (perturbation: 10 random moves)", runMany(distances, costs, [&](int run) {
                           std::mt19937 rng = seeded(1, run);
                           return ils(budget, distances, costs, ls, rng);
                       })});
    report(methods.back());

    const std::vector<std::tuple<Destroy, std::string, std::string>> destroys = {
        {Destroy::RandomNodes, "nodes", "random nodes"},
        {Destroy::SingleSubpath, "path", "single subpath"},
        {Destroy::MultiSubpath, "paths", "3-5 subpaths"},
    };
    int index = 2;
    for (bool useLS : {false, true}) {
        for (const auto &[kind, key, name] : destroys) {
            const int mi = index++;
            methods.push_back({"lns_" + key + (useLS ? "_ls" : ""),
                               "LNS, " + name + (useLS ? ", with LS" : ", no LS"),
                               runMany(distances, costs, [&, kind = kind, useLS](int run) {
                                   std::mt19937 rng = seeded(mi, run);
                                   return lns(budget, kind, useLS, distances, costs, ls, rng);
                               })});
            report(methods.back());
        }
    }

    std::ofstream summary(std::string(RESULTS_DIR) + "/" + instanceName + "_summary.csv");
    summary << "instance,method,label,min,max,avg,time_min_ms,time_max_ms,time_avg_ms,iter_min,iter_max,iter_avg\n";
    summary << std::fixed;
    for (const auto &m : methods) {
        const Stats &s = m.stats;
        summary << std::setprecision(1) << instanceName << "," << m.key << ",\"" << m.label << "\"," << s.min << ","
                << s.max << "," << s.avg << "," << s.timeMinMs << "," << s.timeMaxMs << "," << s.timeAvgMs << ","
                << s.itMin << "," << s.itMax << "," << s.itAvg << "\n";
        writeTour(std::string(RESULTS_DIR) + "/" + instanceName + "_" + m.key + "_best.txt", s.best);
    }
}

int main() {
    std::filesystem::create_directories(RESULTS_DIR);
    for (const std::string &instance : {"A", "B"}) runInstance(instance);
    return 0;
}
