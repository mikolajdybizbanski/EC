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
constexpr const char *RESULTS_DIR = "6_msls_ils/results";

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


// Perturbation for ILS ------------------------------------------------------------
// Two random changes, one per part of the solution, so that the next local search
// cannot simply undo them with a single move:
//   1. double bridge on the cycle: cut it into four segments S1 S2 S3 S4 (three random
//      cut points) and reconnect them as S1 S3 S2 S4. This replaces three edges at once
//      and keeps the direction of every segment; 2-opt cannot reverse it in one step.
//   2. `replacements` random selected nodes are swapped for random unselected nodes, each
//      inserted at the position of the node it replaces (changes the selected set).
struct Perturbation {
    bool doubleBridge = true;
    int replacements = 4;
};

Tour perturb(const Tour &tour, size_t numPoints, const Perturbation &p, std::mt19937 &rng) {
    const int k = static_cast<int>(tour.size());
    Tour result = tour;

    if (p.doubleBridge && k >= 8) {
        std::uniform_int_distribution<int> cut(1, k - 1);
        int c[3];
        do {
            c[0] = cut(rng), c[1] = cut(rng), c[2] = cut(rng);
            std::sort(c, c + 3);
        } while (c[0] == c[1] || c[1] == c[2]);
        result.clear();
        result.insert(result.end(), tour.begin(), tour.begin() + c[0]);         // S1
        result.insert(result.end(), tour.begin() + c[1], tour.begin() + c[2]);  // S3
        result.insert(result.end(), tour.begin() + c[0], tour.begin() + c[1]);  // S2
        result.insert(result.end(), tour.begin() + c[2], tour.end());           // S4
    }

    std::vector<bool> selected(numPoints, false);
    for (int v : result) selected[v] = true;
    std::vector<int> unselected;
    for (size_t v = 0; v < numPoints; ++v) {
        if (!selected[v]) unselected.push_back(static_cast<int>(v));
    }
    std::vector<int> positions(k);
    std::iota(positions.begin(), positions.end(), 0);
    std::shuffle(positions.begin(), positions.end(), rng);
    std::shuffle(unselected.begin(), unselected.end(), rng);
    const int r = std::min<int>({p.replacements, k, static_cast<int>(unselected.size())});
    for (int i = 0; i < r; ++i) result[positions[i]] = unselected[i];
    return result;
}

// MSLS and ILS --------------------------------------------------------------------
struct RunResult {
    Tour tour;
    long long score = 0;
    double ms = 0.0;
    long long lsRuns = 0;
};

// Multiple start local search: `iterations` steepest LS runs from random solutions,
// the result is the best local optimum found.
RunResult msls(int iterations, const DistanceMatrix &distances, const std::vector<int> &costs,
               ListLocalSearch &ls, std::mt19937 &rng) {
    Stopwatch stopwatch;
    RunResult best;
    best.score = std::numeric_limits<long long>::max();
    for (int it = 0; it < iterations; ++it) {
        ListCounters counters;
        Tour tour = ls.run(randomSolution(distances.size(), rng), counters);
        const long long score = evaluateTour(tour, distances, costs);
        if (score < best.score) {
            best.score = score;
            best.tour = std::move(tour);
        }
    }
    best.lsRuns = iterations;
    best.ms = stopwatch.elapsedSeconds() * 1000.0;
    return best;
}

// Iterated local search: x = LS(random); repeat { y = LS(perturb(x)); if f(y) < f(x): x = y }
// until `timeLimitMs` has elapsed (checked after every local search).
RunResult ils(double timeLimitMs, const Perturbation &p, const DistanceMatrix &distances,
              const std::vector<int> &costs, ListLocalSearch &ls, std::mt19937 &rng) {
    Stopwatch stopwatch;
    ListCounters counters;
    RunResult current;
    current.tour = ls.run(randomSolution(distances.size(), rng), counters);
    current.score = evaluateTour(current.tour, distances, costs);
    long long runs = 1;
    while (stopwatch.elapsedSeconds() * 1000.0 < timeLimitMs) {
        Tour candidate = ls.run(perturb(current.tour, distances.size(), p, rng), counters);
        ++runs;
        const long long score = evaluateTour(candidate, distances, costs);
        if (score < current.score) {
            current.score = score;
            current.tour = std::move(candidate);
        }
    }
    current.lsRuns = runs;
    current.ms = stopwatch.elapsedSeconds() * 1000.0;
    return current;
}

// Experiment driver ---------------------------------------------------------------
struct Stats {
    long long min = std::numeric_limits<long long>::max();
    long long max = std::numeric_limits<long long>::min();
    double avg = 0.0;
    double timeMinMs = std::numeric_limits<double>::max(), timeMaxMs = 0.0, timeAvgMs = 0.0;
    long long runsMin = std::numeric_limits<long long>::max(), runsMax = 0;
    double runsAvg = 0.0;
    Tour best;
};

Stats summarize(const std::vector<RunResult> &results) {
    Stats s;
    for (const RunResult &r : results) {
        s.avg += static_cast<double>(r.score);
        s.timeAvgMs += r.ms;
        s.runsAvg += static_cast<double>(r.lsRuns);
        s.timeMinMs = std::min(s.timeMinMs, r.ms);
        s.timeMaxMs = std::max(s.timeMaxMs, r.ms);
        s.runsMin = std::min(s.runsMin, r.lsRuns);
        s.runsMax = std::max(s.runsMax, r.lsRuns);
        if (r.score < s.min) {
            s.min = r.score;
            s.best = r.tour;
        }
        s.max = std::max(s.max, r.score);
    }
    const double n = static_cast<double>(results.size());
    s.avg /= n;
    s.timeAvgMs /= n;
    s.runsAvg /= n;
    return s;
}

void writeTour(const std::string &path, const Tour &tour) {
    std::ofstream out(path);
    for (int idx : tour) out << idx << "\n";
}

struct Instance {
    std::string name;
    DistanceMatrix distances;
    std::vector<int> costs;
};

bool loadInstance(const std::string &name, Instance &inst) {
    const std::string dataPath = "0_setup/" + name + "_data.csv";
    const std::vector<Point> points = readPoints(dataPath);
    if (points.empty()) {
        std::cerr << "No points read from " << dataPath << "\n";
        return false;
    }
    inst.name = name;
    inst.distances = buildDistanceMatrix(points);
    inst.costs = extractCosts(points);
    return true;
}

constexpr int RUNS = 20;
constexpr int MSLS_ITERATIONS = 400;
const Perturbation ILS_PERTURBATION{true, 4};

void runInstance(const Instance &inst, std::mt19937 &rng) {
    std::cout << "Instance " << inst.name << ": " << inst.distances.size() << " points, tour size "
              << tourSize(inst.distances.size()) << "\n";
    ListLocalSearch ls(inst.distances, inst.costs);

    std::vector<RunResult> mslsResults;
    for (int r = 0; r < RUNS; ++r) {
        mslsResults.push_back(msls(MSLS_ITERATIONS, inst.distances, inst.costs, ls, rng));
        std::cout << "  MSLS run " << r + 1 << ": " << mslsResults.back().score << " ("
                  << mslsResults.back().ms << " ms)\n";
    }
    const Stats mslsStats = summarize(mslsResults);

    // ILS gets exactly the average running time of MSLS on this instance.
    const double timeLimitMs = mslsStats.timeAvgMs;
    std::vector<RunResult> ilsResults;
    for (int r = 0; r < RUNS; ++r) {
        ilsResults.push_back(ils(timeLimitMs, ILS_PERTURBATION, inst.distances, inst.costs, ls, rng));
        std::cout << "  ILS run " << r + 1 << ": " << ilsResults.back().score << " (" << ilsResults.back().lsRuns
                  << " LS runs)\n";
    }
    const Stats ilsStats = summarize(ilsResults);

    struct Method {
        std::string key, label;
        const Stats &stats;
    };
    const std::vector<Method> methods = {
        {"msls", "Multiple start local search (MSLS)", mslsStats},
        {"ils", "Iterated local search (ILS)", ilsStats},
    };

    std::ofstream summary(std::string(RESULTS_DIR) + "/" + inst.name + "_summary.csv");
    summary << "instance,method,label,min,max,avg,time_min_ms,time_max_ms,time_avg_ms,ls_runs_min,ls_runs_max,"
               "ls_runs_avg,time_limit_ms\n";
    summary << std::fixed;
    for (const Method &m : methods) {
        const Stats &s = m.stats;
        std::cout << "  " << m.label << ": min=" << s.min << " max=" << s.max << " avg=" << s.avg << " (avg "
                  << s.timeAvgMs << " ms, avg " << s.runsAvg << " LS runs)\n";
        summary << std::setprecision(1) << inst.name << "," << m.key << ",\"" << m.label << "\"," << s.min << ","
                << s.max << "," << s.avg << "," << std::setprecision(2) << s.timeMinMs << "," << s.timeMaxMs << ","
                << s.timeAvgMs << "," << s.runsMin << "," << s.runsMax << "," << std::setprecision(1) << s.runsAvg
                << "," << std::setprecision(2) << timeLimitMs << "\n";
        writeTour(std::string(RESULTS_DIR) + "/" + inst.name + "_" + m.key + "_best.txt", s.best);
    }

    std::ofstream all(std::string(RESULTS_DIR) + "/" + inst.name + "_runs.csv");
    all << "method,run,objective,time_ms,ls_runs\n" << std::fixed << std::setprecision(2);
    for (int r = 0; r < RUNS; ++r) {
        all << "msls," << r << "," << mslsResults[r].score << "," << mslsResults[r].ms << "," << mslsResults[r].lsRuns << "\n";
    }
    for (int r = 0; r < RUNS; ++r) {
        all << "ils," << r << "," << ilsResults[r].score << "," << ilsResults[r].ms << "," << ilsResults[r].lsRuns << "\n";
    }
}

// Short preliminary experiment used to choose the perturbation: every variant gets
// `runs` ILS runs with the same time limit (a fixed `limitMs`).
void tunePerturbation(const Instance &inst, double limitMs, int runs, std::mt19937 &rng) {
    ListLocalSearch ls(inst.distances, inst.costs);
    std::ofstream out(std::string(RESULTS_DIR) + "/" + inst.name + "_tuning.csv");
    out << "double_bridge,replacements,min,max,avg,ls_runs_avg\n" << std::fixed << std::setprecision(1);
    for (bool bridge : {false, true}) {
        for (int repl : {0, 2, 4, 6, 10}) {
            if (!bridge && repl == 0) continue;
            std::vector<RunResult> results;
            for (int r = 0; r < runs; ++r) {
                results.push_back(ils(limitMs, {bridge, repl}, inst.distances, inst.costs, ls, rng));
            }
            const Stats s = summarize(results);
            out << bridge << "," << repl << "," << s.min << "," << s.max << "," << s.avg << "," << s.runsAvg << "\n";
            std::cout << inst.name << " bridge=" << bridge << " repl=" << repl << ": avg=" << s.avg
                      << " min=" << s.min << " runs=" << s.runsAvg << "\n";
        }
    }
}

int main(int argc, char **argv) {
    std::filesystem::create_directories(RESULTS_DIR);
    // Fixed seed: reproducible runs for the report.
    std::mt19937 rng(42);
    const bool tune = argc > 1 && std::string(argv[1]) == "tune";
    for (const std::string name : {"A", "B"}) {
        Instance inst;
        if (!loadInstance(name, inst)) continue;
        if (tune) {
            tunePerturbation(inst, 1500.0, 5, rng);
        } else {
            runInstance(inst, rng);
        }
    }
    return 0;
}
