#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Settings ----------------------------------------------------------------------
constexpr char CSV_DELIMITER = ';';
constexpr const char *RESULTS_DIR = "1_candidates/results";
constexpr int NUM_RUNS = 200;
constexpr int DEFAULT_CANDIDATES = 10;
constexpr unsigned BASE_SEED = 12345;

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

using Tour = std::vector<int>;
using DistanceMatrix = std::vector<std::vector<int>>;
using CandidateLists = std::vector<std::vector<int>>;

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

size_t tourSize(size_t numPoints) { return (numPoints + 1) / 2; }

long long evaluateTour(const Tour &tour, const DistanceMatrix &distances, const std::vector<int> &costs) {
    if (tour.empty()) return 0;

    long long total = 0;
    for (size_t i = 0; i < tour.size(); ++i) {
        const int from = tour[i];
        const int to = tour[(i + 1) % tour.size()];
        total += distances[from][to] + costs[from];
    }
    return total;
}

// Starting solutions ------------------------------------------------------------
Tour randomTour(size_t numPoints, std::mt19937 &rng) {
    Tour all(numPoints);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(tourSize(numPoints));
    return all;
}

// Candidate edges ---------------------------------------------------------------
// For every node a: the k nodes b != a with the smallest distances[a][b] + costs[b].
CandidateLists buildCandidateLists(const DistanceMatrix &distances, const std::vector<int> &costs, int k) {
    const size_t n = distances.size();
    CandidateLists lists(n);
    for (size_t a = 0; a < n; ++a) {
        std::vector<int> others;
        others.reserve(n - 1);
        for (size_t b = 0; b < n; ++b) {
            if (b != a) others.push_back(static_cast<int>(b));
        }
        const size_t take = std::min(static_cast<size_t>(k), others.size());
        std::partial_sort(others.begin(), others.begin() + take, others.end(), [&](int x, int y) {
            const int kx = distances[a][x] + costs[x];
            const int ky = distances[a][y] + costs[y];
            return kx != ky ? kx < ky : x < y;
        });
        others.resize(take);
        lists[a] = std::move(others);
    }
    return lists;
}

// Local search ------------------------------------------------------------------
// Neighborhood: intra-route two-edges exchange (2-opt) + inter-route exchange of a
// selected node with an unselected one. Steepest version: the best improving move
// in the whole (or candidate) neighborhood is applied in each iteration.
struct Move {
    enum class Type { None, TwoOpt, Exchange } type = Type::None;
    int first = 0;   // 2-opt: position of the first removed edge (tour[first], tour[first+1]); exchange: position
    int second = 0;  // 2-opt: position of the second removed edge; exchange: node inserted at `first`
    long long delta = 0;
};

class LocalSearch {
public:
    LocalSearch(const DistanceMatrix &distances, const std::vector<int> &costs)
        : distances_(distances), costs_(costs) {}

    // Baseline: evaluates every move of the neighborhood.
    Tour steepestFull(Tour tour) {
        reset(tour);
        while (true) {
            Move best;
            const int n = static_cast<int>(tour_.size());
            for (int x = 0; x < n; ++x) {
                for (int y = x + 2; y < n; ++y) considerTwoOpt(x, y, best);
            }
            for (int k = 0; k < n; ++k) {
                for (int v = 0; v < static_cast<int>(pos_.size()); ++v) {
                    if (pos_[v] < 0) considerExchange(k, v, best);
                }
            }
            if (best.type == Move::Type::None) break;
            apply(best);
        }
        return tour_;
    }

    // Candidate moves: only moves that introduce at least one candidate edge (a, b),
    // b in candidates[a], are generated - no move outside this set is ever evaluated.
    Tour steepestCandidates(Tour tour, const CandidateLists &candidates) {
        reset(tour);
        const int n = static_cast<int>(tour_.size());
        while (true) {
            Move best;
            for (int i = 0; i < n; ++i) {
                const int a = tour_[i];
                for (int b : candidates[a]) {
                    if (pos_[b] >= 0) {
                        const int j = pos_[b];
                        // Two 2-opt moves add edge (a, b): with the successors or with the predecessors.
                        considerTwoOpt(i, j, best);
                        considerTwoOpt(prev(i), prev(j), best);
                    } else {
                        // Two exchanges add edge (a, b): b replaces the successor or the predecessor of a.
                        considerExchange(next(i), b, best);
                        considerExchange(prev(i), b, best);
                    }
                }
            }
            if (best.type == Move::Type::None) break;
            apply(best);
        }
        return tour_;
    }

    // Self-check: compares the delta of every 2-opt and exchange move on `tour` with a full
    // re-evaluation of the tour after applying the move. Returns {moves checked, mismatches}.
    std::pair<long long, long long> verifyDeltas(const Tour &tour) {
        reset(tour);
        const long long base = evaluateTour(tour_, distances_, costs_);
        const int n = static_cast<int>(tour_.size());
        long long checked = 0, mismatches = 0;

        auto check = [&](const Move &move) {
            if (move.type == Move::Type::None) return;
            LocalSearch copy(distances_, costs_);
            copy.reset(tour);
            copy.apply(move);
            ++checked;
            if (evaluateTour(copy.tour_, distances_, costs_) - base != move.delta) ++mismatches;
        };
        constexpr long long ANY = std::numeric_limits<long long>::max();  // accept every move
        for (int x = 0; x < n; ++x) {
            for (int y = x + 1; y < n; ++y) {
                Move move;
                move.delta = ANY;
                considerTwoOpt(x, y, move);
                check(move);
            }
        }
        for (int k = 0; k < n; ++k) {
            for (int v = 0; v < static_cast<int>(pos_.size()); ++v) {
                if (pos_[v] >= 0) continue;
                Move move;
                move.delta = ANY;
                considerExchange(k, v, move);
                check(move);
            }
        }
        return {checked, mismatches};
    }

private:
    const DistanceMatrix &distances_;
    const std::vector<int> &costs_;
    Tour tour_;
    std::vector<int> pos_;  // position of each node in tour_, -1 if not selected

    void reset(const Tour &tour) {
        tour_ = tour;
        pos_.assign(distances_.size(), -1);
        for (size_t i = 0; i < tour_.size(); ++i) pos_[tour_[i]] = static_cast<int>(i);
    }

    int next(int i) const { return i + 1 == static_cast<int>(tour_.size()) ? 0 : i + 1; }
    int prev(int i) const { return i == 0 ? static_cast<int>(tour_.size()) - 1 : i - 1; }

    // Removes edges (tour[x], tour[x+1]) and (tour[y], tour[y+1]), adds (tour[x], tour[y]) and (tour[x+1], tour[y+1]).
    void considerTwoOpt(int x, int y, Move &best) const {
        if (x > y) std::swap(x, y);
        const int n = static_cast<int>(tour_.size());
        if (y - x < 2 || (x == 0 && y == n - 1)) return;  // edges share a node

        const int a = tour_[x], an = tour_[x + 1];
        const int b = tour_[y], bn = tour_[next(y)];
        const long long delta = distances_[a][b] + distances_[an][bn] - distances_[a][an] - distances_[b][bn];
        if (delta < best.delta) best = {Move::Type::TwoOpt, x, y, delta};
    }

    // Replaces the selected node at position k with the unselected node v.
    void considerExchange(int k, int v, Move &best) const {
        const int u = tour_[k];
        const int p = tour_[prev(k)], q = tour_[next(k)];
        const long long delta = costs_[v] - costs_[u] + distances_[p][v] + distances_[v][q] -
                                distances_[p][u] - distances_[u][q];
        if (delta < best.delta) best = {Move::Type::Exchange, k, v, delta};
    }

    void apply(const Move &move) {
        if (move.type == Move::Type::TwoOpt) {
            std::reverse(tour_.begin() + move.first + 1, tour_.begin() + move.second + 1);
            for (int i = move.first + 1; i <= move.second; ++i) pos_[tour_[i]] = i;
        } else {
            pos_[tour_[move.first]] = -1;
            tour_[move.first] = move.second;
            pos_[move.second] = move.first;
        }
    }
};

// Experiment driver ---------------------------------------------------------------
struct Stats {
    long long min = std::numeric_limits<long long>::max();
    long long max = std::numeric_limits<long long>::min();
    double avg = 0.0;
    double minMs = std::numeric_limits<double>::max();
    double maxMs = 0.0;
    double avgMs = 0.0;
    Tour best;
};

void writeTour(const std::string &path, const Tour &tour) {
    std::ofstream out(path);
    for (int idx : tour) out << idx << "\n";
}

template <typename Method>
Stats runOnStarts(const std::vector<Tour> &starts, const DistanceMatrix &distances, const std::vector<int> &costs,
                  Method method) {
    Stats stats;
    double sum = 0.0, sumMs = 0.0;
    Stopwatch stopwatch;
    for (const Tour &start : starts) {
        stopwatch.restart();
        const Tour tour = method(start);
        const double ms = stopwatch.elapsedSeconds() * 1000.0;
        const long long score = evaluateTour(tour, distances, costs);

        sum += static_cast<double>(score);
        sumMs += ms;
        if (score < stats.min) {
            stats.min = score;
            stats.best = tour;
        }
        stats.max = std::max(stats.max, score);
        stats.minMs = std::min(stats.minMs, ms);
        stats.maxMs = std::max(stats.maxMs, ms);
    }
    stats.avg = sum / static_cast<double>(starts.size());
    stats.avgMs = sumMs / static_cast<double>(starts.size());
    return stats;
}

void runInstance(const std::string &instanceName) {
    const std::string dataPath = "0_setup/" + instanceName + "_data.csv";

    Stopwatch stopwatch;
    const std::vector<Point> points = readPoints(dataPath);
    if (points.empty()) {
        std::cerr << "No points read from " << dataPath << "\n";
        return;
    }

    const DistanceMatrix distances = buildDistanceMatrix(points);
    const std::vector<int> costs = extractCosts(points);
    const size_t n = points.size();
    std::cout << "Instance " << instanceName << ": " << n << " points, tour size "
              << tourSize(n) << " (setup took " << stopwatch.elapsedSeconds() << " s)\n";

    // The same 200 random starting solutions are given to every method.
    std::mt19937 rng(BASE_SEED + static_cast<unsigned>(instanceName[0]));
    std::vector<Tour> starts;
    for (int r = 0; r < NUM_RUNS; ++r) starts.push_back(randomTour(n, rng));

    LocalSearch search(distances, costs);

    long long checked = 0, mismatches = 0;
    for (int r = 0; r < 5; ++r) {
        const auto [c, m] = search.verifyDeltas(starts[r]);
        checked += c;
        mismatches += m;
    }
    std::cout << "  Delta self-check: " << checked << " moves, " << mismatches << " mismatches\n";
    std::filesystem::create_directories(RESULTS_DIR);
    std::ofstream(std::string(RESULTS_DIR) + "/" + instanceName + "_delta_check.txt")
        << checked << " moves, " << mismatches << " mismatches\n";

    struct Method {
        std::string key;
        std::string label;
        int candidates;  // 0 = full neighborhood (baseline)
        Stats stats{};
        double precomputeMs = 0.0;  // building candidate lists, excluded from the per-run times
    };

    std::vector<Method> methods = {
        {"steepest_full", "Steepest LS, full neighborhood (baseline)", 0, {}},
        {"steepest_cand_10", "Steepest LS, candidate moves (k=10)", DEFAULT_CANDIDATES, {}},
        {"steepest_cand_5", "Steepest LS, candidate moves (k=5)", 5, {}},
        {"steepest_cand_15", "Steepest LS, candidate moves (k=15)", 15, {}},
        {"steepest_cand_20", "Steepest LS, candidate moves (k=20)", 20, {}},
    };

    for (Method &m : methods) {
        if (m.candidates == 0) {
            m.stats = runOnStarts(starts, distances, costs, [&](const Tour &start) {
                return search.steepestFull(start);
            });
        } else {
            stopwatch.restart();
            const CandidateLists candidates = buildCandidateLists(distances, costs, m.candidates);
            m.precomputeMs = stopwatch.elapsedSeconds() * 1000.0;
            m.stats = runOnStarts(starts, distances, costs, [&](const Tour &start) {
                return search.steepestCandidates(start, candidates);
            });
        }
    }

    std::filesystem::create_directories(RESULTS_DIR);

    std::ofstream summary(std::string(RESULTS_DIR) + "/" + instanceName + "_summary.csv", std::ios::trunc);
    summary << "instance,method,min,max,avg,min_ms,max_ms,avg_ms,precompute_ms\n";

    for (const auto &m : methods) {
        std::cout << "  " << m.label << ": min=" << m.stats.min << " max=" << m.stats.max
                  << " avg=" << m.stats.avg << " | time per run [ms] min=" << m.stats.minMs
                  << " max=" << m.stats.maxMs << " avg=" << m.stats.avgMs << "\n";
        summary << instanceName << "," << m.key << "," << m.stats.min << "," << m.stats.max << ","
                << m.stats.avg << "," << m.stats.minMs << "," << m.stats.maxMs << "," << m.stats.avgMs << "," << m.precomputeMs << "\n";
        writeTour(std::string(RESULTS_DIR) + "/" + instanceName + "_" + m.key + "_best.txt", m.stats.best);
    }
}

int main() {
    for (const std::string instance : {"A", "B"}) {
        runInstance(instance);
    }
    return 0;
}
