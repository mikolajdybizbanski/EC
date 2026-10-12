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
 
constexpr char CSV_DELIMITER = ';';
constexpr const char *RESULTS_DIR = "1_greedy/results";
 
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
 
Tour randomSolution(size_t numPoints, std::mt19937 &rng) {
    const size_t size = tourSize(numPoints);
    std::vector<int> candidates(numPoints);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::shuffle(candidates.begin(), candidates.end(), rng);
    return Tour(candidates.begin(), candidates.begin() + size);
}
 
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
 
            {
                const long long delta = distances[c][tour.front()] + costs[c];
                if (delta < bestDelta) {
                    bestDelta = delta;
                    bestNode = static_cast<int>(c);
                    bestPos = 0;
                }
            }
            {
                const long long delta = distances[tour.back()][c] + costs[c];
                if (delta < bestDelta) {
                    bestDelta = delta;
                    bestNode = static_cast<int>(c);
                    bestPos = tour.size();
                }
            }
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
 
Tour greedyCycle(const DistanceMatrix &distances, const std::vector<int> &costs, int start) {
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
 
struct Stats {
    long long min = std::numeric_limits<long long>::max();
    long long max = std::numeric_limits<long long>::min();
    double avg = 0.0;
    Tour best;
};
 
void writeTour(const std::string &path, const Tour &tour) {
    std::ofstream out(path);
    for (int idx : tour) out << idx << "\n";
}
 
template <typename Method>
Stats runFromEveryNode(size_t numPoints, const DistanceMatrix &distances, const std::vector<int> &costs,
                        Method method) {
    Stats stats;
    double sum = 0.0;
    for (size_t start = 0; start < numPoints; ++start) {
        const Tour tour = method(static_cast<int>(start));
        const long long score = evaluateTour(tour, distances, costs);
        sum += static_cast<double>(score);
        if (score < stats.min) {
            stats.min = score;
            stats.best = tour;
        }
        if (score > stats.max) stats.max = score;
    }
    stats.avg = sum / static_cast<double>(numPoints);
    return stats;
}
 
Stats runRandom(size_t numPoints, const DistanceMatrix &distances, const std::vector<int> &costs, int iterations,
                 std::mt19937 &rng) {
    Stats stats;
    double sum = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const Tour tour = randomSolution(numPoints, rng);
        const long long score = evaluateTour(tour, distances, costs);
        sum += static_cast<double>(score);
        if (score < stats.min) {
            stats.min = score;
            stats.best = tour;
        }
        if (score > stats.max) stats.max = score;
    }
    stats.avg = sum / static_cast<double>(iterations);
    return stats;
}
 
void runInstance(const std::string &instanceName, std::mt19937 &rng) {
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
 
    struct Method {
        std::string key;
        std::string label;
        Stats stats;
        double seconds = 0.0;
    };
    std::vector<Method> methods = {
        {"random", "Random solution"},
        {"nn_end", "Nearest neighbor (end only)"},
        {"nn_any", "Nearest neighbor (any position)"},
        {"greedy_cycle", "Greedy cycle"},
    };
 
    stopwatch.restart();
    methods[0].stats = runRandom(n, distances, costs, 200, rng);
    methods[0].seconds = stopwatch.elapsedSeconds();
 
    stopwatch.restart();
    methods[1].stats = runFromEveryNode(n, distances, costs, [&](int start) {
        return nearestNeighborEnd(distances, costs, start);
    });
    methods[1].seconds = stopwatch.elapsedSeconds();
 
    stopwatch.restart();
    methods[2].stats = runFromEveryNode(n, distances, costs, [&](int start) {
        return nearestNeighborAnyPosition(distances, costs, start);
    });
    methods[2].seconds = stopwatch.elapsedSeconds();
 
    stopwatch.restart();
    methods[3].stats = runFromEveryNode(n, distances, costs, [&](int start) {
        return greedyCycle(distances, costs, start);
    });
    methods[3].seconds = stopwatch.elapsedSeconds();
 
    std::filesystem::create_directories(RESULTS_DIR);
 
    std::ofstream summary(std::string(RESULTS_DIR) + "/" + instanceName + "_summary.csv", std::ios::app);
    if (summary.tellp() == 0) summary << "instance,method,min,max,avg,seconds\n";
 
    for (const auto &m : methods) {
        std::cout << "  " << m.label << ": min=" << m.stats.min << " max=" << m.stats.max
                  << " avg=" << m.stats.avg << " (" << m.seconds << " s)\n";
        summary << instanceName << "," << m.key << "," << m.stats.min << "," << m.stats.max << ","
                << m.stats.avg << "," << m.seconds << "\n";
        writeTour(std::string(RESULTS_DIR) + "/" + instanceName + "_" + m.key + "_best.txt", m.stats.best);
    }
}
 
int main() {
    std::filesystem::create_directories(RESULTS_DIR);
    std::mt19937 rng(42);
 
    for (const std::string &instance : {"A", "B"}) {
        std::ofstream(std::string(RESULTS_DIR) + "/" + instance + "_summary.csv", std::ios::trunc);
    }
 
    for (const std::string &instance : {"A", "B"}) {
        runInstance(instance, rng);
    }
    return 0;
}