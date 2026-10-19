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
constexpr const char *RESULTS_DIR = "1_greedy/results";

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

// Algorithms --------------------------------------------------------------------

Tour weightedRegretCycle(const DistanceMatrix &distances, const std::vector<int> &costs, int start, double weight) {
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
        double bestScore = -std::numeric_limits<double>::infinity();
        long long bestNodeDelta = std::numeric_limits<long long>::max();

        for (size_t c = 0; c < numPoints; ++c) {
            if (used[c]) continue;

            long long delta1 = std::numeric_limits<long long>::max();
            long long delta2 = std::numeric_limits<long long>::max();
            size_t pos1 = 0;

            for (size_t i = 0; i < tour.size(); ++i) {
                const int prev = tour[i];
                const int next = tour[(i + 1) % tour.size()];
                const long long delta = distances[prev][c] + distances[c][next] - distances[prev][next] + costs[c];

                if (delta < delta1) {
                    delta2 = delta1;
                    delta1 = delta;
                    pos1 = i + 1;
                } else if (delta < delta2) {
                    delta2 = delta;
                }
            }

            long long regret = delta2 - delta1;
            double score = weight * static_cast<double>(regret) - (1.0 - weight) * static_cast<double>(delta1);

            if (score > bestScore || (score == bestScore && delta1 < bestNodeDelta)) {
                bestScore = score;
                bestNode = static_cast<int>(c);
                bestPos = pos1;
                bestNodeDelta = delta1;
            }
        }

        tour.insert(tour.begin() + bestPos, bestNode);
        used[bestNode] = true;
    }
    return tour;
}

// Experiment driver ---------------------------------------------------------------
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
Stats runFromEveryNode(size_t numPoints, const DistanceMatrix &distances, const std::vector<int> &costs, Method method) {
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

    struct Method {
        std::string key;
        std::string label;
        Stats stats;
        double seconds = 0.0;
    };
    
    std::vector<Method> methods = {
        {"regret_1_0", "Greedy 2-regret (W=1.0)"},
        {"regret_0_5", "Weighted Regret (W=0.5)"},
    };

    stopwatch.restart();
    methods[0].stats = runFromEveryNode(n, distances, costs, [&](int start) {
        return weightedRegretCycle(distances, costs, start, 1.0); 
    });
    methods[0].seconds = stopwatch.elapsedSeconds();

    stopwatch.restart();
    methods[1].stats = runFromEveryNode(n, distances, costs, [&](int start) {
        return weightedRegretCycle(distances, costs, start, 0.5); 
    });
    methods[1].seconds = stopwatch.elapsedSeconds();

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

    for (const std::string &instance : {"A", "B"}) {
        std::ofstream(std::string(RESULTS_DIR) + "/" + instance + "_summary.csv", std::ios::trunc);
    }

    for (const std::string &instance : {"A", "B"}) {
        runInstance(instance);
    }
    return 0;
}