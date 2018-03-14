#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <limits>

/**
 * K-Mean Clustering Algorithm
 * @author Mohammad_ZD
 */

class Point {
public:
    std::string name; // Assuming it is unique
    int cluster_id; // Cluster id where this point belong
    std::vector<long double> values; // Values of this point (x, y, ...)

    Point(std::string name, std::vector<long double> values) {
        this->name = name;
        this->values = values;
        cluster_id = -1; // no cluster yet (default)
    }
};

class Cluster {
public:
    int cluster_id; // Cluster Id
    std::vector<long double> centroid_point; // values of the centroid of this cluster
    std::vector<Point> points; // points that are in this cluster

    Cluster(int cluster_id, std::vector<long double> centroid_point) {
        this->cluster_id = cluster_id;
        this->centroid_point = centroid_point;
    }

    // Adds point to the cluster
    void add_point(Point p) {
        points.push_back(p);
    }

    // Removes point from the cluster
    void remove_point(const Point& p) {
        for (int i = 0; i < points.size(); ++i) {
            if (points[i].name == p.name) {
                points.erase(points.begin() + i);
                break;
            }
        }
    }
};

class KMeanClustering {
public:
    int id = 0; // global id
    int k;
    std::vector<Point> points;
    std::vector<Cluster> clusters;

    KMeanClustering(int k, std::vector<Point> points) {
        this->k = k;
        this->points = points;
    }

    // Returns the id of the nearest cluster to Point p
    int get_nearest_cluster(Point p) {
        long double min_dist = std::numeric_limits<long double>::max();
        int nearest_cluster_id = -1;
        for (int i = 0; i < k; ++i) {
            long double sum = 0.0;
            for (int j = 0; j < p.values.size(); ++j) {
                sum += (p.values[j] - clusters[i].centroid_point[j])
                     * (p.values[j] - clusters[i].centroid_point[j]);
            }
            long double dist = std::sqrt(sum);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_cluster_id = i;
            }
        }
        return nearest_cluster_id;
    }

    // Initializes the clusters
    void init_clusters() {
        for (int i = 0; i < k; ++i) {
            clusters.push_back({id++, points[i].values});
        }
    }

    // Runs K-mean Clustering Algorithm
    void run(int max_iterations) {
        init_clusters();
        int iterations = 0;
        while (iterations <= max_iterations) {
            bool change = false;
            for (int i = 0; i < points.size(); ++i) {
                auto& p = points[i];
                const int old_nearest_cluster_id = p.cluster_id;
                const int new_nearest_clusted_id = get_nearest_cluster(p);
                if (new_nearest_clusted_id != -1) {
                    if (old_nearest_cluster_id != -1) {
                        clusters[old_nearest_cluster_id].remove_point(p);
                    }
                    p.cluster_id = new_nearest_clusted_id;
                    clusters[new_nearest_clusted_id].add_point(p);
                    change = true;
                }
            }
            for (int i = 0; i < k; ++i) {
                auto& cluster = clusters[i];
                const int cluster_size = cluster.points.size();
                if (cluster_size == 0) continue;
                long double total_x = 0.0, total_y = 0.0;
                for (int l = 0; l < cluster_size; ++l) {
                    total_x += cluster.points[l].values[0];
                    total_y += cluster.points[l].values[1];
                }
                cluster.centroid_point = {total_x/cluster_size, total_y/cluster_size};
            }
            if (!change) break;
            iterations++;
        }
    }

    // Prints the final clusters
    void print_result() {
        for (const auto& c: clusters) {
            std::cout << "Cluster id: " << c.cluster_id << '\n';
            std::cout << "Centroid coordinates: (" << c.centroid_point[0] << ", " << c.centroid_point[1] << ")\n";
            std::cout << "Points:\n";
            for (const auto& p: c.points) {
                std::cout << p.name << " (" << p.values[0] << ", " << p.values[1] << ")\n";
            }
            std::cout << '\n';
        }
    }
};

int main() {
    std::ifstream in("kmeans_input.in");

    // Read input
    int num_of_points, dim; in >> num_of_points >> dim;
    std::vector<Point> points;
    for (int i = 0; i < num_of_points; ++i) {
        std::vector<long double> data(dim, 0);
        std::string name; in >> name;
        for (int j = 0; j < dim; ++j) {
            in >> data[j];
        }
        Point p = {name, data};
        points.push_back(p);
    }

    KMeanClustering kmean = {3, points};
    kmean.run(1000);
    kmean.print_result();

    return 0;
}
