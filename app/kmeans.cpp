#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <random>
#include <limits>

//g++ -O2 -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` kmeans.cpp -o kmeans_cpp`python3.12-config --extension-suffix`
//c++ -O3 -Wall -shared -std=c++17 -fPIC `python3.12 -m pybind11 --includes` kmeans.cpp -o kmeans_cpp`python3.12-config --extension-suffix`

namespace py = pybind11;

// Calcular la distancia euclidiana entre dos puntos
double euclidean_distance(const std::vector<double>& point1, const std::vector<double>& point2) {
    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return std::sqrt(sum);
}

// Asignar cada punto al centroide más cercano
std::vector<int> assign_points_to_centroids(const std::vector<std::vector<double>>& data, 
                                            const std::vector<std::vector<double>>& centroids) {
    std::vector<int> assignments(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        double min_distance = std::numeric_limits<double>::max();
        int best_centroid = -1;
        for (size_t j = 0; j < centroids.size(); ++j) {
            double distance = euclidean_distance(data[i], centroids[j]);
            if (distance < min_distance) {
                min_distance = distance;
                best_centroid = j;
            }
        }
        assignments[i] = best_centroid;
    }
    return assignments;
}

// Recalcular los centroides
std::vector<std::vector<double>> recompute_centroids(const std::vector<std::vector<double>>& data, 
                                                     const std::vector<int>& assignments, 
                                                     int n_clusters) {
    std::vector<std::vector<double>> centroids(n_clusters, std::vector<double>(data[0].size(), 0.0));
    std::vector<int> counts(n_clusters, 0);

    for (size_t i = 0; i < data.size(); ++i) {
        int cluster = assignments[i];
        for (size_t j = 0; j < data[i].size(); ++j) {
            centroids[cluster][j] += data[i][j];
        }
        counts[cluster] += 1;
    }

    for (int i = 0; i < n_clusters; ++i) {
        if (counts[i] > 0) {
            for (size_t j = 0; j < centroids[i].size(); ++j) {
                centroids[i][j] /= counts[i];
            }
        }
    }

    return centroids;
}

// Función principal de K-Means
std::tuple<std::vector<std::vector<double>>, std::vector<int>> kmeans(const std::vector<std::vector<double>>& data, 
                                                                      int n_clusters, 
                                                                      int n_iters) {
    // Inicialización aleatoria de los centroides
    std::vector<std::vector<double>> centroids;
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(0, data.size() - 1);

    for (int i = 0; i < n_clusters; ++i) {
        centroids.push_back(data[distribution(generator)]);
    }

    std::vector<int> assignments;

    for (int iter = 0; iter < n_iters; ++iter) {
        assignments = assign_points_to_centroids(data, centroids);
        centroids = recompute_centroids(data, assignments, n_clusters);
    }

    return {centroids, assignments};
}

// Exponer las funciones a Python
PYBIND11_MODULE(kmeans_cpp, m) {
    m.def("kmeans", &kmeans, "K-Means Clustering",
          py::arg("data"), py::arg("n_clusters"), py::arg("n_iters") = 10);
}
