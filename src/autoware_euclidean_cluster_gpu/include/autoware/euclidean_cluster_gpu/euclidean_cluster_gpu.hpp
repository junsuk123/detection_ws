// Copyright 2025
// Apache-2.0

#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>

namespace autoware::euclidean_cluster_gpu {

struct Float3 { float x, y, z; };

struct ClusteringParams {
  float voxel_size = 0.3f;        // meters
  float tolerance = 0.7f;         // neighbor distance threshold (m)
  int min_cluster_size = 10;
  int max_cluster_size = 100000;
  int max_points = 2'000'000;     // safety upper bound
  int max_iterations = 10;        // label propagation iterations
};

// Runs on GPU: assigns a cluster label to each point (-1 for noise)
void run_euclidean_clustering_cuda(
  const Float3* d_points,     // device pointer with N points
  int N,
  const ClusteringParams& params,
  int32_t* d_labels,          // device output labels (size N)
  cudaStream_t stream = 0);

// Host wrapper: uploads points and returns labels (host vectors). Useful for tests.
std::vector<int32_t> cluster_host(
  const std::vector<Float3>& pts,
  const ClusteringParams& params);

} // namespace
