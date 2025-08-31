// Copyright 2025
// Apache-2.0
#include <cuda_runtime.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <math.h>
#include <stdio.h>
#include "autoware/euclidean_cluster_gpu/euclidean_cluster_gpu.hpp"

namespace autoware::euclidean_cluster_gpu {

// Helper for error checking
#define CHECK_CUDA_ERROR(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      return; \
    } \
  } while(0)

// 64-bit hash for integer voxel coords (simple mix; sufficient for binning)
__host__ __device__ inline uint64_t hash3(int x, int y, int z) {
  uint64_t h = 1469598103934665603ull;
  h ^= (uint64_t)x + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
  h ^= (uint64_t)y + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
  h ^= (uint64_t)z + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
  return h;
}

struct HashKey {
  uint64_t key;
  int idx;
};

struct HashKeyCompare {
  __host__ __device__ bool operator()(const HashKey& a, const HashKey& b) const {
    return a.key < b.key || (a.key == b.key && a.idx < b.idx);
  }
};

// Compute voxel hash for each point
__global__ void compute_hash_kernel(
  const Float3* pts, int N, float vsize, HashKey* out_keys)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  Float3 p = pts[i];

  // Guard against NaN/Inf – push to a sentinel bin
  if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z) || !(vsize > 0.0f)) {
    out_keys[i].key = 0xFFFFFFFFFFFFFFFFull; // sentinel
    out_keys[i].idx = i;
    return;
  }

  int vx = static_cast<int>(floorf(p.x / vsize));
  int vy = static_cast<int>(floorf(p.y / vsize));
  int vz = static_cast<int>(floorf(p.z / vsize));
  out_keys[i].key = hash3(vx, vy, vz);
  out_keys[i].idx = i;
}

// Build cell starts by detecting boundaries in the sorted key array.
// unique_keys is optional (may be nullptr); cell_starts size is N, num_cells is scalar on device.
__global__ void build_cell_offsets(
  const HashKey* sorted_keys, int N, uint64_t* unique_keys, int* cell_starts, int* num_cells)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  bool is_start = (i == 0) || (sorted_keys[i].key != sorted_keys[i-1].key);
  if (is_start) {
    int cell_id = atomicAdd(num_cells, 1);
    if (unique_keys) unique_keys[cell_id] = sorted_keys[i].key;
    cell_starts[cell_id] = i;
  }
}

// Binary search a key in unique cell keys [0, C)
__device__ int find_cell(uint64_t key, const uint64_t* keys, int C) {
  int lo = 0, hi = C - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    uint64_t mk = keys[mid];
    if (mk < key) lo = mid + 1;
    else if (mk > key) hi = mid - 1;
    else return mid;
  }
  return -1;
}

// Fill cell_ends from cell_starts (CSR-like)
__global__ void fill_cell_ends(const int* cell_starts, int C, int N, int* cell_ends) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= C) return;
  int end = (i+1 < C) ? cell_starts[i+1] : N;
  cell_ends[i] = end;
}

// Label relaxation over neighbor voxels in 3x3x3 window
__global__ void relax_labels(
  const Float3* pts,
  const HashKey* sorted_keys,
  const uint64_t* cell_keys,
  const int* cell_starts,
  const int* cell_ends,
  int N, int C,
  float vsize,
  float tol2,
  int* labels_changed,
  int32_t* labels)
{
  int sidx = blockIdx.x * blockDim.x + threadIdx.x; // index into sorted arrays
  if (sidx >= N) return;

  int i = sorted_keys[sidx].idx;
  Float3 p = pts[i];
  // Skip invalid points (mapped to sentinel cell)
  if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) return;

  int32_t mylabel = labels[i];

  // Recompute voxel coordinate for neighbor search (consistent with hash)
  int vx = static_cast<int>(floorf(p.x / vsize));
  int vy = static_cast<int>(floorf(p.y / vsize));
  int vz = static_cast<int>(floorf(p.z / vsize));

  for (int dx=-1; dx<=1; ++dx) {
    for (int dy=-1; dy<=1; ++dy) {
      for (int dz=-1; dz<=1; ++dz) {
        uint64_t h = hash3(vx+dx, vy+dy, vz+dz);
        int cid = find_cell(h, cell_keys, C);
        if (cid < 0) continue;
        int start = cell_starts[cid];
        int end   = cell_ends[cid];
        for (int k = start; k < end; ++k) {
          int j = sorted_keys[k].idx;
          Float3 q = pts[j];
          float dx_ = p.x - q.x, dy_ = p.y - q.y, dz_ = p.z - q.z;
          if (dx_*dx_ + dy_*dy_ + dz_*dz_ <= tol2) {
            int32_t nl = labels[j];
            if (nl < mylabel) { 
              atomicMin(&labels[i], nl);
              *labels_changed = 1; 
            }
          }
        }
      }
    }
  }
}

// Simplify memory management - avoid global static variables
void run_euclidean_clustering_cuda(
  const Float3* d_points, int N, const ClusteringParams& params, int32_t* d_labels, cudaStream_t stream)
{
  // Validate inputs
  if (N <= 0 || !d_points || !d_labels) {
    printf("Invalid input parameters: N=%d, points=%p, labels=%p\n", N, d_points, d_labels);
    return;
  }

  // Ensure voxel size is reasonable
  float voxel_size = params.voxel_size;
  if (voxel_size <= 0.001f) {
    printf("Warning: voxel_size too small (%.6f), using default 0.2\n", voxel_size);
    voxel_size = 0.2f;
  }

  // Use raw CUDA memory management like the preprocessor package
  HashKey* d_keys = nullptr;
  uint64_t* d_cell_keys = nullptr;
  int* d_cell_starts = nullptr;
  int* d_cell_ends = nullptr;
  int* d_num_cells = nullptr;
  int* d_changed = nullptr;

  try {
    // Allocate device memory with error checking
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_keys, N * sizeof(HashKey)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cell_keys, N * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cell_starts, N * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cell_ends, N * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_num_cells, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_changed, sizeof(int)));

    // Initialize counters
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_num_cells, 0, sizeof(int), stream));
    
    // Setup grid dimensions
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Compute hash values
    compute_hash_kernel<<<blocks, threads, 0, stream>>>(d_points, N, voxel_size, d_keys);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // Sort by hash key using thrust
    thrust::device_ptr<HashKey> thrust_keys(d_keys);
    thrust::sort(thrust::cuda::par.on(stream), thrust_keys, thrust_keys + N, HashKeyCompare());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // Build cell data structure
    build_cell_offsets<<<blocks, threads, 0, stream>>>(d_keys, N, d_cell_keys, d_cell_starts, d_num_cells);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // Get number of cells
    int C = 0;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&C, d_num_cells, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // Skip if no cells or invalid count
    if (C <= 0 || C > N) {
      printf("Invalid cell count: %d (should be 0 < C <= %d)\n", C, N);
      thrust::device_ptr<int32_t> thrust_labels(d_labels);
      thrust::fill(thrust::cuda::par.on(stream), thrust_labels, thrust_labels + N, -1);
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    } else {
      // Create cell ends from starts
      int c_blocks = (C + threads - 1) / threads;
      fill_cell_ends<<<c_blocks, threads, 0, stream>>>(d_cell_starts, C, N, d_cell_ends);
      CHECK_CUDA_ERROR(cudaGetLastError());
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
      
      // Initialize labels to point indices
      thrust::device_ptr<int32_t> thrust_labels(d_labels);
      thrust::counting_iterator<int32_t> first(0);
      thrust::copy(thrust::cuda::par.on(stream), first, first + N, thrust_labels);
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
      
      // Perform label propagation
      float tol2 = params.tolerance * params.tolerance;
      int max_iters = params.max_iterations > 0 ? params.max_iterations : 10;
      
      for (int it = 0; it < max_iters; ++it) {
        // Reset change flag
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_changed, 0, sizeof(int), stream));
        
        // Run relaxation step
        int sblocks = (N + threads - 1) / threads;
        relax_labels<<<sblocks, threads, 0, stream>>>(
          d_points, d_keys, d_cell_keys, d_cell_starts, d_cell_ends,
          N, C, voxel_size, tol2, d_changed, d_labels);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        
        // Check if labels changed
        int host_changed = 0;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(&host_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (host_changed == 0) break;
      }
    }
    
    // Ensure all work is complete
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
  }
  catch (const std::exception &e) {
    printf("Exception during clustering: %s\n", e.what());
    // Initialize all labels to -1 (noise)
    thrust::device_ptr<int32_t> thrust_labels(d_labels);
    thrust::fill(thrust::cuda::par.on(stream), thrust_labels, thrust_labels + N, -1);
    cudaStreamSynchronize(stream);
  }

  // Clean up device memory
  if (d_keys) cudaFree(d_keys);
  if (d_cell_keys) cudaFree(d_cell_keys);
  if (d_cell_starts) cudaFree(d_cell_starts);
  if (d_cell_ends) cudaFree(d_cell_ends);
  if (d_num_cells) cudaFree(d_num_cells);
  if (d_changed) cudaFree(d_changed);
}

// Host wrapper (testing convenience)
std::vector<int32_t> cluster_host(const std::vector<Float3>& pts, const ClusteringParams& params) {
  int N = static_cast<int>(pts.size());
  if (N == 0) return {};
  thrust::device_vector<Float3> d_pts(pts.begin(), pts.end());
  thrust::device_vector<int32_t> d_labels(N);
  run_euclidean_clustering_cuda(
    thrust::raw_pointer_cast(d_pts.data()), N, params,
    thrust::raw_pointer_cast(d_labels.data()), 0);
  thrust::host_vector<int32_t> h = d_labels;
  return std::vector<int32_t>(h.begin(), h.end());
}

} // namespace autoware::euclidean_cluster_gpu
