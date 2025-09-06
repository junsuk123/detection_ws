#include "range_image_clustering/range_image_clustering_cuda.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace range_image_clustering {

// 블록 크기 정의
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8

__device__ float calculateAngleDifference(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dot = x1 * x2 + y1 * y2 + z1 * z2;
    float len1 = sqrt(x1 * x1 + y1 * y1 + z1 * z1);
    float len2 = sqrt(x2 * x2 + y2 * y2 + z2 * z2);
    
    float cosine = dot / (len1 * len2);
    if (cosine > 1.0f) cosine = 1.0f;
    if (cosine < -1.0f) cosine = -1.0f;
    
    return acos(cosine);
}

__device__ float calculateDistance(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__global__ void clusterKernel(
    float* points,        // 포인트 클라우드 데이터 (x,y,z)
    int* indices,         // 레인지 이미지 픽셀에 해당하는 포인트 인덱스
    int* labels,          // 클러스터 레이블
    int num_points,       // 포인트 개수
    int width,            // 레인지 이미지 너비
    int height,           // 레인지 이미지 높이
    float angle_threshold,// 각도 임계값
    float distance_threshold // 거리 임계값
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;
    
    float x = points[idx * 3];
    float y = points[idx * 3 + 1];
    float z = points[idx * 3 + 2];
    
    // 레인지 이미지 좌표 계산
    float range = sqrt(x * x + y * y + z * z);
    float azimuth = atan2(y, x);
    float elevation = asin(z / range);
    
    int col = static_cast<int>((azimuth + M_PI) / (2.0f * M_PI) * width) % width;
    int row = static_cast<int>((elevation + M_PI/6) / (M_PI/3) * height);
    
    if (row < 0 || row >= height) return;
    
    // 이웃 포인트 검사
    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue;
            
            int new_row = row + dr;
            int new_col = col + dc;
            
            // 방위각 방향 순환 처리
            if (new_col < 0) new_col += width;
            if (new_col >= width) new_col -= width;
            
            // 유효하지 않은 행 건너뛰기
            if (new_row < 0 || new_row >= height) continue;
            
            int img_idx = new_row * width + new_col;
            int neighbor_idx = indices[img_idx];
            
            if (neighbor_idx >= 0 && neighbor_idx < num_points && neighbor_idx != idx) {
                float nx = points[neighbor_idx * 3];
                float ny = points[neighbor_idx * 3 + 1];
                float nz = points[neighbor_idx * 3 + 2];
                
                float angle = calculateAngleDifference(x, y, z, nx, ny, nz);
                float dist = calculateDistance(x, y, z, nx, ny, nz);
                
                if (angle < angle_threshold && dist < distance_threshold) {
                    // 원자적 연산을 사용하여 레이블 업데이트
                    if (labels[idx] == -1 && labels[neighbor_idx] >= 0) {
                        labels[idx] = labels[neighbor_idx];
                    } else if (labels[neighbor_idx] == -1 && labels[idx] >= 0) {
                        atomicCAS(&labels[neighbor_idx], -1, labels[idx]);
                    } else if (labels[idx] >= 0 && labels[neighbor_idx] >= 0 && labels[idx] != labels[neighbor_idx]) {
                        int old_label = labels[neighbor_idx];
                        int new_label = labels[idx];
                        
                        // Union-find 유사 연산
                        if (old_label < new_label) {
                            atomicMin(&labels[idx], old_label);
                        } else {
                            atomicMin(&labels[neighbor_idx], new_label);
                        }
                    }
                }
            }
        }
    }
}

// 레이블 초기화 커널 - thrust::sequence 대체
__global__ void initLabelKernel(int* labels, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        labels[idx] = idx;
    }
}

void cudaClusterPointCloud(
    float* d_points,
    int* d_indices,
    int* d_labels,
    int num_points,
    int width,
    int height,
    float angle_threshold,
    float distance_threshold
) {
    // 레이블 초기화
    thrust::device_ptr<int> dev_labels(d_labels);
    thrust::fill(dev_labels, dev_labels + num_points, -1);
    
    // 초기 레이블 설정 - thrust::sequence 대신 커스텀 커널 사용
    dim3 initBlockSize(BLOCK_DIM_X * BLOCK_DIM_Y, 1);
    dim3 initGridSize((num_points + initBlockSize.x - 1) / initBlockSize.x, 1);
    initLabelKernel<<<initGridSize, initBlockSize>>>(d_labels, num_points);
    
    // 그리드 차원 계산
    dim3 blockSize(BLOCK_DIM_X, 1);
    dim3 gridSize((num_points + blockSize.x - 1) / blockSize.x, 1);
    
    // 클러스터링 커널 실행
    clusterKernel<<<gridSize, blockSize>>>(
        d_points,
        d_indices,
        d_labels,
        num_points,
        width,
        height,
        angle_threshold,
        distance_threshold
    );
    
    // 오류 확인
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    // GPU 작업 완료 대기
    cudaDeviceSynchronize();
}

} // namespace range_image_clustering
