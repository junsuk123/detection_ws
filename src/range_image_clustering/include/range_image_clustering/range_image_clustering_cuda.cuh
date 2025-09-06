#ifndef RANGE_IMAGE_CLUSTERING_CUDA_CUH
#define RANGE_IMAGE_CLUSTERING_CUDA_CUH

#include <cuda_runtime.h>

namespace range_image_clustering {

// CUDA 커널 선언
void cudaClusterPointCloud(
    float* d_points,      // 디바이스 메모리의 포인트 클라우드 (x,y,z)
    int* d_indices,       // 레인지 이미지 픽셀에 해당하는 포인트 인덱스
    int* d_labels,        // 각 포인트의 클러스터 레이블
    int num_points,       // 포인트 수
    int width,            // 레인지 이미지 너비
    int height,           // 레인지 이미지 높이
    float angle_threshold,// 각도 임계값
    float distance_threshold // 거리 임계값
);

} // namespace range_image_clustering

#endif // RANGE_IMAGE_CLUSTERING_CUDA_CUH
