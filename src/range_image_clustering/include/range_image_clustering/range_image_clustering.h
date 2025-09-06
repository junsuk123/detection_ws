#ifndef RANGE_IMAGE_CLUSTERING_H
#define RANGE_IMAGE_CLUSTERING_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vector>
#include <Eigen/Dense>

namespace range_image_clustering {

struct ClusteringParams {
  float angle_threshold;      // 라디안 단위
  float distance_threshold;   // 미터 단위
  int min_cluster_size;       // 최소 클러스터 크기
  int max_cluster_size;       // 최대 클러스터 크기
  int range_image_width;      // 레인지 이미지 너비
  int range_image_height;     // 레인지 이미지 높이
  
  // 적응형 임계값 설정
  bool enable_adaptive_threshold = false;  // 거리에 따른 임계값 조정 활성화
  float min_distance_for_adaptive = 5.0f;  // 적응형 임계값이 시작되는 거리 (미터)
  float max_adaptive_factor = 3.0f;        // 최대 적응형 계수
  
  // 다단계 클러스터링 설정
  bool use_multi_level_clustering = false;  // 다단계 클러스터링 사용 여부
  int clustering_levels = 2;                // 클러스터링 단계 수
  float distance_threshold_step = 0.1f;     // 단계별 거리 임계값 증가량
  float angle_threshold_step = 0.05f;       // 단계별 각도 임계값 증가량
  
  // 클러스터 품질 평가
  bool evaluate_cluster_quality = false;     // 클러스터 품질 평가 여부
  float min_cluster_density = 0.01f;         // 최소 클러스터 밀도 (포인트/m³)
  float max_cluster_elongation = 5.0f;       // 최대 클러스터 길이 비율
  
  // 클러스터 병합 설정
  bool merge_clusters = false;               // 클러스터 병합 여부
  float cluster_merge_threshold = 0.3f;      // 클러스터 병합 거리 임계값(m)
  float cluster_merge_angle = 0.5f;          // 클러스터 병합 각도 임계값(rad)
  
  // 포인트 특성 기반 클러스터링
  bool use_point_features = false;       // 포인트 특성 사용 여부
  float feature_weight = 0.5f;           // 특성 가중치 (0-1)
  float curvature_threshold = 0.05f;     // 곡률 임계값
  float normal_angle_weight = 0.3f;      // 법선 각도 가중치
};

class RangeImageClustering {
public:
  // 생성자 및 소멸자
  RangeImageClustering();
  ~RangeImageClustering();

  // 주요 인터페이스 함수
  void setParams(const ClusteringParams& params);
  void process(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
  
  // 클러스터 결과 획득 함수
  void getClusteredCloudWithIntensity(pcl::PointCloud<pcl::PointXYZI>::Ptr& clustered_cloud);
  void getClusteredCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& clustered_cloud);
  void getClusterMarkers(visualization_msgs::msg::MarkerArray& markers, 
                         const std::string& frame_id, double lifetime = 0.1);
  void getAllClusters(std::vector<pcl::PointIndices>& all_clusters) const;

private:
  // 기본 클러스터링 변수
  ClusteringParams params_;
  int width_, height_;  // 먼저 초기화되는 변수들을 앞으로 이동
  pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_;  // width_, height_ 이후 초기화
  std::vector<pcl::PointIndices> clusters_;
  
  // 레인지 이미지 처리 변수
  std::vector<float> range_image_;
  std::vector<int> point_indices_;
  
  // 기본 클러스터링 함수
  void createRangeImage(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
  void performClustering();
  
  // 다단계 클러스터링 관련 함수
  void performMultiLevelClustering();
  void performSingleLevelClustering(float angle_threshold, float distance_threshold, 
                                  std::vector<pcl::PointIndices>& new_clusters,
                                  std::vector<bool>& processed);
  
  // 클러스터 품질 평가 및 병합 관련 함수
  void evaluateAndFilterClusters();
  float computeClusterQuality(const pcl::PointIndices& cluster);
  void mergeSimilarClusters();
  bool shouldMergeClusters(const pcl::PointIndices& cluster1, 
                         const pcl::PointIndices& cluster2);
  
  // 포인트 특성 계산 관련 변수/함수
  void computePointFeatures();
  std::vector<float> point_curvatures_;
  std::vector<Eigen::Vector3f> point_normals_;
  bool features_computed_ = false;
};

} // namespace range_image_clustering

#endif // RANGE_IMAGE_CLUSTERING_H