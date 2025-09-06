#include "range_image_clustering/range_image_clustering.h"
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <queue>
#include <random>
#include <iostream>

namespace range_image_clustering {

RangeImageClustering::RangeImageClustering() : width_(0), height_(0), input_cloud_(new pcl::PointCloud<pcl::PointXYZI>) {
  // 기본 파라미터 설정
  params_.angle_threshold = 0.1f;  // 약 5.7도
  params_.distance_threshold = 0.5f;  // 0.5 미터
  params_.min_cluster_size = 10;
  params_.max_cluster_size = 10000;
  params_.range_image_width = 1800;
  params_.range_image_height = 32;
}

RangeImageClustering::~RangeImageClustering() {}

void RangeImageClustering::setParams(const ClusteringParams& params) {
  params_ = params;
  std::cout << "클러스터링 파라미터 설정됨 - 최소 크기: " << params_.min_cluster_size 
            << ", 최대 크기: " << params_.max_cluster_size << std::endl;
}

void RangeImageClustering::process(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
  if (cloud->empty()) {
    std::cerr << "입력 클라우드가 비어 있습니다!" << std::endl;
    return;
  }
  
  std::cout << "RangeImageClustering::process - 포인트 개수: " << cloud->points.size() << std::endl;
  
  // 입력 클라우드 저장
  input_cloud_ = cloud;
  
  // 포인트 특성 계산 (설정된 경우)
  if (params_.use_point_features) {
    computePointFeatures();
  }
  
  // 레인지 이미지 생성 및 클러스터링
  createRangeImage(cloud);
  performClustering();
  
  // 품질 평가 및 필터링 (설정된 경우)
  if (params_.evaluate_cluster_quality) {
    evaluateAndFilterClusters();
  }
  
  // 유사 클러스터 병합 (설정된 경우)
  if (params_.merge_clusters && clusters_.size() > 1) {
    mergeSimilarClusters();
  }
  
  std::cout << "클러스터링 완료 - 클러스터 수: " << clusters_.size() << std::endl;
}

// 포인트 특성(법선, 곡률) 계산
void RangeImageClustering::computePointFeatures() {
  // 법선 및 곡률 계산을 위한 PCL 함수 사용
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
  
  ne.setInputCloud(input_cloud_);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(0.03);  // 3cm 내 이웃점 사용
  ne.compute(*normals);
  
  // 결과 저장
  point_normals_.resize(input_cloud_->points.size());
  point_curvatures_.resize(input_cloud_->points.size());
  
  for (size_t i = 0; i < normals->points.size(); ++i) {
    point_normals_[i] = Eigen::Vector3f(
        normals->points[i].normal_x,
        normals->points[i].normal_y,
        normals->points[i].normal_z
    );
    point_curvatures_[i] = normals->points[i].curvature;
  }
  
  features_computed_ = true;
  std::cout << "포인트 특성 계산 완료" << std::endl;
}

void RangeImageClustering::createRangeImage(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
  // 레인지 이미지 파라미터 설정
  width_ = params_.range_image_width;  // 레인지 이미지 너비
  height_ = params_.range_image_height;  // 레인지 이미지 높이
  
  range_image_.resize(width_ * height_, -1.0f);  // -1은 데이터 없음을 표시
  point_indices_.resize(width_ * height_, -1);   // -1은 포인트 없음을 표시
  
  int mapped_points = 0;
  int invalid_points = 0;
  
  // 포인트를 레인지 이미지로 투영
  for (size_t i = 0; i < cloud->points.size(); i++) {
    const auto& point = cloud->points[i];
    
    // 무효한 포인트 건너뛰기
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      invalid_points++;
      continue;
    }
    
    // 구면 좌표계 계산
    float range = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    if (range < 0.1f) {
      invalid_points++;
      continue; // 너무 가까운 포인트 제외
    }
    
    // 수정: azimuth와 elevation 계산 방식 개선
    // azimuth는 -pi ~ pi
    float azimuth = atan2(point.y, point.x);
    
    // elevation은 -pi/2 ~ pi/2
    float elevation = atan2(point.z, sqrt(point.x*point.x + point.y*point.y));
    
    // 픽셀 좌표로 변환 개선
    // 방위각을 [0, width_] 범위로 변환
    int col = static_cast<int>((azimuth + M_PI) / (2.0 * M_PI) * width_);
    
    // 경계 검사 개선 - 모듈로 연산으로 모든 각도 포함
    if (col < 0) col += width_;
    if (col >= width_) col %= width_;
    
    // 고도각을 [0, height_] 범위로 변환 - 개선된 공식
    int row = static_cast<int>((elevation + M_PI/2) / M_PI * height_);
    
    // 배열 범위 안전성 검사
    if (row >= 0 && row < height_) {
      int idx = row * width_ + col;
      // 가장 가까운 포인트만 저장 (중복된 픽셀이 있을 경우)
      if (range_image_[idx] < 0 || range < range_image_[idx]) {
        range_image_[idx] = range;
        point_indices_[idx] = i;
        mapped_points++;
      }
    }
  }
  
  std::cout << "레인지 이미지 매핑된 포인트 수: " << mapped_points << "/" << cloud->points.size() 
            << " (무효 포인트: " << invalid_points << ")" << std::endl;
  
  if (mapped_points < 100) {
    std::cout << "경고: 매핑된 포인트 수가 너무 적습니다. 레인지 이미지 파라미터 확인 필요!" << std::endl;
  }
}

void RangeImageClustering::performClustering() {
  if (params_.use_multi_level_clustering) {
    performMultiLevelClustering();
  } else {
    // 기존 단일 레벨 클러스터링
    clusters_.clear();
    
    std::vector<int> labels(input_cloud_->points.size(), -1);
    std::vector<bool> processed(input_cloud_->points.size(), false);
    int current_label = 0;
    
    // 클러스터링 효율성 향상을 위한 통계 변수
    int total_points = 0;
    int processed_points = 0;
    int clusters_found = 0;
    int single_point_clusters = 0;
    
    // 유효한 레인지 이미지 포인트만 카운트
    for (size_t i = 0; i < range_image_.size(); i++) {
      if (range_image_[i] > 0 && 
          point_indices_[i] >= 0 && 
          point_indices_[i] < static_cast<int>(input_cloud_->points.size())) {
        total_points++;
      }
    }
    
    // BFS 기반 클러스터링 - 레인지 이미지에서 직접 시작
    for (int row = 0; row < height_; row++) {
      for (int col = 0; col < width_; col++) {
        int img_idx = row * width_ + col;
        
        // 유효한 레인지 이미지 포인트인지 확인
        if (range_image_[img_idx] <= 0) continue;
        
        int point_idx = point_indices_[img_idx];
        if (point_idx < 0 || point_idx >= static_cast<int>(input_cloud_->points.size())) continue;
        
        if (processed[point_idx]) continue;
        
        // 클러스터 시작
        std::queue<std::pair<int, int>> neighbors;  // <row, col> 쌍으로 저장
        neighbors.push(std::make_pair(row, col));
        processed[point_idx] = true;
        labels[point_idx] = current_label;
        processed_points++;
        
        pcl::PointIndices cluster_indices;
        cluster_indices.indices.push_back(point_idx);
        
        while (!neighbors.empty()) {
          auto [curr_row, curr_col] = neighbors.front();
          neighbors.pop();
          
          int curr_img_idx = curr_row * width_ + curr_col;
          int curr_point_idx = point_indices_[curr_img_idx];
          
          // 현재 포인트 정보
          const auto& point = input_cloud_->points[curr_point_idx];
          float range = range_image_[curr_img_idx];
          
          // 적응형 임계값 계산
          float current_distance_threshold = params_.distance_threshold;
          if (params_.enable_adaptive_threshold && range > params_.min_distance_for_adaptive) {
            float factor = std::min(range / params_.min_distance_for_adaptive, params_.max_adaptive_factor);
            current_distance_threshold *= factor;
          }
          
          // 8방향 이웃 탐색 (개선된 이웃 탐색 알고리즘)
          for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
              if (dr == 0 && dc == 0) continue;
              
              int new_row = curr_row + dr;
              int new_col = curr_col + dc;
              
              // 방위각 방향 순환 처리 (개선)
              if (new_col < 0) new_col += width_;
              else if (new_col >= width_) new_col -= width_;
              
              // 유효하지 않은 행 건너뛰기
              if (new_row < 0 || new_row >= height_) continue;
              
              int new_img_idx = new_row * width_ + new_col;
              
              // 이웃 셀에 유효한 포인트가 있는지 확인
              if (range_image_[new_img_idx] <= 0) continue;
              
              int neighbor_idx = point_indices_[new_img_idx];
              if (neighbor_idx < 0 || neighbor_idx >= static_cast<int>(input_cloud_->points.size())) continue;
              
              if (!processed[neighbor_idx]) {
                // 각도와 거리 기준으로 연결 여부 판단
                const auto& neighbor_point = input_cloud_->points[neighbor_idx];
                
                // 무효한 포인트 건너뛰기
                if (!std::isfinite(neighbor_point.x) || 
                    !std::isfinite(neighbor_point.y) || 
                    !std::isfinite(neighbor_point.z)) {
                  continue;
                }
                
                float dist = pcl::euclideanDistance(point, neighbor_point);
                
                // 정규화된 벡터 간의 각도 계산
                Eigen::Vector3f v1(point.x, point.y, point.z);
                Eigen::Vector3f v2(neighbor_point.x, neighbor_point.y, neighbor_point.z);
                float len1 = v1.norm();
                float len2 = v2.norm();
                
                if (len1 < 1e-6 || len2 < 1e-6) continue;
                
                v1.normalize();
                v2.normalize();
                float angle = acos(std::min(1.0f, std::max(-1.0f, v1.dot(v2))));
                
                if (dist < current_distance_threshold && angle < params_.angle_threshold) {
                  neighbors.push(std::make_pair(new_row, new_col));
                  processed[neighbor_idx] = true;
                  processed_points++;
                  labels[neighbor_idx] = current_label;
                  cluster_indices.indices.push_back(neighbor_idx);
                }
              }
            }
          }
        }
        
        // 클러스터 크기 통계
        if (cluster_indices.indices.size() == 1) {
          single_point_clusters++;
        }
        
        // 크기 기준 충족 여부 확인 - 수정: 단일 포인트도 클러스터로 인정
        if (cluster_indices.indices.size() >= static_cast<size_t>(params_.min_cluster_size) && 
            cluster_indices.indices.size() <= static_cast<size_t>(params_.max_cluster_size)) {
          clusters_.push_back(cluster_indices);
          clusters_found++;
          current_label++;
        }
      }
    }
    
    std::cout << "클러스터링 효율성: 총 " << total_points << " 포인트 중 " << processed_points
              << " 처리됨 (" << (100.0f * processed_points / std::max(1, total_points)) << "%)" << std::endl;
    std::cout << "발견된 클러스터: " << clusters_found << ", 단일 포인트 클러스터: " << single_point_clusters
              << " (" << (100.0f * single_point_clusters / std::max(1, clusters_found)) << "%)" << std::endl;
  }
}

void RangeImageClustering::performMultiLevelClustering() {
  clusters_.clear();
  
  std::vector<bool> processed(input_cloud_->points.size(), false);
  
  // 각 레벨마다 클러스터링 수행
  for (int level = 0; level < params_.clustering_levels; ++level) {
    // 레벨에 따라 임계값 증가
    float current_angle_threshold = params_.angle_threshold + level * params_.angle_threshold_step;
    float current_distance_threshold = params_.distance_threshold + level * params_.distance_threshold_step;
    
    std::vector<pcl::PointIndices> level_clusters;
    performSingleLevelClustering(current_angle_threshold, current_distance_threshold, level_clusters, processed);
    
    // 새로 찾은 클러스터 추가
    clusters_.insert(clusters_.end(), level_clusters.begin(), level_clusters.end());
    
    std::cout << "레벨 " << level << " 클러스터링 완료: " << level_clusters.size() 
              << " 클러스터 발견 (총 " << clusters_.size() << ")" << std::endl;
  }
}

void RangeImageClustering::performSingleLevelClustering(
    float angle_threshold, 
    float distance_threshold, 
    std::vector<pcl::PointIndices>& new_clusters,
    std::vector<bool>& processed) {
  
  int current_label = 0;
  
  for (size_t i = 0; i < input_cloud_->points.size(); i++) {
    if (processed[i] || !std::isfinite(input_cloud_->points[i].x)) continue;
    
    // 새 클러스터 시작
    pcl::PointIndices cluster_indices;
    std::queue<int> neighbors;
    neighbors.push(i);
    processed[i] = true;
    cluster_indices.indices.push_back(i);
    
    while (!neighbors.empty()) {
      int current = neighbors.front();
      neighbors.pop();
      
      // 레인지 이미지에서 현재 포인트의 위치 찾기
      const auto& point = input_cloud_->points[current];
      float range = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
      float azimuth = atan2(point.y, point.x);
      float elevation = asin(point.z / range);
      
      // 현재 포인트에 대한 거리 임계값 계산 (적응형)
      float current_distance_threshold = distance_threshold;
      if (params_.enable_adaptive_threshold && range > params_.min_distance_for_adaptive) {
        float factor = std::min(range / params_.min_distance_for_adaptive, params_.max_adaptive_factor);
        current_distance_threshold *= factor;
      }
      
      int col = static_cast<int>((azimuth + M_PI) / (2.0 * M_PI) * width_) % width_;
      int row = static_cast<int>((elevation + M_PI/2) / M_PI * height_);
      
      // 8방향 이웃 탐색
      for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
          if (dr == 0 && dc == 0) continue;
          
          int new_row = row + dr;
          int new_col = col + dc;
          
          // 방위각 방향 순환 처리
          if (new_col < 0) new_col += width_;
          if (new_col >= width_) new_col -= width_;
          
          // 유효하지 않은 행 건너뛰기
          if (new_row < 0 || new_row >= height_) continue;
          
          int img_idx = new_row * width_ + new_col;
          if (img_idx < 0 || img_idx >= static_cast<int>(range_image_.size())) continue;
          
          if (range_image_[img_idx] < 0) continue;  // 데이터 없음
          
          int neighbor_idx = point_indices_[img_idx];
          if (neighbor_idx < 0 || neighbor_idx >= static_cast<int>(input_cloud_->points.size())) continue;
          
          if (!processed[neighbor_idx]) {
            // 각도와 거리 기준으로 연결 여부 판단
            const auto& neighbor_point = input_cloud_->points[neighbor_idx];
            
            // 무효한 포인트 건너뛰기
            if (!std::isfinite(neighbor_point.x) || 
                !std::isfinite(neighbor_point.y) || 
                !std::isfinite(neighbor_point.z)) {
              continue;
            }
            
            float dist = pcl::euclideanDistance(point, neighbor_point);
            
            // 정규화된 벡터 간의 각도 계산
            Eigen::Vector3f v1(point.x, point.y, point.z);
            Eigen::Vector3f v2(neighbor_point.x, neighbor_point.y, neighbor_point.z);
            float len1 = v1.norm();
            float len2 = v2.norm();
            
            if (len1 < 1e-6 || len2 < 1e-6) continue;
            
            v1.normalize();
            v2.normalize();
            float angle = acos(v1.dot(v2));
            
            if (dist < current_distance_threshold && angle < angle_threshold) {
              neighbors.push(neighbor_idx);
              processed[neighbor_idx] = true;
              cluster_indices.indices.push_back(neighbor_idx);
            }
          }
        }
      }
    }
    
    // 크기 기준 충족 여부 확인
    if (cluster_indices.indices.size() >= static_cast<size_t>(params_.min_cluster_size) &&
        cluster_indices.indices.size() <= static_cast<size_t>(params_.max_cluster_size)) {
      new_clusters.push_back(cluster_indices);
      current_label++;
    }
  }
}

void RangeImageClustering::evaluateAndFilterClusters() {
  std::vector<pcl::PointIndices> filtered_clusters;
  std::vector<float> quality_scores;
  
  // 각 클러스터의 품질 계산
  for (const auto& cluster : clusters_) {
    float quality = computeClusterQuality(cluster);
    if (quality > 0) {  // 품질 임계값 통과
      filtered_clusters.push_back(cluster);
      quality_scores.push_back(quality);
    }
  }
  
  // 필터링된 클러스터로 업데이트
  clusters_ = filtered_clusters;
  std::cout << "품질 평가 후 클러스터 수: " << clusters_.size() << std::endl;
}

float RangeImageClustering::computeClusterQuality(const pcl::PointIndices& cluster) {
  if (cluster.indices.size() < 3) return 0.0f;
  
  // 클러스터 경계 상자 계산
  Eigen::Vector4f min_pt, max_pt;
  pcl::getMinMax3D(*input_cloud_, cluster.indices, min_pt, max_pt);
  
  // 클러스터 부피 계산
  float volume = (max_pt[0] - min_pt[0]) * (max_pt[1] - min_pt[1]) * (max_pt[2] - min_pt[2]);
  if (volume < 1e-6) return 0.0f;
  
  // 클러스터 밀도 계산
  float density = static_cast<float>(cluster.indices.size()) / volume;
  
  // 클러스터 형상 분석 (PCA로 주축 길이 비율 계산)
  Eigen::MatrixXf points(cluster.indices.size(), 3);
  for (size_t i = 0; i < cluster.indices.size(); ++i) {
    points(i, 0) = input_cloud_->points[cluster.indices[i]].x;
    points(i, 1) = input_cloud_->points[cluster.indices[i]].y;
    points(i, 2) = input_cloud_->points[cluster.indices[i]].z;
  }
  
  Eigen::Vector3f centroid = points.colwise().mean();
  Eigen::MatrixXf centered = points.rowwise() - centroid.transpose();
  Eigen::MatrixXf cov = (centered.transpose() * centered) / float(points.rows() - 1);
  
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
  Eigen::Vector3f eigenvalues = eig.eigenvalues();
  
  // 클러스터 길이 비율 (최대축/최소축)
  float elongation = eigenvalues[2] / std::max(eigenvalues[0], 1e-6f);
  
  // 품질 점수 계산 (밀도와 형상 고려)
  float quality_score = 1.0;
  
  // 밀도가 너무 낮으면 감점
  if (density < params_.min_cluster_density) {
    quality_score *= (density / params_.min_cluster_density);
  }
  
  // 길이 비율이 너무 크면 감점 (너무 길쭉하면 좋지 않음)
  if (elongation > params_.max_cluster_elongation) {
    quality_score *= (params_.max_cluster_elongation / elongation);
  }
  
  return quality_score;
}

void RangeImageClustering::mergeSimilarClusters() {
  bool merged_any = false;
  
  do {
    merged_any = false;
    
    for (size_t i = 0; i < clusters_.size(); ++i) {
      for (size_t j = i + 1; j < clusters_.size(); ++j) {
        if (shouldMergeClusters(clusters_[i], clusters_[j])) {
          // 클러스터 병합
          clusters_[i].indices.insert(
              clusters_[i].indices.end(),
              clusters_[j].indices.begin(),
              clusters_[j].indices.end());
          
          // j번째 클러스터 제거
          clusters_.erase(clusters_.begin() + j);
          merged_any = true;
          break;
        }
      }
      if (merged_any) break;
    }
  } while (merged_any);
  
  std::cout << "클러스터 병합 후 클러스터 수: " << clusters_.size() << std::endl;
}

bool RangeImageClustering::shouldMergeClusters(
    const pcl::PointIndices& cluster1, 
    const pcl::PointIndices& cluster2) {
  
  // 두 클러스터의 중심점 계산
  Eigen::Vector3f centroid1(0, 0, 0), centroid2(0, 0, 0);
  
  for (auto idx : cluster1.indices) {
    centroid1[0] += input_cloud_->points[idx].x;
    centroid1[1] += input_cloud_->points[idx].y;
    centroid1[2] += input_cloud_->points[idx].z;
  }
  centroid1 /= cluster1.indices.size();
  
  for (auto idx : cluster2.indices) {
    centroid2[0] += input_cloud_->points[idx].x;
    centroid2[1] += input_cloud_->points[idx].y;
    centroid2[2] += input_cloud_->points[idx].z;
  }
  centroid2 /= cluster2.indices.size();
  
  // 중심점 간 거리
  float distance = (centroid1 - centroid2).norm();
  
  // 병합 결정: 거리가 임계값보다 작으면 병합
  return distance < params_.cluster_merge_threshold;
}

void RangeImageClustering::getClusteredCloudWithIntensity(pcl::PointCloud<pcl::PointXYZI>::Ptr& clustered_cloud) {
  clustered_cloud->clear();
  
  // 클러스터별로 ID 값을 intensity로 사용 - 유효한 클러스터만
  for (size_t i = 0; i < clusters_.size(); i++) {
    // 클러스터 ID를 intensity 값으로 사용 (1부터 시작)
    float cluster_id = static_cast<float>(i + 1); // 0은 미분류 포인트용으로 예약
    
    for (const auto& idx : clusters_[i].indices) {
      pcl::PointXYZI point;
      point.x = input_cloud_->points[idx].x;
      point.y = input_cloud_->points[idx].y;
      point.z = input_cloud_->points[idx].z;
      point.intensity = cluster_id;  // 클러스터 ID 저장
      clustered_cloud->points.push_back(point);
    }
  }
  
  clustered_cloud->width = clustered_cloud->points.size();
  clustered_cloud->height = 1;
  clustered_cloud->is_dense = true;
  
  std::cout << "클러스터링된 포인트 수: " << clustered_cloud->points.size() << std::endl;
}

void RangeImageClustering::getClusteredCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& clustered_cloud) {
  clustered_cloud->clear();
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  
  for (size_t i = 0; i < clusters_.size(); i++) {
    // 클러스터별 랜덤 색상 생성
    uint8_t r = dis(gen);
    uint8_t g = dis(gen);
    uint8_t b = dis(gen);
    
    for (const auto& idx : clusters_[i].indices) {
      pcl::PointXYZRGB colored_point;
      colored_point.x = input_cloud_->points[idx].x;
      colored_point.y = input_cloud_->points[idx].y;
      colored_point.z = input_cloud_->points[idx].z;
      colored_point.r = r;
      colored_point.g = g;
      colored_point.b = b;
      clustered_cloud->points.push_back(colored_point);
    }
  }
  
  clustered_cloud->width = clustered_cloud->points.size();
  clustered_cloud->height = 1;
  clustered_cloud->is_dense = true;
  
  std::cout << "RGB 클라우드 생성 완료 - 포인트 수: " << clustered_cloud->points.size() << std::endl;
}

void RangeImageClustering::getClusterMarkers(visualization_msgs::msg::MarkerArray& markers, const std::string& frame_id, double lifetime) {
  markers.markers.clear();
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  
  for (size_t i = 0; i < clusters_.size(); i++) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = rclcpp::Clock().now();
    marker.ns = "clusters";
    marker.id = i;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.color.r = dis(gen) / 255.0f;
    marker.color.g = dis(gen) / 255.0f;
    marker.color.b = dis(gen) / 255.0f;
    marker.color.a = 1.0;
    marker.lifetime = rclcpp::Duration::from_seconds(lifetime);
    
    for (const auto& idx : clusters_[i].indices) {
      geometry_msgs::msg::Point p;
      p.x = input_cloud_->points[idx].x;
      p.y = input_cloud_->points[idx].y;
      p.z = input_cloud_->points[idx].z;
      marker.points.push_back(p);
    }
    
    markers.markers.push_back(marker);
  }
}

void RangeImageClustering::getAllClusters(std::vector<pcl::PointIndices>& all_clusters) const {
  all_clusters = clusters_;
}

} // namespace range_image_clustering
