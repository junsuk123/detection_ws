# Range Image Clustering

3D LiDAR 포인트 클라우드에 대한 효율적인 클러스터링 알고리즘을 제공하는 ROS 2 패키지입니다. 레인지 이미지 기반 접근 방식을 사용하여 빠른 클러스터링을 수행합니다.

## 알고리즘 개요

이 패키지는 다음과 같은 주요 알고리즘 흐름을 따릅니다:

1. **입력**: 3D 포인트 클라우드 (지면 제거된 상태)
2. **ROI 필터링**: 관심 영역 내 포인트만 선택
3. **레인지 이미지 변환**: 3D 포인트를 2D 레인지 이미지로 투영
4. **클러스터링**: 레인지 이미지에서 BFS 기반 클러스터링 수행
5. **후처리**: 품질 평가, 클러스터 병합, 필터링
6. **출력**: 클러스터링된 포인트 클라우드 및 시각화 마커

### 고급 기능

- **다단계 클러스터링**: 여러 임계값 수준에서 클러스터링을 수행하여 다양한 객체 크기 처리
- **적응형 임계값**: 거리에 따라 클러스터링 임계값을 자동 조정
- **포인트 특성 기반 클러스터링**: 법선 및 곡률과 같은 로컬 특성을 활용
- **ROI 필터링**: 관심 영역 내 포인트만 처리하여 효율성 증가
- **클러스터 품질 평가**: 클러스터 밀도, 형상 등을 평가하여 품질 점수 산출
- **클러스터 병합**: 유사한 인접 클러스터를 하나로 병합

## 설치 방법

### 요구 사항
- ROS 2 Humble 이상
- PCL 1.10 이상
- Eigen 3

### 빌드 방법

```bash
# 워크스페이스로 이동
cd /home/j/detection_ws

# 의존성 설치
sudo apt install ros-humble-pcl-conversions ros-humble-visualization-msgs

# 패키지 빌드
colcon build --symlink-install --packages-select range_image_clustering

# 환경 설정
source install/setup.bash
```

## 사용 방법

### 기본 실행

```bash
# 기본 설정으로 클러스터링 노드 실행
ros2 launch range_image_clustering range_image_clustering_for_tracker.launch.py
```

### 토픽

#### 입력
- `/pointcloud/ground_removed` (sensor_msgs/PointCloud2): 지면이 제거된 포인트 클라우드

#### 출력
- `/pointcloud/clustered` (sensor_msgs/PointCloud2): 클러스터 ID가 포함된 클러스터링된 포인트 클라우드
- `/pointcloud/clustered/markers` (visualization_msgs/MarkerArray): 시각화를 위한 클러스터 마커

### RViz에서 확인

```bash
# RViz 실행
ros2 run rviz2 rviz2

# 다음 토픽 추가:
# - PointCloud2: /pointcloud/clustered (Color Transformer: Intensity 또는 cluster_id로 설정)
# - MarkerArray: /pointcloud/clustered/markers
# - Marker: /visualization/roi_marker
```

## 파라미터 설명

주요 파라미터는 `config/params.yaml` 파일에서 설정할 수 있습니다:

### 클러스터링 파라미터
- `angle_threshold`: 클러스터링을 위한 각도 임계값(라디안)
- `distance_threshold`: 클러스터링을 위한 거리 임계값(미터)
- `min_cluster_size`: 유효한 클러스터의 최소 포인트 수
- `max_cluster_size`: 유효한 클러스터의 최대 포인트 수

### 레인지 이미지 파라미터
- `range_image_width`: 레인지 이미지 가로 해상도
- `range_image_height`: 레인지 이미지 세로 해상도(라이다 수직 레이어 수)

### ROI 파라미터
- `use_roi_filter`: ROI 필터링 사용 여부
- `roi_x_min`, `roi_x_max`: X축 ROI 범위
- `roi_y_min`, `roi_y_max`: Y축 ROI 범위
- `roi_z_min`, `roi_z_max`: Z축 ROI 범위

### 고급 기능 파라미터
- `use_multi_level_clustering`: 다단계 클러스터링 사용 여부
- `clustering_levels`: 클러스터링 단계 수
- `evaluate_cluster_quality`: 클러스터 품질 평가 활성화
- `merge_clusters`: 클러스터 병합 활성화
- `use_point_features`: 포인트 특성 기반 클러스터링 활성화

## 성능 최적화 팁

1. **레인지 이미지 해상도 조정**:
   - `range_image_width`와 `range_image_height`를 하드웨어 및 요구 사항에 맞게 조정
   
2. **ROI 설정**:
   - 관심 영역만 처리하도록 ROI 파라미터 조정

3. **임계값 튜닝**:
   - 클러스터링 결과가 좋지 않으면 `angle_threshold`와 `distance_threshold` 조정
   - 거리가 먼 객체의 경우 `enable_adaptive_threshold`를 활성화

4. **다단계 클러스터링**:
   - 다양한 크기의 객체가 있는 경우 `use_multi_level_clustering` 활성화
   
5. **클러스터 품질 평가 및 병합**:
   - 과분할된 클러스터가 많은 경우 `merge_clusters` 활성화

## 알려진 이슈 및 제한 사항

- 매우 밀집된 포인트 클라우드에서는 과분할 현상이 발생할 수 있음
- 특정 조건에서 레인지 이미지 매핑 효율이 낮을 수 있음
- CPU 사용량이 높을 수 있으므로 필요에 따라 `range_image_width`, `range_image_height` 조정 필요
