# README - Detection Workspace (detection_ws)

## 1. 환경 준비 및 전체 빌드

```bash
cd ~/detection_ws

# ROS2 Humble 환경 설정
source /opt/ros/humble/setup.bash

# Autoware 환경 설정
source ~/autoware/install/setup.bash

# 전체 의존성 패키지 설치
rosdep update
rosdep install --from-paths src -y --ignore-src

# ultralytics_ros 의존성 설치
cd ~/detection_ws/src/ultralytics_ros
pip install -r requirements.txt

cd ~/detection_ws

# 전체 빌드
colcon build --symlink-install
source install/setup.bash
```

---

## 2. 패키지별 주요 기능 및 토픽/런치 명령어

### 1) 라이다 센서 드라이버 (nebula_ros)

- **기능:** Velodyne 라이다 데이터 수집
- **OUTPUT TOPIC:** `/velodyne_points`

```bash
ros2 launch nebula_ros velodyne_launch_all_hw.xml sensor_model:=VLP32
```

---

### 2) GPU 기반 노이즈 제거 (autoware_cuda_pointcloud_preprocessor)

- **기능:** 라이다 포인트클라우드 노이즈 제거 (CUDA)
- **INPUT TOPIC:** `/velodyne_points`
- **OUTPUT TOPIC:** `/pointcloud_noiseremoved`

```bash
ros2 launch autoware_cuda_pointcloud_preprocessor cuda_pointcloud_preprocessor.launch.py
```

---

### 3) 지면 제거 (autoware_ground_segmentation)

- **기능:** 포인트클라우드에서 지면(ground) 제거
- **INPUT TOPIC:** `/pointcloud_noiseremoved`
- **OUTPUT TOPIC:** `/pointcloud/ground_removed`

```bash
ros2 launch autoware_ground_segmentation scan_ground_filter.launch.py
```

---

### 4) 레인지 이미지 기반 클러스터링 (range_image_clustering)

- **기능:** 레인지 이미지 기반 고속 포인트 클라우드 클러스터링 
- **INPUT TOPIC:** `/pointcloud/ground_removed`
- **OUTPUT TOPIC:** `/pointcloud/clustered` (클러스터 ID 포함)
- **특징:** 
  - ROI 기반 필터링
  - 다단계 클러스터링
  - 클러스터 품질 평가 및 병합
  - 적응형 거리 임계값

```bash
ros2 launch range_image_clustering range_image_clustering_for_tracker.launch.py
```

---

### 5) GPU 기반 클러스터링 (autoware_euclidean_cluster_gpu)

- **기능:** ROI 내 포인트클라우드 클러스터링 (CUDA)
- **INPUT TOPIC:** `/pointcloud/ground_removed`
- **OUTPUT TOPIC:** `/perception/clustered/points_only` (ROI 내 포인트만 포함)

```bash
ros2 launch euclidean_cluster_gpu euclidean_cluster_gpu.launch.py
```

---



### 6) 카메라 실행 (usb_cam)

- **기능:** USB 카메라 이미지 토픽 발행
- **런치/설정파일 필요시 수정**
- **OUTPUT TOPIC:** `/camera2/image_raw` 등

```bash
ros2 launch usb_cam camera.launch.py
```

---

### 7) YOLO + LiDAR 3D Projection (ultralytics_ros)

- **기능:** YOLO 객체 인식 결과와 라이다 클러스터를 3D로 매칭
- **INPUT TOPIC:** `/pointcloud/clustered`, `/yolo_result`
- **OUTPUT TOPIC:** `/yolo_3d_result`

```bash
ros2 launch ultralytics_ros tracker_with_cloud.launch.xml debug:=true
```

---

## 3. 전체 데이터 흐름 요약

### 기본 파이프라인
```
[LiDAR] 
  └─/velodyne_points
      ↓
[노이즈 제거]
  └─/pointcloud_noiseremoved
      ↓
[지면 제거]
  └─/pointcloud/ground_removed
      ↓
[클러스터링]
  └─/perception/clustered/points_only
      ↓
[YOLO+3D 매칭]
  └─/yolo_3d_result
```

---

## 4. 참고

- 각 패키지의 config/launch 파일에서 토픽명, 파라미터 등을 환경에 맞게 조정하세요.
- 모델 파일 등은 반드시 설치 경로에 존재해야 합니다.
- RViz 등 시각화는 필요에 따라 추가 실행하세요.

---
