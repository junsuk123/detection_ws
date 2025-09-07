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
  - 깊이 불연속 기반 클러스터링 (소형 객체 분리 성능 향상)
  - 적응형 최소 클러스터 크기 (원거리 객체 검출 개선)
  - 센서 채널 맵 기반 정확한 수직 매핑
  - acos 최적화로 성능 개선
  - 클러스터 병합 기준 개선 (경계 상자 교차 검사)

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

- **기능:** USB 카메라 이미지 토픽 발행 및 카메라-라이다 TF 발행
- **OUTPUT TOPIC:** `/camera1/image_raw`, `/camera2/image_raw`, ... 등
- **TF 설정:** 
  - TFConfig 폴더의 YAML 파일에서 카메라-라이다 변환 행렬 설정
  - 캘리브레이션 결과에 기반한 정확한 TF 자동 발행

```bash
ros2 launch usb_cam camera.launch.py
```

---

### 7) YOLO + LiDAR 3D Projection (ultralytics_ros)

- **기능:** YOLO 객체 인식 결과와 라이다 클러스터를 3D로 매칭
- **INPUT TOPIC:** `/pointcloud/clustered`, `/detection/{camera_name}/yolo_result`
- **OUTPUT TOPIC:** `/yolo_3d_result`
- **특징:**
  - 여러 카메라 자동 감지 및 처리 (최대 4대)
  - 런타임 시 활성화된 카메라만 선택적으로 처리
  - 다중 카메라 결과를 통합하여 3D 객체 생성

```bash
ros2 launch ultralytics_ros tracker_with_cloud.launch.py debug:=true
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
  └─/pointcloud/clustered
      ↓
[YOLO 인식]  ←── [카메라(들)]
  └─/detection/{camera_name}/yolo_result
      ↓
[YOLO+3D 매칭]
  └─/yolo_3d_result
```

---

## 4. 새로운 기능

### 1) 다중 카메라 자동 감지 및 처리
- **기능:** 런치 시 활성화된 카메라 토픽을 자동으로 감지하고 해당 카메라에 대한 YOLO 노드 실행
- **지원:** 최대 4대 카메라 동시 처리 가능

### 2) 카메라-라이다 TF 자동 설정
- **경로:** `/home/j/detection_ws/src/usb_cam/config/TFConfig/`
- **파일형식:** `{camera_name}_tf.yaml`
- **특징:** 캘리브레이션 결과를 YAML 파일로 저장하고 자동으로 TF 발행

### 3) 소형 객체 분리 최적화
- **기능:** 근거리의 소형 객체(콘 등)를 더 정확하게 분리하는 클러스터링 최적화
- **파라미터:** 
  - 깊이 불연속 기반 클러스터링 
  - 더 엄격한 거리/각도 임계값
  - 적응형 최소 클러스터 크기

---

## 5. 참고

- 각 패키지의 config/launch 파일에서 토픽명, 파라미터 등을 환경에 맞게 조정하세요.
- 모델 파일 등은 반드시 설치 경로에 존재해야 합니다.
- RViz 등 시각화는 필요에 따라 추가 실행하세요.
- TF 설정 파일은 카메라 추가/변경 시 캘리브레이션 결과로 업데이트해야 합니다.

---
