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
ros2 launch autoware_cuda_pointcloud_preprocessor cuda_pointcloud_preprocessor.launch.xml
```

---

### 3) 지면 제거 (autoware_ground_segmentation)

- **기능:** 포인트클라우드에서 지면(ground) 제거
- **INPUT TOPIC:** `/pointcloud_noiseremoved`
- **OUTPUT TOPIC:** `/pointcloud_nonground`

```bash
ros2 launch autoware_ground_segmentation scan_ground_filter.launch.py
```

---

### 4) GPU 기반 클러스터링 (autoware_euclidean_cluster_gpu)

- **기능:** ROI 내 포인트클라우드 클러스터링 (CUDA)
- **INPUT TOPIC:** `/pointcloud_nonground`
- **OUTPUT TOPIC:** `/perception/clustered/points_only` (ROI 내 포인트만 포함)

```bash
ros2 launch euclidean_cluster_gpu euclidean_cluster_gpu.launch.py
```

---

### 5) TF 발행 (좌표계 변환)

- **기능:** base_link, velodyne, camera2 간 정적 변환
- **예시:** (실제 값/토픽명은 환경에 맞게 수정)

```bash
ros2 run tf2_ros static_transform_publisher -0.605987064362494 0.186388570753703 0.628623572853739 0 2.2008 -0.623599 velodyne camera2
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
- **INPUT TOPIC:** `/perception/clustered/points_only`, `/yolo_result`
- **OUTPUT TOPIC:** `/yolo_3d_result`

```bash
ros2 launch ultralytics_ros tracker_with_cloud.launch.xml debug:=true
```

---

## 3. 전체 데이터 흐름 요약

```
[LiDAR] 
  └─/velodyne_points
      ↓
[노이즈 제거]
  └─/pointcloud_noiseremoved
      ↓
[지면 제거]
  └─/pointcloud_nonground
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

## 5. Git 저장소(GitHub 등) 연동 및 워크스페이스 동기화

### 대용량 파일 오류(100MB 초과) 해결 방법

- GitHub는 100MB가 넘는 파일(예: rosbag, .db3 등)을 업로드할 수 없습니다.
- 아래와 같이 `.gitignore`에 대용량/불필요 파일(rosbag, 로그, 데이터 등)을 반드시 추가하세요.

#### 예시: .gitignore에 rosbag, db3 등 추가

```bash
echo "
build/
install/
log/
*.pyc
__pycache__/
.env
.venv
models/
*.pt
*.db3
*.bag
*.sqlite3
rosbag2_*/
*.zip
*.tar
*.tar.gz
*.tgz
*.log
*.csv
*.npy
*.npz
*.h5
*.onnx
*.pb
*.pth
*.ckpt
*.weights
" > .gitignore
```

- 이미 커밋된 대용량 파일은 아래 명령으로 git 기록에서 완전히 제거해야 합니다.

#### 이미 커밋된 대용량 파일 완전 삭제

```bash
# 1. BFG Repo-Cleaner 설치 (https://rtyley.github.io/bfg-repo-cleaner/)
sudo apt install openjdk-11-jre
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# 2. git reflog/GC로 안전장치
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 3. 대용량 파일 완전 삭제 (예: *.db3)
java -jar bfg-1.14.0.jar --delete-files *.db3
java -jar bfg-1.14.0.jar --delete-folders rosbag2_ --no-blob-protection

# 4. 다시 GC
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. 강제 푸시
git push --force
```

- 자세한 방법: [https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-a-repositorys-history](https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-a-repositorys-history)

---