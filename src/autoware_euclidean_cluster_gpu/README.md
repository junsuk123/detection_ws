# autoware_euclidean_cluster_gpu

CUDA-accelerated Euclidean clustering for ROS 2 Humble. This package drops into an Autoware-like stack and
subscribes to `sensor_msgs/PointCloud2` with fields `x,y,z`. It publishes:

- A copy of the input cloud with an extra `int32 cluster_id` field (`/perception/clustered/points` by default)
- `vision_msgs/Detection3DArray` axis-aligned bounding boxes (`/perception/clustered/boxes`)

## Build (ROS 2 Humble)

```bash
# In your Autoware workspace
cd ~/autoware/src
# unzip this folder here or git clone your fork
colcon build --packages-select autoware_euclidean_cluster_gpu --cmake-args -DGPU_ARCH=86
source install/setup.bash
```

> Set `-DGPU_ARCH` to your GPU's SM (70=V100, 75=T4, 86=RTX30/Ampere, 89=RTX40/Ada, etc.).

## Run

```bash
ros2 launch autoware_euclidean_cluster_gpu euclidean_cluster_gpu.launch.py   params_file:=`ros2 pkg prefix autoware_euclidean_cluster_gpu`/share/autoware_euclidean_cluster_gpu/config/euclidean_cluster_gpu.param.yaml
```

Or just:

```bash
ros2 launch autoware_euclidean_cluster_gpu euclidean_cluster_gpu.launch.py
```

Tune `voxel_size` (grid binning) and `tolerance` (neighbor radius) for your LiDAR.
Larger `voxel_size` accelerates search but may over-merge clusters; `tolerance` should be roughly
your desired inter-point connection radius (e.g., 0.5~0.8 m for automotive objects).

## Notes

- The kernel uses a hash-grid and iterative label propagation to compute connected components within
  the given `tolerance`. It's designed to be fast and simple to build; no external RAPIDS/CuML required.
- The final labels are compacted and filtered on the CPU to apply `min/max_cluster_size`.
- For best performance, downsample with a voxel grid before clustering (e.g., 0.1–0.2 m).

