/**
 * ultralytics_ros (logged + patched version, Option-B integrated)
 * - Adds richer logs
 * - Optional LiDAR QoS (reliable vs sensor)
 * - Voxel leaf-size clamp
 * - MASK-to-camera resolution scaling
 * - Optional force_bbox parameter to bypass masks
 * - **Option-B**: GPU clustered cloud 사용 + cluster_id ↔ YOLO bbox/mask 매칭
 * - **Multi-cluster matching**: 상위 복수 클러스터 허용 + 독점 할당 옵션 추가
 *
 * Copyright (C) 2023-2024  Alpaca-zip
 * GNU Affero General Public License v3.0+
 */

#include "tracker_with_cloud_node/tracker_with_cloud_node.h"

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <unordered_map>
#include <limits>
#include <cstring>

namespace {
inline bool isFinite(const pcl::PointXYZ &p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

// LiDAR 포인트(원 프레임) + 클러스터 ID
struct PtLidar { float x, y, z; int cid; };
// 카메라 프레임 + 픽셀 좌표
struct PtCam   { float x, y, z; int cid; int u, v; };

// GPU 노드의 ~/clustered 에서 x,y,z,cluster_id를 읽어온다
bool readXYZCID(const sensor_msgs::msg::PointCloud2& msg, std::vector<PtLidar>& out)
{
  bool has_cid = false;
  for (const auto& f : msg.fields) if (f.name == "cluster_id") { has_cid = true; break; }
  if (!has_cid) return false;

  try {
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");
    sensor_msgs::PointCloud2ConstIterator<int32_t> iter_cid(msg, "cluster_id");

    const size_t N = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);
    out.resize(N);
    for (size_t i = 0; i < N; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_cid) {
      out[i].x = *iter_x; out[i].y = *iter_y; out[i].z = *iter_z; out[i].cid = *iter_cid;
    }
    return true;
  } catch (...) {
    return false;
  }
}

void transformLidarToCam(const std::vector<PtLidar>& in_lidar,
                         std::vector<PtCam>& out_cam,
                         const Eigen::Affine3f& T_lidar_to_cam)
{
  out_cam.resize(in_lidar.size());
  for (size_t i = 0; i < in_lidar.size(); ++i) {
    const auto& p = in_lidar[i];
    Eigen::Vector3f v = T_lidar_to_cam * Eigen::Vector3f(p.x, p.y, p.z);
    out_cam[i].x = v.x(); out_cam[i].y = v.y(); out_cam[i].z = v.z();
    out_cam[i].cid = p.cid; out_cam[i].u = -1; out_cam[i].v = -1;
  }
}

inline bool projectToPixel(const image_geometry::PinholeCameraModel& cam_model,
                           float x, float y, float z, int& u, int& v)
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || z <= 0) return false;
  const cv::Point2d uv = cam_model.project3dToPixel(cv::Point3d(x, y, z));
  const auto sz = cam_model.fullResolution(); // cv::Size
  if (uv.x < 0 || uv.y < 0 || uv.x >= sz.width || uv.y >= sz.height) return false;
  u = static_cast<int>(uv.x); v = static_cast<int>(uv.y);
  return true;
}

// 선택된 CID들의 포인트만 모아 하나의 PointCloud2 생성
sensor_msgs::msg::PointCloud2::SharedPtr build_selected_cloud(
  const sensor_msgs::msg::PointCloud2& in,
  const std::unordered_set<int>& selected_cids)
{
  // 출력은 x,y,z만 복사(필요하면 추가 필드도 복사 가능)
  sensor_msgs::msg::PointCloud2 out;
  out.header = in.header;
  out.height = 1;
  out.is_bigendian = in.is_bigendian;
  out.is_dense = false;

  // 필드 구성: x,y,z(float32)
  out.fields.clear();
  auto add_field = [&](const std::string& name, uint32_t offset){
    sensor_msgs::msg::PointField f; f.name = name; f.offset = offset;
    f.datatype = sensor_msgs::msg::PointField::FLOAT32; f.count = 1; out.fields.push_back(f);
  };
  add_field("x", 0); add_field("y", 4); add_field("z", 8);
  out.point_step = 12;

  // 입력에서 iterator로 읽고, CID 필터링
  sensor_msgs::PointCloud2ConstIterator<float> ix(in, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iy(in, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iz(in, "z");
  sensor_msgs::PointCloud2ConstIterator<int32_t> icid(in, "cluster_id");

  std::vector<uint8_t> buf;
  buf.reserve(static_cast<size_t>(in.width) * 12);

  for (; ix != ix.end(); ++ix, ++iy, ++iz, ++icid) {
    const int cid = *icid;
    if (cid < 0) continue;
    if (selected_cids.find(cid) == selected_cids.end()) continue;
    float x = *ix, y = *iy, z = *iz;
    // append 12 bytes
    const uint8_t* px = reinterpret_cast<const uint8_t*>(&x);
    const uint8_t* py = reinterpret_cast<const uint8_t*>(&y);
    const uint8_t* pz = reinterpret_cast<const uint8_t*>(&z);
    buf.insert(buf.end(), px, px+4);
    buf.insert(buf.end(), py, py+4);
    buf.insert(buf.end(), pz, pz+4);
  }

  out.width = static_cast<uint32_t>(buf.size() / 12);
  out.row_step = out.point_step * out.width;
  out.data = std::move(buf);

  return std::make_shared<sensor_msgs::msg::PointCloud2>(std::move(out));
}

} // namespace

TrackerWithCloudNode::TrackerWithCloudNode() : rclcpp::Node("tracker_with_cloud_node")
{
  // ===== Params =====
  this->declare_parameter<bool>("use_reliable_lidar", false);
  this->declare_parameter<bool>("force_bbox", false); // NEW: bypass masks if true
  this->declare_parameter<int>("min_points_in_bbox", 20);
  this->declare_parameter<double>("min_ratio_in_cluster", 0.0); // 0=off
  // NEW: 멀티-클러스터 매칭 파라미터
  this->declare_parameter<int>("max_clusters_per_det", 1);
  this->declare_parameter<bool>("exclusive_cluster", true);

  bool use_reliable_lidar = this->get_parameter("use_reliable_lidar").as_bool();

  this->declare_parameter<std::string>("camera_info_topic", "camera_info");
  this->declare_parameter<std::string>("lidar_topic", "points_raw");
  this->declare_parameter<std::string>("yolo_result_topic", "yolo_result");
  this->declare_parameter<std::string>("yolo_3d_result_topic", "yolo_3d_result");

  // (아래 파라미터는 Option-B 경로에선 사용하지 않지만 유지)
  this->declare_parameter<float>("cluster_tolerance", 0.5f);
  this->declare_parameter<float>("voxel_leaf_size", 0.5f);
  this->declare_parameter<int>("min_cluster_size", 100);
  this->declare_parameter<int>("max_cluster_size", 25000);

  // Read topics
  this->get_parameter("camera_info_topic", camera_info_topic_);
  this->get_parameter("lidar_topic", lidar_topic_);
  this->get_parameter("yolo_result_topic", yolo_result_topic_);

  RCLCPP_INFO(get_logger(), "[INIT] subscribing: camera_info=%s lidar=%s yolo=%s",
              camera_info_topic_.c_str(), lidar_topic_.c_str(), yolo_result_topic_.c_str());

  // ===== Subscribers & QoS =====
  camera_info_sub_.subscribe(this, camera_info_topic_, rmw_qos_profile_sensor_data);
  if (use_reliable_lidar) {
    lidar_sub_.subscribe(this, lidar_topic_, rmw_qos_profile_default);          // Reliable
  } else {
    lidar_sub_.subscribe(this, lidar_topic_, rmw_qos_profile_sensor_data);      // BestEffort
  }
  yolo_result_sub_.subscribe(this, yolo_result_topic_, rmw_qos_profile_sensor_data);
  sync_ = std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(30);
  sync_->connectInput(camera_info_sub_, lidar_sub_, yolo_result_sub_);
  sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(1.0));
  sync_->registerCallback(std::bind(&TrackerWithCloudNode::syncCallback, this, std::placeholders::_1,
                                    std::placeholders::_2, std::placeholders::_3));

  // ===== Publishers =====
  this->get_parameter("yolo_3d_result_topic", yolo_3d_result_topic_);
  detection3d_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(yolo_3d_result_topic_, 1);
  detection_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("detection_cloud", 1);
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("detection_marker", 1);

  RCLCPP_INFO(get_logger(), "[INIT] publishing: yolo_3d=%s detection_cloud=detection_cloud marker=detection_marker",
              yolo_3d_result_topic_.c_str());

  last_call_time_ = this->now();
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

void TrackerWithCloudNode::syncCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg,
                                        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
                                        const ultralytics_ros::msg::YoloResult::ConstSharedPtr& yolo_result_msg)
{
  rclcpp::Time current_call_time = this->now();
  rclcpp::Duration callback_interval = current_call_time - last_call_time_;
  last_call_time_ = current_call_time;

  RCLCPP_INFO(get_logger(), "[CB] stamps: cloud=%.3f yolo=%.3f cam=%.3f (dt=%.3f)",
              rclcpp::Time(cloud_msg->header.stamp).seconds(),
              rclcpp::Time(yolo_result_msg->header.stamp).seconds(),
              rclcpp::Time(camera_info_msg->header.stamp).seconds(),
              callback_interval.seconds());
  RCLCPP_INFO(get_logger(), "[CB] frames: cloud=%s cam_info_frame=%s",
              cloud_msg->header.frame_id.c_str(), camera_info_msg->header.frame_id.c_str());
  RCLCPP_INFO(get_logger(), "[CB] cloud2: height=%u width=%u step=%u row_step=%u",
              cloud_msg->height, cloud_msg->width, cloud_msg->point_step, cloud_msg->row_step);
  RCLCPP_INFO(get_logger(), "[CB] yolo: detections=%zu masks=%zu",
              yolo_result_msg->detections.detections.size(), yolo_result_msg->masks.size());

  // ===== Camera model =====
  try {
    cam_model_.fromCameraInfo(camera_info_msg);
    const cv::Size fr = cam_model_.fullResolution();
    RCLCPP_INFO(get_logger(), "[CAM] fullResolution=%dx%d K=[fx=%.1f fy=%.1f cx=%.1f cy=%.1f] tfFrame='%s'",
                fr.width, fr.height,
                cam_model_.fx(), cam_model_.fy(), cam_model_.cx(), cam_model_.cy(),
                cam_model_.tfFrame().c_str());
    if (cam_model_.tfFrame().empty()) {
      RCLCPP_WARN(get_logger(), "[CAM] cam_model.tfFrame() is empty. Will use camera_info frame: %s",
                  camera_info_msg->header.frame_id.c_str());
    }
  } catch (const std::exception &e) {
    RCLCPP_ERROR(get_logger(), "[CAM] fromCameraInfo failed: %s", e.what());
    return;
  }

  // ===== Read x,y,z,cid from GPU clustered cloud =====
  std::vector<PtLidar> pts_lidar;
  if (!readXYZCID(*cloud_msg, pts_lidar)) {
    RCLCPP_WARN(get_logger(), "[INPUT] 'cluster_id' field missing on input cloud. Did you set lidar_topic to ~/clustered?");
    return;
  }
  if (pts_lidar.empty()) {
    RCLCPP_WARN(get_logger(), "[INPUT] cloud has 0 points");
    return;
  }

  // ===== Transform points to camera frame (for projection) =====
  const std::string target_frame = cam_model_.tfFrame().empty()
                                   ? camera_info_msg->header.frame_id : cam_model_.tfFrame();

  geometry_msgs::msg::TransformStamped tf_stamped;
  try {
    RCLCPP_INFO(get_logger(), "[TF] lookup %s -> %s at %.3f s",
                cloud_msg->header.frame_id.c_str(), target_frame.c_str(),
                rclcpp::Time(cloud_msg->header.stamp).seconds());
    tf_stamped = tf_buffer_->lookupTransform(
        target_frame, cloud_msg->header.frame_id, cloud_msg->header.stamp,
        tf2::durationFromSec(0.05));
  } catch (tf2::TransformException& e) {
    RCLCPP_WARN(get_logger(), "[TF] %s", e.what());
    return;
  }
  const Eigen::Affine3f T = tf2::transformToEigen(tf_stamped.transform).cast<float>();

  std::vector<PtCam> pts_cam; pts_cam.reserve(pts_lidar.size());
  transformLidarToCam(pts_lidar, pts_cam, T);

  // 한 번만 투영
  const cv::Size fr = cam_model_.fullResolution();
  size_t behind = 0;
  for (auto &p : pts_cam) {
    if (p.z <= 0 || !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) { behind++; continue; }
    int u, v; if (projectToPixel(cam_model_, p.x, p.y, p.z, u, v)) { p.u=u; p.v=v; }
  }
  RCLCPP_INFO(get_logger(), "[PROJ] projected=%zu / %zu (behind=%zu)",
              pts_cam.size() - behind, pts_cam.size(), behind);

  // ===== 클러스터 전체 크기 맵 (ratio 용) =====
  std::unordered_map<int,int> cluster_total; cluster_total.reserve(256);
  for (const auto& p : pts_lidar) if (p.cid >= 0) ++cluster_total[p.cid];

  // ===== 파라미터 =====
  const bool force_bbox = this->get_parameter("force_bbox").as_bool();
  const int min_points_in_bbox = this->get_parameter("min_points_in_bbox").as_int();
  const double min_ratio_in_cluster = this->get_parameter("min_ratio_in_cluster").as_double();
  const int max_clusters_per_det = this->get_parameter("max_clusters_per_det").as_int();
  const bool exclusive_cluster = this->get_parameter("exclusive_cluster").as_bool();

  // ===== YOLO 매칭 & 3D 박스 생성 =====
  vision_msgs::msg::Detection3DArray detection3d_array_msg;
  detection3d_array_msg.header = cloud_msg->header;
  detection3d_array_msg.header.stamp = yolo_result_msg->header.stamp;

  std::unordered_set<int> selected_cids; selected_cids.reserve(64);

  if (yolo_result_msg->detections.detections.empty()) {
    RCLCPP_WARN(get_logger(), "[PROJ] YOLO detections empty.");
  }

  for (size_t i = 0; i < yolo_result_msg->detections.detections.size(); ++i)
  {
    const auto& det2d = yolo_result_msg->detections.detections[i];
    std::unordered_map<int,int> votes; votes.reserve(64);

    if (!force_bbox && !yolo_result_msg->masks.empty() && i < yolo_result_msg->masks.size()) {
      // --- MASK 모드 ---
      RCLCPP_INFO(get_logger(), "[PROJ] det[%zu] use MASK", i);
      cv_bridge::CvImagePtr cv_ptr;
      try {
        cv_ptr = cv_bridge::toCvCopy(yolo_result_msg->masks[i], sensor_msgs::image_encodings::MONO8);
      } catch (cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "[MASK] toCvCopy failed: %s", e.what());
        continue;
      }
      const int cam_w = fr.width, cam_h = fr.height;
      const int mask_w = cv_ptr->image.cols, mask_h = cv_ptr->image.rows;
      const double sx = cam_w > 0 ? static_cast<double>(mask_w) / static_cast<double>(cam_w) : 1.0;
      const double sy = cam_h > 0 ? static_cast<double>(mask_h) / static_cast<double>(cam_h) : 1.0;
      const auto& im = cv_ptr->image;

      size_t hit=0, inside=0;
      for (const auto& p : pts_cam) {
        if (p.cid < 0 || p.u < 0 || p.v < 0) continue;
        const int mx = static_cast<int>(p.u * sx);
        const int my = static_cast<int>(p.v * sy);
        if (mx >= 0 && mx < mask_w && my >= 0 && my < mask_h) {
          inside++;
          if (im.at<uchar>(cv::Point(mx, my)) > 0) { ++votes[p.cid]; ++hit; }
        }
      }
      RCLCPP_INFO(get_logger(), "[MASK] det[%zu] inside=%zu matched=%zu uniq_cid=%zu", i, inside, hit, votes.size());
    } else {
      // --- BBOX 모드 ---
      RCLCPP_INFO(get_logger(), "[PROJ] det[%zu] use BBOX", i);
      const double cx = det2d.bbox.center.position.x;
      const double cy = det2d.bbox.center.position.y;
      const double w  = det2d.bbox.size_x;
      const double h  = det2d.bbox.size_y;
      if (!(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(w) && std::isfinite(h)) || w <= 0 || h <= 0) {
        RCLCPP_WARN(get_logger(), "[BBOX] invalid bbox (cx=%.2f cy=%.2f w=%.2f h=%.2f)", cx, cy, w, h);
        continue;
      }
      const int umin = (int)std::floor(cx - w/2.0);
      const int umax = (int)std::ceil (cx + w/2.0);
      const int vmin = (int)std::floor(cy - h/2.0);
      const int vmax = (int)std::ceil (cy + h/2.0);

      size_t hit=0, inside=0;
      for (const auto& p : pts_cam) {
        if (p.cid < 0 || p.u < 0 || p.v < 0) continue;
        if (p.u >= umin && p.u <= umax && p.v >= vmin && p.v <= vmax) { ++votes[p.cid]; ++hit; }
        if (p.u >= 0 && p.u < fr.width && p.v >= 0 && p.v < fr.height) inside++;
      }
      RCLCPP_INFO(get_logger(), "[BBOX] det[%zu] inside=%zu matched=%zu uniq_cid=%zu", i, inside, hit, votes.size());
    }

    // ---- 득표 상위 복수 클러스터 선택 ----
    std::vector<std::pair<int,int>> cand; cand.reserve(votes.size());
    for (const auto& kv : votes) cand.emplace_back(kv.first, kv.second);
    std::sort(cand.begin(), cand.end(), [](auto& a, auto& b){ return a.second > b.second; });

    int added = 0;
    for (const auto& [cid, cnt] : cand) {
      if (cnt < std::max(1, min_points_in_bbox)) continue;
      if (min_ratio_in_cluster > 0.0) {
        auto it = cluster_total.find(cid);
        if (it != cluster_total.end()) {
          const double ratio = (double)cnt / std::max(1, it->second);
          if (ratio < min_ratio_in_cluster) continue;
        }
      }
      if (exclusive_cluster && selected_cids.find(cid) != selected_cids.end()) continue;

      // 선택된 클러스터의 AABB (LiDAR 프레임 기준)
      pcl::PointXYZ min_pt, max_pt;
      min_pt.x=min_pt.y=min_pt.z= std::numeric_limits<float>::max();
      max_pt.x=max_pt.y=max_pt.z=-std::numeric_limits<float>::max();
      int inlier_cnt=0;
      for (const auto& p : pts_lidar) {
        if (p.cid != cid) continue;
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        if (p.x < min_pt.x) min_pt.x = p.x;
        if (p.y < min_pt.y) min_pt.y = p.y;
        if (p.z < min_pt.z) min_pt.z = p.z;
        if (p.x > max_pt.x) max_pt.x = p.x;
        if (p.y > max_pt.y) max_pt.y = p.y;
        if (p.z > max_pt.z) max_pt.z = p.z;
        ++inlier_cnt;
      }
      if (inlier_cnt==0) continue;

      vision_msgs::msg::Detection3D det3d;
      det3d.header = cloud_msg->header;
      det3d.bbox.center.position.x = (max_pt.x + min_pt.x)*0.5f;
      det3d.bbox.center.position.y = (max_pt.y + min_pt.y)*0.5f;
      det3d.bbox.center.position.z = (max_pt.z + min_pt.z)*0.5f;
      det3d.bbox.center.orientation.w = 1.0; // axis-aligned
      det3d.bbox.size.x = std::max(0.f, max_pt.x - min_pt.x);
      det3d.bbox.size.y = std::max(0.f, max_pt.y - min_pt.y);
      det3d.bbox.size.z = std::max(0.f, max_pt.z - min_pt.z);
      det3d.results = det2d.results; // YOLO class 결과 그대로 복사

      RCLCPP_INFO(get_logger(), "[BBOX3D] det[%zu] center=(%.2f,%.2f,%.2f) size=(%.2f,%.2f,%.2f) cid=%d pts=%d/%d",
                  i,
                  det3d.bbox.center.position.x,
                  det3d.bbox.center.position.y,
                  det3d.bbox.center.position.z,
                  det3d.bbox.size.x, det3d.bbox.size.y, det3d.bbox.size.z,
                  cid, cnt, cluster_total[cid]);

      detection3d_array_msg.detections.push_back(std::move(det3d));
      selected_cids.insert(cid);
      if (++added >= std::max(1, max_clusters_per_det)) break;
    }

    if (added==0) {
      int vmax=0, cid_max=-1; for (auto& kv:votes) if (kv.second>vmax){vmax=kv.second; cid_max=kv.first;}
      RCLCPP_WARN(get_logger(), "[MATCH] det[%zu] no candidate accepted (best_cid=%d best_cnt=%d)", i, cid_max, vmax);
    }
  }

  // ===== 선택된 클러스터들로 합성 포인트클라우드 생성 & 퍼블리시 =====
  auto detection_cloud_msg_ptr = build_selected_cloud(*cloud_msg, selected_cids);
  RCLCPP_INFO(get_logger(), "[PUB] 3D detections=%zu detection_cloud: width=%u height=%u bytes=%zu",
              detection3d_array_msg.detections.size(),
              detection_cloud_msg_ptr->width, detection_cloud_msg_ptr->height, detection_cloud_msg_ptr->data.size());

  auto marker_array_msg = createMarkerArray(detection3d_array_msg, callback_interval.seconds());

  detection3d_pub_->publish(detection3d_array_msg);
  detection_cloud_pub_->publish(*detection_cloud_msg_ptr);
  marker_pub_->publish(marker_array_msg);
}

// ===== 아래 함수들은 Option-B 경로에선 사용하지 않지만, 기존 인터페이스 유지용으로 남겨둠 =====

void TrackerWithCloudNode::transformPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                               pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                               const Eigen::Affine3f& transform)
{
  const int n = static_cast<int>(cloud_in->size());
  cloud_out->clear();
  cloud_out->reserve(n);

  int nan_skipped = 0;
  for (int i = 0; i < n; i++)
  {
    const auto &pt = cloud_in->points[i];
    if (!isFinite(pt)) { nan_skipped++; continue; }
    pcl::PointXYZ p;
    p.x = transform(0,0)*pt.x + transform(0,1)*pt.y + transform(0,2)*pt.z + transform(0,3);
    p.y = transform(1,0)*pt.x + transform(1,1)*pt.y + transform(1,2)*pt.z + transform(1,3);
    p.z = transform(2,0)*pt.x + transform(2,1)*pt.y + transform(2,2)*pt.z + transform(2,3);
    cloud_out->push_back(p);
  }
  RCLCPP_INFO(get_logger(), "[TF] transformPointCloud: in=%d out=%zu nan_skipped=%d", n, cloud_out->size(), nan_skipped);
}

void TrackerWithCloudNode::projectCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr&,
                                        const ultralytics_ros::msg::YoloResult::ConstSharedPtr&,
                                        const std_msgs::msg::Header&,
                                        vision_msgs::msg::Detection3DArray&,
                                        sensor_msgs::msg::PointCloud2&)
{
  // (Option-B 통합으로 사용하지 않음)
  RCLCPP_WARN(get_logger(), "[NOTE] projectCloud() legacy path not used in Option-B build.");
}

void TrackerWithCloudNode::processPointsWithBbox(const pcl::PointCloud<pcl::PointXYZ>::Ptr&,
                                                 const vision_msgs::msg::Detection2D&,
                                                 pcl::PointCloud<pcl::PointXYZ>::Ptr&)
{
  // (미사용)
}

void TrackerWithCloudNode::processPointsWithMask(const pcl::PointCloud<pcl::PointXYZ>::Ptr&,
                                                 const sensor_msgs::msg::Image&,
                                                 pcl::PointCloud<pcl::PointXYZ>::Ptr&)
{
  // (미사용)
}

void TrackerWithCloudNode::createBoundingBox(
    vision_msgs::msg::Detection3DArray&,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr&,
    const std::vector<vision_msgs::msg::ObjectHypothesisWithPose>&)
{
  // (미사용: 본문에서 AABB 직접 생성)
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
TrackerWithCloudNode::downsampleCloudMsg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg)
{
  // (Option-B에서는 원본 cluster_id 유지가 중요하므로 미사용 권장. 남겨둔다.)
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*cloud_msg, *cloud);
  this->get_parameter("voxel_leaf_size", voxel_leaf_size_);
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  voxel_grid.setInputCloud(cloud);
  voxel_grid.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
  voxel_grid.filter(*downsampled_cloud);
  return downsampled_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
TrackerWithCloudNode::cloud2TransformedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                             const std::string& source_frame, const std::string& target_frame,
                                             const rclcpp::Time& stamp)
{
  try {
    geometry_msgs::msg::TransformStamped tf_stamped = tf_buffer_->lookupTransform(target_frame, source_frame, stamp);
    Eigen::Affine3f eigen_transform = tf2::transformToEigen(tf_stamped.transform).cast<float>();
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    transformPointCloud(cloud, transformed_cloud, eigen_transform);
    return transformed_cloud;
  } catch (tf2::TransformException& e) {
    RCLCPP_WARN(this->get_logger(), "[TF] %s (use input cloud)", e.what());
    return cloud;
  }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
TrackerWithCloudNode::euclideanClusterExtraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  // (Option-B에서는 GPU 노드가 이미 클러스터링했으므로 미사용)
  return cloud;
}

visualization_msgs::msg::MarkerArray
TrackerWithCloudNode::createMarkerArray(const vision_msgs::msg::Detection3DArray& detection3d_array_msg,
                                        const double& duration)
{
  visualization_msgs::msg::MarkerArray marker_array_msg;

  for (size_t i = 0; i < detection3d_array_msg.detections.size(); i++)
  {
    if (std::isfinite(detection3d_array_msg.detections[i].bbox.size.x) &&
        std::isfinite(detection3d_array_msg.detections[i].bbox.size.y) &&
        std::isfinite(detection3d_array_msg.detections[i].bbox.size.z))
    {
      visualization_msgs::msg::Marker marker_msg;
      marker_msg.header = detection3d_array_msg.header;
      marker_msg.ns = "detection";
      marker_msg.id = static_cast<int>(i);
      marker_msg.type = visualization_msgs::msg::Marker::CUBE;
      marker_msg.action = visualization_msgs::msg::Marker::ADD;
      marker_msg.pose = detection3d_array_msg.detections[i].bbox.center;
      marker_msg.scale.x = detection3d_array_msg.detections[i].bbox.size.x;
      marker_msg.scale.y = detection3d_array_msg.detections[i].bbox.size.y;
      marker_msg.scale.z = detection3d_array_msg.detections[i].bbox.size.z;
      marker_msg.color.r = 0.0;
      marker_msg.color.g = 1.0;
      marker_msg.color.b = 0.0;
      marker_msg.color.a = 0.5;
      marker_msg.lifetime = rclcpp::Duration(std::chrono::duration<double>(duration));
      marker_array_msg.markers.push_back(marker_msg);
    }
  }

  RCLCPP_INFO(get_logger(), "[MARKER] published markers=%zu", marker_array_msg.markers.size());
  return marker_array_msg;
}

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TrackerWithCloudNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
