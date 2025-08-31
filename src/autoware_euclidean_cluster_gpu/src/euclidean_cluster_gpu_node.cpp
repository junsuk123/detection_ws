// Apache-2.0
// Euclidean cluster GPU node — Tracking v2 (Optimized, Humble-fixes)
// - 기존 기능 보존 + 성능 최적화 + Humble 빌드 에러/워닝 수정
//   * 퍼블리시: 역참조 publish (Humble 호환)
//   * QoS: SensorDataQoS().keep_last(1).best_effort()
//   * 마커/박스 decimation 파라미터
//   * GPU 메모리 재사용 + (옵션) pinned host + async memcpy
//   * compute_centroids/to_boxes 최적화 (compact 라벨 가정)
//   * std::max 템플릿 인자 명시(int)로 캐스팅 문제 해결
//   * 람다/if 들여쓰기 경고 해결

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_field_conversion.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <builtin_interfaces/msg/duration.hpp>

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <cmath>
#include <unordered_map>
#include <cstdint>

#include "autoware/euclidean_cluster_gpu/euclidean_cluster_gpu.hpp"

// ========================= helper: compress_and_filter_labels =========================
// (별도 TU였던 구현을 본 파일에 통합)
std::vector<int32_t> compress_and_filter_labels(
  const std::vector<int32_t>& raw, int min_sz, int max_sz)
{
  std::unordered_map<int32_t, int> counts;
  counts.reserve(raw.size()/4+1);
  for (auto v : raw) if (v >= 0) ++counts[v];
  // build mapping for those within [min,max]
  std::unordered_map<int32_t, int32_t> remap;
  remap.reserve(counts.size());
  int32_t cur = 0;
  for (auto& kv : counts) {
    if (kv.second >= min_sz && kv.second <= max_sz) {
      remap[kv.first] = cur++;
    }
  }
  std::vector<int32_t> out(raw.size(), -1);
  for (size_t i=0;i<raw.size();++i) {
    auto it = remap.find(raw[i]);
    if (it != remap.end()) out[i] = it->second;
  }
  return out;
}

// ========================= local utilities =========================
namespace {

struct PointXYZ { float x, y, z; };

bool read_xyz(const sensor_msgs::msg::PointCloud2& msg, std::vector<PointXYZ>& out) {
  int x_off=-1,y_off=-1,z_off=-1;
  for (auto& f : msg.fields) {
    if (f.name == "x") x_off = f.offset;
    else if (f.name == "y") y_off = f.offset;
    else if (f.name == "z") z_off = f.offset;
  }
  if (x_off<0||y_off<0||z_off<0) return false;
  out.resize(static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height));
  const uint8_t* data = msg.data.data();
  for (size_t i=0;i<out.size();++i) {
    const uint8_t* p = data + i*msg.point_step;
    float x = *reinterpret_cast<const float*>(p + x_off);
    float y = *reinterpret_cast<const float*>(p + y_off);
    float z = *reinterpret_cast<const float*>(p + z_off);
    out[i] = {x,y,z};
  }
  return true;
}

sensor_msgs::msg::PointCloud2::SharedPtr add_cluster_id_field(
  const sensor_msgs::msg::PointCloud2& in,
  const std::vector<int32_t>& cluster_ids)
{
  auto out = std::make_shared<sensor_msgs::msg::PointCloud2>();
  out->header = in.header; out->height=in.height; out->width=in.width;
  out->is_bigendian=in.is_bigendian; out->is_dense=in.is_dense;
  out->fields = in.fields;
  sensor_msgs::msg::PointField fld; fld.name="cluster_id"; fld.offset=in.point_step;
  fld.datatype=sensor_msgs::msg::PointField::INT32; fld.count=1; out->fields.push_back(fld);
  out->point_step=in.point_step+4; out->row_step=out->point_step*out->width;
  out->data.resize(static_cast<size_t>(out->row_step) * static_cast<size_t>(out->height));
  const size_t N = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);
  for (size_t i=0;i<N;++i){
    const uint8_t* src=in.data.data()+i*in.point_step; uint8_t* dst=out->data.data()+i*out->point_step;
    memcpy(dst,src,in.point_step);
    int32_t cid = (i<cluster_ids.size())?cluster_ids[i]:-1;
    *reinterpret_cast<int32_t*>(dst+fld.offset)=cid;
  }
  return out;
}

sensor_msgs::msg::PointCloud2::SharedPtr build_clustered_only_cloud(
  const sensor_msgs::msg::PointCloud2& in,
  const std::vector<int32_t>& ids)
{
  const size_t N = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);
  size_t keep=0; for(size_t i=0;i<ids.size();++i) if(ids[i]>=0) ++keep;
  auto out=std::make_shared<sensor_msgs::msg::PointCloud2>();
  out->header=in.header; out->height=1; out->width=(uint32_t)keep;
  out->is_bigendian=in.is_bigendian; out->is_dense=false;
  out->fields=in.fields; sensor_msgs::msg::PointField fld; fld.name="cluster_id"; fld.offset=in.point_step;
  fld.datatype=sensor_msgs::msg::PointField::INT32; fld.count=1; out->fields.push_back(fld);
  out->point_step=in.point_step+4; out->row_step=out->point_step*out->width; out->data.resize(out->row_step);
  uint8_t* dst=out->data.data(); size_t w=0;
  for(size_t i=0;i<N && w<keep;++i){ if(i<ids.size() && ids[i]>=0){ const uint8_t* src=in.data.data()+i*in.point_step; memcpy(dst,src,in.point_step); *reinterpret_cast<int32_t*>(dst+fld.offset)=ids[i]; dst+=out->point_step; ++w; } }
  return out;
}

visualization_msgs::msg::MarkerArray::SharedPtr boxes_to_markers(
  const vision_msgs::msg::Detection3DArray& boxes,
  float r=1.0f,float g=0.2f,float b=0.2f,float a=0.5f)
{
  auto arr=std::make_shared<visualization_msgs::msg::MarkerArray>(); int id=0;
  for(const auto& det:boxes.detections){ visualization_msgs::msg::Marker m; m.header=boxes.header; m.ns="gpu_clusters"; m.id=id++; m.type=visualization_msgs::msg::Marker::CUBE; m.action=visualization_msgs::msg::Marker::ADD; m.pose=det.bbox.center; m.scale=det.bbox.size; m.color.r=r; m.color.g=g; m.color.b=b; m.color.a=a; m.lifetime=builtin_interfaces::msg::Duration(); arr->markers.push_back(std::move(m)); }
  return arr;
}

// ========================= 연산 최적화 버전: compact labels 가정 =========================
struct Centroid { int label; float x,y,z; int count; };

// compute_centroids_compact는 그대로 사용
static std::vector<Centroid> compute_centroids_compact(
  const std::vector<PointXYZ>& pts, const std::vector<int32_t>& ids)
{
  // ids는 compress_and_filter_labels를 거쳐 0..K-1 범위의 compact 라벨이라고 가정
  int K = 0; for (auto v: ids) if (v>=0 && v+1>K) K=v+1;
  std::vector<double> sx(K,0.0), sy(K,0.0), sz(K,0.0); std::vector<int> cnt(K,0);
  for(size_t i=0;i<ids.size();++i){ int lab=ids[i]; if(lab<0) continue; sx[lab]+=pts[i].x; sy[lab]+=pts[i].y; sz[lab]+=pts[i].z; cnt[lab]++; }
  std::vector<Centroid> cs; cs.reserve(K);
  for(int lab=0; lab<K; ++lab){ if(cnt[lab]>0){ float c=(float)cnt[lab]; cs.push_back({lab, (float)(sx[lab]/c), (float)(sy[lab]/c), (float)(sz[lab]/c), cnt[lab]}); } }
  // 결정적 정렬(기존 규칙 유지)
  std::sort(cs.begin(), cs.end(), [](const Centroid &A, const Centroid &B){
    if (A.x != B.x) return A.x < B.x;
    if (A.y != B.y) return A.y < B.y;
    if (A.z != B.z) return A.z < B.z;
    return A.label < B.label;
  });
  return cs;
}

// ========================= Node =========================
class EuclideanClusterGpuNode : public rclcpp::Node {
public:
  explicit EuclideanClusterGpuNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions())
  : Node("euclidean_cluster_gpu", opts)
  {
    using std::placeholders::_1;
    
    RCLCPP_INFO(get_logger(), "=== Starting EuclideanClusterGpuNode ===");
    
    // Topics
    declare_parameter<std::string>("input", "/velodyne_points");
    declare_parameter<std::string>("output_clusters", "~/clustered");
    declare_parameter<std::string>("output_boxes", "~/boxes");
    declare_parameter<std::string>("output_clusters_only", "~/clustered_only");
    declare_parameter<std::string>("output_markers", "~/cluster_markers");

    // Clustering params
    declare_parameter<double>("voxel_size", 0.35);
    declare_parameter<double>("tolerance", 0.8);
    declare_parameter<int>("min_cluster_size", 5);
    declare_parameter<int>("max_cluster_size", 100000);
    declare_parameter<int>("max_iterations", 18);
    declare_parameter<int>("max_points", 2000000);

    // Tracking params
    declare_parameter<double>("base_match_dist", 1.2);          // was track_match_dist
    declare_parameter<double>("vel_gate_mps", 2.0);             // dynamic gate per dt
    declare_parameter<double>("pos_ema_alpha", 0.5);            // was track_ema_alpha
    declare_parameter<double>("vel_ema_alpha", 0.6);            // NEW
    declare_parameter<double>("size_weight", 0.15);             // NEW
    declare_parameter<int>("track_miss_tolerance", 15);

    // External config (compat) params
    declare_parameter<double>("track_match_dist", 1.2);      // config alias for base_match_dist
    declare_parameter<bool>("publish_debug_markers", true);   // config alias for publish_markers
    declare_parameter<bool>("publish_tracking_info", false);  // throttle 로그 on/off

    // Visualization / output decimation
    declare_parameter<bool>("publish_markers", true);
    declare_parameter<bool>("publish_boxes", true);
    declare_parameter<int>("marker_decimation", 5);  // publish every N frames
    declare_parameter<int>("boxes_decimation", 1);   // 1 = every frame

    // ROI 파라미터 선언 및 읽기
    declare_parameter<double>("roi_x_min", -50.0);
    declare_parameter<double>("roi_x_max",  50.0);
    declare_parameter<double>("roi_y_min", -30.0);
    declare_parameter<double>("roi_y_max",  30.0);
    declare_parameter<double>("roi_z_min", -2.5);
    declare_parameter<double>("roi_z_max",  2.5);

    in_topic_ = get_parameter("input").as_string();
    out_topic_ = get_parameter("output_clusters").as_string();
    box_topic_ = get_parameter("output_boxes").as_string();
    out_only_topic_ = get_parameter("output_clusters_only").as_string();
    markers_topic_ = get_parameter("output_markers").as_string();

    params_.voxel_size = (float)get_parameter("voxel_size").as_double();
    params_.tolerance = (float)get_parameter("tolerance").as_double();
    params_.min_cluster_size = get_parameter("min_cluster_size").as_int();
    params_.max_cluster_size = get_parameter("max_cluster_size").as_int();
    params_.max_iterations = get_parameter("max_iterations").as_int();
    params_.max_points = get_parameter("max_points").as_int();

    // prefer track_match_dist if provided (otherwise base_match_dist)
    {
      const double cfg_track = get_parameter("track_match_dist").as_double();
      const double cfg_base  = get_parameter("base_match_dist").as_double();
      base_match_dist_ = (cfg_track != 1.2 ? cfg_track : cfg_base);
    }
    vel_gate_mps_        = get_parameter("vel_gate_mps").as_double();
    pos_ema_alpha_       = std::clamp(get_parameter("pos_ema_alpha").as_double(), 0.0, 1.0);
    vel_ema_alpha_       = std::clamp(get_parameter("vel_ema_alpha").as_double(), 0.0, 1.0);
    size_weight_         = std::max(0.0, get_parameter("size_weight").as_double());
    track_miss_tolerance_= get_parameter("track_miss_tolerance").as_int();

    publish_markers_   = get_parameter("publish_markers").as_bool();
    publish_boxes_     = get_parameter("publish_boxes").as_bool();
    // config alias overrides if set differently
    {
      const bool dbg = get_parameter("publish_debug_markers").as_bool();
      publish_markers_ = dbg; // give priority to config alias
    }
    publish_tracking_info_ = get_parameter("publish_tracking_info").as_bool();
    // Humble: std::max 템플릿 명시 + 캐스팅 필요(as_int -> int64)
    marker_decimation_ = std::max<int>(1, static_cast<int>(get_parameter("marker_decimation").as_int()));
    boxes_decimation_  = std::max<int>(1, static_cast<int>(get_parameter("boxes_decimation").as_int()));

    roi_x_min_ = get_parameter("roi_x_min").as_double();
    roi_x_max_ = get_parameter("roi_x_max").as_double();
    roi_y_min_ = get_parameter("roi_y_min").as_double();
    roi_y_max_ = get_parameter("roi_y_max").as_double();
    roi_z_min_ = get_parameter("roi_z_min").as_double();
    roi_z_max_ = get_parameter("roi_z_max").as_double();

    RCLCPP_INFO(get_logger(), "Parameters loaded:");
    RCLCPP_INFO(get_logger(), "  input: %s", in_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  voxel_size: %.3f", params_.voxel_size);
    RCLCPP_INFO(get_logger(), "  tolerance: %.3f", params_.tolerance);
    RCLCPP_INFO(get_logger(), "  min_cluster_size: %d", params_.min_cluster_size);
    RCLCPP_INFO(get_logger(), "  max_cluster_size: %d", params_.max_cluster_size);

    // QoS settings to match the publisher exactly
    auto qos = rclcpp::QoS(5)  // Match publisher's KEEP_LAST (5)
               .reliability(rclcpp::ReliabilityPolicy::BestEffort)
               .durability(rclcpp::DurabilityPolicy::Volatile)
               .history(rclcpp::HistoryPolicy::KeepLast);

    RCLCPP_INFO(get_logger(), "Creating subscription with QoS: KEEP_LAST(5), BEST_EFFORT, VOLATILE");
    
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      in_topic_, qos, std::bind(&EuclideanClusterGpuNode::onCloud, this, _1));
      
    RCLCPP_INFO(get_logger(), "Subscription created successfully");
    
    pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(out_topic_, qos);
    pub_boxes_ = create_publisher<vision_msgs::msg::Detection3DArray>(box_topic_, qos);
    pub_cloud_only_ = create_publisher<sensor_msgs::msg::PointCloud2>(out_only_topic_, qos);
    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>(markers_topic_, qos);
    roi_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("roi_marker", 1);

    RCLCPP_INFO(get_logger(), "All publishers created successfully");

    RCLCPP_INFO(get_logger(), "GPU node v2(opt): vsize=%.2f tol=%.2f base_match=%.2f vel_gate=%.2f pos_ema=%.2f vel_ema=%.2f size_w=%.2f",
      params_.voxel_size, params_.tolerance, base_match_dist_, vel_gate_mps_, pos_ema_alpha_, vel_ema_alpha_, size_weight_);
    
    // Add debug info about topics
    RCLCPP_INFO(get_logger(), "Subscribing to: %s", in_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Publishing to: %s", out_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Waiting for point cloud data...");

    // Add a timer to periodically check subscription status
    check_timer_ = create_wall_timer(
      std::chrono::seconds(2),  // Check every 2 seconds instead of 5
      [this]() {
        size_t pub_count = sub_->get_publisher_count();
        RCLCPP_INFO(get_logger(), "Publisher count on %s: %zu, frame_count: %zu", 
                    in_topic_.c_str(), pub_count, frame_count_);
        
        if (pub_count == 0) {
          RCLCPP_WARN(get_logger(), "No publishers detected on topic %s", in_topic_.c_str());
        }
      });
      
    RCLCPP_INFO(get_logger(), "=== EuclideanClusterGpuNode initialization completed ===");
  }

  ~EuclideanClusterGpuNode() override {
    // Simplified cleanup following the preprocessor pattern
    if (stream_created_) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
    
    if (d_pts_) {
      cudaFree(d_pts_);
      d_pts_ = nullptr;
    }
    
    if (d_labels_) {
      cudaFree(d_labels_);
      d_labels_ = nullptr;
    }
    
#ifdef USE_PINNED_HOST
    if (h_pinned_) {
      cudaFreeHost(h_pinned_);
      h_pinned_ = nullptr;
    }
#endif
  }

private:
  // Simplified capacity management following preprocessor pattern
  void ensure_capacity(int M) {
    if (M <= capacity_ && d_pts_ && d_labels_) return;
    
    // Free existing memory
    if (d_pts_) {
      cudaFree(d_pts_);
      d_pts_ = nullptr;
    }
    
    if (d_labels_) {
      cudaFree(d_labels_);
      d_labels_ = nullptr;
    }
    
#ifdef USE_PINNED_HOST
    if (h_pinned_) {
      cudaFreeHost(h_pinned_);
      h_pinned_ = nullptr;
    }
#endif
    
    // Allocate new memory
    cudaError_t err1 = cudaMalloc((void**)&d_pts_, M * sizeof(autoware::euclidean_cluster_gpu::Float3));
    cudaError_t err2 = cudaMalloc((void**)&d_labels_, M * sizeof(int32_t));
    
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "Failed to allocate GPU memory");
      if (d_pts_) { cudaFree(d_pts_); d_pts_ = nullptr; }
      if (d_labels_) { cudaFree(d_labels_); d_labels_ = nullptr; }
      return;
    }
    
#ifdef USE_PINNED_HOST
    cudaError_t err3 = cudaHostAlloc((void**)&h_pinned_, 
                                    M * sizeof(autoware::euclidean_cluster_gpu::Float3), 
                                    cudaHostAllocDefault);
    if (err3 != cudaSuccess) {
      h_pinned_ = nullptr; // Continue without pinned memory
    }
#endif
    
    capacity_ = M;
    
    // Create stream if needed
    if (!stream_created_) {
      if (cudaStreamCreate(&stream_) == cudaSuccess) {
        stream_created_ = true;
      }
    }
  }

  void publish_roi_marker(const std_msgs::msg::Header& header) {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "roi";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = (roi_x_min_ + roi_x_max_) * 0.5;
    marker.pose.position.y = (roi_y_min_ + roi_y_max_) * 0.5;
    marker.pose.position.z = (roi_z_min_ + roi_z_max_) * 0.5;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = std::abs(roi_x_max_ - roi_x_min_);
    marker.scale.y = std::abs(roi_y_max_ - roi_y_min_);
    marker.scale.z = std::abs(roi_z_max_ - roi_z_min_);
    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.05f; // 투명하게
    marker.lifetime = rclcpp::Duration(0,0); // forever
    roi_marker_pub_->publish(marker);
  }

void onCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    try {
      RCLCPP_INFO(get_logger(), "=== Processing point cloud frame %zu ===", frame_count_);
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, 
                           "Received point cloud with %d points", 
                           static_cast<int>(msg->width * msg->height));
      
      std::vector<PointXYZ> pts;
      if(!read_xyz(*msg, pts)){
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "PointCloud2 missing x/y/z fields");
        return;
      }

      // ROI 필터링 적용
      std::vector<PointXYZ> roi_pts;
      std::vector<size_t> roi_indices;
      roi_pts.reserve(pts.size());
      roi_indices.reserve(pts.size());
      for(size_t i=0; i<pts.size(); ++i) {
        const auto& p = pts[i];
        if(p.x >= roi_x_min_ && p.x <= roi_x_max_ &&
           p.y >= roi_y_min_ && p.y <= roi_y_max_ &&
           p.z >= roi_z_min_ && p.z <= roi_z_max_) {
          roi_pts.push_back(p);
          roi_indices.push_back(i);
        }
      }
      RCLCPP_INFO(get_logger(), "ROI filtered: %zu -> %zu points", pts.size(), roi_pts.size());

      publish_roi_marker(msg->header);

      const int N = (int)roi_pts.size();
      if(N==0) {
        RCLCPP_DEBUG(get_logger(), "No points in ROI");
        return;
      }
      if(N > params_.max_points){ 
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, 
                            "Too many points: %d > %d; truncating", N, params_.max_points); 
      }
      const int M = std::min(N, params_.max_points);
      RCLCPP_INFO(get_logger(), "Processing %d points (original: %d)", M, N);

      last_stamp_ = msg->header.stamp;

      RCLCPP_INFO(get_logger(), "Ensuring GPU memory capacity for %d points", M);
      ensure_capacity(M);
      
      // Check if memory allocation succeeded
      if (!d_pts_ || !d_labels_) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, 
                             "GPU memory not available, skipping frame");
        return;
      }
      
      RCLCPP_INFO(get_logger(), "GPU memory allocated successfully");

      // Add error checking for CUDA memory operations
      cudaError_t err = cudaSuccess;

#ifdef USE_PINNED_HOST
      if (h_pinned_) {
        RCLCPP_INFO(get_logger(), "Using pinned memory for data transfer");
        for(int i=0;i<M;++i){ 
          h_pinned_[i] = {roi_pts[i].x, roi_pts[i].y, roi_pts[i].z}; 
        }
        err = cudaMemcpyAsync(d_pts_, h_pinned_, 
                             M*sizeof(autoware::euclidean_cluster_gpu::Float3), 
                             cudaMemcpyHostToDevice, stream_);
      } else {
        RCLCPP_INFO(get_logger(), "Using regular memory for data transfer (pinned failed)");
        std::vector<autoware::euclidean_cluster_gpu::Float3> host(M);
        for(int i=0;i<M;++i){ 
          host[i]={roi_pts[i].x,roi_pts[i].y,roi_pts[i].z}; 
        }
        err = cudaMemcpyAsync(d_pts_, host.data(), 
                             M*sizeof(autoware::euclidean_cluster_gpu::Float3), 
                             cudaMemcpyHostToDevice, stream_);
      }
#else
      RCLCPP_INFO(get_logger(), "Using regular memory for data transfer");
      std::vector<autoware::euclidean_cluster_gpu::Float3> host(M);
      for(int i=0;i<M;++i){ 
        host[i]={roi_pts[i].x,roi_pts[i].y,roi_pts[i].z}; 
      }
      err = cudaMemcpyAsync(d_pts_, host.data(), 
                           M*sizeof(autoware::euclidean_cluster_gpu::Float3), 
                           cudaMemcpyHostToDevice, stream_);
#endif

      if (err != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "Failed to copy points to GPU: %s", cudaGetErrorString(err));
        return;
      }
      
      RCLCPP_INFO(get_logger(), "Points copied to GPU successfully");

      err = cudaMemsetAsync(d_labels_, 0xFF, M*sizeof(int32_t), stream_);
      if (err != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "Failed to initialize labels: %s", cudaGetErrorString(err));
        return;
      }
      
      RCLCPP_INFO(get_logger(), "Labels initialized successfully");

      // Wait for memory operations to complete before running kernel
      err = cudaStreamSynchronize(stream_);
      if (err != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "Stream sync failed before kernel: %s", cudaGetErrorString(err));
        return;
      }
      
      RCLCPP_INFO(get_logger(), "Stream synchronized, starting CUDA kernel");

      // 커널 실행 (기본 스트림/동일 스트림 가정)
      auto kernel_start = std::chrono::high_resolution_clock::now();
      autoware::euclidean_cluster_gpu::run_euclidean_clustering_cuda(d_pts_, M, params_, d_labels_, stream_);
      
      // Check for kernel launch errors
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "CUDA kernel launch failed: %s", cudaGetErrorString(err));
        return;
      }
      
      RCLCPP_INFO(get_logger(), "CUDA kernel launched successfully");

      std::vector<int32_t> h_labels(M);
      err = cudaMemcpyAsync(h_labels.data(), d_labels_, M*sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
      if (err != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "Failed to copy labels from GPU: %s", cudaGetErrorString(err));
        return;
      }
      
      RCLCPP_INFO(get_logger(), "Labels copied from GPU, synchronizing stream");
      
      err = cudaStreamSynchronize(stream_);
      if (err != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "Final stream sync failed: %s", cudaGetErrorString(err));
        return;
      }
      
      auto kernel_end = std::chrono::high_resolution_clock::now();
      auto kernel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end - kernel_start);
      RCLCPP_INFO(get_logger(), "CUDA kernel completed in %ld ms", kernel_duration.count());

      RCLCPP_INFO(get_logger(), "Starting post-processing");
      
      // Check first few labels for debugging
      int valid_labels = 0;
      for(int i = 0; i < std::min(10, M); ++i) {
        if(h_labels[i] >= 0) valid_labels++;
      }
      RCLCPP_INFO(get_logger(), "Sample labels (first 10): valid=%d, example values: %d, %d, %d", 
                  valid_labels,
                  M > 0 ? h_labels[0] : -999,
                  M > 1 ? h_labels[1] : -999, 
                  M > 2 ? h_labels[2] : -999);

    // 라벨 후처리
    std::vector<int32_t> compact = compress_and_filter_labels(h_labels, params_.min_cluster_size, params_.max_cluster_size);
    RCLCPP_INFO(get_logger(), "Labels compressed and filtered");

    // 클러스터 중심 계산 (트래킹 없이)
    auto cents = compute_centroids_compact(roi_pts, compact);
    RCLCPP_INFO(get_logger(), "Computed %zu centroids", cents.size());

    // stable_ids는 compact 라벨 그대로 사용
    std::vector<int32_t> stable_ids = compact;

    RCLCPP_INFO(get_logger(), "Creating output messages");

    // === ROI 영역 포인트만 담긴 PointCloud2 생성 ===
    // roi_indices를 이용해 원본 msg에서 ROI 포인트만 추출
    auto roi_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    *roi_cloud = *msg;
    roi_cloud->width = roi_pts.size();
    roi_cloud->height = 1;
    roi_cloud->row_step = roi_cloud->point_step * roi_cloud->width;
    roi_cloud->data.resize(roi_cloud->row_step);
    for(size_t i=0; i<roi_indices.size(); ++i) {
      const size_t orig_idx = roi_indices[i];
      const uint8_t* src = msg->data.data() + orig_idx * msg->point_step;
      uint8_t* dst = roi_cloud->data.data() + i * roi_cloud->point_step;
      memcpy(dst, src, msg->point_step);
    }

    // 클러스터 id 필드 추가 (ROI 포인트만)
    auto cloud_msg = add_cluster_id_field(*roi_cloud, stable_ids);

    // 클러스터된 포인트만 (ROI 내에서)
    auto cloud_only = build_clustered_only_cloud(*roi_cloud, stable_ids);

    // boxes/markers는 decimation
    bool do_boxes  = publish_boxes_   && ((frame_count_ % boxes_decimation_)  == 0);
    bool do_markers= publish_markers_ && ((frame_count_ % marker_decimation_) == 0);

    vision_msgs::msg::Detection3DArray::SharedPtr boxes;
    visualization_msgs::msg::MarkerArray::SharedPtr markers;
    if (do_boxes || do_markers) {
      RCLCPP_INFO(get_logger(), "Creating boxes and markers");
      boxes = std::make_shared<vision_msgs::msg::Detection3DArray>();
      // to_boxes 최적화: AABB 벡터 누적 (compact 라벨 가정)
      int K = 0; for (auto v: stable_ids) if (v>=0 && v+1>K) K=v+1;
      boxes->header = msg->header;
      std::vector<float> xmin(K,  std::numeric_limits<float>::max());
      std::vector<float> ymin(K,  std::numeric_limits<float>::max());
      std::vector<float> zmin(K,  std::numeric_limits<float>::max());
      std::vector<float> xmax(K, -std::numeric_limits<float>::max());
      std::vector<float> ymax(K, -std::numeric_limits<float>::max());
      std::vector<float> zmax(K, -std::numeric_limits<float>::max());
      std::vector<char>  seen(K, 0);
      for(size_t i=0;i<stable_ids.size();++i){ int id=stable_ids[i]; if(id<0) continue; const auto &p=pts[i];
        if (p.x < xmin[id]) { xmin[id] = p.x; }
        if (p.y < ymin[id]) { ymin[id] = p.y; }
        if (p.z < zmin[id]) { zmin[id] = p.z; }
        if (p.x > xmax[id]) { xmax[id] = p.x; }
        if (p.y > ymax[id]) { ymax[id] = p.y; }
        if (p.z > zmax[id]) { zmax[id] = p.z; }
        seen[id] = 1; }
      boxes->detections.reserve(K);
      for(int id=0; id<K; ++id){ if(!seen[id]) continue; vision_msgs::msg::Detection3D det; det.header=msg->header;
        det.bbox.size.x=(xmax[id]-xmin[id]); det.bbox.size.y=(ymax[id]-ymin[id]); det.bbox.size.z=(zmax[id]-zmin[id]);
        det.bbox.center.position.x=(xmax[id]+xmin[id])*0.5f; det.bbox.center.position.y=(ymax[id]+ymin[id])*0.5f; det.bbox.center.position.z=(zmax[id]+zmin[id])*0.5f;
        boxes->detections.push_back(std::move(det)); }
      if (do_markers) { markers = boxes_to_markers(*boxes); }
    }

    RCLCPP_INFO(get_logger(), "Publishing messages");

    // 퍼블리시(역참조: Humble 호환)
    pub_cloud_->publish(*cloud_msg);
    pub_cloud_only_->publish(*cloud_only);
    if (do_boxes)   pub_boxes_->publish(*boxes);
    if (do_markers) pub_markers_->publish(*markers);

    int kept=0,max_id=-1; for(int v:stable_ids){ if(v>=0){ ++kept; if(v>max_id) max_id=v; }} int clusters=max_id+1;
    
    RCLCPP_INFO(get_logger(), "Published all messages successfully");
    RCLCPP_INFO(get_logger(), "GPU v2(opt): clusters=%d, kept=%d/%d", clusters, kept, (int)stable_ids.size());

    ++frame_count_;
    RCLCPP_INFO(get_logger(), "=== Completed frame %zu ===", frame_count_ - 1);
    }
    catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Exception in point cloud processing: %s", e.what());
    }
    catch (...) {
      RCLCPP_ERROR(get_logger(), "Unknown exception in point cloud processing");
    }
  }

  // Topics
  std::string in_topic_, out_topic_, box_topic_, out_only_topic_, markers_topic_;

  // ROS IO
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr pub_boxes_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_only_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr roi_marker_pub_;

  // 실시간 클러스터링만, 트래킹 상태 제거
  rclcpp::Time last_stamp_{};
  size_t frame_count_ = 0;

  // Params
  autoware::euclidean_cluster_gpu::ClusteringParams params_{};
  double base_match_dist_ = 1.2;
  double vel_gate_mps_ = 2.0;
  double pos_ema_alpha_ = 0.5;
  double vel_ema_alpha_ = 0.6;
  double size_weight_ = 0.15;
  int    track_miss_tolerance_ = 15;

  bool publish_markers_ = true;
  bool publish_boxes_   = true;
  bool publish_tracking_info_ = false;
  int  marker_decimation_ = 5;
  int  boxes_decimation_  = 1;

  // ROI 파라미터 멤버
  double roi_x_min_{-50.0}, roi_x_max_{50.0};
  double roi_y_min_{-30.0}, roi_y_max_{30.0};
  double roi_z_min_{-2.5},  roi_z_max_{2.5};

  // CUDA resources (reused)
  autoware::euclidean_cluster_gpu::Float3* d_pts_ = nullptr;
  int32_t* d_labels_ = nullptr;
  int capacity_ = 0;
#ifdef USE_PINNED_HOST
  autoware::euclidean_cluster_gpu::Float3* h_pinned_ = nullptr;
#endif
  cudaStream_t stream_{}; bool stream_created_ = false;
  rclcpp::TimerBase::SharedPtr check_timer_;
}; // <-- 이 중괄호가 반드시 필요합니다!
} // <-- 이 중괄호가 반드시 필요합니다!
int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opts; opts.use_intra_process_comms(true);
  auto node = std::make_shared<EuclideanClusterGpuNode>(opts);

  // 멀티스레드 실행기로 퍼블리시/콜백 겹치기 허용
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}

