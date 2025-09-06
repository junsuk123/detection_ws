#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <csignal>  // 시그널 처리
#include <unistd.h> // usleep 사용
#include <stdlib.h> // exit() 함수

#include "range_image_clustering/range_image_clustering.h"

// 전역 변수로 종료 플래그 선언 - 컴파일 오류 해결
bool g_force_shutdown = false;

// 강제 종료 핸들러 - 강력한 버전
void force_exit_handler(int /*signum*/) {
    std::cerr << "\n강제 종료 요청. 프로세스를 즉시 종료합니다.\n" << std::endl;
    // 종료 플래그 설정
    g_force_shutdown = true;
    // 기본 핸들러로 복원하여 두 번째 Ctrl+C에서 즉시 종료
    std::signal(SIGINT, SIG_DFL);
}

// 노드 클래스
class RangeImageClusteringNode : public rclcpp::Node {
public:
    RangeImageClusteringNode() : Node("range_image_clustering_node") {
        // 1. 파라미터 선언 - 기본값은 의미 없음, YAML에서 덮어씀
        declare_node_parameters();
        
        // 2. 파라미터 로드
        load_parameters();
        
        // 3. 클러스터링 파라미터 설정
        setup_clustering();
        
        // 4. 구독자 및 발행자 생성
        setup_publishers_and_subscribers();
        
        // 5. 데이터 없음 체크 타이머 - 더 간격 늘림
        timer_ = this->create_wall_timer(
            std::chrono::seconds(10), // 5초->10초로 증가
            std::bind(&RangeImageClusteringNode::checkTopicStatus, this));
        
        // 6. 인터럽트 체크 타이머 - 필요 없음
    }

private:
    // 파라미터 선언
    void declare_node_parameters() {
        this->declare_parameter<double>("angle_threshold", 0.1);
        this->declare_parameter<double>("distance_threshold", 0.5);
        this->declare_parameter<int>("min_cluster_size", 10);
        this->declare_parameter<int>("max_cluster_size", 5000);
        
        this->declare_parameter<std::string>("input_topic", "/points_raw");
        this->declare_parameter<std::string>("output_cloud_topic", "/clustered_cloud");
        this->declare_parameter<std::string>("output_markers_topic", "/cluster_markers");
        
        this->declare_parameter<bool>("use_sensor_data_qos", true);
        this->declare_parameter<int>("qos_history_depth", 5);
        
        this->declare_parameter<int>("range_image_width", 1800);
        this->declare_parameter<int>("range_image_height", 32);
        
        this->declare_parameter<double>("marker_lifetime", 0.1);
        this->declare_parameter<double>("point_size", 2.0);
        
        this->declare_parameter<std::string>("fixed_frame", "base_link");
        
        this->declare_parameter<bool>("enable_adaptive_threshold", false);
        this->declare_parameter<double>("min_distance_for_adaptive", 5.0);
        this->declare_parameter<double>("max_adaptive_factor", 3.0);
        
        // ROI 파라미터 선언
        this->declare_parameter<bool>("use_roi_filter", true);
        this->declare_parameter<double>("roi_x_min", 0.0);
        this->declare_parameter<double>("roi_x_max", 2.0);
        this->declare_parameter<double>("roi_y_min", -1.5);
        this->declare_parameter<double>("roi_y_max", 1.5);
        this->declare_parameter<double>("roi_z_min", -0.5);
        this->declare_parameter<double>("roi_z_max", 1.5);
    }

    // 파라미터 로드
    void load_parameters() {
        // YAML 파일에서 로드된 값 가져오기
        angle_threshold_ = this->get_parameter("angle_threshold").as_double();
        distance_threshold_ = this->get_parameter("distance_threshold").as_double();
        min_cluster_size_ = this->get_parameter("min_cluster_size").as_int();
        max_cluster_size_ = this->get_parameter("max_cluster_size").as_int();
        
        input_topic_ = this->get_parameter("input_topic").as_string();
        output_cloud_topic_ = this->get_parameter("output_cloud_topic").as_string();
        output_markers_topic_ = this->get_parameter("output_markers_topic").as_string();
        
        use_sensor_data_qos_ = this->get_parameter("use_sensor_data_qos").as_bool();
        qos_history_depth_ = this->get_parameter("qos_history_depth").as_int();
        
        range_image_width_ = this->get_parameter("range_image_width").as_int();
        range_image_height_ = this->get_parameter("range_image_height").as_int();
        
        marker_lifetime_ = this->get_parameter("marker_lifetime").as_double();
        point_size_ = this->get_parameter("point_size").as_double();
        
        fixed_frame_ = this->get_parameter("fixed_frame").as_string();
        
        enable_adaptive_threshold_ = this->get_parameter("enable_adaptive_threshold").as_bool();
        min_distance_for_adaptive_ = this->get_parameter("min_distance_for_adaptive").as_double();
        max_adaptive_factor_ = this->get_parameter("max_adaptive_factor").as_double();
        
        // ROI 파라미터 로드
        use_roi_filter_ = this->get_parameter("use_roi_filter").as_bool();
        roi_x_min_ = this->get_parameter("roi_x_min").as_double();
        roi_x_max_ = this->get_parameter("roi_x_max").as_double();
        roi_y_min_ = this->get_parameter("roi_y_min").as_double();
        roi_y_max_ = this->get_parameter("roi_y_max").as_double();
        roi_z_min_ = this->get_parameter("roi_z_min").as_double();
        roi_z_max_ = this->get_parameter("roi_z_max").as_double();
        
        // ROI 정보 로그 출력
        if (use_roi_filter_) {
            RCLCPP_INFO(this->get_logger(), "ROI 필터 활성화: X=[%.1f, %.1f], Y=[%.1f, %.1f], Z=[%.1f, %.1f]",
                      roi_x_min_, roi_x_max_, roi_y_min_, roi_y_max_, roi_z_min_, roi_z_max_);
        } else {
            RCLCPP_INFO(this->get_logger(), "ROI 필터 비활성화: 전체 포인트 클라우드 사용");
        }
        
        // 파라미터 설정 출력
        RCLCPP_INFO(this->get_logger(), "===== 파라미터 설정(YAML에서 로드됨) =====");
        RCLCPP_INFO(this->get_logger(), "최소 클러스터 크기: %d", min_cluster_size_);
        RCLCPP_INFO(this->get_logger(), "입력 토픽: %s, 각도: %.3f rad, 거리: %.3f m", 
                    input_topic_.c_str(), angle_threshold_, distance_threshold_);
    }

    // 클러스터링 파라미터 설정
    void setup_clustering() {
        range_image_clustering::ClusteringParams params;
        params.angle_threshold = static_cast<float>(angle_threshold_);
        params.distance_threshold = static_cast<float>(distance_threshold_);
        params.min_cluster_size = min_cluster_size_;
        params.max_cluster_size = max_cluster_size_;
        params.range_image_width = range_image_width_;
        params.range_image_height = range_image_height_;
        
        // 적응형 임계값 설정 (있는 경우)
        if (enable_adaptive_threshold_) {
            params.enable_adaptive_threshold = true;
            params.min_distance_for_adaptive = static_cast<float>(min_distance_for_adaptive_);
            params.max_adaptive_factor = static_cast<float>(max_adaptive_factor_);
            
            RCLCPP_INFO(this->get_logger(), "적응형 임계값 활성화: 시작 거리=%.1f, 최대 계수=%.1f", 
                        min_distance_for_adaptive_, max_adaptive_factor_);
        }
        
        clustering_.setParams(params);
    }

    // 구독자 및 발행자 설정
    void setup_publishers_and_subscribers() {
        // QoS 프로파일 설정
        rclcpp::QoS sub_qos = use_sensor_data_qos_ ? 
            rclcpp::QoS(rclcpp::SensorDataQoS()).keep_last(qos_history_depth_) : 
            rclcpp::QoS(qos_history_depth_);
        
        // 구독자 생성
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, sub_qos, 
            std::bind(&RangeImageClusteringNode::pointCloudCallback, this, std::placeholders::_1));
        
        // 발행자 생성
        cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_cloud_topic_, 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(output_markers_topic_, 10);
        
        RCLCPP_INFO(this->get_logger(), "토픽 구독 중: %s (QoS: %s)", 
                    input_topic_.c_str(), use_sensor_data_qos_ ? "SensorDataQoS" : "기본 QoS");
    }

    void checkTopicStatus() {
        static int warning_count = 0;
        warning_count++;
        
        if (warning_count <= 3) {
            RCLCPP_WARN(this->get_logger(), "포인트 클라우드 데이터가 수신되지 않았습니다.");
        }
    }
  
    // 점유 메모리 체크 함수 (디버깅용)
    void check_memory_usage() {
        // Linux에서만 작동
        FILE* file = fopen("/proc/self/status", "r");
        if (file) {
            char line[128];
            while (fgets(line, 128, file) != NULL) {
                if (strncmp(line, "VmRSS:", 6) == 0) {
                    // 메모리 사용량 로그
                    RCLCPP_DEBUG(this->get_logger(), "메모리 사용량: %s", line+6);
                    break;
                }
            }
            fclose(file);
        }
    }
    
    // 포인트 클라우드 콜백 함수 (로깅 최소화)
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // 타이머 초기화
        if (timer_) {
            timer_->cancel();
            timer_ = nullptr;
        }
        
        // 로깅 최소화
        static int msg_count = 0;
        msg_count++;
        
        // 10번에 한 번만 상세 로깅
        bool verbose_logging = (msg_count % 10 == 0);
        
        if (verbose_logging) {
            RCLCPP_INFO(this->get_logger(), "포인트 클라우드 처리 - 크기: %u x %u = %u 포인트", 
                    msg->width, msg->height, msg->width * msg->height);
        }
        
        try {
            // 인터럽트 체크
            if (g_force_shutdown) {
                RCLCPP_INFO(this->get_logger(), "종료 요청 감지됨");
                rclcpp::shutdown();
                return;
            }
            
            // 포인트 클라우드 변환
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*msg, *cloud);
            
            // ROI 필터링 적용
            pcl::PointCloud<pcl::PointXYZI>::Ptr roi_filtered_cloud = applyROIFilter(cloud);
            
            // 원본 클라우드 저장 (작은 클러스터 처리에 필요)
            input_cloud_ = cloud;
            
            // 필터링된 포인트 수 로깅
            if (verbose_logging) {
                float filter_ratio = cloud->size() > 0 ? 100.0f * (cloud->size() - roi_filtered_cloud->size()) / cloud->size() : 0.0f;
                RCLCPP_INFO(this->get_logger(), "ROI 필터링: %zu -> %zu 포인트 (%.1f%% 제거)",
                          cloud->size(), roi_filtered_cloud->size(), filter_ratio);
            }
            
            // 필터링된 클라우드로 클러스터링 수행
            clustering_.process(roi_filtered_cloud);
            
            // 클러스터링된 클라우드 가져오기 (기본 방식)
            pcl::PointCloud<pcl::PointXYZI>::Ptr clustered_cloud_intensity(new pcl::PointCloud<pcl::PointXYZI>);
            clustering_.getClusteredCloudWithIntensity(clustered_cloud_intensity);
            
            // 추가: 모든 클러스터 정보 가져오기 (작은 것 포함)
            std::vector<pcl::PointIndices> all_clusters;
            clustering_.getAllClusters(all_clusters);
            
            // 최소 크기 미달 클러스터 포함 (시각화 목적)
            includeSmallClustersInResult(clustered_cloud_intensity, all_clusters);
            
            if (verbose_logging) {
                RCLCPP_INFO(this->get_logger(), "클러스터링 결과: %zu 포인트 (모든 클러스터 포함)", 
                          clustered_cloud_intensity->points.size());
            }
            
            // 클러스터 마커 가져오기
            visualization_msgs::msg::MarkerArray markers;
            clustering_.getClusterMarkers(markers, msg->header.frame_id, marker_lifetime_);
            
            // 발행 처리 - PointCloud2 메시지 직접 수정하여 cluster_id 필드 추가
            if (!clustered_cloud_intensity->points.empty()) {
                sensor_msgs::msg::PointCloud2 output_cloud;
                pcl::toROSMsg(*clustered_cloud_intensity, output_cloud);
                output_cloud.header = msg->header;
                
                // cluster_id 필드 추가 (intensity 값을 복사)
                sensor_msgs::msg::PointField cluster_id_field;
                cluster_id_field.name = "cluster_id";
                cluster_id_field.offset = output_cloud.fields[3].offset;  // intensity와 같은 위치 사용
                cluster_id_field.datatype = output_cloud.fields[3].datatype;  // float32
                cluster_id_field.count = 1;
                
                // 기존 필드에 cluster_id 필드 추가
                output_cloud.fields.push_back(cluster_id_field);
                
                // 디버깅을 위해 필드 정보 출력
                if (verbose_logging) {
                    RCLCPP_INFO(this->get_logger(), "출력 클라우드 필드 정보:");
                    for (const auto& field : output_cloud.fields) {
                        RCLCPP_INFO(this->get_logger(), " - 필드: %s (오프셋: %d, 타입: %d)", 
                                  field.name.c_str(), field.offset, field.datatype);
                    }
                }
                
                cloud_pub_->publish(output_cloud);
                
                if (verbose_logging) {
                    RCLCPP_INFO(this->get_logger(), "클러스터링된 클라우드 발행 완료 - 포인트: %zu, 클러스터: %zu", 
                              clustered_cloud_intensity->points.size(), markers.markers.size());
                }
            }
            
            if (!markers.markers.empty()) {
                marker_pub_->publish(markers);
                
                if (verbose_logging) {
                    RCLCPP_INFO(this->get_logger(), "마커 %zu개 발행", markers.markers.size());
                }
            }
            
            // 주기적으로 메모리 사용량 확인
            if (msg_count % 50 == 0) {
                check_memory_usage();
            }
        } 
        catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "처리 중 오류: %s", e.what());
        }
        
        // 잠시 CPU 양보하여 다른 작업이 진행되도록 함
        usleep(1000); // 1ms 대기
    }
    
    // 클러스터링에 사용되는 원본 포인트 클라우드 저장
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_;
    
    // 변수들
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    range_image_clustering::RangeImageClustering clustering_;
    
    // 파라미터 저장 변수
    double angle_threshold_;
    double distance_threshold_;
    int min_cluster_size_;
    int max_cluster_size_;
    std::string input_topic_;
    std::string output_cloud_topic_;
    std::string output_markers_topic_;
    bool use_sensor_data_qos_;
    int qos_history_depth_;
    int range_image_width_;
    int range_image_height_;
    double marker_lifetime_;
    double point_size_;
    std::string fixed_frame_;
    bool enable_adaptive_threshold_;
    double min_distance_for_adaptive_;
    double max_adaptive_factor_;
    
    // ROI 파라미터 변수
    bool use_roi_filter_;
    double roi_x_min_;
    double roi_x_max_;
    double roi_y_min_;
    double roi_y_max_;
    double roi_z_min_;
    double roi_z_max_;
    
    // 클러스터링된 작은 클러스터 추가 처리 메소드 - 선언 추가
    void includeSmallClustersInResult(pcl::PointCloud<pcl::PointXYZI>::Ptr& clustered_cloud,
                                     const std::vector<pcl::PointIndices>& all_clusters) {
        // 모든 클러스터 정보 기록 (통계용)
        int total_clusters = 0;
        int rejected_clusters = 0;
        int single_point_clusters = 0;
        
        for (const auto& cluster : all_clusters) {
            total_clusters++;
            if (cluster.indices.size() == 1) {
                single_point_clusters++;
            }
            if (cluster.indices.size() < static_cast<size_t>(min_cluster_size_)) {
                rejected_clusters++;
            }
        }
        
        if (total_clusters > 0) {
            RCLCPP_INFO(this->get_logger(), "클러스터 통계: 전체=%d, 거부=%d (%.1f%%), 단일포인트=%d (%.1f%%)",
                total_clusters, rejected_clusters, 
                100.0 * rejected_clusters / total_clusters,
                single_point_clusters,
                100.0 * single_point_clusters / total_clusters);
        }
        
        // 단일 포인트 클러스터를 결과에 포함 (intensity = 0)
        for (const auto& cluster : all_clusters) {
            if (cluster.indices.size() < static_cast<size_t>(min_cluster_size_)) {
                for (const auto& idx : cluster.indices) {
                    pcl::PointXYZI point;
                    point.x = input_cloud_->points[idx].x;
                    point.y = input_cloud_->points[idx].y;
                    point.z = input_cloud_->points[idx].z;
                    point.intensity = 0.0f; // 0 = 유효하지 않은 클러스터
                    clustered_cloud->points.push_back(point);
                }
            }
        }
        
        clustered_cloud->width = clustered_cloud->points.size();
        clustered_cloud->height = 1;
        clustered_cloud->is_dense = true;
    }
    
    // ROI 필터링 함수 추가
    pcl::PointCloud<pcl::PointXYZI>::Ptr applyROIFilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        
        // ROI 필터가 비활성화되어 있으면 원본 클라우드 반환
        if (!use_roi_filter_) {
            return input_cloud;
        }
        
        // 각 포인트 검사하여 ROI 내부인 경우만 추가
        for (const auto& point : input_cloud->points) {
            if (point.x >= roi_x_min_ && point.x <= roi_x_max_ &&
                point.y >= roi_y_min_ && point.y <= roi_y_max_ &&
                point.z >= roi_z_min_ && point.z <= roi_z_max_) {
                filtered_cloud->points.push_back(point);
            }
        }
        
        filtered_cloud->width = filtered_cloud->points.size();
        filtered_cloud->height = 1;
        filtered_cloud->is_dense = input_cloud->is_dense;
        
        return filtered_cloud;
    }
};

int main(int argc, char **argv) {
    // 강력한 종료 핸들러 등록
    std::signal(SIGINT, force_exit_handler);
    std::signal(SIGTERM, force_exit_handler);
    
    // ROS 초기화 시 파라미터 디버깅 로그 비활성화
    // 명시적 로깅 레벨 설정 제거
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<RangeImageClusteringNode>();
    
    // 간단한 스핀 루프 - 인터럽트 체크 포함
    while (rclcpp::ok() && !g_force_shutdown) {
        rclcpp::spin_some(node);
        
        // CPU 점유율 줄이기 위해 짧은 대기 추가
        usleep(1000); // 1ms 대기
    }
    
    // 종료 처리
    if (g_force_shutdown) {
        std::cerr << "사용자 요청으로 종료합니다." << std::endl;
    }
    
    rclcpp::shutdown();
    return 0;
}