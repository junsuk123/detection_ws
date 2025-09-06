#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>

class ROIPublisherNode : public rclcpp::Node {
public:
  ROIPublisherNode() : Node("roi_publisher_node") {
    // ROI 파라미터 로드
    this->declare_parameter<bool>("use_roi_filter", true);
    this->declare_parameter<double>("roi_x_min", 0.0);
    this->declare_parameter<double>("roi_x_max", 2.0);
    this->declare_parameter<double>("roi_y_min", -1.5);
    this->declare_parameter<double>("roi_y_max", 1.5);
    this->declare_parameter<double>("roi_z_min", -0.5);
    this->declare_parameter<double>("roi_z_max", 1.5);
    this->declare_parameter<std::string>("fixed_frame", "velodyne");
    
    use_roi_filter_ = this->get_parameter("use_roi_filter").as_bool();
    roi_x_min_ = this->get_parameter("roi_x_min").as_double();
    roi_x_max_ = this->get_parameter("roi_x_max").as_double();
    roi_y_min_ = this->get_parameter("roi_y_min").as_double();
    roi_y_max_ = this->get_parameter("roi_y_max").as_double();
    roi_z_min_ = this->get_parameter("roi_z_min").as_double();
    roi_z_max_ = this->get_parameter("roi_z_max").as_double();
    fixed_frame_ = this->get_parameter("fixed_frame").as_string();
    
    // ROI 마커 발행자
    roi_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/visualization/roi_marker", 10);
    
    // ROI 정보 로그
    RCLCPP_INFO(this->get_logger(), "ROI 시각화 노드 시작");
    RCLCPP_INFO(this->get_logger(), "ROI 영역: X=[%.2f, %.2f], Y=[%.2f, %.2f], Z=[%.2f, %.2f]",
                roi_x_min_, roi_x_max_, roi_y_min_, roi_y_max_, roi_z_min_, roi_z_max_);
    
    // 주기적으로 ROI 마커 발행
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&ROIPublisherNode::publishROIMarker, this));
  }
  
private:
  void publishROIMarker() {
    if (!use_roi_filter_) {
      return;
    }
    
    // ROI 박스 마커 생성
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = fixed_frame_;
    marker.header.stamp = this->now();
    marker.ns = "roi";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // ROI 박스 중심 위치
    marker.pose.position.x = (roi_x_min_ + roi_x_max_) / 2.0;
    marker.pose.position.y = (roi_y_min_ + roi_y_max_) / 2.0;
    marker.pose.position.z = (roi_z_min_ + roi_z_max_) / 2.0;
    marker.pose.orientation.w = 1.0;
    
    // ROI 박스 크기
    marker.scale.x = roi_x_max_ - roi_x_min_;
    marker.scale.y = roi_y_max_ - roi_y_min_;
    marker.scale.z = roi_z_max_ - roi_z_min_;
    
    // ROI 박스 색상 (반투명 녹색)
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 0.3;  // 투명도
    
    // 라이프타임 무한대
    marker.lifetime = rclcpp::Duration::from_seconds(0);
    
    // 마커 발행
    roi_marker_pub_->publish(marker);
  }
  
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr roi_marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  
  bool use_roi_filter_;
  double roi_x_min_, roi_x_max_;
  double roi_y_min_, roi_y_max_;
  double roi_z_min_, roi_z_max_;
  std::string fixed_frame_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ROIPublisherNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
