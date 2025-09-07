#!/usr/bin/env python3
# 여러 카메라의 YOLO 결과를 통합하고 클라우드를 처리하는 노드

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2, CameraInfo
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray
import message_filters
from std_msgs.msg import Header
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import TransformException
import threading

class TrackerWithCloudNode(Node):
    def __init__(self):
        super().__init__('tracker_with_cloud_node')
        
        # QoS 설정
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # 파라미터 선언
        self.declare_parameter('lidar_topic', '/pointcloud/clustered')
        self.declare_parameter('yolo_3d_result_topic', '/detection/yolo_3d_result')
        self.declare_parameter('active_cameras', ['camera3'])
        self.declare_parameter('camera_count', 1)
        self.declare_parameter('yolo_result_topics', ['/detection/camera3/yolo_result'])
        self.declare_parameter('camera_info_topics', ['/camera3/camera_info'])
        
        # 파라미터 가져오기
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.yolo_3d_result_topic = self.get_parameter('yolo_3d_result_topic').value
        self.active_cameras = self.get_parameter('active_cameras').value
        self.camera_count = self.get_parameter('camera_count').value
        self.yolo_result_topics = self.get_parameter('yolo_result_topics').value
        self.camera_info_topics = self.get_parameter('camera_info_topics').value
        
        self.get_logger().info(f"활성 카메라: {', '.join(self.active_cameras)}")
        self.get_logger().info(f"카메라 개수: {self.camera_count}")
        
        # TF 설정
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 카메라별 정보 저장소
        self.camera_infos = {}       # 카메라 정보
        self.latest_detections = {}  # 최신 감지 결과
        self.camera_transformations = {}  # 카메라->라이다 변환 행렬
        
        # 라이다 구독자
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            sensor_qos
        )
        
        # 각 카메라에 대한 구독자 설정
        for idx, camera_name in enumerate(self.active_cameras):
            # CameraInfo 구독자
            self.create_subscription(
                CameraInfo,
                self.camera_info_topics[idx],
                lambda msg, cam=camera_name: self.camera_info_callback(msg, cam),
                10
            )
            
            # 카메라별 YOLO 결과 구독자
            self.create_subscription(
                Detection2DArray,
                self.yolo_result_topics[idx],
                lambda msg, cam=camera_name: self.yolo_detection_callback(msg, cam),
                10
            )
        
        # 3D 결과 발행자
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            self.yolo_3d_result_topic,
            10
        )
        
        # TF 변환 캐시를 주기적으로 업데이트
        self.tf_timer = self.create_timer(1.0, self.update_transformations)
        
        self.get_logger().info('다중 카메라 3D 트래커 초기화 완료')

    def camera_info_callback(self, msg, camera_name):
        """카메라 정보 콜백"""
        self.camera_infos[camera_name] = msg
        self.get_logger().debug(f"{camera_name} 카메라 정보 업데이트됨")

    def yolo_detection_callback(self, msg, camera_name):
        """YOLO 감지 결과 콜백"""
        self.latest_detections[camera_name] = msg
        self.get_logger().debug(f"{camera_name} YOLO 감지 업데이트됨: {len(msg.detections)}개 객체")

    def lidar_callback(self, msg):
        """클러스터링된 라이다 포인트 클라우드 콜백"""
        # 모든 카메라의 결과를 합쳐서 처리
        self.process_all_detections(msg)

    def update_transformations(self):
        """카메라에서 라이다로의 변환을 주기적으로 업데이트"""
        for camera_name in self.active_cameras:
            try:
                # 카메라->라이다 변환 확인
                transform = self.tf_buffer.lookup_transform(
                    'velodyne',  # target frame (라이다 프레임)
                    camera_name,  # source frame (카메라 프레임)
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                self.camera_transformations[camera_name] = transform
                self.get_logger().debug(f"{camera_name}->velodyne TF 변환 업데이트됨")
            except TransformException as ex:
                self.get_logger().warning(f"{camera_name}->velodyne TF 변환 실패: {ex}")

    def process_all_detections(self, cloud_msg):
        """모든 카메라의 감지 결과를 처리하여 3D 마커로 변환"""
        combined_markers = MarkerArray()
        
        # 현재 활성화된 모든 카메라의 감지 결과 처리
        for camera_name in self.active_cameras:
            # 이 카메라에 필요한 모든 정보가 있는지 확인
            if (camera_name not in self.latest_detections or
                camera_name not in self.camera_infos or
                camera_name not in self.camera_transformations):
                continue
            
            # 이 카메라의 감지 결과에서 3D 마커 생성
            camera_markers = self.create_3d_markers_from_detections(
                self.latest_detections[camera_name],
                self.camera_infos[camera_name],
                self.camera_transformations[camera_name],
                camera_name,
                cloud_msg
            )
            
            # 마커 ID 충돌 방지
            for idx, marker in enumerate(camera_markers.markers):
                marker.id += 1000 * (self.active_cameras.index(camera_name) + 1)
                combined_markers.markers.append(marker)
        
        # 마커 발행
        if combined_markers.markers:
            self.marker_publisher.publish(combined_markers)

    def create_3d_markers_from_detections(self, detections, camera_info, transform, camera_name, cloud_msg):
        """감지 결과를 3D 마커로 변환"""
        # 여기에 2D 감지를 3D로 투영하고 마커를 생성하는 코드 구현
        # 이 부분은 기존 코드를 유지하고 필요에 따라 카메라별 처리 로직 추가
        
        # 예시 코드 (실제 구현은 기존 로직에 맞게 수정 필요)
        markers = MarkerArray()
        
        # TODO: 2D 감지 결과와 포인트 클라우드를 결합하여 3D 마커 생성
        
        return markers

def main(args=None):
    rclpy.init(args=args)
    node = TrackerWithCloudNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
