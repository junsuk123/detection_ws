#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
from std_msgs.msg import String

class PointCloudDebugger(Node):
    def __init__(self):
        super().__init__('pointcloud_debugger')
        
        # 파라미터 정의
        self.declare_parameter('input_topic', '/points_raw')
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        
        # 구독자 생성
        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.listener_callback,
            10)
        
        # 디버그 메시지 발행자
        self.debug_publisher = self.create_publisher(String, '/pointcloud_debug', 10)
        
        self.get_logger().info(f'디버그 노드가 {input_topic} 토픽을 모니터링합니다.')
        
        # 타이머 생성 (5초마다 상태 체크)
        self.timer = self.create_timer(5.0, self.check_status)
        self.last_msg_time = None
        
    def listener_callback(self, msg):
        self.last_msg_time = self.get_clock().now()
        
        # 포인트 클라우드 정보 로깅
        debug_msg = String()
        debug_msg.data = f"PointCloud 수신: {msg.width * msg.height} 포인트, 프레임: {msg.header.frame_id}"
        self.debug_publisher.publish(debug_msg)
        
        self.get_logger().info(f"PointCloud 수신: {msg.width * msg.height} 포인트, 프레임: {msg.header.frame_id}")
    
    def check_status(self):
        if self.last_msg_time is None:
            self.get_logger().warn('아직 포인트 클라우드 메시지를 수신하지 못했습니다.')
        else:
            now = self.get_clock().now()
            time_diff = (now - self.last_msg_time).nanoseconds / 1e9
            self.get_logger().info(f'마지막 메시지 수신 후 {time_diff:.2f}초 경과')

def main(args=None):
    rclpy.init(args=args)
    
    debugger = PointCloudDebugger()
    
    rclpy.spin(debugger)
    
    debugger.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
