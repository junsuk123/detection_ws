#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import struct

class TestPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('test_point_cloud_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, '/test_points_raw', 10)
        timer_period = 1.0  # 초 단위
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('테스트 포인트 클라우드 발행자 시작')

    def timer_callback(self):
        # 테스트용 포인트 클라우드 생성 (원통 모양)
        points = []
        num_circles = 20
        points_per_circle = 100
        radius = 5.0
        
        for i in range(num_circles):
            z = float(i) / 2.0 - 5.0
            for j in range(points_per_circle):
                angle = 2.0 * 3.14159 * float(j) / float(points_per_circle)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                points.append([x, y, z, 100])  # x, y, z, intensity
        
        # 또 다른 물체 추가 (구 모양)
        sphere_center = [10.0, 0.0, 0.0]
        sphere_radius = 3.0
        sphere_points = 1000
        
        for i in range(sphere_points):
            theta = 2.0 * 3.14159 * np.random.random()
            phi = 3.14159 * np.random.random()
            x = sphere_center[0] + sphere_radius * np.sin(phi) * np.cos(theta)
            y = sphere_center[1] + sphere_radius * np.sin(phi) * np.sin(theta)
            z = sphere_center[2] + sphere_radius * np.cos(phi)
            points.append([x, y, z, 200])  # 다른 intensity 값
        
        # PointCloud2 메시지 생성
        msg = PointCloud2()
        msg.header = std_msgs.msg.Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "lidar_frame"  # RViz에서 사용할 프레임 ID
        
        msg.height = 1
        msg.width = len(points)
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        
        # 바이트 데이터로 변환
        buffer = []
        for p in points:
            buffer.append(struct.pack('ffff', p[0], p[1], p[2], p[3]))
        
        msg.data = b''.join(buffer)
        msg.is_dense = True
        
        # 메시지 발행
        self.publisher_.publish(msg)
        self.get_logger().info(f'테스트 포인트 클라우드 발행: {len(points)} 포인트')

def main(args=None):
    rclpy.init(args=args)
    test_publisher = TestPointCloudPublisher()
    rclpy.spin(test_publisher)
    test_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
