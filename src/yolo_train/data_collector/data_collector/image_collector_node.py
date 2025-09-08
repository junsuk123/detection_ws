#!/usr/bin/env python3
# File: data_collector/src/data_collector/collector_node.py

import os
from datetime import datetime, timedelta

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageCollectorNode(Node):
    def __init__(self):
        super().__init__('image_collector')
        self.bridge = CvBridge()

        # --- 파라미터 선언 및 읽어오기 ---
        # image_topic: 구독할 이미지 토픽
        # save_hz: 초당 저장할 프레임 수
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('save_hz', 1.0)

        topic_name = self.get_parameter('image_topic').value
        save_hz    = self.get_parameter('save_hz').value

        # 저장 주기(interval) 계산
        if save_hz > 0:
            self.interval = timedelta(seconds=1.0 / save_hz)
        else:
            self.interval = timedelta(0)

        # --- 결과 저장 폴더 설정 (워크스페이스 루트 기준) ---
        cwd      = os.getcwd()  # ros2 run or launch 시 meta_ws 루트여야 함
        base_dir = os.path.join(cwd, 'imageData')
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_dir = os.path.join(base_dir, now)
        os.makedirs(self.out_dir, exist_ok=True)
        self.get_logger().info(f"Saving images to: {self.out_dir}")

        # --- 이미지 토픽 구독 ---
        self.create_subscription(
            Image,
            topic_name,
            self.image_cb,
            10
        )

        # 내부 카운터 및 타이머 초기화
        self.counter        = 0
        self.last_save_time = datetime.min

    def image_cb(self, msg: Image):
        now = datetime.now()
        # 주기보다 빠른 콜백은 스킵
        if self.interval.total_seconds() > 0 and (now - self.last_save_time) < self.interval:
            return

        # ROS Image → OpenCV BGR
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 파일명 및 저장
        fname = f"image_{self.counter:06d}.png"
        path  = os.path.join(self.out_dir, fname)
        cv2.imwrite(path, cv_img)
        self.get_logger().info(f"Saved {fname}")

        # 카운터 및 타임스탬프 갱신
        self.counter        += 1
        self.last_save_time  = now

def main(args=None):
    rclpy.init(args=args)
    node = ImageCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
