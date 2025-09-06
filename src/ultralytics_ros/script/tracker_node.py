#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ultralytics_ros
# Copyright (C) 2023-2024  Alpaca-zip
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import cv_bridge
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult
import re


class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")
        
        # 파라미터 선언
        self.declare_parameter("yolo_model", "best.pt")
        self.declare_parameter("model_path", "/home/j/detection_ws/src/ultralytics_ros/models")
        self.declare_parameter("input_topic", "image_raw")
        self.declare_parameter("result_topic", "yolo_result")
        self.declare_parameter("result_image_topic", "yolo_image")
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("classes", list(range(80)))
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("device", "cuda:0")  # 기본값을 GPU로 변경
        self.declare_parameter("result_conf", True)
        self.declare_parameter("result_line_width", 1)
        self.declare_parameter("result_font_size", 1)
        self.declare_parameter("result_font", "Arial.ttf")
        self.declare_parameter("result_labels", True)
        self.declare_parameter("result_boxes", True)
        self.declare_parameter("force_autocast", False)  # GPU 메모리 최적화를 위한 옵션

        # 모델 파일 경로 설정
        path = get_package_share_directory("ultralytics_ros")
        yolo_model = self.get_parameter("yolo_model").get_parameter_value().string_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value

        # 모델 경로가 절대 경로가 아닌 경우 처리
        if not os.path.isabs(yolo_model):
            if model_path:
                yolo_model = os.path.join(model_path, yolo_model)
            else:
                yolo_model = os.path.join(path, "models", yolo_model)

        # 모델 로드 및 분석
        try:
            # 1. 먼저 .pt 파일을 분석하여 기본 정보 추출
            self.model_info = self._analyze_model(yolo_model)
            self.get_logger().info(f"Loading YOLO model: {yolo_model}")
            self.get_logger().info(f"Initial model analysis: Version {self.model_info['version']}, "
                                  f"Type {self.model_info['type']}, Size {self.model_info['size']}")
            
            # device 파라미터 가져오기
            device = self.get_parameter("device").get_parameter_value().string_value
            
            # 2. 모델 로드
            self.model = YOLO(yolo_model)
            
            # 3. 로드된 모델 객체에서 추가 정보 분석
            self._update_model_info_after_load()
            
            # 4. 모델 최적화 (YOLOv8의 경우)
            if self.model_info['version'] == 8:
                self.model.fuse()
                if device and device.startswith('cuda') and hasattr(self.model, 'warmup'):
                    try:
                        self.model.warmup()  # GPU 워밍업
                    except Exception as e:
                        self.get_logger().warn(f"Model warmup failed: {e}")
                
            self.get_logger().info(f"Successfully loaded model: {yolo_model}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise e
            
        # OpenCV 브릿지 초기화
        self.bridge = cv_bridge.CvBridge()
        
        # 세그멘테이션 모델 확인
        self.use_segmentation = self.model_info['type'] == 'seg'
        
        # GPU 메모리 최적화 옵션 (작은 모델에는 불필요할 수 있음)
        self.force_autocast = self.get_parameter("force_autocast").get_parameter_value().bool_value
        
        # 입출력 토픽 설정
        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        result_topic = self.get_parameter("result_topic").get_parameter_value().string_value
        result_image_topic = self.get_parameter("result_image_topic").get_parameter_value().string_value
        
        # 토픽 구독 및 발행 설정
        self.create_subscription(Image, input_topic, self.image_callback, 1)
        self.results_pub = self.create_publisher(YoloResult, result_topic, 1)
        self.result_image_pub = self.create_publisher(Image, result_image_topic, 1)

    def _parse_model_name(self, model_path):
        """모델 파일명에서 버전, 크기, 타입 정보 추출"""
        filename = os.path.basename(model_path)
        
        # 기본값 설정
        info = {
            'version': 8,  # 기본 버전
            'size': 's',   # 기본 크기
            'type': 'det'  # 기본 타입 (detection)
        }
        
        # YOLOv8 또는 YOLOv11 패턴 확인
        v_match = re.search(r'yolov(\d+)', filename.lower())
        if v_match:
            info['version'] = int(v_match.group(1))
            
        # 모델 크기 (n, s, m, l, x) 확인
        size_match = re.search(r'yolov\d+([nsmltx])', filename.lower())
        if size_match:
            info['size'] = size_match.group(1)
            
        # 모델 타입 확인 (seg, pose, cls, det)
        if '-seg' in filename.lower():
            info['type'] = 'seg'
        elif '-pose' in filename.lower():
            info['type'] = 'pose'
        elif '-cls' in filename.lower():
            info['type'] = 'cls'
            
        return info

    # _parse_model_name 메서드 대체 (모델 파일 분석 기능)
    def _analyze_model(self, model_path):
        """모델 파일(.pt)에서 직접 버전, 크기, 타입 정보 추출"""
        # 기본값 설정 (파일 분석 실패시 사용)
        info = {
            'version': 8,  # 기본 버전
            'size': 's',   # 기본 크기
            'type': 'det'  # 기본 타입 (detection)
        }
        
        # 1. 먼저 파일명에서 기본 정보 추출 (fallback용)
        filename = os.path.basename(model_path)
        
        # 버전 추출 (YOLOv8 또는 YOLOv11 등)
        v_match = re.search(r'yolov(\d+)', filename.lower())
        if v_match:
            info['version'] = int(v_match.group(1))
        
        # 크기 추출 (n, s, m, l, x)
        size_match = re.search(r'yolov\d+([nsmltx])', filename.lower())
        if size_match:
            info['size'] = size_match.group(1)
        
        # 타입 추출 (seg, pose, cls)
        if '-seg' in filename.lower():
            info['type'] = 'seg'
        elif '-pose' in filename.lower():
            info['type'] = 'pose'
        elif '-cls' in filename.lower():
            info['type'] = 'cls'
            
        try:
            # 2. PT 파일 직접 분석 (더 정확한 방법)
            import torch
            from pathlib import Path
            
            # PyTorch 2.6 호환성을 위해 weights_only=False 지정
            model_data = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 모델 구조 정보 분석
            if 'model' in model_data and hasattr(model_data['model'], 'yaml'):
                # YOLOv8/YOLOv5 스타일 모델
                yaml_data = model_data['model'].yaml
                
                # 모델 타입 확인
                if 'task' in yaml_data:
                    if yaml_data['task'] == 'segment':
                        info['type'] = 'seg'
                    elif yaml_data['task'] == 'pose':
                        info['type'] = 'pose'
                    elif yaml_data['task'] == 'classify':
                        info['type'] = 'cls'
                
                # 모델 크기 추정 (파라미터 수 기반)
                if 'parameters' in model_data and isinstance(model_data['parameters'], int):
                    param_count = model_data['parameters']
                    # 대략적인 파라미터 수에 따른 모델 크기 추정
                    if param_count < 3_000_000:
                        info['size'] = 'n'  # nano
                    elif param_count < 12_000_000:
                        info['size'] = 's'  # small
                    elif param_count < 30_000_000:
                        info['size'] = 'm'  # medium
                    elif param_count < 70_000_000:
                        info['size'] = 'l'  # large
                    else:
                        info['size'] = 'x'  # xlarge
            
            # YOLO 버전 확인 (가능한 경우)
            if 'model_type' in model_data and 'yolo' in str(model_data['model_type']).lower():
                version_match = re.search(r'yolov?(\d+)', str(model_data['model_type']).lower())
                if version_match:
                    info['version'] = int(version_match.group(1))
            
            # YOLOv11 특정 구조 확인
            if any('v11' in k for k in model_data.keys() if isinstance(k, str)):
                info['version'] = 11
                
            self.get_logger().info(f"Model analysis from .pt file: {info}")
            
        except Exception as e:
            self.get_logger().warn(f"Failed to analyze model file directly, using filename-based info: {e}")
    
        return info

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 파라미터 읽기
        conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        classes = self.get_parameter("classes").get_parameter_value().integer_array_value
        tracker = self.get_parameter("tracker").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value or None
        
        try:
            # GPU 메모리 최적화가 필요한 경우 (큰 모델)
            if self.force_autocast and device and device.startswith('cuda'):
                import torch
                with torch.cuda.amp.autocast():
                    results = self._run_inference(cv_image, conf_thres, iou_thres, 
                                                max_det, classes, tracker, device)
            else:
                results = self._run_inference(cv_image, conf_thres, iou_thres, 
                                            max_det, classes, tracker, device)
                                        
            if results is not None:
                yolo_result_msg = YoloResult()
                yolo_result_image_msg = Image()
                yolo_result_msg.header = msg.header
                yolo_result_image_msg.header = msg.header
                yolo_result_msg.detections = self.create_detections_array(results)
                yolo_result_image_msg = self.create_result_image(results)
                if self.use_segmentation:
                    yolo_result_msg.masks = self.create_segmentation_masks(results)
                self.results_pub.publish(yolo_result_msg)
                self.result_image_pub.publish(yolo_result_image_msg)
            
        except torch.cuda.OutOfMemoryError:
            self.get_logger().error("CUDA out of memory! Try using a smaller model or increasing voxel_leaf_size")
        except Exception as e:
            self.get_logger().error(f"Error during inference: {str(e)}")

    def _run_inference(self, image, conf_thres, iou_thres, max_det, classes, tracker, device):
        """버전별 추론 메서드"""
        try:
            # YOLOv8과 YOLOv11 모두 같은 API 구조를 사용하지만, 
            # 향후 API가 변경될 경우 여기서 분기 처리
            if hasattr(self.model, 'track'):
                # 추론 실행 - 모든 모델 호환
                return self.model.track(
                    source=image,
                    conf=conf_thres,
                    iou=iou_thres,
                    max_det=max_det,
                    classes=classes,
                    tracker=tracker,
                    device=device,
                    verbose=False,
                    retina_masks=True if self.use_segmentation else False,
                )
            else:
                # 추적 기능이 없는 경우 기본 예측 사용
                self.get_logger().warn("Track method not available, falling back to predict")
                return self.model.predict(
                    source=image,
                    conf=conf_thres,
                    iou=iou_thres,
                    max_det=max_det,
                    classes=classes,
                    device=device,
                    verbose=False,
                    retina_masks=True if self.use_segmentation else False,
                )
        except AttributeError as e:
            self.get_logger().error(f"API compatibility issue: {str(e)}")
            # 대체 추론 시도
            try:
                return self.model.predict(
                    source=image,
                    conf=conf_thres,
                    iou=iou_thres,
                    max_det=max_det,
                    classes=classes,
                    device=device,
                    verbose=False,
                )
            except Exception as e2:
                self.get_logger().error(f"Fallback inference failed: {str(e2)}")
                return None

    def create_detections_array(self, results):
        """결과를 ROS 메시지로 변환"""
        detections_msg = Detection2DArray()
        
        try:
            # YOLOv8, YOLOv11 모두 호환되는 방식으로 결과 처리
            if hasattr(results[0].boxes, 'xywh'):
                # YOLOv8 스타일
                bounding_box = results[0].boxes.xywh
                classes = results[0].boxes.cls
                confidence_score = results[0].boxes.conf
            elif hasattr(results[0], 'xywh'):  
                # 다른 형식으로 결과 반환될 경우
                bounding_box = results[0].xywh
                classes = results[0].cls
                confidence_score = results[0].conf
            else:
                # 구조를 예측할 수 없는 경우
                self.get_logger().warn("Unexpected results format - unable to extract bounding boxes")
                return detections_msg
                
            # 결과 메시지 생성
            for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
                detection = Detection2D()
                detection.bbox.center.position.x = float(bbox[0])
                detection.bbox.center.position.y = float(bbox[1])
                detection.bbox.size_x = float(bbox[2])
                detection.bbox.size_y = float(bbox[3])
                hypothesis = ObjectHypothesisWithPose()
                
                # 클래스 ID 설정
                class_id = int(cls)
                if hasattr(results[0], 'names') and class_id in results[0].names:
                    hypothesis.hypothesis.class_id = results[0].names[class_id]
                else:
                    # 이름이 없는 경우 기본값 사용
                    hypothesis.hypothesis.class_id = f"class_{class_id}"
                    
                hypothesis.hypothesis.score = float(conf)
                detection.results.append(hypothesis)
                detections_msg.detections.append(detection)
                
        except Exception as e:
            self.get_logger().error(f"Error creating detection array: {str(e)}")
            
        return detections_msg

    def create_result_image(self, results):
        """결과 시각화 이미지 생성"""
        try:
            # 시각화 파라미터 가져오기
            result_conf = self.get_parameter("result_conf").get_parameter_value().bool_value
            result_line_width = self.get_parameter("result_line_width").get_parameter_value().integer_value
            result_font_size = self.get_parameter("result_font_size").get_parameter_value().integer_value
            result_font = self.get_parameter("result_font").get_parameter_value().string_value
            result_labels = self.get_parameter("result_labels").get_parameter_value().bool_value
            result_boxes = self.get_parameter("result_boxes").get_parameter_value().bool_value
            
            # 모든 버전 호환성을 위한 시각화 처리
            if hasattr(results[0], 'plot'):
                plotted_image = results[0].plot(
                    conf=result_conf,
                    line_width=result_line_width,
                    font_size=result_font_size,
                    font=result_font,
                    labels=result_labels,
                    boxes=result_boxes,
                )
            else:
                # 대체 시각화 방법
                self.get_logger().warn("Standard plot method not available, using fallback visualization")
                from ultralytics.utils.plotting import Annotator
                img = results[0].orig_img.copy()
                annotator = Annotator(img)
                boxes = results[0].boxes.xyxy if hasattr(results[0], 'boxes') else results[0].xyxy
                cls = results[0].boxes.cls if hasattr(results[0], 'boxes') else results[0].cls
                
                names = results[0].names if hasattr(results[0], 'names') else None
                for box, c in zip(boxes, cls):
                    label = None
                    if result_labels and names and int(c) in names:
                        label = f"{names[int(c)]} {results[0].boxes.conf[i]:.2f}" if result_conf else names[int(c)]
                    annotator.box_label(box, label, color=(0, 255, 0))
                
                plotted_image = annotator.result()
                
            result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
            return result_image_msg
            
        except Exception as e:
            self.get_logger().error(f"Error creating result image: {str(e)}")
            # 오류 시 빈 이미지 반환
            empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
            return self.bridge.cv2_to_imgmsg(empty_img, encoding="bgr8")

    def create_segmentation_masks(self, results):
        """세그멘테이션 마스크 생성"""
        masks_msg = []
        
        try:
            if not self.use_segmentation:
                return masks_msg
                
            for result in results:
                if hasattr(result, "masks") and result.masks is not None:
                    # YOLOv8 세그멘테이션 마스크 처리
                    for mask_tensor in result.masks:
                        # GPU→CPU 전송 및 넘파이 변환
                        if hasattr(mask_tensor, 'data'):
                            # torch.Tensor 형식인 경우
                            mask_numpy = np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(np.uint8) * 255
                        elif isinstance(mask_tensor, np.ndarray):
                            # 이미 numpy 배열인 경우
                            mask_numpy = np.squeeze(mask_tensor).astype(np.uint8) * 255
                        else:
                            self.get_logger().warn(f"Unsupported mask type: {type(mask_tensor)}")
                            continue
                            
                        mask_image_msg = self.bridge.cv2_to_imgmsg(mask_numpy, encoding="mono8")
                        masks_msg.append(mask_image_msg)
                elif hasattr(result, "masks") and isinstance(result.masks, list):
                    # YOLOv11 또는 다른 형식의 세그멘테이션 마스크 처리
                    for mask in result.masks:
                        if isinstance(mask, np.ndarray):
                            mask_numpy = (mask * 255).astype(np.uint8)
                            mask_image_msg = self.bridge.cv2_to_imgmsg(mask_numpy, encoding="mono8")
                            masks_msg.append(mask_image_msg)
                            
        except Exception as e:
            self.get_logger().error(f"Error processing segmentation masks: {str(e)}")
            
        return masks_msg

    def _update_model_info_after_load(self):
        """로드된 모델 객체에서 추가 정보 분석"""
        try:
            # YOLOv8/YOLOv5 스타일 모델의 경우
            if hasattr(self.model, 'yaml'):
                yaml_data = self.model.yaml
                
                # 모델 타입 확인
                if 'task' in yaml_data:
                    if yaml_data['task'] == 'segment':
                        self.model_info['type'] = 'seg'
                    elif yaml_data['task'] == 'pose':
                        self.model_info['type'] = 'pose'
                    elif yaml_data['task'] == 'classify':
                        self.model_info['type'] = 'cls'
            
            # 모델 크기 추정 (파라미터 수 기반)
            if hasattr(self.model, 'params'):
                param_count = sum(p.numel() for p in self.model.parameters())
                # 대략적인 파라미터 수에 따른 모델 크기 추정
                if param_count < 3_000_000:
                    self.model_info['size'] = 'n'  # nano
                elif param_count < 12_000_000:
                    self.model_info['size'] = 's'  # small
                elif param_count < 30_000_000:
                    self.model_info['size'] = 'm'  # medium
                elif param_count < 70_000_000:
                    self.model_info['size'] = 'l'  # large
                else:
                    self.model_info['size'] = 'x'  # xlarge

            self.get_logger().info(f"Updated model info after load: {self.model_info}")
        
        except Exception as e:
            self.get_logger().error(f"Error updating model info after load: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
