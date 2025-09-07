# Copyright 2018 Lucas Walter
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Lucas Walter nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
from pathlib import Path  # noqa: E402
import sys
import numpy as np
import yaml
from tf_transformations import quaternion_from_matrix

# Hack to get relative import of .camera_config file working
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from camera_config import CameraConfig, USB_CAM_DIR  # noqa: E402

from launch import LaunchDescription  # noqa: E402
from launch.actions import GroupAction  # noqa: E402
from launch_ros.actions import Node  # noqa: E402

# TF Config 디렉토리 설정
TF_CONFIG_DIR = Path(USB_CAM_DIR, 'config', 'TFConfig')

CAMERAS = []
CAMERAS.append(
    CameraConfig(
        name='camera1',
        param_path=Path(USB_CAM_DIR, 'config', 'params_1.yaml')
    )
)
CAMERAS.append(
    CameraConfig(
        name='camera2',
        param_path=Path(USB_CAM_DIR, 'config', 'params_2.yaml')
    )
)
CAMERAS.append(
    CameraConfig(
        name='camera3',
        param_path=Path(USB_CAM_DIR, 'config', 'params_3.yaml')
    )
)
CAMERAS.append(
    CameraConfig(
        name='camera4',
        param_path=Path(USB_CAM_DIR, 'config', 'params_4.yaml')
    )
)
# Add more Camera's here and they will automatically be launched below


def load_tf_config(camera_name):
    """TF 설정 파일을 로드하여 변환 매개변수를 반환합니다."""
    tf_config_path = Path(TF_CONFIG_DIR, f"{camera_name}_tf.yaml")
    
    if not tf_config_path.exists():
        return None
    
    with open(tf_config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # 변환 매트릭스에서 쿼터니언 계산
    rot_matrix = np.array([
        [config['rotation_matrix']['r11'], config['rotation_matrix']['r12'], config['rotation_matrix']['r13'], 0],
        [config['rotation_matrix']['r21'], config['rotation_matrix']['r22'], config['rotation_matrix']['r23'], 0],
        [config['rotation_matrix']['r31'], config['rotation_matrix']['r32'], config['rotation_matrix']['r33'], 0],
        [0, 0, 0, 1]
    ])
    
    quat = quaternion_from_matrix(rot_matrix)
    
    # TF 매개변수 반환
    return {
        'parent_frame': config['parent_frame'],
        'child_frame': config['child_frame'],
        'x': config['translation']['x'],
        'y': config['translation']['y'],
        'z': config['translation']['z'],
        'qx': quat[0],
        'qy': quat[1],
        'qz': quat[2],
        'qw': quat[3]
    }


def generate_launch_description():
    ld = LaunchDescription()

    parser = argparse.ArgumentParser(description='usb_cam demo')
    parser.add_argument('-n', '--node-name', dest='node_name', type=str,
                        help='name for device', default='usb_cam')

    camera_nodes = []
    tf_nodes = []
    
    # 카메라 노드 및 TF 노드 생성
    for camera in CAMERAS:
        # 카메라 노드 추가
        camera_nodes.append(
            Node(
                package='usb_cam', executable='usb_cam_node_exe', output='screen',
                name=camera.name,
                namespace=camera.namespace,
                parameters=[camera.param_path],
                remappings=camera.remappings
            )
        )
        
        # TF 설정 로드 및 TF 노드 추가
        tf_config = load_tf_config(camera.name)
        if tf_config:
            print(f"TF 설정을 {camera.name}에 대해 로드했습니다.")
            tf_nodes.append(
                Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name=f'tf_{camera.name}_from_velodyne',
                    arguments=[
                        str(tf_config['x']), str(tf_config['y']), str(tf_config['z']),
                        str(tf_config['qx']), str(tf_config['qy']), str(tf_config['qz']), str(tf_config['qw']),
                        tf_config['parent_frame'], tf_config['child_frame']
                    ],
                    output='screen'
                )
            )
        else:
            print(f"TF 설정이 {camera.name}에 대해 발견되지 않았습니다.")

    # 카메라 노드와 TF 노드를 그룹으로 묶어서 실행
    all_nodes = camera_nodes + tf_nodes
    node_group = GroupAction(all_nodes)

    ld.add_action(node_group)
    return ld
