import os
import subprocess
import re
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import launch


def get_active_cameras(context):
    """현재 발행 중인 카메라 토픽을 감지하여 활성 카메라 목록을 반환합니다"""
    
    # ros2 topic list 명령어 실행
    try:
        result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True, check=True)
        topic_list = result.stdout.strip().split('\n')
    except subprocess.SubprocessError:
        print("경고: ROS2 토픽 목록을 가져오는 데 실패했습니다. 기본 카메라만 사용합니다.")
        return ['camera3']  # 기본 카메라 하나만 가정
    
    # 카메라 토픽 패턴 매칭
    camera_pattern = r'/(\w+)/image_raw'
    
    # 모든 카메라 이름 찾기
    cameras = set()
    for topic in topic_list:
        match = re.match(camera_pattern, topic)
        if match:
            camera_name = match.group(1)
            # 이름과 camera_info 토픽이 모두 존재하는지 확인
            if f'/{camera_name}/camera_info' in topic_list:
                cameras.add(camera_name)
    
    if not cameras:
        print("경고: 활성 카메라가 감지되지 않았습니다. 기본 카메라(camera3)를 사용합니다.")
        return ['camera3']  # 기본 카메라 사용
    
    print(f"감지된 활성 카메라: {', '.join(cameras)}")
    return list(cameras)


def generate_launch_description():
    
    # Package directory
    pkg_dir = get_package_share_directory('ultralytics_ros')
    
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')
    debug_arg = DeclareLaunchArgument('debug', default_value='false')
    config_file_arg = DeclareLaunchArgument(
        'config_file', 
        default_value=os.path.join(pkg_dir, 'config', 'tracker_with_cloud.param.yaml')
    )
    
    # Topic override arguments
    input_image_topic_arg = DeclareLaunchArgument('input_image_topic', default_value='')
    camera_info_topic_arg = DeclareLaunchArgument('camera_info_topic', default_value='')
    lidar_topic_arg = DeclareLaunchArgument('lidar_topic', default_value='')
    yolo_result_topic_arg = DeclareLaunchArgument('yolo_result_topic', default_value='')
    yolo_result_image_topic_arg = DeclareLaunchArgument('yolo_result_image_topic', default_value='')
    yolo_3d_result_topic_arg = DeclareLaunchArgument('yolo_3d_result_topic', default_value='')
    
    # Model arguments
    yolo_model_arg = DeclareLaunchArgument('yolo_model', default_value='')
    conf_thres_arg = DeclareLaunchArgument('conf_thres', default_value='')
    iou_thres_arg = DeclareLaunchArgument('iou_thres', default_value='')
    device_arg = DeclareLaunchArgument('device', default_value='')

    # Base parameters from config file
    config_file = LaunchConfiguration('config_file')

    # RViz2 Node (only when debug=true)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_dir, 'rviz2', 'default.rviz')],
        condition=IfCondition(LaunchConfiguration('debug')),
        output='screen'
    )

    # 동적으로 노드를 생성하는 함수
    def launch_setup(context):
        # 활성 카메라 목록 가져오기
        active_cameras = get_active_cameras(context)
        
        # 기본 런치 엔티티 (정적 TF 노드는 제외 - USB_CAM에서 처리)
        launch_entities = [rviz_node]  # TF 노드 제거됨
        
        # 각 카메라마다 YOLO 트래커 노드 생성
        yolo_nodes = []
        for idx, camera_name in enumerate(active_cameras):
            # 고유한 YOLO 노드 이름 생성
            node_name = f'yolo_tracker_{camera_name}'
            topic_prefix = f'/{camera_name}'
            
            # YOLO 트래커 노드 파라미터 설정
            yolo_params = {
                'model_path': os.path.join(pkg_dir, 'models'),
                'input_image_topic': 'image_raw',
                'camera_info_topic': 'camera_info',
                'yolo_result_topic': 'yolo_result',
                'yolo_result_image_topic': 'yolo_image',  # 상대 경로
                'camera_name': camera_name
            }
            yolo_node = Node(
                package='ultralytics_ros',
                executable='tracker_node.py',
                namespace=camera_name,
                name='yolo_tracker',
                output='screen',
                parameters=[config_file, yolo_params],
                remappings=[
                    # 입력을 반드시 카메라 고유 토픽으로 고정
                    ('/image_raw',  f'/{camera_name}/image_raw'),
                    ('image_raw',   'image_raw'),
                    ('/camera_info',f'/{camera_name}/camera_info'),
                    ('camera_info', 'camera_info'),
                    # 출력 이미지도 네임스페이스로 고정
                    ('/yolo_image', 'yolo_image'),
                ],
            )


            yolo_nodes.append(yolo_node)
            launch_entities.append(yolo_node)
        
        # 다중 카메라 지원을 위한 트래커 파라미터 설정
        tracker_params = {
            'active_cameras': active_cameras,
            'camera_count': len(active_cameras),
            'camera_info_topics': [f'/{c}/camera_info' for c in active_cameras],
            'yolo_result_topics': [f'/{c}/yolo_result' for c in active_cameras],
            # 여기를 반드시 지정 (예시: euclidean_cluster_gpu 노드의 출력)
            'lidar_topic': '/pointcloud/clustered',
            # 선택: 매칭 동작 튜닝 파라미터
            'force_bbox': False,
            'min_points_in_bbox': 4,
            'min_ratio_in_cluster': 0.5,
            'max_clusters_per_det': 2,
            'exclusive_cluster': True,
        }
        
        # 3D 트래커 노드 생성
        tracker_3d_node = Node(
            package='ultralytics_ros',
            executable='tracker_with_cloud_node',
            name='tracker_3d_node',
            output='screen',
            parameters=[tracker_params],   # ← YAML 빼기!
            arguments=['--ros-args', '--log-level', 'info'],
        )
        launch_entities.append(tracker_3d_node)
        
        return launch_entities

    # OpaqueFunction을 사용하여 런치 타임에 활성 카메라 감지 및 노드 구성
    return LaunchDescription([
        # Arguments
        use_sim_time_arg,
        debug_arg,
        config_file_arg,
        input_image_topic_arg,
        camera_info_topic_arg,
        lidar_topic_arg,
        yolo_result_topic_arg,
        yolo_result_image_topic_arg,
        yolo_3d_result_topic_arg,
        yolo_model_arg,
        conf_thres_arg,
        iou_thres_arg,
        device_arg,
        
        # 동적으로 노드를 생성하는 함수 실행
        OpaqueFunction(function=launch_setup)
    ])