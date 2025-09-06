from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import ExecuteProcess, LogInfo
import os

def generate_launch_description():
    # 설정 파일 경로
    config_dir = os.path.join(
        get_package_share_directory('range_image_clustering'),
        'config')
    config_file = os.path.join(config_dir, 'params.yaml')
    
    # 파라미터 파일 존재 확인
    if not os.path.exists(config_file):
        return LaunchDescription([
            LogInfo(msg=f"오류: 설정 파일이 존재하지 않습니다: {config_file}")
        ])
    
    # 클러스터링 노드 실행 - 트래커에 적합한 출력 설정
    clustering_node = Node(
        package='range_image_clustering',
        executable='range_image_clustering_node',
        name='range_image_clustering_node',
        output='screen',
        parameters=[config_file, {
            # 트래커용으로 출력 토픽 이름 조정
            'output_cloud_topic': '/pointcloud/clustered',
            'input_topic': '/pointcloud/ground_removed',
            
            # 디버그 모드 활성화 - 클러스터링 문제 진단
            'debug_mode': True,
            'verbose_logging': True,
            
            # 최적화된 설정 직접 적용
            'range_image_width': 720,       # 해상도 조정
            'range_image_height': 64, 
            'angle_threshold': 0.15,
            'min_cluster_size': 3
        }],
        emulate_tty=True,
        prefix=['nice -n 10']
    )
    
    # ROI 시각화 노드 추가
    roi_publisher_node = Node(
        package='range_image_clustering',
        executable='roi_publisher_node',
        name='roi_publisher_node',
        parameters=[config_file],
        output='screen',
        emulate_tty=True,
        prefix=['nice -n 15']
    )
    
    return LaunchDescription([
        clustering_node,
        roi_publisher_node
    ])
