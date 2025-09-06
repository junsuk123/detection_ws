from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # RViz 설정 파일 경로
    rviz_config_dir = os.path.join(
        get_package_share_directory('range_image_clustering'),
        'rviz',
        'clustering_visualization.rviz')
    
    # 테스트 포인트 클라우드 발행 노드
    test_publisher_node = Node(
        package='range_image_clustering',
        executable='test_point_cloud.py',
        name='test_point_cloud_publisher',
        output='screen'
    )
    
    # 클러스터링 노드 - 테스트 토픽 입력
    clustering_node = Node(
        package='range_image_clustering',
        executable='range_image_clustering_node',
        name='range_image_clustering_node',
        output='screen',
        parameters=[{
            'input_topic': '/test_points_raw',
            'output_cloud_topic': '/clustered_cloud',
            'output_markers_topic': '/cluster_markers',
            'angle_threshold': 0.3,  # 테스트용으로 더 관대한 임계값
            'distance_threshold': 1.0,
            'min_cluster_size': 5,
            'max_cluster_size': 10000,
        }],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    # RViz 노드
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        output='screen'
    )
    
    return LaunchDescription([
        test_publisher_node,
        clustering_node,
        rviz_node
    ])
