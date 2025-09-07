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
            
            # 소형 객체 분리를 위한 최적화된 설정
            'range_image_width': 720,       # 해상도 증가
            'range_image_height': 64, 
            'angle_threshold': 0.07,        # 더 엄격한 각도 임계값
            'distance_threshold': 0.03,     # 더 엄격한 거리 임계값 (3cm)
            'min_cluster_size': 2,
            
            # 깊이 불연속 기반 클러스터링 강화
            'depth_discontinuity_threshold': 0.08,
            'adaptive_min_cluster_size': True,
            'enable_acos_optimization': True,
            
            # 병합 비활성화로 소형 객체 보존
            'merge_clusters': False,
            'use_improved_merge_criteria': True,
            'merge_box_expansion_factor': 1.03,
            
            # 소형 객체 추가 처리 활성화
            'enable_small_object_refinement': True,
            'small_object_max_volume': 0.01,
            'min_object_separation': 0.05
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
