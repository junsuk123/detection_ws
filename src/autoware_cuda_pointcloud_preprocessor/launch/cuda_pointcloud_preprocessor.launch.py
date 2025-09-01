import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('autoware_cuda_pointcloud_preprocessor')
    
    # Config 파일에서 기본 토픽 읽기
    config_file = os.path.join(pkg_dir, 'config', 'cuda_pointcloud_preprocessor.param.yaml')
    
    # Config 파일 파싱하여 기본값 추출
    default_topics = {
        'input_pointcloud': '/velodyne_points',
        'input_imu': '/iahrs/imu', 
        'input_twist': '/sensing/vehicle_velocity_converter/twist_with_covariance',
        'output_pointcloud': '/pointcloud/noiseremoved'
    }
    
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            params = config_data['/**']['ros__parameters']
            default_topics['input_pointcloud'] = params.get('input_pointcloud_topic', default_topics['input_pointcloud'])
            default_topics['input_imu'] = params.get('input_imu_topic', default_topics['input_imu'])
            default_topics['input_twist'] = params.get('input_twist_topic', default_topics['input_twist'])
            default_topics['output_pointcloud'] = params.get('output_pointcloud_topic', default_topics['output_pointcloud'])
    except:
        pass  # Use defaults if config parsing fails

    # =============================================================================
    # Static Transform Publisher
    # =============================================================================
    
    # base_link -> velodyne : 항등(원점·무회전)
    tf_base_to_velodyne = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_velodyne',
        arguments=['0', '0', '0', '0', '0', '0', '1', 'base_link', 'velodyne'],
    )

    # =============================================================================
    # Main Processing Node
    # =============================================================================
    
    cuda_preprocessor_node = Node(
        package='autoware_cuda_pointcloud_preprocessor',
        executable='cuda_pointcloud_preprocessor_node',
        name='cuda_pointcloud_preprocessor',
        output='screen',
        parameters=[LaunchConfiguration('cuda_pointcloud_preprocessor_param_file')],
        remappings=[
            ('~/input/pointcloud', LaunchConfiguration('input_pointcloud')),
            ('~/input/imu', LaunchConfiguration('input_imu')),
            ('~/input/twist', LaunchConfiguration('input_twist')),
            ('~/output/pointcloud', LaunchConfiguration('output_pointcloud')),
        ]
    )

    return LaunchDescription([
        # Arguments with defaults from config file
        DeclareLaunchArgument('cuda_pointcloud_preprocessor_param_file', 
                            default_value=config_file),
        DeclareLaunchArgument('input_pointcloud', 
                            default_value=default_topics['input_pointcloud']),
        DeclareLaunchArgument('input_imu', 
                            default_value=default_topics['input_imu']),
        DeclareLaunchArgument('input_twist', 
                            default_value=default_topics['input_twist']),
        DeclareLaunchArgument('output_pointcloud', 
                            default_value=default_topics['output_pointcloud']),

        # Static Transform Publisher
        tf_base_to_velodyne,
        
        # Main Node
        cuda_preprocessor_node,
    ])