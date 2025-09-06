import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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

    # Create dynamic parameter list
    def create_parameters():
        # Base parameters from config file
        params = [LaunchConfiguration('config_file')]
        
        # Dynamic overrides based on launch arguments
        override_params = {}
        
        # Model path (always set)
        override_params['model_path'] = os.path.join(pkg_dir, 'models')
        
        return params + [override_params]

    # =============================================================================
    # Static Transform Publishers (TF Tree)
    # =============================================================================

    # velodyne -> camera3 : cam3 A의 역행렬
    tf_cam3_from_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_cam3_from_lidar',
        arguments=[
            '0', '0', '0.015',     # x y z (z만 1.5cm)
            '-0.5', '0.5', '-0.5', '0.5',    # qx qy qz qw (회전 없음)
            'velodyne', 'camera3'  # parent child
        ]

    )

    # =============================================================================
    # Main Processing Nodes
    # =============================================================================

    # YOLO Tracker Node
    yolo_tracker_node = Node(
        package='ultralytics_ros',
        executable='tracker_node.py',
        name='yolo_tracker_node',
        output='screen',
        parameters=create_parameters(),
    )

    # 3D Tracker with Cloud Node
    tracker_3d_node = Node(
        package='ultralytics_ros',
        executable='tracker_with_cloud_node',
        name='tracker_3d_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )

    # RViz2 Node (only when debug=true)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_dir, 'rviz2', 'default.rviz')],
        condition=IfCondition(LaunchConfiguration('debug')),
        output='screen'
    )

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
        
        # Static Transform Publishers
        tf_cam3_from_lidar,

        # Processing Nodes
        yolo_tracker_node,
        tracker_3d_node,
        rviz_node,
    ])