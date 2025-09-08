from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1) 런치 아규먼트 선언
    declare_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Subscribe Image topic name'
    )
    declare_hz_arg = DeclareLaunchArgument(
        'save_hz',
        default_value='0.5',
        description='Image saving frequency (Hz)'
    )

    # 2) node 실행 시 파라미터로 전달
    return LaunchDescription([
        declare_topic_arg,
        declare_hz_arg,
        Node(
            package='data_collector',
            executable='image_collector_node',
            name='image_collector',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'save_hz':    LaunchConfiguration('save_hz'),
            }],
        ),
    ])
