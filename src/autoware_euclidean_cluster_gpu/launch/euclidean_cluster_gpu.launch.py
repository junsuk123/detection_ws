from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('euclidean_cluster_gpu')
    param = os.path.join(pkg, 'config', 'euclidean_cluster_gpu.param.yaml')
    
    print(f"Package directory: {pkg}")
    print(f"Parameter file: {param}")
    print(f"Parameter file exists: {os.path.exists(param)}")
    
    return LaunchDescription([
        Node(
            package='euclidean_cluster_gpu',
            executable='euclidean_cluster_gpu_node',
            name='euclidean_cluster_gpu',
            parameters=[param],
            output='screen',
            emulate_tty=True,
            arguments=['--ros-args', '--log-level', 'INFO']
        )
    ])
