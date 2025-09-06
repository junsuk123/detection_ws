from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # 토픽 정보 확인 명령
    check_topics_cmd = ExecuteProcess(
        cmd=['bash', '-c', 
             'echo "===== 포인트 클라우드 토픽 확인 =====" && '
             'echo "1. /velodyne_points:" && ros2 topic info /velodyne_points && echo && '
             'echo "2. /pointcloud/ground_removed:" && ros2 topic info /pointcloud/ground_removed && echo && '
             'echo "3. /cuda_pointcloud_preprocessor/output/pointcloud/cuda:" && '
             'ros2 topic info /cuda_pointcloud_preprocessor/output/pointcloud/cuda && echo && '
             'echo "4. 프레임 ID 확인:" && '
             'ros2 topic echo /pointcloud/ground_removed --once | grep frame_id'
            ],
        output='screen'
    )
    
    return LaunchDescription([
        check_topics_cmd
    ])
