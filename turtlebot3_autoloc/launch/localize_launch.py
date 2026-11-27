"""Start TurtleBot3 Nav2 (map_server + AMCL + navigation + RViz) and auto-localize.

Usage:
  ros2 launch tb3_autoloc localize_launch.py map:=$HOME/maps/lab.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    map_file = LaunchConfiguration('map')

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('turtlebot3_navigation2'),
            'launch', 'navigation2.launch.py')),
        launch_arguments={'map': map_file, 'use_sim_time': 'False'}.items(),
    )

    auto_localize = Node(
        package='tb3_autoloc',
        executable='auto_localize',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(os.path.expanduser('~'), 'maps', 'map.yaml'),
            description='Full path to the map .yaml file'),
        nav2,
        auto_localize,
    ])
