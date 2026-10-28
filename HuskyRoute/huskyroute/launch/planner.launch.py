"""Nav2 global planner on a saved map, no robot: map_server + planner_server (with its global costmap) + lifecycle
manager, and identity map -> odom -> base_link transforms so the costmap can start.
    ros2 launch planner.launch.py params:=<nav2 params yaml> map:=<grid yaml>
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params, grid = LaunchConfiguration("params"), LaunchConfiguration("map")
    tf = lambda a, b: Node(package="tf2_ros", executable="static_transform_publisher", name=f"tf_{a}_{b}",
                           arguments=["0", "0", "0", "0", "0", "0", a, b])
    return LaunchDescription([
        DeclareLaunchArgument("params"), DeclareLaunchArgument("map"),
        tf("map", "odom"), tf("odom", "base_link"),
        Node(package="nav2_map_server", executable="map_server", name="map_server",
             parameters=[{"yaml_filename": grid, "use_sim_time": False}]),
        Node(package="nav2_planner", executable="planner_server", name="planner_server", parameters=[params]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_route",
             parameters=[{"autostart": True, "node_names": ["map_server", "planner_server"], "use_sim_time": False}]),
    ])
