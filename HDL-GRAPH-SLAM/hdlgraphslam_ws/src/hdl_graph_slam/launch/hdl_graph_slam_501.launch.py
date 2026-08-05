from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    raw_points_topic = LaunchConfiguration("raw_points_topic")
    points_topic = LaunchConfiguration("points_topic")
    raw_points_qos = LaunchConfiguration("raw_points_qos")
    filtered_points_qos = LaunchConfiguration("filtered_points_qos")
    enable_floor_detection = LaunchConfiguration("enable_floor_detection")
    base_link_frame_id = LaunchConfiguration("base_link_frame_id")
    lidar_frame_id = LaunchConfiguration("lidar_frame_id")
    distance_far_thresh = LaunchConfiguration("distance_far_thresh")
    prefilter_downsample_method = LaunchConfiguration("prefilter_downsample_method")
    prefilter_downsample_resolution = LaunchConfiguration("prefilter_downsample_resolution")
    prefilter_outlier_removal_method = LaunchConfiguration("prefilter_outlier_removal_method")
    scan_downsample_method = LaunchConfiguration("scan_downsample_method")
    scan_downsample_resolution = LaunchConfiguration("scan_downsample_resolution")
    scan_reg_num_threads = LaunchConfiguration("scan_reg_num_threads")
    scan_reg_maximum_iterations = LaunchConfiguration("scan_reg_maximum_iterations")
    scan_reg_max_optimizer_iterations = LaunchConfiguration("scan_reg_max_optimizer_iterations")
    scan_reg_correspondence_randomness = LaunchConfiguration("scan_reg_correspondence_randomness")
    graph_keyframe_delta_trans = LaunchConfiguration("graph_keyframe_delta_trans")
    graph_keyframe_delta_angle = LaunchConfiguration("graph_keyframe_delta_angle")

    common_params = {
        "use_sim_time": use_sim_time,
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("raw_points_topic", default_value="/velodyne_points"),
            DeclareLaunchArgument("points_topic", default_value="/filtered_points"),
            DeclareLaunchArgument("raw_points_qos", default_value="best_effort"),
            DeclareLaunchArgument("filtered_points_qos", default_value="reliable"),
            DeclareLaunchArgument("enable_floor_detection", default_value="false"),
            DeclareLaunchArgument("base_link_frame_id", default_value="base_link"),
            DeclareLaunchArgument("lidar_frame_id", default_value="velodyne"),
            DeclareLaunchArgument("distance_far_thresh", default_value="20.0"),
            DeclareLaunchArgument("prefilter_downsample_method", default_value="APPROX_VOXELGRID"),
            DeclareLaunchArgument("prefilter_downsample_resolution", default_value="0.1"),
            DeclareLaunchArgument("prefilter_outlier_removal_method", default_value="NONE"),
            DeclareLaunchArgument("scan_downsample_method", default_value="NONE"),
            DeclareLaunchArgument("scan_downsample_resolution", default_value="0.1"),
            DeclareLaunchArgument("scan_reg_num_threads", default_value="0"),
            DeclareLaunchArgument("scan_reg_maximum_iterations", default_value="64"),
            DeclareLaunchArgument("scan_reg_max_optimizer_iterations", default_value="20"),
            DeclareLaunchArgument("scan_reg_correspondence_randomness", default_value="20"),
            # graph_slam_node's own keyframe gate. Its code defaults are 2.0 m /
            # 2.0 rad (115 deg), which on a 174 s indoor run yields ~30 graph
            # nodes — too coarse to serve as a reference trajectory and too few
            # to give the loop detector much to work with.
            DeclareLaunchArgument("graph_keyframe_delta_trans", default_value="2.0"),
            DeclareLaunchArgument("graph_keyframe_delta_angle", default_value="2.0"),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="lidar2base_publisher",
                output="screen",
                arguments=[
                    "--frame-id",
                    base_link_frame_id,
                    "--child-frame-id",
                    lidar_frame_id,
                ],
            ),
            Node(
                package="hdl_graph_slam",
                executable="prefiltering_node",
                name="prefiltering_node",
                output="screen",
                parameters=[
                    common_params,
                    {
                        "input_points_topic": raw_points_topic,
                        "output_points_topic": points_topic,
                        "input_points_qos": raw_points_qos,
                        "output_points_qos": filtered_points_qos,
                        "base_link_frame": base_link_frame_id,
                        "use_distance_filter": True,
                        "distance_near_thresh": 0.5,
                        "distance_far_thresh": ParameterValue(distance_far_thresh, value_type=float),
                        "downsample_method": prefilter_downsample_method,
                        "downsample_resolution": ParameterValue(prefilter_downsample_resolution, value_type=float),
                        "outlier_removal_method": prefilter_outlier_removal_method,
                        "statistical_mean_k": 20,
                        "statistical_stddev": 1.0,
                        "radius_radius": 0.5,
                        "radius_min_neighbors": 2,
                    },
                ],
            ),
            Node(
                package="hdl_graph_slam",
                executable="scan_matching_odometry_node",
                name="scan_matching_odometry_node",
                output="screen",
                parameters=[
                    common_params,
                    {
                        "points_topic": points_topic,
                        "points_qos": filtered_points_qos,
                        "published_odom_topic": "/odom",
                        "odom_frame_id": "odom",
                        "keyframe_delta_trans": 0.25,
                        "keyframe_delta_angle": 0.3,
                        "keyframe_delta_time": 10000.0,
                        "transform_thresholding": False,
                        "downsample_method": scan_downsample_method,
                        "downsample_resolution": ParameterValue(scan_downsample_resolution, value_type=float),
                        "registration_method": "FAST_GICP",
                        "reg_num_threads": ParameterValue(scan_reg_num_threads, value_type=int),
                        "reg_transformation_epsilon": 0.1,
                        "reg_maximum_iterations": ParameterValue(scan_reg_maximum_iterations, value_type=int),
                        "reg_max_correspondence_distance": 1.0,
                        "reg_max_optimizer_iterations": ParameterValue(scan_reg_max_optimizer_iterations, value_type=int),
                        "reg_use_reciprocal_correspondences": False,
                        "reg_correspondence_randomness": ParameterValue(scan_reg_correspondence_randomness, value_type=int),
                        "reg_resolution": 1.0,
                        "reg_nn_search_method": "DIRECT7",
                    },
                ],
            ),
            Node(
                package="hdl_graph_slam",
                executable="graph_slam_node",
                name="graph_slam_node",
                output="screen",
                parameters=[
                    common_params,
                    {
                        "points_topic": points_topic,
                        "points_qos": filtered_points_qos,
                        "published_odom_topic": "/odom",
                        "keyframe_delta_trans": ParameterValue(graph_keyframe_delta_trans,
                                                               value_type=float),
                        "keyframe_delta_angle": ParameterValue(graph_keyframe_delta_angle,
                                                               value_type=float),
                        "use_const_inf_matrix": True,
                        "publish_map_odom_tf": True,
                        "publish_map_points": True,
                        "map_frame_id": "map",
                        "map_cloud_resolution": 0.1,
                        "fix_first_node": True,
                        "fix_first_node_stddev": "10 10 10 1 1 1",
                        "fix_first_node_adaptive": True,
                        "enable_floor_constraints": enable_floor_detection,
                        "floor_association_tolerance_sec": 0.05,
                        "g2o_optimization_interval_sec": 2.0,
                        "loop_accum_distance_thresh": 3.0,
                        "loop_min_edge_interval": 2.0,
                        "loop_distance_thresh": 5.0,
                        "loop_fitness_score_thresh": 0.6,
                    },
                ],
            ),
            Node(
                package="hdl_graph_slam",
                executable="floor_detection_node",
                name="floor_detection_node",
                output="screen",
                condition=IfCondition(enable_floor_detection),
                parameters=[
                    common_params,
                    {
                        "points_topic": points_topic,
                        "points_qos": filtered_points_qos,
                        "tilt_deg": 0.0,
                        "sensor_height": 0.5,
                        "height_clip_range": 0.5,
                        "floor_pts_thresh": 256,
                        "use_normal_filtering": True,
                        "normal_filter_thresh": 20.0,
                    },
                ],
            ),
        ]
    )
