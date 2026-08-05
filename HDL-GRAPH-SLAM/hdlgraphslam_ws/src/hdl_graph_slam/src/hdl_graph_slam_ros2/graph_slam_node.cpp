// SPDX-License-Identifier: BSD-2-Clause

#include <chrono>
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d_addons/vertex_plane.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <std_msgs/msg/int32.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <hdl_graph_slam/graph_slam.hpp>
#include <hdl_graph_slam/information_matrix_calculator.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/keyframe_updater.hpp>
#include <hdl_graph_slam/loop_detector.hpp>
#include <hdl_graph_slam/map_cloud_generator.hpp>
#include <hdl_graph_slam/msg/floor_coeffs.hpp>
#include <hdl_graph_slam/qos.hpp>
#include <hdl_graph_slam/srv/dump_graph.hpp>
#include <hdl_graph_slam/srv/load_graph.hpp>
#include <hdl_graph_slam/srv/save_map.hpp>

namespace hdl_graph_slam {
namespace {

Eigen::Isometry3d odom2isometry(const nav_msgs::msg::Odometry& odom) {
  const auto& p = odom.pose.pose.position;
  const auto& q = odom.pose.pose.orientation;

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(p.x, p.y, p.z);
  pose.linear() = Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized().toRotationMatrix();
  return pose;
}

}  // namespace

class GraphSlamNode : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;
  using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::PointCloud2>;

  explicit GraphSlamNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : rclcpp::Node("graph_slam_node", options),
    keyframe_updater_(
      declare_parameter<double>("keyframe_delta_trans", 2.0),
      declare_parameter<double>("keyframe_delta_angle", 2.0)),
    graph_slam_(declare_parameter<std::string>("g2o_solver_type", "lm_var")),
    information_matrix_calculator_(*this),
    loop_detector_(*this) {
    published_odom_topic_ = declare_parameter<std::string>("published_odom_topic", "/odom");
    points_topic_ = declare_parameter<std::string>("points_topic", "/filtered_points");
    points_qos_ = declare_parameter<std::string>("points_qos", "reliable");
    point_cloud_qos_depth_ = declare_parameter<int>("point_cloud_qos_depth", 32);
    map_frame_id_ = declare_parameter<std::string>("map_frame_id", "map");
    odom_frame_id_ = declare_parameter<std::string>("odom_frame_id", "odom");
    sync_queue_size_ = declare_parameter<int>("sync_queue_size", 32);

    g2o_optimization_interval_sec_ = declare_parameter<double>("g2o_optimization_interval_sec", 5.0);
    g2o_optimization_max_iterations_ = declare_parameter<int>("g2o_optimization_max_iterations", 1024);
    g2o_optimization_min_edges_ = declare_parameter<int>("g2o_optimization_min_edges", 2);
    map_cloud_resolution_ = declare_parameter<double>("map_cloud_resolution", 0.1);
    enable_debug_topics_ = declare_parameter<bool>("enable_debug_topics", true);
    publish_map_odom_tf_ = declare_parameter<bool>("publish_map_odom_tf", true);
    publish_map_points_ = declare_parameter<bool>("publish_map_points", true);
    fix_first_node_ = declare_parameter<bool>("fix_first_node", false);
    fix_first_node_stddev_ = declare_parameter<std::string>("fix_first_node_stddev", "1 1 1 1 1 1");
    fix_first_node_adaptive_ = declare_parameter<bool>("fix_first_node_adaptive", true);
    enable_floor_constraints_ = declare_parameter<bool>("enable_floor_constraints", false);
    floor_coeffs_topic_ = declare_parameter<std::string>("floor_coeffs_topic", "/floor_detection/floor_coeffs");
    floor_edge_stddev_ = declare_parameter<double>("floor_edge_stddev", 10.0);
    floor_association_tolerance_sec_ = declare_parameter<double>("floor_association_tolerance_sec", 0.05);
    enable_imu_constraints_ = declare_parameter<bool>("enable_imu_constraints", false);
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/imu/data");
    imu_orientation_stddev_ = declare_parameter<double>("imu_orientation_stddev", 1.0);
    enable_gps_constraints_ = declare_parameter<bool>("enable_gps_constraints", false);
    gps_topic_ = declare_parameter<std::string>("gps_topic", "/gps/fix");
    gps_edge_stddev_ = declare_parameter<double>("gps_edge_stddev", 1.0);

    odom_sub_.subscribe(
      this,
      published_odom_topic_,
      rclcpp::QoS(rclcpp::KeepLast(sync_queue_size_)).get_rmw_qos_profile());
    cloud_sub_.subscribe(
      this,
      points_topic_,
      point_cloud_qos(points_qos_, point_cloud_qos_depth_).get_rmw_qos_profile());

    RCLCPP_INFO(
      get_logger(),
      "graph_slam_node: subscribing to %s with %s QoS",
      points_topic_.c_str(),
      points_qos_.c_str());

    sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(
      ApproxSyncPolicy(sync_queue_size_), odom_sub_, cloud_sub_);
    sync_->registerCallback(
      std::bind(&GraphSlamNode::cloud_callback, this, std::placeholders::_1, std::placeholders::_2));

    map_odom_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    if(publish_map_points_) {
      map_points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/hdl_graph_slam/map_points",
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
    }
    if(enable_floor_constraints_) {
      floor_coeffs_sub_ = create_subscription<hdl_graph_slam::msg::FloorCoeffs>(
        floor_coeffs_topic_,
        rclcpp::QoS(256),
        std::bind(&GraphSlamNode::floor_coeffs_callback, this, std::placeholders::_1));
    }
    if(enable_imu_constraints_) {
      imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_,
        rclcpp::QoS(256),
        std::bind(&GraphSlamNode::imu_callback, this, std::placeholders::_1));
    }
    if(enable_gps_constraints_) {
      gps_sub_ = create_subscription<sensor_msgs::msg::NavSatFix>(
        gps_topic_,
        rclcpp::QoS(256),
        std::bind(&GraphSlamNode::gps_callback, this, std::placeholders::_1));
    }

    if(enable_debug_topics_) {
      auto count_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
      keyframe_count_pub_ = create_publisher<std_msgs::msg::Int32>("/hdl_graph_slam/debug/keyframe_count", count_qos);
      graph_vertices_pub_ = create_publisher<std_msgs::msg::Int32>("/hdl_graph_slam/debug/graph_vertices", count_qos);
      graph_edges_pub_ = create_publisher<std_msgs::msg::Int32>("/hdl_graph_slam/debug/graph_edges", count_qos);
      odom2map_debug_pub_ = create_publisher<geometry_msgs::msg::TransformStamped>(
        "/hdl_graph_slam/debug/odom2map", count_qos);
      loop_count_debug_pub_ = create_publisher<std_msgs::msg::Int32>(
        "/hdl_graph_slam/debug/loop_count", count_qos);
      floor_count_debug_pub_ = create_publisher<std_msgs::msg::Int32>(
        "/hdl_graph_slam/debug/floor_constraint_count", count_qos);
    }

    save_map_service_ = create_service<hdl_graph_slam::srv::SaveMap>(
      "/hdl_graph_slam/save_map",
      std::bind(&GraphSlamNode::save_map_service, this, std::placeholders::_1, std::placeholders::_2));
    dump_graph_service_ = create_service<hdl_graph_slam::srv::DumpGraph>(
      "/hdl_graph_slam/dump_graph",
      std::bind(&GraphSlamNode::dump_graph_service, this, std::placeholders::_1, std::placeholders::_2));
    load_graph_service_ = create_service<hdl_graph_slam::srv::LoadGraph>(
      "/hdl_graph_slam/load_graph",
      std::bind(&GraphSlamNode::load_graph_service, this, std::placeholders::_1, std::placeholders::_2));

    const auto interval = std::chrono::duration<double>(g2o_optimization_interval_sec_);
    optimize_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(interval),
      std::bind(&GraphSlamNode::optimize_timer_callback, this));

    RCLCPP_INFO(
      get_logger(),
      "graph_slam_node stage8 waiting for synchronized %s and %s, optimize every %.2fs (min_edges=%d), loop_distance_thresh=%.2f, fix_first_node=%s adaptive=%s, floor_constraints=%s tolerance=%.3fs, publish_map_odom_tf=%s, publish_map_points=%s, debug=%s",
      published_odom_topic_.c_str(),
      points_topic_.c_str(),
      g2o_optimization_interval_sec_,
      g2o_optimization_min_edges_,
      loop_detector_.get_distance_thresh(),
      fix_first_node_ ? "enabled" : "disabled",
      fix_first_node_adaptive_ ? "enabled" : "disabled",
      enable_floor_constraints_ ? "enabled" : "disabled",
      floor_association_tolerance_sec_,
      publish_map_odom_tf_ ? "enabled" : "disabled",
      publish_map_points_ ? "enabled" : "disabled",
      enable_debug_topics_ ? "enabled" : "disabled");
  }

private:
  void floor_coeffs_callback(const hdl_graph_slam::msg::FloorCoeffs::SharedPtr msg) {
    floor_coeffs_queue_.push_back(msg);
    while(floor_coeffs_queue_.size() > 512) {
      floor_coeffs_queue_.pop_front();
    }
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    imu_queue_.push_back(msg);
    while(imu_queue_.size() > 512) {
      imu_queue_.pop_front();
    }
  }

  void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
    gps_queue_.push_back(msg);
    while(gps_queue_.size() > 512) {
      gps_queue_.pop_front();
    }
  }

  void cloud_callback(
    const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg) {
    const Eigen::Isometry3d odom = odom2isometry(*odom_msg);
    if(!keyframe_updater_.update(odom)) {
      return;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(cloud->empty()) {
      return;
    }
    pcl_conversions::toPCL(cloud_msg->header, cloud->header);

    auto keyframe = std::make_shared<KeyFrame>(
      rclcpp::Time(cloud_msg->header.stamp),
      odom,
      keyframe_updater_.get_accum_distance(),
      cloud);

    keyframe->node = graph_slam_.add_se3_node(odom);
    maybe_add_first_node_anchor(keyframe);

    const KeyFrame::Ptr prev_keyframe = latest_keyframe();
    if(prev_keyframe) {
      const Eigen::Isometry3d relative_pose = keyframe->odom.inverse() * prev_keyframe->odom;
      const Eigen::MatrixXd information = information_matrix_calculator_.calc_information_matrix(
        keyframe->cloud, prev_keyframe->cloud, relative_pose);
      graph_slam_.add_se3_edge(keyframe->node, prev_keyframe->node, relative_pose, information);
    }
    add_optional_constraints(keyframe);

    new_keyframes_.push_back(keyframe);

    if(enable_debug_topics_) {
      std_msgs::msg::Int32 count_msg;
      count_msg.data = total_keyframe_count();
      keyframe_count_pub_->publish(count_msg);
      publish_graph_counts();
    }

    RCLCPP_INFO(
      get_logger(),
      "registered keyframe %d stamp %.9f accum_distance %.3f cloud_size %zu vertices %d edges %d",
      total_keyframe_count(),
      rclcpp::Time(cloud_msg->header.stamp).seconds(),
      keyframe->accum_distance,
      keyframe->cloud->size(),
      graph_slam_.num_vertices(),
      graph_slam_.num_edges());
  }

  void add_optional_constraints(const KeyFrame::Ptr& keyframe) {
    add_floor_constraint(keyframe);
    add_imu_constraint(keyframe);
    add_gps_constraint(keyframe);
  }

  void maybe_add_first_node_anchor(const KeyFrame::Ptr& keyframe) {
    if(!fix_first_node_ || anchor_node_ || total_keyframe_count() != 0) {
      return;
    }

    Eigen::MatrixXd information = Eigen::MatrixXd::Identity(6, 6);
    std::stringstream sst(fix_first_node_stddev_);
    for(int i = 0; i < 6; ++i) {
      double stddev = 1.0;
      if(!(sst >> stddev) || stddev <= 0.0) {
        RCLCPP_WARN(
          get_logger(),
          "invalid fix_first_node_stddev component at index %d, using 1.0",
          i);
        stddev = 1.0;
      }
      information(i, i) = 1.0 / stddev;
    }

    anchor_node_ = graph_slam_.add_se3_node(Eigen::Isometry3d::Identity());
    anchor_node_->setFixed(true);
    anchor_edge_ = graph_slam_.add_se3_edge(
      anchor_node_, keyframe->node, Eigen::Isometry3d::Identity(), information);

    RCLCPP_INFO(
      get_logger(),
      "fix_first_node anchor added: anchor_node=%d anchor_edge=%d target_keyframe_stamp=%.9f",
      anchor_node_->id(), anchor_edge_->id(), keyframe->stamp.seconds());
  }

  bool add_floor_constraint(const KeyFrame::Ptr& keyframe) {
    if(!enable_floor_constraints_) {
      return false;
    }
    if(keyframe->floor_coeffs) {
      return false;
    }

    const auto found = find_nearest_floor_coeff(keyframe);
    if(found == floor_coeffs_queue_.end()) {
      return false;
    }

    if(!floor_plane_node_) {
      floor_plane_node_ = graph_slam_.add_plane_node(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0));
      floor_plane_node_->setFixed(true);
    }

    const auto& coeffs_msg = *found;
    Eigen::Vector4d coeffs(
      coeffs_msg->coeffs[0],
      coeffs_msg->coeffs[1],
      coeffs_msg->coeffs[2],
      coeffs_msg->coeffs[3]);
    Eigen::Matrix3d information = Eigen::Matrix3d::Identity() * (1.0 / floor_edge_stddev_);
    graph_slam_.add_se3_plane_edge(keyframe->node, floor_plane_node_, coeffs, information);
    keyframe->floor_coeffs = coeffs;
    floor_constraint_count_++;

    const double dt = (rclcpp::Time(coeffs_msg->header.stamp) - keyframe->stamp).seconds();
    RCLCPP_INFO(
      get_logger(),
      "floor constraint added: total=%d keyframe_stamp=%.9f coeff_stamp=%.9f dt=%.6f vertices=%d edges=%d",
      floor_constraint_count_,
      keyframe->stamp.seconds(),
      rclcpp::Time(coeffs_msg->header.stamp).seconds(),
      dt,
      graph_slam_.num_vertices(),
      graph_slam_.num_edges());

    floor_coeffs_queue_.erase(found);
    publish_floor_constraint_count();
    return true;
  }

  std::deque<hdl_graph_slam::msg::FloorCoeffs::SharedPtr>::iterator find_nearest_floor_coeff(
    const KeyFrame::Ptr& keyframe) {
    auto found = floor_coeffs_queue_.end();
    double best_abs_dt = floor_association_tolerance_sec_;
    for(auto itr = floor_coeffs_queue_.begin(); itr != floor_coeffs_queue_.end(); ++itr) {
      if((*itr)->coeffs.size() < 4) {
        continue;
      }
      const double abs_dt = std::abs((rclcpp::Time((*itr)->header.stamp) - keyframe->stamp).seconds());
      if(abs_dt <= best_abs_dt) {
        best_abs_dt = abs_dt;
        found = itr;
      }
    }
    return found;
  }

  bool retry_floor_constraints() {
    if(!enable_floor_constraints_ || floor_coeffs_queue_.empty()) {
      return false;
    }

    bool updated = false;
    for(const auto& keyframe : keyframes_) {
      updated |= add_floor_constraint(keyframe);
    }
    for(const auto& keyframe : new_keyframes_) {
      updated |= add_floor_constraint(keyframe);
    }
    prune_floor_coeffs_queue();
    return updated;
  }

  void prune_floor_coeffs_queue() {
    if(floor_coeffs_queue_.empty()) {
      return;
    }

    const auto latest = latest_keyframe();
    if(!latest) {
      return;
    }

    const int64_t stale_before_ns = latest->stamp.nanoseconds()
      - static_cast<int64_t>(floor_association_tolerance_sec_ * 1.0e9);
    while(!floor_coeffs_queue_.empty()
      && rclcpp::Time(floor_coeffs_queue_.front()->header.stamp).nanoseconds() < stale_before_ns) {
      floor_coeffs_queue_.pop_front();
    }
  }

  void add_imu_constraint(const KeyFrame::Ptr& keyframe) {
    if(!enable_imu_constraints_) {
      return;
    }
    auto found = imu_queue_.end();
    for(auto itr = imu_queue_.begin(); itr != imu_queue_.end(); ++itr) {
      if(rclcpp::Time((*itr)->header.stamp).nanoseconds() <= keyframe->stamp.nanoseconds()) {
        found = itr;
      } else {
        break;
      }
    }
    if(found == imu_queue_.end()) {
      return;
    }

    const auto& q = (*found)->orientation;
    Eigen::Quaterniond orientation(q.w, q.x, q.y, q.z);
    if(orientation.norm() == 0.0) {
      return;
    }
    orientation.normalize();
    Eigen::MatrixXd information = Eigen::MatrixXd::Identity(3, 3) * (1.0 / imu_orientation_stddev_);
    graph_slam_.add_se3_prior_quat_edge(keyframe->node, orientation, information);
    keyframe->orientation = orientation;
    imu_constraint_count_++;
    imu_queue_.erase(imu_queue_.begin(), std::next(found));
  }

  void add_gps_constraint(const KeyFrame::Ptr& keyframe) {
    if(!enable_gps_constraints_) {
      return;
    }
    auto found = gps_queue_.end();
    for(auto itr = gps_queue_.begin(); itr != gps_queue_.end(); ++itr) {
      if(rclcpp::Time((*itr)->header.stamp).nanoseconds() <= keyframe->stamp.nanoseconds()) {
        found = itr;
      } else {
        break;
      }
    }
    if(found == gps_queue_.end()) {
      return;
    }

    const auto& fix = *found;
    if(std::isnan(fix->latitude) || std::isnan(fix->longitude)) {
      return;
    }

    if(!gps_origin_set_) {
      gps_origin_lat_rad_ = fix->latitude * M_PI / 180.0;
      gps_origin_lon_rad_ = fix->longitude * M_PI / 180.0;
      gps_origin_alt_ = std::isnan(fix->altitude) ? 0.0 : fix->altitude;
      gps_origin_set_ = true;
    }

    constexpr double earth_radius_m = 6378137.0;
    const double lat = fix->latitude * M_PI / 180.0;
    const double lon = fix->longitude * M_PI / 180.0;
    const double x = (lon - gps_origin_lon_rad_) * std::cos(gps_origin_lat_rad_) * earth_radius_m;
    const double y = (lat - gps_origin_lat_rad_) * earth_radius_m;
    const double z = (std::isnan(fix->altitude) ? gps_origin_alt_ : fix->altitude) - gps_origin_alt_;

    Eigen::MatrixXd information = Eigen::MatrixXd::Identity(3, 3) * (1.0 / gps_edge_stddev_);
    graph_slam_.add_se3_prior_xyz_edge(keyframe->node, Eigen::Vector3d(x, y, z), information);
    keyframe->utm_coord = Eigen::Vector3d(x, y, z);
    gps_constraint_count_++;
    gps_queue_.erase(gps_queue_.begin(), std::next(found));
  }

  KeyFrame::Ptr latest_keyframe() const {
    if(!new_keyframes_.empty()) return new_keyframes_.back();
    if(!keyframes_.empty()) return keyframes_.back();
    return nullptr;
  }

  int total_keyframe_count() const {
    return static_cast<int>(keyframes_.size() + new_keyframes_.size());
  }

  void publish_graph_counts() {
    std_msgs::msg::Int32 vertices_msg;
    vertices_msg.data = graph_slam_.num_vertices();
    graph_vertices_pub_->publish(vertices_msg);

    std_msgs::msg::Int32 edges_msg;
    edges_msg.data = graph_slam_.num_edges();
    graph_edges_pub_->publish(edges_msg);
  }

  void publish_floor_constraint_count() {
    if(!enable_debug_topics_ || !floor_count_debug_pub_) {
      return;
    }

    std_msgs::msg::Int32 msg;
    msg.data = floor_constraint_count_;
    floor_count_debug_pub_->publish(msg);
  }

  void optimize_timer_callback() {
    retry_floor_constraints();
    flush_loop_closures();
    update_adaptive_anchor();

    const int vertices = graph_slam_.num_vertices();
    const int edges = graph_slam_.num_edges();
    if(edges < g2o_optimization_min_edges_) {
      RCLCPP_INFO(
        get_logger(),
        "stage4 optimize tick skipped: vertices=%d edges=%d min_edges=%d",
        vertices, edges, g2o_optimization_min_edges_);
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "stage4 optimize tick start: vertices=%d edges=%d",
      vertices, edges);

    const auto t1 = std::chrono::steady_clock::now();
    const int iterations = graph_slam_.optimize(
      g2o_optimization_max_iterations_,
      g2o_optimization_min_edges_);
    const auto t2 = std::chrono::steady_clock::now();
    const double dt_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    RCLCPP_INFO(
      get_logger(),
      "stage4 optimize tick done: iterations=%d dt_ms=%.3f vertices=%d edges=%d floor_constraints=%d loop_constraints=%d",
      iterations, dt_ms, graph_slam_.num_vertices(), graph_slam_.num_edges(), floor_constraint_count_, loop_total_count_);

    publish_map_odom_outputs();
    publish_map_points();
  }

  void flush_loop_closures() {
    if(new_keyframes_.empty()) {
      return;
    }

    const auto loops = loop_detector_.detect(keyframes_, new_keyframes_, graph_slam_);
    for(const auto& loop : loops) {
      const Eigen::Isometry3d relpose(loop->relative_pose.cast<double>());
      const Eigen::MatrixXd information = information_matrix_calculator_.calc_information_matrix(
        loop->key1->cloud, loop->key2->cloud, relpose);
      graph_slam_.add_se3_edge(loop->key1->node, loop->key2->node, relpose, information);
      ++loop_total_count_;
    }

    keyframes_.insert(keyframes_.end(), new_keyframes_.begin(), new_keyframes_.end());
    new_keyframes_.clear();

    if(!loops.empty()) {
      RCLCPP_INFO(
        get_logger(),
        "stage6 loop closures added this tick: %zu, total=%d",
        loops.size(), loop_total_count_);
    }

    if(enable_debug_topics_ && loop_count_debug_pub_) {
      std_msgs::msg::Int32 msg;
      msg.data = loop_total_count_;
      loop_count_debug_pub_->publish(msg);
      publish_graph_counts();
    }
  }

  void update_adaptive_anchor() {
    if(!anchor_node_ || !anchor_edge_ || !fix_first_node_adaptive_) {
      return;
    }

    auto* target_node = static_cast<g2o::VertexSE3*>(anchor_edge_->vertices()[1]);
    if(!target_node) {
      return;
    }
    anchor_node_->setEstimate(target_node->estimate());
  }

  void publish_map_odom_outputs() {
    const auto latest = latest_keyframe();
    if(!latest) {
      return;
    }
    const Eigen::Isometry3d map_T_odom = latest->estimate() * latest->odom.inverse();

    geometry_msgs::msg::TransformStamped msg = tf2::eigenToTransform(map_T_odom);
    msg.header.stamp = latest->stamp;
    msg.header.frame_id = map_frame_id_;
    msg.child_frame_id = odom_frame_id_;

    const auto& t = msg.transform.translation;
    const auto& q = msg.transform.rotation;

    if(publish_map_odom_tf_ && map_odom_broadcaster_) {
      map_odom_broadcaster_->sendTransform(msg);
      RCLCPP_INFO(
        get_logger(),
        "stage7 map->odom tf published: kf=%zu t=[%.3f %.3f %.3f] q=[%.3f %.3f %.3f %.3f]",
        keyframes_.size(), t.x, t.y, t.z, q.x, q.y, q.z, q.w);
    }

    if(!enable_debug_topics_ || !odom2map_debug_pub_) {
      return;
    }

    odom2map_debug_pub_->publish(msg);
    RCLCPP_INFO(
      get_logger(),
      "stage5 odom2map published: kf=%zu t=[%.3f %.3f %.3f] q=[%.3f %.3f %.3f %.3f]",
      keyframes_.size(), t.x, t.y, t.z, q.x, q.y, q.z, q.w);
  }

  void publish_map_points() {
    if(!publish_map_points_ || !map_points_pub_ || keyframes_.empty()) {
      return;
    }

    std::vector<KeyFrameSnapshot::Ptr> snapshot;
    snapshot.reserve(keyframes_.size());
    for(const auto& keyframe : keyframes_) {
      snapshot.push_back(std::make_shared<KeyFrameSnapshot>(keyframe));
    }

    auto cloud = map_cloud_generator_.generate(snapshot, map_cloud_resolution_);
    if(!cloud || cloud->empty()) {
      return;
    }

    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(*cloud, map_msg);
    map_msg.header.stamp = keyframes_.back()->stamp;
    map_msg.header.frame_id = map_frame_id_;
    map_points_pub_->publish(map_msg);

    RCLCPP_INFO(
      get_logger(),
      "stage8 map_points published: keyframes=%zu points=%zu resolution=%.3f",
      snapshot.size(), cloud->size(), map_cloud_resolution_);
  }

  std::vector<KeyFrameSnapshot::Ptr> make_keyframe_snapshot() const {
    std::vector<KeyFrameSnapshot::Ptr> snapshot;
    snapshot.reserve(keyframes_.size());
    for(const auto& keyframe : keyframes_) {
      snapshot.push_back(std::make_shared<KeyFrameSnapshot>(keyframe));
    }
    return snapshot;
  }

  void save_map_service(
    const std::shared_ptr<hdl_graph_slam::srv::SaveMap::Request> req,
    std::shared_ptr<hdl_graph_slam::srv::SaveMap::Response> res) {
    const auto snapshot = make_keyframe_snapshot();
    if(snapshot.empty() || req->destination.empty()) {
      res->success = false;
      return;
    }

    const double resolution = req->resolution > 0.0 ? req->resolution : map_cloud_resolution_;
    auto cloud = map_cloud_generator_.generate(snapshot, resolution);
    if(!cloud || cloud->empty()) {
      res->success = false;
      return;
    }

    cloud->header.frame_id = map_frame_id_;
    const int ret = pcl::io::savePCDFileBinary(req->destination, *cloud);
    res->success = ret == 0;
    RCLCPP_INFO(get_logger(), "save_map service destination=%s success=%s", req->destination.c_str(), res->success ? "true" : "false");
  }

  void dump_graph_service(
    const std::shared_ptr<hdl_graph_slam::srv::DumpGraph::Request> req,
    std::shared_ptr<hdl_graph_slam::srv::DumpGraph::Response> res) {
    std::string directory = req->destination;
    if(directory.empty()) {
      directory = "hdl_graph_slam_dump";
    }
    try {
      boost::filesystem::create_directories(directory);
      graph_slam_.save(directory + "/graph.g2o");
      for(size_t i = 0; i < keyframes_.size(); ++i) {
        std::ostringstream sst;
        sst << directory << "/" << std::setw(6) << std::setfill('0') << i;
        keyframes_[i]->save(sst.str());
      }
      std::ofstream ofs(directory + "/special_nodes.csv");
      ofs << "anchor_node " << (anchor_node_ == nullptr ? -1 : anchor_node_->id()) << std::endl;
      ofs << "anchor_edge " << (anchor_edge_ == nullptr ? -1 : anchor_edge_->id()) << std::endl;
      ofs << "floor_node " << (floor_plane_node_ == nullptr ? -1 : floor_plane_node_->id()) << std::endl;
      res->success = true;
    } catch(const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "dump_graph service failed: %s", e.what());
      res->success = false;
    }
  }

  void load_graph_service(
    const std::shared_ptr<hdl_graph_slam::srv::LoadGraph::Request> req,
    std::shared_ptr<hdl_graph_slam::srv::LoadGraph::Response> res) {
    if(req->path.empty() || graph_slam_.num_vertices() != 0) {
      res->success = false;
      return;
    }
    try {
      if(!graph_slam_.load(req->path + "/graph.g2o")) {
        res->success = false;
        return;
      }
      keyframes_.clear();
      new_keyframes_.clear();
      anchor_node_ = nullptr;
      anchor_edge_ = nullptr;
      floor_plane_node_ = nullptr;
      for(int i = 0;; ++i) {
        std::ostringstream sst;
        sst << req->path << "/" << std::setw(6) << std::setfill('0') << i;
        if(!boost::filesystem::is_directory(sst.str())) {
          break;
        }
        keyframes_.push_back(std::make_shared<KeyFrame>(sst.str(), graph_slam_.graph.get()));
      }
      publish_graph_counts();
      publish_map_odom_outputs();
      publish_map_points();
      res->success = !keyframes_.empty();
    } catch(const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "load_graph service failed: %s", e.what());
      res->success = false;
    }
  }

  std::string published_odom_topic_;
  std::string points_topic_;
  std::string points_qos_;
  int point_cloud_qos_depth_;
  std::string map_frame_id_;
  std::string odom_frame_id_;
  std::string floor_coeffs_topic_;
  std::string imu_topic_;
  std::string gps_topic_;
  int sync_queue_size_;

  double g2o_optimization_interval_sec_;
  double map_cloud_resolution_;
  double floor_edge_stddev_;
  double floor_association_tolerance_sec_;
  double imu_orientation_stddev_;
  double gps_edge_stddev_;
  int g2o_optimization_max_iterations_;
  int g2o_optimization_min_edges_;
  bool enable_debug_topics_;
  bool publish_map_odom_tf_;
  bool publish_map_points_;
  bool fix_first_node_;
  bool fix_first_node_adaptive_;
  bool enable_floor_constraints_;
  bool enable_imu_constraints_;
  bool enable_gps_constraints_;
  std::string fix_first_node_stddev_;

  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
  std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;
  rclcpp::Subscription<hdl_graph_slam::msg::FloorCoeffs>::SharedPtr floor_coeffs_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;

  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr keyframe_count_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr graph_vertices_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr graph_edges_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr odom2map_debug_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr loop_count_debug_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr floor_count_debug_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_points_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> map_odom_broadcaster_;
  rclcpp::Service<hdl_graph_slam::srv::SaveMap>::SharedPtr save_map_service_;
  rclcpp::Service<hdl_graph_slam::srv::DumpGraph>::SharedPtr dump_graph_service_;
  rclcpp::Service<hdl_graph_slam::srv::LoadGraph>::SharedPtr load_graph_service_;

  rclcpp::TimerBase::SharedPtr optimize_timer_;

  KeyframeUpdater keyframe_updater_;
  GraphSLAM graph_slam_;
  InformationMatrixCalculator information_matrix_calculator_;
  LoopDetector loop_detector_;
  MapCloudGenerator map_cloud_generator_;
  std::vector<KeyFrame::Ptr> keyframes_;
  std::deque<KeyFrame::Ptr> new_keyframes_;
  std::deque<hdl_graph_slam::msg::FloorCoeffs::SharedPtr> floor_coeffs_queue_;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_queue_;
  std::deque<sensor_msgs::msg::NavSatFix::SharedPtr> gps_queue_;
  g2o::VertexSE3* anchor_node_ = nullptr;
  g2o::EdgeSE3* anchor_edge_ = nullptr;
  g2o::VertexPlane* floor_plane_node_ = nullptr;
  int loop_total_count_ = 0;
  int floor_constraint_count_ = 0;
  int imu_constraint_count_ = 0;
  int gps_constraint_count_ = 0;
  bool gps_origin_set_ = false;
  double gps_origin_lat_rad_ = 0.0;
  double gps_origin_lon_rad_ = 0.0;
  double gps_origin_alt_ = 0.0;
};

}  // namespace hdl_graph_slam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hdl_graph_slam::GraphSlamNode>());
  rclcpp::shutdown();
  return 0;
}
