// SPDX-License-Identifier: BSD-2-Clause

#include <memory>
#include <chrono>
#include <string>

#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <hdl_graph_slam/msg/scan_matching_odometry_debug.hpp>
#include <hdl_graph_slam/msg/scan_matching_status.hpp>
#include <hdl_graph_slam/qos.hpp>
#include <hdl_graph_slam/registrations.hpp>

namespace hdl_graph_slam {
namespace {

geometry_msgs::msg::TransformStamped matrix2transform(
  const rclcpp::Time& stamp,
  const Eigen::Matrix4f& pose,
  const std::string& frame_id,
  const std::string& child_frame_id) {
  Eigen::Quaternionf quat(pose.block<3, 3>(0, 0));
  quat.normalize();

  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = stamp;
  transform.header.frame_id = frame_id;
  transform.child_frame_id = child_frame_id;
  transform.transform.translation.x = pose(0, 3);
  transform.transform.translation.y = pose(1, 3);
  transform.transform.translation.z = pose(2, 3);
  transform.transform.rotation.w = quat.w();
  transform.transform.rotation.x = quat.x();
  transform.transform.rotation.y = quat.y();
  transform.transform.rotation.z = quat.z();
  return transform;
}

geometry_msgs::msg::Pose isometry2pose(const Eigen::Isometry3d& mat) {
  Eigen::Quaterniond quat(mat.linear());
  Eigen::Vector3d trans = mat.translation();

  geometry_msgs::msg::Pose pose;
  pose.position.x = trans.x();
  pose.position.y = trans.y();
  pose.position.z = trans.z();
  pose.orientation.w = quat.w();
  pose.orientation.x = quat.x();
  pose.orientation.y = quat.y();
  pose.orientation.z = quat.z();
  return pose;
}

}  // namespace

class ScanMatchingOdometryNode : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  explicit ScanMatchingOdometryNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : rclcpp::Node("scan_matching_odometry_node", options),
    odom_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)),
    keyframe_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)) {
    initialize_params();

    points_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic_, point_cloud_qos(points_qos_, point_cloud_qos_depth_),
      std::bind(&ScanMatchingOdometryNode::cloud_callback, this, std::placeholders::_1));

    read_until_pub_ = create_publisher<std_msgs::msg::Header>("/scan_matching_odometry/read_until", 32);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(published_odom_topic_, 32);
    trans_pub_ = create_publisher<geometry_msgs::msg::TransformStamped>("/scan_matching_odometry/transform", 32);
    status_pub_ = create_publisher<hdl_graph_slam::msg::ScanMatchingStatus>("/scan_matching_odometry/status", 8);
    debug_pub_ = create_publisher<hdl_graph_slam::msg::ScanMatchingOdometryDebug>(
      "/scan_matching_odometry/debug",
      rclcpp::QoS(rclcpp::KeepLast(32)));
    aligned_points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/aligned_points", 32);
  }

private:
  struct DebugMetrics {
    size_t input_point_count = 0;
    size_t downsampled_point_count = 0;
    double registration_time_ms = 0.0;
    bool registration_triggered = false;
    bool has_converged = false;
    double matching_error = 0.0;
  };

  template<typename T>
  T param(const std::string& name, const T& default_value) {
    return declare_parameter<T>(name, default_value);
  }

  void initialize_params() {
    published_odom_topic_ = param<std::string>("published_odom_topic", "/odom");
    points_topic_ = param<std::string>("points_topic", "/filtered_points");
    points_qos_ = param<std::string>("points_qos", "reliable");
    point_cloud_qos_depth_ = param<int>("point_cloud_qos_depth", 32);
    odom_frame_id_ = param<std::string>("odom_frame_id", "odom");

    RCLCPP_INFO(
      get_logger(),
      "scan_matching_odometry_node: subscribing to %s with %s QoS",
      points_topic_.c_str(),
      points_qos_.c_str());

    keyframe_delta_trans_ = param<double>("keyframe_delta_trans", 0.25);
    keyframe_delta_angle_ = param<double>("keyframe_delta_angle", 0.15);
    keyframe_delta_time_ = param<double>("keyframe_delta_time", 1.0);
    transform_thresholding_ = param<bool>("transform_thresholding", false);
    max_acceptable_trans_ = param<double>("max_acceptable_trans", 1.0);
    max_acceptable_angle_ = param<double>("max_acceptable_angle", 1.0);

    std::string downsample_method = param<std::string>("downsample_method", "VOXELGRID");
    double downsample_resolution = param<double>("downsample_resolution", 0.1);
    if(downsample_method == "VOXELGRID") {
      auto voxelgrid = std::make_shared<pcl::VoxelGrid<PointT>>();
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter_ = voxelgrid;
    } else if(downsample_method == "APPROX_VOXELGRID") {
      auto approx_voxelgrid = std::make_shared<pcl::ApproximateVoxelGrid<PointT>>();
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter_ = approx_voxelgrid;
    } else {
      downsample_filter_ = std::make_shared<pcl::PassThrough<PointT>>();
    }

    registration_ = select_registration_method(*this);
  }

  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    const auto callback_start = std::chrono::steady_clock::now();

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(cloud->empty()) {
      return;
    }
    pcl_conversions::toPCL(cloud_msg->header, cloud->header);

    DebugMetrics debug;
    debug.input_point_count = cloud->size();

    Eigen::Matrix4f pose = matching(cloud_msg->header.stamp, cloud_msg->header.frame_id, cloud, debug);
    publish_odometry(cloud_msg->header.stamp, cloud_msg->header.frame_id, pose);
    ++published_odom_count_;

    std_msgs::msg::Header read_until;
    read_until.frame_id = points_topic_;
    read_until.stamp = rclcpp::Time(cloud_msg->header.stamp) + rclcpp::Duration(1, 0);
    read_until_pub_->publish(read_until);

    publish_debug(*cloud_msg, debug, callback_start);
  }

  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter_) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter_->setInputCloud(cloud);
    downsample_filter_->filter(*filtered);
    filtered->header = cloud->header;
    return filtered;
  }

  Eigen::Matrix4f matching(
    const builtin_interfaces::msg::Time& stamp_msg,
    const std::string& frame_id,
    const pcl::PointCloud<PointT>::ConstPtr& cloud,
    DebugMetrics& debug) {
    const rclcpp::Time stamp(stamp_msg);

    if(!keyframe_) {
      prev_time_ = rclcpp::Time(0, 0, stamp.get_clock_type());
      prev_trans_.setIdentity();
      keyframe_pose_.setIdentity();
      keyframe_stamp_ = stamp;
      keyframe_ = downsample(cloud);
      debug.downsampled_point_count = keyframe_->size();
      registration_->setInputTarget(keyframe_);
      return Eigen::Matrix4f::Identity();
    }

    auto filtered = downsample(cloud);
    debug.downsampled_point_count = filtered->size();
    registration_->setInputSource(filtered);

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    const auto registration_start = std::chrono::steady_clock::now();
    registration_->align(*aligned, prev_trans_);
    const auto registration_end = std::chrono::steady_clock::now();
    debug.registration_time_ms = std::chrono::duration<double, std::milli>(registration_end - registration_start).count();
    debug.registration_triggered = true;
    debug.has_converged = registration_->hasConverged();
    debug.matching_error = registration_->getFitnessScore();

    publish_scan_matching_status(stamp_msg, frame_id, aligned);

    if(!registration_->hasConverged()) {
      RCLCPP_INFO(get_logger(), "scan matching has not converged; ignoring frame");
      return keyframe_pose_ * prev_trans_;
    }

    Eigen::Matrix4f trans = registration_->getFinalTransformation();
    Eigen::Matrix4f odom = keyframe_pose_ * trans;

    if(transform_thresholding_) {
      Eigen::Matrix4f delta = prev_trans_.inverse() * trans;
      double dx = delta.block<3, 1>(0, 3).norm();
      double da = std::acos(Eigen::Quaternionf(delta.block<3, 3>(0, 0)).w());
      if(dx > max_acceptable_trans_ || da > max_acceptable_angle_) {
        RCLCPP_INFO(get_logger(), "too large transform: %.3f m %.3f rad; ignoring frame", dx, da);
        return keyframe_pose_ * prev_trans_;
      }
    }

    prev_time_ = stamp;
    prev_trans_ = trans;

    keyframe_broadcaster_->sendTransform(matrix2transform(stamp, keyframe_pose_, odom_frame_id_, "keyframe"));

    double delta_trans = trans.block<3, 1>(0, 3).norm();
    double delta_angle = std::acos(Eigen::Quaternionf(trans.block<3, 3>(0, 0)).w());
    double delta_time = (stamp - keyframe_stamp_).seconds();
    if(delta_trans > keyframe_delta_trans_ || delta_angle > keyframe_delta_angle_ || delta_time > keyframe_delta_time_) {
      keyframe_ = filtered;
      registration_->setInputTarget(keyframe_);
      keyframe_pose_ = odom;
      keyframe_stamp_ = stamp;
      prev_time_ = stamp;
      prev_trans_.setIdentity();
    }

    if(aligned_points_pub_->get_subscription_count() > 0) {
      pcl::transformPointCloud(*cloud, *aligned, odom);
      sensor_msgs::msg::PointCloud2 msg;
      pcl::toROSMsg(*aligned, msg);
      msg.header.stamp = stamp_msg;
      msg.header.frame_id = odom_frame_id_;
      aligned_points_pub_->publish(msg);
    }

    return odom;
  }

  void publish_odometry(
    const builtin_interfaces::msg::Time& stamp_msg,
    const std::string& base_frame_id,
    const Eigen::Matrix4f& pose) {
    rclcpp::Time stamp(stamp_msg);
    auto odom_trans = matrix2transform(stamp, pose, odom_frame_id_, base_frame_id);
    trans_pub_->publish(odom_trans);
    odom_broadcaster_->sendTransform(odom_trans);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp_msg;
    odom.header.frame_id = odom_frame_id_;
    odom.child_frame_id = base_frame_id;
    odom.pose.pose.position.x = pose(0, 3);
    odom.pose.pose.position.y = pose(1, 3);
    odom.pose.pose.position.z = pose(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;
    odom_pub_->publish(odom);
  }

  void publish_scan_matching_status(
    const builtin_interfaces::msg::Time& stamp,
    const std::string& frame_id,
    const pcl::PointCloud<PointT>::ConstPtr& aligned) {
    if(status_pub_->get_subscription_count() == 0) {
      return;
    }

    hdl_graph_slam::msg::ScanMatchingStatus status;
    status.header.stamp = stamp;
    status.header.frame_id = frame_id;
    status.has_converged = registration_->hasConverged();
    status.matching_error = registration_->getFitnessScore();

    const double max_correspondence_dist = 0.5;
    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    if(registration_->getSearchMethodTarget()) {
      for(const auto& pt : aligned->points) {
        registration_->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
        if(!k_sq_dists.empty() && k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
          num_inliers++;
        }
      }
    }

    status.inlier_fraction = aligned->empty() ? 0.0f : static_cast<float>(num_inliers) / aligned->size();
    status.relative_pose = isometry2pose(Eigen::Isometry3f(registration_->getFinalTransformation()).cast<double>());
    status_pub_->publish(status);
  }

  void publish_debug(
    const sensor_msgs::msg::PointCloud2& cloud_msg,
    const DebugMetrics& metrics,
    const std::chrono::steady_clock::time_point& callback_start) {
    if(debug_pub_->get_subscription_count() == 0) {
      return;
    }

    const auto callback_end = std::chrono::steady_clock::now();
    const double callback_ms = std::chrono::duration<double, std::milli>(callback_end - callback_start).count();

    hdl_graph_slam::msg::ScanMatchingOdometryDebug debug;
    debug.header = cloud_msg.header;
    debug.input_point_count = static_cast<uint32_t>(metrics.input_point_count);
    debug.downsampled_point_count = static_cast<uint32_t>(metrics.downsampled_point_count);
    debug.registration_time_ms = static_cast<float>(metrics.registration_time_ms);
    debug.callback_time_ms = static_cast<float>(callback_ms);
    debug.odom_count = published_odom_count_;
    debug.registration_triggered = metrics.registration_triggered;
    debug.has_converged = metrics.has_converged;
    debug.matching_error = static_cast<float>(metrics.matching_error);
    debug_pub_->publish(debug);
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr read_until_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr trans_pub_;
  rclcpp::Publisher<hdl_graph_slam::msg::ScanMatchingStatus>::SharedPtr status_pub_;
  rclcpp::Publisher<hdl_graph_slam::msg::ScanMatchingOdometryDebug>::SharedPtr debug_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_points_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> odom_broadcaster_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> keyframe_broadcaster_;

  std::string published_odom_topic_;
  std::string points_topic_;
  std::string points_qos_;
  int point_cloud_qos_depth_;
  std::string odom_frame_id_;

  double keyframe_delta_trans_;
  double keyframe_delta_angle_;
  double keyframe_delta_time_;
  bool transform_thresholding_;
  double max_acceptable_trans_;
  double max_acceptable_angle_;

  rclcpp::Time prev_time_;
  Eigen::Matrix4f prev_trans_;
  Eigen::Matrix4f keyframe_pose_;
  rclcpp::Time keyframe_stamp_;
  pcl::PointCloud<PointT>::ConstPtr keyframe_;

  pcl::Filter<PointT>::Ptr downsample_filter_;
  pcl::Registration<PointT, PointT>::Ptr registration_;
  uint64_t published_odom_count_ = 0;
};

}  // namespace hdl_graph_slam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hdl_graph_slam::ScanMatchingOdometryNode>());
  rclcpp::shutdown();
  return 0;
}
