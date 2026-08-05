// SPDX-License-Identifier: BSD-2-Clause

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <hdl_graph_slam/msg/prefiltering_debug.hpp>
#include <hdl_graph_slam/qos.hpp>

namespace hdl_graph_slam {

class PrefilteringNode : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  explicit PrefilteringNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : rclcpp::Node("prefiltering_node", options),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_) {
    initialize_params();

    auto input_qos = point_cloud_qos(input_points_qos_, point_cloud_qos_depth_);
    auto output_qos = point_cloud_qos(output_points_qos_, point_cloud_qos_depth_);

    points_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_points_topic_, input_qos,
      std::bind(&PrefilteringNode::cloud_callback, this, std::placeholders::_1));
    points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_points_topic_, output_qos);
    debug_pub_ = create_publisher<hdl_graph_slam::msg::PrefilteringDebug>(
      "/prefiltering/debug",
      rclcpp::QoS(rclcpp::KeepLast(32)));

    RCLCPP_INFO(
      get_logger(),
      "prefiltering_node: %s (%s) -> %s (%s), distance=%s [%.2f, %.2f], downsample=%s %.3f, outlier=%s",
      input_points_topic_.c_str(),
      input_points_qos_.c_str(),
      output_points_topic_.c_str(),
      output_points_qos_.c_str(),
      use_distance_filter_ ? "enabled" : "disabled",
      distance_near_thresh_,
      distance_far_thresh_,
      downsample_method_.c_str(),
      downsample_resolution_,
      outlier_removal_method_.c_str());
  }

private:
  template<typename T>
  T param(const std::string& name, const T& default_value) {
    return declare_parameter<T>(name, default_value);
  }

  void initialize_params() {
    input_points_topic_ = param<std::string>("input_points_topic", "/velodyne_points");
    output_points_topic_ = param<std::string>("output_points_topic", "/filtered_points");
    input_points_qos_ = param<std::string>("input_points_qos", "best_effort");
    output_points_qos_ = param<std::string>("output_points_qos", "reliable");
    point_cloud_qos_depth_ = param<int>("point_cloud_qos_depth", 32);
    base_link_frame_ = param<std::string>("base_link_frame", "base_link");

    use_distance_filter_ = param<bool>("use_distance_filter", true);
    distance_near_thresh_ = param<double>("distance_near_thresh", 0.5);
    distance_far_thresh_ = param<double>("distance_far_thresh", 20.0);

    downsample_method_ = param<std::string>("downsample_method", "VOXELGRID");
    downsample_resolution_ = param<double>("downsample_resolution", 0.1);
    if(downsample_method_ == "VOXELGRID") {
      auto voxelgrid = std::make_shared<pcl::VoxelGrid<PointT>>();
      voxelgrid->setLeafSize(downsample_resolution_, downsample_resolution_, downsample_resolution_);
      downsample_filter_ = voxelgrid;
    } else if(downsample_method_ == "APPROX_VOXELGRID") {
      auto approx_voxelgrid = std::make_shared<pcl::ApproximateVoxelGrid<PointT>>();
      approx_voxelgrid->setLeafSize(downsample_resolution_, downsample_resolution_, downsample_resolution_);
      downsample_filter_ = approx_voxelgrid;
    } else if(downsample_method_ != "NONE") {
      RCLCPP_WARN(get_logger(), "unknown downsample_method '%s'; disabling downsample", downsample_method_.c_str());
      downsample_method_ = "NONE";
    }

    outlier_removal_method_ = param<std::string>("outlier_removal_method", "RADIUS");
    if(outlier_removal_method_ == "STATISTICAL") {
      auto sor = std::make_shared<pcl::StatisticalOutlierRemoval<PointT>>();
      sor->setMeanK(param<int>("statistical_mean_k", 30));
      sor->setStddevMulThresh(param<double>("statistical_stddev", 1.2));
      outlier_removal_filter_ = sor;
    } else if(outlier_removal_method_ == "RADIUS") {
      auto radius = std::make_shared<pcl::RadiusOutlierRemoval<PointT>>();
      radius->setRadiusSearch(param<double>("radius_radius", 0.5));
      radius->setMinNeighborsInRadius(param<int>("radius_min_neighbors", 2));
      outlier_removal_filter_ = radius;
    } else if(outlier_removal_method_ != "NONE") {
      RCLCPP_WARN(get_logger(), "unknown outlier_removal_method '%s'; disabling outlier removal", outlier_removal_method_.c_str());
      outlier_removal_method_ = "NONE";
    }
  }

  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    const auto callback_start = std::chrono::steady_clock::now();

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(cloud->empty()) {
      return;
    }
    pcl_conversions::toPCL(cloud_msg->header, cloud->header);

    pcl::PointCloud<PointT>::ConstPtr filtered = transform_to_base_link(cloud);
    const size_t transformed_count = filtered->size();
    filtered = distance_filter(filtered);
    const size_t distance_filtered_count = filtered->size();
    filtered = downsample(filtered);
    const size_t downsampled_count = filtered->size();
    filtered = outlier_removal(filtered);
    const size_t output_count = filtered->size();

    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*filtered, output);
    pcl_conversions::fromPCL(filtered->header, output.header);
    points_pub_->publish(output);
    ++published_cloud_count_;

    publish_debug(
      *cloud_msg,
      cloud->size(),
      transformed_count,
      distance_filtered_count,
      downsampled_count,
      output_count,
      callback_start);
  }

  pcl::PointCloud<PointT>::ConstPtr transform_to_base_link(const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    if(base_link_frame_.empty() || cloud->header.frame_id == base_link_frame_) {
      return cloud;
    }

    geometry_msgs::msg::TransformStamped transform;
    try {
      transform = tf_buffer_.lookupTransform(base_link_frame_, cloud->header.frame_id, tf2::TimePointZero);
    } catch(const tf2::TransformException& e) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "failed to transform cloud from %s to %s: %s",
        cloud->header.frame_id.c_str(),
        base_link_frame_.c_str(),
        e.what());
      return cloud;
    }

    const auto& t = transform.transform.translation;
    const auto& r = transform.transform.rotation;
    Eigen::Quaternionf q(r.w, r.x, r.y, r.z);
    q.normalize();

    Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
    matrix.block<3, 3>(0, 0) = q.toRotationMatrix();
    matrix.block<3, 1>(0, 3) = Eigen::Vector3f(t.x, t.y, t.z);

    pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>());
    pcl::transformPointCloud(*cloud, *transformed, matrix);
    transformed->header = cloud->header;
    transformed->header.frame_id = base_link_frame_;
    return transformed;
  }

  pcl::PointCloud<PointT>::ConstPtr distance_filter(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!use_distance_filter_) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    filtered->reserve(cloud->size());
    std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const PointT& p) {
      const auto range = p.getVector3fMap().norm();
      return std::isfinite(range) && range > distance_near_thresh_ && range < distance_far_thresh_;
    });

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;
    filtered->header = cloud->header;
    return filtered;
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

  pcl::PointCloud<PointT>::ConstPtr outlier_removal(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!outlier_removal_filter_) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    outlier_removal_filter_->setInputCloud(cloud);
    outlier_removal_filter_->filter(*filtered);
    filtered->header = cloud->header;
    return filtered;
  }

  void publish_debug(
    const sensor_msgs::msg::PointCloud2& cloud_msg,
    size_t input_count,
    size_t transformed_count,
    size_t distance_filtered_count,
    size_t downsampled_count,
    size_t output_count,
    const std::chrono::steady_clock::time_point& callback_start) {
    if(debug_pub_->get_subscription_count() == 0) {
      return;
    }

    const auto callback_end = std::chrono::steady_clock::now();
    const double callback_ms = std::chrono::duration<double, std::milli>(callback_end - callback_start).count();

    hdl_graph_slam::msg::PrefilteringDebug debug;
    debug.header = cloud_msg.header;
    debug.input_point_count = static_cast<uint32_t>(input_count);
    debug.transformed_point_count = static_cast<uint32_t>(transformed_count);
    debug.distance_filtered_point_count = static_cast<uint32_t>(distance_filtered_count);
    debug.downsampled_point_count = static_cast<uint32_t>(downsampled_count);
    debug.output_point_count = static_cast<uint32_t>(output_count);
    debug.callback_time_ms = static_cast<float>(callback_ms);
    debug.output_count = published_cloud_count_;
    debug_pub_->publish(debug);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub_;
  rclcpp::Publisher<hdl_graph_slam::msg::PrefilteringDebug>::SharedPtr debug_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string input_points_topic_;
  std::string output_points_topic_;
  std::string input_points_qos_;
  std::string output_points_qos_;
  int point_cloud_qos_depth_;
  std::string base_link_frame_;

  bool use_distance_filter_;
  double distance_near_thresh_;
  double distance_far_thresh_;

  std::string downsample_method_;
  double downsample_resolution_;
  std::string outlier_removal_method_;
  pcl::Filter<PointT>::Ptr downsample_filter_;
  pcl::Filter<PointT>::Ptr outlier_removal_filter_;
  uint64_t published_cloud_count_ = 0;
};

}  // namespace hdl_graph_slam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hdl_graph_slam::PrefilteringNode>());
  rclcpp::shutdown();
  return 0;
}
