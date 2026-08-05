// SPDX-License-Identifier: BSD-2-Clause

#include <deque>
#include <mutex>
#include <string>

#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <hdl_graph_slam/qos.hpp>

namespace hdl_graph_slam {
namespace {

Eigen::Isometry3f odom2isometry(const nav_msgs::msg::Odometry& odom) {
  const auto& q = odom.pose.pose.orientation;
  const auto& p = odom.pose.pose.position;
  Eigen::Quaternionf quat(q.w, q.x, q.y, q.z);
  quat.normalize();

  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.linear() = quat.toRotationMatrix();
  pose.translation() = Eigen::Vector3f(p.x, p.y, p.z);
  return pose;
}

}  // namespace

class SimpleGraphSlamNode : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  explicit SimpleGraphSlamNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : rclcpp::Node("hdl_graph_slam_node", options) {
    points_topic_ = declare_parameter<std::string>("points_topic", "/filtered_points");
    points_qos_ = declare_parameter<std::string>("points_qos", "reliable");
    point_cloud_qos_depth_ = declare_parameter<int>("point_cloud_qos_depth", 32);
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom");
    map_frame_id_ = declare_parameter<std::string>("map_frame_id", "map");
    map_cloud_resolution_ = declare_parameter<double>("map_cloud_resolution", 0.1);
    max_frames_ = declare_parameter<int>("max_frames", 250);

    map_cloud_.reset(new pcl::PointCloud<PointT>());

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 100, std::bind(&SimpleGraphSlamNode::odom_callback, this, std::placeholders::_1));
    points_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic_, point_cloud_qos(points_qos_, point_cloud_qos_depth_),
      std::bind(&SimpleGraphSlamNode::cloud_callback, this, std::placeholders::_1));
    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/hdl_graph_slam/map_points", rclcpp::QoS(1).transient_local());
  }

private:
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_odom_ = *odom_msg;
    has_odom_ = true;
  }

  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    nav_msgs::msg::Odometry odom;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if(!has_odom_) {
        return;
      }
      odom = latest_odom_;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(cloud->empty()) {
      return;
    }

    pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>());
    pcl::transformPointCloud(*cloud, *transformed, odom2isometry(odom).matrix());
    frames_.push_back(transformed);
    while(static_cast<int>(frames_.size()) > max_frames_) {
      frames_.pop_front();
    }

    map_cloud_->clear();
    for(const auto& frame : frames_) {
      *map_cloud_ += *frame;
    }

    if(map_cloud_resolution_ > 0.0) {
      pcl::VoxelGrid<PointT> voxelgrid;
      voxelgrid.setLeafSize(map_cloud_resolution_, map_cloud_resolution_, map_cloud_resolution_);
      voxelgrid.setInputCloud(map_cloud_);
      pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
      voxelgrid.filter(*filtered);
      map_cloud_ = filtered;
    }

    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(*map_cloud_, map_msg);
    map_msg.header.stamp = cloud_msg->header.stamp;
    map_msg.header.frame_id = map_frame_id_;
    map_pub_->publish(map_msg);
  }

private:
  std::mutex mutex_;
  bool has_odom_ = false;
  nav_msgs::msg::Odometry latest_odom_;

  std::string points_topic_;
  std::string points_qos_;
  int point_cloud_qos_depth_;
  std::string odom_topic_;
  std::string map_frame_id_;
  double map_cloud_resolution_;
  int max_frames_;

  std::deque<pcl::PointCloud<PointT>::Ptr> frames_;
  pcl::PointCloud<PointT>::Ptr map_cloud_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
};

}  // namespace hdl_graph_slam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hdl_graph_slam::SimpleGraphSlamNode>());
  rclcpp::shutdown();
  return 0;
}
