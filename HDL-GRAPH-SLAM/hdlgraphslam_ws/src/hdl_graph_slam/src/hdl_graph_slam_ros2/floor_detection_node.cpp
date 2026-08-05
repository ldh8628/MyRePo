// SPDX-License-Identifier: BSD-2-Clause

#include <memory>
#include <string>

#include <boost/optional.hpp>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/impl/plane_clipper3D.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include <hdl_graph_slam/msg/floor_coeffs.hpp>
#include <hdl_graph_slam/qos.hpp>

namespace hdl_graph_slam {

class FloorDetectionNode : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  explicit FloorDetectionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : rclcpp::Node("floor_detection_node", options) {
    initialize_params();

    points_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic_, point_cloud_qos(points_qos_, point_cloud_qos_depth_),
      std::bind(&FloorDetectionNode::cloud_callback, this, std::placeholders::_1));

    floor_pub_ = create_publisher<hdl_graph_slam::msg::FloorCoeffs>("/floor_detection/floor_coeffs", 32);
    read_until_pub_ = create_publisher<std_msgs::msg::Header>("/floor_detection/read_until", 32);
    floor_filtered_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/floor_detection/floor_filtered_points", 32);
    floor_points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/floor_detection/floor_points", 32);

    RCLCPP_INFO(
      get_logger(),
      "floor_detection_node: subscribing to %s with %s QoS",
      points_topic_.c_str(),
      points_qos_.c_str());
  }

private:
  template<typename T>
  T param(const std::string& name, const T& default_value) {
    return declare_parameter<T>(name, default_value);
  }

  void initialize_params() {
    points_topic_ = param<std::string>("points_topic", "/filtered_points");
    points_qos_ = param<std::string>("points_qos", "reliable");
    point_cloud_qos_depth_ = param<int>("point_cloud_qos_depth", 32);
    tilt_deg_ = param<double>("tilt_deg", 0.0);
    sensor_height_ = param<double>("sensor_height", 2.0);
    height_clip_range_ = param<double>("height_clip_range", 1.0);
    floor_pts_thresh_ = param<int>("floor_pts_thresh", 512);
    floor_normal_thresh_ = param<double>("floor_normal_thresh", 10.0);
    use_normal_filtering_ = param<bool>("use_normal_filtering", true);
    normal_filter_thresh_ = param<double>("normal_filter_thresh", 20.0);
  }

  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(cloud->empty()) {
      return;
    }
    pcl_conversions::toPCL(cloud_msg->header, cloud->header);

    boost::optional<Eigen::Vector4f> floor = detect(cloud);

    hdl_graph_slam::msg::FloorCoeffs coeffs;
    coeffs.header = cloud_msg->header;
    if(floor) {
      coeffs.coeffs.resize(4);
      for(int i = 0; i < 4; i++) {
        coeffs.coeffs[i] = (*floor)[i];
      }
    }
    floor_pub_->publish(coeffs);

    std_msgs::msg::Header read_until;
    read_until.frame_id = points_topic_;
    read_until.stamp = rclcpp::Time(cloud_msg->header.stamp) + rclcpp::Duration(1, 0);
    read_until_pub_->publish(read_until);
  }

  boost::optional<Eigen::Vector4f> detect(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    Eigen::Matrix4f tilt_matrix = Eigen::Matrix4f::Identity();
    tilt_matrix.topLeftCorner(3, 3) =
      Eigen::AngleAxisf(tilt_deg_ * M_PI / 180.0f, Eigen::Vector3f::UnitY()).toRotationMatrix();

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*cloud, *filtered, tilt_matrix);
    filtered = plane_clip(filtered, Eigen::Vector4f(0.0f, 0.0f, 1.0f, sensor_height_ + height_clip_range_), false);
    filtered = plane_clip(filtered, Eigen::Vector4f(0.0f, 0.0f, 1.0f, sensor_height_ - height_clip_range_), true);

    if(use_normal_filtering_) {
      filtered = normal_filtering(filtered);
    }

    pcl::transformPointCloud(*filtered, *filtered, static_cast<Eigen::Matrix4f>(tilt_matrix.inverse()));

    if(floor_filtered_pub_->get_subscription_count() > 0) {
      sensor_msgs::msg::PointCloud2 msg;
      pcl::toROSMsg(*filtered, msg);
      pcl_conversions::fromPCL(filtered->header, msg.header);
      floor_filtered_pub_->publish(msg);
    }

    if(static_cast<int>(filtered->size()) < floor_pts_thresh_) {
      return boost::none;
    }

    pcl::SampleConsensusModelPlane<PointT>::Ptr model_p(new pcl::SampleConsensusModelPlane<PointT>(filtered));
    pcl::RandomSampleConsensus<PointT> ransac(model_p);
    ransac.setDistanceThreshold(0.1);
    ransac.computeModel();

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    ransac.getInliers(inliers->indices);

    if(static_cast<int>(inliers->indices.size()) < floor_pts_thresh_) {
      return boost::none;
    }

    Eigen::Vector4f reference = tilt_matrix.inverse() * Eigen::Vector4f::UnitZ();
    Eigen::VectorXf coeffs;
    ransac.getModelCoefficients(coeffs);

    double dot = coeffs.head<3>().dot(reference.head<3>());
    if(std::abs(dot) < std::cos(floor_normal_thresh_ * M_PI / 180.0)) {
      return boost::none;
    }

    if(coeffs.head<3>().dot(Eigen::Vector3f::UnitZ()) < 0.0f) {
      coeffs *= -1.0f;
    }

    if(floor_points_pub_->get_subscription_count() > 0) {
      pcl::PointCloud<PointT>::Ptr inlier_cloud(new pcl::PointCloud<PointT>);
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud(filtered);
      extract.setIndices(inliers);
      extract.filter(*inlier_cloud);

      sensor_msgs::msg::PointCloud2 msg;
      pcl::toROSMsg(*inlier_cloud, msg);
      pcl_conversions::fromPCL(filtered->header, msg.header);
      floor_points_pub_->publish(msg);
    }

    return Eigen::Vector4f(coeffs);
  }

  pcl::PointCloud<PointT>::Ptr plane_clip(
    const pcl::PointCloud<PointT>::Ptr& src_cloud,
    const Eigen::Vector4f& plane,
    bool negative) const {
    pcl::PlaneClipper3D<PointT> clipper(plane);
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    clipper.clipPointCloud3D(*src_cloud, indices->indices);

    pcl::PointCloud<PointT>::Ptr dst_cloud(new pcl::PointCloud<PointT>);
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(src_cloud);
    extract.setIndices(indices);
    extract.setNegative(negative);
    extract.filter(*dst_cloud);
    dst_cloud->header = src_cloud->header;
    return dst_cloud;
  }

  pcl::PointCloud<PointT>::Ptr normal_filtering(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>));

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(10);
    ne.setViewPoint(0.0f, 0.0f, sensor_height_);
    ne.compute(*normals);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    filtered->reserve(cloud->size());
    for(size_t i = 0; i < cloud->size(); i++) {
      float dot = normals->at(i).getNormalVector3fMap().normalized().dot(Eigen::Vector3f::UnitZ());
      if(std::abs(dot) > std::cos(normal_filter_thresh_ * M_PI / 180.0)) {
        filtered->push_back(cloud->at(i));
      }
    }

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;
    filtered->header = cloud->header;
    return filtered;
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Publisher<hdl_graph_slam::msg::FloorCoeffs>::SharedPtr floor_pub_;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr read_until_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr floor_filtered_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr floor_points_pub_;

  std::string points_topic_;
  std::string points_qos_;
  int point_cloud_qos_depth_;
  double tilt_deg_;
  double sensor_height_;
  double height_clip_range_;
  int floor_pts_thresh_;
  double floor_normal_thresh_;
  bool use_normal_filtering_;
  double normal_filter_thresh_;
};

}  // namespace hdl_graph_slam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hdl_graph_slam::FloorDetectionNode>());
  rclcpp::shutdown();
  return 0;
}
