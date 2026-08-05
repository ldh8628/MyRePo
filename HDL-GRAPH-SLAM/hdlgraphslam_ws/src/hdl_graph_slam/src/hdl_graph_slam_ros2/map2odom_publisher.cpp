// SPDX-License-Identifier: BSD-2-Clause

#include <memory>
#include <string>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>

class Map2OdomPublisher : public rclcpp::Node {
public:
  Map2OdomPublisher()
  : rclcpp::Node("map2odom_publisher"),
    broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)) {
    map_frame_id_ = declare_parameter<std::string>("map_frame_id", "map");
    odom_frame_id_ = declare_parameter<std::string>("odom_frame_id", "odom");
    publish_period_ = declare_parameter<double>("publish_period", 0.05);

    timer_ = create_wall_timer(
      std::chrono::duration<double>(publish_period_),
      std::bind(&Map2OdomPublisher::publish_transform, this));
  }

private:
  void publish_transform() {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = now();
    transform.header.frame_id = map_frame_id_;
    transform.child_frame_id = odom_frame_id_;
    transform.transform.rotation.w = 1.0;
    broadcaster_->sendTransform(transform);
  }

  std::string map_frame_id_;
  std::string odom_frame_id_;
  double publish_period_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Map2OdomPublisher>());
  rclcpp::shutdown();
  return 0;
}
