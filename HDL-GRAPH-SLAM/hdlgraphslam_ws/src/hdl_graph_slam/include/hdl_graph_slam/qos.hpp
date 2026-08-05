// SPDX-License-Identifier: BSD-2-Clause

#ifndef HDL_GRAPH_SLAM_QOS_HPP
#define HDL_GRAPH_SLAM_QOS_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>

#include <rclcpp/rclcpp.hpp>

namespace hdl_graph_slam {

inline std::string normalize_qos_profile(std::string profile) {
  std::transform(profile.begin(), profile.end(), profile.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  std::replace(profile.begin(), profile.end(), '-', '_');
  return profile;
}

inline rclcpp::QoS point_cloud_qos(const std::string& profile, std::size_t depth = 32) {
  auto qos = rclcpp::QoS(rclcpp::KeepLast(depth)).durability_volatile();
  const auto normalized = normalize_qos_profile(profile);

  if(normalized == "reliable") {
    return qos.reliable();
  }
  if(normalized == "system_default" || normalized == "default") {
    return qos.reliability(RMW_QOS_POLICY_RELIABILITY_SYSTEM_DEFAULT);
  }

  return qos.best_effort();
}

}  // namespace hdl_graph_slam

#endif  // HDL_GRAPH_SLAM_QOS_HPP
