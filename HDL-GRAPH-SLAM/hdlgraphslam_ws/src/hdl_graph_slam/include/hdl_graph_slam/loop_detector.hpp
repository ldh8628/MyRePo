// SPDX-License-Identifier: BSD-2-Clause

#ifndef LOOP_DETECTOR_HPP
#define LOOP_DETECTOR_HPP

#include <chrono>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <boost/format.hpp>
#include <rclcpp/rclcpp.hpp>

#include <g2o/types/slam3d/vertex_se3.h>

#include <hdl_graph_slam/graph_slam.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/registrations.hpp>

namespace hdl_graph_slam {

struct Loop {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Loop>;

  Loop(const KeyFrame::Ptr& key1, const KeyFrame::Ptr& key2, const Eigen::Matrix4f& relpose)
  : key1(key1), key2(key2), relative_pose(relpose) {}

  KeyFrame::Ptr key1;
  KeyFrame::Ptr key2;
  Eigen::Matrix4f relative_pose;
};

class LoopDetector {
public:
  using PointT = pcl::PointXYZI;

  explicit LoopDetector(rclcpp::Node& node) {
    distance_thresh_ = node.declare_parameter<double>("loop_distance_thresh", 5.0);
    accum_distance_thresh_ = node.declare_parameter<double>("loop_accum_distance_thresh", 8.0);
    distance_from_last_edge_thresh_ = node.declare_parameter<double>("loop_min_edge_interval", 5.0);

    fitness_score_max_range_ = node.declare_parameter<double>(
      "loop_fitness_score_max_range", std::numeric_limits<double>::max());
    fitness_score_thresh_ = node.declare_parameter<double>("loop_fitness_score_thresh", 0.5);

    registration_ = select_registration_method(node);
    last_edge_accum_distance_ = 0.0;
  }

  std::vector<Loop::Ptr> detect(
    const std::vector<KeyFrame::Ptr>& keyframes,
    const std::deque<KeyFrame::Ptr>& new_keyframes,
    hdl_graph_slam::GraphSLAM& graph_slam) {
    std::vector<Loop::Ptr> detected_loops;
    for(const auto& new_keyframe : new_keyframes) {
      auto candidates = find_candidates(keyframes, new_keyframe);
      auto loop = matching(candidates, new_keyframe, graph_slam);
      if(loop) {
        detected_loops.push_back(loop);
      }
    }
    return detected_loops;
  }

  double get_distance_thresh() const {
    return distance_thresh_;
  }

private:
  std::vector<KeyFrame::Ptr> find_candidates(
    const std::vector<KeyFrame::Ptr>& keyframes,
    const KeyFrame::Ptr& new_keyframe) const {
    if(new_keyframe->accum_distance - last_edge_accum_distance_ < distance_from_last_edge_thresh_) {
      return {};
    }

    std::vector<KeyFrame::Ptr> candidates;
    candidates.reserve(32);

    for(const auto& k : keyframes) {
      if(new_keyframe->accum_distance - k->accum_distance < accum_distance_thresh_) {
        continue;
      }

      const auto& pos1 = k->estimate().translation();
      const auto& pos2 = new_keyframe->estimate().translation();
      const double dist = (pos1.head<2>() - pos2.head<2>()).norm();
      if(dist > distance_thresh_) {
        continue;
      }

      candidates.push_back(k);
    }

    return candidates;
  }

  Loop::Ptr matching(
    const std::vector<KeyFrame::Ptr>& candidate_keyframes,
    const KeyFrame::Ptr& new_keyframe,
    hdl_graph_slam::GraphSLAM& /*graph_slam*/) {
    if(candidate_keyframes.empty()) {
      return nullptr;
    }

    registration_->setInputTarget(new_keyframe->cloud);

    double best_score = std::numeric_limits<double>::max();
    KeyFrame::Ptr best_matched;
    Eigen::Matrix4f relative_pose = Eigen::Matrix4f::Identity();

    std::cout << "\n--- loop detection ---\n";
    std::cout << "num_candidates: " << candidate_keyframes.size() << "\n";
    std::cout << "matching" << std::flush;
    const auto t1 = std::chrono::steady_clock::now();

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    for(const auto& candidate : candidate_keyframes) {
      registration_->setInputSource(candidate->cloud);
      Eigen::Isometry3d new_keyframe_estimate = new_keyframe->estimate();
      new_keyframe_estimate.linear() =
        Eigen::Quaterniond(new_keyframe_estimate.linear()).normalized().toRotationMatrix();
      Eigen::Isometry3d candidate_estimate = candidate->estimate();
      candidate_estimate.linear() =
        Eigen::Quaterniond(candidate_estimate.linear()).normalized().toRotationMatrix();
      Eigen::Matrix4f guess = (new_keyframe_estimate.inverse() * candidate_estimate).matrix().cast<float>();
      guess(2, 3) = 0.0;
      registration_->align(*aligned, guess);
      std::cout << "." << std::flush;

      const double score = registration_->getFitnessScore(fitness_score_max_range_);
      if(!registration_->hasConverged() || score > best_score) {
        continue;
      }

      best_score = score;
      best_matched = candidate;
      relative_pose = registration_->getFinalTransformation();
    }

    const auto t2 = std::chrono::steady_clock::now();
    const double dt_sec = std::chrono::duration<double>(t2 - t1).count();
    std::cout << " done\nbest_score: " << boost::format("%.3f") % best_score
              << "    time: " << boost::format("%.3f") % dt_sec << "[sec]\n";

    if(best_score > fitness_score_thresh_) {
      std::cout << "loop not found...\n";
      return nullptr;
    }

    std::cout << "loop found!!\nrelpose: "
              << relative_pose.block<3, 1>(0, 3).transpose() << " - "
              << Eigen::Quaternionf(relative_pose.block<3, 3>(0, 0)).coeffs().transpose() << "\n";

    last_edge_accum_distance_ = new_keyframe->accum_distance;

    return std::make_shared<Loop>(new_keyframe, best_matched, relative_pose);
  }

  double distance_thresh_;
  double accum_distance_thresh_;
  double distance_from_last_edge_thresh_;
  double fitness_score_max_range_;
  double fitness_score_thresh_;
  double last_edge_accum_distance_;

  pcl::Registration<PointT, PointT>::Ptr registration_;
};

}  // namespace hdl_graph_slam

#endif  // LOOP_DETECTOR_HPP
