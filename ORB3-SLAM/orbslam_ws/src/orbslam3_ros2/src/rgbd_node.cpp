  #include <algorithm>
  #include <chrono>
  #include <cmath>
  #include <cstdint>
  #include <cstdlib>
  #include <numeric>
  #include <vector>
  #include <fstream>
  #include <sstream>
  #include <iomanip>
  #include <functional>
  #include <limits>
  #include <memory>
  #include <mutex>
  #include <string>
  #include <unordered_map>

  #include <deque>

  #include <rclcpp/rclcpp.hpp>
  #include <sensor_msgs/msg/image.hpp>
  #include <sensor_msgs/msg/imu.hpp>
  #include <std_msgs/msg/string.hpp>
  #include <geometry_msgs/msg/pose_stamped.hpp>
  #include <geometry_msgs/msg/transform_stamped.hpp>
  #include <nav_msgs/msg/odometry.hpp>
  #include <tf2_ros/transform_broadcaster.h>

  #include <message_filters/subscriber.h>
  #include <message_filters/synchronizer.h>
  #include <message_filters/sync_policies/approximate_time.h>

  #include <opencv2/core.hpp>
  #include <opencv2/core/persistence.hpp>
  #include <opencv2/imgproc.hpp>

  #include <sophus/se3.hpp>

  #include "System.h"
  #include "ImuTypes.h"

  // for core accessor
  #include <cstring>
  #include <vector>
  #include <sensor_msgs/msg/point_cloud2.hpp>
  #include <sensor_msgs/msg/point_field.hpp>

  class RgbdNode : public rclcpp::Node
  {
  public:
    RgbdNode()
    : Node("orbslam3_rgbd_node")
    {
      const char * home_env = std::getenv("HOME");
      const std::string home = home_env ? std::string(home_env) : std::string("");

      vocabulary_path_ = this->declare_parameter<std::string>(
        "vocabulary_path",
        home + "/orbslam3_ws/src/ORB_SLAM3/Vocabulary/ORBvoc.txt"
      );

      settings_path_ = this->declare_parameter<std::string>(
        "settings_path",
        home + "/orbslam3_ws/src/ORB_SLAM3/Examples/RGB-D/TUM1.yaml"
      );

      rgb_topic_ = this->declare_parameter<std::string>(
        "rgb_topic",
        "/camera/rgb/image_color"
      );

      depth_topic_ = this->declare_parameter<std::string>(
        "depth_topic",
        "/camera/depth/image"
      );

      use_imu_ = this->declare_parameter<bool>(
        "use_imu",
        false
      );

      imu_topic_ = this->declare_parameter<std::string>(
        "imu_topic",
        "/camera/camera/imu"
      );

      // When true, the System is switched to pure LOCALIZATION mode right after
      // construction: local mapping + loop closing are disabled so a previously
      // loaded Atlas (via System.LoadAtlasFromFile in the settings YAML) stays
      // FIXED and the incoming sequence only relocalizes + tracks against it.
      // This is the "estimate current pose inside a prior map" workflow.
      localization_mode_ = this->declare_parameter<bool>(
        "localization_mode",
        false
      );

      world_frame_ = this->declare_parameter<std::string>(
        "world_frame",
        "map"
      );

      camera_frame_ = this->declare_parameter<std::string>(
        "camera_frame",
        "openni_rgb_optical_frame"
      );

      sync_queue_size_ = this->declare_parameter<int>(
        "sync_queue_size",
        30
      );

      // for core accessor
      publish_map_points_ = this->declare_parameter<bool>(
        "publish_map_points",
        true
      );

      // for core accessor
      map_points_topic_ = this->declare_parameter<std::string>(
        "map_points_topic",
        "/orbslam3/map_points"
      );

      // for core accessor
      map_points_publish_period_sec_ = this->declare_parameter<double>(
        "map_points_publish_period_sec",
        1.0
      );

      save_map_points_on_shutdown_ = this->declare_parameter<bool>(
        "save_map_points_on_shutdown",
        false
      );

      map_points_output_path_ = this->declare_parameter<std::string>(
        "map_points_output_path",
        "orbslam3_map_points.pcd"
      );

      dense_map_enabled_ = this->declare_parameter<bool>(
        "dense_map_enabled",
        false
      );

      dense_map_topic_ = this->declare_parameter<std::string>(
        "dense_map_topic",
        "/orbslam3/dense_map"
      );

      dense_map_publish_period_sec_ = this->declare_parameter<double>(
        "dense_map_publish_period_sec",
        2.0
      );

      dense_map_frame_stride_ = std::max<int>(1, static_cast<int>(this->declare_parameter<int>(
        "dense_map_frame_stride",
        5
      )));

      dense_map_pixel_stride_ = std::max<int>(1, static_cast<int>(this->declare_parameter<int>(
        "dense_map_pixel_stride",
        4
      )));

      dense_map_voxel_size_ = this->declare_parameter<double>(
        "dense_map_voxel_size",
        0.03
      );

      dense_map_min_depth_m_ = this->declare_parameter<double>(
        "dense_map_min_depth_m",
        0.15
      );

      dense_map_max_depth_m_ = this->declare_parameter<double>(
        "dense_map_max_depth_m",
        6.0
      );

      dense_map_max_points_ = static_cast<size_t>(std::max<int>(0, static_cast<int>(this->declare_parameter<int>(
        "dense_map_max_points",
        2000000
      ))));

      dense_map_include_color_ = this->declare_parameter<bool>(
        "dense_map_include_color",
        true
      );

      dense_map_save_pcd_ = this->declare_parameter<bool>(
        "dense_map_save_pcd",
        true
      );

      dense_map_pcd_output_path_ = this->declare_parameter<std::string>(
        "dense_map_pcd_output_path",
        "orbslam3_dense_map.pcd"
      );

      dense_map_save_ply_ = this->declare_parameter<bool>(
        "dense_map_save_ply",
        false
      );

      dense_map_ply_output_path_ = this->declare_parameter<std::string>(
        "dense_map_ply_output_path",
        "orbslam3_dense_map.ply"
      );

      // FIX (ghosting): when true, the SAVED dense map is rebuilt at shutdown by
      // reprojecting per-frame depth with the OPTIMIZED (post loop-closure / BA)
      // camera poses instead of the live front-end poses used during runtime.
      // This removes duplicated/ghosted geometry caused by accumulating points at
      // pre-loop-closure positions that are never re-integrated. Default ON so it
      // applies to every future run (incl. newly collected data) automatically.
      dense_map_reproject_optimized_ = this->declare_parameter<bool>(
        "dense_map_reproject_optimized",
        true
      );

      // Safety cap on buffered per-frame points (only used when reprojection is
      // ON). Prevents OOM on very long sequences; if hit, buffering stops and the
      // saved map covers only the buffered portion (a warning is logged).
      dense_map_reproject_max_points_ = static_cast<size_t>(std::max<long>(0, static_cast<long>(
        this->declare_parameter<int>("dense_map_reproject_max_points", 80000000))));

      tracking_state_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/orbslam3/tracking_state",
        10
      );

      pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/orbslam3/pose",
        10
      );

      odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
        "/orbslam3/odom",
        10
      );

      // for core accessor
      map_points_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        map_points_topic_,
        rclcpp::QoS(1).reliable().transient_local()
      );

      // for core accessor
      RCLCPP_INFO(this->get_logger(), "Publishing map points : %s", map_points_topic_.c_str());

      if (dense_map_enabled_) {
        if (!loadDenseCameraSettings()) {
          RCLCPP_ERROR(this->get_logger(), "Dense map disabled because camera settings could not be loaded.");
          dense_map_enabled_ = false;
        } else if (dense_map_voxel_size_ <= 0.0) {
          RCLCPP_ERROR(this->get_logger(), "Dense map disabled because dense_map_voxel_size must be positive.");
          dense_map_enabled_ = false;
        } else {
          dense_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            dense_map_topic_,
            rclcpp::QoS(1).reliable().transient_local()
          );
          RCLCPP_INFO(this->get_logger(), "Publishing dense map : %s", dense_map_topic_.c_str());
        }
      }

      tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

      RCLCPP_INFO(this->get_logger(), "Loading ORB-SLAM3 vocabulary: %s", vocabulary_path_.c_str());
      RCLCPP_INFO(this->get_logger(), "Loading ORB-SLAM3 settings: %s", settings_path_.c_str());

      const ORB_SLAM3::System::eSensor sensor_mode =
        use_imu_ ? ORB_SLAM3::System::IMU_RGBD : ORB_SLAM3::System::RGBD;

      slam_ = std::make_unique<ORB_SLAM3::System>(
        vocabulary_path_,
        settings_path_,
        sensor_mode,
        false
      );

      if (localization_mode_) {
        // Disable local mapping + loop closing; only relocalization + tracking
        // run against the (loaded) Atlas. The prior map is not modified.
        slam_->ActivateLocalizationMode();
        RCLCPP_INFO(this->get_logger(),
          "LOCALIZATION mode ACTIVE: prior Atlas is fixed; estimating pose against it.");
      }

      if (use_imu_) {
        // IMU is published BEST_EFFORT by the RealSense driver / rosbag; match it.
        // Dedicated reentrant callback group so the high-rate IMU keeps buffering
        // (under a MultiThreadedExecutor) while TrackRGBD is busy on another thread.
        imu_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        rclcpp::SubscriptionOptions imu_opts;
        imu_opts.callback_group = imu_cb_group_;
        auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(2000)).best_effort();
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
          imu_topic_,
          imu_qos,
          std::bind(&RgbdNode::imuCallback, this, std::placeholders::_1),
          imu_opts
        );
        RCLCPP_INFO(this->get_logger(), "IMU-RGBD mode: subscribed IMU topic: %s", imu_topic_.c_str());
      } else {
        RCLCPP_INFO(this->get_logger(), "RGB-D mode (IMU disabled).");
      }

      rgb_sub_.subscribe(this, rgb_topic_);
      depth_sub_.subscribe(this, depth_topic_);

      sync_ = std::make_shared<Synchronizer>(
        SyncPolicy(sync_queue_size_),
        rgb_sub_,
        depth_sub_
      );

      sync_->registerCallback(
        std::bind(&RgbdNode::rgbdCallback, this, std::placeholders::_1, std::placeholders::_2)
      );

      RCLCPP_INFO(this->get_logger(), "Subscribed RGB topic: %s", rgb_topic_.c_str());
      RCLCPP_INFO(this->get_logger(), "Subscribed depth topic: %s", depth_topic_.c_str());
      RCLCPP_INFO(this->get_logger(), "Publishing pose: /orbslam3/pose");
      RCLCPP_INFO(this->get_logger(), "Publishing odom: /orbslam3/odom");
      RCLCPP_INFO(this->get_logger(), "Publishing TF: %s -> %s", world_frame_.c_str(), camera_frame_.c_str());
      RCLCPP_INFO(this->get_logger(), "ORB-SLAM3 RGB-D node started with viewer disabled.");
    }

    ~RgbdNode() override
    {
      if (slam_) {
        RCLCPP_INFO(this->get_logger(), "Shutting down ORB-SLAM3.");
        slam_->Shutdown();
        saveMapPointsOnShutdown();
        // Save the optimized (post loop-closure / BA) trajectory FIRST so the
        // dense-map reprojection below can read the corrected per-frame poses.
        slam_->SaveTrajectoryTUM("CameraTrajectory.txt");
        slam_->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
        slam_->SaveLoopEdges("loop_edges.txt");
        saveDenseMapOnShutdown();
      }
      if (!track_times_ms_.empty()) {
        std::vector<double> v = track_times_ms_;
        std::sort(v.begin(), v.end());
        const double sum = std::accumulate(v.begin(), v.end(), 0.0);
        const double mean = sum / v.size();
        const double median = v[v.size() / 2];
        const double p95 = v[static_cast<size_t>(v.size() * 0.95)];
        RCLCPP_INFO(this->get_logger(),
          "TIMING: frames_tracked=%zu track_ms mean=%.1f median=%.1f p95=%.1f max=%.1f min=%.1f mean_fps=%.1f",
          v.size(), mean, median, p95, v.back(), v.front(), 1000.0 / mean);
      }
    }

  private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image,
      sensor_msgs::msg::Image
    >;

    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

    void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr & imu_msg)
    {
      const double t =
        static_cast<double>(imu_msg->header.stamp.sec) +
        static_cast<double>(imu_msg->header.stamp.nanosec) * 1e-9;
      const cv::Point3f acc(
        static_cast<float>(imu_msg->linear_acceleration.x),
        static_cast<float>(imu_msg->linear_acceleration.y),
        static_cast<float>(imu_msg->linear_acceleration.z));
      const cv::Point3f gyr(
        static_cast<float>(imu_msg->angular_velocity.x),
        static_cast<float>(imu_msg->angular_velocity.y),
        static_cast<float>(imu_msg->angular_velocity.z));
      std::lock_guard<std::mutex> lock(imu_mutex_);
      imu_buf_.emplace_back(acc, gyr, t);
    }

    // Pop all buffered IMU samples with timestamp <= frame_timestamp, in order.
    std::vector<ORB_SLAM3::IMU::Point> drainImuUntil(double frame_timestamp)
    {
      std::vector<ORB_SLAM3::IMU::Point> out;
      std::lock_guard<std::mutex> lock(imu_mutex_);
      while (!imu_buf_.empty() && imu_buf_.front().t <= frame_timestamp) {
        out.push_back(imu_buf_.front());
        imu_buf_.pop_front();
      }
      return out;
    }

    void rgbdCallback(
      const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg)
    {
      cv::Mat rgb;
      cv::Mat depth;

      if (!imageToBgr(rgb_msg, rgb)) {
        RCLCPP_WARN(this->get_logger(), "Unsupported RGB encoding: %s", rgb_msg->encoding.c_str());
        return;
      }

      if (!imageToDepth(depth_msg, depth)) {
        RCLCPP_WARN(this->get_logger(), "Unsupported depth encoding: %s", depth_msg->encoding.c_str());
        return;
      }

      if (rgb.empty() || depth.empty()) {
        RCLCPP_WARN(this->get_logger(), "Received empty RGB or depth image.");
        return;
      }

      const double timestamp =
        static_cast<double>(rgb_msg->header.stamp.sec) +
        static_cast<double>(rgb_msg->header.stamp.nanosec) * 1e-9;

      std::vector<ORB_SLAM3::IMU::Point> imu_meas;
      if (use_imu_) {
        imu_meas = drainImuUntil(timestamp);
      }

      Sophus::SE3f Tcw;
      {
        std::lock_guard<std::mutex> lock(slam_mutex_);
        const auto _t0 = std::chrono::steady_clock::now();
        Tcw = slam_->TrackRGBD(rgb, depth, timestamp, imu_meas);
        track_times_ms_.push_back(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count());
      }

      std_msgs::msg::String state_msg;

      if (Tcw.matrix().isZero(0)) {
        state_msg.data = "not_tracking";
        tracking_state_pub_->publish(state_msg);
        return;
      }

      const Sophus::SE3f Twc = Tcw.inverse();
      publishPose(rgb_msg->header.stamp, Twc);
      accumulateDenseMapIfNeeded(rgb, depth, Twc, timestamp);
      publishDenseMapIfNeeded(rgb_msg->header.stamp, timestamp);
      publishMapPointsIfNeeded(rgb_msg->header.stamp, timestamp); // for core accessor

      state_msg.data = "tracking";
      tracking_state_pub_->publish(state_msg);

      frame_count_++;
      if (frame_count_ % 30 == 0) {
        RCLCPP_INFO(this->get_logger(), "Processed %zu RGB-D frames.", frame_count_);
      }
    }

    bool imageToBgr(const sensor_msgs::msg::Image::ConstSharedPtr & msg, cv::Mat & out) const
    {
      if (msg->encoding == "bgr8") {
        out = cv::Mat(
          static_cast<int>(msg->height),
          static_cast<int>(msg->width),
          CV_8UC3,
          const_cast<unsigned char *>(msg->data.data()),
          msg->step).clone();
        return true;
      }

      if (msg->encoding == "rgb8") {
        cv::Mat rgb(
          static_cast<int>(msg->height),
          static_cast<int>(msg->width),
          CV_8UC3,
          const_cast<unsigned char *>(msg->data.data()),
          msg->step);
        cv::cvtColor(rgb, out, cv::COLOR_RGB2BGR);
        return true;
      }

      if (msg->encoding == "bgra8") {
        cv::Mat bgra(
          static_cast<int>(msg->height),
          static_cast<int>(msg->width),
          CV_8UC4,
          const_cast<unsigned char *>(msg->data.data()),
          msg->step);
        cv::cvtColor(bgra, out, cv::COLOR_BGRA2BGR);
        return true;
      }

      if (msg->encoding == "rgba8") {
        cv::Mat rgba(
          static_cast<int>(msg->height),
          static_cast<int>(msg->width),
          CV_8UC4,
          const_cast<unsigned char *>(msg->data.data()),
          msg->step);
        cv::cvtColor(rgba, out, cv::COLOR_RGBA2BGR);
        return true;
      }

      return false;
    }

    bool imageToDepth(const sensor_msgs::msg::Image::ConstSharedPtr & msg, cv::Mat & out) const
    {
      if (msg->encoding == "16UC1") {
        out = cv::Mat(
          static_cast<int>(msg->height),
          static_cast<int>(msg->width),
          CV_16UC1,
          const_cast<unsigned char *>(msg->data.data()),
          msg->step).clone();
        return true;
      }

      if (msg->encoding == "32FC1") {
        out = cv::Mat(
          static_cast<int>(msg->height),
          static_cast<int>(msg->width),
          CV_32FC1,
          const_cast<unsigned char *>(msg->data.data()),
          msg->step).clone();
        return true;
      }

      return false;
    }

    void publishPose(const builtin_interfaces::msg::Time & stamp, const Sophus::SE3f & Twc)
    {
      const Eigen::Vector3f t = Twc.translation();
      const Eigen::Quaternionf q(Twc.unit_quaternion());

      // [METRIC#2 / method B] Append the LIVE (pre-loop-closure) front-end pose.
      // SaveTrajectoryTUM() at shutdown is post-optimization; this stream is the
      // raw odometry-like trajectory before global corrections.
      if (!live_traj_file_.is_open()) {
        live_traj_file_.open("CameraTrajectory_live.txt");
        live_traj_file_ << std::fixed << std::setprecision(9);
      }
      if (live_traj_file_.is_open()) {
        const double ts =
          static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
        live_traj_file_ << ts << " " << t.x() << " " << t.y() << " " << t.z() << " "
                        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
      }

      geometry_msgs::msg::PoseStamped pose_msg;
      pose_msg.header.stamp = stamp;
      pose_msg.header.frame_id = world_frame_;
      pose_msg.pose.position.x = static_cast<double>(t.x());
      pose_msg.pose.position.y = static_cast<double>(t.y());
      pose_msg.pose.position.z = static_cast<double>(t.z());
      pose_msg.pose.orientation.x = static_cast<double>(q.x());
      pose_msg.pose.orientation.y = static_cast<double>(q.y());
      pose_msg.pose.orientation.z = static_cast<double>(q.z());
      pose_msg.pose.orientation.w = static_cast<double>(q.w());
      pose_pub_->publish(pose_msg);

      nav_msgs::msg::Odometry odom_msg;
      odom_msg.header.stamp = stamp;
      odom_msg.header.frame_id = world_frame_;
      odom_msg.child_frame_id = camera_frame_;
      odom_msg.pose.pose = pose_msg.pose;
      odom_pub_->publish(odom_msg);

      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header.stamp = stamp;
      tf_msg.header.frame_id = world_frame_;
      tf_msg.child_frame_id = camera_frame_;
      tf_msg.transform.translation.x = pose_msg.pose.position.x;
      tf_msg.transform.translation.y = pose_msg.pose.position.y;
      tf_msg.transform.translation.z = pose_msg.pose.position.z;
      tf_msg.transform.rotation = pose_msg.pose.orientation;
      tf_broadcaster_->sendTransform(tf_msg);
    }

    struct DenseVoxelKey
    {
      int64_t x{0};
      int64_t y{0};
      int64_t z{0};

      bool operator==(const DenseVoxelKey & other) const
      {
        return x == other.x && y == other.y && z == other.z;
      }
    };

    struct DenseVoxelKeyHash
    {
      size_t operator()(const DenseVoxelKey & key) const
      {
        size_t seed = std::hash<int64_t>()(key.x);
        seed ^= std::hash<int64_t>()(key.y) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>()(key.z) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
      }
    };

    struct DensePoint
    {
      float x{0.0f};
      float y{0.0f};
      float z{0.0f};
      float r{255.0f};
      float g{255.0f};
      float b{255.0f};
      uint32_t count{0};
    };

    // One strided RGB-D frame's depth points in CAMERA-LOCAL coordinates, tagged
    // with the frame timestamp. Buffered during runtime and reprojected at
    // shutdown with the optimized per-frame pose (ghosting fix).
    struct DenseFrame
    {
      double timestamp{0.0};
      std::vector<float> xyz;     // 3 * N camera-local coordinates
      std::vector<uint8_t> rgb;   // 3 * N colors (r,g,b)
    };

    bool loadDenseCameraSettings()
    {
      cv::FileStorage fs(settings_path_, cv::FileStorage::READ);
      if (!fs.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open ORB-SLAM3 settings: %s", settings_path_.c_str());
        return false;
      }

      dense_camera_fx_ = readSettingDouble(fs, "Camera1.fx", readSettingDouble(fs, "Camera.fx", 0.0));
      dense_camera_fy_ = readSettingDouble(fs, "Camera1.fy", readSettingDouble(fs, "Camera.fy", 0.0));
      dense_camera_cx_ = readSettingDouble(fs, "Camera1.cx", readSettingDouble(fs, "Camera.cx", 0.0));
      dense_camera_cy_ = readSettingDouble(fs, "Camera1.cy", readSettingDouble(fs, "Camera.cy", 0.0));
      dense_depth_factor_ = readSettingDouble(fs, "RGBD.DepthMapFactor", 1000.0);

      if (dense_camera_fx_ <= 0.0 || dense_camera_fy_ <= 0.0 || dense_depth_factor_ <= 0.0) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Invalid dense camera settings fx=%f fy=%f depth_factor=%f",
          dense_camera_fx_,
          dense_camera_fy_,
          dense_depth_factor_);
        return false;
      }

      return true;
    }

    double readSettingDouble(const cv::FileStorage & fs, const std::string & key, double default_value) const
    {
      const cv::FileNode node = fs[key];
      if (node.empty()) {
        return default_value;
      }
      return static_cast<double>(node);
    }

    void accumulateDenseMapIfNeeded(
      const cv::Mat & rgb,
      const cv::Mat & depth,
      const Sophus::SE3f & Twc,
      double timestamp)
    {
      if (!dense_map_enabled_) {
        return;
      }

      const size_t dense_frame_index = dense_frame_count_++;
      if (dense_frame_index % static_cast<size_t>(dense_map_frame_stride_) != 0) {
        return;
      }

      if (rgb.rows != depth.rows || rgb.cols != depth.cols) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          5000,
          "Skipping dense map frame because RGB and depth sizes differ: rgb=%dx%d depth=%dx%d",
          rgb.cols,
          rgb.rows,
          depth.cols,
          depth.rows);
        return;
      }

      const Eigen::Matrix3f Rwc = Twc.rotationMatrix();
      const Eigen::Vector3f twc = Twc.translation();

      // Per-frame camera-local points, buffered for optimized-pose reprojection
      // at shutdown (the ghosting fix). Built only when reprojection is enabled.
      DenseFrame frame;
      const bool buffering = dense_map_reproject_optimized_;
      if (buffering) {
        frame.timestamp = timestamp;
      }

      std::lock_guard<std::mutex> lock(dense_map_mutex_);
      for (int v = 0; v < depth.rows; v += dense_map_pixel_stride_) {
        for (int u = 0; u < depth.cols; u += dense_map_pixel_stride_) {
          const float z = depthMetersAt(depth, v, u);
          if (!std::isfinite(z) || z < dense_map_min_depth_m_ || z > dense_map_max_depth_m_) {
            continue;
          }

          const float x = static_cast<float>((static_cast<double>(u) - dense_camera_cx_) * z / dense_camera_fx_);
          const float y = static_cast<float>((static_cast<double>(v) - dense_camera_cy_) * z / dense_camera_fy_);

          float r = 255.0f;
          float g = 255.0f;
          float b = 255.0f;
          readBgrColor(rgb, v, u, r, g, b);

          if (buffering) {
            frame.xyz.push_back(x);
            frame.xyz.push_back(y);
            frame.xyz.push_back(z);
            frame.rgb.push_back(static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, r))));
            frame.rgb.push_back(static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, g))));
            frame.rgb.push_back(static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, b))));
          }

          // Live (front-end pose) world voxel map -- kept for real-time publishing
          // and as the legacy fallback when reprojection is disabled.
          const Eigen::Vector3f point_world = Rwc * Eigen::Vector3f(x, y, z) + twc;
          const DenseVoxelKey key{
            static_cast<int64_t>(std::floor(static_cast<double>(point_world.x()) / dense_map_voxel_size_)),
            static_cast<int64_t>(std::floor(static_cast<double>(point_world.y()) / dense_map_voxel_size_)),
            static_cast<int64_t>(std::floor(static_cast<double>(point_world.z()) / dense_map_voxel_size_))
          };

          auto iter = dense_map_.find(key);
          if (iter == dense_map_.end()) {
            if (dense_map_max_points_ > 0 && dense_map_.size() >= dense_map_max_points_) {
              continue;
            }

            DensePoint point;
            point.x = point_world.x();
            point.y = point_world.y();
            point.z = point_world.z();
            point.count = 1;
            point.r = r;
            point.g = g;
            point.b = b;
            dense_map_.emplace(key, point);
            continue;
          }

          DensePoint & point = iter->second;
          const float count = static_cast<float>(point.count);
          const float next_count = count + 1.0f;
          point.x = (point.x * count + point_world.x()) / next_count;
          point.y = (point.y * count + point_world.y()) / next_count;
          point.z = (point.z * count + point_world.z()) / next_count;
          point.r = (point.r * count + r) / next_count;
          point.g = (point.g * count + g) / next_count;
          point.b = (point.b * count + b) / next_count;
          if (point.count < std::numeric_limits<uint32_t>::max()) {
            ++point.count;
          }
        }
      }

      if (buffering && !frame.xyz.empty()) {
        const size_t n_pts = frame.xyz.size() / 3;
        std::lock_guard<std::mutex> flock(dense_frames_mutex_);
        if (dense_map_reproject_max_points_ == 0 ||
            dense_reproject_point_count_ + n_pts <= dense_map_reproject_max_points_) {
          dense_reproject_point_count_ += n_pts;
          dense_frames_.push_back(std::move(frame));
        } else if (!dense_reproject_cap_warned_) {
          RCLCPP_WARN(
            this->get_logger(),
            "Dense-map reprojection buffer cap (%zu pts) reached; saved dense map "
            "will cover only the buffered portion.",
            dense_map_reproject_max_points_);
          dense_reproject_cap_warned_ = true;
        }
      }
    }

    float depthMetersAt(const cv::Mat & depth, int row, int col) const
    {
      if (depth.type() == CV_16UC1) {
        const uint16_t raw = depth.at<uint16_t>(row, col);
        if (raw == 0) {
          return std::numeric_limits<float>::quiet_NaN();
        }
        return static_cast<float>(static_cast<double>(raw) / dense_depth_factor_);
      }

      if (depth.type() == CV_32FC1) {
        return depth.at<float>(row, col);
      }

      return std::numeric_limits<float>::quiet_NaN();
    }

    void readBgrColor(const cv::Mat & rgb, int row, int col, float & r, float & g, float & b) const
    {
      if (!dense_map_include_color_ || rgb.type() != CV_8UC3) {
        r = 255.0f;
        g = 255.0f;
        b = 255.0f;
        return;
      }

      const cv::Vec3b color = rgb.at<cv::Vec3b>(row, col);
      b = static_cast<float>(color[0]);
      g = static_cast<float>(color[1]);
      r = static_cast<float>(color[2]);
    }

    std::vector<DensePoint> denseMapSnapshot() const
    {
      std::vector<DensePoint> points;
      std::lock_guard<std::mutex> lock(dense_map_mutex_);
      points.reserve(dense_map_.size());
      for (const auto & item : dense_map_) {
        points.push_back(item.second);
      }
      return points;
    }

    float packedRgbFloat(const DensePoint & point) const
    {
      const uint32_t r = static_cast<uint32_t>(std::max(0.0f, std::min(255.0f, point.r)));
      const uint32_t g = static_cast<uint32_t>(std::max(0.0f, std::min(255.0f, point.g)));
      const uint32_t b = static_cast<uint32_t>(std::max(0.0f, std::min(255.0f, point.b)));
      const uint32_t packed = (r << 16) | (g << 8) | b;
      float rgb_float = 0.0f;
      std::memcpy(&rgb_float, &packed, sizeof(float));
      return rgb_float;
    }

    void publishDenseMapIfNeeded(const builtin_interfaces::msg::Time & stamp, double timestamp_sec)
    {
      if (!dense_map_enabled_ || !dense_map_pub_) {
        return;
      }

      if (last_dense_map_publish_stamp_sec_ >= 0.0 &&
          timestamp_sec - last_dense_map_publish_stamp_sec_ < dense_map_publish_period_sec_) {
        return;
      }

      const std::vector<DensePoint> points = denseMapSnapshot();
      if (points.empty()) {
        return;
      }

      sensor_msgs::msg::PointCloud2 cloud_msg;
      cloud_msg.header.stamp = stamp;
      cloud_msg.header.frame_id = world_frame_;
      cloud_msg.height = 1;
      cloud_msg.width = static_cast<uint32_t>(points.size());
      cloud_msg.is_bigendian = false;
      cloud_msg.is_dense = false;
      cloud_msg.point_step = dense_map_include_color_ ? 16 : 12;
      cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

      cloud_msg.fields.resize(dense_map_include_color_ ? 4 : 3);
      cloud_msg.fields[0].name = "x";
      cloud_msg.fields[0].offset = 0;
      cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
      cloud_msg.fields[0].count = 1;

      cloud_msg.fields[1].name = "y";
      cloud_msg.fields[1].offset = 4;
      cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
      cloud_msg.fields[1].count = 1;

      cloud_msg.fields[2].name = "z";
      cloud_msg.fields[2].offset = 8;
      cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
      cloud_msg.fields[2].count = 1;

      if (dense_map_include_color_) {
        cloud_msg.fields[3].name = "rgb";
        cloud_msg.fields[3].offset = 12;
        cloud_msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
        cloud_msg.fields[3].count = 1;
      }

      cloud_msg.data.resize(cloud_msg.row_step);
      for (size_t i = 0; i < points.size(); ++i) {
        const size_t offset = i * cloud_msg.point_step;
        std::memcpy(&cloud_msg.data[offset + 0], &points[i].x, sizeof(float));
        std::memcpy(&cloud_msg.data[offset + 4], &points[i].y, sizeof(float));
        std::memcpy(&cloud_msg.data[offset + 8], &points[i].z, sizeof(float));
        if (dense_map_include_color_) {
          const float rgb = packedRgbFloat(points[i]);
          std::memcpy(&cloud_msg.data[offset + 12], &rgb, sizeof(float));
        }
      }

      dense_map_pub_->publish(cloud_msg);
      last_dense_map_publish_stamp_sec_ = timestamp_sec;
    }

    // for core accessor
    void publishMapPointsIfNeeded(const builtin_interfaces::msg::Time & stamp, double timestamp_sec)
    {
      if (!publish_map_points_) {
        return;
      }

      if (last_map_points_publish_stamp_sec_ >= 0.0 && timestamp_sec - last_map_points_publish_stamp_sec_ < map_points_publish_period_sec_) {
        return;
      }

      std::vector<Eigen::Vector3f> points;
      {
        std::lock_guard<std::mutex> lock(slam_mutex_);
        points = slam_->GetAllMapPointsWorld();
      }

      if (points.empty()) {
        return;
      }

      sensor_msgs::msg::PointCloud2 cloud_msg;
      cloud_msg.header.stamp = stamp;
      cloud_msg.header.frame_id = world_frame_;
      cloud_msg.height = 1;
      cloud_msg.width = static_cast<uint32_t>(points.size());
      cloud_msg.is_bigendian = false;
      cloud_msg.is_dense = false;
      cloud_msg.point_step = 12;
      cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

      cloud_msg.fields.resize(3);
      cloud_msg.fields[0].name = "x";
      cloud_msg.fields[0].offset = 0;
      cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
      cloud_msg.fields[0].count = 1;

      cloud_msg.fields[1].name = "y";
      cloud_msg.fields[1].offset = 4;
      cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
      cloud_msg.fields[1].count = 1;

      cloud_msg.fields[2].name = "z";
      cloud_msg.fields[2].offset = 8;
      cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
      cloud_msg.fields[2].count = 1;

      cloud_msg.data.resize(cloud_msg.row_step);

      for (size_t i = 0; i < points.size(); ++i)
      {
        const float x = points[i].x();
        const float y = points[i].y();
        const float z = points[i].z();
        const size_t offset = i * cloud_msg.point_step;

        std::memcpy(&cloud_msg.data[offset + 0], &x, sizeof(float));
        std::memcpy(&cloud_msg.data[offset + 4], &y, sizeof(float));
        std::memcpy(&cloud_msg.data[offset + 8], &z, sizeof(float));
      }

      map_points_pub_->publish(cloud_msg);
      last_map_points_publish_stamp_sec_ = timestamp_sec;
    }

    void saveMapPointsOnShutdown()
    {
      if (!save_map_points_on_shutdown_) {
        return;
      }

      std::vector<Eigen::Vector3f> points;
      {
        std::lock_guard<std::mutex> lock(slam_mutex_);
        points = slam_->GetAllMapPointsWorld();
      }

      std::ofstream out(map_points_output_path_);
      if (!out) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Failed to open map point output: %s",
          map_points_output_path_.c_str());
        return;
      }

      out << "# .PCD v0.7 - Point Cloud Data file format\n";
      out << "VERSION 0.7\n";
      out << "FIELDS x y z\n";
      out << "SIZE 4 4 4\n";
      out << "TYPE F F F\n";
      out << "COUNT 1 1 1\n";
      out << "WIDTH " << points.size() << "\n";
      out << "HEIGHT 1\n";
      out << "VIEWPOINT 0 0 0 1 0 0 0\n";
      out << "POINTS " << points.size() << "\n";
      out << "DATA ascii\n";
      for (const auto & point : points) {
        out << point.x() << " " << point.y() << " " << point.z() << "\n";
      }

      RCLCPP_INFO(
        this->get_logger(),
        "Saved %zu ORB-SLAM3 map points to %s",
        points.size(),
        map_points_output_path_.c_str());
    }

    void saveDenseMapOnShutdown()
    {
      if (!dense_map_enabled_) {
        return;
      }

      std::vector<DensePoint> points;
      if (dense_map_reproject_optimized_) {
        points = buildReprojectedDenseMap("CameraTrajectory.txt");
        if (points.empty()) {
          RCLCPP_WARN(
            this->get_logger(),
            "Optimized-pose reprojection produced no points; falling back to the "
            "live (front-end pose) dense map.");
          points = denseMapSnapshot();
        }
      } else {
        points = denseMapSnapshot();
      }

      if (dense_map_save_pcd_) {
        saveDenseMapPcd(points);
      }
      if (dense_map_save_ply_) {
        saveDenseMapPly(points);
      }
    }

    // Ghosting fix: rebuild the dense map from buffered per-frame camera-local
    // points using the OPTIMIZED per-frame poses written to `traj_path`
    // (TUM: "ts tx ty tz qx qy qz qw", camera-to-world Twc). Loop-closure / BA
    // corrections are baked into those poses, so revisited geometry lands in the
    // same voxels instead of duplicating.
    std::vector<DensePoint> buildReprojectedDenseMap(const std::string & traj_path)
    {
      // 1) parse optimized trajectory: sorted (timestamp, Twc)
      std::vector<std::pair<double, Sophus::SE3f>> traj;
      {
        std::ifstream tf(traj_path);
        if (!tf) {
          RCLCPP_ERROR(this->get_logger(),
            "Reprojection: cannot open trajectory %s", traj_path.c_str());
          return {};
        }
        std::string line;
        while (std::getline(tf, line)) {
          if (line.empty() || line[0] == '#') {
            continue;
          }
          std::istringstream ss(line);
          double t, tx, ty, tz, qx, qy, qz, qw;
          if (!(ss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
            continue;
          }
          Eigen::Quaternionf q(static_cast<float>(qw), static_cast<float>(qx),
                               static_cast<float>(qy), static_cast<float>(qz));
          q.normalize();
          Sophus::SE3f Twc(q, Eigen::Vector3f(
            static_cast<float>(tx), static_cast<float>(ty), static_cast<float>(tz)));
          traj.emplace_back(t, Twc);
        }
      }
      if (traj.empty()) {
        return {};
      }
      std::sort(traj.begin(), traj.end(),
                [](const auto & a, const auto & b) { return a.first < b.first; });
      std::vector<double> traj_ts;
      traj_ts.reserve(traj.size());
      for (const auto & e : traj) {
        traj_ts.push_back(e.first);
      }

      // 2) reproject each buffered frame with its optimized pose, voxel-average
      const double kMatchTolSec = 0.02;  // unambiguous at typical 30 Hz framerate
      std::unordered_map<DenseVoxelKey, DensePoint, DenseVoxelKeyHash> rebuilt;
      size_t matched = 0;
      size_t dropped = 0;
      std::lock_guard<std::mutex> flock(dense_frames_mutex_);
      for (const auto & frame : dense_frames_) {
        auto it = std::lower_bound(traj_ts.begin(), traj_ts.end(), frame.timestamp);
        size_t best = (it == traj_ts.end()) ? traj_ts.size() - 1 : (it - traj_ts.begin());
        if (best > 0 &&
            std::abs(traj_ts[best - 1] - frame.timestamp) < std::abs(traj_ts[best] - frame.timestamp)) {
          --best;
        }
        if (std::abs(traj_ts[best] - frame.timestamp) > kMatchTolSec) {
          ++dropped;
          continue;  // frame was lost / not localized in the optimized trajectory
        }
        ++matched;
        const Sophus::SE3f & Twc = traj[best].second;
        const Eigen::Matrix3f Rwc = Twc.rotationMatrix();
        const Eigen::Vector3f twc = Twc.translation();
        const size_t n = frame.xyz.size() / 3;
        for (size_t i = 0; i < n; ++i) {
          const Eigen::Vector3f p_local(frame.xyz[3 * i], frame.xyz[3 * i + 1], frame.xyz[3 * i + 2]);
          const Eigen::Vector3f pw = Rwc * p_local + twc;
          const DenseVoxelKey key{
            static_cast<int64_t>(std::floor(static_cast<double>(pw.x()) / dense_map_voxel_size_)),
            static_cast<int64_t>(std::floor(static_cast<double>(pw.y()) / dense_map_voxel_size_)),
            static_cast<int64_t>(std::floor(static_cast<double>(pw.z()) / dense_map_voxel_size_))
          };
          const float r = static_cast<float>(frame.rgb[3 * i]);
          const float g = static_cast<float>(frame.rgb[3 * i + 1]);
          const float b = static_cast<float>(frame.rgb[3 * i + 2]);
          auto iter = rebuilt.find(key);
          if (iter == rebuilt.end()) {
            if (dense_map_max_points_ > 0 && rebuilt.size() >= dense_map_max_points_) {
              continue;
            }
            DensePoint point;
            point.x = pw.x();
            point.y = pw.y();
            point.z = pw.z();
            point.r = r;
            point.g = g;
            point.b = b;
            point.count = 1;
            rebuilt.emplace(key, point);
            continue;
          }
          DensePoint & point = iter->second;
          const float count = static_cast<float>(point.count);
          const float next_count = count + 1.0f;
          point.x = (point.x * count + pw.x()) / next_count;
          point.y = (point.y * count + pw.y()) / next_count;
          point.z = (point.z * count + pw.z()) / next_count;
          point.r = (point.r * count + r) / next_count;
          point.g = (point.g * count + g) / next_count;
          point.b = (point.b * count + b) / next_count;
          if (point.count < std::numeric_limits<uint32_t>::max()) {
            ++point.count;
          }
        }
      }

      RCLCPP_INFO(
        this->get_logger(),
        "Dense map reprojected with optimized poses: %zu/%zu frames matched "
        "(%zu dropped), %zu voxels.",
        matched, dense_frames_.size(), dropped, rebuilt.size());

      std::vector<DensePoint> points;
      points.reserve(rebuilt.size());
      for (const auto & item : rebuilt) {
        points.push_back(item.second);
      }
      return points;
    }

    void saveDenseMapPcd(const std::vector<DensePoint> & points)
    {
      std::ofstream out(dense_map_pcd_output_path_);
      if (!out) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Failed to open dense PCD output: %s",
          dense_map_pcd_output_path_.c_str());
        return;
      }

      out << "# .PCD v0.7 - Point Cloud Data file format\n";
      out << "VERSION 0.7\n";
      if (dense_map_include_color_) {
        out << "FIELDS x y z rgb\n";
        out << "SIZE 4 4 4 4\n";
        out << "TYPE F F F F\n";
        out << "COUNT 1 1 1 1\n";
      } else {
        out << "FIELDS x y z\n";
        out << "SIZE 4 4 4\n";
        out << "TYPE F F F\n";
        out << "COUNT 1 1 1\n";
      }
      out << "WIDTH " << points.size() << "\n";
      out << "HEIGHT 1\n";
      out << "VIEWPOINT 0 0 0 1 0 0 0\n";
      out << "POINTS " << points.size() << "\n";
      out << "DATA ascii\n";
      for (const auto & point : points) {
        out << point.x << " " << point.y << " " << point.z;
        if (dense_map_include_color_) {
          out << " " << packedRgbFloat(point);
        }
        out << "\n";
      }

      RCLCPP_INFO(
        this->get_logger(),
        "Saved %zu dense map points to %s",
        points.size(),
        dense_map_pcd_output_path_.c_str());
    }

    void saveDenseMapPly(const std::vector<DensePoint> & points)
    {
      std::ofstream out(dense_map_ply_output_path_);
      if (!out) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Failed to open dense PLY output: %s",
          dense_map_ply_output_path_.c_str());
        return;
      }

      out << "ply\n";
      out << "format ascii 1.0\n";
      out << "element vertex " << points.size() << "\n";
      out << "property float x\n";
      out << "property float y\n";
      out << "property float z\n";
      out << "property uchar red\n";
      out << "property uchar green\n";
      out << "property uchar blue\n";
      out << "end_header\n";
      for (const auto & point : points) {
        const int r = static_cast<int>(std::max(0.0f, std::min(255.0f, point.r)));
        const int g = static_cast<int>(std::max(0.0f, std::min(255.0f, point.g)));
        const int b = static_cast<int>(std::max(0.0f, std::min(255.0f, point.b)));
        out << point.x << " " << point.y << " " << point.z << " " << r << " " << g << " " << b << "\n";
      }

      RCLCPP_INFO(
        this->get_logger(),
        "Saved %zu dense map points to %s",
        points.size(),
        dense_map_ply_output_path_.c_str());
    }

    std::string vocabulary_path_;
    std::string settings_path_;
    std::string rgb_topic_;
    std::string depth_topic_;
    std::string imu_topic_;
    bool use_imu_{false};
    bool localization_mode_{false};
    std::string world_frame_;
    std::string camera_frame_;
    std::string map_points_topic_; // for core accessor
    std::string map_points_output_path_;
    std::string dense_map_topic_;
    std::string dense_map_pcd_output_path_;
    std::string dense_map_ply_output_path_;

    int sync_queue_size_{30};
    size_t frame_count_{0};
    std::vector<double> track_times_ms_;  // per-frame TrackRGBD wall time
    std::ofstream live_traj_file_;        // [METRIC#2/method B] pre-loop front-end live pose

    // for core accessor
    bool publish_map_points_{true};
    bool save_map_points_on_shutdown_{false};
    double map_points_publish_period_sec_{1.0};
    double last_map_points_publish_stamp_sec_{-1.0};

    bool dense_map_enabled_{false};
    bool dense_map_include_color_{true};
    bool dense_map_save_pcd_{true};
    bool dense_map_save_ply_{false};
    int dense_map_frame_stride_{5};
    int dense_map_pixel_stride_{4};
    double dense_map_voxel_size_{0.03};
    double dense_map_min_depth_m_{0.15};
    double dense_map_max_depth_m_{6.0};
    double dense_map_publish_period_sec_{2.0};
    double last_dense_map_publish_stamp_sec_{-1.0};
    double dense_camera_fx_{0.0};
    double dense_camera_fy_{0.0};
    double dense_camera_cx_{0.0};
    double dense_camera_cy_{0.0};
    double dense_depth_factor_{1000.0};
    size_t dense_map_max_points_{2000000};
    size_t dense_frame_count_{0};

    // Ghosting fix (optimized-pose reprojection of the saved dense map).
    bool dense_map_reproject_optimized_{true};
    size_t dense_map_reproject_max_points_{80000000};
    size_t dense_reproject_point_count_{0};
    bool dense_reproject_cap_warned_{false};
    mutable std::mutex dense_frames_mutex_;
    std::vector<DenseFrame> dense_frames_;

    std::mutex slam_mutex_;
    std::unique_ptr<ORB_SLAM3::System> slam_;
    mutable std::mutex dense_map_mutex_;
    std::unordered_map<DenseVoxelKey, DensePoint, DenseVoxelKeyHash> dense_map_;

    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    std::shared_ptr<Synchronizer> sync_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::CallbackGroup::SharedPtr imu_cb_group_;
    std::mutex imu_mutex_;
    std::deque<ORB_SLAM3::IMU::Point> imu_buf_;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tracking_state_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // for core accessor
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_points_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dense_map_pub_;
  };

  int main(int argc, char ** argv)
  {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RgbdNode>();
    // MultiThreadedExecutor lets the IMU callback group buffer samples
    // concurrently with the (heavier) RGB-D tracking callback.
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
  }
