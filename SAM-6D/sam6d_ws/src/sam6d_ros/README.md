# sam6d_ros

ROS 2 package for running the local SAM-6D template rendering, image playback,
and RGB-D 6D pose inference workflow.

The package assumes the SAM-6D source tree is available under
`~/ros2_ws/sam6d_master/SAM-6D`. Relative paths in the package configuration are
resolved from `~/ros2_ws`.

## Workflows

Build the package after changing launch files, configuration, or Python entry
points:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam6d_test
colcon build --packages-select sam6d_ros --symlink-install
source install/setup.bash
```

Render templates before inference:

```bash
ros2 run sam6d_ros sam6d_rendering --config src/sam6d_ros/config/params.yaml
```

The renderer reads `cad_path`, `output_dir`, `normalize`, `colorize`, and
`base_color` from `config/params.yaml`. It writes template files under:

```text
sam6d_master/SAM-6D/Data/Example/outputs_obj5/templates
```

Publish RGB and depth images from folders:

```bash
ros2 launch sam6d_ros image_folder_publisher.launch.py
```

The publisher uses `config/publisher_params.yaml` by default. The default input
topics are:

```text
/camera/rgb/image_raw
/camera/depth/image_raw
```

Run the SAM-6D inference node:

```bash
ros2 launch sam6d_ros sam6d_inference.launch.py
```

The launch file loads `config/params.yaml` unless a different config is passed:

```bash
ros2 launch sam6d_ros sam6d_inference.launch.py \
  config_file:=/absolute/path/to/params.yaml
```

## ROS Interfaces

The inference node subscribes to synchronized RGB and depth topics configured in
`params.yaml`:

```text
/camera/rgb/image_raw
/camera/depth/image_raw
```

It also exposes a manual trigger service:

```text
/sam6d_inference/run
```

Pose and visualization outputs are published on:

```text
/sam6d_inference/poses
/sam6d_results/pose_6dof
/sam6d_inference/result_image
```

## Configuration

Primary configuration files:

```text
config/params.yaml
config/publisher_params.yaml
```

Important inference parameters:

```text
sam6d_root
cad_path
camera_json
template_dir
output_dir
segmentor_model
top_k_for_pem
max_proposals
no_vis
visualization.show_imshow
benchmark.skip_file_save
```

`sam6d_inference.launch.py` and `image_folder_publisher.launch.py` require an
active conda environment. `CONDA_PREFIX` is used automatically. For nonstandard
launch wrappers, `SAM6D_CONDA_PREFIX` can point at the conda environment
explicitly.

## Troubleshooting

If `template_dir` is missing, run `sam6d_rendering` before starting inference.

If CUDA is unavailable, the inference node logs the detected PyTorch CUDA state
and continues in CPU mode where possible.

If `cv2.imshow` fails in a headless or non-X11 environment, set
`visualization.show_imshow: false` in `params.yaml`. Result images can still be
published and written by the SAM-6D pipeline.
