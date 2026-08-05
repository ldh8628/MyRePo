#!/usr/bin/env python3
"""ROS 2 wrapper for SAM-6D RGB-D pose inference.

The node expects templates to be rendered beforehand and runs the SAM-6D
ISM/PEM pipeline on synchronized RGB and depth image topics.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import threading
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger


@dataclass(frozen=True)
class PoseDetection:
    object_id: int
    R: np.ndarray
    t_mm: np.ndarray


class Sam6dInferenceNode(Node):
    RESULT_WINDOW_NAME = "SAM-6D Inference Result"

    def __init__(self) -> None:
        super().__init__("sam6d_inference_node")
        self._imshow_available = True
        self._display_window_created = False

        self._declare_parameters()
        self._load_parameters()
        self._setup_sys_path()
        self._check_environment()
        self._setup_ros_interfaces()

        self.get_logger().info("SAM-6D 모델 로딩 시작...")
        self._load_models()
        self.get_logger().info(
            "준비 완료 — '/sam6d_inference/run' 서비스로 추론을 실행하세요."
        )

    @staticmethod
    def _workspace_root() -> Path:
        pkg_share = Path(get_package_share_directory("sam6d_ros")).resolve()
        for candidate in (pkg_share, *pkg_share.parents):
            if (candidate / "sam6d_master" / "SAM-6D").exists():
                return candidate
            if candidate.name == "install":
                return candidate.parent
        return Path.cwd()

    def _declare_parameters(self) -> None:
        workspace = self._workspace_root()
        base = os.path.join(str(workspace), "sam6d_master", "SAM-6D")
        ex   = os.path.join(base, "Data", "Example")
        out  = os.path.join(base, "Data", "Example", "outputs_obj5")

        self.declare_parameter("sam6d_root",             base)
        self.declare_parameter("cad_path",               os.path.join(ex,  "obj_000005.ply"))
        self.declare_parameter("camera_json",            os.path.join(ex,  "camera.json"))
        self.declare_parameter("template_dir",           os.path.join(out, "templates"))
        self.declare_parameter("output_dir",             out)
        self.declare_parameter("segmentor_model",        "fastsam")
        self.declare_parameter("stability_score_thresh", 0.97)
        self.declare_parameter("det_score_thresh",       0.2)
        self.declare_parameter("top_k_for_pem",          3)
        self.declare_parameter("max_proposals",          50)
        self.declare_parameter("ism_rerank.enabled",     False)
        self.declare_parameter("ism_rerank.candidates",  20)
        self.declare_parameter("ism_rerank.semantic_window", 0.04)
        self.declare_parameter("ism_rerank.min_score",   0.45)
        self.declare_parameter("ism_rerank.force_top1_for_pem", False)
        self.declare_parameter("no_vis",                 False)
        self.declare_parameter("skip_file_save",         False)

        # ── 신뢰성 개선 파라미터 (코드 기본값 = 기존 동작 유지; params.yaml 에서 활성화) ──
        # FR-1~4: confidence floor + no-object 판정
        self.declare_parameter("confidence.pose_score_min", 0.0)   # PEM pred_pose_score floor
        self.declare_parameter("confidence.ism_score_min",  0.0)   # ISM final_score floor
        # FR-5~7: temporal stabilization (EMA + outlier/jump rejection)
        self.declare_parameter("stabilize.enabled",        False)
        self.declare_parameter("stabilize.ema_alpha",      0.0)    # 직전 결과 가중치(0=평활 없음)
        self.declare_parameter("stabilize.jump_trans_mm",  0.0)    # 위치 점프 임계(0=비활성)
        self.declare_parameter("stabilize.jump_rot_deg",   0.0)    # 회전 점프 임계(0=비활성)
        self.declare_parameter("stabilize.jump_max_hold",  3)      # 연속 점프 N회 후 새 값 수용
        # FR-8: deterministic inference
        self.declare_parameter("deterministic.enabled",   False)
        self.declare_parameter("deterministic.seed",       1)
        # FR-9: debug logging
        self.declare_parameter("debug_log.enabled",        False)
        self.declare_parameter("debug_log.dir",            "")
        # Depth/Pose Consistency Secondary Gate (Phase 0). 기본 off → 기존 동작 보존.
        # score floor(confidence.*)는 건드리지 않는다. mode: off | shadow | hard
        self.declare_parameter("depth_gate.mode",          "off")
        self.declare_parameter("depth_gate.tol_mm",        40.0)
        self.declare_parameter("depth_gate.agree_min",     0.5)
        self.declare_parameter("depth_gate.vr_min",        0.3)
        self.declare_parameter("depth_gate.err_max",       100.0)
        self.declare_parameter("depth_gate.sample_points", 2000)
        self.declare_parameter("depth_gate.depth_scale",   1.0)   # depth.png 값×scale = mm (16UC1=1.0)
        self.declare_parameter("workspace.enabled",        False)
        self.declare_parameter("workspace.z_min",          100.0)
        self.declare_parameter("workspace.z_max",          1500.0)
        self.declare_parameter("workspace.xy_max",         800.0)
        self.declare_parameter("benchmark.skip_file_save", False)
        self.declare_parameter("rgb_topic",              "/camera/rgb/image_raw")
        self.declare_parameter("depth_topic",            "/camera/depth/image_raw")
        self.declare_parameter("sync_queue_size",        5)
        self.declare_parameter("sync_slop",              0.05)
        self.declare_parameter("result_image_topic",     "/sam6d_inference/result_image")
        self.declare_parameter("pose_6dof_topic",        "/sam6d_results/pose_6dof")
        self.declare_parameter("visualization.show_imshow", True)

    def _load_parameters(self) -> None:
        workspace = self._workspace_root()

        def _abs(p: str) -> str:
            """Resolve package-relative paths from the workspace root."""
            p = os.path.expanduser(p)
            return p if os.path.isabs(p) else os.path.abspath(os.path.join(str(workspace), p))

        self.sam6d_root  = _abs(self.get_parameter("sam6d_root").value)
        self.cad_path    = _abs(self.get_parameter("cad_path").value)
        self.camera_json = _abs(self.get_parameter("camera_json").value)
        self.template_dir = _abs(self.get_parameter("template_dir").value)
        self.output_dir  = _abs(self.get_parameter("output_dir").value)

        self.segmentor_model        = self.get_parameter("segmentor_model").value
        self.stability_score_thresh = float(self.get_parameter("stability_score_thresh").value)
        self.det_score_thresh       = float(self.get_parameter("det_score_thresh").value)
        self.top_k_for_pem          = int(self.get_parameter("top_k_for_pem").value)
        self.max_proposals          = int(self.get_parameter("max_proposals").value)
        self.ism_rerank_enabled     = bool(self.get_parameter("ism_rerank.enabled").value)
        self.ism_rerank_candidates  = int(self.get_parameter("ism_rerank.candidates").value)
        self.ism_rerank_semantic_window = float(
            self.get_parameter("ism_rerank.semantic_window").value
        )
        self.ism_rerank_min_score   = float(self.get_parameter("ism_rerank.min_score").value)
        self.ism_rerank_force_top1_for_pem = bool(
            self.get_parameter("ism_rerank.force_top1_for_pem").value
        )
        self.no_vis                 = bool(self.get_parameter("no_vis").value)
        self.skip_file_save         = bool(
            self.get_parameter("skip_file_save").value
            or self.get_parameter("benchmark.skip_file_save").value
        )

        # ── 신뢰성 개선 파라미터 로드 ──
        self.pose_score_min   = float(self.get_parameter("confidence.pose_score_min").value)
        self.ism_score_min    = float(self.get_parameter("confidence.ism_score_min").value)
        self.stabilize_enabled    = bool(self.get_parameter("stabilize.enabled").value)
        self.stabilize_ema_alpha  = float(self.get_parameter("stabilize.ema_alpha").value)
        self.stabilize_jump_trans = float(self.get_parameter("stabilize.jump_trans_mm").value)
        self.stabilize_jump_rot   = float(self.get_parameter("stabilize.jump_rot_deg").value)
        self.stabilize_jump_max_hold = int(self.get_parameter("stabilize.jump_max_hold").value)
        self.deterministic_enabled = bool(self.get_parameter("deterministic.enabled").value)
        self.deterministic_seed    = int(self.get_parameter("deterministic.seed").value)
        self.debug_log_enabled = bool(self.get_parameter("debug_log.enabled").value)
        self.debug_log_dir     = self.get_parameter("debug_log.dir").value or ""
        # Depth/Pose gate 파라미터
        self.depth_gate_mode   = str(self.get_parameter("depth_gate.mode").value or "off").lower()
        self.dg_tol_mm         = float(self.get_parameter("depth_gate.tol_mm").value)
        self.dg_agree_min      = float(self.get_parameter("depth_gate.agree_min").value)
        self.dg_vr_min         = float(self.get_parameter("depth_gate.vr_min").value)
        self.dg_err_max        = float(self.get_parameter("depth_gate.err_max").value)
        self.dg_sample         = int(self.get_parameter("depth_gate.sample_points").value)
        self.dg_depth_scale    = float(self.get_parameter("depth_gate.depth_scale").value)
        self.ws_enabled        = bool(self.get_parameter("workspace.enabled").value)
        self.ws_z_min          = float(self.get_parameter("workspace.z_min").value)
        self.ws_z_max          = float(self.get_parameter("workspace.z_max").value)
        self.ws_xy_max         = float(self.get_parameter("workspace.xy_max").value)
        self._depth_np = None          # 최신 depth 이미지(mm 가정) — gate 접근용
        self._cam_K = None             # cam_K (3x3) 캐시
        self._depth_scale_logged = False
        self._prev_pose: dict[int, dict] = {}  # object_id -> {"raw":(R,t), "stab":(R,t)} (delta 계산)
        # 안정화 상태: object_id -> {"t": np(3,), "R": np(3,3), "hold": int}
        self._stab_state: dict[int, dict] = {}
        self._frame_counter = 0
        self._debug_csv_path = None  # 최초 기록 시 헤더와 함께 생성
        self.rgb_topic              = self.get_parameter("rgb_topic").value
        self.depth_topic            = self.get_parameter("depth_topic").value
        self.sync_queue_size        = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop              = float(self.get_parameter("sync_slop").value)
        self.result_image_topic     = self.get_parameter("result_image_topic").value
        self.pose_6dof_topic        = self.get_parameter("pose_6dof_topic").value
        self.show_imshow            = bool(
            self.get_parameter("visualization.show_imshow").value
        )

        self.live_input_dir = os.path.join(self.output_dir, "live_input")
        self.live_rgb_path = os.path.join(self.live_input_dir, "rgb.png")
        self.live_depth_path = os.path.join(self.live_input_dir, "depth.png")
        self.live_camera_path = os.path.join(self.live_input_dir, "camera.json")
        self.rgb_path = self.live_rgb_path
        self.depth_path = self.live_depth_path
        self.camera_for_inference_path = self.camera_json
        self._warned_scene_camera_json = False

        for label, value in [
            ("sam6d_root  ", self.sam6d_root),
            ("cad_path    ", self.cad_path),
            ("camera_json ", self.camera_json),
            ("template_dir", self.template_dir),
            ("output_dir  ", self.output_dir),
        ]:
            self.get_logger().info(f"  {label}: {value}")
        self.get_logger().info(
            "  ism_rerank : "
            f"enabled={self.ism_rerank_enabled}, "
            f"candidates={self.ism_rerank_candidates}, "
            f"semantic_window={self.ism_rerank_semantic_window}, "
            f"min_score={self.ism_rerank_min_score}, "
            f"force_top1_for_pem={self.ism_rerank_force_top1_for_pem}"
        )
        self.get_logger().info(
            "  reliability: "
            f"pose_score_min={self.pose_score_min}, ism_score_min={self.ism_score_min}, "
            f"stabilize={self.stabilize_enabled}(alpha={self.stabilize_ema_alpha}, "
            f"jump={self.stabilize_jump_trans}mm/{self.stabilize_jump_rot}deg), "
            f"deterministic={self.deterministic_enabled}(seed={self.deterministic_seed}), "
            f"debug_log={self.debug_log_enabled}"
        )

    def _setup_sys_path(self) -> None:
        """Add the configured SAM-6D checkout and conda packages to sys.path."""
        candidates = []
        requested_conda = os.environ.get("SAM6D_CONDA_PREFIX")
        if requested_conda:
            candidates.append(os.path.expanduser(requested_conda))

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidates.append(os.path.expanduser(conda_prefix))

        candidates.append(sys.prefix)

        legacy_override = os.environ.get("SAM6D_VENV")
        if legacy_override:
            candidates.append(os.path.expanduser(legacy_override))

        for candidate in candidates:
            if os.path.isdir(os.path.join(candidate, "conda-meta")):
                conda_dir = candidate
                break
        else:
            raise RuntimeError(
                "conda environment not detected. Activate sam6d_test before "
                "launching sam6d_ros."
            )

        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        conda_site = os.path.join(conda_dir, "lib", py_ver, "site-packages")
        if os.path.isdir(conda_site):
            if conda_site not in sys.path:
                sys.path.insert(0, conda_site)
            self.get_logger().info(f"  conda site-packages: {conda_site}")
        else:
            self.get_logger().warn(
                f"conda site-packages 없음: {conda_site}\n"
                "  torch/trimesh/gorilla 등이 다른 경로에 있어야 합니다."
            )

        ism_dir = os.path.join(self.sam6d_root, "Instance_Segmentation_Model")
        pem_dir = os.path.join(self.sam6d_root, "Pose_Estimation_Model")

        candidates = [
            self.sam6d_root,
            ism_dir,
            os.path.join(pem_dir, "provider"),
            os.path.join(pem_dir, "utils"),
            os.path.join(pem_dir, "model"),
            os.path.join(pem_dir, "model", "pointnet2"),
        ]
        for p in reversed(candidates):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        self.get_logger().debug(f"sys.path 갱신 완료 (기준: {self.sam6d_root})")

    def _check_environment(self) -> None:
        """Validate runtime dependencies and configured input paths."""
        import importlib.util

        required_pkgs = [
            "torch",
            "cv2",
            "numpy",
            "trimesh",
            "imageio",
            "gorilla",
            "distinctipy",
        ]
        missing = [p for p in required_pkgs if importlib.util.find_spec(p) is None]
        if missing:
            msg = f"필수 패키지 미설치: {missing}"
            self.get_logger().error(msg)
            raise RuntimeError(msg)

        import torch
        cuda_ok = torch.cuda.is_available()
        self.get_logger().info(f"[GPU] torch.cuda.is_available() = {cuda_ok}")
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            self.get_logger().info(
                f"[GPU] 이름: {torch.cuda.get_device_name(0)} | "
                f"VRAM: {props.total_memory // 1024**2} MB | "
                f"CUDA Runtime: {torch.version.cuda} | "
                f"torch: {torch.__version__}"
            )
            if self.skip_file_save:
                self.get_logger().info(
                    "[PERF] skip_file_save=True — 입력 파일 쓰기 및 시각화 생략 모드 활성화"
                )
        else:
            self.get_logger().warn(
                "[GPU] CUDA 불가 — CPU 모드로 실행됩니다. "
                "우분투에서 CPU 추론 중일 수 있습니다! "
                "nvidia-smi 및 CUDA 드라이버를 확인하세요."
            )

        for label, path in [
            ("cad_path",    self.cad_path),
            ("camera_json", self.camera_json),
        ]:
            if not os.path.isfile(path):
                self.get_logger().warn(f"파일 없음 [{label}]: {path}")

        self.camera_for_inference_path = self._prepare_camera_json_for_inference()

        if not os.path.isdir(self.template_dir):
            self.get_logger().warn(
                f"template_dir 없음: {self.template_dir}\n"
                "  → rendering.py 를 먼저 실행하여 템플릿을 생성하세요."
            )
        else:
            n_rgb = len([
                f for f in os.listdir(self.template_dir) if f.startswith("rgb_")
            ])
            self.get_logger().info(f"template_dir: {n_rgb}개 뷰 확인")

    def _setup_ros_interfaces(self) -> None:
        self._lock = threading.Lock()
        self._display_lock = threading.Lock()
        self._pending_vis_path = ""
        self._displayed_vis_key = None
        self._last_visualization_log_key = None
        self.bridge = CvBridge()
        self.pose_pub = self.create_publisher(PoseArray, "/sam6d_inference/poses", 10)
        self.pose_6dof_pub = self.create_publisher(
            TransformStamped, self.pose_6dof_topic, 10
        )
        self.result_image_pub = self.create_publisher(Image, self.result_image_topic, 10)
        self.create_service(Trigger, "/sam6d_inference/run", self._trigger_cb)
        self._input_image_index = 0
        self._active_output_stem = None

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self._synced_image_cb)
        self.visualization_timer = None
        if self.show_imshow:
            self.visualization_timer = self.create_timer(0.03, self._visualization_timer_cb)
        self.get_logger().info(
            f"RGB/Depth 구독 대기: {self.rgb_topic}, {self.depth_topic}"
        )

    def _load_models(self) -> None:
        import torch

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.output_dir, exist_ok=True)
        self._install_pynvml_stub_if_unavailable()

        # FR-8: GPU 결정성 (1회 설정). 정지 장면 jitter 저감.
        if self.deterministic_enabled:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.get_logger().info(
                "  결정성 모드: cudnn.deterministic=True, benchmark=False"
            )

        # Some SAM-6D loaders resolve Hydra config paths from the current directory.
        orig_cwd = os.getcwd()
        os.chdir(self.sam6d_root)
        try:
            from run_batch_inference_fast import (
                load_ism_model,
                load_ism_templates,
                load_pem_model_and_templates,
                preload_pem_mesh,
            )

            self.get_logger().info("  [1/4] ISM (FastSAM + DINOv2) 로딩...")
            self.ism_model = load_ism_model(
                self.segmentor_model,
                self.stability_score_thresh,
                self.device,
            )

            self.get_logger().info("  [2/4] ISM 템플릿 feature 추출...")
            self.ism_model = load_ism_templates(
                self.ism_model, self.template_dir, self.cad_path, self.device
            )

            self.get_logger().info("  [3/4] PEM 모델 + 템플릿 로딩...")
            (
                self.pem_model,
                self.all_tem_pts,
                self.all_tem_feat,
                self.pem_cfg,
            ) = load_pem_model_and_templates(self.template_dir, self.device)

            self.get_logger().info("  [4/4] CAD 메시 포인트클라우드 샘플링...")
            self.model_points, self.radius = preload_pem_mesh(
                self.cad_path, self.pem_cfg.test_dataset.n_sample_model_point
            )
        finally:
            os.chdir(orig_cwd)

    def _install_pynvml_stub_if_unavailable(self) -> None:
        """Allow gorilla-core to import on CPU-only systems without NVML."""
        try:
            import pynvml

            pynvml.nvmlInit()
            return
        except Exception as exc:
            self.get_logger().warn(
                f"NVML 사용 불가 — CPU 환경용 pynvml stub 적용: {exc}"
            )

        stub = types.ModuleType("pynvml")

        def _raise_unavailable(*args, **kwargs):
            raise RuntimeError("NVML is unavailable in this environment")

        stub.NVMLError = RuntimeError
        stub.NVMLError_LibraryNotFound = RuntimeError
        stub.nvmlInit = lambda *args, **kwargs: None
        stub.nvmlShutdown = lambda *args, **kwargs: None
        stub.nvmlDeviceGetCount = lambda *args, **kwargs: 0
        stub.nvmlDeviceGetHandleByIndex = _raise_unavailable
        stub.nvmlDeviceGetMemoryInfo = _raise_unavailable
        sys.modules["pynvml"] = stub

        gpustat_stub = types.ModuleType("gpustat")

        class _GPUStatCollection:
            @staticmethod
            def new_query():
                return []

        gpustat_stub.GPUStatCollection = _GPUStatCollection
        sys.modules["gpustat"] = gpustat_stub

    def _trigger_cb(self, request, response):
        if not self._lock.acquire(blocking=False):
            response.success = False
            response.message = "이전 추론이 아직 실행 중입니다."
            return response
        start_time = time.perf_counter()
        try:
            poses, timing = self._run_inference()
            end_time = self._publish_poses(poses)
            self._log_processing_summary(self.rgb_path, start_time, end_time, timing)
            self._publish_result_image()
            response.success = True
            response.message = f"{len(poses)}개 포즈 추정 완료"
        except Exception as exc:
            self.get_logger().error(f"추론 오류: {exc}")
            response.success = False
            response.message = str(exc)
        finally:
            self._lock.release()
        return response

    def _synced_image_cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        image_label = f"{self._input_image_index:06d}.png"
        self._input_image_index += 1

        if not self._lock.acquire(blocking=False):
            self.get_logger().debug("추론 실행 중이라 수신 프레임을 건너뜁니다.")
            return
        start_time = time.perf_counter()

        try:
            load_t0 = time.perf_counter()
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            input_load_time = time.perf_counter() - load_t0
        except Exception as exc:
            self._lock.release()
            self.get_logger().error(f"이미지 변환 오류: {exc}")
            return

        worker = threading.Thread(
            target=self._run_inference_from_images,
            args=(
                rgb.copy(),
                np.asarray(depth).copy(),
                start_time,
                input_load_time,
                image_label,
            ),
            daemon=True,
        )
        worker.start()

    def _run_inference_from_images(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        start_time: float,
        input_load_time: float,
        image_label: str,
    ) -> None:
        original_rgb_path = self.rgb_path
        original_depth_path = self.depth_path
        file_save_time = 0.0
        try:
            os.makedirs(self.live_input_dir, exist_ok=True)

            if self.skip_file_save:
                if not (
                    os.path.exists(self.live_rgb_path)
                    and os.path.exists(self.live_depth_path)
                ):
                    raise RuntimeError(
                        "benchmark.skip_file_save=True 이지만 live_input/rgb.png 또는 "
                        "depth.png가 없습니다. 먼저 skip_file_save=False로 한 프레임을 "
                        "실행해 기준 입력 파일을 생성하세요."
                    )
                self.get_logger().debug(
                    "skip_file_save=True: 입력 파일 쓰기 생략, 이전 프레임 재사용"
                )
            else:
                t0_write = time.perf_counter()
                cv2.imwrite(self.live_rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(self.live_depth_path, depth)
                file_save_time = time.perf_counter() - t0_write

            self.rgb_path = self.live_rgb_path
            self.depth_path = self.live_depth_path
            self._active_output_stem = os.path.splitext(image_label)[0]
            self._depth_np = depth   # depth/pose gate 가 접근할 최신 depth (passthrough, mm 가정)

            poses, timing = self._run_inference()
            timing["load"] = round(timing.get("load", 0.0) + input_load_time, 3)
            end_time = self._publish_poses(poses)
            self._log_processing_summary(
                image_label, start_time, end_time, timing, file_save_time
            )
            self._publish_result_image()
        except Exception as exc:
            self.get_logger().error(f"자동 추론 오류: {exc}")
        finally:
            self.rgb_path = original_rgb_path
            self.depth_path = original_depth_path
            self._active_output_stem = None
            self._lock.release()

    def _prepare_camera_json_for_inference(self) -> str:
        if not os.path.isfile(self.camera_json):
            return self.camera_json

        with open(self.camera_json, "r") as f:
            camera_data = json.load(f)

        if "cam_K" in camera_data:
            return self.camera_json

        frame_entries = [
            (key, value)
            for key, value in camera_data.items()
            if isinstance(value, dict) and "cam_K" in value
        ]
        if not frame_entries:
            return self.camera_json

        frame_entries.sort(key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))
        frame_id, camera_info = frame_entries[0]
        os.makedirs(self.live_input_dir, exist_ok=True)
        with open(self.live_camera_path, "w") as f:
            json.dump(
                {
                    "cam_K": camera_info["cam_K"],
                    "depth_scale": camera_info.get("depth_scale", 1.0),
                },
                f,
            )

        if not self._warned_scene_camera_json:
            self.get_logger().warn(
                "camera_json이 frame_id별 scene_camera 형식입니다. "
                f"현재 추론 API는 단일 camera.json을 요구하므로 frame {frame_id}의 cam_K를 "
                f"{self.live_camera_path}로 변환해 사용합니다."
            )
            self._warned_scene_camera_json = True

        return self.live_camera_path

    def _visualize_ism_rerank_rank0(self, rb, rgb_path: str, detections: list[dict], save_path: str) -> None:
        if not detections:
            return
        image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return
        overlay = image_bgr.copy()
        mask = rb.rle_to_mask(detections[0]["segmentation"]).astype(bool)
        color = np.array([0, 213, 255], dtype=np.uint8)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
        x, y, w, h = [int(round(v)) for v in detections[0]["bbox"]]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 213, 255), 2)
        label = f"rerank #0 score={float(detections[0].get('score', 0.0)):.3f}"
        text_org = (max(0, x), max(18, y - 8))
        cv2.putText(overlay, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 213, 255), 1, cv2.LINE_AA)
        combined = np.concatenate([image_bgr, overlay], axis=1)
        cv2.imwrite(save_path, combined)

    def _run_ism_single_area_rerank(
        self,
        rb,
        rgb_path: str,
        depth_path: str,
        cam_path: str,
        img_out: str,
        top_k_for_pem: int,
        max_proposals: int,
        no_vis: bool,
    ) -> dict:
        import torch

        timing = {}

        def tick(name: str, t0: float) -> float:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing[name] = round(time.time() - t0, 3)
            return time.time()

        try:
            t = time.time()
            rgb = rb.Image.open(rgb_path).convert("RGB")
            rgb_np = np.array(rgb)
            t = tick("image_load", t)

            raw = self.ism_model.segmentor_model.generate_masks(rgb_np)
            n_raw = len(raw["masks"]) if isinstance(raw, dict) else len(raw)
            detections = rb.Detections(raw)
            if max_proposals > 0 and len(detections) > max_proposals:
                areas = (detections.boxes[:, 2] - detections.boxes[:, 0]) * (
                    detections.boxes[:, 3] - detections.boxes[:, 1]
                )
                _, top_idx = torch.topk(areas, max_proposals)
                detections.filter(top_idx)
            n_kept = len(detections)
            timing["proposals"] = f"{n_raw}->{n_kept}"
            t = tick("mask_generation", t)

            query_desc, query_appe_desc = self.ism_model.descriptor_model.forward(rgb_np, detections)
            t = tick("dinov2_feature", t)

            idx_selected, pred_idx_objects, semantic_score, best_template = (
                self.ism_model.compute_semantic_score(query_desc)
            )
            t = tick("semantic_score", t)

            detections.filter(idx_selected)
            query_appe_desc = query_appe_desc[idx_selected, :]

            appe_scores, ref_aux_desc = self.ism_model.compute_appearance_score(
                best_template, pred_idx_objects, query_appe_desc
            )
            t = tick("appearance_score", t)

            batch = rb._batch_input_data(depth_path, cam_path, self.device)
            image_uv = self.ism_model.project_template_to_image(
                best_template, pred_idx_objects, batch, detections.masks
            )
            geometric_score, visible_ratio = self.ism_model.compute_geometric_score(
                image_uv,
                detections,
                query_appe_desc,
                ref_aux_desc,
                visible_thred=self.ism_model.visible_thred,
            )
            t = tick("geometric_score", t)

            final_score = (
                semantic_score + appe_scores + geometric_score * visible_ratio
            ) / (1 + 1 + visible_ratio)
            detections.add_attribute("scores", final_score)
            detections.add_attribute("object_ids", torch.zeros_like(final_score))

            sem_np = semantic_score.detach().cpu().numpy()
            app_np = appe_scores.detach().cpu().numpy()
            geo_np = geometric_score.detach().cpu().numpy()
            vis_np = visible_ratio.detach().cpu().numpy()
            fin_np = final_score.detach().cpu().numpy()

            detections.to_numpy()
            masks_np = detections.masks
            det_all = rb.detections_to_json_direct(detections, top_k=0)
            rows = []
            for i, det in enumerate(det_all):
                rows.append(
                    {
                        "source_index": i,
                        "det": det,
                        "mask_area": int(rb.force_binary_mask(masks_np[i]).sum()),
                        "semantic": float(sem_np[i]),
                        "appearance": float(app_np[i]),
                        "geometric": float(geo_np[i]),
                        "visible_ratio": float(vis_np[i]),
                        "original_final": float(fin_np[i]),
                    }
                )

            rows_sorted = sorted(
                rows, key=lambda row: row["original_final"], reverse=True
            )[: self.ism_rerank_candidates]
            max_semantic = max((row["semantic"] for row in rows_sorted), default=0.0)
            rerank_rows = []
            for original_rank, row in enumerate(rows_sorted):
                eligible = (
                    row["semantic"] >= max_semantic - self.ism_rerank_semantic_window
                    and row["original_final"] >= self.ism_rerank_min_score
                )
                rerank_score = (
                    10.0 + row["semantic"] + row["mask_area"] / 100000.0
                    if eligible
                    else row["original_final"]
                )
                row = dict(row)
                row["original_rank"] = original_rank
                row["eligible"] = bool(eligible)
                row["rerank_score"] = float(rerank_score)
                rerank_rows.append(row)
            rerank_rows = sorted(rerank_rows, key=lambda row: row["rerank_score"], reverse=True)

            effective_top_k = 1 if self.ism_rerank_force_top1_for_pem else top_k_for_pem
            det_list = []
            meta = []
            for new_rank, row in enumerate(rerank_rows[:effective_top_k]):
                det = dict(row["det"])
                det["original_rank"] = int(row["original_rank"])
                det["source_index"] = int(row["source_index"])
                det["rerank_score"] = float(row["rerank_score"])
                det_list.append(det)
                meta.append(
                    {
                        "new_rank": new_rank,
                        "original_rank": int(row["original_rank"]),
                        "source_index": int(row["source_index"]),
                        "bbox_xywh": [float(x) for x in det["bbox"]],
                        "mask_area": int(row["mask_area"]),
                        "semantic": float(row["semantic"]),
                        "appearance": float(row["appearance"]),
                        "geometric": float(row["geometric"]),
                        "visible_ratio": float(row["visible_ratio"]),
                        "original_final": float(row["original_final"]),
                        "rerank_score": float(row["rerank_score"]),
                        "eligible": bool(row["eligible"]),
                    }
                )

            rb.save_json_bop23(os.path.join(img_out, "detection_ism.json"), det_list)
            with open(os.path.join(img_out, "ism_rerank_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "rule": (
                            "semantic >= max_semantic - semantic_window and "
                            "original_final >= min_score, then prefer larger mask_area"
                        ),
                        "max_semantic": max_semantic,
                        "semantic_window": self.ism_rerank_semantic_window,
                        "min_score": self.ism_rerank_min_score,
                        "candidates": self.ism_rerank_candidates,
                        "force_top1_for_pem": self.ism_rerank_force_top1_for_pem,
                        "effective_top_k_for_pem": effective_top_k,
                        "meta": meta,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            if not no_vis:
                self._visualize_ism_rerank_rank0(
                    rb, rgb_path, det_list, os.path.join(img_out, "vis_ism.png")
                )
            tick("save_json_visualization", t)

            return {
                "ok": True,
                "timing": timing,
                "scores": [
                    {
                        "final": round(float(det["score"]), 4),
                        "rerank": round(float(det["rerank_score"]), 4),
                    }
                    for det in det_list
                ],
                "rerank": {
                    "enabled": True,
                    "force_top1_for_pem": self.ism_rerank_force_top1_for_pem,
                    "effective_top_k_for_pem": effective_top_k,
                    "max_semantic": max_semantic,
                    "top": meta[: min(5, len(meta))],
                },
            }
        except Exception as exc:
            return {
                "ok": False,
                "timing": timing,
                "scores": [],
                "error": str(exc),
                "rerank": {"enabled": True},
            }

    # ──────────────────────────────────────────────────────────────────────
    # 신뢰성 개선 헬퍼 (FR-1~9)
    # ──────────────────────────────────────────────────────────────────────
    def _apply_determinism(self) -> None:
        """FR-8: 프레임 단위로 전역 RNG 를 고정한다."""
        if not self.deterministic_enabled:
            return
        seed = self.deterministic_seed
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception as exc:  # torch 미가용 환경 방어
            self.get_logger().warn(f"torch seed 적용 실패: {exc}")

    @staticmethod
    def _ism_breakdown_lookup(ism_scores: list) -> dict:
        """ISM final score(반올림) -> 분해 점수 dict. pem_poses 의 ism_score 와 조인용."""
        table: dict[float, dict] = {}
        for s in ism_scores or []:
            key = round(float(s.get("final", 0.0)), 4)
            table.setdefault(key, s)
        return table

    def _filter_and_decide(self, pem_poses, ism_scores, best_idx):
        """FR-1~4: confidence floor 적용 + no-object 판정.

        반환: (poses, debug_records, decision)
        decision ∈ {PUBLISH, NO_OBJECT, NO_DETECTION}
        """
        breakdown = self._ism_breakdown_lookup(ism_scores)
        poses: list[PoseDetection] = []
        records = []
        # 단일 인스턴스 가정(PRD #3): stabilize 시 object_id 별 최고 final_score 1개만
        # 안정화·발행한다. (top_k_for_pem>1 로 같은 object_id 가 여러 PASS 후보로 나오면
        # _stabilize_poses 의 EMA 상태가 object_id 키로 충돌해 오염되므로.)
        best_pose_by_obj: dict = {}
        for idx, p in enumerate(pem_poses):
            ism_s   = float(p.get("ism_score", 0.0))
            pem_raw = p.get("pem_score", None)
            pem_s   = float(pem_raw) if pem_raw is not None else None
            final_s = float(p.get("final_score", 0.0))
            has_R   = "R" in p

            bd = breakdown.get(round(ism_s, 4), {})
            similarity = bd.get("similarity")
            appearance = bd.get("appearance")   # mask 영역 patch 외관 → mask_score 대표
            geometric  = bd.get("geometric")    # bbox IoU 기반 → bbox_score 대표
            # [instrumentation] template 선택 메타 (계측 전용). bd 미존재 시 None/[] → null 기록.
            best_template_id     = bd.get("best_template_id")
            best_template_score  = bd.get("best_template_score")
            top5_template_ids    = bd.get("top5_template_ids")
            top5_template_scores = bd.get("top5_template_scores")

            pass_ism = (self.ism_score_min <= 0.0) or (ism_s >= self.ism_score_min)
            pass_pem = (self.pose_score_min <= 0.0) or (pem_s is None) or (pem_s >= self.pose_score_min)
            passed   = has_R and pass_ism and pass_pem

            object_id = int(p.get("obj_id", p.get("object_id", p.get("category_id", idx + 1))))
            tx, ty, tz = (p.get("t_mm") or [0.0, 0.0, 0.0])
            records.append({
                "candidate":          idx,
                "object_id":          object_id,
                "similarity_score":   similarity,
                "appearance_score":   appearance,
                "geometric_score":    geometric,
                "ism_score":          round(ism_s, 4),
                "pose_score":         round(pem_s, 4) if pem_s is not None else None,
                "final_score":        round(final_s, 4),
                "is_best":            (idx == best_idx),
                "candidate_decision": "PASS" if passed else "FILTERED",
                "t_mm":               [round(float(tx), 1), round(float(ty), 1), round(float(tz), 1)],
                "best_template_id":     best_template_id,
                "best_template_score":  best_template_score,
                "top5_template_ids":    top5_template_ids,
                "top5_template_scores": top5_template_scores,
            })

            if passed:
                R = np.array(p["R"], dtype=np.float64)
                t = np.array(p["t_mm"], dtype=np.float64)
                pd = PoseDetection(object_id=object_id, R=R, t_mm=t)
                if self.stabilize_enabled:
                    # object_id 별 최고 final_score 1개만 유지 (단일 인스턴스)
                    cur = best_pose_by_obj.get(object_id)
                    if cur is None or final_s > cur[0]:
                        best_pose_by_obj[object_id] = (final_s, pd)
                else:
                    poses.append(pd)   # stabilize off → 기존 다중 pose 동작 보존(NFR-3)

        if self.stabilize_enabled:
            poses = [pd for _, pd in best_pose_by_obj.values()]

        if not pem_poses:
            decision = "NO_DETECTION"
        elif not poses:
            decision = "NO_OBJECT"
        else:
            decision = "PUBLISH"
        return poses, records, decision

    def _stabilize_poses(self, poses):
        """FR-5~7: object_id 별 EMA 평활 + jump/outlier rejection."""
        from scipy.spatial.transform import Rotation, Slerp

        alpha = self.stabilize_ema_alpha          # 직전 결과 가중치 (0~1)
        out = []
        for pose in poses:
            key  = pose.object_id
            prev = self._stab_state.get(key)
            R_new, t_new = pose.R, pose.t_mm

            if prev is None:
                self._stab_state[key] = {"t": t_new.copy(), "R": R_new.copy(), "hold": 0}
                out.append(PoseDetection(object_id=key, R=R_new, t_mm=t_new))
                continue

            t_prev, R_prev = prev["t"], prev["R"]
            d_trans = float(np.linalg.norm(t_new - t_prev))
            r_rel   = Rotation.from_matrix(R_prev).inv() * Rotation.from_matrix(R_new)
            d_rot   = float(np.degrees(r_rel.magnitude()))

            is_jump = (
                (self.stabilize_jump_trans > 0.0 and d_trans > self.stabilize_jump_trans) or
                (self.stabilize_jump_rot   > 0.0 and d_rot   > self.stabilize_jump_rot)
            )

            if is_jump and prev["hold"] < self.stabilize_jump_max_hold:
                # outlier: 직전 안정값 유지(hold). 연속 hold 가 한계 넘으면 새 값 수용.
                prev["hold"] += 1
                R_s, t_s = R_prev, t_prev
                self.get_logger().debug(
                    f"[stabilize] obj {key} jump 거부 "
                    f"(Δt={d_trans:.1f}mm, Δr={d_rot:.1f}°, hold={prev['hold']})"
                )
            else:
                a = 0.0 if is_jump else alpha   # hold 한계 초과 → 새 값 즉시 수용
                t_s = a * t_prev + (1.0 - a) * t_new
                key_rots = Rotation.from_matrix(np.stack([R_prev, R_new]))
                slerp = Slerp([0.0, 1.0], key_rots)
                R_s = slerp([1.0 - a])[0].as_matrix()
                self._stab_state[key] = {"t": np.array(t_s).copy(),
                                         "R": np.array(R_s).copy(), "hold": 0}

            out.append(PoseDetection(object_id=key,
                                     R=np.array(R_s, dtype=np.float64),
                                     t_mm=np.array(t_s, dtype=np.float64)))
        return out

    def _load_cam_K(self):
        """cam_K(3x3) 로드·캐시. camera_for_inference_path JSON 의 cam_K 사용."""
        if self._cam_K is not None:
            return self._cam_K
        try:
            with open(self.camera_for_inference_path, "r") as f:
                cd = json.load(f)
            K = cd.get("cam_K")
            if K is None:
                for v in cd.values():
                    if isinstance(v, dict) and "cam_K" in v:
                        K = v["cam_K"]; break
            if K is not None:
                self._cam_K = np.array(K, dtype=np.float64).reshape(3, 3)
        except Exception:
            self._cam_K = None
        return self._cam_K

    def _depth_pose_gate(self, R, t_mm):
        """Depth/Pose Consistency Secondary Gate (Phase 0).
        model_points 를 (R,t) 로 투영해 projected depth 와 sensor depth(mm) 를 비교한다.
        반환: metric + workspace/depth 판정 + 사유 dict. score floor 는 건드리지 않음.
        """
        res = {"depth_valid_ratio": None, "depth_error_median": None, "depth_error_p90": None,
               "depth_agreement_ratio": None, "workspace_gate_pass": True,
               "workspace_gate_reason": "OK", "depth_gate_pass": True, "depth_gate_reason": "OK"}
        t = np.asarray(t_mm, dtype=np.float64).reshape(3)
        Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)

        # 1) workspace plausibility
        if self.ws_enabled:
            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
            if not (self.ws_z_min < tz < self.ws_z_max):
                res["workspace_gate_pass"] = False
                res["workspace_gate_reason"] = f"WORKSPACE_Z_OUT({tz:.0f})"
            elif abs(tx) > self.ws_xy_max or abs(ty) > self.ws_xy_max:
                res["workspace_gate_pass"] = False
                res["workspace_gate_reason"] = f"WORKSPACE_XY_OUT({tx:.0f},{ty:.0f})"

        # 2) depth consistency
        depth = self._depth_np
        K = self._load_cam_K()
        mp = self.model_points
        if depth is None:
            res["depth_gate_reason"] = "NO_DEPTH"
        elif K is None:
            res["depth_gate_reason"] = "NO_DEPTH"
        elif mp is None or len(mp) == 0:
            res["depth_gate_reason"] = "NO_MODEL_POINTS"
        else:
            mp = np.asarray(mp, dtype=np.float64).reshape(-1, 3)
            if self.dg_sample > 0 and len(mp) > self.dg_sample:
                idx = np.linspace(0, len(mp) - 1, self.dg_sample).astype(int)
                mp = mp[idx]
            Pc = (Rm @ mp.T).T + t                      # (N,3) mm, 카메라 좌표
            z = Pc[:, 2]
            front = z > 1e-3
            Pc, z = Pc[front], z[front]
            if len(Pc) == 0:
                res["depth_gate_reason"] = "PROJECTION_EMPTY"
            else:
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                u = fx * Pc[:, 0] / z + cx
                v = fy * Pc[:, 1] / z + cy
                H, W = depth.shape[:2]
                inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                if int(inb.sum()) == 0:
                    res["depth_gate_reason"] = "PROJECTION_EMPTY"
                else:
                    ui = u[inb].astype(int); vi = v[inb].astype(int); zp = z[inb]
                    sensor = depth[vi, ui].astype(np.float64) * self.dg_depth_scale
                    valid = np.isfinite(sensor) & (sensor > 1e-3)
                    vr = float(valid.mean())
                    res["depth_valid_ratio"] = round(vr, 4)
                    if not self._depth_scale_logged:
                        self._depth_scale_logged = True
                        sv = sensor[valid]
                        med = float(np.median(sv)) if sv.size else -1.0
                        self.get_logger().info(
                            f"[depth_gate] depth 단위 검증(frame {self._frame_counter}): "
                            f"sensor median={med:.1f}(×scale {self.dg_depth_scale}), "
                            f"pose z={float(t[2]):.1f}mm → mm 가정 {'타당' if 50<med<5000 else '재확인 필요'}")
                    if int(valid.sum()) >= 1:
                        err = np.abs(zp[valid] - sensor[valid])
                        res["depth_error_median"] = round(float(np.median(err)), 2)
                        res["depth_error_p90"] = round(float(np.percentile(err, 90)), 2)
                        res["depth_agreement_ratio"] = round(float((err < self.dg_tol_mm).mean()), 4)
                    # 판정
                    if vr < self.dg_vr_min:
                        res["depth_gate_reason"] = f"DEPTH_UNRELIABLE(vr={vr:.2f})"  # pass=True 유지(보류)
                    elif res["depth_error_median"] is not None and res["depth_error_median"] > self.dg_err_max:
                        res["depth_gate_pass"] = False
                        res["depth_gate_reason"] = f"DEPTH_ERR_HIGH(med={res['depth_error_median']:.0f})"
                    elif res["depth_agreement_ratio"] is not None and res["depth_agreement_ratio"] < self.dg_agree_min:
                        res["depth_gate_pass"] = False
                        res["depth_gate_reason"] = f"DEPTH_DISAGREE(agree={res['depth_agreement_ratio']:.2f})"

        res["gate_pass"] = bool(res["workspace_gate_pass"] and res["depth_gate_pass"])
        return res

    @staticmethod
    def _geodesic_deg(Ra, Rb):
        """두 회전행렬 사이 geodesic angle(deg)."""
        if Ra is None or Rb is None:
            return None
        m = np.asarray(Ra).reshape(3, 3).T @ np.asarray(Rb).reshape(3, 3)
        c = max(-1.0, min(1.0, (float(np.trace(m)) - 1.0) / 2.0))
        return round(float(np.degrees(np.arccos(c))), 3)

    @staticmethod
    def _mat_to_quat(R):
        """3x3 회전행렬 → quaternion [x, y, z, w]. (디버그 로깅 전용, 의존성 최소)"""
        if R is None:
            return [None, None, None, None]
        m = np.asarray(R, dtype=np.float64).reshape(3, 3)
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return [round(float(x), 6), round(float(y), 6), round(float(z), 6), round(float(w), 6)]

    @staticmethod
    def _rle_area(seg):
        """COCO uncompressed RLE → mask 픽셀 수. 실패 시 None."""
        try:
            c = seg.get("counts")
            if isinstance(c, str) or c is None:
                return None
            s = 0; val = 0
            for x in c:
                if val == 1:
                    s += int(x)
                val ^= 1
            return s
        except Exception:
            return None

    def _read_pem_areas(self, frame_id):
        """detection_pem.json 에서 candidate index → (bbox_area, mask_area). 없으면 {}."""
        p = os.path.join(self.output_dir, str(frame_id), "detection_pem.json")
        out = {}
        if not os.path.isfile(p):
            return out
        try:
            dets = json.load(open(p))
            for i, d in enumerate(dets):
                bb = d.get("bbox")
                ba = (int(bb[2]) * int(bb[3])) if bb and len(bb) >= 4 else None
                out[i] = (ba, self._rle_area(d.get("segmentation", {})))
        except Exception:
            pass
        return out

    def _write_debug_log(self, frame_id, records, decision, best_idx,
                         raw_by_obj=None, stab_by_obj=None, gate_info=None) -> None:
        """FR-9/10 + Depth gate: CSV(append) + per-frame JSON 저장.

        raw_by_obj/stab_by_obj: object_id → PoseDetection (안정화 전/후).
        gate_info: {before, after, actual, mode, gate_by_obj} (depth/workspace gate 결과).
        """
        try:
            base = self.debug_log_dir or os.path.join(self.output_dir, "debug")
            base = os.path.expanduser(base)
            os.makedirs(base, exist_ok=True)

            ts = datetime.now().isoformat(timespec="seconds")
            if self._debug_csv_path is None:
                self._debug_csv_path = os.path.join(base, "sam6d_debug.csv")
            csv_fields = [
                "timestamp", "frame_counter", "frame_id", "candidate", "object_id",
                "similarity_score", "appearance_score", "geometric_score",
                "ism_score", "pose_score", "final_score",
                "is_best", "selected_candidate", "candidate_decision", "final_decision",
                "t_x_mm", "t_y_mm", "t_z_mm",                       # raw (안정화 전)
                "t_x_stab_mm", "t_y_stab_mm", "t_z_stab_mm",        # stabilized (FR-10)
                "qx_raw", "qy_raw", "qz_raw", "qw_raw",
                "qx_stab", "qy_stab", "qz_stab", "qw_stab",
                # ── Depth/Pose gate (Phase 0) ──
                "bbox_area", "mask_area",
                "pose_x", "pose_y", "pose_z", "pose_distance_mm",
                "raw_delta_translation", "raw_delta_rotation_deg",
                "stabilized_delta_translation", "stabilized_delta_rotation_deg",
                "depth_valid_ratio", "depth_error_median", "depth_error_p90",
                "depth_agreement_ratio", "depth_gate_pass", "depth_gate_reason",
                "workspace_gate_pass", "workspace_gate_reason",
                "final_publish_decision_before_gate", "final_publish_decision_after_gate",
                "actual_publish_decision", "depth_gate_mode",
                # ── Template selection 계측 (append-only; 기존 컬럼 순서 불변) ──
                "best_template_id", "best_template_score",
                "top5_template_ids", "top5_template_scores",
            ]
            # ── gate / delta / area 사전 계산 ──
            gi = gate_info or {}
            gate_by_obj = gi.get("gate_by_obj", {}) or {}
            g_before = gi.get("before", decision)
            g_after  = gi.get("after", decision)
            g_actual = gi.get("actual", decision)
            g_mode   = gi.get("mode", self.depth_gate_mode)
            areas = self._read_pem_areas(frame_id)
            delta_by_obj = {}
            for oid, rawp in (raw_by_obj or {}).items():
                stbp = (stab_by_obj or {}).get(oid)
                prev = self._prev_pose.get(oid)
                drt = drr = dst = dsr = None
                if prev is not None:
                    pr = prev.get("raw"); ps = prev.get("stab")
                    if pr is not None and rawp is not None:
                        drt = round(float(np.linalg.norm(np.asarray(rawp.t_mm) - pr[1])), 3)
                        drr = self._geodesic_deg(pr[0], rawp.R)
                    if ps is not None and stbp is not None:
                        dst = round(float(np.linalg.norm(np.asarray(stbp.t_mm) - ps[1])), 3)
                        dsr = self._geodesic_deg(ps[0], stbp.R)
                delta_by_obj[oid] = (drt, drr, dst, dsr)
            for oid, rawp in (raw_by_obj or {}).items():
                stbp = (stab_by_obj or {}).get(oid)
                self._prev_pose[oid] = {
                    "raw": (np.asarray(rawp.R), np.asarray(rawp.t_mm)) if rawp is not None else None,
                    "stab": (np.asarray(stbp.R), np.asarray(stbp.t_mm)) if stbp is not None else None}

            write_header = not os.path.exists(self._debug_csv_path)
            with open(self._debug_csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=csv_fields)
                if write_header:
                    w.writeheader()
                if not records:
                    w.writerow({
                        "timestamp": ts, "frame_counter": self._frame_counter,
                        "frame_id": frame_id, "candidate": -1, "object_id": -1,
                        "selected_candidate": best_idx,
                        "candidate_decision": "NONE", "final_decision": decision,
                        "final_publish_decision_before_gate": g_before,
                        "final_publish_decision_after_gate": g_after,
                        "actual_publish_decision": g_actual, "depth_gate_mode": g_mode,
                    })
                for r in records:
                    tx, ty, tz = r["t_mm"]
                    oid  = r["object_id"]
                    rawp = (raw_by_obj or {}).get(oid)
                    stbp = (stab_by_obj or {}).get(oid)
                    q_raw  = self._mat_to_quat(rawp.R) if rawp is not None else [None] * 4
                    q_stab = self._mat_to_quat(stbp.R) if stbp is not None else [None] * 4
                    t_stab = ([round(float(v), 1) for v in stbp.t_mm]
                              if stbp is not None else [None, None, None])
                    gp = gate_by_obj.get(oid, {})
                    dr = delta_by_obj.get(oid, (None, None, None, None))
                    ba, ma = areas.get(int(r["candidate"]), (None, None)) \
                        if str(r["candidate"]).lstrip("-").isdigit() else (None, None)
                    # pose(x/y/z): 발행 기준 = stabilized 우선, 없으면 raw
                    pt = (stbp.t_mm if stbp is not None else (rawp.t_mm if rawp is not None else None))
                    px = py = pz = pdist = None
                    if pt is not None:
                        px, py, pz = round(float(pt[0]), 1), round(float(pt[1]), 1), round(float(pt[2]), 1)
                        pdist = round(float(np.linalg.norm(np.asarray(pt))), 1)
                    w.writerow({
                        "timestamp": ts, "frame_counter": self._frame_counter,
                        "frame_id": frame_id, "candidate": r["candidate"],
                        "object_id": oid,
                        "similarity_score": r["similarity_score"],
                        "appearance_score": r["appearance_score"],
                        "geometric_score": r["geometric_score"],
                        "ism_score": r["ism_score"], "pose_score": r["pose_score"],
                        "final_score": r["final_score"], "is_best": r["is_best"],
                        "selected_candidate": best_idx,
                        "candidate_decision": r["candidate_decision"],
                        "final_decision": decision,
                        "t_x_mm": tx, "t_y_mm": ty, "t_z_mm": tz,
                        "t_x_stab_mm": t_stab[0], "t_y_stab_mm": t_stab[1], "t_z_stab_mm": t_stab[2],
                        "qx_raw": q_raw[0], "qy_raw": q_raw[1], "qz_raw": q_raw[2], "qw_raw": q_raw[3],
                        "qx_stab": q_stab[0], "qy_stab": q_stab[1], "qz_stab": q_stab[2], "qw_stab": q_stab[3],
                        "bbox_area": ba, "mask_area": ma,
                        "pose_x": px, "pose_y": py, "pose_z": pz, "pose_distance_mm": pdist,
                        "raw_delta_translation": dr[0], "raw_delta_rotation_deg": dr[1],
                        "stabilized_delta_translation": dr[2], "stabilized_delta_rotation_deg": dr[3],
                        "depth_valid_ratio": gp.get("depth_valid_ratio"),
                        "depth_error_median": gp.get("depth_error_median"),
                        "depth_error_p90": gp.get("depth_error_p90"),
                        "depth_agreement_ratio": gp.get("depth_agreement_ratio"),
                        "depth_gate_pass": gp.get("depth_gate_pass"),
                        "depth_gate_reason": gp.get("depth_gate_reason"),
                        "workspace_gate_pass": gp.get("workspace_gate_pass"),
                        "workspace_gate_reason": gp.get("workspace_gate_reason"),
                        "final_publish_decision_before_gate": g_before,
                        "final_publish_decision_after_gate": g_after,
                        "actual_publish_decision": g_actual,
                        "depth_gate_mode": g_mode,
                        "best_template_id": r.get("best_template_id"),
                        "best_template_score": r.get("best_template_score"),
                        # list 는 CSV 셀 안전을 위해 JSON 문자열로 직렬화
                        "top5_template_ids": (json.dumps(r.get("top5_template_ids"))
                                              if r.get("top5_template_ids") else None),
                        "top5_template_scores": (json.dumps(r.get("top5_template_scores"))
                                                 if r.get("top5_template_scores") else None),
                    })

            pose_pairs = []
            for oid, rawp in (raw_by_obj or {}).items():
                stbp = (stab_by_obj or {}).get(oid)
                pose_pairs.append({
                    "object_id": oid,
                    "raw":  {"t_mm": [round(float(v), 2) for v in rawp.t_mm],
                             "quat": self._mat_to_quat(rawp.R)},
                    "stab": ({"t_mm": [round(float(v), 2) for v in stbp.t_mm],
                              "quat": self._mat_to_quat(stbp.R)} if stbp is not None else None),
                })
            json_path = os.path.join(base, f"frame_{self._frame_counter:06d}_{frame_id}.json")
            with open(json_path, "w") as f:
                json.dump({
                    "timestamp":          ts,
                    "frame_counter":      self._frame_counter,
                    "frame_id":           frame_id,
                    "selected_candidate": best_idx,
                    "final_decision":     decision,
                    "stabilize_enabled":  self.stabilize_enabled,
                    "depth_gate": {"mode": g_mode, "before": g_before, "after": g_after,
                                   "actual": g_actual,
                                   "by_object": {str(k): v for k, v in gate_by_obj.items()}},
                    "candidates":         records,
                    "poses":              pose_pairs,
                }, f, indent=2)
        except Exception as exc:
            self.get_logger().warn(f"debug 로그 저장 실패: {exc}")

    def _run_inference(self) -> tuple[list[PoseDetection], dict]:
        import run_batch_inference_fast as rb

        # FR-8: 프레임 단위 재시드 — 백엔드의 전역 numpy/torch RNG(np.random.choice,
        # torch.rand)를 고정해 정지 입력 시 동일 출력을 보장한다.
        self._apply_determinism()
        self._frame_counter += 1

        stem    = self._active_output_stem or os.path.splitext(os.path.basename(self.rgb_path))[0]
        img_out = os.path.join(self.output_dir, stem)
        os.makedirs(img_out, exist_ok=True)
        seg_path = os.path.join(img_out, "detection_ism.json")
        self.get_logger().debug(
            f"추론 시작: rgb={self.rgb_path}, depth={self.depth_path}, "
            f"camera={self.camera_for_inference_path}, output={img_out}"
        )

        self.get_logger().debug("── ISM 추론 실행...")
        no_vis_effective = self.no_vis or self.skip_file_save or not self.show_imshow
        if self.ism_rerank_enabled:
            ism_ret = self._run_ism_single_area_rerank(
                rb,
                self.rgb_path,
                self.depth_path,
                self.camera_for_inference_path,
                img_out,
                top_k_for_pem=self.top_k_for_pem,
                max_proposals=self.max_proposals,
                no_vis=no_vis_effective,
            )
        else:
            ism_ret = rb.run_ism_single(
                self.ism_model,
                self.rgb_path,
                self.depth_path,
                self.camera_for_inference_path,
                img_out,
                self.device,
                top_k_for_pem=self.top_k_for_pem,
                max_proposals=self.max_proposals,
                no_vis=no_vis_effective,
            )
        if not ism_ret["ok"]:
            raise RuntimeError(f"ISM 실패: {ism_ret.get('error', '알 수 없는 오류')}")

        scores = ism_ret["scores"]
        self.get_logger().debug(f"   ISM 완료 — {len(scores)}개 검출  {ism_ret['timing']}")
        for i, s in enumerate(scores):
            marker = " ◀ best" if i == 0 else ""
            self.get_logger().debug(f"   [{i}] final={s['final']:.4f}{marker}")

        self.get_logger().debug("── PEM 추론 실행...")
        pem_ret = rb.run_pem_single(
            self.pem_model,
            self.all_tem_pts,
            self.all_tem_feat,
            self.pem_cfg,
            self.rgb_path,
            self.depth_path,
            self.camera_for_inference_path,
            seg_path,
            img_out,
            self.det_score_thresh,
            self.model_points,
            self.radius,
            device=self.device,
            no_vis=no_vis_effective,
        )
        if not pem_ret["ok"]:
            raise RuntimeError(f"PEM 실패: {pem_ret.get('error', '알 수 없는 오류')}")

        pem_poses = pem_ret["poses"]
        self.get_logger().debug(f"   PEM 완료 — {len(pem_poses)}개 포즈  {pem_ret['timing']}")
        best_idx = max(range(len(pem_poses)), key=lambda i: pem_poses[i]["final_score"]) if pem_poses else -1
        for i, p in enumerate(pem_poses):
            marker = " ◀ best" if i == best_idx else ""
            tx, ty, tz = p["t_mm"]
            pem_s = f"{p['pem_score']:.4f}" if p["pem_score"] is not None else "N/A"
            self.get_logger().debug(
                f"   [{i}] ism={p['ism_score']:.4f}  pem={pem_s}  "
                f"final={p['final_score']:.4f}  "
                f"t=[{tx:.1f},{ty:.1f},{tz:.1f}]mm{marker}"
            )

        # FR-1~4: confidence floor 적용 + no-object 판정. (debug_records 는 FR-9 용)
        poses, debug_records, decision = self._filter_and_decide(pem_poses, scores, best_idx)

        # Fall back to the PEM result file for older SAM-6D return formats.
        # (filter 로 모두 제거된 NO_OBJECT 가 아니라, 애초에 pem_poses 가 비어있던 경우에만)
        if not self.skip_file_save and not poses and decision == "NO_DETECTION":
            det_path = os.path.join(img_out, "detection_pem.json")
            if os.path.isfile(det_path):
                with open(det_path, "r") as f:
                    detections = json.load(f)
                for idx, det in enumerate(detections):
                    if "R" not in det or "t" not in det:
                        continue
                    object_id = int(
                        det.get("obj_id", det.get("object_id", det.get("category_id", idx + 1)))
                    )
                    R = np.array(det["R"], dtype=np.float64)
                    t = np.array(det["t"], dtype=np.float64)
                    poses.append(PoseDetection(object_id=object_id, R=R, t_mm=t))
                if poses:
                    decision = "PUBLISH_LEGACY"

        # FR-5~7: temporal stabilization (EMA + outlier/jump rejection)
        # FR-10: raw pose 를 안정화 직전에 캡쳐해 stabilized 와 함께 로깅(ON/OFF 정량 비교).
        poses_raw = list(poses)
        if self.stabilize_enabled and poses:
            poses = self._stabilize_poses(poses)
        raw_by_obj  = {p.object_id: p for p in poses_raw}
        stab_by_obj = {p.object_id: p for p in poses}

        # Depth/Pose Secondary Gate (Phase 0): score 기반 decision '이후' 단계.
        # off/shadow = 발행 불변, hard = gate 실패 pose 발행 제외. score floor 불변.
        decision_before_gate = decision
        gate_by_obj = {}
        if self.depth_gate_mode != "off" and poses:
            gate_by_obj = {p.object_id: self._depth_pose_gate(p.R, p.t_mm) for p in poses}
        poses_gate_pass = [p for p in poses
                           if gate_by_obj.get(p.object_id, {}).get("gate_pass", True)]
        # after_gate = gate 가 판정한 결과(shadow 에서도 시뮬 기록). 발행 변경은 hard 에서만.
        decision_after_gate = decision_before_gate
        if (self.depth_gate_mode != "off" and poses
                and decision_before_gate in ("PUBLISH", "PUBLISH_LEGACY")
                and not poses_gate_pass):
            decision_after_gate = "NO_OBJECT_GATED"
        if self.depth_gate_mode == "hard":
            poses = poses_gate_pass     # hard 에서만 실제 발행 목록 변경
        actual_publish_decision = (decision_after_gate if self.depth_gate_mode == "hard"
                                   else decision_before_gate)
        gate_info = {"before": decision_before_gate, "after": decision_after_gate,
                     "actual": actual_publish_decision, "mode": self.depth_gate_mode,
                     "gate_by_obj": gate_by_obj}

        if decision in ("NO_OBJECT", "NO_DETECTION"):
            # poses 가 비어 _publish_poses 는 빈 PoseArray 만 발행(개별 pose/TF 없음).
            self.get_logger().info(
                f"[no-object] pose 미발행 (빈 PoseArray) — decision={decision}, "
                f"raw={len(pem_poses)}개 후보 모두 임계 미달"
            )

        # FR-9/10: debug 저장 (CSV + per-frame JSON, raw+stabilized pose + gate 포함)
        if self.debug_log_enabled:
            self._write_debug_log(stem, debug_records, decision, best_idx,
                                  raw_by_obj, stab_by_obj, gate_info)

        ism_t = ism_ret.get("timing", {})
        pem_t = pem_ret.get("timing", {})

        t_load = float(ism_t.get("image_load", 0))
        t_preprocess = float(pem_t.get("data_preprocessing", 0))
        t_gpu = (
            float(ism_t.get("mask_generation", 0)) +
            float(ism_t.get("dinov2_feature", 0)) +
            float(ism_t.get("semantic_score", 0)) +
            float(ism_t.get("appearance_score", 0)) +
            float(ism_t.get("geometric_score", 0)) +
            float(pem_t.get("model_forward", 0))
        )
        t_render = (
            float(ism_t.get("save_json", 0)) +
            float(ism_t.get("visualization", 0)) +
            float(pem_t.get("save_json", 0)) +
            float(pem_t.get("visualization", 0))
        )

        timing_breakdown = {
            "load":          round(t_load, 3),
            "preprocess":    round(t_preprocess, 3),
            "gpu_inference": round(t_gpu, 3),
            "render":        round(t_render, 3),
            "ism_detail":    ism_t,
            "pem_detail":    pem_t,
        }

        return poses, timing_breakdown

    def _publish_poses(self, poses: list[PoseDetection]) -> float:
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"

        for pose in poses:
            p = Pose()
            p.position.x = float(pose.t_mm[0]) / 1000.0
            p.position.y = float(pose.t_mm[1]) / 1000.0
            p.position.z = float(pose.t_mm[2]) / 1000.0
            q = Rotation.from_matrix(pose.R).as_quat()  # (x, y, z, w)
            p.orientation.x = float(q[0])
            p.orientation.y = float(q[1])
            p.orientation.z = float(q[2])
            p.orientation.w = float(q[3])
            msg.poses.append(p)

        self.pose_pub.publish(msg)
        self.get_logger().debug(
            f"PoseArray 퍼블리시: {len(poses)}개 → /sam6d_inference/poses"
        )
        return self._publish_pose_transforms(poses, msg.header.stamp, msg.header.frame_id)

    def _publish_pose_transforms(
        self, poses: list[PoseDetection], stamp, frame_id: str
    ) -> float:
        end_time = time.perf_counter()
        for idx, pose in enumerate(poses):
            q = Rotation.from_matrix(pose.R).as_quat()  # (x, y, z, w)
            msg = TransformStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = frame_id
            msg.child_frame_id = f"sam6d_object_{pose.object_id}_{idx}"
            msg.transform.translation.x = float(pose.t_mm[0]) / 1000.0
            msg.transform.translation.y = float(pose.t_mm[1]) / 1000.0
            msg.transform.translation.z = float(pose.t_mm[2]) / 1000.0
            msg.transform.rotation.x = float(q[0])
            msg.transform.rotation.y = float(q[1])
            msg.transform.rotation.z = float(q[2])
            msg.transform.rotation.w = float(q[3])
            self.pose_6dof_pub.publish(msg)
            end_time = time.perf_counter()
            self.get_logger().debug(
                "6DoF 발행 직후: "
                f"child_frame_id={msg.child_frame_id}, "
                f"xyz=({msg.transform.translation.x:.4f}, "
                f"{msg.transform.translation.y:.4f}, "
                f"{msg.transform.translation.z:.4f}), "
                f"quat=({msg.transform.rotation.x:.4f}, "
                f"{msg.transform.rotation.y:.4f}, "
                f"{msg.transform.rotation.z:.4f}, "
                f"{msg.transform.rotation.w:.4f})"
            )

        self.get_logger().debug(
            f"6DoF TransformStamped 퍼블리시: {len(poses)}개 → {self.pose_6dof_topic}"
        )
        return end_time

    def _log_processing_summary(
        self,
        image_path: str,
        start_time: float,
        end_time: float,
        timing: dict | None = None,
        file_save_time: float = 0.0,
    ) -> None:
        elapsed = end_time - start_time
        image_name = os.path.basename(image_path) if image_path else "unknown"

        if timing:
            t_load = timing.get("load", 0.0)
            t_pre = timing.get("preprocess", 0.0)
            t_gpu = timing.get("gpu_inference", 0.0)
            t_render = timing.get("render", 0.0)
            t_other = max(
                0.0,
                elapsed - t_load - file_save_time - t_pre - t_gpu - t_render,
            )

            self.get_logger().info(
                f"[{image_name}] 총: {elapsed:.2f}s | "
                f"로드: {t_load:.2f}s | "
                f"파일저장: {file_save_time:.2f}s | "
                f"전처리: {t_pre:.2f}s | "
                f"GPU추론: {t_gpu:.2f}s | "
                f"렌더링: {t_render:.2f}s | "
                f"기타: {t_other:.2f}s"
            )
            self.get_logger().debug(
                f"  └─ [상세] 파일저장(입력): {file_save_time:.3f}s | "
                f"skip_file_save={self.skip_file_save}"
            )
            ism_d = timing.get("ism_detail", {})
            pem_d = timing.get("pem_detail", {})
            if ism_d:
                ism_parts = " | ".join(
                    f"{k}={v:.3f}s" for k, v in ism_d.items() if isinstance(v, (int, float))
                )
                self.get_logger().debug(f"  [ISM] {ism_parts}")
            if pem_d:
                pem_parts = " | ".join(
                    f"{k}={v:.3f}s" for k, v in pem_d.items() if isinstance(v, (int, float))
                )
                self.get_logger().debug(f"  [PEM] {pem_parts}")
        else:
            self.get_logger().info(
                f"처리 완료: 이미지 {image_name} | 입력~포즈발행 총 소요시간: {elapsed:.2f}초"
            )

    def _publish_result_image(self) -> None:
        if not self.show_imshow:
            return

        vis_path = self._find_latest_result_image()
        if not vis_path:
            self.get_logger().debug("시각화 결과 이미지 없음: vis_pem.png/vis_ism.png를 찾지 못했습니다.")
            return

        image_bgr = cv2.imread(vis_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            self.get_logger().debug(f"시각화 결과 이미지 로드 실패: {vis_path}")
            return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        msg = self.bridge.cv2_to_imgmsg(image_rgb, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        self.result_image_pub.publish(msg)
        self._queue_result_image_for_display(vis_path)
        self.get_logger().debug(f"Result image 퍼블리시: {vis_path} → {self.result_image_topic}")

    def _queue_result_image_for_display(self, vis_path: str) -> None:
        with self._display_lock:
            self._pending_vis_path = vis_path
        self.get_logger().debug(f"imshow 대기열 갱신: {vis_path}")

    def _visualization_timer_cb(self) -> None:
        if not self.show_imshow or not self._imshow_available:
            return

        with self._display_lock:
            vis_path = self._pending_vis_path

        try:
            if vis_path:
                exists = os.path.exists(vis_path)
                if exists:
                    mtime = os.path.getmtime(vis_path)
                    display_key = (vis_path, mtime)
                    if display_key != self._displayed_vis_key:
                        self.get_logger().debug(
                            f"imshow 타이머: path={vis_path}, exists={exists}"
                        )
                        self.get_logger().debug(f"imshow 이미지 읽기 시작: {vis_path}")
                        image_bgr = cv2.imread(vis_path, cv2.IMREAD_COLOR)
                        if image_bgr is None:
                            self.get_logger().debug(
                                f"imshow 이미지 로드 실패(None): {vis_path}"
                            )
                        else:
                            self.get_logger().debug(
                                f"imshow 호출 직전: path={vis_path}, "
                                f"shape={image_bgr.shape}, dtype={image_bgr.dtype}"
                            )
                            if not self._display_window_created:
                                cv2.namedWindow(self.RESULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
                                self._display_window_created = True
                            cv2.imshow(self.RESULT_WINDOW_NAME, image_bgr)
                            self._displayed_vis_key = display_key
                elif self._last_visualization_log_key != (vis_path, exists):
                    self.get_logger().debug(
                        f"imshow 타이머: path={vis_path}, exists={exists}"
                    )
                    self._last_visualization_log_key = (vis_path, exists)

            cv2.waitKey(1)
        except cv2.error as exc:
            self._imshow_available = False
            self.get_logger().warn(
                f"cv2.imshow 비활성화: {exc}. DISPLAY/X11 환경을 확인하세요. "
                f"이미지 파일은 계속 생성됩니다: {vis_path}"
            )

    def _find_latest_result_image(self) -> str:
        stem = self._active_output_stem or os.path.splitext(os.path.basename(self.rgb_path))[0]
        img_out = os.path.join(self.output_dir, stem)
        candidates = [
            os.path.join(img_out, "vis_pem.png"),
            os.path.join(img_out, "vis_ism.png"),
        ]
        for path in candidates:
            exists = os.path.exists(path)
            self.get_logger().debug(f"시각화 후보 확인: path={path}, exists={exists}")
            if exists and os.path.isfile(path):
                return path
        return ""

    def destroy_node(self) -> bool:
        if self.show_imshow and self._imshow_available:
            try:
                cv2.destroyWindow(self.RESULT_WINDOW_NAME)
                cv2.waitKey(1)
            except cv2.error:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Sam6dInferenceNode()
    interrupted = False
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if interrupted:
            os._exit(0)


if __name__ == "__main__":
    main()
