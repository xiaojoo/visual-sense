from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


# =========================================================
# 状态机
# =========================================================

STATE_OFF = "off"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_FAILED = "failed"


@dataclass
class DetectorConfig:
    """
    V0.2 检测器配置。

    对应 config/config.json 的 "ai" 段。
    """

    enabled: bool = False

    # 权重路径。相对路径按项目根目录解析。
    model_path: str = "weights/yolo11n.pt"

    # auto / cpu / cuda / 0 / cuda:0
    device: str = "auto"

    imgsz: int = 640

    conf: float = 0.25

    iou: float = 0.70

    # 推理精度。None = FP32。
    # ultralytics 8.4 起用 quantize 取代已废弃的 half，
    # "fp16" / 16 表示半精度。
    quantize: Optional[str] = None

    # 每 N 帧推理一次。1 = 每帧。
    # 用于对比 "推理耗时" 与 "端到端 FPS"，
    # 网络摄像头 30 FPS 时可用 2~3 换取流畅度。
    detect_every: int = 1

    # 传给 ultralytics 的 max_det
    max_det: int = 100

    # V0.3：把 detect 换成 track，同一目标跨帧给同一个 ID
    track_enabled: bool = False

    # ultralytics 内置的跟踪器配置名
    tracker: str = "bytetrack.yaml"

    # 每个 ID 保留多少个历史中心点，用来画轨迹
    trail_frames: int = 40

    # ---- V0.6 多模型 ----

    # 这路模型在结果里的身份。track_id 只在单路模型内唯一，
    # 两路模型都从 1 开始编号，不带 key 就会撞成同一个目标。
    key: str = "mosquito"

    # 页面上显示的名字
    label: str = ""

    # 类别白名单，None = 不过滤。
    # 这是"拿现成 COCO 权重只认人"的关键 —— 没有它，
    # 换通用权重就等于把 80 个类全开出来。
    classes: Optional[list[int]] = None

    # 这路的输出能不能驱动激光。默认 False。
    # 通用框架里"任何模型都能触发发射"是不可接受的默认值：
    # 接上人模型的那天，就是激光可能对着人那天。
    targetable: bool = False


@dataclass
class Detection:
    """
    单个检测框，坐标为原始视频帧像素坐标。
    """

    x1: float
    y1: float
    x2: float
    y2: float

    confidence: float
    class_id: int
    class_name: str

    # 哪一路模型给的框。多模型同时跑时，
    # 光靠 class_name 分不出是两个模型还是两次误检。
    source: str = ""

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )


@dataclass
class Track(Detection):
    """
    V0.3 带 ID 的检测框。

    track_id 由跟踪器分配，同一只目标跨帧保持不变。
    """

    track_id: int = -1


class Detector:
    """
    V0.2 YOLO 检测器封装。

    Camera
        ↓
    frame (BGR ndarray)
        ↓
    Detector.detect()
        ↓
    list[Detection]

    设计约束：

        1. ultralytics 只在 enabled=True 时才 import。
           V0.1 的轻量环境（只有 opencv + numpy）
           依然可以直接运行本文件。

        2. 加载失败不抛出，只记录 last_error，
           主程序降级为纯显示，V0.1 行为不变。

        3. 模型路径可配置。
           换成自己训练的 weights/mosquito.pt 时，
           本文件和主循环都不需要修改。
    """

    def __init__(
        self,
        config: DetectorConfig,
        root: Path,
    ):
        self.config = config
        self.root = Path(root)

        self._model: Optional[Any] = None
        self._names: dict[int, str] = {}
        self._device: Optional[str] = None

        self.state = STATE_OFF
        self.last_error = ""

        self.load_ms = 0.0
        self.last_infer_ms = 0.0
        self.last_count = 0

        # 跟踪器状态是否已经建立
        self._tracking = False

    # =========================================================
    # Properties
    # =========================================================

    @property
    def ready(self) -> bool:
        return self.state == STATE_READY and self._model is not None

    @property
    def model_name(self) -> str:
        return Path(self.config.model_path).name

    @property
    def device(self) -> str:
        """
        实际使用的推理设备。

        加载前返回配置值，
        加载后返回解析结果（auto 会展开成 cuda / cpu）。
        """

        return self._device or self.config.device

    @property
    def class_names(self) -> dict[int, str]:
        """
        权重自带的类别表。采集时要原样写进 metadata，
        否则日后看到 labels 里的 class_id=0 会不知道指什么。
        """

        return dict(self._names)

    # =========================================================
    # Path
    # =========================================================

    def resolve_model_path(self) -> Path:
        """
        相对路径 → 项目根目录下的绝对路径。
        """

        path = Path(self.config.model_path).expanduser()

        if path.is_absolute():
            return path

        return (self.root / path).resolve()

    # =========================================================
    # Device
    # =========================================================

    def resolve_device(self) -> str:
        """
        auto → 有 CUDA 用 CUDA，否则 CPU。

        显式写 cpu / cuda / 0 时按配置走，
        但 CUDA 不可用会退回 CPU，
        避免整条链路直接起不来。
        """

        requested = str(self.config.device).strip().lower()

        if requested in ("", "auto"):

            try:
                import torch
            except ImportError:
                return "cpu"

            if torch.cuda.is_available():
                return "cuda"

            return "cpu"

        if requested in ("cuda", "gpu"):
            try:
                import torch

                if torch.cuda.is_available():
                    return requested
            except ImportError:
                pass

            print(
                "[Detector] CUDA 不可用，"
                "回退到 CPU"
            )

            return "cpu"

        return requested

    # =========================================================
    # Load
    # =========================================================

    def load(self) -> bool:
        """
        加载权重。

        幂等：已 ready 时直接返回 True。
        """

        if self.ready:
            return True

        self.state = STATE_LOADING
        self.last_error = ""

        started = time.perf_counter()

        # -----------------------------------------------------
        # import 放在函数内：
        # V0.1 环境没有 ultralytics，
        # 只要 enabled=False 就永远不会走到这里
        # -----------------------------------------------------

        try:
            from ultralytics import YOLO

        except ImportError:

            self.state = STATE_FAILED

            self.last_error = (
                "ultralytics 未安装，"
                "请使用 run-ai.bat 启动"
            )

            print(
                f"[Detector] {self.last_error}"
            )

            return False

        model_path = self.resolve_model_path()

        print(
            f"[Detector] Loading "
            f"{model_path}"
        )

        try:

            self._device = self.resolve_device()

            print(
                f"[Detector] Device = "
                f"{self._device}"
            )

            self._model = YOLO(str(model_path))

            names = getattr(self._model, "names", None) or {}

            self._names = {
                int(k): str(v)
                for k, v in names.items()
            }

            self.load_ms = (
                time.perf_counter() - started
            ) * 1000.0

            self.state = STATE_READY

            print(
                f"[Detector] Loaded in "
                f"{self.load_ms:.0f} ms, "
                f"{len(self._names)} classes"
            )

            self._warmup()

            return True

        except Exception as exc:

            self._model = None

            self.state = STATE_FAILED

            self.last_error = str(exc)

            print(
                f"[Detector] Load failed: "
                f"{self.last_error}"
            )

            return False

    def _warmup(self) -> None:
        """
        第一次 predict 会触发 kernel 编译 / 显存分配，
        耗时远高于稳态。

        先跑一张纯色图，
        让后续测到的推理耗时是稳态值。
        """

        if not self.ready:
            return

        try:

            dummy = np.zeros(
                (self.config.imgsz,) * 2 + (3,),
                dtype=np.uint8,
            )

            self.detect(dummy)

            # 预热的目的就是把它自己的耗时冲掉。
            # 不清零的话，这一发的 kernel 编译成本会被当成"稳态推理耗时"
            # 喂给任何读 last_infer_ms 的人 —— 实测 1290 ms vs 稳态 9 ms。
            self.last_infer_ms = 0.0
            self.last_count = 0

        except Exception:
            # 预热失败不影响可用性
            pass

    def unload(self) -> None:
        self._model = None
        self._names = {}
        self.state = STATE_OFF
        self.last_count = 0
        self._tracking = False

    # =========================================================
    # Detect / Track
    # =========================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[Detection]:
        """
        对一帧做推理，不带 ID。

        返回帧坐标系的检测框，
        并把耗时记入 last_infer_ms。
        """

        return self._run(frame, track=False)

    def track(
        self,
        frame: np.ndarray,
    ) -> list[Detection]:
        """
        V0.3：推理 + 分配跟踪 ID。

        同一只目标跨帧拿到同一个 track_id，
        这是后面算速度和预测位置的前提。
        """

        return self._run(frame, track=True)

    def reset_tracking(self) -> None:
        """
        让下一帧重新起头。

        开关跟踪、换权重、摄像头重连之后都要调一次，
        否则跟踪器会拿着上一段画面的运动历史继续猜。
        """

        self._tracking = False

    def _run(
        self,
        frame: np.ndarray,
        track: bool,
    ) -> list[Detection]:

        if not self.ready:
            return []

        started = time.perf_counter()

        arguments = {
            "source": frame,
            "imgsz": self.config.imgsz,
            "conf": self.config.conf,
            "iou": self.config.iou,
            "max_det": self.config.max_det,
            "device": self._device,
            "verbose": False,
        }

        # 类别白名单。不传就是全开 —— 通用权重会把 80 个类都框出来。
        if self.config.classes:
            arguments["classes"] = list(self.config.classes)

        # 不传 quantize 就是 FP32。
        # 传 quantize=None 会被当成"显式清除精度"，
        # 而不传则保留模型自带设置。
        if self.config.quantize:

            arguments["quantize"] = str(
                self.config.quantize
            )

        if track:

            arguments["tracker"] = self.config.tracker

            # 第一帧 False 建状态，之后 True 续状态
            arguments["persist"] = self._tracking

        try:

            if track:

                results = self._model.track(
                    **arguments
                )

            else:

                results = self._model.predict(
                    **arguments
                )

        except Exception as exc:

            self.state = STATE_FAILED

            self.last_error = str(exc)
            self.last_infer_ms = 0.0
            self.last_count = 0

            print(
                f"[Detector] Inference failed: "
                f"{self.last_error}"
            )

            return []

        self.last_infer_ms = (
            time.perf_counter() - started
        ) * 1000.0

        if track:
            self._tracking = True

        found = [
            item
            for result in results
            for item in self._parse(result, with_id=track)
        ]

        self.last_count = len(found)

        return found

    def _parse(
        self,
        result: Any,
        with_id: bool = False,
    ) -> list[Detection]:
        """
        ultralytics Results → Detection / Track 列表。
        """

        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        ids = None

        if with_id:

            raw_id = getattr(boxes, "id", None)

            if raw_id is not None:
                ids = raw_id.cpu().numpy().astype(int)

        found: list[Detection] = []

        for index, ((x1, y1, x2, y2), score, class_id) in enumerate(
            zip(
                xyxy,
                conf,
                cls,
            )
        ):

            fields = {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(score),
                "class_id": int(class_id),
                "class_name": self._names.get(
                    int(class_id),
                    str(int(class_id)),
                ),
                "source": self.config.key,
            }

            if ids is not None:

                fields["track_id"] = int(ids[index])

                found.append(Track(**fields))

            else:

                found.append(Detection(**fields))

        return found
