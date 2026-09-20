"""
V0.6 多模型组：一个画面，N 路检测器同时跑，各自带自己的跟踪和预测。

为什么不共用一个 Tracks / MotionPredictor：
ByteTrack 的 track_id 只在单个跟踪器内唯一。两路模型都从 1 开始编号，
合并到一起就会把"蚊子#1"和"人#1"当成同一个目标 ——
轨迹串味、预测拿错历史、激光锁错东西。
所以每路模型一套 Detector + Tracks + MotionPredictor，
对外再合并成一个目标列表，每条带 source。

detect_every 按模型给，这是多模型能负担的关键：
人不需要 25 fps，蚊子需要。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .detector import Detection, Detector, DetectorConfig
from .motion import MotionPredictor
from .runtime import toggle_detector
from .tracks import Tracks


@dataclass
class PredictConfig:
    """每路模型一套预测参数。大目标和小目标的最优窗口不是一回事。"""

    horizon_ms: float = 100.0
    window: int = 6
    min_samples: int = 4
    max_gap_s: float = 0.5
    enabled: bool = False


@dataclass
class Bundle:
    """一路模型 = 一个检测器 + 一套自己的轨迹和预测。"""

    config: DetectorConfig
    predict_config: PredictConfig
    detector: Detector
    trails: Tracks
    predictor: MotionPredictor
    last_detections: list = field(default_factory=list)
    last_states: list = field(default_factory=list)
    _samples: deque = field(default_factory=lambda: deque(maxlen=60), repr=False)

    @property
    def key(self) -> str:
        return self.config.key

    @property
    def name(self) -> str:
        return self.config.label or self.config.key

    @property
    def running(self) -> bool:
        return self.config.enabled and self.detector.ready

    def avg_infer_ms(self) -> float:
        """纯读。采样在 run() 里做 —— 快照每秒调它四次，
        在读取函数里改状态会把同一个值重复计入均值。"""

        return round(sum(self._samples) / len(self._samples), 1) if self._samples else 0.0

    def run(self, frame: np.ndarray, frame_counter: int, now: float) -> tuple[list[Detection], list]:
        """
        到点就推理，没到点就交回上一批结果。

        抽帧期间沿用旧框是有意的：画面不会一闪一闪。
        代价是采集时必须只存"真跑过推理"的那些帧，
        否则标签会配到别的帧上 —— 见 pipeline._record。
        """

        if not self.running:
            return [], []

        if frame_counter % max(1, self.config.detect_every) != 0:
            return list(self.last_detections), list(self.last_states)

        if self.config.track_enabled:
            found = self.detector.track(frame)
            self.trails.update(found, frame_counter, now)
            states = self.predictor.estimate_all(self.trails) if self.predictor.enabled else []
        else:
            found = self.detector.detect(frame)
            states = []

        if self.detector.last_infer_ms > 0:
            self._samples.append(self.detector.last_infer_ms)

        self.last_detections = found
        self.last_states = states

        return list(found), list(states)

    def snapshot(self) -> dict:
        return {
            "key": self.config.key,
            "label": self.name,
            "enabled": self.config.enabled,
            "running": self.running,
            "state": self.detector.state,
            "model": self.detector.model_name or self.config.model_path,
            "error": self.detector.last_error,
            "classes": self.config.classes,
            "imgsz": self.config.imgsz,
            "conf": self.config.conf,
            "detect_every": self.config.detect_every,
            "tracking": self.config.track_enabled,
            "predicting": self.predictor.enabled,
            "predict_window": self.predictor.window,
            "targetable": self.config.targetable,
            "tracks": len(self.trails),
            "last_count": self.detector.last_count,
            "infer_ms": self.avg_infer_ms(),
        }


class ModelGroup:
    def __init__(self, bundles: list[Bundle]):
        self.bundles = bundles
        self.detections: list[Detection] = []
        self.states: list = []
        self.inferred = False

    # -----------------------------------------------------

    def __iter__(self):
        return iter(self.bundles)

    def get(self, key: str) -> Bundle | None:
        return next((b for b in self.bundles if b.config.key == key), None)

    @property
    def any_running(self) -> bool:
        return any(b.running for b in self.bundles)

    def load_all(self) -> list[str]:
        """逐个加载。一个失败不影响另一个 —— 否则一路坏权重会拖死整个界面。"""

        notes = []

        for bundle in self.bundles:
            if not bundle.config.enabled:
                notes.append(f"[{bundle.name}] 未启用")
                continue

            notes.append(f"[{bundle.name}] {toggle_detector(bundle.detector)}")

        return notes

    def unload_all(self) -> None:
        for bundle in self.bundles:
            if bundle.detector.ready:
                bundle.detector.unload()

    def reset_tracking(self) -> None:
        for bundle in self.bundles:
            bundle.detector.reset_tracking()
            bundle.trails.reset()

    # -----------------------------------------------------

    def run(self, frame: np.ndarray, frame_counter: int) -> tuple[list[Detection], list]:
        """
        跑完这一帧该跑的所有路，合并结果。

        合并只做拼接，不做去重：
        两路模型框住同一个虫子时，那是两条独立信息，
        谁该被丢掉是任务决定，不是我能替它决定的。
        """

        now = time.perf_counter()

        self.detections = []
        self.states = []
        self.inferred = False

        for bundle in self.bundles:
            found, states = bundle.run(frame, frame_counter, now)

            if bundle.running and frame_counter % max(1, bundle.config.detect_every) == 0:
                self.inferred = True

            self.detections.extend(found)
            self.states.extend(states)

        return self.detections, self.states

    def targetable_detections(self) -> list[Detection]:
        """
        激光能看见的目标。

        只有显式标了 targetable 的那几路会出现在这里 ——
        接上"人"或"车牌"之后，这两路的框对激光是不可见的。
        """

        return [
            detection
            for bundle in self.bundles
            if bundle.config.targetable
            for detection in bundle.last_detections
        ]

    def snapshots(self) -> list[dict]:
        return [bundle.snapshot() for bundle in self.bundles]

    def total_infer_ms(self) -> float:
        return round(sum(b.avg_infer_ms() for b in self.bundles if b.running), 1)


# =========================================================
# 配置解析
#
# 放在这一层而不是 runtime.py，是因为 PredictConfig 属于这里，
# 而 runtime.py 不能反过来 import 本模块（会绕成环）。
# =========================================================

# 每路模型可以覆盖的字段。没写的沿用 ai 段的扁平默认值，
# 所以旧的 config.json 一个字不改也能继续跑 —— 只是变成"只有一路模型"。
_OVERRIDABLE = (
    "model_path",
    "device",
    "imgsz",
    "conf",
    "iou",
    "quantize",
    "detect_every",
    "max_det",
    "track_enabled",
    "tracker",
    "trail_frames",
    "classes",
)

_PREDICT_KEYS = ("predict_horizon_ms", "predict_window", "predict_min_samples", "predict_max_gap_s")


def model_configs_from(config: dict) -> list[tuple[DetectorConfig, PredictConfig]]:
    """
    config.json → 每路模型一份 (DetectorConfig, PredictConfig)。

    ai.models 缺失时退化成单路，键名和默认值全部沿用旧的扁平字段。
    """

    from .runtime import detector_config_from, predictor_from

    ai = config.get("ai") or {}
    base_detector = detector_config_from(config)
    base_predict = predictor_from(config)

    entries = ai.get("models")

    if not entries:
        # 旧的单模型写法：激光本来就围着这一路做的，保持可驱动。
        # 一旦改用 ai.models 显式列了多路，targetable 就变成每路自己点名，
        # 默认不可见 —— 免得哪天加一路"人"进去，激光就能对着人。
        base_detector.targetable = True
        return [(base_detector, base_predict)]

    groups = []

    for index, entry in enumerate(entries):

        if not isinstance(entry, dict):
            raise ValueError(f"ai.models[{index}] 不是对象")

        detector = DetectorConfig(**{
            **{k: getattr(base_detector, k) for k in _OVERRIDABLE},
            "enabled": bool(entry.get("enabled", True)),
            "key": str(entry.get("key") or f"model{index}"),
            "label": str(entry.get("label") or ""),
            "targetable": bool(entry.get("targetable", False)),
            **{k: entry[k] for k in _OVERRIDABLE if k in entry},
        })

        predict = PredictConfig(
            horizon_ms=float(entry.get("predict_horizon_ms", base_predict.horizon_ms)),
            window=int(entry.get("predict_window", base_predict.window)),
            min_samples=int(entry.get("predict_min_samples", base_predict.min_samples)),
            max_gap_s=float(entry.get("predict_max_gap_s", base_predict.max_gap_s)),
            enabled=bool(entry.get("predict_enabled", base_predict.enabled)),
        )

        groups.append((detector, predict))

    keys = [g[0].key for g in groups]

    if len(set(keys)) != len(keys):
        raise ValueError(f"ai.models 里的 key 重复：{keys}")

    return groups


def build_group(config: dict, root) -> ModelGroup:
    """按配置装配出模型组。这一步不加载权重，加载在 load_all()。"""

    bundles = []

    for detector_config, predict_config in model_configs_from(config):

        bundles.append(
            Bundle(
                config=detector_config,
                predict_config=predict_config,
                detector=Detector(detector_config, root=root),
                trails=Tracks(max_points=detector_config.trail_frames),
                predictor=MotionPredictor(
                    horizon_ms=predict_config.horizon_ms,
                    window=predict_config.window,
                    min_samples=predict_config.min_samples,
                    max_gap_s=predict_config.max_gap_s,
                    enabled=predict_config.enabled,
                ),
            )
        )

    return ModelGroup(bundles)
