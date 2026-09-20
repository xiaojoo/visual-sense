"""
运行时装配层：config.json → 各个对象，以及运行时开关。

为什么从 main.py 里搬出来：
V0.5 之后主循环跑在后台线程，Web 层在另一个线程下发命令，
两边都要构造和切换这些对象。留在 main.py 里的话
pipeline 要 import main、main 又要 import pipeline，绕成环。

这里**不 import cv2**：
Web 层和激光状态机只需要这一层，不该被渲染依赖拖住。
"""

from __future__ import annotations

import json
from pathlib import Path

from .detector import Detector, DetectorConfig
from .motion import MotionPredictor
from .recorder import Recorder, RecorderConfig
from .tracks import Tracks


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / "config" / "config.json"


def load_config(root: Path | None = None) -> dict:
    path = config_path(root)

    if not path.exists():
        raise FileNotFoundError(
            "配置文件不存在:\n" + str(path)
            + "\n先复制一份改：copy config\\config.example.json config\\config.json"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_config(config: dict, root: Path | None = None) -> None:
    """
    写回 config.json。

    页面能改的只有几个数值开关，所以整份覆盖。
    先写临时文件再替换：直接 open("w") 之后如果序列化中途抛错，
    原文件已经被截断，配置就没了。
    """

    path = config_path(root)
    tmp = path.with_suffix(".json.tmp")

    with tmp.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)

    tmp.replace(path)


def detector_config_from(config: dict) -> DetectorConfig:
    """
    config.json 的 "ai" 段 → DetectorConfig。

    缺省即 enabled=False，所以旧配置文件不用改也能跑。
    """

    ai = config.get("ai") or {}

    # quantize 支持 null / "fp16" / "int8" / 16 等写法，
    # 空值统一归一成 None（FP32）。
    quantize = ai.get("quantize")

    return DetectorConfig(
        enabled=bool(ai.get("enabled", False)),
        model_path=str(ai.get("model_path", "weights/yolo11n.pt")),
        device=str(ai.get("device", "auto")),
        imgsz=int(ai.get("imgsz", 640)),
        conf=float(ai.get("conf", 0.25)),
        iou=float(ai.get("iou", 0.70)),
        quantize=str(quantize) if quantize else None,
        detect_every=max(1, int(ai.get("detect_every", 1))),
        max_det=int(ai.get("max_det", 100)),
        track_enabled=bool(ai.get("track_enabled", False)),
        tracker=str(ai.get("tracker", "bytetrack.yaml")),
        trail_frames=int(ai.get("trail_frames", 40)),
    )


def predictor_from(config: dict) -> MotionPredictor:
    """
    预测参数放在 ai 里而不是单独一段：
    没有检测就没有轨迹，没有轨迹就没有预测，它们是同一条链上的开关。
    """

    ai = config.get("ai") or {}

    return MotionPredictor(
        horizon_ms=float(ai.get("predict_horizon_ms", 100.0)),
        window=int(ai.get("predict_window", 6)),
        min_samples=int(ai.get("predict_min_samples", 4)),
        max_gap_s=float(ai.get("predict_max_gap_s", 0.5)),
        enabled=bool(ai.get("predict_enabled", False)),
    )


def recorder_config_from(config: dict) -> RecorderConfig:
    record = config.get("record") or {}

    return RecorderConfig(
        dataset_dir=str(record.get("dataset_dir", "dataset")),
        stride=max(1, int(record.get("stride", 5))),
        max_frames=int(record.get("max_frames", 3000)),
        jpeg_quality=int(record.get("jpeg_quality", 90)),
        auto_labels=bool(record.get("auto_labels", False)),
        save_preview=bool(record.get("save_preview", False)),
    )


# =========================================================
# 运行时开关
#
# 这几个函数原来绑在键盘事件上（D/T/P/R），
# 现在同样被 Web 的命令队列调用 —— 逻辑一行没改，
# 只是换了触发源。全部只在管线线程上调用，不加锁。
# =========================================================

def toggle_detector(detector: Detector) -> str:
    """
    运行时开关 AI。

    关掉时释放显存，再开会重新加载，
    方便在同一进程里对比开/关 AI 的 FPS。
    """

    if detector.ready:
        detector.unload()
        return "检测已关闭"

    from .detector import STATE_LOADING

    if detector.state == STATE_LOADING:
        return "模型正在加载，请稍候"

    if detector.load():
        return "检测已开启"

    return f"加载失败：{detector.last_error}"


def toggle_tracking(detector: Detector, trails: Tracks) -> str:
    """
    运行时切换 检测 / 跟踪。

    直接改 config.track_enabled：它是唯一的状态来源，
    读的一方和写的一方看的是同一个值，不会两边对不上。

    切换时必须清掉跟踪器状态和旧轨迹，
    否则关回来会接着上一段的运动历史猜。
    """

    detector.config.track_enabled = not detector.config.track_enabled
    detector.reset_tracking()
    trails.reset()

    return ("跟踪开启" if detector.config.track_enabled else "跟踪关闭") + f"（{detector.config.tracker}）"


def toggle_prediction(predictor: MotionPredictor) -> str:
    """
    不用清任何状态：拟合是每帧从轨迹里重算的，
    关掉只是不算，轨迹照旧累积，再开回来立刻就有结果。
    """

    predictor.enabled = not predictor.enabled

    return (
        f"预测{'开启' if predictor.enabled else '关闭'}"
        f"  {predictor.horizon_ms:.0f} ms  窗口 {predictor.window} 点"
    )


def toggle_record(recorder: Recorder, detector: Detector) -> str:
    """
    运行时开关采集。

    关掉才真正写 metadata 收尾；
    开着直接退出程序时，finally 里也会补一次，
    不然会话目录里会只剩一份 status=recording 的头信息。
    """

    if recorder.recording:
        summary = recorder.stop()
        return f"采集已停止，共 {recorder.saved} 帧" + (
            f"（{summary.get('session', '')}）" if summary else "（没有存下帧，未建目录）"
        )

    recorder.model_name = detector.model_name if detector.ready else ""
    recorder.class_names = detector.class_names if detector.ready else {}
    session = recorder.start()

    return f"采集已开始 → {session.name}" if session else "采集启动失败"
