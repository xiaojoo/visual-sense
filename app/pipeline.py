"""
V0.5 管线线程：摄像头 → 检测/跟踪/预测 → 采集 + 激光 + 推流。

和 V0.1~V0.4 的 main.py 主循环是同一套逻辑，
区别只有两处：

1. 渲染不再画文字 HUD —— 页面画得比 cv2.putText 好一万倍，
   这里只留框、轨迹、预测线。
2. 所有按键换成命令队列，由本线程在帧与帧之间消费，
   所以 detector / recorder / laser 依然只被一个线程碰。
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera, CameraConfig
from .detectors import build_group
from .draw import draw_detections, draw_predictions, draw_trails
from .hub import CommandQueue, Hub, Shutdown
from .laser import LaserConfig, LaserController
from .metrics import Metrics
from .recorder import Recorder
from .runtime import (
    recorder_config_from,
    toggle_detector,
    toggle_prediction,
    toggle_record,
    toggle_tracking,
)


@dataclass
class StreamConfig:
    """
    推流参数。

    桌面窗口时代不存在这个问题：帧就在本机内存里。
    换成局域网里的手机之后，每一帧都要编码 + 走 WiFi，
    所以必须能单独限宽和降质量。
    """

    max_width: int = 960
    jpeg_quality: int = 72
    every_n_frames: int = 1
    snapshot_every_s: float = 0.25


def stream_config_from(config: dict) -> StreamConfig:
    web = config.get("web") or {}
    stream = web.get("stream") or {}

    return StreamConfig(
        max_width=int(stream.get("max_width", 960)),
        jpeg_quality=int(stream.get("jpeg_quality", 72)),
        every_n_frames=max(1, int(stream.get("every_n_frames", 1))),
        snapshot_every_s=float(stream.get("snapshot_every_s", 0.25)),
    )


def laser_config_from(config: dict) -> LaserConfig:
    laser = config.get("laser") or {}

    return LaserConfig(
        enabled=bool(laser.get("enabled", False)),
        backend=str(laser.get("backend", "none")),
        max_dwell_s=float(laser.get("max_dwell_s", 1.5)),
        cooldown_s=float(laser.get("cooldown_s", 1.0)),
        watchdog_s=float(laser.get("watchdog_s", 3.0)),
        power_w=float(laser.get("power_w", 0.0)),
        aperture_mm=float(laser.get("aperture_mm", 0.0)),
    )


class Pipeline:
    def __init__(
        self,
        config: dict,
        hub: Hub,
        commands: CommandQueue,
        shutdown: Shutdown,
        root,
    ):
        self.config = config
        self.hub = hub
        self.commands = commands
        self.shutdown = shutdown
        self.root = root

        self.stream = stream_config_from(config)

        self.camera = Camera(
            CameraConfig(
                mode=str(config.get("camera_mode", "usb")).strip().lower(),
                index=int(config.get("camera_index", 0)),
                url=str(config.get("camera_url", "")).strip(),
                reconnect_interval=float(config.get("reconnect_interval", 2.0)),
            )
        )

        # V0.6：一路或多路。每路自带 Tracks 和 MotionPredictor，
        # 因为 track_id 只在单个跟踪器内唯一，两路共用会把
        # "蚊子#1"和"人#1"当成同一个目标。
        self.group = build_group(config, root)

        self.recorder = Recorder(
            recorder_config_from(config),
            root=root,
            camera_info={"mode": self.camera.config.mode, "url": self.camera.config.url},
        )
        self.laser = LaserController(laser_config_from(config))

        # 预标注按路分目录写，见 recorder._write_labels
        self.recorder.sources = [bundle.key for bundle in self.group]

        self.metrics = Metrics(sample_window=int(config.get("metrics_window", 120)))

        self._thread: threading.Thread | None = None
        self._started_at = time.perf_counter()
        self._last_snapshot_at = 0.0
        self._frame_counter = 0
        self._detections: list = []
        self._states: list = []
        # 锁的是 (来源, track_id)：两路模型都有 #1，只存数字就锁错东西
        self._laser_lock: tuple[str, int] | None = None
        self._last_frame: np.ndarray | None = None
        self._camera_error = ""
        self._note = ""
        self.inferred = False
        self._ever_connected = False
        self._attempt_logged = False
        self._last_fail_log = ""

    # =========================================================
    # 生命周期
    # =========================================================

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _run(self) -> None:
        # 先交一份快照再去做任何可能阻塞的事。
        # 摄像头 IP 不通时 cv2 的 connect 会卡十几秒，
        # 那期间页面不能是一片空白 —— 它得如实显示"未连接"。
        self.hub.publish_snapshot(self.snapshot())

        for note in self.group.load_all():
            self.hub.log(note)

        if self.laser.config.enabled:
            self.hub.log(f"[激光] {self.laser.connect()[1]}")

        try:
            self._loop()
        finally:
            if self.recorder.recording:
                self.recorder.stop()
            self.laser.disconnect()
            self.camera.disconnect()

    # =========================================================
    # 主循环
    # =========================================================

    def _loop(self) -> None:
        encode_index = 0

        while not self.shutdown.requested:

            self.commands.drain(self._handle)

            self._connect_if_needed()

            frame = self._read()

            if frame is not None:
                self._infer(frame)
                self._record(frame)

            self._aim_laser()

            if frame is not None and encode_index % self.stream.every_n_frames == 0:
                self.hub.publish_frame(self._encode(frame))
            encode_index += 1

            if time.perf_counter() - self._last_snapshot_at >= self.stream.snapshot_every_s:
                self._last_snapshot_at = time.perf_counter()
                self.hub.publish_snapshot(self.snapshot())

            self._note = ""

    def _connect_if_needed(self) -> None:
        """
        首连和重连走同一条路。

        原来 main.py 在进循环之前先 connect 一次，循环里只管重连，
        于是"连接中"这段时间整个进程是静止的。
        挪进循环之后，页面每 0.25 秒还能拿到一份新快照。
        """

        if self.camera.connected:
            return

        if not self.camera.should_reconnect():
            time.sleep(0.01)
            return

        # 每 2 秒重试一次，全写进日志的话两分钟就把真正的事件冲走了。
        # 所以只报"状态变了"：一次连接尝试一条，失败原因变了才再记一条。
        if not self._attempt_logged:
            self._attempt_logged = True
            self.hub.log("[摄像头] 正在连接…" if not self._ever_connected else "[摄像头] 尝试重新连接…")

        if self.camera.connect():
            self._ever_connected = True
            self._attempt_logged = False
            self._camera_error = ""
            self._last_fail_log = ""
            self.camera.set_buffer_size(int(self.config.get("buffer_size", 1)))
            self.metrics.reset()
            # 断流期间的空档会让运动历史失真，重连后必须让跟踪器重新起头
            self.group.reset_tracking()
            self.hub.log("[摄像头] 已连接")
            return

        self._camera_error = self.camera.last_error

        if self._last_fail_log != self._camera_error:
            self._last_fail_log = self._camera_error
            self.hub.log(f"[摄像头] 连接失败：{self._camera_error}")

    def _read(self) -> np.ndarray | None:
        if not self.camera.connected:
            return None

        ok, frame, read_ms = self.camera.read()

        if not (ok and frame is not None):
            return None

        self._last_frame = frame
        self._frame_counter += 1
        self.metrics.update(read_ms)

        return frame

    def _infer(self, frame: np.ndarray) -> None:
        """
        跑模型组。抽帧的那几路保留上一批框，画面不会一闪一闪。

        detect_every 按模型给，这是多模型能负担的关键：
        人不需 25 fps，蚊子需要。两路都全帧率就是两倍的推理时间。
        """

        if not self.group.any_running:
            self.inferred = False
            self._detections = []
            self._states = []
            return

        self._detections, self._states = self.group.run(frame, self._frame_counter)
        self.inferred = self.group.inferred

        if self.inferred:
            self.metrics.update_infer(self.group.total_infer_ms(), len(self._detections))

    def _record(self, frame: np.ndarray) -> None:
        """
        只存"这一帧真跑过推理"的画面。

        抽帧推理时，没跑的那几张要么配不上标签、
        要么会拿到上一帧的框，两种都会污染数据集。
        检测器没开时不推理，那就只存图。
        """

        if not self.recorder.recording:
            return

        if not (self.inferred or not self.group.any_running):
            return

        self.recorder.write(
            frame,
            self._detections if self.inferred else [],
            time.time(),
        )

    # =========================================================
    # 激光
    # =========================================================

    def _candidates(self) -> list:
        """
        激光能看见的目标。

        只有配置里显式标了 targetable 的那几路模型的框会出现在这里。
        默认不可见 —— 通用框架里"任何模型都能触发发射"意味着
        接上"人"这个模型的那天，激光就可能对着人。
        """

        return self.group.targetable_detections()

    def _as_target(self, detection) -> tuple[float, float, str, int]:
        height_px, width_px = self._frame_size()

        return (
            (detection.x1 + detection.x2) / 2 / max(1, width_px),
            (detection.y1 + detection.y2) / 2 / max(1, height_px),
            getattr(detection, "source", ""),
            int(getattr(detection, "track_id", -1)),
        )

    def _primary_target(self) -> tuple | None:
        """
        当前该打谁：可驱动的那几路里面积最大的目标。

        刻意不做"选最近的""选最快的" ——
        有多只的时候任何启发式选择都是没量过的猜测，
        而"最大"是唯一一个不需要理由的规则。
        页面上也允许手动锁定目标。
        """

        best = None

        for detection in self._candidates():
            area = (detection.x2 - detection.x1) * (detection.y2 - detection.y1)

            if best is None or area > best[0]:
                best = (area, self._as_target(detection))

        return None if best is None else best[1]

    def _target_of(self, source: str, track_id: int) -> tuple | None:
        """
        按 (来源, ID) 找目标。

        两路模型的 track_id 都从 1 开始，只比数字会把
        "蚊子#1"和"人#1"当成同一个东西。
        """

        for detection in self._candidates():

            if getattr(detection, "source", "") == source and int(
                getattr(detection, "track_id", -1)
            ) == track_id:
                return self._as_target(detection)

        return None

    def _aim_laser(self) -> None:
        """
        每帧喂给状态机的瞄准点。

        锁定某个目标时只认它 —— 目标一丢就交 None，
        状态机收到 None 会立刻收光，不会原地乱扫。
        """

        if self._laser_lock is not None:
            target = self._target_of(*self._laser_lock)
        else:
            target = self._primary_target()

        self.laser.tick(None if target is None else target[:2])

    def _aim_point(self) -> tuple[float, float] | None:
        """发射瞬间真正用的瞄准点：锁定目标优先，否则最大目标。"""

        if self._laser_lock is not None:
            locked = self._target_of(*self._laser_lock)
            return None if locked is None else locked[:2]

        target = self._primary_target()

        return None if target is None else target[:2]

    def _frame_size(self) -> tuple[int, int]:
        if self._last_frame is None:
            return 720, 960
        height, width = self._last_frame.shape[:2]
        return height, width

    # =========================================================
    # 渲染
    # =========================================================

    def _encode(self, frame: np.ndarray) -> bytes:
        canvas = frame.copy()

        height, width = canvas.shape[:2]

        if width > self.stream.max_width:
            scale = self.stream.max_width / width
            canvas = cv2.resize(
                canvas,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        # 轨迹和预测线按各路模型各画一次：它们各自是一套 Tracks，
        # 合并成一个 Tracks 会让两路的 #1 串成一条线。
        for bundle in self.group:
            if len(bundle.trails):
                draw_trails(canvas, bundle.trails)

            if bundle.predictor.enabled and bundle.last_states:
                draw_predictions(canvas, bundle.last_states)

        if self._detections:
            draw_detections(canvas, self._detections, with_label=False)

        if self.recorder.recording:
            cv2.circle(canvas, (26, 26), 11, (60, 60, 235), -1, cv2.LINE_AA)

        ok, buffer = cv2.imencode(
            ".jpg",
            canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.stream.jpeg_quality],
        )

        return buffer.tobytes() if ok else b""

    # =========================================================
    # 命令
    # =========================================================

    def _handle(self, name: str, args: dict) -> dict:
        """
        返回 {"ok", "message"}，不返回裸字符串。

        成败必须由代码分支决定，不能由页面上的文案决定 ——
        "加载失败"里含"失败"两个字这种判断，改一个字就错了。
        """

        if name in ("toggle_ai", "toggle_track", "toggle_predict"):
            ok, message = True, self._toggle_model(name, args)
        elif name == "toggle_record":
            ok, message = True, toggle_record(self.recorder, self.primary.detector)
        elif name == "laser":
            ok, message = self._laser_command(args.get("action", ""), args)
        elif name == "clear_log":
            self.hub.clear_log()
            ok, message = True, "日志已清空"
            self.hub.publish_snapshot(self.snapshot())
            return {"ok": ok, "message": message}
        elif name == "stop":
            self.shutdown.request()
            ok, message = True, "正在停止"
        else:
            ok, message = False, f"未知命令：{name}"

        self._note = message
        self.hub.log(message)

        # 立刻补发一份快照。
        # 快照平时每 0.25 秒才发布一次，不补的话
        # 页面点完"解锁"拿回来的还是点之前的状态，
        # 看起来就像按钮没反应。
        self.hub.publish_snapshot(self.snapshot())

        return {"ok": ok, "message": message}

    def _toggle_model(self, name: str, args: dict) -> str:
        """
        开关一类能力。

        带 key 就只动那一路；不带就动全部，并在回执里说清是"全部"——
        顶部那几个 chip 是全组快捷方式，不写清楚会让人以为只切了一个。
        """

        key = args.get("key")
        targets = [b for b in self.group if b.key == key] if key else list(self.group)

        if key and not targets:
            return f"没有这一路模型：{key}"

        verb = {"toggle_ai": "检测", "toggle_track": "跟踪", "toggle_predict": "预测"}[name]
        scope = targets[0].name if key else "全部"
        notes = []

        for bundle in targets:
            if name == "toggle_ai":
                note = toggle_detector(bundle.detector)
            elif name == "toggle_track":
                note = toggle_tracking(bundle.detector, bundle.trails)
            else:
                note = toggle_prediction(bundle.predictor)
            notes.append(note)

        return f"[{scope}] " + "；".join(notes)

    @property
    def primary(self):
        """第一路模型。采集器记权重名和类别表时用它 —— 多路时这是已知局限。"""

        return self.group.bundles[0]

    def _laser_command(self, action: str, args: dict) -> tuple[bool, str]:
        """
        全部只在管线线程执行，所以状态机不需要锁。

        每个分支都返回一句"人话"，页面直接显示 ——
        点了按钮却没反应，是这类控制台最难查的 bug。
        """

        if action == "connect":
            return self.laser.connect()

        if action == "arm":
            return self.laser.arm()

        if action == "disarm":
            self._laser_lock = None
            return self.laser.disarm()

        if action == "fire":
            ok, message = self.laser.fire(self._aim_point())

            if ok:
                self.laser.heartbeat()

            return ok, message

        if action == "stop":
            self._laser_lock = None
            return self.laser.stop()

        if action == "reset":
            return self.laser.reset()

        if action == "lock":
            return self._lock_target(args.get("track_id"), args.get("source"))

        if action == "heartbeat":
            self.laser.heartbeat()
            return True, "心跳已收到"

        return False, f"未知激光动作：{action}"

    def _lock_target(self, track_id, source=None) -> tuple[bool, str]:
        """
        锁定某个 ID。

        锁定的意义是别在两只虫交错的时候甩过去打另一只 ——
        不锁的话瞄准点永远是"当前最大"，交叉瞬间会跳。
        """

        try:
            wanted = int(track_id)
        except (TypeError, ValueError):
            self._laser_lock = None
            return True, "已取消锁定，瞄准点回到最大目标"

        if wanted < 0:
            return False, "该目标还没有稳定跟踪 ID，锁定无意义"

        origin = str(source or "")

        if self._target_of(origin, wanted) is None:
            return False, f"目标 {origin}#{wanted} 不可锁定：不在可驱动的那几路里，或已离开画面"

        self._laser_lock = (origin, wanted)

        return True, f"已锁定目标 {origin}#{wanted}"

    # =========================================================
    # 快照
    # =========================================================

    def targets(self) -> list[dict]:
        """
        页面上的目标列表，多路模型合并成一张表。

        坐标一律归一化：画面在浏览器里会被缩放，
        像素坐标换过去就对不上，比例永远正确。
        source 必须带上 —— 两路模型的 track_id 都会从 1 开始。
        """

        height_px, width_px = self._frame_size()

        # (来源, track_id) → 运动状态。两路模型的 #1 不是一回事，键必须成对
        by_track = {}

        for bundle in self.group:
            for state in bundle.last_states:
                by_track[(bundle.key, int(getattr(state, "track_id", -1)))] = state

        lock = self._laser_lock

        rows = []

        for detection in self._detections:

            source = getattr(detection, "source", "")
            track_id = int(getattr(detection, "track_id", -1))
            width = detection.x2 - detection.x1
            height = detection.y2 - detection.y1
            state = by_track.get((source, track_id))
            bundle = self.group.get(source)

            rows.append(
                {
                    "source": source,
                    "label": bundle.name if bundle else source,
                    "targetable": bool(bundle and bundle.config.targetable),
                    "track_id": track_id,
                    "class": detection.class_name,
                    "conf": round(float(detection.confidence), 3),
                    "x": round(detection.x1 / max(1, width_px), 4),
                    "y": round(detection.y1 / max(1, height_px), 4),
                    "w": round(width / max(1, width_px), 4),
                    "h": round(height / max(1, height_px), 4),
                    "short_px": round(min(width, height), 1),
                    "speed_px_s": round(state.speed, 1) if state else None,
                    "heading_deg": round(math.degrees(state.heading), 1) if state else None,
                    "predicted": (
                        [
                            round(state.predicted_x / max(1, width_px), 4),
                            round(state.predicted_y / max(1, height_px), 4),
                        ]
                        if state
                        else None
                    ),
                    "locked": lock is not None and lock == (source, track_id),
                }
            )

        return sorted(rows, key=lambda row: -row["short_px"])

    def snapshot(self) -> dict:
        metrics = self.metrics.snapshot()

        return {
            "uptime_s": round(time.perf_counter() - self._started_at, 1),
            "frame": self._frame_counter,
            "note": self._note,
            "camera": {
                "connected": self.camera.connected,
                "mode": self.camera.config.mode,
                "url": getattr(self.camera.config, "url", ""),
                "resolution": list(self.camera.get_resolution()),
                "error": self._camera_error or self.camera.last_error,
            },
            "metrics": {
                "fps": round(metrics.fps, 1),
                "frame_interval_ms": round(metrics.frame_interval_ms, 1),
                "read_ms": round(metrics.read_time_ms, 1),
                "read_avg_ms": round(metrics.avg_read_ms, 1),
                "read_min_ms": round(metrics.min_read_ms, 1),
                "read_max_ms": round(metrics.max_read_ms, 1),
                "infer_ms": round(metrics.avg_infer_ms, 1),
                "infer_min_ms": round(metrics.min_infer_ms, 1),
                "infer_max_ms": round(metrics.max_infer_ms, 1),
                "avg_detections": round(metrics.avg_detections, 2),
                "frames": metrics.frames,
            },
            "ai": {
                "count": len(self.group.bundles),
                "running": sum(1 for b in self.group if b.running),
                "models": self.group.snapshots(),
                "infer_total_ms": self.group.total_infer_ms(),
                "tracks": sum(len(b.trails) for b in self.group),
                "any_tracking": any(b.config.track_enabled for b in self.group),
                "any_predicting": any(b.predictor.enabled for b in self.group),
                "any_ready": self.group.any_running,
                "targetable": [b.name for b in self.group if b.config.targetable],
            },
            "record": {
                "recording": self.recorder.recording,
                "saved": self.recorder.saved,
                "session": self.recorder.session_dir.name if self.recorder.session_dir else "",
                "seconds": round(self.recorder.elapsed_s, 1),
                "stride": self.recorder.config.stride,
                "dropped_boxes": self.recorder.clipped_dropped,
            },
            "laser": {
                **self.laser.snapshot(),
                "locked_id": self._laser_lock[1] if self._laser_lock else None,
                "locked_source": self._laser_lock[0] if self._laser_lock else None,
            },
            "targets": self.targets(),
            "stream": {
                "max_width": self.stream.max_width,
                "jpeg_quality": self.stream.jpeg_quality,
                "frame_age_s": round(self.hub.frame_age_s, 2),
            },
        }
