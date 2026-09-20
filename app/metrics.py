from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class MetricsSnapshot:
    fps: float
    frame_interval_ms: float
    read_time_ms: float

    min_read_ms: float
    max_read_ms: float
    avg_read_ms: float

    frames: int

    # V0.2 推理统计
    avg_infer_ms: float
    min_infer_ms: float
    max_infer_ms: float

    inferred_frames: int
    detections_total: int

    avg_detections: float


class Metrics:
    """
    视频性能统计。

    重点观察：

        FPS
        Frame Interval
        read() 时间

    V0.2 起额外统计推理侧：

        detect() 耗时
        推理帧数
        检测框数量

    读帧和推理分开计数，
    才能区分"网络慢"还是"模型慢"。
    """

    def __init__(
        self,
        sample_window: int = 120,
    ):
        self.sample_window = max(10, sample_window)

        self.frame_times = deque(
            maxlen=self.sample_window
        )

        self.read_times = deque(
            maxlen=self.sample_window
        )

        self.infer_times = deque(
            maxlen=self.sample_window
        )

        self.last_frame_time: float | None = None

        self.total_frames = 0

        self.inferred_frames = 0
        self.detections_total = 0

    def reset(self) -> None:
        self.frame_times.clear()
        self.read_times.clear()
        self.infer_times.clear()

        self.last_frame_time = None

        self.total_frames = 0

        self.inferred_frames = 0
        self.detections_total = 0

    def update(
        self,
        read_time_ms: float,
    ) -> None:
        now = time.perf_counter()

        if self.last_frame_time is not None:
            interval_ms = (
                now - self.last_frame_time
            ) * 1000.0

            self.frame_times.append(interval_ms)

        self.last_frame_time = now

        self.read_times.append(read_time_ms)

        self.total_frames += 1

    def update_infer(
        self,
        infer_time_ms: float,
        detections: int,
    ) -> None:
        """
        记录一次推理。

        只在真正跑了 detect() 的帧上调用，
        所以 detect_every>1 时这里统计的仍是单帧耗时，
        而不是摊到每一帧的平均值。
        """

        self.infer_times.append(infer_time_ms)

        self.inferred_frames += 1

        self.detections_total += max(0, int(detections))

    @property
    def fps(self) -> float:
        """
        根据最近帧间隔计算 FPS。
        """

        if not self.frame_times:
            return 0.0

        avg_interval = (
            sum(self.frame_times)
            / len(self.frame_times)
        )

        if avg_interval <= 0:
            return 0.0

        return 1000.0 / avg_interval

    @property
    def frame_interval_ms(self) -> float:
        if not self.frame_times:
            return 0.0

        return (
            sum(self.frame_times)
            / len(self.frame_times)
        )

    @property
    def avg_read_ms(self) -> float:
        if not self.read_times:
            return 0.0

        return (
            sum(self.read_times)
            / len(self.read_times)
        )

    @property
    def min_read_ms(self) -> float:
        if not self.read_times:
            return 0.0

        return min(self.read_times)

    @property
    def max_read_ms(self) -> float:
        if not self.read_times:
            return 0.0

        return max(self.read_times)

    @property
    def avg_infer_ms(self) -> float:
        if not self.infer_times:
            return 0.0

        return (
            sum(self.infer_times)
            / len(self.infer_times)
        )

    @property
    def min_infer_ms(self) -> float:
        if not self.infer_times:
            return 0.0

        return min(self.infer_times)

    @property
    def max_infer_ms(self) -> float:
        if not self.infer_times:
            return 0.0

        return max(self.infer_times)

    @property
    def avg_detections(self) -> float:
        """
        每个推理帧的平均检测框数。
        """

        if self.inferred_frames <= 0:
            return 0.0

        return (
            self.detections_total
            / self.inferred_frames
        )

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            fps=self.fps,
            frame_interval_ms=self.frame_interval_ms,
            read_time_ms=self.avg_read_ms,
            min_read_ms=self.min_read_ms,
            max_read_ms=self.max_read_ms,
            avg_read_ms=self.avg_read_ms,
            frames=self.total_frames,
            avg_infer_ms=self.avg_infer_ms,
            min_infer_ms=self.min_infer_ms,
            max_infer_ms=self.max_infer_ms,
            inferred_frames=self.inferred_frames,
            detections_total=self.detections_total,
            avg_detections=self.avg_detections,
        )