from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .detector import Detection


@dataclass
class TrackPoint:
    """
    某个 ID 在第 frame_index 帧的中心位置。

    同时存帧号和时间戳：

        帧号差  → 丢帧时才是真实的间隔
        时间差  → 算 px/s 只能用真实经过的秒数

    只用帧号会把速度算成"像素每帧"，
    帧率一变（10 fps 和 25 fps）同一个数就代表不同速度。
    """

    frame_index: int
    x: float
    y: float

    timestamp: float = 0.0


class Tracks:
    """
    V0.3 轨迹存储。

    track_id → 最近的若干中心点。

    只做两件事：

        1. 把每帧的跟踪结果按 ID 归档，
           供画轨迹和后面算速度用。
        2. 目标消失后保留一段时间再丢弃。

    保留期是必要的：跟踪器短暂跟丢（蚊子被遮挡、检测掉一帧）
    之后会拿回同一个 ID，如果轨迹被立刻清空，
    屏幕上就会出现一条断成两截的路径。
    """

    def __init__(
        self,
        max_points: int = 40,
        max_lost_frames: int = 60,
    ):
        self.max_points = max(2, max_points)

        self.max_lost_frames = max(0, max_lost_frames)

        self._trails: dict[int, deque[TrackPoint]] = {}

        self._lost: dict[int, int] = {}

        # 建过多少条轨迹。
        # 同一个 ID 被丢弃后重新出现会再记一次，
        # 所以这里是"轨迹段数"，不是"目标个数"。
        self.total_ids = 0

    # =========================================================
    # Update
    # =========================================================

    def update(
        self,
        tracks: list[Detection],
        frame_index: int,
        timestamp: float = 0.0,
    ) -> None:
        """
        归档一帧的跟踪结果。

        没有 track_id 的目标（跟踪未开启、
        或这一帧没被关联上）直接跳过。
        """

        seen: set[int] = set()

        for track in tracks:

            track_id = getattr(track, "track_id", -1)

            if track_id is None or track_id < 0:
                continue

            track_id = int(track_id)

            seen.add(track_id)

            trail = self._trails.get(track_id)

            if trail is None:

                trail = deque(maxlen=self.max_points)

                self._trails[track_id] = trail

                self.total_ids += 1

            center_x, center_y = track.center

            trail.append(
                TrackPoint(
                    frame_index=frame_index,
                    x=center_x,
                    y=center_y,
                    timestamp=timestamp,
                )
            )

            self._lost[track_id] = 0

        for track_id in list(self._trails):

            if track_id in seen:
                continue

            self._lost[track_id] = self._lost.get(track_id, 0) + 1

            if self._lost[track_id] > self.max_lost_frames:

                del self._trails[track_id]

                self._lost.pop(track_id, None)

    def reset(self) -> None:
        self._trails.clear()
        self._lost.clear()
        self.total_ids = 0

    # =========================================================
    # Read
    # =========================================================

    def active_ids(self) -> list[int]:
        return sorted(self._trails)

    def points(
        self,
        track_id: int,
    ) -> list[TrackPoint]:
        return list(self._trails.get(track_id, ()))

    def recent(
        self,
        track_id: int,
        window: int = 6,
        max_gap_s: float = 0.5,
    ) -> list[TrackPoint]:
        """
        取最近一段**连续**的轨迹点，最多 window 个。

        连续是指相邻点之间的时间差不超过 max_gap_s。
        跟丢过一次的轨迹不能整条拿去拟合：
        中间断掉的那段会让最小二乘把两个位置的差
        摊成一条匀速直线，预测点会指到半空中。
        """

        points = self._trails.get(track_id)

        if not points:
            return []

        segment = [points[-1]]

        for index in range(len(points) - 2, -1, -1):

            current = points[index]

            ahead = points[index + 1]

            if (
                current.timestamp > 0.0
                and ahead.timestamp > 0.0
                and ahead.timestamp - current.timestamp > max_gap_s
            ):
                break

            segment.append(current)

            if len(segment) >= window:
                break

        segment.reverse()

        return segment

    def is_stale(
        self,
        track_id: int,
    ) -> bool:
        """
        这条轨迹当前没被跟到，
        只是还在保留期内。
        """

        return self._lost.get(track_id, 0) > 0

    def __len__(self) -> int:
        return len(self._trails)
