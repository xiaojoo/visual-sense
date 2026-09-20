from __future__ import annotations

import math
from dataclasses import dataclass

from .tracks import TrackPoint, Tracks


@dataclass
class TrackState:
    """
    某个 ID 在这一刻的运动状态。

    x / y 是拟合直线在当前时刻的取值，
    不是最后一个检测框的中心 —— 这是有意的：
    检测框本身在抖，直接拿最后一个点当"当前位置"，
    预测起点就会跟着噪声跳。

    predicted_x / predicted_y 是同一条直线外推
    horizon_ms 之后的位置。
    """

    track_id: int

    x: float
    y: float

    vx: float
    vy: float

    predicted_x: float
    predicted_y: float

    samples: int

    horizon_ms: float

    @property
    def speed(self) -> float:
        """
        合速度，px/s。
        """

        return math.hypot(self.vx, self.vy)

    @property
    def heading(self) -> float:
        """
        运动方向，弧度。0 = 向右，逆时针为正。
        """

        return math.atan2(-self.vy, self.vx)


@dataclass
class LineFit:
    """
    一条最小二乘拟合出来的直线轨迹。

    保存"在 at 时刻位于 (x, y)、速度 (vx, vy)"，
    于是任意时刻的位置都能算出来：

        position(t) = (x + vx*(t-at), y + vy*(t-at))

    之所以不直接返回四个数：调用方常常要在**多个**时刻上求值
    （标注传递就是在一个时间跨度上逐帧求值），
    每次重算一遍拟合是浪费，也容易在别处写错外推公式。
    """

    x: float

    y: float

    vx: float

    vy: float

    at: float

    samples: int

    span: tuple[float, float] = (0.0, 0.0)

    # "s" = 时间戳是秒，速度单位 px/s；
    # "frame" = 没有时间戳，只能按帧序号拟合，速度是 px/帧。
    # 标出来是因为这两者差一个帧率，混用不会报错，只会算错。
    unit: str = "s"

    def at_time(self, timestamp: float) -> tuple[float, float]:
        delta = timestamp - self.at

        return (self.x + self.vx * delta, self.y + self.vy * delta)

    def inside_span(self, timestamp: float, tolerance: float = 0.0) -> bool:
        """
        这个时刻是否落在参与拟合的观测范围内。

        内插可信，外推不可信：
        一个目标只被跟到第 5~12 帧，
        拿这条直线去猜第 40 帧的位置就是在编数据。
        """

        return self.span[0] - tolerance <= timestamp <= self.span[1] + tolerance


class MotionPredictor:
    """
    V0.4-A：匀速（Constant Velocity）预测。

    Trajectory Buffer → 最小二乘拟合 → 速度 → 外推

    刻意不用这些东西：

        相邻两帧差分  → 框一抖速度就跳成 350 px/s
        卡尔曼        → 要先有过程噪声/观测噪声的实测值才能调，
                        现在连蚊子长什么样都还没有
        LSTM / 光流   → 需要数据，而数据是 V0.4-B 的事

    拟合窗口内所有点这条直线，同时给出两样东西：
    平滑后的当前位置，和速度。
    抖动被平均掉，不需要额外的低通滤波器。

    窗口长度是量出来的，不是随手填的：
    在周期 1.67 秒的合成往复运动上，
    4~6 点比"原地不动"基线好 36~45%，
    拉到 15 点反而输 8~13% ——
    窗口比运动周期长，拟合出的速度就被平均掉了，
    还带相位滞后。蚊子这种目标只能用短窗口。
    """

    def __init__(
        self,
        horizon_ms: float = 100.0,
        window: int = 6,
        min_samples: int = 4,
        max_gap_s: float = 0.5,
        enabled: bool = False,
    ):
        self.horizon_ms = float(horizon_ms)

        self.window = max(2, window)

        self.min_samples = max(2, min_samples)

        self.max_gap_s = float(max_gap_s)

        # 运行时开关（P 键）。
        # 拟合本身不贵，但关掉之后画面回到 V0.3 的样子，
        # 方便对比"预测点是不是真的稳"。
        self.enabled = enabled

    # =========================================================
    # Fit
    # =========================================================

    @staticmethod
    def fit(
        points: list[TrackPoint],
    ) -> LineFit | None:
        """
        对一段连续轨迹做最小二乘直线拟合。

        时间轴平移到最后一个点（dt <= 0）再拟合，
        否则 t 是 perf_counter 的绝对值（1e9 量级），
        法方程里 Σt² 会大到吃掉有效数字。
        """

        if len(points) < 2:
            return None

        now = points[-1].timestamp

        # 判据必须是"时间戳在往前走"，
        # 不能是"所有时间戳都大于 0"：
        # 第一段轨迹的起始时间戳常常就是 0.0，
        # 那样会被误判成没有时间戳、退化成按帧号拟合，
        # 速度单位悄悄变成 px/帧，乘上秒数后预测点几乎不动。
        use_time = points[-1].timestamp > points[0].timestamp

        if use_time:
            times = [point.timestamp - now for point in points]
        else:
            # 没有时间戳就退回按帧序号拟合，
            # 此时速度单位是 px/帧，不是 px/s
            times = [
                float(point.frame_index - points[-1].frame_index)
                for point in points
            ]

        count = len(points)

        sum_t = 0.0
        sum_tt = 0.0
        sum_x = 0.0
        sum_y = 0.0
        sum_tx = 0.0
        sum_ty = 0.0

        for time, point in zip(times, points):

            sum_t += time
            sum_tt += time * time
            sum_x += point.x
            sum_y += point.y
            sum_tx += time * point.x
            sum_ty += time * point.y

        denominator = count * sum_tt - sum_t * sum_t

        if abs(denominator) < 1e-9:
            # 所有点挤在同一时刻，斜率无意义
            return None

        vx = (count * sum_tx - sum_t * sum_x) / denominator
        vy = (count * sum_ty - sum_t * sum_y) / denominator

        x_now = (sum_x - vx * sum_t) / count
        y_now = (sum_y - vy * sum_t) / count

        return LineFit(
            x=x_now,
            y=y_now,
            vx=vx,
            vy=vy,
            at=now,
            samples=count,
            span=(points[0].timestamp, points[-1].timestamp),
            unit="s" if use_time else "frame",
        )

    # =========================================================
    # Estimate
    # =========================================================

    def estimate(
        self,
        track_id: int,
        points: list[TrackPoint],
        horizon_ms: float | None = None,
    ) -> TrackState | None:
        """
        给一个 ID 算运动状态。

        点不够、或者拟合不出斜率，就返回 None：
        宁可不画预测点，也不要画一个乱跳的点。
        """

        horizon = self.horizon_ms if horizon_ms is None else horizon_ms

        if len(points) < self.min_samples:
            return None

        fitted = self.fit(points)

        if fitted is None:
            return None

        if fitted.unit != "s":

            # 没有真实时间戳就不能按毫秒外推，
            # 硬算会把 px/帧 当成 px/s，预测点几乎不动
            return None

        predicted_x, predicted_y = fitted.at_time(
            fitted.at + horizon / 1000.0
        )

        return TrackState(
            track_id=track_id,
            x=fitted.x,
            y=fitted.y,
            vx=fitted.vx,
            vy=fitted.vy,
            predicted_x=predicted_x,
            predicted_y=predicted_y,
            samples=fitted.samples,
            horizon_ms=horizon,
        )

    def estimate_all(
        self,
        trails: Tracks,
    ) -> list[TrackState]:
        """
        对当前所有轨迹算一遍。
        """

        states: list[TrackState] = []

        for track_id in trails.active_ids():

            points = trails.recent(
                track_id,
                window=self.window,
                max_gap_s=self.max_gap_s,
            )

            state = self.estimate(track_id, points)

            if state is not None:
                states.append(state)

        return states
