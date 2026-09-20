from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection
from .motion import TrackState
from .tracks import Tracks


# BGR 调色板，按 class_id 取色
PALETTE = [
    (80, 220, 80),
    (255, 200, 60),
    (60, 200, 255),
    (230, 120, 255),
    (255, 130, 100),
    (140, 255, 140),
    (255, 255, 255),
    (200, 160, 90),
]


def box_color(key: int) -> tuple[int, int, int]:
    return PALETTE[abs(int(key)) % len(PALETTE)]


def build_label(detection: Detection) -> str:
    """
    框上那行字。

    带 track_id 时把 ID 放在最前面，
    这样同一个目标跨帧扫过去，
    能一眼看出 ID 有没有换人。
    """

    track_id = getattr(detection, "track_id", None)

    prefix = f"#{track_id} " if track_id is not None and track_id >= 0 else ""

    return (
        f"{prefix}"
        f"{detection.class_name} "
        f"{detection.confidence:.2f}"
    )


def draw_trails(
    canvas: np.ndarray,
    trails: Tracks,
    thickness: int = 2,
) -> None:
    """
    画每个 ID 的历史轨迹。

    先画轨迹再画框，
    框压在线上面，读图时不会把线和框的边缘混在一起。

    颜色按 ID 取，和框一致。
    """

    for track_id in trails.active_ids():

        points = trails.points(track_id)

        if len(points) < 2:
            continue

        color = box_color(track_id)

        previous = points[0]

        for point in points[1:]:

            cv2.line(
                canvas,
                (int(previous.x), int(previous.y)),
                (int(point.x), int(point.y)),
                color,
                thickness,
                cv2.LINE_AA,
            )

            previous = point

        head = points[-1]

        cv2.circle(
            canvas,
            (int(head.x), int(head.y)),
            max(2, thickness + 1),
            color,
            -1,
        )


def draw_predictions(
    canvas: np.ndarray,
    states: list[TrackState],
    thickness: int = 2,
) -> None:
    """
    V0.4-A：当前位置 → 预测位置。

    实心点是拟合出来的当前位置，
    不是检测框中心 —— 两者不重合正是平滑在起作用。

    末端画空心圈，表示这个点是猜的。
    颜色跟同一个 ID 的框和轨迹保持一致，
    三只蚊子的时候能一眼看出哪条预测属于谁。

    刻意不给每个目标标速度文字：
    九个目标一起画的时候文字会叠成一团，反而什么都读不出来。
    线长本身就是速度，数字在 HUD 上有汇总。
    """

    if not states:
        return

    height = canvas.shape[0]

    radius = max(3, int(height / 180))

    for state in states:

        color = box_color(state.track_id)

        current = (int(round(state.x)), int(round(state.y)))

        predicted = (
            int(round(state.predicted_x)),
            int(round(state.predicted_y)),
        )

        cv2.line(canvas, current, predicted, color, thickness, cv2.LINE_AA)

        cv2.circle(canvas, current, radius, color, -1, cv2.LINE_AA)

        cv2.circle(
            canvas,
            predicted,
            radius * 2,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_detections(
    canvas: np.ndarray,
    detections: list[Detection],
    line_thickness: int = 2,
    with_label: bool = True,
) -> None:
    """
    在画面上绘制检测框。

    坐标直接使用帧像素坐标，
    因此必须在 resize 到窗口之前调用，
    缩放会由主程序的 resize_to_window 一并完成。

    with_label=False 给 V0.5 的推流用：
    网页自己在帧上叠一层可点击的框并标 ID，
    再烧一份 cv2 的文字进去就是两份互相打架的标签。
    """

    if not detections:
        return

    height = canvas.shape[0]

    # 文字大小随分辨率变化，低分屏不会糊成一片
    scale = max(0.42, min(0.9, height / 720.0 * 0.55))
    text_thickness = max(1, int(round(scale * 1.8)))
    padding = 3

    for detection in detections:

        x1 = int(round(detection.x1))
        y1 = int(round(detection.y1))
        x2 = int(round(detection.x2))
        y2 = int(round(detection.y2))

        # 有 ID 就按 ID 取色，让同一只目标跨帧颜色不变
        track_id = getattr(detection, "track_id", None)

        if track_id is not None and track_id >= 0:
            color = box_color(track_id)
        else:
            color = box_color(detection.class_id)

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            color,
            line_thickness,
            cv2.LINE_AA,
        )

        if not with_label:
            continue

        label = build_label(detection)

        (text_w, text_h), baseline = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                text_thickness,
            )
        )

        # 标签默认放在框上方，贴边时翻到框内侧
        top_y = y1 - padding

        if top_y - text_h - baseline < 0:
            top_y = y1 + text_h + baseline + padding

        cv2.rectangle(
            canvas,
            (x1, top_y - text_h - baseline),
            (x1 + text_w + padding * 2, top_y + padding),
            color,
            -1,
        )

        cv2.putText(
            canvas,
            label,
            (x1 + padding, top_y - padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (15, 15, 15),
            text_thickness,
            cv2.LINE_AA,
        )
