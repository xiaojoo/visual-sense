"""
离线渲染 + HUD 文字。

V0.5 之前这里还管着 OpenCV 窗口和主循环。界面搬到浏览器之后
（app/server.py + app/web.py），窗口整条删掉了，只留两样
"必须画到像素上"的东西：

    create_status_frame   框 + 轨迹 + 预测线 + HUD 合成一张图
    resize_to_window      等比缩到目标尺寸

tools/check_detector.py 出图给人眼复核时用的就是这两个。
配置工厂和运行时开关在 app/runtime.py，不要加回这里。
"""

from __future__ import annotations

import cv2
import numpy as np

from .camera import Camera
from .detector import STATE_FAILED, STATE_LOADING, Detection, Detector
from .draw import draw_detections, draw_predictions, draw_trails
from .metrics import MetricsSnapshot
from .motion import MotionPredictor, TrackState
from .recorder import Recorder
from .tracks import Tracks


APP_NAME = "VisualSense"
APP_VERSION = "V0.5"

# HUD 排版参数
HUD_LINE_HEIGHT = 21
HUD_FIRST_BASELINE = 26
HUD_PADDING = 12

# HUD 配色（BGR）
TEXT_TITLE = (235, 235, 235)
TEXT_NORMAL = (220, 220, 220)
TEXT_DIM = (190, 190, 190)
TEXT_MUTED = (180, 180, 180)

COLOR_OK = (80, 220, 80)
COLOR_WARN = (80, 200, 255)
COLOR_ERROR = (80, 80, 255)

# 采集中：红色，且单独占一行，绝不可能看漏
COLOR_REC = (60, 60, 235)

def build_ai_lines(
    detector: Detector,
    snapshot: MetricsSnapshot,
    trails: Tracks,
    states: list[TrackState],
    predictor: MotionPredictor,
) -> list[tuple[str, tuple[int, int, int], float, int]]:
    """
    HUD 的 AI 区块。

    按检测器状态给出行数：

        ready  → 4~6 行（开跟踪/预测各多一行）
        loading→ 1 行
        off    → 1 行
        failed → 2 行
    """

    if detector.state == STATE_LOADING:

        return [
            (
                "AI        : LOADING",
                COLOR_WARN,
                0.58,
                1,
            ),
        ]

    if not detector.ready:

        if detector.state == STATE_FAILED:

            return [
                (
                    "AI        : FAILED",
                    COLOR_ERROR,
                    0.58,
                    1,
                ),
                (
                    f"AI Error  : "
                    f"{detector.last_error[:52]}",
                    TEXT_DIM,
                    0.53,
                    1,
                ),
            ]

        return [
            (
                "AI        : OFF  (press D)",
                TEXT_MUTED,
                0.58,
                1,
            ),
        ]

    lines = [
        (
            f"AI        : {detector.model_name}",
            TEXT_NORMAL,
            0.58,
            1,
        ),
        (
            f"AI Infer  : "
            f"{snapshot.avg_infer_ms:.1f} ms"
            f"  "
            f"[{snapshot.min_infer_ms:.1f}"
            f" ~ "
            f"{snapshot.max_infer_ms:.1f}]",
            COLOR_OK,
            0.58,
            1,
        ),
        (
            f"AI Obj    : "
            f"{detector.last_count}"
            f"   avg "
            f"{snapshot.avg_detections:.1f}"
            f"   "
            f"frames "
            f"{snapshot.inferred_frames}",
            COLOR_OK,
            0.58,
            1,
        ),
        (
            f"AI Cfg    : "
            f"{detector.device}"
            f"  imgsz="
            f"{detector.config.imgsz}"
            f"  conf="
            f"{detector.config.conf:.2f}"
            f"  "
            f"{detector.config.quantize or 'fp32'}"
            f"  every "
            f"{detector.config.detect_every}",
            TEXT_DIM,
            0.53,
            1,
        ),
    ]

    if detector.config.track_enabled:

        # 插在 Cfg 之前，让绿色的运行数据都排在灰色配置行上面
        lines.insert(
            len(lines) - 1,
            (
                f"AI Track  : "
                f"{detector.config.tracker}"
                f"   ids "
                f"{len(trails)}"
                f"   total "
                f"{trails.total_ids}"
                f"   trail "
                f"{detector.config.trail_frames}",
                COLOR_OK,
                0.58,
                1,
            ),
        )

    if predictor.enabled:

        lines.insert(
            len(lines) - 1,
            (
                f"AI Pred   : "
                f"{predictor.horizon_ms:.0f} ms"
                f"   ok "
                f"{len(states)}"
                f" / "
                f"{len(trails)}"
                f"   win "
                f"{predictor.window}"
                f"   min "
                f"{predictor.min_samples}",
                COLOR_OK,
                0.58,
                1,
            ),
        )

    return lines


def build_record_line(
    recorder: Recorder,
) -> list[tuple[str, tuple[int, int, int], float, int]]:
    """
    采集状态行。

    不在录的时候整行不出现，
    免得平时 HUD 上挂一个 REC 0/3000 让人误以为在写盘。
    """

    if not recorder.recording:
        return []

    return [
        (
            f"REC     : "
            f"{recorder.saved}"
            f" / "
            f"{recorder.config.max_frames}"
            f"   "
            f"{recorder.fps:.1f} fps"
            f"   "
            f"{recorder.elapsed_s:.0f} s"
            f"   stride "
            f"{recorder.config.stride}",
            COLOR_REC,
            0.58,
            1,
        ),
    ]


def build_hud_lines(
    camera: Camera,
    detector: Detector,
    config: dict,
    snapshot: MetricsSnapshot,
    trails: Tracks,
    states: list[TrackState],
    predictor: MotionPredictor,
    recorder: Recorder,
) -> list[tuple[str, tuple[int, int, int], float, int]]:
    """
    组装 HUD 全部文字行。

    返回 (文本, BGR 颜色, 字号, 线宽)。
    """

    camera_width, camera_height = camera.get_resolution()

    if camera_width > 0 and camera_height > 0:
        resolution_text = (
            f"{camera_width} x {camera_height}"
        )
    else:
        resolution_text = "--"

    if camera.connected:
        status_text = "CONNECTED"
        status_color = COLOR_OK
    else:
        status_text = "DISCONNECTED"
        status_color = COLOR_ERROR

    mode = str(
        config.get("camera_mode", "usb")
    ).upper()

    lines = [
        (
            f"{APP_NAME} {APP_VERSION}",
            TEXT_TITLE,
            0.75,
            2,
        ),
        (
            f"Camera    : {status_text}",
            status_color,
            0.62,
            1,
        ),
        (
            f"Mode      : {mode}",
            TEXT_NORMAL,
            0.58,
            1,
        ),
        (
            f"Resolution: {resolution_text}",
            TEXT_NORMAL,
            0.58,
            1,
        ),
        (
            f"FPS       : {snapshot.fps:.1f}",
            TEXT_NORMAL,
            0.58,
            1,
        ),
        (
            f"Interval  : "
            f"{snapshot.frame_interval_ms:.1f} ms",
            TEXT_NORMAL,
            0.58,
            1,
        ),
        (
            f"Read Avg  : "
            f"{snapshot.avg_read_ms:.1f} ms",
            TEXT_NORMAL,
            0.58,
            1,
        ),
        (
            f"Read Min  : "
            f"{snapshot.min_read_ms:.1f} ms",
            TEXT_DIM,
            0.53,
            1,
        ),
        (
            f"Read Max  : "
            f"{snapshot.max_read_ms:.1f} ms",
            TEXT_DIM,
            0.53,
            1,
        ),
    ]

    lines.extend(
        build_ai_lines(detector, snapshot, trails, states, predictor)
    )

    lines.extend(build_record_line(recorder))

    return lines



def create_status_frame(
    frame: np.ndarray | None,
    camera: Camera,
    detector: Detector,
    metrics: Metrics,
    config: dict,
    detections: list[Detection],
    trails: Tracks,
    states: list[TrackState],
    predictor: MotionPredictor,
    recorder: Recorder,
) -> np.ndarray:
    """
    创建带状态信息的视频画面。

    绘制顺序：

        视频帧
        ↓
        轨迹线
        ↓
        检测框
        ↓
        当前点 → 预测点
        ↓
        顶部 HUD
        ↓
        底部操作提示

    预测标记画在最上面：
    它本来就常常落在框外面，
    被框压住就看不出预测到哪了。
    """

    # =========================================================
    # 没有视频帧时创建黑色画布
    # =========================================================

    if frame is None:

        width = int(
            config.get(
                "window_width",
                1280,
            )
        )

        height = int(
            config.get(
                "window_height",
                720,
            )
        )

        canvas = np.zeros(
            (
                height,
                width,
                3,
            ),
            dtype=np.uint8,
        )

    else:

        canvas = frame.copy()

    height, width = canvas.shape[:2]

    # =========================================================
    # 轨迹 + 检测框
    # =========================================================

    if len(trails):
        draw_trails(canvas, trails)

    if detections:
        draw_detections(canvas, detections)

    if predictor.enabled and states:
        draw_predictions(canvas, states)

    # =========================================================
    # HUD 内容
    # =========================================================

    snapshot = metrics.snapshot()

    lines = build_hud_lines(
        camera=camera,
        detector=detector,
        config=config,
        snapshot=snapshot,
        trails=trails,
        states=states,
        predictor=predictor,
        recorder=recorder,
    )

    # =========================================================
    # 顶部状态区域
    #
    # 高度按行数算，
    # 开/关 AI 时面板不会留白或被截断。
    # =========================================================

    overlay_height = min(
        height,
        HUD_FIRST_BASELINE
        + HUD_LINE_HEIGHT * len(lines)
        + HUD_PADDING,
    )

    overlay = canvas[0:overlay_height, 0:width].copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, overlay_height),
        (20, 20, 20),
        -1,
    )

    canvas[0:overlay_height, 0:width] = cv2.addWeighted(
        overlay,
        0.78,
        canvas[0:overlay_height, 0:width],
        0.22,
        0,
    )

    # =========================================================
    # 绘制状态文字
    # =========================================================

    y = HUD_FIRST_BASELINE

    for text, color, scale, thickness in lines:

        cv2.putText(
            canvas,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        y += HUD_LINE_HEIGHT

    # =========================================================
    # 断开时显示错误
    # =========================================================

    if not camera.connected:

        error_text = camera.last_error

        if error_text:

            error_text = error_text[:100]

            cv2.putText(
                canvas,
                error_text,
                (
                    18,
                    max(
                        30,
                        height - 30,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (80, 80, 255),
                1,
                cv2.LINE_AA,
            )

    # =========================================================
    # 底部操作提示
    # =========================================================

    if height > 80:

        cv2.putText(
            canvas,
            "Q/ESC:Exit  D:AI  T:Track  P:Pred  R:Record",
            (
                18,
                height - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

    return canvas


def resize_to_window(
    frame: np.ndarray,
    window_width: int,
    window_height: int,
) -> np.ndarray:
    """
    将视频缩放到窗口大小。

    保持原始宽高比。

    不会强行拉伸视频。
    """

    height, width = frame.shape[:2]

    if (
        width <= 0
        or height <= 0
    ):
        return frame

    scale_x = (
        window_width
        / width
    )

    scale_y = (
        window_height
        / height
    )

    scale = min(
        scale_x,
        scale_y,
    )

    new_width = max(
        1,
        int(width * scale),
    )

    new_height = max(
        1,
        int(height * scale),
    )

    if (
        new_width == width
        and new_height == height
    ):
        return frame

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    return cv2.resize(
        frame,
        (
            new_width,
            new_height,
        ),
        interpolation=interpolation,
    )


