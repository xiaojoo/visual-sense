"""
轻量标注工具：在采集帧上拖框，直接写 YOLO 格式。

    .venv\\Scripts\\python.exe -m tools.label_frames                # 全部会话
    .venv\\Scripts\\python.exe -m tools.label_frames --session 0    # 指定会话
    .venv\\Scripts\\python.exe -m tools.label_frames --names mosquito,fly

只需要 opencv + numpy，**不用 .venv-ai**，
标注重在快和准，不需要模型在场。

数据约定（重要）：

    labels/<帧>.txt   = 人工真值。这个工具是唯一写它的东西。
    annotations.jsonl = 模型当时说了什么（含 track_id），只读。

    所以覆盖 labels/ 不会丢信息 —— 模型的原话一直在
    annotations.jsonl 里，随时能对比"人改了什么"。

按键：

    左键拖拽      画框
    A / D 或 ← →  上一帧 / 下一帧
    1 2 3 ...     切换类别
    U             撤销最后一个框
    X             删掉鼠标所在的框
    N             标记这帧没有目标（写空文件，负样本）
    P             按 track_id 把本帧的框传递到相邻帧
    M             显示 / 隐藏模型原框（对照用）
    滚轮          以光标为中心缩放
    右键拖拽      平移
    0             复位缩放
    S             保存
    Q / ESC       退出

蚊子只有十几像素，不缩放根本没法标，
所以缩放平移是必备功能而不是加分项。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.motion import MotionPredictor  # noqa: E402
from app.tracks import TrackPoint  # noqa: E402

WIN = "label"

MIN_BOX_PX = 2


@dataclass
class Box:
    """
    像素坐标框。
    """

    x1: float
    y1: float
    x2: float
    y2: float

    cls: int = 0

    def normalized(self) -> tuple[float, float, float, float]:
        """
        保证 x1<x2、y1<y2 —— 反向拖拽要能纠正。
        """

        return (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def contains(self, x: float, y: float) -> bool:
        left, top, right, bottom = self.normalized()

        return left <= x <= right and top <= y <= bottom

    def clipped(self, width: int, height: int) -> "Box | None":
        """
        裁到画面内；退化成一一线段就返回 None。

        在落框那一刻就裁，而不是存盘时才裁：
        否则屏幕上画的是大框、重开看到的是小框，
        标注的人无法相信自己眼睛。
        """

        left, top, right, bottom = self.normalized()

        left = max(left, 0.0)

        top = max(top, 0.0)

        right = min(right, float(width))

        bottom = min(bottom, float(height))

        if right - left < MIN_BOX_PX or bottom - top < MIN_BOX_PX:

            return None

        return Box(left, top, right, bottom, self.cls)


# =========================================================
# 纯逻辑：坐标换算
# =========================================================

@dataclass
class View:
    """
    图像坐标 <-> 屏幕坐标。

    缩放以"图像坐标"为基准保存，
    换算时才乘窗口尺寸，
    这样窗口大小变了不会让已画的框漂移。
    """

    scale: float = 1.0

    offset_x: float = 0.0

    offset_y: float = 0.0

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(round(x * self.scale + self.offset_x)),
            int(round(y * self.scale + self.offset_y)),
        )

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        if self.scale == 0:
            return (0.0, 0.0)

        return (
            (x - self.offset_x) / self.scale,
            (y - self.offset_y) / self.scale,
        )

    def zoom_at(self, screen_x: float, screen_y: float, factor: float) -> "View":
        """
        以光标为不动点缩放。

        先算出光标对应的图像点，缩放后重新把那个点放回
        光标原来的屏幕位置 —— 否则每次滚轮都会把目标甩出视野。
        """

        anchor_x, anchor_y = self.to_image(screen_x, screen_y)

        scale = min(max(self.scale * factor, 0.2), 40.0)

        return View(
            scale=scale,
            offset_x=screen_x - anchor_x * scale,
            offset_y=screen_y - anchor_y * scale,
        )


def to_yolo(
    box: Box,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """
    像素框 -> YOLO 归一化，并裁到画面内。

    裁剪而不是夹中心点：夹中心会把贴边目标整体挪位，
    和 app/recorder.py 保持同一套几何约定。
    """

    left, top, right, bottom = box.normalized()

    left = max(left, 0.0)

    top = max(top, 0.0)

    right = min(right, float(width))

    bottom = min(bottom, float(height))

    if right - left <= 0.0 or bottom - top <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)

    return (
        ((left + right) / 2.0) / width,
        ((top + bottom) / 2.0) / height,
        (right - left) / width,
        (bottom - top) / height,
    )


def from_yolo(
    line: str,
    width: int,
    height: int,
) -> Box | None:
    """
    一行 YOLO 标签 -> 像素框。读不懂返回 None。
    """

    parts = line.split()

    if len(parts) < 5:
        return None

    try:
        cls = int(parts[0])
        cx, cy, w, h = (float(value) for value in parts[1:5])
    except ValueError:
        return None

    if w <= 0.0 or h <= 0.0:
        return None

    box_w = w * width

    box_h = h * height

    return Box(
        x1=cx * width - box_w / 2.0,
        y1=cy * height - box_h / 2.0,
        x2=cx * width + box_w / 2.0,
        y2=cy * height + box_h / 2.0,
        cls=cls,
    )


def format_yolo(box: Box, width: int, height: int) -> str:
    cx, cy, w, h = to_yolo(box, width, height)

    return f"{box.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# =========================================================
# 纯逻辑：按 track_id 传递框
# =========================================================

def find_track(
    box: Box,
    model_boxes: list[dict],
    tolerance: float = 0.75,
) -> int | None:
    """
    人画的框对应模型跟踪里的哪个 ID。

    取中心点最近、且距离不超过框对角线 tolerance 倍的那个。

    用相对距离而不是绝对像素：
    蚊子框十几像素、人框可能偏几像素，
    绝对阈值要么太松要么太紧。
    """

    left, top, right, bottom = box.normalized()

    center_x = (left + right) / 2.0

    center_y = (top + bottom) / 2.0

    reach = max(8.0, ((right - left) ** 2 + (bottom - top) ** 2) ** 0.5)

    reach *= tolerance

    best_id: int | None = None

    best_distance = float("inf")

    for entry in model_boxes:

        track_id = entry.get("track_id")

        if track_id is None:
            continue

        x1, y1, x2, y2 = entry["bbox_px"]

        distance = (
            ((x1 + x2) / 2.0 - center_x) ** 2
            + ((y1 + y2) / 2.0 - center_y) ** 2
        ) ** 0.5

        if distance <= reach and distance < best_distance:

            best_distance = distance

            best_id = int(track_id)

    return best_id


class TrackBook:
    """
    整个会话的模型轨迹索引 + 滑窗平滑。

    为什么要平滑：ByteTrack 自己的框在相邻帧之间有实测抖动，
    直接照它的位移搬，等于把这份抖动写进人工真值
    （实测：吸附法传出去的轨迹偏离直线 0.73px 中位 / 1.47px 最大，
    和模型观测本身一模一样，就是 1:1 复制）。
    同一条结论现在由 check_propagate_window 用合成轨迹钉住。

    为什么是滑窗而不是整段一条直线：
    整段拟合只能表达匀速，蚊子急转、悬停、变向时
    它会给出一个"太平滑"因而错误的位置。
    滑窗在跟随真实运动和压制抖动之间取折中。

    窗口以**目标帧**为中心，不是以源帧为中心：
    以源帧为中心的窗口去外推远处的帧，
    比它要解决的问题更糟。
    """

    def __init__(self, session: Session, window: int = 4, min_points: int = 3):
        self.frames = session.model

        self.window = max(2, window)

        self.min_points = max(2, min_points)

        self.use_time = (
            len(session.times) == len(session.model)
            and len(session.times) >= 2
            and session.times[-1] > session.times[0]
        )

        self.times: list[float] = [
            (session.times[index] if self.use_time else float(index))
            for index in range(len(session.model))
        ]

        self.series: dict[int, list[TrackPoint]] = {}

        for index, detections in enumerate(session.model):

            for entry in detections:

                track_id = entry.get("track_id")

                if track_id is None:
                    continue

                x1, y1, x2, y2 = entry["bbox_px"]

                self.series.setdefault(int(track_id), []).append(
                    TrackPoint(
                        frame_index=index,
                        x=(x1 + x2) / 2.0,
                        y=(y1 + y2) / 2.0,
                        timestamp=self.times[index],
                    )
                )

        self._cache: dict[tuple[int, int], tuple[float, float] | None] = {}

    def _window_points(
        self,
        track_id: int,
        index: int,
    ) -> list[TrackPoint]:
        """
        以 index 为中心取窗口内的观测点。

        不够 min_points 就向两侧扩，
        因为 2 个点定一条线没有任何平滑可言 ——
        那等于直接把抖动搬过去。
        """

        points = self.series.get(track_id) or []

        if not points:
            return []

        half = max(1, self.window // 2)

        while True:

            selected = [
                point
                for point in points
                if abs(point.frame_index - index) <= half
            ]

            if len(selected) >= self.min_points:
                return selected

            # 扩到能凑够点数，或者覆盖整条轨迹为止
            if half > len(points):
                return selected

            half += 1

    def smoothed(
        self,
        track_id: int,
        index: int,
    ) -> tuple[float, float] | None:
        """
        这个 ID 在第 index 帧的平滑中心。

        邻域内做最小二乘直线拟合，取该时刻直线上的位置。
        相邻帧的窗口高度重叠，所以结果本身是连续的。
        """

        key = (track_id, index)

        if key in self._cache:
            return self._cache[key]

        selected = self._window_points(track_id, index)

        result = None

        if len(selected) >= 2:

            line = MotionPredictor.fit(selected)

            if line is not None and line.inside_span(self.times[index]):

                result = line.at_time(self.times[index])

        self._cache[key] = result

        return result

    def observed(self, track_id: int) -> int:
        return len(self.series.get(track_id) or [])


def transfer(
    boxes: list[Box],
    source: list[dict],
    book: TrackBook,
    source_index: int,
    target_index: int,
    width: int,
    height: int,
) -> tuple[list[Box], list[str]]:
    """
    把一帧上的人工框搬到另一帧。

    搬的是**滑窗平滑后的位移**：该 ID 在两帧上的平滑位置之差，
    人工框就跟着走多少。

    三条理由：

    - 人框的大小和形状保持不变。直接用目标帧的模型框，
      等于把模型的误差写进真值。
    - 位移来自邻域拟合而不是相邻两帧之差，
      所以不会把跟踪器的逐帧抖动传进标注结果。
    - 用滑窗而不是整段一条直线，才能跟随真实的转向。

    目标帧附近没有观测就不搬：平滑位置是从观测里算出来的，
    跟踪器在那一帧根本没看到目标时，搬过去的是猜的。
    """

    moved: list[Box] = []

    notes: list[str] = []

    for box in boxes:

        track_id = find_track(box, source)

        if track_id is None:

            notes.append("找不到对应的跟踪 ID")

            continue

        if book.observed(track_id) < 2:

            notes.append(
                f"ID {track_id} 只有 "
                f"{book.observed(track_id)} 个观测点，无法拟合"
            )

            continue

        source_position = book.smoothed(track_id, source_index)

        target_position = book.smoothed(track_id, target_index)

        if source_position is None or target_position is None:

            notes.append(f"ID {track_id} 在目标帧附近没有观测")

            continue

        dx = target_position[0] - source_position[0]

        dy = target_position[1] - source_position[1]

        moved_box = Box(
            x1=box.x1 + dx,
            y1=box.y1 + dy,
            x2=box.x2 + dx,
            y2=box.y2 + dy,
            cls=box.cls,
        ).clipped(width, height)

        if moved_box is None:

            notes.append(f"ID {track_id} 搬过去就出画面了")

            continue

        moved.append(moved_box)

    return moved, notes


# =========================================================
# 会话读写
# =========================================================

@dataclass
class Session:
    """
    一个采集会话：帧、模型记录、人工标签。
    """

    root: Path

    frames: list[Path] = field(default_factory=list)

    model: list[list[dict]] = field(default_factory=list)

    # 每帧的墙钟时间戳，拟合轨迹要用。
    # 没有它就只能按帧号拟合，速度单位会变成 px/帧
    times: list[float] = field(default_factory=list)

    size: list[int] = field(default_factory=lambda: [0, 0])

    @property
    def label_dir(self) -> Path:
        return self.root / "labels"

    @classmethod
    def open(cls, root: Path) -> "Session | None":
        """
        annotations.jsonl 是权威清单：
        它记了哪些帧真的落盘了，按它读顺序才不会错。
        """

        images_dir = root / "images"

        if not images_dir.is_dir():
            return None

        annotations = root / "annotations.jsonl"

        records: list[dict] = []

        if annotations.exists():

            for line in annotations.read_text(
                encoding="utf-8"
            ).splitlines():

                line = line.strip()

                if line:

                    try:

                        records.append(json.loads(line))

                    except json.JSONDecodeError:

                        continue

        if records:

            frames = []

            model = []

            times: list[float] = []

            for record in records:

                # 崩溃时最后一行可能是写全的 JSON 但缺字段，
                # 也可能指向一张没落盘的图，两种都要跳过
                name = record.get("file")

                if not name:
                    continue

                path = root / name

                if not path.exists():
                    continue

                frames.append(path)

                model.append(record.get("detections", []))

                times.append(float(record.get("timestamp") or 0.0))

        else:

            frames = sorted(
                path
                for path in images_dir.iterdir()
                if path.suffix.lower() in (".jpg", ".jpeg", ".png")
            )

            model = [[] for _ in frames]

            times = [0.0 for _ in frames]

        if not frames:
            return None

        size = None

        for record in records:

            # size 记在每帧记录上，不在单个检测里
            found = record.get("size")

            if found and len(found) == 2 and all(found):

                size = list(found)

                break

        if size is None:

            probe = cv2.imread(str(frames[0]))

            size = (
                [0, 0]
                if probe is None
                else [probe.shape[1], probe.shape[0]]
            )

        return cls(
            root=root,
            frames=frames,
            model=model,
            times=times,
            size=size,
        )

    def boxes(self, index: int) -> list[Box]:
        """
        已有人工标签。
        """

        path = self.label_path(index)

        if not path.exists():
            return []

        return [
            box
            for box in (
                from_yolo(line, self.width, self.height)
                for line in path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
            if box is not None
        ]

    def save(self, index: int, boxes: list[Box]) -> None:
        """
        写人工真值。空列表也写空文件 —— 那是负样本标记。
        """

        self.label_dir.mkdir(parents=True, exist_ok=True)

        body = "\n".join(
            format_yolo(box, self.width, self.height) for box in boxes
        )

        self.label_path(index).write_text(
            body + ("\n" if body else ""),
            encoding="utf-8",
        )

    def label_path(self, index: int) -> Path:
        return self.label_dir / f"{self.frames[index].stem}.txt"

    @property
    def width(self) -> int:
        return self.size[0] or 1

    @property
    def height(self) -> int:
        return self.size[1] or 1


# =========================================================
# 自检：P 为什么是滑窗
# =========================================================

WAVE_FRAMES = 150

WAVE_FPS = 30.0

# 周期不能再短：轨迹自身弯曲会在二阶差分里和抖动混成一团，
# 那时候「滑窗压掉了抖动」这条就量不出来了。
# 2.5s 时弯曲项约 0.5px，抖动项 1.65px，两个量级才分得开。
WAVE_PERIOD = 2.5

WAVE_RADIUS_X = 60.0

WAVE_RADIUS_Y = 40.0

# x、y 同频率不同相位 —— 椭圆，一段一直在转弯的轨迹
WAVE_PHASE = 0.9

# 顶替 ByteTrack 的框抖动；量级取自实测（观测二阶差分中位 1.65px）
WAVE_JITTER = 0.9

WAVE_TRACK_ID = 1


def _wave_session() -> Session:
    """
    合成一段一直在转弯的轨迹：x、y 同频率不同相位，合起来是一个椭圆，
    再叠 ±0.9px 抖动。

    种子固定，所以数字每次都一样 —— 自检变红只可能是拟合变了，
    不是随机数变了。

    椭圆取代了最初那轮扫描用的正弦片段：同样能把「整段一条直线」
    和「滑窗」分开，但转弯更缓，弯曲项不会盖过抖动项。
    """

    rng = random.Random(20260920)

    model: list[list[dict]] = []

    times: list[float] = []

    omega = 2.0 * math.pi / WAVE_PERIOD

    for index in range(WAVE_FRAMES):

        moment = index / WAVE_FPS

        times.append(moment)

        x = 480.0 + WAVE_RADIUS_X * math.sin(omega * moment)

        y = 260.0 + WAVE_RADIUS_Y * math.sin(omega * moment + WAVE_PHASE)

        bx = x + rng.uniform(-WAVE_JITTER, WAVE_JITTER)

        by = y + rng.uniform(-WAVE_JITTER, WAVE_JITTER)

        model.append(
            [
                {
                    "track_id": WAVE_TRACK_ID,
                    "class_id": 0,
                    "confidence": 0.5,
                    "bbox_px": [bx - 12.0, by - 6.0, bx + 12.0, by + 6.0],
                }
            ]
        )

    return Session(root=Path("."), model=model, times=times)


def _second_differences(
    points: list[tuple[float, float]],
) -> list[float]:
    return [
        math.hypot(
            points[i + 1][0] - 2 * points[i][0] + points[i - 1][0],
            points[i + 1][1] - 2 * points[i][1] + points[i - 1][1],
        )
        for i in range(1, len(points) - 1)
    ]


def check_propagate_window(window: int = 4) -> list[str]:
    """
    跑在每次启动前的无头自检，不开窗口、不碰数据集，几毫秒。

    它钉住的是当初那轮窗口扫描的全部结论：

      整段一条直线在转弯处偏得很远 —— 滑窗存在的理由
      滑窗把抖动压掉了一大半 —— 不吸附模型框的理由
      窗口放大到 20 帧偏差就爬回来 —— 默认取小不取大的理由
      窗口缩到 3 帧又变抖 —— 默认停在拐点而不是贴着下界

    偏差 = 传递结果偏离跟踪器观测多少 px；
    粗糙度 = 二阶差分中位，即还剩多少抖动写进真值。
    和 README 里那张表同一个定义、同一组数：
    --verbose 会把这张表打出来。

    窗口 N 在 TrackBook 里是以目标帧为中心 ±N//2 帧的邻域，
    所以 4 实际取到 5 个观测点。
    """

    session = _wave_session()

    observed = [
        (point.x, point.y)
        for point in TrackBook(session, window=window).series[WAVE_TRACK_ID]
    ]

    span = range(2, len(observed) - 2)

    def measure(size: int) -> tuple[float, float]:

        book = TrackBook(session, window=size)

        centres: list[tuple[float, float]] = []

        for index in span:

            position = book.smoothed(WAVE_TRACK_ID, index)

            if position is None:
                raise AssertionError(f"自检轨迹第 {index} 帧拟合不出来")

            centres.append(position)

        deviation = [
            math.hypot(a[0] - b[0], a[1] - b[1])
            for a, b in zip(centres, observed[span.start:])
        ]

        return (
            statistics.median(deviation),
            statistics.median(_second_differences(centres)),
        )

    # 窗口比轨迹还长 = 一整条轨迹拟合一条直线
    whole_bias, whole_rough = measure(WAVE_FRAMES * 2)

    sizes = sorted({3, window, 6, 20})

    results = {size: measure(size) for size in sizes}

    win_bias, win_rough = results[window]

    wide_bias = results[20][0]

    # 吸附就是观测本身：偏差恒为 0，粗糙度就是它自带的抖动
    snap_rough = statistics.median(
        _second_differences(observed[span.start : -2])
    )

    report = [
        f"  整段一条直线  偏差 {whole_bias:6.2f}px  粗糙度 {whole_rough:5.2f}px",
    ]

    report += [
        f"  滑窗 {size:>2} 帧   偏差 {results[size][0]:6.2f}px"
        f"  粗糙度 {results[size][1]:5.2f}px"
        + ("   <- 默认" if size == window else "")
        for size in sizes
    ]

    report.append(
        f"  逐帧吸附      偏差   0.00px  粗糙度 {snap_rough:5.2f}px"
    )

    def require(ok: bool, why: str) -> None:

        if ok:
            return

        raise AssertionError(
            "P 的传递方式不再满足滑窗的前提：\n"
            + "\n".join(report)
            + f"\n  -> {why}"
        )

    require(
        whole_bias >= 6 * win_bias,
        "整段拟合的偏差没有远大于滑窗，滑窗白留了",
    )

    require(
        whole_rough <= win_rough,
        "整段拟合不再是更平滑的那个，粗糙度这个指标坏了",
    )

    require(
        win_rough <= 0.6 * snap_rough,
        "滑窗没把抖动压下去，那就该直接吸附模型框",
    )

    require(
        win_bias <= snap_rough,
        "滑窗换来的偏差已经不小于它消掉的抖动，不划算",
    )

    require(
        win_bias <= wide_bias,
        "窗口放大偏差随之上升这条关系断了，拐点位置要重测",
    )

    require(
        win_rough <= results[3][1],
        "再缩小窗口反而更平滑，默认的 4 帧不是拐点了",
    )

    return report


# =========================================================
# 命令行
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集帧标注工具",
    )

    parser.add_argument(
        "--dataset",
        default="dataset",
        help="采集根目录",
    )

    parser.add_argument(
        "--session",
        type=int,
        default=-1,
        help="打开第几个会话，默认 0",
    )

    parser.add_argument(
        "--names",
        default="mosquito",
        help="类别名，逗号分隔；数字键切换",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="起始帧号",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="只列出会话和标注进度，不开窗口",
    )

    parser.add_argument(
        "--propagate-window",
        type=int,
        default=4,
        help="P 传递时的滑窗帧数（邻域 ±N//2 帧）。拐点在 4："
             "抖动 1.65->0.43px，偏差只花 0.67px；"
             "20 帧平滑度几乎不涨，偏差却爬到 6.27px。"
             "自检恒按默认 4 帧跑，--verbose 打印整张表",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印 P 传递自检表，并把每个鼠标和按键事件打出来，"
             "用来确认事件有没有到窗口",
    )

    return parser.parse_args()


def discover_sessions(dataset: Path) -> list[Path]:
    """
    找会话目录。
    """

    if (dataset / "images").is_dir():
        return [dataset]

    return sorted(
        child
        for child in dataset.iterdir()
        if (child / "images").is_dir()
    )


def main() -> int:
    args = parse_args()

    # 恒用默认窗口自检，不跟着 --propagate-window 走：
    # 换窗口大小是用法，不是这套传递逻辑成不成立的前提。
    numbers = check_propagate_window()

    if args.verbose:

        print("P 传递自检:")

        print("\n".join(numbers))

    dataset = Path(args.dataset)

    if not dataset.is_absolute():
        dataset = ROOT / dataset

    if not dataset.is_dir():

        print(f"[ERROR] 没有 {dataset}，先在窗口里按 R 采集")

        return 1

    sessions = discover_sessions(dataset)

    if not sessions:

        print(f"[ERROR] {dataset} 下没有会话目录")

        return 1

    if args.list:

        print(f"{'会话':<20} {'帧':>5} {'已标':>5} {'负样本':>7} {'待标':>5}")

        print("-" * 48)

        for position, root in enumerate(sessions):

            session = Session.open(root)

            if session is None:

                print(f"{root.name:<20}  打不开")

                continue

            done = 0

            negative = 0

            for index in range(len(session.frames)):

                path = session.label_path(index)

                if not path.exists():

                    continue

                done += 1

                if not path.read_text(encoding="utf-8").strip():

                    negative += 1

            print(
                f"{position:2d} {root.name:<17} "
                f"{len(session.frames):>5} "
                f"{done:>5} "
                f"{negative:>7} "
                f"{len(session.frames) - done:>5}"
            )

        return 0

    index = max(0, min(args.session, len(sessions) - 1))

    session = Session.open(sessions[index])

    if session is None:

        print(f"[ERROR] 会话打不开：{sessions[index]}")

        return 1

    names = [
        item.strip() for item in args.names.split(",") if item.strip()
    ] or ["class0"]

    print(f"会话 {sessions[index].name}  {len(session.frames)} 帧  "
          f"{session.width}x{session.height}")
    print(f"类别: {list(enumerate(names))}")
    print(f"传递滑窗: {args.propagate_window} 帧")
    print("左键拖框  A/D 翻帧  P 按ID传递  N 无目标  S 保存  Q 退出  滚轮缩放")

    try:
        run(
            session,
            names,
            max(0, args.start),
            args.verbose,
            args.propagate_window,
        )

    except KeyboardInterrupt:

        print("\n中断退出")

    return 0


# =========================================================
# GUI
# =========================================================

def run(
    session: Session,
    names: list[str],
    start: int,
    verbose: bool = False,
    window: int = 6,
) -> None:
    """
    交互层。

    刻意做薄：所有算数（换算、传递、读写格式）都在上面的
    纯函数里，那部分有离线测试。
    这里只管窗口、鼠标和按键。

    verbose 存在的原因很实际：OpenCV 的 HighGUI 只在窗口
    自己有焦点时收键盘事件，"我按了没反应"和"按键没送到"
    从屏幕上看完全一样，只能靠日志区分。
    """

    def log(message: str) -> None:

        if verbose:

            print(message, flush=True)

    state = {
        "index": min(start, len(session.frames) - 1),
        "boxes": [],
        "dirty": False,
        "cls": 0,
        "show_model": True,
        "view": View(),
        "drag": None,
        "pan": None,
        "mouse": (0, 0),
    }

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):

        image_x, image_y = state["view"].to_image(x, y)

        # 自己记光标位置：这个 OpenCV 构建里没有
        # cv2.getMousePosition，而"删鼠标所在的框"
        # 不能退化成"删画面中心那个框"——
        # 那样按下去删掉的是别的东西，还看起来一切正常。
        state["mouse"] = (x, y)

        # 移动事件太密，只记按键和滚轮，
        # 否则日志会淹掉真正有用的那条
        if event != cv2.EVENT_MOUSEMOVE:

            log(
                f"mouse event={event} screen=({x},{y}) "
                f"image=({image_x:.1f},{image_y:.1f}) flags={flags}"
            )

        if event == cv2.EVENT_LBUTTONDOWN:

            state["drag"] = (image_x, image_y)

        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:

            state["drag"] = state["drag"][0:2] + (image_x, image_y)

        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:

            start_x, start_y = state["drag"][:2]

            box = Box(start_x, start_y, image_x, image_y, state["cls"]).clipped(
                session.width,
                session.height,
            )

            if box is not None:

                state["boxes"].append(box)

                state["dirty"] = True

                log(f"  -> 新框 {box}")

            else:

                log(
                    f"  -> 丢弃：太小或完全在画面外 "
                    f"({start_x:.0f},{start_y:.0f})-({image_x:.0f},{image_y:.0f})"
                )

            state["drag"] = None

        elif event == cv2.EVENT_RBUTTONDOWN:

            state["pan"] = (x, y, state["view"].offset_x, state["view"].offset_y)

        elif event == cv2.EVENT_RBUTTONUP:

            state["pan"] = None

        elif event == cv2.EVENT_MOUSEMOVE and state["pan"]:

            origin_x, origin_y, off_x, off_y = state["pan"]

            state["view"] = View(
                scale=state["view"].scale,
                offset_x=off_x + (x - origin_x),
                offset_y=off_y + (y - origin_y),
            )

        elif event == cv2.EVENT_MOUSEWHEEL:

            factor = 1.25 if flags > 0 else 0.8

            state["view"] = state["view"].zoom_at(x, y, factor)

    cv2.setMouseCallback(WIN, on_mouse)

    def load(index: int) -> None:
        """
        切帧，先自动保存上一帧。

        画完的框因为顺手按了一下 D 就静默消失，
        是标注工具最不能接受的失败方式 ——
        脏标记只有眼睛盯着才看得见，而人不会一直盯着。
        """

        if state["dirty"]:

            save()

        state["index"] = index

        state["boxes"] = session.boxes(index)

        state["dirty"] = False

        state["drag"] = None

    def save() -> None:

        session.save(state["index"], state["boxes"])

        state["dirty"] = False

        print(f"  保存 {session.frames[state['index']].name}  "
              f"{len(state['boxes'])} 个框")

    def propagate() -> None:
        """
        把本帧的框按拟合轨迹搬到前后各 30 帧。

        只覆盖"还没有人工标签"的帧：
        已经标过的地方不能被机器改动，
        否则标一百帧的结果可能被一次误触抹掉。
        """

        index = state["index"]

        boxes = list(state["boxes"])

        if not boxes:

            print("  本帧没有框，没什么可传递")

            return

        book = TrackBook(session, window=window)

        touched = 0

        reasons: dict[str, int] = {}

        for step in (-1, 1):
            for offset in range(1, 31):

                target = index + step * offset

                if target < 0 or target >= len(session.frames):
                    break

                if session.boxes(target):
                    continue

                moved, notes = transfer(
                    boxes,
                    session.model[index],
                    book,
                    index,
                    target,
                    session.width,
                    session.height,
                )

                if not moved:

                    # 不再一遇错就停下整侧：
                    # 位移来自整段拟合，目标中途没跟到
                    # 不影响它两头的位置
                    for note in notes:

                        reasons[note] = reasons.get(note, 0) + 1

                    continue

                session.save(target, moved)

                touched += 1

        save()

        print(f"  已传递 {touched} 帧（只写空白的帧）")

        for reason, count in sorted(
            reasons.items(),
            key=lambda item: -item[1],
        ):

            print(f"     跳过 {count:3d} 次：{reason}")

    load(state["index"])

    last_key = -1

    last_key_at = 0.0

    # 按住不放时 Windows 会以每秒二三十次的速率重发同一个键。
    # 翻帧重复无所谓，但 U（撤销）连发会把一帧的框全清空，
    # X 连发会把整帧删光，N 连发是反复覆写。
    # 所以破坏性键做一次去抖，导航键不限制，
    # 不然快速连按 D 翻帧会跟着变卡。
    DESTRUCTIVE = {
        ord("u"), ord("U"),
        ord("x"), ord("X"),
        ord("n"), ord("N"),
    }

    REPEAT_MS = 180

    while True:

        # 标题栏的 X 会直接销毁窗口。
        # 不检测的话下一次 imshow 抛异常，
        # 当前帧刚画完的框就跟着进程一起没了。
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:

            log("窗口被关闭，保存当前帧后退出")

            if state["dirty"]:
                save()

            break

        frame = cv2.imread(str(session.frames[state["index"]]))

        if frame is None:

            print(f"[ERROR] 读不出 {session.frames[state['index']]}")

            break

        view: View = state["view"]

        canvas = cv2.copyMakeBorder(
            frame,
            1,
            1,
            1,
            1,
            cv2.BORDER_CONSTANT,
            value=(40, 40, 40),
        )

        canvas = cv2.resize(
            canvas,
            None,
            fx=view.scale,
            fy=view.scale,
            interpolation=(
                cv2.INTER_NEAREST
                if view.scale > 3.0
                else cv2.INTER_AREA
            ),
        )

        def screen(x: float, y: float) -> tuple[int, int]:
            return (
                int(round(x * view.scale + view.offset_x + 1)),
                int(round(y * view.scale + view.offset_y + 1)),
            )

        if state["show_model"]:

            for entry in session.model[state["index"]]:

                x1, y1, x2, y2 = entry["bbox_px"]

                cv2.rectangle(
                    canvas,
                    screen(x1, y1),
                    screen(x2, y2),
                    (200, 200, 200),
                    1,
                )

        for box in state["boxes"]:

            left, top, right, bottom = box.normalized()

            color = (80, 220, 80) if box.cls == 0 else (255, 200, 60)

            cv2.rectangle(
                canvas,
                screen(left, top),
                screen(right, bottom),
                color,
                2,
            )

            cv2.putText(
                canvas,
                names[box.cls] if box.cls < len(names) else str(box.cls),
                (screen(left, top)[0], max(12, screen(left, top)[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        if state["drag"] and len(state["drag"]) == 4:

            start_x, start_y, now_x, now_y = state["drag"]

            cv2.rectangle(
                canvas,
                screen(start_x, start_y),
                screen(now_x, now_y),
                (80, 220, 80),
                1,
            )

        bar = (
            f"{state['index'] + 1}/{len(session.frames)}  "
            f"cls={names[state['cls']]}  "
            f"boxes={len(state['boxes'])}  "
            f"{'*' if state['dirty'] else ''}  "
            f"zoom={view.scale:.1f}x  "
            f"{'model' if state['show_model'] else 'human'}"
        )

        cv2.putText(
            canvas,
            bar,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
        )

        cv2.putText(
            canvas,
            bar,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
        )

        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(20) & 0xFF

        now_ms = time.perf_counter() * 1000.0

        if (
            key != 255
            and key in DESTRUCTIVE
            and key == last_key
            and now_ms - last_key_at < REPEAT_MS
        ):

            # 同一个破坏性键的自动重复，直接吞掉
            key = 255

        if key != 255:

            last_key = key

            last_key_at = now_ms

            log(f"key={key} frame={state['index']} dirty={state['dirty']}")

        if key in (ord("q"), 27):

            if state["dirty"]:

                save()

            break

        if key in (ord("a"), ord("A"), 81):

            load(max(0, state["index"] - 1))

        elif key in (ord("d"), ord("D"), 83):

            load(min(len(session.frames) - 1, state["index"] + 1))

        elif key in (ord("s"), ord("S")):

            save()

        elif key in (ord("u"), ord("U")):

            if state["boxes"]:

                state["boxes"].pop()

                state["dirty"] = True

        elif key in (ord("x"), ord("X")):

            image_x, image_y = view.to_image(*state["mouse"])

            hit = next(
                (
                    position
                    for position, box in enumerate(state["boxes"])
                    if box.contains(image_x, image_y)
                ),
                None,
            )

            if hit is not None:

                state["boxes"].pop(hit)

                state["dirty"] = True

        elif key in (ord("n"), ord("N")):

            state["boxes"] = []

            save()

        elif key in (ord("p"), ord("P")):

            propagate()

        elif key in (ord("m"), ord("M")):

            state["show_model"] = not state["show_model"]

        elif key == ord("0"):

            state["view"] = View()

        elif ord("1") <= key <= ord("9"):

            slot = key - ord("1")

            if slot < len(names):

                state["cls"] = slot

    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
