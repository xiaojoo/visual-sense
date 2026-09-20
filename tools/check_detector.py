"""
V0.2/V0.3 离线验证工具。

不开摄像头、不开窗口，
用静态图片和合成视频验证检测与跟踪链路：

    .venv-ai\\Scripts\\python.exe -m tools.check_detector

    .venv-ai\\Scripts\\python.exe -m tools.check_detector --track

    .venv-ai\\Scripts\\python.exe -m tools.check_detector --track --video clip.mp4

    .venv-ai\\Scripts\\python.exe -m tools.check_detector --model weights/mosquito.pt

每个输入输出两张图：

    <名字>_boxes.png  只有检测框 / 轨迹
    <名字>_hud.png    框 + 顶部 HUD + 缩放到窗口尺寸，
                      和主程序实际显示的画面一致

这样"框画错了"和"HUD 画错了"可以分开判断。

--track 还会打一张 ID 连续性表：
每个 ID 出现在哪些帧、断了几次；pan 片段上它进一步核对
每个长命 ID 的逐帧位移是否等于画面真实位移，不达标就非零退出。
静态图片看不出错，跟踪必须看跨帧。
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# 合成片段的帧率与波形周期。
# 周期按秒定义，这样 --clip-frames 只改时长、不改运动速度。
CLIP_FPS = 30.0
WAVE_PERIOD_S = 1.5

# pan 片段的真实平移速度。自检拿它核对每个 ID 的逐帧位移，
# 所以合成器和断言必须读同一个数 —— 各写一份迟早会对不上。
PAN_SPEED_PX_S = 120.0

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.detector import Detector, DetectorConfig, Detection  # noqa: E402
from app.draw import draw_detections, draw_predictions, draw_trails  # noqa: E402
from app.render import create_status_frame, resize_to_window  # noqa: E402
from app.metrics import Metrics  # noqa: E402
from app.motion import MotionPredictor  # noqa: E402
from app.recorder import Recorder, RecorderConfig  # noqa: E402
from app.tracks import Tracks  # noqa: E402


class Backtest:
    """
    预测误差回测。

    每帧把当时的预测挂起来，
    等真实时刻追上预测目标时刻，再拿同一 ID 的实际位置对答案。

    同时记一条基线："假设它从那一刻起就没动过" 的误差。
    没有这条基线，"平均误差 12 px" 这种数字毫无意义 ——
    目标本来就没怎么动的时候，站着不动猜反而更准。

    真值用的是检测框中心（不是拟合位置）：
    用拟合值当 ground truth 会自己给自己打分，
    误差会被平滑掉一半。代价是这个数字里含检测框自身的抖动，
    所以它是误差的上界。
    """

    def __init__(self, horizons_ms: list[float], max_pending: int = 4000):
        self.horizons = horizons_ms

        self.errors: dict[float, list[float]] = {
            h: [] for h in horizons_ms
        }

        self.baseline: dict[float, list[float]] = {
            h: [] for h in horizons_ms
        }

        self.hit_time: dict[float, list[float]] = {
            h: [] for h in horizons_ms
        }

        self.pending: list[tuple] = []

        self.max_pending = max_pending

        self.dropped = 0

    def submit(
        self,
        predictor: MotionPredictor,
        trails: Tracks,
        now: float,
    ) -> None:
        """
        对当前所有 ID、所有时距各挂一条待验证预测。
        """

        for track_id in trails.active_ids():

            points = trails.recent(
                track_id,
                window=predictor.window,
                max_gap_s=predictor.max_gap_s,
            )

            for horizon in self.horizons:

                state = predictor.estimate(
                    track_id,
                    points,
                    horizon_ms=horizon,
                )

                if state is None:
                    continue

                self.pending.append(
                    (
                        horizon,
                        now + horizon / 1000.0,
                        track_id,
                        state.predicted_x,
                        state.predicted_y,
                        state.x,
                        state.y,
                    )
                )

        if len(self.pending) > self.max_pending:

            self.dropped += len(self.pending) - self.max_pending

            self.pending = self.pending[-self.max_pending:]

    def resolve(
        self,
        centers: dict[int, tuple[float, float]],
        now: float,
    ) -> None:
        """
        到点的预测就地对答案，没到点的留着。
        """

        keep: list[tuple] = []

        for item in self.pending:

            horizon, target_at, track_id, px, py, base_x, base_y = item

            if now < target_at:

                keep.append(item)

                continue

            actual = centers.get(track_id)

            if actual is None:

                # 目标在预测落地之前跟丢了，这条作废
                self.dropped += 1

                continue

            self.errors[horizon].append(
                ((px - actual[0]) ** 2 + (py - actual[1]) ** 2) ** 0.5
            )

            self.baseline[horizon].append(
                ((base_x - actual[0]) ** 2
                 + (base_y - actual[1]) ** 2) ** 0.5
            )

            self.hit_time[horizon].append((now - target_at) * 1000.0)

        self.pending = keep

    def report(self) -> None:
        import statistics as st

        print()
        print("=" * 72)
        print(
            "时距    样本    实际落地    匀速预测 px        "
            "原地不动 px    改善"
        )
        print("-" * 72)

        for horizon in self.horizons:

            errors = self.errors[horizon]

            if not errors:

                print(f"{horizon:5.0f}ms      0       --        没有可验证的预测")

                continue

            base = self.baseline[horizon]

            gain = (
                (st.mean(base) - st.mean(errors)) / st.mean(base) * 100.0
                if st.mean(base) > 0
                else 0.0
            )

            print(
                f"{horizon:5.0f}ms  {len(errors):6d}  "
                f"{st.mean(self.hit_time[horizon]):5.0f} ms  "
                f"avg {st.mean(errors):6.1f} p50 {st.median(errors):6.1f} "
                f"p95 {np.percentile(errors, 95):6.1f}  "
                f"avg {st.mean(base):6.1f} p50 {st.median(base):6.1f}  "
                f"{gain:+6.1f}%"
            )

        print("-" * 72)

        if self.dropped:

            print(f"作废（目标提前跟丢 / 队列溢出）：{self.dropped} 条")

        print("=" * 72)


class CameraStub:
    """
    离线验证不需要真实摄像头，
    只需要 create_status_frame 用到的那几个字段。
    """

    connected = False

    last_error = "offline check"

    def get_resolution(self) -> tuple[int, int]:
        return 1280, 720


def default_samples() -> list[Path]:
    """
    ultralytics 自带的样例图片，离线可用。
    """

    try:
        import ultralytics
    except ImportError:
        print(
            "[ERROR] ultralytics 未安装，"
            "请使用 .venv-ai 或 run-ai.bat"
        )

        return []

    assets = Path(ultralytics.__file__).parent / "assets"

    return sorted(
        path
        for name in ("bus.jpg", "zidane.jpg")
        if (path := assets / name).exists()
    )


def synthetic_clip(
    source: Path,
    frames: int,
    out: Path,
    motion: str = "wave",
) -> Path:
    """
    把一张图水平平移，合成一段视频。

    目的不是造目标，而是造"同一批真实检测框跨帧移动"：
    跟踪器的 ID 连续性和预测误差只能在这种输入上验证。
    用纯色块自己画是没有用的——YOLO 不认它，
    一个框都不会出。

    motion 决定平移规律：

        pan    匀速直线 120 px/s。匀速外推对它是送分题，
               误差会好看得毫无意义。
        wave   周期 1.5 秒的正弦往复叠加 45 px/s 漂移，
               速度一直在变还会反向，
               这才是蚊子那种目标该有的对照组。

    两个规律都按**秒**定义，不按帧数：
    早先版本让周期跟 --clip-frames 挂钩，
    结果改一下片段长度就等于改了运动速度，
    60 帧和 150 帧跑出来的误差根本不可比。

    默认 wave。
    """

    image = cv2.imread(str(source))

    height, width = image.shape[:2]

    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        CLIP_FPS,
        (width, height),
    )

    amplitude = width / 6

    for index in range(frames):

        t = index / CLIP_FPS

        if motion == "pan":

            shift = PAN_SPEED_PX_S * t

        else:

            shift = (
                amplitude * math.sin(2.0 * math.pi * t / WAVE_PERIOD_S)
                + 45.0 * t
            )

        writer.write(np.roll(image, int(shift), axis=1))

    writer.release()

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisualSense 检测/跟踪离线验证",
    )

    parser.add_argument(
        "images",
        nargs="*",
        help="待检测图片或视频，默认使用 ultralytics 样例",
    )

    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="视频片段，可重复；等价于把它写在位置参数里",
    )

    parser.add_argument(
        "--model",
        default="weights/yolo11n.pt",
        help="权重路径，相对项目根目录",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto / cpu / cuda / 0",
    )

    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument("--conf", type=float, default=0.25)

    parser.add_argument("--iou", type=float, default=0.70)

    parser.add_argument(
        "--track",
        action="store_true",
        help="开 V0.3 跟踪：分配 ID 并画轨迹",
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="开 V0.4 预测并回测误差（隐含 --track）",
    )

    parser.add_argument(
        "--horizons",
        default="50,100,150",
        help="回测时距，毫秒，逗号分隔",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=6,
        help="拟合窗口点数，和 config.ai.predict_window 保持一致",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=4,
        help="少于这个点数不出预测",
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
    )

    parser.add_argument(
        "--trail",
        type=int,
        default=40,
        help="每个 ID 保留多少个历史点",
    )

    parser.add_argument(
        "--clip-frames",
        type=int,
        default=60,
        help="合成视频帧数（只在 --track 且没给视频时用）",
    )

    parser.add_argument(
        "--motion",
        default="auto",
        choices=("auto", "pan", "wave"),
        help="合成片段运动规律；auto = 回测用 wave、纯跟踪自检用 pan",
    )

    parser.add_argument(
        "--out",
        default="runs/check",
        help="输出目录，相对项目根目录",
    )

    return parser.parse_args()


def run_still_images(
    detector: Detector,
    trails: Tracks,
    metrics: Metrics,
    images: list[Path],
    out_dir: Path,
) -> None:
    """
    一张一张跑，帧之间没有连续性。
    """

    for image_path in images:

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"[SKIP] 无法读取: {image_path}")

            continue

        height, width = frame.shape[:2]

        if detector.config.track_enabled:

            detections = detector.track(frame)

            trails.update(detections, 0)

        else:

            detections = detector.detect(frame)

        metrics.update(0.0)

        metrics.update_infer(
            detector.last_infer_ms,
            len(detections),
        )

        print()
        print(
            f"{image_path.name}  "
            f"{width}x{height}  "
            f"infer={detector.last_infer_ms:.2f} ms  "
            f"det={len(detections)}"
        )

        for detection in detections:

            print(f"    {describe(detection)}")

        write_outputs(
            frame,
            detections,
            trails,
            metrics,
            detector,
            out_dir / image_path.stem,
        )


def run_video(
    detector: Detector,
    trails: Tracks,
    metrics: Metrics,
    video: Path,
    out_dir: Path,
    predictor: MotionPredictor | None = None,
    backtest: Backtest | None = None,
) -> tuple[dict[int, list[tuple[int, float, float, float, float]]], int]:
    """
    逐帧跑一段视频，返回 ID → [(帧号, x1, y1, x2, y2)] 和总帧数。

    这是唯一能验证"同一只目标跨帧是不是同一个 ID"的入口。
    帧号留着是为了算断裂，整只框留着是为了核对位移和形状 ——
    一个 ID 活了 60 帧不代表它跟的是同一个对象。
    """

    capture = cv2.VideoCapture(str(video))

    if not capture.isOpened():
        print(f"[SKIP] 无法打开视频: {video}")

        return {}, 0

    # 录下来的片段里没有真实时钟，
    # 用帧号除以录制帧率造一个，
    # 这样速度和预测时距的单位才和直播时一致（px/s、ms）
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    seen: dict[int, list[tuple[int, float, float, float, float]]] = {}

    index = 0

    detections: list[Detection] = []

    states: list = []

    last: np.ndarray | None = None

    middle = None

    while True:

        ok, frame = capture.read()

        if not ok:
            break

        last = frame

        now = index / fps

        detections = detector.track(frame)

        trails.update(detections, index, now)

        metrics.update(0.0)

        metrics.update_infer(detector.last_infer_ms, len(detections))

        centers: dict[int, tuple[float, float]] = {}

        for detection in detections:

            track_id = getattr(detection, "track_id", -1)

            if track_id is None or track_id < 0:
                continue

            seen.setdefault(int(track_id), []).append(
                (
                    index,
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2,
                )
            )

            centers[int(track_id)] = detection.center

        if predictor is not None:

            states = predictor.estimate_all(trails)

        if backtest is not None and predictor is not None:

            # 先对答案再挂新预测：
            # 同一帧挂上去的东西不该在同一帧被判掉
            backtest.resolve(centers, now)

            backtest.submit(predictor, trails, now)

        if index == 30:
            middle = frame.copy()

        index += 1

    capture.release()

    print()
    print(f"{video.name}  {index} 帧  "
          f"平均 infer={metrics.avg_infer_ms:.2f} ms")

    for track_id, entries in sorted(seen.items()):

        frames = [entry[0] for entry in entries]

        gaps = [
            b - a - 1
            for a, b in zip(frames, frames[1:])
            if b - a > 1
        ]

        print(
            f"    ID {track_id:<3d} "
            f"出现 {len(frames):3d} 帧  "
            f"[{frames[0]}~{frames[-1]}]  "
            f"最大断裂 {max(gaps) if gaps else 0} 帧"
        )

    if last is not None:

        write_outputs(
            last,
            detections,
            trails,
            metrics,
            detector,
            out_dir / f"{video.stem}_last",
            states,
            predictor,
        )

    if middle is not None:

        cv2.imwrite(
            str(out_dir / f"{video.stem}_mid.png"),
            middle,
        )

    return seen, index


# =========================================================
# 自检断言
# =========================================================

# 覆盖 ≥ 这个比例的 ID 才参与核对
ID_COVERAGE = 0.9

# 逐帧位移允许偏离画面真实位移多少：检测框本身有抖动
# （实测长命 ID 偏离 0.02~0.31 px），但跟错对象差的是几十像素
ID_STEP_TOL = 1.0

# 框的宽/高在这个范围内浮动才算"形状稳定"。
# 被画面边缘切开的那只不会稳定：实测贴边目标框宽摆 200 px、
# 完整目标摆 7 px，差着二十多倍，阈值落在中间怎么都不会错。
ID_SHAPE_TOL = 20.0

# 至少要有这么多个形状稳定的目标全程单 ID，否则这条自检是空的
ID_MIN_RIGID = 2

# 形状稳定的目标单帧最多挪多少：画面真值是 4 px，
# 实测最坏 6.4 px（检测框自身的抖动）。
# 中位数对单次跳变免疫，而换对象就是一次跳变，所以要单独判这条。
ID_JUMP_PX = 12.0

# 少于这么多可验证样本，增益百分比就没有意义
MIN_BACKTEST_SAMPLES = 100

# 时距 ≤100ms 上至少要打赢静止基线这么多
MIN_GAIN_PCT = 10.0


def check_track_ids(
    seen: dict[int, list[tuple[int, float, float, float, float]]],
    total_frames: int,
) -> list[str]:
    """
    钉住 V0.3 的结论：pan 片段上形状稳定的目标，全程是同一个 ID。

    画面是整幅刚性平移，真值位移 = PAN_SPEED_PX_S / CLIP_FPS，逐帧恒定。
    所以这里不看"ID 活了几帧"—— 那只能证明分配器没撒手 ——
    看的是相邻帧框心位移对不对得上画面真实位移：
    中途换了对象的 ID 活得再久，位移也会露馅。

    只核对形状稳定的 ID。框尺寸本身在变的 ID 不满足这个前提，
    不是豁免：画面平移时它的可见部分在变（贴边被切开、被遮挡、
    检测尺度跳变），框心根本不跟着画面走。
    实测贴边那只逐帧只走 1.62 px（真值 4.00）而框宽摆 200 px，
    框尺寸稳在 7 px 以内的两只走 4.02 / 4.08。
    这条判据写反的话，自检会把正常的裁剪目标报成跟丢。

    np.roll 还会让出界的目标从另一侧回来，那一帧位移必然跳变，
    所以取中位数而不是均值。只在 pan 片段上跑：
    wave 片段里目标会合法地出画再入画，断裂不是故障。
    """

    step = PAN_SPEED_PX_S / CLIP_FPS

    threshold = max(2, int(total_frames * ID_COVERAGE))

    failures: list[str] = []

    candidates = 0

    rigid = 0

    for track_id, entries in sorted(seen.items()):

        if len(entries) < threshold:
            continue

        candidates += 1

        widths = [entry[3] - entry[1] for entry in entries]

        heights = [entry[4] - entry[2] for entry in entries]

        wobble = max(
            max(widths) - min(widths),
            max(heights) - min(heights),
        )

        deltas = [
            (
                (later[1] + later[3]) / 2.0 - (earlier[1] + earlier[3]) / 2.0,
                (later[2] + later[4]) / 2.0 - (earlier[2] + earlier[4]) / 2.0,
            )
            for earlier, later in zip(entries, entries[1:])
            if later[0] - earlier[0] == 1
        ]

        gaps = [
            later[0] - earlier[0] - 1
            for earlier, later in zip(entries, entries[1:])
            if later[0] - earlier[0] > 1
        ]

        max_gap = max(gaps) if gaps else 0

        dx = statistics.median(delta[0] for delta in deltas)

        dy = statistics.median(delta[1] for delta in deltas)

        # 中位数天生对"单次跳变"免疫，而 ID 中途换对象恰恰就是一次跳变
        # —— 只看中位数会把这件事正好漏掉，所以单独记一个最大值
        jump = max(
            (math.hypot(*delta) for delta in deltas),
            default=0.0,
        )

        if wobble > ID_SHAPE_TOL:

            print(
                f"    ID {track_id:<3d} 覆盖 {len(entries)}/{total_frames}  "
                f"框尺寸摆动 {wobble:5.1f}px  -> 跳过"
                f"（框本身在变形，框心位移不等于画面位移）"
            )

            continue

        rigid += 1

        off = abs(dx - step) > ID_STEP_TOL or abs(dy) > ID_STEP_TOL

        print(
            f"    ID {track_id:<3d} 覆盖 {len(entries)}/{total_frames}  "
            f"最大断裂 {max_gap} 帧  框尺寸摆动 {wobble:4.1f}px  "
            f"逐帧位移 dx={dx:+.2f} dy={dy:+.2f}"
            f"（真值 {step:+.2f} / 0.00）  "
            f"单帧最大跳变 {jump:5.1f}px  "
            f"{'FAIL' if off else 'OK'}"
        )

        if max_gap > 1:

            failures.append(
                f"ID {track_id} 中途断了 {max_gap} 帧，"
                f"重新出现的已经不是同一个目标"
            )

        if off:

            failures.append(
                f"ID {track_id} 逐帧位移 ({dx:.2f}, {dy:.2f})，"
                f"画面真实位移是 ({step:.2f}, 0) —— 这个 ID 跟丢了对象"
            )

        if jump > ID_JUMP_PX:

            failures.append(
                f"ID {track_id} 有一帧挪了 {jump:.1f}px"
                f"（画面真值 {step:.2f}px/帧）—— "
                f"中位数看不出来，但这是一次换对象"
            )

    if candidates < ID_MIN_RIGID:

        failures.append(
            f"只有 {candidates} 个 ID 覆盖 ≥{ID_COVERAGE:.0%}，"
            f"跟踪器没能把任何目标全程认下来"
        )

    elif rigid < ID_MIN_RIGID:

        failures.append(
            f"长命 ID 有 {candidates} 个，但形状稳定的只有 {rigid} 个"
            f"（要求 ≥{ID_MIN_RIGID}）—— "
            f"这些目标的框在平移中一直在变形，多半是片段太长、"
            f"目标被推到画面边缘切开了，位移判据在这段上没有成立的前提。"
            f"缩短 --clip-frames 再看"
        )

    return failures


def check_prediction(backtest: Backtest) -> list[str]:
    """
    钉住 V0.4-A 的结论：匀速外推打得赢"原地不动"这条基线。

    单看"平均误差 15 px"没有意义，目标不动的时候站着猜更准，
    所以判的是相对静止基线的增益；
    样本量太小的"赢"同样没有意义，所以两条一起判。
    """

    failures: list[str] = []

    marks: list[str] = []

    previous: tuple[float, float] | None = None

    for horizon in sorted(backtest.horizons):

        errors = backtest.errors[horizon]

        base = backtest.baseline[horizon]

        if len(errors) < MIN_BACKTEST_SAMPLES:

            failures.append(
                f"{horizon:g}ms 只有 {len(errors)} 条可验证预测，"
                f"少于 {MIN_BACKTEST_SAMPLES}，增益数字不成立"
            )

            continue

        mean = statistics.mean(errors)

        mean_base = statistics.mean(base)

        gain = (
            (mean_base - mean) / mean_base * 100.0
            if mean_base > 0
            else 0.0
        )

        floor = MIN_GAIN_PCT if horizon <= 100.0 else 0.0

        marks.append(f"{horizon:g}ms {gain:+.1f}%")

        if gain <= floor:

            failures.append(
                f"{horizon:g}ms 相对静止基线只赢 {gain:+.1f}%（要求 >{floor:g}%），"
                f"匀速外推这一档不值钱了"
            )

        if previous is not None and mean < previous[1]:

            failures.append(
                f"{horizon:g}ms 的平均误差 {mean:.1f}px 比 "
                f"{previous[0]:g}ms 的 {previous[1]:.1f}px 还小 —— "
                f"看得越远反而越准，这个数不可能对"
            )

        previous = (horizon, mean)

    print(f"    预测自检：{'  '.join(marks) if marks else '没有可核对的时距'}")

    return failures


def describe(detection: Detection) -> str:
    track_id = getattr(detection, "track_id", None)

    prefix = f"#{track_id} " if track_id is not None and track_id >= 0 else ""

    return (
        f"{prefix}"
        f"{detection.class_name:<12} "
        f"{detection.confidence:.3f}  "
        f"({detection.x1:6.0f}, {detection.y1:6.0f}) - "
        f"({detection.x2:6.0f}, {detection.y2:6.0f})  "
        f"{detection.width:5.0f} x {detection.height:4.0f}"
    )


def write_outputs(
    frame: np.ndarray,
    detections: list[Detection],
    trails: Tracks,
    metrics: Metrics,
    detector: Detector,
    stem: Path,
    states: list | None = None,
    predictor: MotionPredictor | None = None,
) -> None:
    """
    分别输出"只有框"和"框 + HUD"两张图。
    """

    stem.parent.mkdir(parents=True, exist_ok=True)

    states = states or []

    predictor = predictor or MotionPredictor(enabled=False)

    boxes = frame.copy()

    if len(trails):
        draw_trails(boxes, trails)

    draw_detections(boxes, detections)

    if states and predictor.enabled:
        draw_predictions(boxes, states)

    cv2.imwrite(f"{stem}_boxes.png", boxes)

    status = create_status_frame(
        frame,
        CameraStub(),
        detector,
        metrics,
        {"camera_mode": "network"},
        detections,
        trails,
        states,
        predictor,
        Recorder(RecorderConfig(), ROOT),
    )

    status = resize_to_window(status, 1280, 720)

    cv2.imwrite(f"{stem}_hud.png", status)

    print(f"    -> {stem}_boxes.png")
    print(f"    -> {stem}_hud.png")


def main() -> int:
    args = parse_args()

    if args.predict:

        # 预测吃的是带 ID 的轨迹，没有跟踪就没有轨迹
        args.track = True

    horizons = [
        float(item)
        for item in str(args.horizons).split(",")
        if item.strip()
    ]

    inputs = [
        Path(item)
        for item in list(args.images) + list(args.video)
    ]

    if not inputs:
        inputs = default_samples()

    if not inputs:
        print("[ERROR] 没有可用的输入")

        return 1

    detector = Detector(
        DetectorConfig(
            enabled=True,
            model_path=str(args.model),
            device=str(args.device),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            track_enabled=args.track,
            tracker=args.tracker,
            trail_frames=args.trail,
        ),
        ROOT,
    )

    if not detector.load():
        print(f"[ERROR] 模型加载失败: {detector.last_error}")

        return 1

    trails = Tracks(max_points=args.trail)

    predictor = (
        MotionPredictor(
            horizon_ms=horizons[len(horizons) // 2] if horizons else 100.0,
            window=args.window,
            min_samples=args.min_samples,
            enabled=args.predict,
        )
        if args.predict
        else None
    )

    backtest = (
        Backtest(horizons)
        if args.predict and predictor is not None
        else None
    )

    metrics = Metrics(sample_window=120)

    out_dir = Path(args.out)

    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "检测"

    if args.track:

        mode = f"跟踪 {args.tracker}"

    if args.predict:

        mode += f" + 预测 {'/'.join(f'{h:g}' for h in horizons)} ms"

    print()
    print("=" * 64)
    print(
        f"Model  : {detector.model_name}\n"
        f"Device : {detector.device}\n"
        f"Mode   : {mode}\n"
        f"Fit    : 窗口 {args.window} 点，最少 {args.min_samples} 点\n"
        f"ImgSz  : {args.imgsz}  Conf: {args.conf}  IoU: {args.iou}"
    )
    print("=" * 64)

    videos = [
        path
        for path in inputs
        if path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    ]

    images = [path for path in inputs if path not in videos]

    # 只有合成的片段才知道真值位移，才谈得上断言
    synthetic_motion: str | None = None

    problems: list[str] = []

    checks: list[str] = []

    if args.track and not videos and not args.images and not args.video:

        # 没给视频就用样例图合成一段，
        # 至少把跨帧 ID 这条路径跑通。
        # 回测预测时必须用 wave：
        # 匀速平移对匀速外推是送分题，误差好看得没有意义。
        motion = args.motion

        if motion == "auto":

            motion = "wave" if args.predict else "pan"

        clip = synthetic_clip(
            images[0],
            args.clip_frames,
            out_dir / f"synthetic_{motion}.mp4",
            motion,
        )

        videos.append(clip)

        synthetic_motion = motion

        print(f"[synth] {clip.name} {args.clip_frames} 帧  motion={motion}")

        images = []

    if images:
        run_still_images(detector, trails, metrics, images, out_dir)

    for video in videos:

        detector.reset_tracking()

        trails.reset()

        seen, total_frames = run_video(
            detector,
            trails,
            metrics,
            video,
            out_dir,
            predictor,
            backtest,
        )

        if synthetic_motion == "pan" and seen:

            print("    ID 位移自检:")

            checks.append("ID 位移")

            problems += check_track_ids(seen, total_frames)

    snapshot = metrics.snapshot()

    print()
    print("=" * 64)
    print(
        f"推理次数   : {snapshot.inferred_frames}\n"
        f"平均耗时   : {snapshot.avg_infer_ms:.2f} ms\n"
        f"最快/最慢  : {snapshot.min_infer_ms:.2f} / "
        f"{snapshot.max_infer_ms:.2f} ms\n"
        f"累计框数   : {snapshot.detections_total}"
    )
    print("=" * 64)

    if backtest is not None:

        backtest.report()

        print()

        checks.append("预测增益")

        problems += check_prediction(backtest)

    print()

    if problems:

        print("=" * 64)

        print(f"[SELFTEST FAIL] {len(problems)} 条不达标：")

        for item in problems:

            print(f"  - {item}")

        print("=" * 64)

        return 1

    if checks:

        print(f"[SELFTEST] {' + '.join(checks)} 全部达标")

    return 0


if __name__ == "__main__":
    sys.exit(main())
