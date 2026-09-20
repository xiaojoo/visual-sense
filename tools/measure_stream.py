"""
手机视频流延迟 / 清晰度实测工具。

和主程序走同一套代码（app.camera / app.detector），
所以这里测出来的数就是实际运行时的数。

    .venv-ai\\Scripts\\python.exe -m tools.measure_stream
    .venv-ai\\Scripts\\python.exe -m tools.measure_stream --seconds 15
    .venv-ai\\Scripts\\python.exe -m tools.measure_stream --watch     # 边调手机边看
    .venv-ai\\Scripts\\python.exe -m tools.measure_stream --no-ai     # 只测流，不跑模型

清晰度用 Laplacian 方差：

    数值越高越锐利。失焦画面通常只有个位数到几十，
    对焦正常一般能到几百。

调手机对焦时盯这个数，比用眼睛判断"好像清楚了"可靠。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.camera import Camera, CameraConfig  # noqa: E402
from app.detector import Detector, DetectorConfig  # noqa: E402
from app.draw import draw_detections  # noqa: E402


def read_phone_status(url: str) -> dict:
    """
    读取摄像头端自己上报的状态。

    PhoneCamera 提供 /status，里面有 fps_target 和实时 fps。

    这里的作用是把"手机上设成了多少"和"PC 上实测到多少"
    放在同一份输出里：两个数不一致时，
    说明瓶颈在手机编码而不是网络或 PC。

    读不到就返回空 dict，不影响测量本身
    （USB 摄像头、或没有该接口的 app 都会走这条路）。
    """

    try:

        parsed = urlparse(url)

        base = f"{parsed.scheme}://{parsed.netloc}"

        with urllib.request.urlopen(base + "/status", timeout=2) as response:

            return json.load(response)

    except Exception:

        return {}


def sharpness(image: np.ndarray, center: bool = False) -> float:
    """
    Laplacian 方差。

    center=True 只看中央 1/2 区域，
    因为自动对焦通常锁的是画面中心。
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if center:

        h, w = gray.shape

        gray = gray[
            h // 4: h * 3 // 4,
            w // 4: w * 3 // 4,
        ]

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    return float(np.percentile(values, p))


def load_config() -> dict:
    with (ROOT / "config" / "config.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="手机视频流延迟与清晰度实测",
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="采样时长，默认 8 秒",
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="每秒打印一行，边调手机边看",
    )

    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="不加载模型，只测摄像头链路",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--save",
        default="runs/live",
        help="样本帧输出目录，相对项目根目录",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_config()

    camera = Camera(
        CameraConfig(
            mode=str(config.get("camera_mode", "usb")),
            index=int(config.get("camera_index", 0)),
            url=str(config.get("camera_url", "")),
            reconnect_interval=float(
                config.get("reconnect_interval", 2.0)
            ),
        )
    )

    print(f"[Camera] {camera.config.mode} -> {camera.config.url}")

    if not camera.connect():

        print(f"[ERROR] 连接失败: {camera.last_error}")

        return 1

    camera.set_buffer_size(int(config.get("buffer_size", 1)))

    phone_status = read_phone_status(camera.config.url)

    if phone_status:

        print(
            f"[Phone] fps_target={phone_status.get('fps_target')}  "
            f"fps={phone_status.get('fps')}  "
            f"camera={phone_status.get('camera')}  "
            f"light={phone_status.get('light')}"
        )

    else:

        print("[Phone] /status 不可用，只报 PC 侧实测值")

    detector = None

    if not args.no_ai:

        detector = Detector(
            DetectorConfig(
                enabled=True,
                model_path="weights/yolo11n.pt",
                device="auto",
                imgsz=args.imgsz,
                conf=0.25,
                iou=0.70,
            ),
            ROOT,
        )

        if not detector.load():

            print(
                f"[WARN] 模型加载失败，"
                f"只测摄像头: {detector.last_error}"
            )

            detector = None

    out_dir = Path(args.save)

    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    reads: list[float] = []

    intervals: list[float] = []

    infers: list[float] = []

    counts: list[int] = []

    sharp_full: list[float] = []

    sharp_center: list[float] = []

    last_print = 0.0

    prev = None

    frame: np.ndarray | None = None

    deadline = time.perf_counter() + args.seconds

    print()
    print("采样中...")

    while time.perf_counter() < deadline:

        ok, current, read_ms = camera.read()

        if not ok:

            print(f"[ERROR] 读帧失败: {camera.last_error}")

            break

        now = time.perf_counter()

        reads.append(read_ms)

        if prev is not None:
            intervals.append((now - prev) * 1000.0)

        prev = now

        frame = current

        sharp_full.append(sharpness(frame))

        sharp_center.append(sharpness(frame, center=True))

        if detector is not None:

            detections = detector.detect(frame)

            infers.append(detector.last_infer_ms)

            counts.append(len(detections))

        else:

            detections = []

        # -----------------------------------------------------
        # watch 模式：每秒一行 + 持续覆盖样本帧，
        # 手机上调整、PC 上立刻看效果
        # -----------------------------------------------------

        if args.watch and now - last_print >= 1.0:

            last_print = now

            cv2.imwrite(
                str(out_dir / "last_frame.png"),
                frame,
            )

            if detections:

                marked = frame.copy()

                draw_detections(marked, detections)

                cv2.imwrite(
                    str(out_dir / "last_frame_ai.png"),
                    marked,
                )

            print(
                f"fps={1000.0 / st.mean(intervals) if intervals else 0:5.1f}  "
                f"read={st.mean(reads):6.2f} ms  "
                f"infer={(st.mean(infers) if infers else 0):6.2f} ms  "
                f"sharp={sharp_full[-1]:7.1f}  "
                f"center={sharp_center[-1]:7.1f}  "
                f"det={len(detections)}",
                flush=True,
            )

    camera.disconnect()

    if not reads:

        print("[ERROR] 没有读到任何帧")

        return 1

    if frame is not None:

        cv2.imwrite(str(out_dir / "sample_frame.png"), frame)

        if detector is not None:

            marked = frame.copy()

            draw_detections(
                marked,
                detector.detect(frame),
            )

            cv2.imwrite(
                str(out_dir / "sample_frame_ai.png"),
                marked,
            )

    height, width = frame.shape[:2]

    def report_sharp(values: list[float], label: str) -> None:
        """
        清晰度不能只看平均值。

        自动对焦来回拉的时候，一半帧清晰一半帧糊，
        平均值会落在中间，看不出真实情况。
        所以给 p50 / p95 / max，
        再给一个"清晰帧占比"：达到 p95 一半以上的帧有多少。
        """

        p95 = percentile(values, 95)

        good = (
            sum(1 for value in values if value >= p95 * 0.5)
            / len(values)
            * 100.0
        )

        print(
            f"清晰度 {label} : "
            f"p50 {percentile(values, 50):7.1f}  "
            f"p95 {p95:7.1f}  "
            f"max {max(values):7.1f}  "
            f"清晰帧 {good:5.1f}%"
        )

    total_ms = st.mean(reads) + (
        st.mean(infers) if infers else 0.0
    )

    measured_fps = (
        1000.0 / st.mean(intervals) if intervals else 0.0
    )

    print()
    print("=" * 62)
    print(f"分辨率        : {width} x {height}")
    print(f"采样帧数      : {len(reads)}")
    print(f"实际帧率      : {measured_fps:.1f} fps")

    if phone_status:

        print(
            f"手机自报      : "
            f"{phone_status.get('fps')} fps"
            f"  (目标 {phone_status.get('fps_target')})"
        )

    print(
        f"read()  ms    : "
        f"avg {st.mean(reads):6.2f}  "
        f"p50 {percentile(reads, 50):6.2f}  "
        f"p95 {percentile(reads, 95):6.2f}  "
        f"max {max(reads):6.2f}"
    )
    print(
        f"帧间隔   ms   : "
        f"avg {st.mean(intervals):6.2f}  "
        f"p95 {percentile(intervals, 95):6.2f}"
    )

    if infers:

        print(
            f"detect() ms  : "
            f"avg {st.mean(infers):6.2f}  "
            f"p50 {percentile(infers, 50):6.2f}  "
            f"p95 {percentile(infers, 95):6.2f}"
        )

        print(
            f"检测占用      : "
            f"{100.0 * st.mean(infers) / st.mean(intervals):.0f}% "
            f"的帧间隔"
        )

    print(
        f"read+detect   : "
        f"{total_ms:6.2f} ms  "
        f"-> 上限 {1000.0 / total_ms:.1f} fps"
    )

    report_sharp(sharp_full, "全画面")

    report_sharp(sharp_center, "中心区")

    print(f"样本帧        : {out_dir / 'sample_frame.png'}")
    print("=" * 62)

    return 0


if __name__ == "__main__":
    sys.exit(main())
