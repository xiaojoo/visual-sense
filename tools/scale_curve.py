"""
尺度曲线：这个模型能看见多小的蚊子。

    .venv-ai\\Scripts\\python.exe -m tools.scale_curve --model weights/mosquito_synth.pt
    .venv-ai\\Scripts\\python.exe -m tools.scale_curve --model a.pt --model b.pt --imgsz 640 --imgsz 1280

做法：从 dataset_yolo 的 **val** 划分里取单框蚊子图，把蚊子缩放到短边 = T 像素
再贴回一张 960x720 的画面（模拟"手机画面里一只小蚊子"），按 T 逐档统计命中率。
命中 = 预测框与真值框 IoU ≥ 0.3。

为什么值得单独成一个工具而不是每次现写：这条曲线是后面所有决定的依据
（要不要自采、要不要改 ai.imgsz、素材池够不够），而它的组合逻辑很容易写错 ——
第一版就把 alpha 用了原尺寸去过缩放后的旋转矩阵，量出来"64px 也 0%"，
那是脚本 bug 不是模型不行。**曲线错了比模型错了更糟**，因为它是判断模型的尺子。

两个不能改的口径：

- 真值用**变换后的框**，不是贴图块的外接框；
- 采样用固定 seed，换模型时样本必须是同一批，否则两条曲线不可比。
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

YOLO_DIR = Path("dataset_yolo")

OWN_DIR = Path("dataset")

FRAME_W = 960

FRAME_H = 720

# 蚊子在原图里短边小于这个数的不选：那类图本身已经糊到底，
# 再缩小就只是在测压缩噪声
MIN_SOURCE_SIDE = 24

DEFAULT_TARGETS = (64, 48, 32, 24, 16, 12)

IOU_HIT = 0.3

CONF = 0.25


def load_samples(count: int, seed: int) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
    """
    取 val 划分里"一张图一个蚊子框"的样本。

    只用 val 是有意的：如果测的图参与过训练（或者参与过剪影素材），
    曲线量到的就是记忆而不是泛化。
    """

    images_dir = YOLO_DIR / "images" / "val"

    labels_dir = YOLO_DIR / "labels" / "val"

    if not images_dir.is_dir():

        raise SystemExit(f"[ERROR] 找不到 {images_dir}，先跑 tools.build_dataset")

    items = []

    for path in sorted(images_dir.iterdir()):

        label = labels_dir / f"{path.stem}.txt"

        if not label.exists():

            continue

        rows = [line.split() for line in
                label.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]

        if len(rows) != 1 or len(rows[0]) != 5:

            continue

        frame = cv2.imread(str(path))

        if frame is None:

            continue

        height, width = frame.shape[:2]

        _, cx, cy, bw, bh = [float(value) for value in rows[0]]

        cx, cy, bw, bh = cx * width, cy * height, bw * width, bh * height

        if min(bw, bh) < MIN_SOURCE_SIDE:

            continue

        items.append((frame, (cx, cy, bw, bh)))

    random.Random(seed).shuffle(items)

    return items[:count]


def compose(frame: np.ndarray, box: tuple[float, float, float, float], target: float) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    把蚊子缩放到短边 = target，贴回 960x720 的画面，返回 (画面, 真值框)。

    底图用同一张原图拉伸到 960x720：这样背景统计量至少是"照片级"的，
    而不是纯色块。注意这是代理，不等于手机 MJPEG 的真实纹理。
    """

    cx, cy, bw, bh = box

    scale = target / min(bw, bh)

    half = max(bw, bh) * 1.5

    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))

    x2, y2 = int(min(frame.shape[1], cx + half)), int(min(frame.shape[0], cy + half))

    crop = frame[y1:y2, x1:x2]

    sw, sh = max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))

    crop = cv2.resize(
        crop,
        (sw, sh),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )

    canvas = cv2.resize(frame, (FRAME_W, FRAME_H))

    ox, oy = max(0, (FRAME_W - sw) // 2), max(0, (FRAME_H - sh) // 2)

    canvas[oy:oy + sh, ox:ox + sw] = crop[:FRAME_H - oy, :FRAME_W - ox]

    tx, ty = ox + (cx - x1) * scale, oy + (cy - y1) * scale

    return canvas, (tx - bw * scale / 2, ty - bh * scale / 2,
                    tx + bw * scale / 2, ty + bh * scale / 2)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))

    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    inter = ix * iy

    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter

    return inter / union if union > 0 else 0.0


def own_frames() -> list[np.ndarray]:
    return [f for p in sorted(OWN_DIR.glob("*/images/*.jpg"))
            if (f := cv2.imread(str(p))) is not None]


def matched_conf(model: YOLO, frames: list[np.ndarray], imgsz: int) -> tuple[float, int]:
    """
    找一个"在自己采集帧上一个假框都不开"的最低 conf。

    为什么必须有这一步：命中率是在固定 conf 上量的，而**不同模型的
    激进程度不一样**。实测同一个 640 训的权重在 imgsz=1280 下对着干净墙面
    开出 70 个假框，而 1280 训的权重一个不开 —— 这时"65% vs 35%"比的
    其实是谁敢乱开框，不是谁看得见蚊子。
    """

    seen = []

    for frame in frames:

        result = model.predict(frame, conf=CONF, imgsz=imgsz, verbose=False)[0]

        seen += [float(b.conf) for b in result.boxes]

    if not seen:

        return CONF, 0

    # 高于最大的那个假框才算干净，留 0.02 余量避免正好压线
    return min(0.9, max(seen) + 0.02), len(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description="小目标尺度曲线")

    parser.add_argument("--model", action="append", default=[], help="要测的权重，可重复")

    parser.add_argument("--imgsz", type=int, action="append", default=[],
                        help="推理输入分辨率，可重复；默认 640")

    parser.add_argument("--targets", default=",".join(str(t) for t in DEFAULT_TARGETS))

    parser.add_argument("--samples", type=int, default=60)

    parser.add_argument("--seed", type=int, default=11)

    parser.add_argument("--no-own", action="store_true", help="跳过自己采集帧的误报统计")

    parser.add_argument("--match-fp", action="store_true",
                        help="先把每个模型的 conf 抬到「在自己帧上零误报」，再比命中率")

    args = parser.parse_args()

    if not args.model:

        print("[ERROR] 至少给一个 --model")

        return 1

    sizes = args.imgsz or [640]

    targets = [int(item) for item in args.targets.split(",") if item.strip()]

    samples = load_samples(args.samples, args.seed)

    if not samples:

        print("[ERROR] val 划分里没有可用的单框样本")

        return 1

    print()
    print(f"样本 {len(samples)} 张（dataset_yolo/val，seed={args.seed}）  "
          f"命中判据 IoU>={IOU_HIT}  conf>={CONF}")

    from ultralytics import YOLO

    own = own_frames()

    models = [(path, YOLO(path)) for path in args.model]

    # 名字宽度按最长的权重名算：截断成 8 个字符会把
    # mosquito_synth / mosquito_synth2 打成同一个 "mosquito"，
    # 对比表里这种重名比不整齐危险得多。
    name_width = max(len(Path(path).stem) for path in args.model) + 8

    header = " " * name_width + "".join(f"{t:>9}px" for t in targets)

    for size in sizes:

        print()
        print(f"推理 imgsz = {size}")
        print(header)

        for path, model in models:

            conf = CONF

            if args.match_fp:

                conf, false_boxes = matched_conf(model, own, size)

                label = (f"{Path(path).stem}"
                         f"[conf {conf:.2f} 屏蔽 {false_boxes} 假框]").ljust(name_width)

            else:

                label = Path(path).stem.ljust(name_width)

            row = [label]

            for target in targets:

                hit = 0

                for frame, box in samples:

                    canvas, truth = compose(frame, box, target)

                    result = model.predict(canvas, conf=conf, imgsz=size, verbose=False)[0]

                    if any(iou(tuple(b.xyxy[0].tolist()), truth) >= IOU_HIT for b in result.boxes):

                        hit += 1

                row.append(f"{100 * hit / len(samples):>8.1f}%")

            print("  ".join(row))

    if not args.no_own:

        frames = own

        print()
        print(f"自己采集的 {len(frames)} 帧（画面里没有蚊子）—— 误报数")

        for path, model in models:

            t0 = time.perf_counter()

            images = boxes = 0

            for frame in frames:

                result = model.predict(frame, conf=CONF, imgsz=sizes[-1], verbose=False)[0]

                images += 1 if len(result.boxes) else 0

                boxes += len(result.boxes)

            ms = (time.perf_counter() - t0) / max(1, len(frames)) * 1000

            print(f"  {Path(path).stem:<22} 出框的图 {images}/{len(frames)}"
                  f"  共 {boxes} 框  {ms:.1f} ms/帧 @imgsz={sizes[-1]}"
                  f"（后台还有训练在跑时这个 ms 不算数）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
