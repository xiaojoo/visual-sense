"""
把外部 YOLO 数据集转成 tools.build_dataset 认的会话目录。

    .venv\\Scripts\\python.exe -m tools.import_external

默认输入 dataset/external/fly_mosquito_yolo/unpacked/FLY_MOS_Dataset
（Mendeley 77hr7mxd3h，1764 图 + 1764 txt），
默认输出 dataset/external/mosquito_yolo/{images,labels}。

为什么不把外部目录直接喂给 build_dataset，中间要过这一层：

1. **类别号不一样。** 这个集子里 1 才是蚊子，0/2/3 全是苍蝇
   （作者标漏了口径，对照表 dataset/external/class_contact_sheet.png 判读后确认）。
   我们的训练集只有一类 0，所以要重映射。
2. **带苍蝇框的图整张丢，不是把苍蝇框删掉留图当背景。**
   那些图里还有根本没框出来的苍蝇，留下就是教模型"看见一群只报一个"。
3. **文件名带空格和括号，还有 webp。** build_dataset 的 IMAGE_SUFFIXES
   不认 webp，那些图会被静默跳过 —— 静默跳过是最难查的一种丢数据。

输出目录非空时直接退出，不覆盖：这个工具跑一次几十秒，
覆盖掉一份已经用过的数据集不值得省这点事。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SRC = Path("external/fly_mosquito_yolo/unpacked/FLY_MOS_Dataset")

DEFAULT_OUT = Path("external/mosquito_yolo")

# 外部集里蚊子的类别号。换一份数据集要改的就是这一行。
SOURCE_MOSQUITO_ID = 1

# 训练集里蚊子固定是 0
OUTPUT_ID = 0

READABLE = (".jpg", ".jpeg", ".png", ".webp")

# build_dataset 认的后缀；不在这里的要转格式，否则会被静默跳过
ACCEPTED = (".jpg", ".jpeg", ".png")


@dataclass
class Report:
    scanned: int = 0

    kept: int = 0

    kept_boxes: int = 0

    negatives: int = 0

    converted: int = 0

    no_label: int = 0

    orphan_label: int = 0

    dropped_other_class: int = 0

    bad_lines: int = 0

    short_sides: list[float] = field(default_factory=list)


def parse_line(line: str) -> tuple[int, float, float, float, float] | None:
    """
    一行 YOLO 标签 -> (class, cx, cy, w, h)；不合法返回 None。
    """

    fields = line.split()

    if len(fields) != 5:
        return None

    try:

        cls = int(float(fields[0]))

        values = [float(item) for item in fields[1:]]

    except ValueError:

        return None

    if any(not 0.0 <= value <= 1.0 for value in values):
        return None

    return (cls, *values)


def import_session(src: Path, out: Path, report: Report) -> list[str]:
    """
    逐张搬运，返回 manifest 行。
    """

    images = sorted(
        path
        for path in src.iterdir()
        if path.suffix.lower() in READABLE
    )

    stems = {path.stem for path in images}

    report.orphan_label = sum(
        1
        for path in src.glob("*.txt")
        if path.stem not in stems
    )

    manifest: list[str] = ["output\tsource\tboxes"]

    for position, image in enumerate(images):

        report.scanned += 1

        label = src / f"{image.stem}.txt"

        if not label.exists():

            # 没有标签 = 没人看过这张，不等于这张没有蚊子
            report.no_label += 1

            continue

        lines = [
            parse_line(line)
            for line in label.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            if line.strip()
        ]

        parsed = [item for item in lines if item is not None]

        report.bad_lines += len(lines) - len(parsed)

        if any(item[0] != SOURCE_MOSQUITO_ID for item in parsed):

            report.dropped_other_class += 1

            continue

        frame = cv2.imread(str(image))

        if frame is None:

            report.no_label += 1

            continue

        height, width = frame.shape[:2]

        kept_lines: list[str] = []

        for _, cx, cy, bw, bh in parsed:

            if bw <= 0.0 or bh <= 0.0:
                continue

            kept_lines.append(
                f"{OUTPUT_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )

            report.short_sides.append(min(bw * width, bh * height))

        target = out / "images" / f"ext_{position:05d}"

        suffix = image.suffix.lower()

        if suffix not in ACCEPTED:

            target = target.with_suffix(".jpg")

            cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            report.converted += 1

        else:

            target = target.with_suffix(suffix)

            shutil.copyfile(image, target)

        (out / "labels" / f"{target.stem}.txt").write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""),
            encoding="utf-8",
        )

        report.kept += 1

        report.kept_boxes += len(kept_lines)

        if not kept_lines:

            report.negatives += 1

        manifest.append(
            f"{target.name}\t{image.name}\t{len(kept_lines)}"
        )

    return manifest


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)

    return ordered[int(ratio * (len(ordered) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="外部 YOLO 数据集 -> 本项目会话目录",
    )

    parser.add_argument("--src", default=str(DEFAULT_SRC))

    parser.add_argument("--out", default=str(DEFAULT_OUT))

    args = parser.parse_args()

    src = Path(args.src)

    out = Path(args.out)

    if not src.is_absolute():

        src = ROOT / src

    if not out.is_absolute():

        out = ROOT / out

    if not src.is_dir():

        print(f"[ERROR] 没有输入目录 {src}")

        return 1

    if (out / "images").exists() and any((out / "images").iterdir()):

        print(f"[ERROR] {out} 里已经有东西，先删掉再跑")

        return 1

    (out / "images").mkdir(parents=True, exist_ok=True)

    (out / "labels").mkdir(parents=True, exist_ok=True)

    report = Report()

    manifest = import_session(src, out, report)

    (out / "manifest.tsv").write_text(
        "\n".join(manifest) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print(f"输入      : {src}")
    print(f"输出      : {out}")
    print(f"扫过      : {report.scanned}")
    print(f"搬入      : {report.kept} 张（其中蚊子框 {report.kept_boxes} 个）")
    print(f"负样本    : {report.negatives} 张空标签")
    print(f"转格式    : {report.converted} 张（webp -> jpg）")
    print(f"丢弃 有非蚊子框 : {report.dropped_other_class} 张")
    print(f"跳过 没有标签   : {report.no_label} 张")
    print(f"标签没有图片 : {report.orphan_label} 个")
    print(f"坏行      : {report.bad_lines}")

    if report.short_sides:

        values = report.short_sides

        print("-" * 60)
        print(
            "蚊子框短边 px   "
            f"p10 {percentile(values, 0.10):.0f}  "
            f"p50 {percentile(values, 0.50):.0f}  "
            f"p90 {percentile(values, 0.90):.0f}  "
            f"min {min(values):.0f}  max {max(values):.0f}"
        )
        print(
            f"              <32px {sum(1 for v in values if v < 32)} 个"
            f"   <48px {sum(1 for v in values if v < 48)} 个"
            f"   <64px {sum(1 for v in values if v < 64)} 个"
        )

    if report.negatives:

        share = 100.0 * report.negatives / max(1, report.kept)

        print("-" * 60)
        print(
            f"[注意] {report.negatives} 张负样本（{share:.0f}%）来自原作者自称的"
            "background，没有逐张核验过里面到底有没有虫。"
        )

    print("=" * 60)

    if not report.kept:

        print("[ERROR] 一张都没搬进来")

        return 1

    print()
    print(
        "下一步：\n"
        f"  .venv-ai\\Scripts\\python.exe -m tools.build_dataset "
        f"--sessions {out.relative_to(ROOT).as_posix()} "
        "--labels human --names mosquito --keep-classes 0 --dry-run"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
