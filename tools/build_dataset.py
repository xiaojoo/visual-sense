"""
把采集会话转成 YOLO 训练目录。

    .venv-ai\\Scripts\\python.exe -m tools.build_dataset --dry-run
    .venv-ai\\Scripts\\python.exe -m tools.build_dataset --names mosquito
    .venv-ai\\Scripts\\python.exe -m tools.build_dataset --sessions dataset/2026*

输出：

    dataset_yolo/
    ├── images/train/  images/val/
    ├── labels/train/  labels/val/
    └── mosquito.yaml

**划分按会话，不按图片。**

同一个会话里的帧是连续拍下的同一场景，画面几乎重复。
按图片随机划分的话，验证集里全是训练集的邻居帧，
mAP 会虚高，而且高得完全看不出真实水平。
按会话划分才能保证验证集是"没见过的场景"。

标签本身会被重新校验和重映射：
采集时写进去的是 COCO 类别号，
换成蚊子权重之前那些号没有意义，
所以 --keep-classes / --names 用来把源类别映射成训练类别。
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

LABEL_SUFFIX = ".txt"


@dataclass
class Sample:
    """
    一张图加它对应的标签文件。
    """

    image: Path
    label: Path | None
    session: str

    # human = 人工真值，auto = 模型预标注，none = 没有标签文件
    origin: str = "none"

    @property
    def key(self) -> str:
        return f"{self.session}/{self.image.name}"


@dataclass
class Report:
    """
    转换过程的统计。

    分成"跳过"和"坏样本"两类是有意的：
    空标签是合法的负样本，
    坐标越界是数据错误，
    两者混在一个计数里就看不出数据集到底能不能用。
    """

    sessions: list[str] = field(default_factory=list)

    images: int = 0

    labelled: int = 0

    empty: int = 0

    unlabelled: int = 0

    dropped_empty: int = 0

    bad_lines: int = 0

    bad_boxes: int = 0

    remapped: int = 0

    dropped_classes: Counter = field(default_factory=Counter)

    class_counts: Counter = field(default_factory=Counter)

    problems: list[str] = field(default_factory=list)


def discover_sessions(paths: list[Path], max_depth: int = 3) -> list[Path]:
    """
    把参数展开成会话目录。

    一个会话目录的判据是里面有 images/。

    往下找而不是只看一层：只找一层的话，
    目录稍微套深一点就报"没找到会话"，
    而这几乎总是路径写错或手滑嵌套，
    报错信息又不说它找去了哪里，很难查。
    这里限定 max_depth 层，避免一头扎进 images/ 里翻。
    """

    sessions: list[Path] = []

    seen: set[Path] = set()

    skip = {"images", "labels", "preview", "__pycache__"}

    queue: list[tuple[Path, int]] = [
        (path, 0) for path in paths if path.is_dir()
    ]

    while queue:

        current, depth = queue.pop(0)

        if current in seen:

            continue

        seen.add(current)

        if (current / "images").is_dir():

            sessions.append(current)

            continue

        if depth >= max_depth:

            continue

        try:

            children = sorted(
                child
                for child in current.iterdir()
                if child.is_dir() and child.name not in skip
            )

        except OSError:

            continue

        queue += [(child, depth + 1) for child in children]

    return sessions


def collect_samples(session: Path, source: str = "prefer") -> list[Sample]:
    """
    一个会话里的所有图。

    标签有两个来源，含义完全不同：

        labels/        人工真值
        labels_auto/   模型当时猜的

    source=prefer 时人工优先、缺失才退回预标注，
    但这种情况会被计数并警告 ——
    把模型猜测混进真值里训练，
    等于让模型学自己的错误，而且报表上看不出来。
    """

    images = sorted(
        path
        for path in (session / "images").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )

    human_dir = session / "labels"

    auto_dir = session / "labels_auto"

    samples: list[Sample] = []

    for image in images:

        human = human_dir / f"{image.stem}{LABEL_SUFFIX}"

        auto = auto_dir / f"{image.stem}{LABEL_SUFFIX}"

        label: Path | None = None

        origin = "none"

        if source in ("human", "prefer") and human.exists():

            label = human

            origin = "human"

        elif source in ("auto", "prefer") and auto.exists():

            label = auto

            origin = "auto"

        samples.append(
            Sample(
                image=image,
                label=label,
                session=session.name,
                origin=origin,
            )
        )

    return samples


def parse_label(
    path: Path,
    keep: set[int] | None,
    remap: dict[int, int],
    report: Report,
    owner: str,
) -> list[str] | None:
    """
    读一个 YOLO 标签文件，校验并重映射类别。

    返回改写后的行；文件读不动返回 None。

    校验项目：字段数、类别在保留集合内、
    数值能转 float、中心点和宽高都在 [0,1] 内、宽高大于 0。
    越界的框直接丢掉并计数 ——
    采集时的框是从像素坐标换算来的，
    目标贴边时归一化后可能略微超出 1.0，
    这种留着会让训练端报错。
    """

    try:

        raw = path.read_text(encoding="utf-8")

    except OSError as exc:

        report.problems.append(f"{owner}: 读标签失败 {exc}")

        return None

    lines: list[str] = []

    for number, line in enumerate(raw.splitlines(), start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:

            report.bad_lines += 1

            report.problems.append(
                f"{owner}: 第 {number} 行字段数 {len(parts)} < 5"
            )

            continue

        try:

            class_id = int(parts[0])

            values = [float(value) for value in parts[1:5]]

        except ValueError:

            report.bad_lines += 1

            report.problems.append(
                f"{owner}: 第 {number} 行不是数字"
            )

            continue

        cx, cy, width, height = values

        # 先校验几何，再过滤类别：
        # 框越界跟它是哪一类无关，
        # 反过来先按类别筛掉，坏数据就会被算成"正常丢弃"，
        # 一个坐标写错的数据集看起来完全健康。
        if (
            not 0.0 <= cx <= 1.0
            or not 0.0 <= cy <= 1.0
            or not 0.0 < width <= 1.0
            or not 0.0 < height <= 1.0
        ):

            report.bad_boxes += 1

            report.problems.append(
                f"{owner}: 第 {number} 行框越界 "
                f"({cx:.3f}, {cy:.3f}, {width:.3f}, {height:.3f})"
            )

            continue

        if keep is not None and class_id not in keep:

            report.dropped_classes[class_id] += 1

            continue

        target = remap.get(class_id, class_id)

        if target != class_id:

            report.remapped += 1

        report.class_counts[target] += 1

        lines.append(
            f"{target} "
            f"{cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
        )

    return lines


def split_by_session(
    prepared: list[tuple[Sample, list[str]]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Sample, list[str]]], list[tuple[Sample, list[str]]], str]:
    """
    按会话整块划分 train / val。

    第三个返回值是实际用的划分方式："session" 或 "image"。
    必须把它报出来 —— 划分方式一变，验证分数就不可比，
    而这件事只在退化时才发生，最容易悄悄溜过去。

    会话数太少时（比如只有 1 个会话标过真值）没法按会话划分，
    这时退回按图片划分：分数会明显虚高，但工具不该直接罢工。
    """

    by_session: dict[str, list[tuple[Sample, list[str]]]] = {}

    for item in prepared:

        by_session.setdefault(item[0].session, []).append(item)

    report_sessions = sorted(by_session)

    if len(report_sessions) < 2:

        rng = random.Random(seed)

        merged = list(prepared)

        rng.shuffle(merged)

        cut = int(len(merged) * (1.0 - val_ratio))

        return merged[:cut], merged[cut:], "image"

    total = sum(len(by_session[name]) for name in report_sessions)

    target_val = max(1.0, total * val_ratio)

    # 按会话大小升序填 val：两个会话、目标 20% 的时候，
    # 乱序可能先撞上大的那个，把绝大部分数据划去验证，
    # 甚至两个都进 val（train 直接空掉）。
    ordered = sorted(report_sessions, key=lambda name: (len(by_session[name]), name))

    val_sessions: list[str] = []

    val_count = 0

    for name in ordered:

        # 至少留一个会话给 train，否则训无可训
        if val_count >= target_val or len(val_sessions) >= len(ordered) - 1:

            break

        val_sessions.append(name)

        val_count += len(by_session[name])

    train = [
        item
        for name in report_sessions
        if name not in val_sessions
        for item in by_session[name]
    ]

    val = [
        item
        for name in val_sessions
        for item in by_session[name]
    ]

    return train, val, "session"


def place(
    sample: Sample,
    lines: list[str],
    out_dir: Path,
    split: str,
    link: bool,
) -> None:
    """
    把图片和标签放进 <split>/ 下。

    文件名前缀上会话名：不同会话可能有同名的 000000.jpg，
    不加前缀会互相覆盖，而且覆盖是静默的。
    """

    image_dir = out_dir / "images" / split

    label_dir = out_dir / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)

    label_dir.mkdir(parents=True, exist_ok=True)

    name = f"{sample.session}_{sample.image.stem}"

    target_image = image_dir / f"{name}{sample.image.suffix}"

    if link:

        try:

            if target_image.exists():

                target_image.unlink()

            target_image.hardlink_to(sample.image)

        except OSError:

            shutil.copy2(sample.image, target_image)

    else:

        shutil.copy2(sample.image, target_image)

    (label_dir / f"{name}{LABEL_SUFFIX}").write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def write_yaml(
    out_dir: Path,
    names: list[str],
) -> Path:

    path = out_dir / "mosquito.yaml"

    body = [
        "# 由 tools/build_dataset.py 生成，别手改",
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]

    body += [f"  {index}: {name}" for index, name in enumerate(names)]

    path.write_text("\n".join(body) + "\n", encoding="utf-8")

    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集会话 → YOLO 训练目录",
    )

    parser.add_argument(
        "--sessions",
        nargs="*",
        default=["dataset"],
        help="会话目录或父目录，可多个",
    )

    parser.add_argument(
        "--out",
        default="dataset_yolo",
        help="输出目录，相对项目根目录",
    )

    parser.add_argument(
        "--labels",
        default="prefer",
        choices=("prefer", "human", "auto"),
        help="标签来源：prefer=人工优先、缺失用模型预标注；"
             "human=只用人工真值；auto=只用预标注",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="划分随机种子，固定下来才能保证可复现",
    )

    parser.add_argument(
        "--names",
        default="",
        help="输出类别名，逗号分隔，例如 mosquito 或 mosquito,fly",
    )

    parser.add_argument(
        "--keep-classes",
        default="",
        help="只保留这些源类别号，逗号分隔；默认全保留",
    )

    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="丢掉没有框的图片（默认留下当负样本）",
    )

    parser.add_argument(
        "--link",
        action="store_true",
        help="用硬链接而不是复制，省磁盘",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计和校验，不写文件",
    )

    parser.add_argument(
        "--check-images",
        action="store_true",
        help="用 OpenCV 真读一遍每张图，抓损坏文件（慢）",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sessions = discover_sessions(
        [Path(item) for item in args.sessions]
    )

    if not sessions:

        print(
            "[ERROR] 没找到会话目录。"
            "先按 R 采集，或者用 --sessions 指到具体目录。"
        )

        for item in args.sessions:

            print(f"  查过：{item}")

        return 1

    samples: list[Sample] = []

    slots: list[str] = []

    origins: Counter = Counter()

    for index, session in enumerate(sessions):

        found = collect_samples(session, args.labels)

        # 会话名只到秒，同一秒开两次就会重名。
        # 前缀一个序号，分组和落盘文件名都不会再撞车。
        slot = f"{index:02d}_{session.name}"

        here: Counter = Counter()

        for sample in found:

            sample.session = slot

            here[sample.origin] += 1

        origins += here

        slots.append(slot)

        print(
            f"[会话] {slot}: {len(found)} 张  "
            f"人工 {here['human']}  预标注 {here['auto']}  "
            f"无标签 {here['none']}"
        )

        samples += found

    if not samples:

        print("[ERROR] 会话里一张图都没有")

        return 1

    keep: set[int] | None = None

    if args.keep_classes:

        keep = {
            int(item)
            for item in args.keep_classes.split(",")
            if item.strip()
        }

    names = (
        [item.strip() for item in args.names.split(",") if item.strip()]
        if args.names
        else []
    )

    source_ids: set[int] = set()

    for sample in samples:

        if sample.label is None:
            continue

        try:

            raw = sample.label.read_text(encoding="utf-8")

        except OSError:

            continue

        for line in raw.splitlines():

            line = line.strip()

            if line:

                try:

                    source_ids.add(int(line.split()[0]))

                except (IndexError, ValueError):

                    continue

    source_ids_sorted = sorted(source_ids)

    if names and keep is not None and len(names) != len(keep):

        print(
            f"[ERROR] --names 有 {len(names)} 个，"
            f"--keep-classes 有 {len(keep)} 个，对不上"
        )

        return 1

    # 保留下来的源类别按顺序映射成 0..n-1
    if keep is None:

        ordered = source_ids_sorted

    else:

        ordered = sorted(keep)

    remap = {
        source: index
        for index, source in enumerate(ordered)
    }

    if not names:

        names = [f"class{index}" for index in range(len(ordered))]

    report = Report(sessions=slots)

    prepared: list[tuple[Sample, list[str]]] = []

    for sample in samples:

        report.images += 1

        if args.check_images:

            if cv2.imread(str(sample.image)) is None:

                report.problems.append(
                    f"{sample.key}: 图片读不出来"
                )

                continue

        if sample.label is None:

            # 没有标签文件 ≠ 这帧没有目标。
            # 没人看过的帧写成空标签，就是给它打了"无蚊子"，
            # 而这种假负样本会直接教模型漏检——
            # 恰恰是蚊子检测最怕的错。所以跳过，不猜。
            report.unlabelled += 1

            continue

        lines = parse_label(
            sample.label,
            keep,
            remap,
            report,
            sample.key,
        )

        if lines is None:

            continue

        if lines:

            report.labelled += 1

        else:

            report.empty += 1

            if args.drop_empty:

                report.dropped_empty += 1

                continue

        prepared.append((sample, lines))

    if not prepared:

        print(
            "[ERROR] 没有任何可用标签，"
            "输出目录会是空的。"
        )

        print(
            "        先用 tools.label_frames 标注，"
            "或换 --labels auto 看模型预标注。"
        )

        return 1

    train, val, split_mode = split_by_session(
        prepared,
        args.val_ratio,
        args.seed,
    )

    out_dir = Path(args.out)

    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    print()
    print("=" * 62)
    print(f"会话        : {len(sessions)} 个")
    print(f"图片        : {report.images}")
    print(f"有框        : {report.labelled}")
    print(f"空标签      : {report.empty}"
          f"{f'（已丢弃 {report.dropped_empty}）' if report.dropped_empty else ''}")

    if report.unlabelled:

        print(
            f"未标注跳过  : {report.unlabelled} 帧"
            "（没有标签文件，不等于没有目标，所以不写进去）"
        )

    if origins["auto"]:

        print(
            f"!! {origins['auto']} 帧用的是模型预标注，"
            "不是人工真值。"
        )

        print(
            "!! 拿它训练等于让模型学自己的错误。"
            "标完之后用 --labels human 只取真值。"
        )

    if origins["human"] and origins["auto"]:

        print("!! 人工真值和模型猜测混在一起，结果无法解释")

    print(f"源类别      : {source_ids_sorted}")
    print(f"映射        : {remap}")
    print(f"输出类别    : {names}")
    print(f"框统计      : {dict(report.class_counts)}")

    if report.dropped_classes:

        print(f"按类别丢弃  : {dict(report.dropped_classes)}")

    if report.bad_lines or report.bad_boxes:

        print(
            f"坏行 {report.bad_lines} 个，"
            f"越界框 {report.bad_boxes} 个"
        )

    print("-" * 62)

    def ratio(part: int) -> str:
        return f"{100.0 * part / max(1, len(prepared)):.0f}%"

    print(
        f"train       : {len(train)} 张"
        f"   val: {len(val)} 张  "
        f"(val 占比 {ratio(len(val))}，目标 {args.val_ratio:.0%})"
    )

    val_sessions = sorted({item[0].session for item in val})

    if split_mode == "image":

        print(
            "!!  有标签的会话不足 2 个，"
            "已退回按图片随机划分。"
        )

        print(
            "!!  这种 val 和 train 是同一场景的相邻帧，"
            "分数会明显虚高，不能用来判断真实水平。"
        )

    else:

        print(f"划分        : 按会话（{len(val_sessions)} 个会话进 val）")

        print(f"val 会话     : {val_sessions}")

    if report.empty and not args.drop_empty and report.labelled:

        empty_ratio = report.empty / max(1, report.images)

        if empty_ratio > 0.5:

            print(
                f"!!  {empty_ratio:.0%} 的图是空标签。"
                "少量负样本有用，超过一半就要确认是不是白采了。"
            )

    print("=" * 62)

    if report.problems:

        print()
        print("问题（最多列 20 条）：")

        for problem in report.problems[:20]:

            print(f"  - {problem}")

    if args.dry_run:

        print()
        print("[dry-run] 没有写任何文件")

        return 0

    if out_dir.exists():

        shutil.rmtree(out_dir)

    for split, items in (("train", train), ("val", val)):

        for sample, lines in items:

            place(sample, lines, out_dir, split, args.link)

    yaml_path = write_yaml(out_dir, names)

    print()
    print(f"输出        : {out_dir}")
    print(f"配置        : {yaml_path}")
    print()
    print("训练：")
    print(
        f"  yolo detect train data={yaml_path.as_posix()} "
        f"model=weights/yolo11n.pt epochs=100 imgsz=640"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
