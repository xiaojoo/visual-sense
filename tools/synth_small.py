"""
把大 mosquito 剪影贴成小目标，造一份尺度对得上的训练集。

    .venv\\Scripts\\python.exe -m tools.synth_small
    .venv\\Scripts\\python.exe -m tools.synth_small --count 400 --seed 5

为什么需要这一层：mosquito_ext.pt 实测 64px 以下检出率从 80% 掉到 13%，
24px 以下是 0%（见 docs/notes.md 的尺度曲线）。外部集里 ≤32px 的蚊子框只有 36 个，
靠它本身学不会看小目标。而真蚊子还没拍到，所以先用合成把尺度这一课补上。

三条口径，都是为了让这份数据不骗自己：

1. **剪影只用 train 划分的图，尺度曲线只用 val 划分的图。**
   同一只蚊子既当训练素材又当测试素材的话，曲线会自己给自己打分。
2. **底图优先用你自己采集的 40 帧**（每组里 85% 的图来自它们），
   因为要学的就是"你这台相机、这个光照、这面墙"。外部背景只占一小部分，
   作用是别让背景种类少到能被记住。
   两个分组的**底图照片互不相交**：同一张墙派生的合成图只出现在一侧。
   上一版按"采集帧 / 外部背景"分家，val 变成只考"没见过的背景种类"，
   实测同一份代码只差素材池，val mAP 能差 7 倍而训练损失几乎重合 ——
   那种分数不能用来比模型。
3. **贴完之后按目标尺寸做光学模糊，再整图重压 JPEG。**
   15px 的蚊子本来就没有纹理细节；不模糊的话模型学到的是
   "一块高清矩形"，那就完全白做。

接缝是这类增广最容易翻车的地方，所以每个会话都出一张
`contact_sheet.png`：贴得像不像，眼睛一扫就知道，不用等训练结果。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

YOLO_DIR = Path("dataset_yolo")

# tools/import_cutouts.py 从 Mosquito Alert 抠好的 RGBA 剪影
CUTOUT_DIR = Path("external/mosquitoalert/cutouts")

OWN_DIR = Path("dataset")

OUT_DIR = Path("synth")

FRAME_W = 960

FRAME_H = 720

# 目标短边范围（像素）。上限 48 是为了覆盖 64px 那个拐点附近，
# 下限 8 比预估的 15px 更狠一点：宁可训得偏保守。
MIN_TARGET = 8

MAX_TARGET = 48

# 每张图放几只：0 只 = 纯负样本
COUNT_WEIGHTS = {0: 15, 1: 55, 2: 22, 3: 8}

# 每个分组里，底图来自你采集帧的比例。外部背景只占少数，
# 作用是别让背景种类少到能被记住。
OWN_SHARE = 0.85


@dataclass
class Foreground:

    crop: np.ndarray

    # 抠图用的 alpha：剪影边框一圈的中位数当背景色，
    # 和背景差得越多越保留。不抠的话贴上去的是一整块原图背景，
    # 浅色墙上的蚊子贴到深色西装上就是一块白斑 —— 模型学到的是那块斑。
    alpha: np.ndarray

    # 蚊子在 crop 内的框（像素），裁图时留了余量，所以要单独记
    box: tuple[float, float, float, float]


def matte(crop: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """
    从剪影里把目标抠出来，返回 0-255 的 alpha。

    做法是"和边框背景差得多的像素算前景"，然后只保留**连着框中心
    的那个连通域**。不取连通域的话，纹理背景（叶子、布纹）整片都会被
    判成前景，贴过去就是一块带硬边的矩形 —— 模型学到的是那块矩形。
    """

    ring = np.concatenate(
        [
            crop[:2].reshape(-1, 3),
            crop[-2:].reshape(-1, 3),
            crop[:, :2].reshape(-1, 3),
            crop[:, -2:].reshape(-1, 3),
        ]
    )

    background = np.median(ring, axis=0)

    distance = np.linalg.norm(crop.astype(np.float32) - background, axis=-1) / 255.0

    scaled = (distance * 255.0).astype(np.uint8)

    threshold, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = (scaled > max(int(threshold), 30)).astype(np.uint8) * 255

    count, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    box_x, box_y, box_w, box_h = box

    center = labels[
        int(np.clip(box_y + box_h / 2.0, 0, crop.shape[0] - 1)),
        int(np.clip(box_x + box_w / 2.0, 0, crop.shape[1] - 1)),
    ]

    if center > 0:

        binary = np.where(labels == center, 255, 0).astype(np.uint8)

    # 腿是单像素宽的细线，不胀一下会在缩放里整个丢掉
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8))

    edge = max(1, min(crop.shape[:2]) // 8)

    ramp = np.ones(crop.shape[:2], np.float32)

    ramp[:edge] *= np.linspace(0.0, 1.0, edge)[:, None]

    ramp[-edge:] *= np.linspace(1.0, 0.0, edge)[:, None]

    ramp[:, :edge] *= np.linspace(0.0, 1.0, edge)[None, :]

    ramp[:, -edge:] *= np.linspace(1.0, 0.0, edge)[None, :]

    alpha = cv2.GaussianBlur(binary.astype(np.float32) * ramp, (3, 3), 0.8)

    return (alpha * 255.0).clip(0, 255).astype(np.uint8)


def matte_score(alpha: np.ndarray, box: tuple[float, float, float, float]) -> tuple[float, float]:
    """
    (框内平均 alpha, 框外一圈平均 alpha)。
    前者要高、后者要低，才说明真的抠出来了而不是整块搬过来。
    """

    x1, y1, x2, y2 = [int(v) for v in (box[0], box[1], box[0] + box[2], box[1] + box[3])]

    inside = float(alpha[max(0, y1):y2, max(0, x1):x2].mean()) / 255.0

    margin = int(max(box[2], box[3]) * 0.6)

    outer = np.zeros(alpha.shape, bool)

    outer[max(0, y1 - margin):min(alpha.shape[0], y2 + margin),
          max(0, x1 - margin):min(alpha.shape[1], x2 + margin)] = True

    outside = float(alpha[outer].mean()) / 255.0

    return inside, outside


def load_foregrounds(images_dir: Path, labels_dir: Path) -> list[Foreground]:
    """
    train 划分里每个蚊子框裁出来当剪影素材。
    """

    items: list[Foreground] = []

    rejected = 0

    for image in sorted(images_dir.iterdir()):

        label = labels_dir / f"{image.stem}.txt"

        if image.suffix.lower() not in (".jpg", ".jpeg", ".png") or not label.exists():
            continue

        frame = cv2.imread(str(image))

        if frame is None:
            continue

        height, width = frame.shape[:2]

        for line in label.read_text(encoding="utf-8", errors="replace").splitlines():

            fields = line.split()

            if len(fields) != 5:
                continue

            _, cx, cy, bw, bh = [float(item) for item in fields]

            box_w, box_h = bw * width, bh * height

            margin = max(box_w, box_h) * 0.08

            x1 = int(max(0, cx * width - box_w / 2 - margin))

            y1 = int(max(0, cy * height - box_h / 2 - margin))

            x2 = int(min(width, cx * width + box_w / 2 + margin))

            y2 = int(min(height, cy * height + box_h / 2 + margin))

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0 or min(box_w, box_h) < 24:

                # 太小的原图放大回去只会更糊，不如不用
                continue

            box = (
                cx * width - x1 - box_w / 2.0,
                cy * height - y1 - box_h / 2.0,
                box_w,
                box_h,
            )

            alpha = matte(crop, box)

            inside, outside = matte_score(alpha, box)

            # 抠不干净的素材直接不要：宁可少，也不要一块带硬边的矩形。
            # 阈值按实测分布定：蚊子是细腿长喙的东西，框内本来就只有
            # 两三成像素是它（中位 0.23），拿"框内要过半"当标准会全灭。
            # 真正要卡的是框外那圈 —— 它接近 0 才说明背景被丢掉了。
            if inside < 0.12 or outside > 0.15:

                rejected += 1

                continue

            items.append(Foreground(crop=crop, alpha=alpha, box=box))

    print(f"  抠图不合格被剔除的素材: {rejected}")

    return items


def load_cutouts(directory: Path) -> list[Foreground]:
    """
    读 tools.import_cutouts 抠好的 RGBA 剪影。

    这批素材的价值在于**背景和目标已经分开了**：外部集那 131 个剪影是从
    带框照片里现抠的，抠不干净就得丢；Mosquito Alert 的图本来就是微距特写，
    抠出来的轮廓完整得多，腿和翅膀都在。
    """

    items: list[Foreground] = []

    for path in sorted(directory.glob("*.png")):

        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:

            continue

        alpha = rgba[..., 3]

        rows, cols = np.nonzero(alpha > 128)

        if len(rows) < 40:

            continue

        y1, y2 = int(rows.min()), int(rows.max())

        x1, x2 = int(cols.min()), int(cols.max())

        if min(y2 - y1, x2 - x1) < 24:

            continue

        items.append(
            Foreground(
                crop=rgba[..., :3].copy(),
                alpha=alpha.copy(),
                box=(
                    float(x1),
                    float(y1),
                    float(x2 - x1 + 1),
                    float(y2 - y1 + 1),
                ),
            )
        )

    return items


def load_backgrounds(rng: random.Random) -> tuple[list[Path], list[Path]]:
    """
    底图分两堆：你自己的采集帧、外部集里作者自称背景的图。
    """

    own = sorted(OWN_DIR.glob("*/images/*.jpg"))

    ext_dir = ROOT / "external/mosquito_yolo"

    ext: list[Path] = []

    for image in sorted((ext_dir / "images").iterdir()):

        label = ext_dir / "labels" / f"{image.stem}.txt"

        if label.exists() and not label.read_text(encoding="utf-8", errors="replace").strip():

            ext.append(image)

    rng.shuffle(own)

    rng.shuffle(ext)

    return own, ext


def color_match(fg: np.ndarray, dst: np.ndarray, strength: float) -> np.ndarray:
    """
    把剪影的亮度和对比度往落点区域的统计量上靠。

    不匹配的话贴上去的东西要么明显比背景亮一块，要么暗一块，
    模型学到的就是那个亮度差而不是蚊子形状。
    """

    result = fg.astype(np.float32)

    for channel in range(3):

        src = result[..., channel]

        ref = dst[..., channel].astype(np.float32)

        src_mean, src_std = float(src.mean()), float(src.std()) + 1e-3

        ref_mean, ref_std = float(ref.mean()), float(ref.std()) + 1e-3

        # 只往落点的曝光靠一点，不敢拉满：
        # 深色蚊子贴到亮墙上，拉满等于把它洗成灰色
        gain = float(np.clip(ref_std / src_std, 0.7, 1.4))

        matched = (src - src_mean) * gain + ref_mean

        result[..., channel] = src * (1.0 - strength) + matched * strength

    return np.clip(result, 0, 255).astype(np.uint8)


def paste(
    canvas: np.ndarray,
    item: Foreground,
    rng: random.Random,
) -> tuple[float, float, float, float] | None:
    """
    随机尺度、随机旋转、羽化贴到画布上，返回蚊子自己的归一化框。

    框是把原框四个角跟着缩放和旋转一起变换后取外接矩形算的，
    不是拿贴图块的大小当框 —— 贴图块带着 18% 余量，
    用它会系统性地把框训大，尺度曲线也跟着虚。
    """

    target = float(np.exp(rng.uniform(np.log(MIN_TARGET), np.log(MAX_TARGET))))

    box_x, box_y, box_w, box_h = item.box

    scale = target / min(box_w, box_h)

    resized = cv2.resize(
        item.crop,
        (
            max(1, int(round(item.crop.shape[1] * scale))),
            max(1, int(round(item.crop.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )

    angle = rng.uniform(0.0, 360.0)

    matrix = cv2.getRotationMatrix2D(
        (resized.shape[1] / 2.0, resized.shape[0] / 2.0),
        angle,
        1.0,
    )

    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])

    width = int(resized.shape[0] * sin + resized.shape[1] * cos)

    height = int(resized.shape[0] * cos + resized.shape[1] * sin)

    matrix[0, 2] += width / 2.0 - resized.shape[1] / 2.0

    matrix[1, 2] += height / 2.0 - resized.shape[0] / 2.0

    patch = cv2.warpAffine(resized, matrix, (width, height), flags=cv2.INTER_LINEAR)

    # alpha 是先按同样比例缩到 resized 尺寸、再跟着旋转的。
    # 直接拿原尺寸的 alpha 去过这个矩阵，整块轮廓会被转到画面外，
    # 结果 mask 全零、什么都没贴上去，而框照样返回 —— 最难查的一种安静失败。
    alpha_small = cv2.resize(
        item.alpha,
        (resized.shape[1], resized.shape[0]),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )

    alpha = cv2.warpAffine(
        alpha_small,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
    )

    mask = (alpha.astype(np.float32) / 255.0)[..., None]

    if float(mask.max()) <= 0.01:

        # 轮廓被转到画面外之类的安静失败：框照样算得出来，画面却什么都没变。
        # 宁可这张不贴，也不能留一个指着空地方的框。
        return None

    if width >= FRAME_W or height >= FRAME_H:

        return None

    x = rng.randint(0, FRAME_W - width - 1)

    y = rng.randint(0, FRAME_H - height - 1)

    window = canvas[y:y + height, x:x + width]

    blended = color_match(patch, window, 0.25)

    canvas[y:y + height, x:x + width] = (
        window * (1.0 - mask) + blended * mask
    ).astype(np.uint8)

    # 15px 的目标本来就没有细节：按尺寸补一点光学模糊。
    # 只能糊目标自己的像素 —— 糊整块矩形窗口会在清晰背景上留一个
    # 软焦方块，那个方块比蚊子更好学。
    if target < 28:

        soft = cv2.GaussianBlur(canvas[y:y + height, x:x + width], (3, 3), 0.6)

        canvas[y:y + height, x:x + width] = (
            canvas[y:y + height, x:x + width] * (1.0 - mask) + soft * mask
        ).astype(np.uint8)

    corners = np.array(
        [
            [box_x, box_y],
            [box_x + box_w, box_y],
            [box_x + box_w, box_y + box_h],
            [box_x, box_y + box_h],
        ],
        dtype=np.float32,
    ) * scale

    corners = corners @ matrix[:, :2].T + matrix[:, 2] + np.array([x, y], np.float32)

    left, top = corners.min(axis=0)

    right, bottom = corners.max(axis=0)

    return (
        (left + right) / 2.0 / FRAME_W,
        (top + bottom) / 2.0 / FRAME_H,
        (right - left) / FRAME_W,
        (bottom - top) / FRAME_H,
    )


def make_session(
    session: Path,
    wanted: int,
    own: list[Path],
    ext: list[Path],
    foregrounds: list[Foreground],
    rng: random.Random,
    seed: int,
    own_share: float,
    tag: str,
) -> int:
    """
    生成一个分组的合成图，返回负样本数。
    """

    (session / "images").mkdir(parents=True, exist_ok=True)

    (session / "labels").mkdir(parents=True, exist_ok=True)

    negatives = 0

    for index in range(wanted):

        pool = own if (own and (not ext or rng.random() < own_share)) else ext

        base = cv2.imread(str(rng.choice(pool)))

        if base is None:

            continue

        canvas = cv2.resize(base, (FRAME_W, FRAME_H))

        boxes: list[str] = []

        for _ in range(rng.choices(list(COUNT_WEIGHTS), weights=list(COUNT_WEIGHTS.values()))[0]):

            item = rng.choice(foregrounds)

            box = paste(canvas, item, rng)

            if box is not None:

                boxes.append("0 " + " ".join(f"{value:.6f}" for value in box))

        if not boxes:

            negatives += 1

        name = f"{index:05d}"

        cv2.imwrite(
            str(session / "images" / f"{name}.jpg"),
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 85)],
        )

        (session / "labels" / f"{name}.txt").write_text(
            "\n".join(boxes) + ("\n" if boxes else ""),
            encoding="utf-8",
        )

    (session / "metadata.json").write_text(
        json.dumps(
            {
                "source": "tools/synth_small.py",
                "seed": seed,
                "group": tag,
                "backgrounds": {"own": len(own), "external": len(ext)},
                "images": wanted,
                "negatives": negatives,
                "target_short_side_px": [MIN_TARGET, MAX_TARGET],
                "foreground_cutouts": len(foregrounds),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return negatives


def build(count: int, seed: int, out_root: Path, val_share: float) -> tuple[Path, Path, int]:
    """
    生成两个**底图互不相交**的分组会话。

    切分按"底图是哪一张照片"分，不按"底图来自哪一堆"分。
    上一版把 40 张采集帧做成一个会话、571 张外部背景做成另一个会话，
    于是 val 只测"没见过的背景种类"——实测同一份代码、只差素材池，
    两个模型的 val mAP 差 7 倍，而它们的训练损失几乎重合
    （box 1.874 vs 1.924），说明那个分数量的是背景押注，不是眼力。

    现在两组都用同样的 85/15 混合比例，只是**具体照片不重叠**：
    同一张墙派生出的所有合成图必然落在同一侧，
    val 既不会见过同一张墙，也不会只见过另一种墙。
    """

    rng = random.Random(seed)

    np.random.seed(seed)

    train_dir = YOLO_DIR / "images" / "train"

    train_labels = YOLO_DIR / "labels" / "train"

    if not train_dir.is_dir():

        raise SystemExit(f"[ERROR] 找不到 {train_dir}，先跑 build_dataset 生成 dataset_yolo")

    from_external_boxes = load_foregrounds(train_dir, train_labels)

    from_cutouts = load_cutouts(ROOT / CUTOUT_DIR)

    foregrounds = from_external_boxes + from_cutouts

    if not foregrounds:

        raise SystemExit("[ERROR] 一个剪影素材都没有")

    print(f"  素材：外部框现抠 {len(from_external_boxes)} + 抠好的剪影 {len(from_cutouts)}"
          f" = {len(foregrounds)}")

    own, ext = load_backgrounds(rng)

    if not own:

        raise SystemExit("[ERROR] dataset/ 下没有采集帧可当底图")

    # 按底图分家：own 只有 40 张，切完 val 侧只剩几张墙，
    # 所以 val_share 不能太大
    own_cut = int(len(own) * (1.0 - val_share))

    ext_cut = int(len(ext) * (1.0 - val_share))

    groups = [
        ("gA", own[:own_cut], ext[:ext_cut], 1.0 - val_share),
        ("gB", own[own_cut:], ext[ext_cut:], val_share),
    ]

    stamp = time.strftime("%Y%m%d_%H%M%S")

    made: list[Path] = []

    for tag, own_pool, ext_pool, share in groups:

        if not own_pool and not ext_pool:

            raise SystemExit(f"[ERROR] 分组 {tag} 没有底图，val_share 太大")

        session = out_root / f"{stamp}_{tag}"

        wanted = max(1, int(round(count * share)))

        negatives = make_session(
            session, wanted, own_pool, ext_pool, foregrounds, rng, seed, OWN_SHARE, tag
        )

        made.append(session)

        print(f"  {session.name}: {wanted} 张  负样本 {negatives}"
              f"  底图 采集帧 {len(own_pool)} + 外部 {len(ext_pool)}")

    print(f"  合计素材 {len(foregrounds)} 个"
          "（外部框只用 train 划分，尺度曲线测的是没见过的个体）")

    return made[0], made[1], len(foregrounds)


def contact_sheet(session: Path, limit: int = 12) -> Path:
    """
    把带框样本拼一张出来，用来肉眼判接缝。
    """

    images = sorted((session / "images").iterdir())

    rng = random.Random(3)

    picks = rng.sample(images, min(limit, len(images)))

    tiles = []

    for path in picks:

        frame = cv2.imread(str(path))

        label = session / "labels" / f"{path.stem}.txt"

        for line in label.read_text(encoding="utf-8", errors="replace").splitlines():

            fields = line.split()

            if len(fields) != 5:
                continue

            _, cx, cy, bw, bh = [float(item) for item in fields]

            x1 = int((cx - bw / 2) * FRAME_W)

            y1 = int((cy - bh / 2) * FRAME_H)

            x2 = int((cx + bw / 2) * FRAME_W)

            y2 = int((cy + bh / 2) * FRAME_H)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # 旁边再画一个 4 倍放大，方便看细节
            pad = max(14, int(max(x2 - x1, y2 - y1) * 1.6))

            crop = frame[
                max(0, y1 - pad):min(FRAME_H, y2 + pad),
                max(0, x1 - pad):min(FRAME_W, x2 + pad),
            ]

            zoom = cv2.resize(crop, (110, 110), interpolation=cv2.INTER_NEAREST)

            frame[4:114, 4:114] = zoom

        tiles.append(cv2.resize(frame, (480, 360)))

    cols = 4

    blank = np.full((360, 480, 3), 30, np.uint8)

    while len(tiles) % cols:

        tiles.append(blank)

    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]

    sheet = np.vstack(rows)

    out = session / "contact_sheet.png"

    cv2.imwrite(str(out), sheet)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="小目标合成集生成器")

    parser.add_argument("--count", type=int, default=2000, help="总共生成多少张")

    parser.add_argument("--seed", type=int, default=3)

    parser.add_argument("--out", default=str(OUT_DIR))

    parser.add_argument("--val-share", type=float, default=0.15,
                        help="分给验证组的底图比例（采集帧只有 40 张，别调太大）")

    args = parser.parse_args()

    out_root = Path(args.out)

    if not out_root.is_absolute():

        out_root = ROOT / out_root

    print()
    print(f"输出 {out_root}   目标短边 {MIN_TARGET}-{MAX_TARGET}px   seed={args.seed}")

    first, last, cutouts = build(args.count, args.seed, out_root, args.val_share)

    sheet = contact_sheet(first)

    print()
    print(f"对照表 {sheet}")
    print(
        "下一步：\n"
        f"  .venv-ai\\Scripts\\python.exe -m tools.build_dataset "
        f"--sessions {out_root.relative_to(ROOT).as_posix()} "
        "--labels human --names mosquito --keep-classes 0 "
        "--out dataset_yolo_synth"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
