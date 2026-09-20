"""
从 Mosquito Alert 公开集（BioStudies S-BIAD249，40,978 张）里取图并抠剪影。

    .venv\\Scripts\\python.exe -m tools.import_cutouts                 # 默认 1200 张 / 上限 400MB
    .venv\\Scripts\\python.exe -m tools.import_cutouts --limit 300     # 先小批试

为什么只要剪影不要框：那批图只有物种标签、没有框，本来就不能直接训检测。
但它们是**蚊子形状的素材库**，抠出来交给 tools.synth_small 往真实场景上贴，
正好补上"剪影只有 131 个"这个瓶颈。

清单接口：https://www.ebi.ac.uk/biostudies/api/v1/files/S-BIAD249?path=&start=N&length=M
（length 上限 1000，超了返回空列表而不是报错，所以分页要按 1000 走）
图片接口：https://www.ebi.ac.uk/biostudies/files/S-BIAD249/<物种目录>/<uuid>.jpg

路径第一层就是物种名，所以采样时按物种轮转，避免整池都是同一种蚊子。

许可：CC BY。模型若要对外发布，得署 Mosquito Alert 的源。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

OUT_DIR = Path("external/mosquitoalert")

API = "https://www.ebi.ac.uk/biostudies/api/v1/files/S-BIAD249"

FILE_BASE = "https://www.ebi.ac.uk/biostudies/files/S-BIAD249"

TOTAL_RECORDS = 40978

PAGE = 1000

# 单张超过这个体积的不要：4K 原图对抠剪影没好处，只让下载变慢
MAX_BYTES = 1_200_000

# 剪影短边小于这个数的不要：贴到 8-48px 会变成放大，糊得没有参考价值
MIN_SILOUETTE = 40

# 抠图前先缩到这个长边。原图动辄 12MP，形态学和连通域在 4000x3000 上
# 每张要好几秒，而 Mosquito Alert 的蚊子在缩到 1000px 后仍有 200px 以上，
# 足够贴成 8-48px 的目标。不缩的话 1200 张要跑一个多小时。
MAX_SIDE = 1000

DOWNLOAD_WORKERS = 6

# 实心度上限：alpha 面积 / 外接框面积。蚊子是细腿长翅的东西，实测真蚊子
# 剪影在 0.08~0.33，手指、镜头白盘、虚化块这类误抠都在 0.52~0.79。
# 只按"最大连通域"选的话 60 张里 38 张抠到的是背景块，必须靠这条筛掉。
MAX_FILL = 0.35

UA = "VisualSense/0.6 (research; small-insect detection)"


def fetch_listing(pages: int, rng: random.Random) -> list[dict]:
    """
    在整份清单上等距取若干页，而不是从头截一段。

    清单是按上传时间排的，只取前 N 条的话拿到的全是 2014 年的同一批拍摄，
    物种和拍摄条件都偏。
    """

    records: list[dict] = []

    starts = rng.sample(range(0, TOTAL_RECORDS - PAGE), pages) if pages else []

    for start in sorted(starts):

        url = f"{API}?path=&start={start}&length={PAGE}"

        request = urllib.request.Request(url, headers={"User-Agent": UA})

        try:

            with urllib.request.urlopen(request, timeout=60) as response:

                payload = json.loads(response.read().decode("utf-8"))

        except (urllib.error.URLError, json.JSONDecodeError) as exc:

            print(f"  [警告] 清单 start={start} 取失败：{exc}")

            continue

        batch = payload.get("data") or []

        records.extend(batch)

        print(f"  清单 start={start}: {len(batch)} 条")

    return records


def pick(records: list[dict], limit: int, max_bytes: int) -> list[dict]:
    """
    按物种轮转取样，直到够数或够体积。
    """

    by_species: dict[str, list[dict]] = defaultdict(list)

    for record in records:

        path = record.get("path") or ""

        size = int(record.get("Size") or 0)

        if "/" not in path or size <= 0 or size > MAX_BYTES:
            continue

        by_species[path.split("/")[0]].append(record)

    for bucket in by_species.values():

        random.Random(7).shuffle(bucket)

    ordered: list[dict] = []

    total = 0

    rank = 0

    while len(ordered) < limit and total < max_bytes:

        added = False

        for bucket in by_species.values():

            if rank >= len(bucket):
                continue

            record = bucket[rank]

            size = int(record["Size"])

            if total + size > max_bytes:

                continue

            ordered.append(record)

            total += size

            added = True

            if len(ordered) >= limit:

                break

        rank += 1

        if not added:

            break

    return ordered


def segment(image: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """
    整图里把最像虫子的连通域抠出来，返回 (RGBA, 该域外接框)。

    和 tools.synth_small.matte 同一套判据（边框中位数当背景 + Otsu +
    取最大连通域），区别是这里没有框可参考，只能取最大的那块 ——
    微距照片里最大的非背景区域就是虫子。
    """

    edge = max(2, min(image.shape[:2]) // 40)

    ring = np.concatenate(
        [
            image[:edge].reshape(-1, 3),
            image[-edge:].reshape(-1, 3),
            image[:, :edge].reshape(-1, 3),
            image[:, -edge:].reshape(-1, 3),
        ]
    )

    background = np.median(ring, axis=0)

    scaled = (
        np.linalg.norm(image.astype(np.float32) - background, axis=-1) / 255.0 * 255.0
    ).astype(np.uint8)

    threshold, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = (scaled > max(int(threshold), 40)).astype(np.uint8) * 255

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if count <= 1:

        return None, None

    areas = stats[1:, cv2.CC_STAT_AREA]

    winner = int(np.argmax(areas)) + 1

    x, y, w, h = stats[winner, :4]

    mask = np.where(labels == winner, 255, 0).astype(np.uint8)

    mask = cv2.GaussianBlur(mask, (5, 5), 1.2)

    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    rgba[..., 3] = mask

    return rgba, (int(x), int(y), int(w), int(h))


_LOCK = threading.Lock()

_MANIFEST: list[dict] = []


def handle(index: int, record: dict, out: Path) -> tuple[str, int]:
    """
    下一张图、抠出剪影、能要就落盘。返回 (判定, 下载字节)。

    判定只是计数用，不抛异常 —— 一批里总有几十张是手指、白盘、
    或者根本没虫，为这些中断整轮下载不值得。
    """

    path = record["path"]

    request = urllib.request.Request(f"{FILE_BASE}/{path}", headers={"User-Agent": UA})

    try:

        with urllib.request.urlopen(request, timeout=60) as response:

            blob = response.read()

    except (urllib.error.URLError, OSError):

        return "failed", 0

    image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)

    if image is None:

        return "failed", len(blob)

    long_side = max(image.shape[:2])

    if long_side > MAX_SIDE:

        scale = MAX_SIDE / long_side

        image = cv2.resize(
            image,
            (int(image.shape[1] * scale), int(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    rgba, box = segment(image)

    if rgba is None or box is None:

        return "no_alpha", len(blob)

    x, y, w, h = box

    if min(w, h) < MIN_SILOUETTE:

        return "small", len(blob)

    # 实心度 + 是否触边：把"最大连通域其实是手指/白盘/虚化块"那类挑掉
    fill = float((rgba[..., 3] > 128).sum()) / float(w * h)

    touches = (
        x <= 2
        or y <= 2
        or x + w >= rgba.shape[1] - 2
        or y + h >= rgba.shape[0] - 2
    )

    if touches or fill > MAX_FILL:

        return "blob", len(blob)

    pad = int(max(w, h) * 0.12)

    x1, y1 = max(0, x - pad), max(0, y - pad)

    x2 = min(rgba.shape[1], x + w + pad)

    y2 = min(rgba.shape[0], y + h + pad)

    crop = rgba[y1:y2, x1:x2]

    coverage = float((crop[..., 3] > 128).mean())

    if coverage < 0.05 or coverage > 0.92:

        # 太稀 = 抠到噪声；太满 = 背景根本没分开，整张搬过来了
        return "bad_alpha", len(blob)

    species = path.split("/")[0]

    name = f"{index:04d}_{species}.png"

    cv2.imwrite(str(out / "cutouts" / name), crop)

    with _LOCK:

        _MANIFEST.append(
            {
                "file": name,
                "species": species,
                "source": path,
                "silhouette": [int(w), int(h)],
                "coverage": round(coverage, 3),
                "bytes": len(blob),
            }
        )

    return "ok", len(blob)


def import_all(records: list[dict], out: Path) -> dict:
    """
    并发下载 + 抠图。

    瓶颈是 EBI 的往返延迟（单张 1-3 秒），不是本机算力，所以开线程池；
    串行跑 1200 张要一个多小时，6 路并发几分钟。
    """

    (out / "cutouts").mkdir(parents=True, exist_ok=True)

    _MANIFEST.clear()

    tally = Counter()

    downloaded = 0

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:

        futures = {
            pool.submit(handle, index, record, out): index
            for index, record in enumerate(records)
        }

        for done, future in enumerate(as_completed(futures), start=1):

            status, size = future.result()

            tally[status] += 1

            downloaded += size

            if done % 100 == 0:

                print(
                    f"  {done}/{len(futures)}  已下 {downloaded/1e6:.0f}MB"
                    f"  可用 {tally['ok']}  背景块 {tally['blob']}"
                )

    return {
        "downloaded_bytes": downloaded,
        "failed": tally["failed"],
        "rejected_small": tally["small"],
        "rejected_blob": tally["blob"],
        "rejected_bad_alpha": tally["no_alpha"] + tally["bad_alpha"],
        "usable": tally["ok"],
        "records": sorted(_MANIFEST, key=lambda item: item["file"]),
        "license": "CC BY (Mosquito Alert, BioStudies S-BIAD249)",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Mosquito Alert 剪影导入")

    parser.add_argument("--limit", type=int, default=1200, help="下载多少张原图")

    parser.add_argument("--pages", type=int, default=5, help="清单取几页（每页 1000 条）")

    parser.add_argument("--max-mb", type=int, default=400, help="下载体积上限")

    parser.add_argument("--out", default=str(OUT_DIR))

    args = parser.parse_args()

    out = Path(args.out)

    if not out.is_absolute():

        out = ROOT / out

    rng = random.Random(args.limit)

    print()
    print(f"取清单 {args.pages} 页 ...")

    records = fetch_listing(args.pages, rng)

    print(f"候选 {len(records)} 条，按物种轮转取 {args.limit} 张（上限 {args.max_mb}MB）")

    picked = pick(records, args.limit, args.max_mb * 1_000_000)

    species = sorted({r["path"].split("/")[0] for r in picked})

    print(f"选中 {len(picked)} 张，覆盖 {len(species)} 个物种目录")

    stats = import_all(picked, out)

    (out / "cutouts.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    usable = stats["records"]

    print()
    print("=" * 56)
    print(f"下载        : {stats['downloaded_bytes']/1e6:.0f} MB  失败 {stats['failed']}")
    print(f"可用剪影    : {stats['usable']}")
    print(f"剔除 太小    : {stats['rejected_small']}（短边 < {MIN_SILOUETTE}px）")
    print(f"剔除 抠到背景: {stats['rejected_blob']}（触边或实心度 > {MAX_FILL}）")
    print(f"剔除 alpha差 : {stats['rejected_bad_alpha']}（抠空或整张搬回）")

    if usable:

        sides = sorted(min(item["silhouette"]) for item in usable)

        print(
            "剪影短边    "
            f"p10 {sides[int(0.1*(len(sides)-1))]}  "
            f"p50 {sides[len(sides)//2]}  "
            f"p90 {sides[int(0.9*(len(sides)-1))]}  "
            f"max {sides[-1]} px"
        )

        counts: dict[str, int] = {}

        for item in usable:

            counts[item["species"]] = counts.get(item["species"], 0) + 1

        print("物种分布    " + ", ".join(
            f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        ))

    print(f"清单        {out / 'cutouts.json'}")
    print("=" * 56)

    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
