from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .detector import Detection


@dataclass
class RecorderConfig:
    """
    V0.4-B 采集配置，对应 config.json 的 "record" 段。
    """

    # 采集目录，相对项目根目录。
    # 每次按 R 会在下面开一个时间戳会话目录。
    dataset_dir: str = "dataset"

    # 每 N 帧存一张。20 fps 下 stride=5 就是 4 张/秒。
    stride: int = 5

    # 单次会话上限，防止忘关把盘写满
    max_frames: int = 3000

    jpeg_quality: int = 90

    # 把当前模型的框写成预标注。
    #
    # 写进 labels_auto/ 而不是 labels/：
    # labels/ 是人工真值的位置，由 tools/label_frames.py 独占。
    # 两者混在一个目录里，"标了多少帧"就没法统计，
    # 而且一次误触就可能把人工结果盖掉。
    #
    # 注意：COCO 权重里没有 mosquito 类，
    # 现在写出来的预标注对训练没有意义，
    # 留着只是为了以后换成蚊子权重时能直接用。
    auto_labels: bool = False

    # 额外存一份带框的缩略图，方便事后肉眼翻。
    # 占空间，只在抽查时用。
    save_preview: bool = False


class Recorder:
    """
    V0.4-B 帧采集器。

    一次会话产出：

        dataset/<会话>/images/000123.jpg       原始帧，无标注
        dataset/<会话>/labels/000123.txt       人工真值（由标注工具写）
        dataset/<会话>/labels_auto/000123.txt  模型预标注（可选）
        dataset/<会话>/preview/000123.jpg      带框图（可选）
        dataset/<会话>/annotations.jsonl       每帧一行，含 track_id
        dataset/<会话>/metadata.json           这次会话的来龙去脉

    为什么存了 labels 还要再存 annotations.jsonl：

        labels.txt 是给训练用的，只有 5 个数，
        一旦要重算速度、加速度、预测误差就不够了 ——
        帧与帧的对应关系、真实时间戳、跟踪 ID 全丢了，
        只能把检测再跑一遍。

    所以 track_id 和 timestamp 必须一起落盘。
    """

    def __init__(
        self,
        config: RecorderConfig,
        root: Path,
        model_name: str = "",
        class_names: Optional[dict[int, str]] = None,
        camera_info: Optional[dict[str, Any]] = None,
    ):
        self.config = config
        self.root = Path(root)

        self.model_name = model_name

        self.class_names = dict(class_names or {})

        self.camera_info = dict(camera_info or {})

        self.recording = False

        self.saved = 0

        self.session_dir: Optional[Path] = None

        self.started_at = 0.0

        self.last_error = ""

        # 有哪些路模型在写预标注。由 pipeline 在装配时填。
        # 空文件要按路补，否则"这一路这帧没看见"和"这一路没跑"分不开。
        self.sources: list[str] = []

        # 裁剪后退化的框数量。
        # 不写出来的话，丢掉的框就悄无声息地消失了。
        self.clipped_dropped = 0

        self._frame_counter = 0

        self._sizes: list[tuple[int, int]] = []

        self._limit_hit = False

        # session_dir 只是占好的名字，目录要等第一帧真正落盘时才建。
        self._dir_created = False

    # =========================================================
    # Session
    # =========================================================

    @property
    def elapsed_s(self) -> float:
        if not self.recording or not self.started_at:
            return 0.0

        return time.perf_counter() - self.started_at

    @property
    def fps(self) -> float:
        if self.saved < 2 or self.elapsed_s <= 0:
            return 0.0

        return self.saved / self.elapsed_s

    def start(self) -> Optional[Path]:
        """
        占一个会话名，开始采集。

        目录名用本地时间，
        这样翻文件夹时不用查表就知道是哪一次采集。

        这里**不建目录**：
        连按 R 开/停会瞬间产生一堆 0 帧空目录
        （2026-09-20 实测 16 个会话里 7 个是空的），
        事后既污染 dataset/ 的通配，
        也要人工去分辨哪些是真的没采到。
        目录和头信息推迟到第一帧落盘时一起建，见 _ensure_dir()。
        """

        if self.recording:
            return self.session_dir

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base = (
            Path(self.config.dataset_dir)
            if Path(self.config.dataset_dir).is_absolute()
            else self.root / self.config.dataset_dir
        )

        # 名字只精确到秒，同一秒内开两次（连按两下 R、
        # 或者脚本批量起会话）会撞进同一个目录，
        # 两批数据静默混在一起，事后完全分不开。
        # 撞名判据用 exists()：上一会话没落过帧就不存在，
        # 让它复用同一个名字正是想要的 —— 空会话不该占号。
        session = base / stamp

        suffix = 2

        while session.exists():

            session = base / f"{stamp}_{suffix}"

            suffix += 1

        self.session_dir = session

        self.recording = True

        self.saved = 0

        self._frame_counter = 0

        self._sizes = []

        self._limit_hit = False

        self.clipped_dropped = 0

        self._dir_created = False

        self.started_at = time.perf_counter()

        print(f"[Record] 开始采集 -> {session}")

        return session

    def _ensure_dir(self) -> bool:
        """
        第一帧落盘前把目录和头信息建出来。

        返回 False 表示建不出来，调用方这帧就别写了 ——
        宁可少一帧，也不要 images/ 都不存在时留下半份标签。
        """

        if self._dir_created:
            return True

        if self.session_dir is None:
            return False

        try:

            (self.session_dir / "images").mkdir(parents=True, exist_ok=True)

            if self.config.auto_labels:
                # 每路模型一个子目录，见 _write_labels
                for source in self.sources or ["model"]:
                    (self.session_dir / "labels_auto" / source).mkdir(parents=True, exist_ok=True)

            if self.config.save_preview:
                (self.session_dir / "preview").mkdir(parents=True, exist_ok=True)

        except OSError as exc:

            self.last_error = str(exc)

            print(f"[Record] 建目录失败: {exc}")

            return False

        self._dir_created = True

        # 先落一份头信息：中途崩溃也能知道这次在采什么
        self._write_metadata(status="recording")

        return True

    def stop(self) -> dict:
        """
        收尾并返回这次会话的摘要。
        """

        summary: dict = {}

        if self._dir_created:

            summary = self._write_metadata(status="stopped")

        print(
            f"[Record] 停止采集，共 {self.saved} 帧"
            + (f"  ({summary.get('seconds', 0):.1f} s)" if summary else "")
            + (
                f"  丢弃退化框 {self.clipped_dropped} 个"
                if self.clipped_dropped
                else ""
            )
            + ("  没有存下帧，未建目录" if not summary else "")
        )

        self.recording = False

        return summary

    # =========================================================
    # Write
    # =========================================================

    def write(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        timestamp: float,
    ) -> bool:
        """
        按 stride 存帧。

        返回 True 表示这一帧真的落盘了。

        写盘是同步的：960x720 的 JPEG 编码加写文件大约 10~20 ms，
        所以 stride 不要设成 1，
        否则采集本身会吃掉主循环的预算。
        """

        if not self.recording or self.session_dir is None:
            return False

        if self.saved >= self.config.max_frames:

            if not self._limit_hit:

                self._limit_hit = True

                self.last_error = "max_frames reached"

                print(
                    f"[Record] 到达上限 "
                    f"{self.config.max_frames} 帧，"
                    f"停止写入（再按一次 R 结束会话）"
                )

            return False

        self._frame_counter += 1

        if self._frame_counter % max(1, self.config.stride) != 0:
            return False

        if not self._ensure_dir():
            return False

        height, width = frame.shape[:2]

        self._sizes.append((width, height))

        usable = self._usable(detections, width, height)

        name = f"{self.saved:06d}"

        ok = cv2.imwrite(
            str(self.session_dir / "images" / f"{name}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )

        if not ok:

            self.last_error = "imwrite failed"

            return False

        if self.config.auto_labels:

            self._write_labels(name, usable)

        if self.config.save_preview:

            self._write_preview(name, frame, detections)

        self._append_annotation(
            name,
            usable,
            timestamp,
            width,
            height,
        )

        self.saved += 1

        return True

    # =========================================================
    # Pieces
    # =========================================================

    def _norm_box(
        self,
        detection: Detection,
        width: int,
        height: int,
    ) -> Optional[tuple[float, float, float, float]]:
        """
        帧像素 → YOLO 归一化 (x_center, y_center, w, h)。

        先按画面边界裁剪，再归一化。

        之前是直接 min/max 夹住归一化后的数，
        那等于把出框的目标整体往里挪：
        一只半只蚊子趴在画面边缘时，
        框的中心会被强行改到边上，几何就错了。
        裁剪才是标准做法 —— 留在画面里的那半截才是真目标。

        完全在画面外、或者裁到退化的框返回 None，
        这种框写进去只会教模型"这里有个不存在的东西"。
        """

        x1 = max(detection.x1, 0.0)

        y1 = max(detection.y1, 0.0)

        x2 = min(detection.x2, float(width))

        y2 = min(detection.y2, float(height))

        box_w = x2 - x1

        box_h = y2 - y1

        if box_w <= 1.0 or box_h <= 1.0:

            return None

        return (
            ((x1 + x2) / 2.0) / width,
            ((y1 + y2) / 2.0) / height,
            box_w / width,
            box_h / height,
        )

    def _usable(
        self,
        detections: list[Detection],
        width: int,
        height: int,
    ) -> list[tuple[Detection, tuple[float, float, float, float]]]:
        """
        一次裁剪，标签和 jsonl 共用同一批结果。

        分开算的话两处可能对不上：
        标签里有第 3 个框、jsonl 里没有，
        事后按 annotations 复核就会找不到。
        """

        usable = []

        for detection in detections:

            box = self._norm_box(detection, width, height)

            if box is None:

                self.clipped_dropped += 1

                continue

            usable.append((detection, box))

        return usable

    def _write_labels(
        self,
        name: str,
        usable: list[tuple[Detection, tuple[float, float, float, float]]],
    ) -> None:
        """
        YOLO 标签：每行 "class xc yc w h"，按模型分目录写。

        为什么要分开：两路模型的 class_id 会撞号 ——
        A 路的 0 是蚊子，B 路的 0 是人，混进同一个 txt
        就成了一份谁都解释不了的标签。分开存之后，
        真要合并必须是一次显式决定，而不是顺手写进同一个文件。

        没有目标时也写一个空文件 —— 这就是负样本，训练时同样有用。
        """

        grouped: dict[str, list[str]] = {}

        for detection, (cx, cy, w, h) in usable:

            source = getattr(detection, "source", "") or "model"

            grouped.setdefault(source, []).append(
                f"{detection.class_id} "
                f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            )

        for source in self.sources or list(grouped) or ["model"]:

            lines = grouped.get(source, [])

            path = self.session_dir / "labels_auto" / source / f"{name}.txt"

            path.parent.mkdir(parents=True, exist_ok=True)

            path.write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

    def _write_preview(
        self,
        name: str,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> None:
        from .draw import draw_detections

        marked = frame.copy()

        draw_detections(marked, detections)

        cv2.imwrite(
            str(self.session_dir / "preview" / f"{name}.jpg"),
            marked,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

    def _append_annotation(
        self,
        name: str,
        usable: list[tuple[Detection, tuple[float, float, float, float]]],
        timestamp: float,
        width: int,
        height: int,
    ) -> None:
        """
        一行一帧的完整记录。

        这里留着 track_id 和 bbox 像素值：
        labels.txt 里那 5 个数撑不起轨迹和误差复算。

        bbox_px 是模型原话（未裁剪），
        bbox_norm 是裁剪后、和 labels.txt 完全一致的那一份。
        两个都留：复核时想知道模型本来报的是什么，
        训练时又必须和标签对得上。
        """

        record = {
            "file": f"images/{name}.jpg",
            "index": self.saved,
            "frame": self._frame_counter,
            "timestamp": round(timestamp, 6),
            "size": [width, height],
            "detections": [
                {
                    "track_id": getattr(
                        detection,
                        "track_id",
                        None,
                    ),
                    "source": getattr(detection, "source", ""),
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "confidence": round(detection.confidence, 4),
                    "bbox_px": [
                        round(detection.x1, 1),
                        round(detection.y1, 1),
                        round(detection.x2, 1),
                        round(detection.y2, 1),
                    ],
                    "bbox_norm": [round(value, 6) for value in box],
                }
                for detection, box in usable
            ],
        }

        path = self.session_dir / "annotations.jsonl"

        with path.open("a", encoding="utf-8") as file:

            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_metadata(self, status: str) -> dict:
        """
        metadata.json。

        把"这次是在什么条件下采的"写全：
        光照、距离、分辨率、权重、配置。
        缺了这些，几个月后面对一堆图完全无法判断能不能用。
        """

        sizes = {
            f"{width}x{height}"
            for width, height in self._sizes
        }

        summary = {
            "status": status,
            "session": self.session_dir.name if self.session_dir else "",
            "started_at": datetime.now().astimezone().isoformat(),
            "saved_frames": self.saved,
            "seconds": round(self.elapsed_s, 2),
            "record_fps": round(self.fps, 2),
            "resolutions": sorted(sizes),
            "dropped_degenerate_boxes": self.clipped_dropped,
            "camera": self.camera_info,
            "model": {
                "weights": self.model_name,
                "classes": self.class_names,
            },
            "config": asdict(self.config),
            "label_format": "YOLO: class x_center y_center width height (normalized)",
            "warning": (
                "预标注来自当前权重。COCO 权重没有 mosquito 类，"
                "在换上蚊子模型之前，labels/ 里的内容不能直接用于训练。"
            ),
        }

        path = self.session_dir / "metadata.json"

        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return summary
