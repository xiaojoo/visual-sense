"""
管线线程和 Web 线程之间唯一的交界处。

为什么要有这个文件：
原来所有状态都是主循环的局部变量，窗口每帧直接读。
换成浏览器之后有两个线程，"谁持有状态"必须说清楚，
否则就是隔着线程抢 numpy 数组。

规矩很简单：**数据单向流**。
管线线程只往 Hub 里写（最新帧 + 最新快照 + 历史采样），
Web 线程只读，并且读到的都是拷贝或不可变值。
反方向只有命令，走 CommandQueue，由管线线程自己在帧间消费，
所以检测器、录制器、激光这些对象始终只被一个线程碰。
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Hub:
    """管线 → Web 的单向数据出口。"""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _jpeg: bytes = field(default=b"", repr=False)
    _frame_seq: int = field(default=0, repr=False)
    _frame_at: float = field(default=0.0, repr=False)
    _snapshot: dict = field(default_factory=dict, repr=False)
    _history: deque = field(default_factory=lambda: deque(maxlen=240), repr=False)
    _log: deque = field(default_factory=lambda: deque(maxlen=200), repr=False)
    _events: int = field(default=0, repr=False)

    # -----------------------------------------------------
    # 写侧（管线线程）
    # -----------------------------------------------------

    def publish_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._frame_seq += 1
            self._frame_at = time.time()

    def publish_snapshot(self, snapshot: dict) -> None:
        with self._lock:
            self._snapshot = snapshot
            self._history.append(
                (
                    snapshot.get("uptime_s", 0.0),
                    round(snapshot.get("metrics", {}).get("fps", 0.0), 1),
                    round(snapshot.get("metrics", {}).get("infer_ms", 0.0), 1),
                    snapshot.get("targets", []),
                )
            )

    def log(self, message: str) -> None:
        """
        给页面看的操作日志。

        原来所有反馈都打在终端上，换成浏览器之后没人看得见终端。
        点了"解锁激光"必须有个地方告诉你结果，这里就是那个地方。

        连续相同的内容只留一条并计数：摄像头不在线时每 2 秒重连一次，
        不去重的话日志会被同一句话刷满，真正的事件反而看不见。
        """

        with self._lock:
            if self._log and self._log[0]["text"] == message:
                self._log[0]["count"] += 1
                self._log[0]["at"] = round(time.time(), 3)
                return

            self._events += 1
            self._log.appendleft(
                {"at": round(time.time(), 3), "text": message, "seq": self._events, "count": 1}
            )

    # -----------------------------------------------------
    # 读侧（Web 线程）
    # -----------------------------------------------------

    def latest_jpeg(self) -> tuple[bytes, int]:
        with self._lock:
            return self._jpeg, self._frame_seq

    def wait_frame(self, after_seq: int, timeout: float = 1.0) -> tuple[bytes, int]:
        """
        阻塞等到比 after_seq 更新的一帧。

        MJPEG 不能像桌面窗口那样"每帧都画"，
        否则浏览器慢一点就开始囤积旧帧，看起来像画面卡死。
        这里让推流端跟着新帧走，旧帧直接跳过。
        """

        deadline = time.perf_counter() + timeout

        while time.perf_counter() < deadline:
            jpeg, seq = self.latest_jpeg()
            if seq > after_seq:
                return jpeg, seq
            time.sleep(0.008)

        return b"", after_seq

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snapshot)

    def history(self) -> list:
        with self._lock:
            return list(self._history)

    def events(self, limit: int = 40) -> list:
        """
        最近若干条，新→旧。

        不做增量游标：去重会把已有条目的计数改写，
        客户端按 seq 追加就永远看不到改写后的值。
        整段返回、整段重建，反而没有这类对不上的可能。
        """

        with self._lock:
            return [dict(e) for e in list(self._log)[:limit]]

    def clear_log(self) -> None:
        with self._lock:
            self._log.clear()

    @property
    def frame_age_s(self) -> float:
        with self._lock:
            return 0.0 if not self._frame_at else time.time() - self._frame_at


class CommandQueue:
    """
    Web → 管线 的命令通道。

    命令不在 Web 线程执行，只入队，
    由管线线程在帧与帧之间取出来跑。
    所以 detector / recorder / laser 永远只被一个线程碰，
    不需要给它们加锁 —— 这也是原来键盘事件的处理方式。
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, dict, Callable[[dict], None]]] = queue.Queue()

    def submit(self, name: str, **args) -> dict:
        """
        同步提交并等结果。

        页面点一下按钮要立刻看到成败，不能等下一帧。
        超时也要返回一个结构，不能让调用方去接一个异常 ——
        管线正在加载模型时卡十几秒是正常情况。
        """

        reply: queue.Queue[dict] = queue.Queue()
        self._queue.put((name, args, reply.put))

        try:
            return reply.get(timeout=20.0)
        except queue.Empty:
            return {
                "ok": False,
                "message": "管线没有响应（可能正在加载模型或摄像头），请稍后再试",
            }

    def drain(self, handler: Callable[[str, dict], dict]) -> None:
        """管线线程调用：把排队的命令跑完，结果回给提交方。"""

        while True:
            try:
                name, args, reply = self._queue.get_nowait()
            except queue.Empty:
                return
            reply(handler(name, args))


class Shutdown:
    """一个事件，两处等：Ctrl+C 和页面上的"停止"。"""

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self) -> None:
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)
