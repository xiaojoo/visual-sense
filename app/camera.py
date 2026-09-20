from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraConfig:
    mode: str = "usb"

    # USB 摄像头
    index: int = 0

    # 网络摄像头
    url: str = ""

    reconnect_interval: float = 2.0


class Camera:
    """
    V0.1 Camera abstraction.

    支持：

        USB:
            Redmi K40
                ↓
            DroidCam
                ↓
            Windows Camera
                ↓
            OpenCV

        Network:
            Redmi K40
                ↓
            IP Webcam
                ↓
            HTTP/MJPEG
                ↓
            OpenCV
    """

    def __init__(self, config: CameraConfig):
        self.config = config

        self._capture: Optional[cv2.VideoCapture] = None

        self.connected = False
        self.last_error = ""

        self.last_connect_time = 0.0

        self.total_frames = 0

        self.width = 0
        self.height = 0

    # =========================================================
    # Connect
    # =========================================================

    def connect(self) -> bool:
        """
        打开 USB 或网络摄像头。
        """

        self.disconnect()

        self.last_connect_time = time.perf_counter()

        try:
            if self.config.mode.lower() == "usb":
                return self._connect_usb()

            if self.config.mode.lower() == "network":
                return self._connect_network()

            self.last_error = (
                f"未知 camera mode: {self.config.mode}"
            )

            return False

        except Exception as exc:
            self.connected = False
            self.last_error = str(exc)

            return False

    # =========================================================
    # USB
    # =========================================================

    def _connect_usb(self) -> bool:
        """
        打开 Windows USB Camera。

        index=0:
            第一个摄像头

        index=1:
            第二个摄像头

        ...
        """

        print(
            f"[Camera] Opening USB camera "
            f"index={self.config.index}"
        )

        capture = cv2.VideoCapture(
            self.config.index,
            cv2.CAP_DSHOW,
        )

        if not capture.isOpened():

            print(
                "[Camera] DirectShow failed, "
                "trying default backend..."
            )

            capture.release()

            capture = cv2.VideoCapture(
                self.config.index
            )

        if not capture.isOpened():

            capture.release()

            self.connected = False

            self.last_error = (
                f"无法打开 USB 摄像头 "
                f"index={self.config.index}"
            )

            return False

        self._capture = capture

        self.connected = True

        self.last_error = ""

        print(
            f"[Camera] USB camera "
            f"{self.config.index} connected"
        )

        return True

    # =========================================================
    # Network
    # =========================================================

    def _connect_network(self) -> bool:
        """
        打开网络摄像头。
        """

        print(
            f"[Camera] Opening network camera: "
            f"{self.config.url}"
        )

        capture = cv2.VideoCapture(
            self.config.url
        )

        if not capture.isOpened():

            capture.release()

            self.connected = False

            self.last_error = (
                f"无法打开网络摄像头: "
                f"{self.config.url}"
            )

            return False

        self._capture = capture

        self.connected = True

        self.last_error = ""

        print(
            "[Camera] Network camera connected"
        )

        return True

    # =========================================================
    # Disconnect
    # =========================================================

    def disconnect(self) -> None:

        if self._capture is not None:

            try:
                self._capture.release()

            except Exception:
                pass

        self._capture = None

        self.connected = False

    # =========================================================
    # Read
    # =========================================================

    def read(
        self,
    ) -> tuple[
        bool,
        Optional[np.ndarray],
        float,
    ]:

        if (
            self._capture is None
            or not self.connected
        ):
            return False, None, 0.0

        start = time.perf_counter()

        try:

            success, frame = (
                self._capture.read()
            )

        except Exception as exc:

            self.connected = False

            self.last_error = str(exc)

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000.0

            return False, None, elapsed

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000.0

        if (
            not success
            or frame is None
        ):

            self.connected = False

            self.last_error = (
                "摄像头读取失败"
            )

            return False, None, elapsed

        self.total_frames += 1

        self.height, self.width = (
            frame.shape[:2]
        )

        return (
            True,
            frame,
            elapsed,
        )

    # =========================================================
    # Reconnect
    # =========================================================

    def should_reconnect(self) -> bool:

        if self.connected:
            return False

        now = time.perf_counter()

        return (
            now - self.last_connect_time
            >= self.config.reconnect_interval
        )

    def reconnect(self) -> bool:

        if not self.should_reconnect():
            return False

        return self.connect()

    # =========================================================
    # Buffer
    # =========================================================

    def set_buffer_size(
        self,
        size: int,
    ) -> None:

        if self._capture is None:
            return

        try:

            self._capture.set(
                cv2.CAP_PROP_BUFFERSIZE,
                size,
            )

        except Exception:
            pass

    # =========================================================
    # Resolution
    # =========================================================

    def get_resolution(
        self,
    ) -> tuple[int, int]:

        return (
            self.width,
            self.height,
        )

    # =========================================================
    # Context manager
    # =========================================================

    def __enter__(self):

        self.connect()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.disconnect()