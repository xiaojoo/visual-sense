"""
V0.5 激光驱赶 / 灭杀模块的状态机与驱动接缝。

两条硬规矩，写在代码里而不是写在文档里：

1. **绝不让界面假装硬件在线。**
   没有驱动时所有命令都返回 unavailable，状态灯是灰的。
   一个显示"已发射"但实际什么都没接的按钮，
   比没有按钮危险得多。

2. **发射必须有第二个条件。**
   单按钮 = 单故障点。这里要求 驱动就绪 + 互锁闭合 + 已解算出目标，
   三者同时成立才允许进入 firing，且有最长停留时间。
   激光对着的是会飞进人眼高度的小虫，这个设备默认按"能伤眼"对待。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# 状态
OFFLINE = "offline"        # 没有驱动，或驱动报不在线
IDLE = "idle"              # 驱动就绪，未解锁
ARMED = "armed"            # 已解锁，可以发射
FIRING = "firing"          # 正在出光
FAULT = "fault"            # 驱动报错，必须手动复位


@dataclass
class LaserConfig:
    enabled: bool = False

    # backend 决定接哪个驱动：none / sim / 以后加的串口子类
    backend: str = "none"

    # 单次连续发射上限（秒）。到点自动收光，不等人。
    max_dwell_s: float = 1.5

    # 两次发射之间的最小间隔（秒），留给散热和重新锁定
    cooldown_s: float = 1.0

    # 心跳超时：页面/上位机失联这么久就自动锁死并收光。
    # 浏览器标签被手机切到后台是常事，不能指望它一定点到"急停"。
    watchdog_s: float = 3.0

    # 像素 → 云台角度的标定还没做，这里只存标称值，
    # 有值也不代表能真的对准。
    aperture_mm: float = 0.0
    power_w: float = 0.0


class LaserDriver:
    """
    驱动接口。真硬件实现这个类即可，上层不用改。

    simulated 的默认值是 True：
    新写的驱动如果忘了声明，界面会把它当成假的，
    这比把它当成真的按下去要安全。
    """

    name = "abstract"
    simulated = True

    def connect(self) -> bool:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def healthy(self) -> bool:
        raise NotImplementedError

    def set_aim(self, x_norm: float, y_norm: float) -> None:
        """x_norm / y_norm 是 0~1 的画面归一化坐标。"""
        raise NotImplementedError

    def emit(self, on: bool) -> None:
        raise NotImplementedError

    def fault(self) -> str:
        return ""


class NullDriver(LaserDriver):
    """
    没配硬件时的默认驱动：什么都不做，且明确说自己在装死。

    simulated=False —— 它不是在"模拟出光"，是根本没有这个东西。
    把"没有"标成"假的"，页面上就会显示"模拟驱动"，
    让人误以为接上就能用。
    """

    name = "none"
    simulated = False

    def connect(self) -> bool:
        return False

    def disconnect(self) -> None:
        return None

    def healthy(self) -> bool:
        return False

    def set_aim(self, x_norm: float, y_norm: float) -> None:
        return None

    def emit(self, on: bool) -> None:
        return None

    def fault(self) -> str:
        return "未配置激光驱动（laser.backend = none）"


class SimDriver(LaserDriver):
    """
    演示驱动：把状态记在内存里，让界面能整条链路走通。

    它存在的唯一理由是"没硬件也能验收交互"。
    simulated=True 会一路带到页面上 ——
    任何显示"已出光"的地方都必须能一眼看出那是假的。
    """

    name = "sim"
    simulated = True

    def __init__(self) -> None:
        self.online = False
        self.emitting = False
        self.aim = (0.5, 0.5)
        self._error = ""

    def connect(self) -> bool:
        self.online = True
        self._error = ""
        return True

    def disconnect(self) -> None:
        self.online = False
        self.emitting = False

    def healthy(self) -> bool:
        return self.online

    def set_aim(self, x_norm: float, y_norm: float) -> None:
        self.aim = (x_norm, y_norm)

    def emit(self, on: bool) -> None:
        if not self.online:
            self._error = "模拟驱动未连接"
            return
        self.emitting = on

    def fault(self) -> str:
        return self._error


DRIVERS: dict[str, type[LaserDriver]] = {
    "none": NullDriver,
    "sim": SimDriver,
}


@dataclass
class LaserController:
    config: LaserConfig
    _driver: LaserDriver = field(default=None, repr=False)
    _state: str = field(default=OFFLINE, repr=False)
    _since: float = field(default=0.0, repr=False)
    _last_emit_at: float = field(default=0.0, repr=False)
    _last_beat_at: float = field(default=0.0, repr=False)
    _error: str = ""

    def __post_init__(self) -> None:
        backend = self.config.backend if self.config.enabled else "none"
        self._driver = DRIVERS.get(backend, NullDriver)()
        self._since = time.perf_counter()

    # -----------------------------------------------------
    # 属性
    # -----------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def simulated(self) -> bool:
        return self._driver.simulated

    @property
    def available(self) -> bool:
        return self._driver.healthy()

    @property
    def seconds_in_state(self) -> float:
        return time.perf_counter() - self._since

    # -----------------------------------------------------
    # 命令
    # -----------------------------------------------------

    def connect(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "激光模块在配置里是关闭的"

        if self._driver.connect():
            self._goto(IDLE)
            return True, f"{self._driver.name} 驱动已连接"

        self._goto(OFFLINE)
        return False, self._driver.fault() or "驱动无法连接"

    def arm(self) -> tuple[bool, str]:
        if self._state == OFFLINE:
            return False, "驱动未连接，无法解锁"

        if self._state == FAULT:
            return False, "处于故障状态，必须先复位"

        if not self._driver.healthy():
            self._goto(FAULT)
            return False, "驱动自检不通过"

        self._goto(ARMED)
        return True, "已解锁"

    def disarm(self) -> tuple[bool, str]:
        self._driver.emit(False)
        self._goto(IDLE)
        return True, "已上锁"

    def fire(self, target: tuple[float, float] | None) -> tuple[bool, str]:
        """
        target 是 (x_norm, y_norm, track_id) 里的前两项；
        None 表示当前没有可打的目标。
        """

        if self._state != ARMED:
            return False, f"未解锁（当前 {self._state}），拒绝发射"

        if target is None:
            return False, "当前没有锁定目标，拒绝发射"

        if self._state != FIRING and time.perf_counter() - self._last_emit_at < self.config.cooldown_s:
            return False, f"冷却中，还需 {self.config.cooldown_s - (time.perf_counter() - self._last_emit_at):.1f} s"

        self._driver.set_aim(target[0], target[1])
        self._driver.emit(True)
        self._last_beat_at = time.perf_counter()
        self._goto(FIRING)

        return True, "出光中"

    def stop(self) -> tuple[bool, str]:
        """急停：收光并退回 armed，不解锁。任何状态都能调。"""

        self._driver.emit(False)

        if self._state == FIRING:
            self._last_emit_at = time.perf_counter()
            self._goto(ARMED)

        return True, "已收光"

    def reset(self) -> tuple[bool, str]:
        if self._state != FAULT:
            return False, "当前不是故障状态"

        self._error = self._driver.fault()
        self._goto(IDLE if self._driver.healthy() else OFFLINE)

        return True, "已复位"

    def heartbeat(self) -> None:
        self._last_beat_at = time.perf_counter()

    def disconnect(self) -> tuple[bool, str]:
        """收光 + 断驱动。退出时无论处于哪个状态都必须走到这里。"""

        self._driver.emit(False)
        self._driver.disconnect()
        self._goto(OFFLINE)

        return True, "已断开"

    # -----------------------------------------------------
    # 每帧
    # -----------------------------------------------------

    def tick(self, target: tuple[float, float] | None = None) -> None:
        """
        由管线线程每帧调用。

        自动收光的三个条件，任何一个成立就收：
        超过最长停留、目标丢了、上位机失联。
        全部是"往安全方向退化"，不需要谁去确认。
        """

        if self._state == FIRING:
            over_dwell = self.seconds_in_state >= self.config.max_dwell_s
            lost_target = target is None
            stale_beat = (
                self.config.watchdog_s > 0
                and self._last_beat_at > 0
                and time.perf_counter() - self._last_beat_at > self.config.watchdog_s
            )

            if over_dwell or lost_target or stale_beat:
                self._driver.emit(False)
                self._last_emit_at = time.perf_counter()
                self._goto(ARMED)
                self._error = (
                    "已达最长停留，自动收光" if over_dwell
                    else "目标丢失，自动收光" if lost_target
                    else "上位机失联，自动收光"
                )

        if self._state not in (OFFLINE, FAULT) and not self._driver.healthy():
            self._driver.emit(False)
            self._error = self._driver.fault() or "驱动掉线"
            self._goto(FAULT)

    # -----------------------------------------------------

    def _goto(self, state: str) -> None:
        self._state = state
        self._since = time.perf_counter()

    def snapshot(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "state": self._state,
            "available": self.available,
            "simulated": self.simulated,
            "backend": self._driver.name,
            "seconds_in_state": round(self.seconds_in_state, 1),
            "max_dwell_s": self.config.max_dwell_s,
            "cooldown_s": self.config.cooldown_s,
            "power_w": self.config.power_w,
            "aperture_mm": self.config.aperture_mm,
            "calibrated": False,
            "error": self._error,
        }
