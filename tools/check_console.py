"""
V0.5 控制台自检：激光安全性质 + 界面文案完整性。

为什么这两件事值得断言：

1. **激光的"不许发射"是安全边界，不是功能开关。**
   它一旦被改错，界面上看不出来 —— 按钮还是灰的、状态还是那几档，
   只有真的接上硬件才会发现它锁不住。所以这里把每一条拒绝路径钉死。

2. **中英两套文案会各自漂移。**
   加一个键只写中文，切到英文就露出 key 名，
   这种事在代码评审里看不见，只有比长度才会。

跑法：
    .venv\\Scripts\\python.exe -m tools.check_console
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.laser import (  # noqa: E402
    ARMED,
    FAULT,
    FIRING,
    IDLE,
    OFFLINE,
    LaserConfig,
    LaserController,
)


def make(**over) -> LaserController:
    config = LaserConfig(enabled=True, backend="sim", max_dwell_s=0.2, cooldown_s=0.0, watchdog_s=0.3)

    for key, value in over.items():
        setattr(config, key, value)

    return LaserController(config)


def refusals() -> list[str]:
    """每一条"应该拒绝"的路径都必须真的拒绝。"""

    problems: list[str] = []

    def expect(condition: bool, text: str) -> None:
        if not condition:
            problems.append(text)

    # 没接驱动：arm 必须被拒
    dead = LaserController(LaserConfig(enabled=True, backend="none"))
    ok, message = dead.arm()
    expect(not ok and dead.state == OFFLINE, f"未连接时 arm 竟然通过了（{message}）")

    ok, message = dead.fire((0.5, 0.5))
    expect(not ok, f"未连接时 fire 竟然通过了（{message}）")

    # 未解锁：fire 必须被拒
    live = make()
    live.connect()
    expect(live.state == IDLE, f"connect 之后不是 idle，而是 {live.state}")

    ok, message = live.fire((0.5, 0.5))
    expect(not ok and live.state == IDLE, f"未解锁时 fire 竟然通过了（{message}）")

    # 解锁了但没有目标：仍然必须被拒
    live.arm()
    expect(live.state == ARMED, f"arm 之后不是 armed，而是 {live.state}")

    ok, message = live.fire(None)
    expect(not ok and live.state == ARMED, f"无目标时 fire 竟然通过了（{message}）")

    # 有目标才允许出光
    ok, message = live.fire((0.4, 0.6))
    expect(ok and live.state == FIRING, f"armed + 有目标却发不出去（{message}）")
    expect(live._driver.emitting, "fire 返回成功但驱动没有出光")

    # 最长停留：超时必须自动收光并退回 armed
    deadline = time.perf_counter() + 2.0
    while live.state == FIRING and time.perf_counter() < deadline:
        live.tick((0.4, 0.6))

    expect(live.state == ARMED, f"超过 max_dwell 没有自动收光，停在 {live.state}")
    expect(not live._driver.emitting, "自动收光之后驱动仍在出光")

    # 心跳失联：armed 期间不心跳，出光必须起不来或立刻收
    watch = make()
    watch.connect()
    watch.arm()
    watch.fire((0.5, 0.5))
    time.sleep(watch.config.watchdog_s + 0.05)
    watch.tick((0.5, 0.5))
    expect(watch.state == ARMED and not watch._driver.emitting,
           f"失联 {watch.config.watchdog_s}s 后仍在出光（{watch.state}）")

    # 目标中途消失：必须立刻收光
    lose = make()
    lose.connect()
    lose.arm()
    lose.fire((0.5, 0.5))
    lose.tick(None)
    expect(lose.state == ARMED and not lose._driver.emitting, "目标丢失后仍在出光")

    # 急停在任何状态都能收光
    stop = make()
    stop.connect()
    stop.arm()
    stop.fire((0.5, 0.5))
    ok, message = stop.stop()
    expect(ok and stop.state == ARMED and not stop._driver.emitting, f"急停没收光（{message}）")

    return problems


def copy_parity() -> list[str]:
    """HTML 用到的每个键，中英两套都得有。"""

    problems: list[str] = []

    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

    used = set(re.findall(r'data-i18n="([^"]+)"', html))

    # t("...") 的调用点，排除掉同名参数会撞上的内置字符串
    called = set(re.findall(r'(?<![A-Za-z0-9_$.])t\("([^"]+)"\)', js))
    used |= called

    blocks = {}
    for lang in ("zh", "en"):
        start = js.index(f'  {lang}: {{')
        end = js.index("\n  },", start)
        # 不能用 ^\s*"key"：字典里一行写好几个键，那样每行只数得到第一个
        blocks[lang] = set(re.findall(r'"([a-z0-9.]+)":\s*"', js[start:end]))

    for lang in ("zh", "en"):
        missing = sorted(used - blocks[lang])
        if missing:
            problems.append(f"{lang} 缺 {len(missing)} 个键：{missing[:6]}")

    only_zh = sorted(blocks["zh"] - blocks["en"])
    only_en = sorted(blocks["en"] - blocks["zh"])

    if only_zh or only_en:
        problems.append(f"中英键集不等：只在 zh {only_zh[:5]} 只在 en {only_en[:5]}")

    dead = sorted(blocks["zh"] - used)
    if dead:
        problems.append(f"定义了但没人用：{dead[:8]}")

    return problems


def driver_honesty() -> list[str]:
    """
    驱动必须如实声明自己是不是假的。

    这条不是洁癖：模拟驱动曾经把 simulated 写成 False，
    于是页面上"已出光"看起来和接了真硬件一模一样。
    界面对硬件存在与否的任何误报，都会直接变成操作误判。
    """

    problems: list[str] = []

    from app.laser import NullDriver, SimDriver

    if not SimDriver().simulated:
        problems.append("SimDriver.simulated 不是 True —— 页面会以为光是真的")

    if NullDriver().simulated:
        problems.append("NullDriver.simulated 不是 False —— 没接驱动会被显示成模拟驱动")

    from app.laser import LaserDriver

    if LaserDriver.simulated is not True:
        problems.append("LaserDriver 基类默认应为 True：新驱动忘了声明就当假的")

    sim = LaserController(LaserConfig(enabled=True, backend="sim"))
    sim.connect()
    sim.arm()

    if not sim.snapshot()["simulated"]:
        problems.append("sim 后端的快照没带上 simulated")

    none = LaserController(LaserConfig(enabled=True, backend="none"))
    none.connect()

    snapshot = none.snapshot()

    if snapshot["state"] != OFFLINE:
        problems.append(f"none 后端连接后状态是 {snapshot['state']}，应为 offline")

    if snapshot["available"]:
        problems.append("none 后端报告 available=True")

    return problems


def model_gating() -> list[str]:
    """
    多路模型的两条硬性质。

    1. **没标 targetable 的那一路，对激光必须完全不可见。**
       这条是安全边界，不是偏好：通用框架里接上"人"这个模型之后，
       如果它的框还能驱动云台，那激光就会去追人。
    2. **两路模型的 track_id 撞号时不能合并。**
       ByteTrack 每路都从 1 开始编号，合并就等于把"蚊子#1"和"人#1"
       当成同一个目标 —— 轨迹串味、预测拿错历史、锁错东西。
    """

    problems: list[str] = []

    from app.detector import Detection, Detector, DetectorConfig, Track
    from app.detectors import Bundle, ModelGroup, PredictConfig
    from app.motion import MotionPredictor
    from app.tracks import Tracks

    def bundle(key: str, targetable: bool) -> Bundle:
        config = DetectorConfig(key=key, targetable=targetable)
        return Bundle(
            config=config,
            predict_config=PredictConfig(),
            detector=Detector(config, root=ROOT),
            trails=Tracks(),
            predictor=MotionPredictor(),
        )

    fly = bundle("fly", True)
    person = bundle("person", False)

    # 两路都拿到 #1 —— ByteTrack 每路独立编号，这正是会撞的那个号
    fly.last_detections = [Track(10.0, 10.0, 40.0, 60.0, 0.5, 0, "mosquito", source="fly", track_id=1)]
    person.last_detections = [Track(20.0, 20.0, 300.0, 400.0, 0.9, 0, "person", source="person", track_id=1)]

    group = ModelGroup([fly, person])
    visible = group.targetable_detections()

    if len(visible) != 1:
        problems.append(f"可驱动目标应有 1 个（只有 fly），实际 {len(visible)} 个")

    if any(d.class_name == "person" for d in visible):
        problems.append("person 这一路没标 targetable，却出现在激光候选里")

    # 两路各自的跟踪器互不知情，同一个数字必须能同时存在
    fly.trails.update(fly.last_detections, 1, 0.0)
    person.trails.update(person.last_detections, 1, 0.0)

    fly.trails.update(fly.last_detections, 2, 0.02)
    person.trails.update(person.last_detections, 2, 0.02)

    if len(fly.trails) != 1 or len(person.trails) != 1:
        problems.append(
            f"两路各自应有 1 条轨迹，实际 fly={len(fly.trails)} person={len(person.trails)}"
        )

    if fly.trails is person.trails:
        problems.append("两路共用了同一个 Tracks 对象")

    # 各自的两条 #1 轨迹必须互不污染：fly 只该看见小框那条
    fly_points = fly.trails.points(1)
    person_points = person.trails.points(1)

    if len(fly_points) != 2 or len(person_points) != 2:
        problems.append(
            f"#1 的轨迹点数不对：fly={len(fly_points)} person={len(person_points)}，"
            "两路的同号目标被混进同一条轨迹了"
        )

    # 关掉唯一可驱动的那一路之后，激光应该没有目标可锁
    fly.config.targetable = False

    if group.targetable_detections():
        problems.append("把 targetable 全关掉后仍有可驱动目标")

    return problems


# 卡片里"其余内容"的缩进规则，选择器要和 app.css 里写的那几条一字不差
CARD_INSET = (
    ".card > :not(header):not(.table-scroll):not(.card-body)",
    ".card-body > :not(.table-scroll)",
)

# 表格容器不留左右内边距（钉住的首末列要真的贴到卡片边缘），
# 左右这一维改由首末格自己的 padding 提供。
TABLE_EDGE = {
    "表格首格": (".table-scroll .table td:first-child", ".table-scroll .table th:first-child"),
    "表格末格": (".table-scroll .table td:last-child", ".table-scroll .table th:last-child"),
}

FIT_MEDIA = "(min-width: 1181px)"


def _split_selectors(selector: str) -> list[str]:
    """按逗号切选择器组，但 :not(a, b) 括号里的逗号不算分隔。"""

    out: list[str] = []
    depth = 0
    current = ""

    for char in selector:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

        if char == "," and depth == 0:
            out.append(" ".join(current.split()))
            current = ""
        else:
            current += char

    out.append(" ".join(current.split()))

    return [one for one in out if one]


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _flatten(css: str) -> list[tuple[str, str, dict[str, str]]]:
    """
    样式表摊平成有序的 (作用域, 选择器, {属性: 值})。

    作用域为 "" 是顶层规则，否则是它所在 @media / @supports 的条件。
    """

    rules: list[tuple[str, str, dict[str, str]]] = []

    def walk(scope: str, chunk: str) -> None:
        index = 0

        while True:
            brace = chunk.find("{", index)

            if brace < 0:
                return

            end = brace + 1
            depth = 1

            while end < len(chunk) and depth:
                if chunk[end] == "{":
                    depth += 1
                elif chunk[end] == "}":
                    depth -= 1
                end += 1

            selector = " ".join(chunk[index:brace].split())
            inner = chunk[brace + 1:end - 1]

            if selector.startswith(("@media", "@supports")):
                walk(selector, inner)
            elif selector:
                decls: dict[str, str] = {}

                for line in inner.split(";"):
                    if ":" in line:
                        prop, _, value = line.partition(":")
                        decls[" ".join(prop.split()).lower()] = value.strip()

                for one in _split_selectors(selector):
                    rules.append((scope, one, decls))

            index = end

    walk("", _strip_comments(css))

    return rules


def _parts(value: str) -> list[str]:
    """按顶层空白切 shorthand，calc(...) 内部不切。"""

    parts: list[str] = []
    depth = 0
    current = ""

    for char in value:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

        if char.isspace() and depth == 0:
            if current:
                parts.append(current)
                current = ""
        else:
            current += char

    if current:
        parts.append(current)

    return parts


def _inline_of(prop: str, value: str | None) -> list[str]:
    """
    一条 margin/padding 声明里"横向"那几段。

    shorthand 的四段是 上 右 下 左：一段时全向，两段时第二段是左右，
    三段时第二段是左右，四段时取第一下标 1 和 3。
    竖向那两段不算 —— 负的竖向边距长不出横向滚动条，是设计里有的手法。
    """

    if not value:
        return []

    parts = _parts(value)

    if not parts:
        return []

    if prop.endswith(("left", "right", "inline")):
        return parts

    if len(parts) == 1:
        return parts

    if len(parts) == 4:
        return [parts[1], parts[3]]

    return [parts[1]]


def _is_negative(part: str) -> bool:
    return bool(re.match(r"-[\d.]", part)) or bool(re.search(r"calc\(\s*-", part))


def _var_name(part: str | None) -> str | None:
    """var(--x) → --x；写死的数字或 calc 原样返回，好让断言报出"这里写死了"。"""

    if part is None:
        return None

    match = re.fullmatch(r"var\((--[\w-]+)\)", part)

    return match.group(1) if match else part


def _last_value(
    rules: list[tuple[str, str, dict[str, str]]],
    selector: str,
    prop: str,
) -> str | None:
    value = None

    for _, sel, decls in rules:
        if sel == selector and prop in decls:
            value = decls[prop]

    return value


def _spacing_part(
    rules: list[tuple[str, str, dict[str, str]]],
    chain: list[str],
    kind: str,
    side: str,
) -> str | None:
    """
    一条链上（.card → .modules .card）取横向某一边真正生效的那一段值。

    后写的选择器更具体，压得过前写的；shorthand 之后还能被 longhand 覆盖。
    返回原始那一段（带 calc 和负号），符号信息要留着 —— 判断"是不是负边距"靠它。
    """

    value = None

    for selector in chain:
        shorthand = _last_value(rules, selector, kind)

        if shorthand is not None:
            parts = _parts(shorthand)

            if parts:
                if len(parts) == 1:
                    value = parts[0]
                elif len(parts) == 4:
                    value = parts[3 if side == "left" else 1]
                else:
                    value = parts[1]

        inline = _last_value(rules, selector, f"{kind}-inline")

        if inline is not None:
            value = _parts(inline)[0] if _parts(inline) else value

        longhand = _last_value(rules, selector, f"{kind}-{side}")

        if longhand is not None:
            value = longhand

    return value


def _pairing_problems(css: str) -> list[str]:
    """
    卡片这一层不许再出现"负外边距顶到边缘"的写法。

    以前是卡片左右各留 16px，标题栏和表格用 -16 的负边距抵消回去，
    两处数字得手动对齐 —— 我就是这么踩了两次横向滚动条（右栏一次、模块行一次），
    人眼还看不出来。现在改成：卡片不留左右内边距，贴边的元素自己带 padding，
    其余子元素用 margin-inline 缩进，三处都引用同一个 --card-pad，
    "对不齐"在写法上就不成立了。所以这里查三件事：

      1. .card 不能有左右内边距（有了就得靠负边距找齐，等于把坑挖回来）；
      2. 卡片相关的选择器里，任何地方都不许出现负的横向外边距（竖向不管，
         竖向负边距长不出横向滚动条）；
      3. 标题栏留白、表格留白、内容缩进必须是同一个 var() ——
         写死数字就等于换号的时候漏改一处。
    """

    rules_all = _flatten(css)
    problems: list[str] = []

    # 1) 和 2) 不分模式，整张表扫一遍：这两条是"永远不许这样写"
    for scope, selector, decls in rules_all:
        tag = scope or "顶层"

        if selector == ".card":
            pad = (
                _inline_of("padding", decls.get("padding"))
                + _inline_of("padding-inline", decls.get("padding-inline"))
                + _inline_of("padding-left", decls.get("padding-left"))
                + _inline_of("padding-right", decls.get("padding-right"))
            )

            if any(part != "0" for part in pad):
                problems.append(
                    f"{tag} .card 有了左右内边距 {pad} —— 贴边的子元素就得用负边距找齐，"
                    "那个坑会回来。左右这一维交给子元素"
                )

        if "card" not in selector and "table-scroll" not in selector:
            continue

        for prop in ("margin", "margin-left", "margin-right", "margin-inline"):
            parts = _inline_of(prop, decls.get(prop))

            if any(_is_negative(part) for part in parts):
                problems.append(
                    f"{tag} {selector} 的 {prop}: {decls[prop]} 用了负外边距 —— "
                    "横向贴边请改成引用 --card-pad"
                )

    # 3) 三处横向值必须是同一个 var()，按视口模式各判一次
    for mode, media in (("基线（窄/矮屏）", None), ("贴合视口", FIT_MEDIA)):
        rules = [
            rule for rule in rules_all
            if rule[0] == "" or (media is not None and media in rule[0])
        ]

        for name, selectors, kind, sides, want in (
            ("标题栏", (".card > header",), "padding", ("left", "right"), "--card-pad"),
            ("内容缩进", CARD_INSET, "margin", ("left", "right"), "--card-pad"),
            ("表格首格", TABLE_EDGE["表格首格"], "padding", ("left",), "--card-pad"),
            ("表格末格", TABLE_EDGE["表格末格"], "padding", ("right",), "--card-pad"),
            # 表格容器必须是 0：留了内边距，钉住列和卡片边缘之间就有一条缝，
            # 横向滚的时候内容会从缝里露出来。
            ("表格外层", (".table-scroll",), "padding", ("left", "right"), "0"),
        ):
            values = {
                _var_name(_spacing_part(rules, [one], kind, side))
                for one in selectors
                for side in sides
            }

            got = values.pop() if len(values) == 1 else f"内部不一致 {sorted(values)}"

            if got != want:
                problems.append(
                    f"{mode} {name} 的横向留白是 {got}，应该是 {want} —— "
                    "这套值只有 --card-pad 一个来源，写死数字或漏掉一处就会对不齐"
                )

    return problems


def css_pairing() -> list[str]:
    """
    跑一次，并且顺手证明这次跑不是空转。

    反证：把 .card > header 改回旧的负边距写法，必须报错。
    上一版这里就是个假断言 —— 它按整个 @media 块做集合相减，
    块里只要还剩一处配套的负边距就算通过，真滚动条还在屏幕上的时候它报了 OK。
    """

    css = (ROOT / "web" / "app.css").read_text(encoding="utf-8")

    problems = _pairing_problems(css)

    broken, removed = re.subn(
        r"(\.card > header \{[^}]*?)margin: 0 0 var\(--s-3\);",
        r"\1margin: calc(-1 * var(--s-4)) calc(-1 * var(--s-4)) var(--s-3);",
        css,
        count=1,
    )

    if removed != 1:
        problems.append(
            f"反证样本造不出来（改写了 {removed}/1 处标题栏 margin）—— "
            "要么样式已改名，要么这条检查已经抓不到真问题"
        )
    elif not _pairing_problems(broken):
        problems.append("把标题栏改回负边距写法后仍然通过 —— 断言是假的")

    return problems


def endpoints() -> list[str]:
    """
    服务在跑就查，没在跑就跳过。

    自检不该为了测试去抢 8060 端口 ——
    他很可能正开着控制台在调别的参数。
    """

    import json
    from urllib.error import URLError
    from urllib.request import urlopen

    try:
        with urlopen("http://127.0.0.1:8060/api/snapshot", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, OSError):
        return []

    want = {"camera", "ai", "metrics", "record", "laser", "targets", "stream"}

    missing = want - set(payload)

    return [f"/api/snapshot 缺字段：{sorted(missing)}"] if missing else []


def main() -> int:
    checks = [("激光安全性质", refusals), ("驱动如实声明自己不是真硬件", driver_honesty),
              ("多路模型的激光门控", model_gating),
              ("卡片不再用负边距贴边", css_pairing),
              ("界面文案完整性", copy_parity), ("在线端点", endpoints)]

    print()
    print("=" * 60)
    print("  控制台自检")
    print("=" * 60)

    total = 0

    for name, run in checks:
        problems = run()
        total += len(problems)
        print(f"  {'OK  ' if not problems else 'FAIL'}  {name}")
        for item in problems:
            print(f"        - {item}")

    print("-" * 60)

    if total:
        print(f"[SELFTEST FAIL] {total} 项不达标")
        return 1

    print("[SELFTEST] 控制台自检 全部达标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
