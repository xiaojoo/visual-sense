"""
一条命令跑完全部自检。

    .venv\\Scripts\\python.exe -m tools.selftest             # 全跑（含要 GPU 的 4 项）
    .venv\\Scripts\\python.exe -m tools.selftest --stream    # 再加线上手机流实测
    .venv\\Scripts\\python.exe -m tools.selftest --verbose   # 把每项的原始输出也打出来

退出码 0 = 没有 FAIL。SKIP 不算失败，但会写在表里。

这里刻意不叫"测试"：项目里没有 pytest，也没有 tests/。
这一层跑的是**已经建成断言的那几项**，其余是跑通检查 ——
表上全绿只说明没崩，不说明行为对。所以每一项都带一个 marker：
**退出码 0 但没打出该有的那行字，一样算 FAIL**，
免得一条静默跳过的检查冒充成一条通过的检查。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TIMEOUT_S = 900.0


def venv_python(name: str) -> Path:
    """
    .venv / .venv-ai 里的解释器。
    """

    windows = sys.platform == "win32"

    return (
        ROOT
        / name
        / ("Scripts" if windows else "bin")
        / ("python.exe" if windows else "python")
    )


@dataclass
class Check:
    """
    一项自检。

    marker 是这层唯一能证明"断言真的跑了"的东西，
    所以带断言的检查项必须填。
    """

    name: str

    env: str  # light | ai

    args: list[str]

    marker: str = ""

    # 输出里出现这段话就算 SKIP 而不是 FAIL：
    # 有些项要外部硬件，硬件不在不是代码的错
    skip_if: str = ""


def code_checks() -> list[Check]:
    """
    不需要模型的部分：轻环境里就能跑完。
    """

    return [
        Check(
            name="编译 app/ + tools/",
            env="light",
            args=["-m", "compileall", "-q", "app", "tools"],
        ),
        Check(
            name="模块导入（无 torch）",
            env="light",
            args=[
                "-c",
                "import app.camera, app.draw, app.metrics, app.motion, "
                "app.recorder, app.tracks, app.render, app.runtime, app.hub, "
                "app.laser, app.pipeline, app.web, app.server; print('ok')",
            ],
        ),
        Check(
            name="Detector 缺 torch 时降级",
            env="light",
            args=[
                "-c",
                "from pathlib import Path;"
                "from app.detector import Detector, DetectorConfig;"
                "import numpy as np;"
                "d = Detector(DetectorConfig(enabled=True), Path('.'));"
                "d.load();"
                "assert d.detect(np.zeros((8, 8, 3), np.uint8)) == [];"
                "print('degraded ok')",
            ],
            marker="degraded ok",
        ),
        Check(
            name="P 传递滑窗自检",
            env="light",
            args=[
                "-c",
                "from tools.label_frames import check_propagate_window as c;"
                "print('\\n'.join(c()))",
            ],
            marker="整段一条直线",
        ),
        Check(
            name="数据集构建校验（dry-run）",
            env="light",
            args=["-m", "tools.build_dataset", "--dry-run"],
            marker="划分",
        ),
        Check(
            name="config.json 可解析",
            env="light",
            args=[
                "-c",
                "import json,pathlib;"
                "c=json.loads(pathlib.Path('config/config.json')"
                ".read_text(encoding='utf-8'));"
                "print('camera_mode', c['camera_mode'])",
            ],
        ),
    ]


def model_checks() -> list[Check]:
    """
    要 torch + CUDA 的部分。这两项是真断言，不达标就非零退出。
    """

    return [
        Check(
            name="torch / ultralytics 可用",
            env="ai",
            args=[
                "-c",
                "import torch, ultralytics;"
                "print(torch.__version__, torch.cuda.is_available(), "
                "ultralytics.__version__)",
            ],
        ),
        Check(
            name="Yolo 加载 + 一帧检测/跟踪",
            env="ai",
            args=[
                "-c",
                "from pathlib import Path;"
                "import numpy as np;"
                "from app.detector import Detector, DetectorConfig, Track;"
                "d = Detector(DetectorConfig(device='cuda'), Path('.'));"
                "assert d.load();"
                "f = np.zeros((720, 960, 3), np.uint8);"
                "d.detect(f);"
                "assert isinstance(d.track(f), list);"
                "print('inference ok')",
            ],
            marker="inference ok",
        ),
        Check(
            name="ID 持续性断言（pan 合成片段）",
            env="ai",
            args=[
                "-m",
                "tools.check_detector",
                "--track",
                "--motion",
                "pan",
                "--clip-frames",
                "60",
                "--out",
                "runs/selftest",
            ],
            marker="[SELFTEST] ID 位移 全部达标",
        ),
        Check(
            name="预测增益断言（wave 合成片段）",
            env="ai",
            args=[
                "-m",
                "tools.check_detector",
                "--predict",
                "--clip-frames",
                "120",
                "--out",
                "runs/selftest",
            ],
            marker="[SELFTEST] 预测增益 全部达标",
        ),
        Check(
            name="激光安全性质 + 界面文案完整性",
            env="light",
            args=["-m", "tools.check_console"],
            marker="[SELFTEST] 控制台自检 全部达标",
        ),
    ]


def stream_check() -> Check:
    return Check(
        name="手机流实测（需要手机在线）",
        env="ai",
        args=["-m", "tools.measure_stream", "--seconds", "5"],
        marker="FPS",
        skip_if="无法打开网络摄像头",
    )


def run(check: Check, interpreter: Path, verbose: bool) -> tuple[str, float, str]:
    """
    跑一项，返回 (PASS|FAIL|SKIP, 秒, 输出)。
    """

    if not interpreter.exists():
        return "SKIP", 0.0, f"没有 {interpreter.parent.parent.name}"

    started = time.perf_counter()

    try:

        done = subprocess.run(
            [str(interpreter), "-X", "utf8", *check.args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_S,
        )

    except subprocess.TimeoutExpired:

        return "FAIL", time.perf_counter() - started, f"超过 {TIMEOUT_S:.0f}s"

    except OSError as exc:

        return "FAIL", time.perf_counter() - started, str(exc)

    elapsed = time.perf_counter() - started

    output = (done.stdout or "") + (done.stderr or "")

    if done.returncode != 0:

        if check.skip_if and check.skip_if in output:

            return "SKIP", elapsed, check.skip_if

        return "FAIL", elapsed, output

    if check.marker and check.marker not in output:

        return (
            "FAIL",
            elapsed,
            f"退出码 0 但没打出「{check.marker}」—— "
            f"这一项很可能根本没核对任何东西\n{output}",
        )

    if verbose:

        print(output.rstrip())

    return "PASS", elapsed, output


def tail(text: str, lines: int = 8) -> str:
    kept = [line for line in text.rstrip().splitlines() if line.strip()]

    return "\n".join(kept[-lines:])


def main() -> int:
    parser = argparse.ArgumentParser(description="项目自检 runner")

    parser.add_argument(
        "--stream",
        action="store_true",
        help="加跑线上手机流实测（需要手机开着 PhoneCamera）",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每项的原始输出",
    )

    args = parser.parse_args()

    checks = code_checks() + model_checks()

    if args.stream:
        checks.append(stream_check())

    interpreters = {
        "light": venv_python(".venv"),
        "ai": venv_python(".venv-ai"),
    }

    print()
    print("=" * 68)
    print(f"自检 {len(checks)} 项   根目录 {ROOT}")
    print("=" * 68)

    results: list[tuple[str, Check, float, str]] = []

    for check in checks:

        status, elapsed, output = run(check, interpreters[check.env], args.verbose)

        results.append((status, check, elapsed, output))

        print(
            f"  {status:<4}  {check.name:<34}  "
            f"{elapsed:6.1f}s"
            + (f"  ({output.strip().splitlines()[0][:40]})" if status == "SKIP" else "")
        )

        if status == "FAIL":

            for line in tail(output).splitlines():

                print(f"          | {line}")

    failed = [item for item in results if item[0] == "FAIL"]

    skipped = [item for item in results if item[0] == "SKIP"]

    print("-" * 68)
    print(
        f"PASS {len(results) - len(failed) - len(skipped)}   "
        f"FAIL {len(failed)}   SKIP {len(skipped)}"
    )

    if skipped:

        print("  SKIP 明细：" + "；".join(
            f"{check.name} — {tail(output, 1)}"
            for _, check, _, output in skipped
        ))

    print("=" * 68)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
