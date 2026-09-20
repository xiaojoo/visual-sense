"""
V0.5 入口：浏览器控制台。

跑法：
    .venv\\Scripts\\python.exe -m app.server            # 只起界面，不开 AI
    .venv-ai\\Scripts\\python.exe -m app.server         # 带检测

和 V0.1~V0.4 的区别：不再有 OpenCV 窗口。
所有状态、开关、目标列表、激光控制、操作日志都在网页上，
本机浏览器开 http://127.0.0.1:8060/，
局域网里的手机和平板开打印出来的那个 IP。
"""

from __future__ import annotations

import sys

from .hub import CommandQueue, Hub, Shutdown
from .pipeline import Pipeline
from .runtime import load_config, project_root
from .web import WebConfig, create_app, local_addresses, web_config_from


def print_banner(web: WebConfig, config: dict) -> None:
    print()
    print("=" * 60)
    print("  VisualSense V0.6  控制台")
    print("=" * 60)
    print()

    camera = config.get("camera_mode", "usb")

    print(f"  摄像头源   {camera}" + (f"  {config.get('camera_url', '')}" if camera == "network" else ""))
    print(f"  检测模型   {(config.get('ai') or {}).get('model_path', '-')}")
    print()
    print("  本机    " + f"http://127.0.0.1:{web.port}/")

    for ip in [a for a in local_addresses() if a != "127.0.0.1"]:
        print(f"  局域网    http://{ip}:{web.port}/")

    print()
    print("  手机和平板用上面的局域网地址；Ctrl+C 停止。")
    print()


def main() -> int:
    import uvicorn

    # 重定向到文件时 stdout 是块缓冲的，
    # 局域网地址恰恰是最需要立刻看到的一行。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)

    try:
        config = load_config()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    root = project_root()
    web = web_config_from(config)

    hub = Hub()
    commands = CommandQueue()
    shutdown = Shutdown()

    pipeline = Pipeline(config, hub, commands, shutdown, root)
    pipeline.start()

    app = create_app(pipeline, hub, commands, shutdown, web)

    print_banner(web, config)

    try:
        uvicorn.run(app, host=web.host, port=web.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        shutdown.request()
        pipeline.join(timeout=5.0)
        print("\n[INFO] 管线已停止。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
