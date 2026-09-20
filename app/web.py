"""
V0.5 Web 服务层。

FastAPI + uvicorn，只装在 .venv 里，不进 .venv-ai：
这一层不 import torch，装了 torch 的机器才需要 .venv-ai，
两者分开之后，没有 GPU 的笔记本也能把界面跑起来看布局和交互。

线程模型：
uvicorn 跑在主线程的事件循环里，管线在后台线程。
两边只通过 Hub（读）和 CommandQueue（写）打交道，
命令一律排队到管线的帧间执行，见 app/hub.py。
"""

from __future__ import annotations

import io
import socket
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .hub import CommandQueue, Hub, Shutdown
from .pipeline import Pipeline

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

BOUNDARY = "mosquitovision"


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8060
    title: str = "VisualSense"


def web_config_from(config: dict) -> WebConfig:
    web = config.get("web") or {}

    return WebConfig(
        host=str(web.get("host", "0.0.0.0")),
        port=int(web.get("port", 8060)),
        title=str(web.get("title", "VisualSense")),
    )


def local_addresses() -> list[str]:
    """
    局域网可达地址。

    用 connect(UDP) 而不是枚举网卡：
    前者让操作系统自己决定"哪块网卡能出网"，
    多网卡 / 虚拟网卡 / VPN 同时开着的时候不会列出一堆没用的 IP。
    """

    addresses: list[str] = []

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        probe.connect(("8.8.8.8", 80))
        addresses.append(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in addresses and not ip.startswith("127."):
                addresses.append(ip)
    except OSError:
        pass

    return addresses or ["127.0.0.1"]


def create_app(
    pipeline: Pipeline,
    hub: Hub,
    commands: CommandQueue,
    shutdown: Shutdown,
    web: WebConfig,
) -> FastAPI:
    app = FastAPI(title=web.title, docs_url=None, redoc_url=None)

    NO_STORE = {"Cache-Control": "no-store"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = WEB_DIR / "index.html"

        if not page.exists():
            return HTMLResponse("<h1>web/index.html 缺失</h1>", status_code=500)

        return HTMLResponse(page.read_text(encoding="utf-8"), headers=NO_STORE)

    @app.get("/app.js")
    def script() -> StreamingResponse:
        return StreamingResponse(
            io.BytesIO((WEB_DIR / "app.js").read_bytes()),
            media_type="text/javascript; charset=utf-8",
            headers=NO_STORE,
        )

    @app.get("/app.css")
    def style() -> StreamingResponse:
        # 不调样式的时候最恨的就是刷新了没变化，所以显式禁缓存。
        # 这是局域网里的调试台，不是公网站点，缓存省不了什么。
        return StreamingResponse(
            io.BytesIO((WEB_DIR / "app.css").read_bytes()),
            media_type="text/css; charset=utf-8",
            headers=NO_STORE,
        )

    @app.get("/favicon.ico")
    def favicon() -> Response:
        """一个内联小圆点。不注册这个，浏览器每次开页都刷一条 404。"""

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
            '<circle cx="16" cy="16" r="13" fill="#0b0f14" stroke="#4c8dff" stroke-width="3"/>'
            '<circle cx="16" cy="16" r="5" fill="#4c8dff"/></svg>'
        )

        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/api/snapshot")
    def snapshot() -> JSONResponse:
        state = hub.snapshot()

        return JSONResponse(
            {
                "server_time": time.time(),
                "events": hub.events(),
                "history": hub.history()[-120:],
                "running": not shutdown.requested,
                **{
                    k: v
                    for k, v in state.items()
                    if k in ("uptime_s", "frame", "note", "camera", "metrics", "ai", "record", "laser", "targets", "stream")
                },
            }
        )

    @app.post("/api/command")
    async def command(request: Request) -> JSONResponse:
        """
        所有控制入口。

        故意不做成 /api/arm、/api/fire 一堆端点：
        命令是排队到管线线程执行的，一个入口 + 一个名字，
        加新功能时不用两头改路由表。
        """

        try:
            payload = await request.json()
        except ValueError:
            payload = {}

        name = str(payload.get("name", "")).strip()

        if not name:
            return JSONResponse({"ok": False, "message": "缺少 name"}, status_code=400)

        args = payload.get("args") or {}

        if not isinstance(args, dict):
            args = {}

        result = commands.submit(name, **args)

        return JSONResponse(
            {
                **result,
                "laser": hub.snapshot().get("laser", {}),
                "server_time": time.time(),
            }
        )

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "frame_age_s": round(hub.frame_age_s, 2),
            "addresses": [f"http://{ip}:{web.port}/" for ip in local_addresses()],
        }

    @app.get("/stream.mjpg")
    def stream() -> StreamingResponse:
        """
        MJPEG 而不是 WebSocket：
        浏览器 <img> 直接能显示，手机 Safari 也认，
        省掉一层前端解码和重绘。

        旧帧直接跳过（hub.wait_frame），
        否则客户端慢一点就会开始囤积历史帧，看起来像画面卡死。
        """

        def parts():
            seq = 0

            while not shutdown.requested:
                jpeg, seq = hub.wait_frame(seq, timeout=2.0)

                if not jpeg:
                    continue

                yield (
                    f"--{BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode() + jpeg + b"\r\n"

        return StreamingResponse(
            parts(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-store", "Connection": "close"},
        )

    return app
