"""
量真实渲染出来的布局，而不是量 CSS 里写了什么。

为什么要这么麻烦：卡片溢出这种问题，读代码读不出来，
`scrollWidth > clientWidth` 也不算数 —— 只有**真的画出滚动条**
（offsetWidth - clientWidth > 0，滚动条占了宽度）才是他眼睛看见的那件事。

用 CDP 而不是 `chrome --screenshot`：这一页有 MJPEG，响应永远不结束，
Chrome 等不到网络空闲会挂住；`window.load` 同理不触发，所以脚本里
只等 DOMContentLoaded 之后的一段稳定期。

跑法（控制台得先在跑，页面要拉 /api/snapshot）：
    .venv\\Scripts\\python.exe -m tools.check_layout --save before.json
    改样式
    .venv\\Scripts\\python.exe -m tools.check_layout --save after.json --against before.json

退出码：滚动条数不为 0，或者 --against 时几何变了，都算失败。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
URL = "http://127.0.0.1:8060/"
PORT = 9333
EVENTS = 3             # 合成日志条数，--events 可改
TARGETS = 3            # 合成目标条数，--targets 可改
SHOT = ""            # --shot 给了就顺手截图，{w} {h} 会替换成视口尺寸

# 覆盖三种真实场景：贴合视口（含 2560 / 3440 宽屏）、单列（窄）、手机
VIEWPORTS = [
    (3440, 1440),
    (2560, 1080),
    (1920, 1080),
    (1600, 1050),
    (1440, 900),
    (1280, 800),
    (915, 800),      # 他说的那条线：这一档不许变
    (914, 800),      # 小于 915：当前目标要独占一行
    (900, 800),
    (390, 844),
]

JS = """
(async () => {
  // 先等页面真的渲染出卡片。上一版这里写死一个 settle 毫秒数，
  // 有一次页面还没起来就开量，结果"零条滚动条、全部达标"——
  // 量了个空页面却报通过，这是最坏的一种假通过。
  let seen = 0;

  for (let i = 0; i < 60; i++) {
    seen = document.querySelectorAll('.card').length;
    if (seen) break;
    await new Promise(r => setTimeout(r, 200));
  }

  await new Promise(r => setTimeout(r, __SETTLE__));

  // 目标表平时是空的（没相机就没目标），那三列 num 的数据格就永远量不到。
  // 页面自己暴露了 renderTargets，所以喂 N 条合成目标，走的是真渲染路径 ——
  // N 可调是为了同时量到"最少 4 行"和"最多 20 行"两头。
  if (window.renderTargets && !document.querySelector('#target-body tr')) {
    const list = [];

    for (let i = 0; i < __TARGETS__; i++) {
      const person = i % 4 === 3;
      list.push({
        source: person ? "person" : "mosquito",
        label: person ? "person" : "mosquito",
        track_id: 100 + i,
        class: person ? "person" : "insect",
        short_px: 16 + i,
        conf: 0.2 + (i % 7) / 20,
        speed_px_s: i % 5 === 0 ? null : 20 + i,
        targetable: !person,
        locked: i === 1,
        x: (i % 5) / 6, y: (i % 4) / 5, w: 0.04, h: 0.03,
      });
    }

    renderTargets(list);
  }

  // 日志条数也要可调：他截图里那张日志卡是灌满的样子，
  // 只有两三条日志时"两张卡等高"这种问题根本量不出来。
  if (window.renderEvents) {
    const now = Date.now() / 1000;
    const feed = [];

    for (let i = 0; i < __EVENTS__; i++) {
      feed.push({
        text: i % 3 === 0 ? "【摄像头】已连接" : "【摄像头】尝试重新连接…",
        at: now - i * 61, seq: i + 1, count: i % 9 === 0 ? 2 : 1,
      });
    }

    renderEvents(feed);
  }

  const px = v => Math.round(parseFloat(v) * 100) / 100;
  const box = el => {
    const b = el.getBoundingClientRect();
    return { x: px(b.x), y: px(b.y), w: px(b.width), h: px(b.height) };
  };
  // 三张右栏卡片都没有 id，className 首段也一样 —— 只叫 "section.card"
  // 的话比对时后一张会盖掉前一张，看不出到底哪张变了。所以同名按出现顺序编号。
  const seenNames = new Map();
  const name = el => {
    const base = el.id
      ? '#' + el.id
      : el.tagName.toLowerCase() + '.' + String(el.className || '').trim().split(/\\s+/)[0];
    const n = (seenNames.get(base) || 0) + 1;
    seenNames.set(base, n);
    return n === 1 ? base : base + '@' + n;
  };
  const sides = el => {
    const s = getComputedStyle(el);
    return [s.marginTop, s.marginRight, s.marginBottom, s.marginLeft,
            '|', s.paddingTop, s.paddingRight, s.paddingBottom, s.paddingLeft].join(' ');
  };

  const cards = [...document.querySelectorAll('.card')].map(card => {
    const head = card.querySelector(':scope > header');
    const table = card.querySelector(':scope > .table-scroll');
    const body = card.querySelector(':scope > .card-body');

    // 标题钉没钉住，只能真的滚一下内容区再看：把 body 滚到底，
    // 量标题的 y 有没有跟着走。滚不动（没溢出）的卡不测。
    let pin = null;

    if (body && head && body.scrollHeight - body.clientHeight > 4) {
      const before = head.getBoundingClientRect().y;
      body.scrollTop = 9999;
      pin = {
        travel: Math.round(body.scrollTop),
        moved: Math.round((head.getBoundingClientRect().y - before) * 100) / 100,
      };
      body.scrollTop = 0;
    }

    return {
      card: name(card),
      box: box(card),
      pad: sides(card),
      header: head ? { box: box(head), sides: sides(head) } : null,
      table: table ? { box: box(table), sides: sides(table) } : null,
      body: body ? { box: box(body), pin } : null,
      clipX: px(card.scrollWidth - card.clientWidth),
      bleed: [...card.querySelectorAll(':scope > *')]
        .filter(el => el !== head && el !== table && el !== body)
        .map(el => name(el) + ' ml=' + getComputedStyle(el).marginLeft
                          + ' mr=' + getComputedStyle(el).marginRight),
    };
  });

  // 怎么才算"真的画出了一条横向滚动条"：
  //   1) 内容确实比盒子宽（scrollWidth > clientWidth）—— 但 overflow:hidden 也满足，
  //   2) 所以再看盒子高度：滚动条是画在内框底部的，会把 clientHeight 压掉一条的厚度。
  //      用 rect.height - clientHeight - 上下边框 量这个厚度，>=5px 才算滚动条，
  //      1~2px 是边框和四舍五入。门槛不能贴着滚动条宽度写 —— 样式把滚动条改成 8px
  //      之后，门槛 >=8 就会因为半个像素的舍入漏判。
  // offsetWidth-clientWidth 量的是**竖向**滚动条，别拿它当横向的证据。
  const sum = (s, a, b) => (parseFloat(s[a]) || 0) + (parseFloat(s[b]) || 0);

  const scrollbars = el => {
    const s = getComputedStyle(el);
    const rect = el.getBoundingClientRect();

    if (s.display === 'inline' || rect.width < 24 || rect.height < 12) return [0, 0];

    const thick = (spare, over) => (over && spare >= 5 ? Math.round(spare) : 0);

    return [
      thick(rect.height - el.clientHeight - sum(s, 'borderTopWidth', 'borderBottomWidth'),
            s.overflowX !== 'visible' && el.scrollWidth - el.clientWidth > 1),
      thick(rect.width - el.clientWidth - sum(s, 'borderLeftWidth', 'borderRightWidth'),
            s.overflowY !== 'visible' && el.scrollHeight - el.clientHeight > 1),
    ];
  };

  const both = [...document.querySelectorAll('body *')]
    .map(el => [el, ...scrollbars(el)]);

  const hbars = both.filter(([, h]) => h).map(([el, h]) => name(el) + ':' + h);
  const vbars = both.filter(([, , v]) => v).map(([el, , v]) => name(el) + ':' + v);

  // 两张数据表：行不许换行、首末两列要钉住、中间挤不下就横向滚。
  // "钉住"只能真的横滚一下再量格子贴没贴住容器内沿 —— 静态几何看不出来。
  const tables = [...document.querySelectorAll('.table')].map(t => {
    const scroller = t.closest('.table-scroll');
    const s = scroller ? getComputedStyle(scroller) : null;
    const head = t.querySelector('thead tr');
    const first = t.querySelector('thead th:first-child');
    const last = t.querySelector('thead th:last-child');

    const wrap = [...t.querySelectorAll('th, td')]
      .filter(el => getComputedStyle(el).whiteSpace !== 'nowrap').length;

    // 两张表统一居中（他定的）。没写对齐方式时 Chrome 报 start，那算左，也要抓出来。
    const offCenter = [...t.querySelectorAll('th, td')]
      .filter(el => getComputedStyle(el).textAlign !== 'center')
      .map(el => (el.textContent.trim().slice(0, 8) || el.tagName.toLowerCase())
                 + '@' + el.cellIndex + '=' + getComputedStyle(el).textAlign);

    // 操作列里的按钮被列宽裁掉是看不出来的（td 有 ellipsis），单独量：
    // 按钮自己的宽度不能大过格子去掉内衬之后的可用宽。
    const btnClipped = [...t.querySelectorAll('tbody td:last-child button')]
      .map(b => {
        const cs = getComputedStyle(b.parentElement);
        const pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
        const cell = b.parentElement;
        return [b.textContent.trim(), Math.round(b.getBoundingClientRect().width),
                Math.round(cell.clientWidth - pad),
                `格${Math.round(cell.getBoundingClientRect().width)}/衬${pad}`];
      })
      .filter(([, need, room]) => need > room + 1)
      .map(([txt, need, room, why]) => `${txt} ${need}px需/${room}px有 ${why}`);

    // 列宽是 px 写死的，字却是按内容长的：来源列塞的是模型名（mosquito 8 个字母）。
    // 数字列截断是有意的（权重列只给 88px，app.js 把全文塞进了 title），
    // 所以只抓"裁了又没给悬停全文"的格子。
    const cellClipped = [...t.querySelectorAll('tbody td')]
      .filter(el => {
        const txt = el.textContent.trim();
        return /[A-Za-z\u4e00-\u9fff]/.test(txt)
          && !el.querySelector('button')
          && !el.getAttribute('title')
          && el.scrollWidth - el.clientWidth > 1;
      })
      .map(el => head.children[el.cellIndex].textContent.trim() + ':'
                  + el.textContent.trim().slice(0, 12)
                  + ` +${el.scrollWidth - el.clientWidth}px`
                  + ` 格${Math.round(el.getBoundingClientRect().width)}`);

    let pin = null;

    if (scroller && first && last && scroller.scrollWidth - scroller.clientWidth > 4) {
      const r = scroller.getBoundingClientRect();
      // 竖向滚动条会占掉右边一条，容器"看得见"的右沿要往内收这么多
      const vbar = scroller.offsetWidth - scroller.clientWidth
        - (parseFloat(s.borderLeftWidth) || 0) - (parseFloat(s.borderRightWidth) || 0);
      const inLeft = r.left + (parseFloat(s.paddingLeft) || 0);
      const inRight = r.right - (parseFloat(s.paddingRight) || 0) - Math.max(vbar, 0);

      scroller.scrollLeft = 9999;
      pin = {
        travel: Math.round(scroller.scrollLeft),
        gapLeft: Math.round((first.getBoundingClientRect().x - inLeft) * 10) / 10,
        gapRight: Math.round((inRight - last.getBoundingClientRect().right) * 10) / 10,
      };
      scroller.scrollLeft = 0;
    }

    return {
      name: name(t),
      cols: head ? head.children.length : 0,
      headH: head ? Math.round(head.getBoundingClientRect().height * 10) / 10 : 0,
      // 表头不许换行，那列宽不够就会被截断 —— 数值截断是有意的（权重列），
      // 表头截断是设计失误，单独量出来。
      clipped: head ? [...head.children]
        .filter(el => el.scrollWidth - el.clientWidth > 1)
        .map(el => el.textContent.trim() + '+' + (el.scrollWidth - el.clientWidth)) : [],
      wrap,
      offCenter,
      btnClipped,
      cellClipped: [...new Set(cellClipped)],
      pin,
    };
  });

  // 换行之后不许留两种毛病：
  //   holes  —— 同一行卡片宽度合计 + 列间距 应当等于容器宽（差 4px 以内）；
  //   uneven —— 只有一行时（宽屏放得下全部卡）几张卡必须等宽，
  //            这时 auto-fit 会收掉空轨道，任何跨列都会把最后一张撑成两倍宽。
  const lineCheck = (() => {
    const box = document.querySelector(".modules");
    if (!box) return { holes: [], uneven: [] };

    const gap = parseFloat(getComputedStyle(box).columnGap) || 0;
    const W = box.getBoundingClientRect().width;
    const lines = new Map();

    for (const c of box.querySelectorAll(":scope > .card")) {
      const r = c.getBoundingClientRect();
      const key = Math.round(r.y);
      const line = lines.get(key) || [];
      line.push(r.width);
      lines.set(key, line);
    }

    const rows = [...lines.entries()].map(([y, w]) => ({ y, w }));
    const holes = rows
      .filter((r) => W - (r.w.reduce((a, b) => a + b, 0) + (r.w.length - 1) * gap) > 4)
      .map((r) => `y=${r.y} ${r.w.length}张 空`
        + `${Math.round(W - r.w.reduce((a, b) => a + b, 0) - (r.w.length - 1) * gap)}px`);

    const spread = rows.length === 1 && rows[0].w.length > 1
      ? Math.max(...rows[0].w) - Math.min(...rows[0].w)
      : 0;

    return {
      holes,
      uneven: spread > 2 ? [`单行 ${rows[0].w.length} 张宽度差 ${Math.round(spread)}px`] : [],
    };
  })();

  const root = document.documentElement;

  return JSON.stringify({
    viewport: [innerWidth, innerHeight],
    seen: cards.length,
    holes: lineCheck.holes,
    uneven: lineCheck.uneven,
    tables,
    // 模块行一旦换行，代价全在画面区高度上，所以这三块必须一起量
    stage: box(document.querySelector('.stage')),
    video: box(document.querySelector('.video-wrap')),
    modules: box(document.querySelector('.modules')),
    railW: getComputedStyle(root).getPropertyValue('--rail-w').trim(),
    rail: (() => {
      const r = document.querySelector(".rail");
      const first = r && r.querySelector(":scope > .card");
      return r && first
        ? { w: px(r.getBoundingClientRect().width), first: px(first.getBoundingClientRect().width) }
        : null;
    })(),
    rowH: getComputedStyle(root).getPropertyValue('--row-h').trim(),
    cards,
    bars: hbars,
    vbars: vbars,
    page: {
      x: px(root.scrollWidth - root.clientWidth),
      y: px(root.scrollHeight - root.clientHeight),
    },
  });
})()
"""

# 只有这四档走"贴合视口"，页面本身不许滚；窄屏/手机本来就是单列长页，
# 竖向滚动是设计内的，不能当异常。
FIT = {"3440x1440", "2560x1080", "1920x1080", "1600x1050", "1440x900", "1280x800"}

# 390 这一档的列宽裁字是 2026-09-21 他看过数字之后留下的：手机档容器比 #target-table 的
# min-width 398px 窄，Chrome 把写死的列宽按比例放大 398/369≈1.079 倍，格子自己还要吃掉
# 左右各 4px 内衬 —— "已锁定"缺 3px、"mosquito" 缺 2px。要治得 c-op→57、c-src→63，
# 代价是这张表在手机上横滚得更远。只豁免这两条，同视口的别的毛病照样报。
CLIP_OK_ON = {"390x844"}


async def probe(width: int, height: int, settle_ms: int) -> dict:
    async with websockets.connect(WS_URL, max_size=None) as ws:
        mid = 0

        async def call(method: str, **params) -> dict:
            nonlocal mid
            mid += 1
            want = mid
            await ws.send(json.dumps({"id": want, "method": method, "params": params}))

            while True:
                reply = json.loads(await ws.recv())

                if reply.get("id") == want:
                    if "error" in reply:
                        raise RuntimeError(f"{method}: {reply['error']}")
                    return reply.get("result", {})

        await call("Runtime.enable")
        await call("Emulation.setDeviceMetricsOverride",
                   width=width, height=height, deviceScaleFactor=1, mobile=False)
        raw = await call("Runtime.evaluate",
                         expression=(JS.replace("__SETTLE__", str(settle_ms))
                              .replace("__TARGETS__", str(TARGETS))
                              .replace("__EVENTS__", str(EVENTS))),
                         awaitPromise=True, returnByValue=True)

        if SHOT:
            shot = await call("Page.captureScreenshot", format="png")
            Path(SHOT.format(w=width, h=height)).write_bytes(
                base64.b64decode(shot["data"])
            )

        await call("Emulation.clearDeviceMetricsOverride")

    return json.loads(raw["result"]["value"])


def target_url() -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=5) as response:
        pages = [t for t in json.loads(response.read().decode()) if t.get("type") == "page"]

    if not pages:
        raise RuntimeError("Chrome 没给出 page target")

    return pages[0]["webSocketDebuggerUrl"]


def collect(settle_ms: int) -> dict:
    global WS_URL

    profile = Path(tempfile.mkdtemp(prefix="mv-layout-"))
    chrome = subprocess.Popen(
        [CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
         f"--user-data-dir={profile}", "--no-first-run", "--disable-gpu",
         "--force-device-scale-factor=1", URL],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        for _ in range(60):
            time.sleep(0.25)

            try:
                WS_URL = target_url()
                break
            except Exception:
                continue
        else:
            raise RuntimeError("等不到 Chrome 的调试端口")

        return {
            f"{w}x{h}": asyncio.run(probe(w, h, settle_ms))
            for w, h in VIEWPORTS
        }
    finally:
        chrome.terminate()
        time.sleep(0.4)


def flatten(data: dict) -> dict[str, dict]:
    """把 {视口: {...}} 摊成 {"1920x1080 #mod-models.box.w": 值}，好逐项比。"""

    out: dict[str, dict] = {}

    for view, payload in data.items():
        out[f"{view} page.x"] = payload["page"]["x"]

        if view in FIT:
            # 单列那几档页面本来就要滚，长度随日志条数走，比了只会吵
            out[f"{view} page.y"] = payload["page"]["y"]

        out[f"{view} railW"] = payload["railW"]
        out[f"{view} bars"] = payload["bars"]

        for part in ("stage", "video", "modules"):
            if part in payload:
                out[f"{view} {part}"] = payload[part]
        # vbars（竖向滚动条）不参与比对：它是设计内的，
        # 而且条目多少跟着目标数/日志条数走，两次跑本来就不一样。

        for card in payload["cards"]:
            key = f"{view} {card['card']}"
            out[f"{key} box"] = card["box"]
            out[f"{key} clipX"] = card["clipX"]
            out[f"{key} pad"] = card["pad"]
            out[f"{key} bleed"] = card["bleed"]

            for part in ("header", "table"):
                if card[part]:
                    out[f"{key} {part}"] = card[part]

    return out


def report(data: dict) -> list[str]:
    """
    只把"横向"当异常。

    竖向滚动条在 .table-scroll / .log 里是设计内的（模块卡装不下就在卡内滚），
    页面竖向滚动只在贴合视口那几档算事故。
    """

    problems: list[str] = []

    for view, payload in data.items():
        if not payload.get("seen"):
            problems.append(f"{view} 页面上一张卡片都没渲染出来，这一档的测量无效")
            continue

        for card in payload["cards"]:
            body = card.get("body") or {}
            pin = body.get("pin")

            if card["card"].startswith("#mod") and not body:
                problems.append(
                    f"{view} {card['card']} 没有 .card-body —— 内容会连着标题一起滚"
                )

            if pin and pin["moved"]:
                problems.append(
                    f"{view} {card['card']} 内容区滚到底 {pin['travel']}px，"
                    f"标题栏跟着移了 {pin['moved']}px —— 标题没钉住"
                )

        for table in payload.get("tables", []):
            if table.get("clipped"):
                problems.append(
                    f"{view} {table['name']} 表头被截断：{table['clipped']} —— "
                    "行不许换行之后，列宽不够首先牺牲的就是表头"
                )

            if table.get("btnClipped") and view not in CLIP_OK_ON:
                problems.append(
                    f"{view} {table['name']} 操作列的按钮被列宽裁掉：{table['btnClipped']} —— "
                    "td 有 ellipsis，这种情况肉眼只会看到按钮缺半截"
                )

            if table.get("cellClipped") and view not in CLIP_OK_ON:
                problems.append(
                    f"{view} {table['name']} 正文格子被列宽裁掉：{table['cellClipped']} —— "
                    "列宽是写死的 px，模型名比列长就少几个字母"
                )

            if table.get("offCenter"):
                problems.append(
                    f"{view} {table['name']} 有没居中的格子：{table['offCenter']} —— "
                    "两张表的表头和数值统一居中"
                )

            if table["wrap"]:
                problems.append(
                    f"{view} {table['name']} 有 {table['wrap']} 个格子允许换行 —— "
                    "行信息必须不换行（挤不下就截断 + 横向滚）"
                )

            roll = table.get("pin")

            if roll and (abs(roll["gapLeft"]) > 1 or abs(roll["gapRight"]) > 1):
                problems.append(
                    f"{view} {table['name']} 横滚 {roll['travel']}px 后，"
                    f"首列离容器左沿 {roll['gapLeft']}px、末列离右沿 {roll['gapRight']}px —— 两头没钉住"
                )

        if payload.get("holes"):
            problems.append(
                f"{view} 模块行有没摊满的一行：{payload['holes']} —— "
                "grid 不会自己摊最后一行，靠 syncLayout() 给最后一张卡补跨列"
            )

        # 914 及以下：右栏第一张卡（当前目标）必须独占一行，另外两张并排
        if payload.get("rail") and int(view.split("x")[0]) <= 914                 and payload["rail"]["first"] < payload["rail"]["w"] - 4:
            problems.append(
                f"{view} 当前目标只占了 {payload['rail']['first']}px，"
                f"右栏宽 {payload['rail']['w']}px —— 这一档它要独占一行"
            )

        if payload.get("uneven"):
            problems.append(
                f"{view} 模块卡只有一行却不等宽：{payload['uneven']} —— "
                "放得下全部卡时不该跨列，auto-fit 自己会收掉空轨道"
            )

        # 表格容器横滚是设计内的（首末列钉住、中间滚），别当事故报
        bars = [b for b in payload["bars"] if "table-scroll" not in b]
        rolls = [b for b in payload["bars"] if "table-scroll" in b]

        if bars:
            problems.append(f"{view} 画出**横向**滚动条的容器：{bars}")

        if rolls:
            print(f"             （设计内）横向滚动的表格容器：{rolls}")

        if payload["page"]["x"]:
            problems.append(f"{view} 页面横向溢出 {payload['page']['x']}px")

        if view in FIT and payload["page"]["y"]:
            problems.append(f"{view} 页面竖向还能滚 {payload['page']['y']}px，没贴合住")

    return problems


def diff(old: dict, new: dict) -> list[str]:
    changed: list[str] = []

    for key in sorted(set(old) | set(new)):
        before, after = old.get(key), new.get(key)

        if before == after:
            continue

        if isinstance(before, dict) and isinstance(after, dict):
            moved = {k: (before.get(k), after.get(k))
                     for k in set(before) | set(after) if before.get(k) != after.get(k)}
            changed.append(f"{key}: {moved}")
        else:
            changed.append(f"{key}: {before!r} -> {after!r}")

    return changed


def serve(port: int) -> tuple:
    """
    自己起一份控制台，量完就关。

    为什么不直接量他开着的那一份：
    1. 相机被抢走一次，他页面上的画面就断一次；
    2. 自带一份才能把画面源掐了（camera_mode=usb 且没有设备 → 恒定"无信号"），
       两次跑的几何才是可比的；
    3. 没有 torch 的 .venv 也能起，AI 关掉不影响布局。
    """

    import threading

    import uvicorn

    from app.hub import CommandQueue, Hub, Shutdown
    from app.pipeline import Pipeline
    from app.runtime import load_config, project_root
    from app.web import create_app, web_config_from

    config = load_config()
    config["camera_mode"] = "usb"
    config["camera_url"] = ""
    config["web"] = {**(config.get("web") or {}), "host": "127.0.0.1", "port": port}

    hub, commands, shutdown = Hub(), CommandQueue(), Shutdown()
    pipeline = Pipeline(config, hub, commands, shutdown, project_root())
    pipeline.start()

    server = uvicorn.Server(uvicorn.Config(
        create_app(pipeline, hub, commands, shutdown, web_config_from(config)),
        host="127.0.0.1", port=port, log_level="critical",
    ))

    threading.Thread(target=server.run, daemon=True).start()

    import urllib.error

    for _ in range(80):
        time.sleep(0.25)

        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1.0):
                break
        except urllib.error.HTTPError:
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"自带的控制台实例 {port} 端口起不来")

    return server, pipeline, shutdown


def main() -> int:
    parser = argparse.ArgumentParser(description="量控制台真实渲染出来的几何与滚动条")
    parser.add_argument("--port", type=int, default=8061, help="自带实例的端口")
    parser.add_argument("--url", help="改成量已经开着的那一份（会抢它的相机）")
    parser.add_argument("--save", type=Path, help="把这次的测量结果写成 json")
    parser.add_argument("--against", type=Path, help="和这份基线逐项比，几何变了就报")
    parser.add_argument("--events", type=int, default=3, help="喂几条合成日志（量日志卡撑高之后的行为）")
    parser.add_argument("--targets", type=int, default=3, help="喂几条合成目标（量 4 行下限和 20 行上限）")
    parser.add_argument("--settle", type=int, default=1400, help="每档视口等多少毫秒再量")
    parser.add_argument("--shot", help="顺手截图，路径里用 {w} {h} 区分视口")
    args = parser.parse_args()

    global URL, SHOT, TARGETS, EVENTS

    SHOT = args.shot or ""
    TARGETS = args.targets
    EVENTS = args.events

    if args.url:
        URL = args.url
        data = collect(args.settle)
    else:
        URL = f"http://127.0.0.1:{args.port}/"
        server, pipeline, shutdown = serve(args.port)

        try:
            data = collect(args.settle)
        finally:
            server.should_exit = True
            time.sleep(0.4)
            shutdown.request()
            pipeline.join(timeout=3.0)

    if args.save:
        args.save.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已写入 {args.save}")

    problems = report(data)

    for view, payload in data.items():
        pins = [c["body"]["pin"] for c in payload["cards"] if (c.get("body") or {}).get("pin")]

        print(f"  {view:<10} 卡片={payload.get('seen', 0)}"
              f"  画面区 h={round(payload['stage']['h']):>4}"
              f" 视频 h={round(payload['video']['h']):>4}"
              f" 模块行 h={round(payload['modules']['h']):>4}"
              f"  页面溢出 x={payload['page']['x']:>4} y={payload['page']['y']:>4}"
              f"  --rail-w={payload['railW'] or '-':<7}"
              f" 横向滚动条={len(payload['bars'])} 竖向={len(payload['vbars'])}"
              f" 没摊满的行={len(payload.get('holes') or [])}"
              f" 不等宽={len(payload.get('uneven') or [])}"
              f" 内容区可滚={len(pins)} 张，标题位移={sorted({p['moved'] for p in pins})}")

        for table in payload.get("tables", []):
            roll = table.get("pin")
            print(f"             {table['name']:<16} {table['cols']} 列 表头高 {table['headH']}"
                  f" 可换行格子={table['wrap']} 没居中格子={len(table.get('offCenter') or [])}"
                  f" 按钮被裁={table.get('btnClipped') or '无'}"
                  f" 表头截断={table.get('clipped') or '无'}"
                  f" 正文截断={table.get('cellClipped') or '无'}"
                  + (f" 横滚 {roll['travel']}px 后首列偏移 {roll['gapLeft']} 末列偏移 {roll['gapRight']}"
                     if roll else " 不用横滚"))

        if payload["bars"]:
            print(f"             -> {payload['bars']}")

    if args.against:
        changes = diff(flatten(json.loads(args.against.read_text(encoding="utf-8"))),
                       flatten(data))

        if changes:
            print(f"\n与 {args.against.name} 相比几何有 {len(changes)} 处变化：")

            for line in changes[:40]:
                print("   ", line)

            if len(changes) > 40:
                print(f"    …另有 {len(changes) - 40} 处")

        else:
            print("\n与基线逐项一致：渲染出来的几何没有任何一处变化。")

        problems += [f"几何变化 {len(changes)} 处"] if changes else []

    print()

    if problems:
        for line in problems:
            print("  !", line)

        print("\n[SELFTEST] 布局测量 有异常")
        return 1

    print("[SELFTEST] 布局测量 全部达标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
