> 这份是工程记录：逐版本的验收数据、量法、踩过的坑和各项测量的原始数字。
> 只想跑起来和看功能，请读 [../README.md](../README.md)。

# VisualSense

小飞虫实时检测控制台：手机摄像头当网络相机 → 本机 GPU 跑 YOLO → 浏览器里看画面、
目标、指标、采集和激光模块。局域网内的手机和平板都能打开，界面中文优先、可切英文。

**现在能跑的**：取流、检测、跟踪（ByteTrack 稳定 ID）、匀速预测、按帧采集 + YOLO 预标、
多路模型并发、浏览器控制台。
**还没有**：真虫数据集上的可用精度（当前模型对侧视姿态的虫子全漏）、激光硬件驱动、
像素→云台角度标定。

```
copy config\config.example.json config\config.json    # 填你的相机地址
run-ai.bat                                            # 带检测；run.bat 是不要 torch 的基线
                                                      # 然后浏览器开 http://127.0.0.1:8060/
```

License: CC0 1.0 Universal — 见 `LICENSE`。
在法律允许的范围内，作者放弃对本作品的全部版权及相关权利：复制、修改、分发、商用都不必打招呼。
数据和模型权重不进版本库（`.gitignore`）。

Current version: V0.6

---

## Version History

V0.1

Camera pipeline verified:

Redmi K40 → IP Webcam → Wi-Fi → Windows PC → OpenCV → real-time video

V0.2

Detection pipeline added:

OpenCV → YOLO → Detections → Bounding Box → HUD

V0.3

Tracking added:

Detections → ByteTrack → stable per-target ID → trajectory drawn on screen

V0.4

Motion added:

Trajectory → least-squares fit → smoothed position + velocity → constant-velocity
prediction, plus an offline back-test that compares the prediction against a
"assume it did not move" baseline

V0.4-B

Capture added: record frames (the **采集** chip; it was `R` in the desktop window), YOLO pre-labels, and a per-frame
`annotations.jsonl` carrying frame index, wall-clock timestamp, track id and
confidence — so motion analysis can be redone from disk without re-running
detection.

V0.5

Console added: the OpenCV window is replaced by a browser dashboard served on
the LAN (`run-ai.bat` → `http://127.0.0.1:8060/`). Live MJPEG, target list,
performance metrics, capture control, an activity log, and a laser repel/kill
module with a driver seam. Chinese first with an in-app EN toggle. Phone and
tablet layouts included.

V0.6

Multi-model added: one frame can run several detectors at once, each with its own
tracker and predictor, merged into one target list tagged with its source.
Class allow-listing makes off-the-shelf COCO weights usable without retraining.
Only models explicitly marked targetable can drive the laser — the default is off.

---

## V0.5 Console

```
run-ai.bat                                   # 带检测，浏览器开 http://127.0.0.1:8060/
run.bat                                      # 同一套界面，但没有 torch，检测会报"未安装"
.venv-ai\Scripts\python.exe -u -m app.server # 手动版
```

线程模型只有一条规矩：**数据单向流**。
管线在后台线程，uvicorn 在主线程，两边只通过 `app/hub.py` 见面 ——
帧和快照往下走（Hub），命令往上走（CommandQueue），
命令一律排到管线的帧间执行，所以 detector / recorder / laser 始终只被一个线程碰，
不需要给它们加锁。

| 端点 | 干什么 |
| --- | --- |
| `GET /` | 单页界面（`web/index.html`，无框架、无 CDN、无构建步骤） |
| `GET /stream.mjpg` | MJPEG 推流。旧帧直接跳过，客户端慢也不会囤历史帧 |
| `GET /api/snapshot` | 全量状态：摄像头 / 指标 / 目标 / 采集 / 激光 + 增量日志 |
| `POST /api/command` | 所有控制入口，`{name, args}`。加新功能不用改路由表 |
| `GET /api/health` | 存活 + 局域网地址列表 |

推流参数在 `config.json` 的 `web.stream`：`max_width` 960、`jpeg_quality` 72。
桌面窗口时代不存在这个问题（帧就在本机内存里），换成局域网里的手机之后，
每一帧都要编码 + 走 WiFi，所以必须能单独限宽降质量。

**大屏布局：≥1181 px 宽且 ≥700 px 高时整页不滚。**
画面区不再是固定 4:3，改成吃掉"视口减去模块行"的剩余高度，
指标块从 2 列 3 行变 3 列 2 行，空目标表塌下去不给它装位子。

这张表是 `python -m tools.check_layout` 量出来的（自带一份控制台实例，
掐掉相机源，所以两次跑可比；判据是**真的画出滚动条**，不是 `scrollWidth` 那种猜测）：

| 视口 | 模块列数 | 单卡宽 | 画面区高 | 视频高 | 横向滚动条 |
| --- | --- | --- | --- | --- | --- |
| 3440×1440 | 一行 5 张 | 667 px（等宽） | 934 px | 820 px | 无 |
| 2560×1080 | 一行 5 张 | 491 px（等宽） | 574 px | 460 px | 无 |
| 1920×1080 | 4 + 1 | 458 px（落单那张 1880 px） | 534 px | 420 px | 无 |
| 1600×1050 | 3 + 2 | 509 px（第二行 509 + 1035） | 516 px | 402 px | 无 |
| 1440×900 | 3 + 2 | 456 px（第二行 456 + 928） | 426 px | 312 px | 无 |
| 1280×800 | 2 + 2 + 1 | 612 px（落单那张 1240 px） | 366 px | 205 px | 无（卡内竖滚） |
| 390×844（手机） | 单列长页 | 351 px | — | — | **1 条**：顶栏 chips |

模块卡的最小宽度是 `--card-min: 410px`（单列那档 292px），`auto-fit` 放不下就换行 ——
卡数变了不用再改列数。代价说清楚：**1280 那档会折成 3 行**，
每行只剩约 96px，四张卡各自内部滚，画面被压到 205px 高；
1440 以上看不出问题，反而因为卡片变宽干掉了原来目标表和使用说明卡上的两条横向滚动条。
再往下要救 1280，就在 `@media (max-width: 1400px)` 里把 `--card-min` 降回 320px。

换行后落单的那张卡**跨掉整行剩下的轨道**，不留空白：`syncLayout()` 用
"容器宽 ÷ `--card-min`"算出列数（**不能去量 `gridTemplateColumns`** ——
跨列本身会逼 grid 多开轨道，量回来的列数再算出更大的跨列，正反馈把 1440 撑成
4 条轨道、横向溢出 48px，踩过），写进 `--last-span`，CSS 只有一条
`.modules .card:last-child { grid-column: span var(--last-span, 1) }`。
跨列只在**真的换行且最后一行没坐满**时才给：`span = (列数 ≥ 卡数 或 整除) ? 1 : 1 + 列数 - 余数`。
宽屏放得下全部卡时 auto-fit 会把空轨道收掉，五张卡本来就等宽，
早先那版公式 `列数 - (卡数-1) % 列数` 在 3440（7 列）下算出跨 3，
把使用说明撑成别人的两倍宽、右栏跟着塌到 411px 还多出两条横向滚动条。
`check_layout` 有两条断言对着这两件事：`没摊满的行`（同行卡宽合计 + 列间距必须等于容器宽，
容差 4px）和 `不等宽`（只有一行时五张卡宽度差必须 ≤ 2px）。
反证做过：换回旧公式，3440 那档立刻报 `单行 5 张宽度差 …px`。
试过整段换成 flex 让最后一行自己撑开 —— 宽度是满了，
但 flex 的行高按内容算，40vh 那个上限压不住它，实测 1920 下最后一张卡有 243px
被 `.layout` 的 `overflow: hidden` 裁掉、1280 下裁掉 592px，所以留在 grid。

右栏宽度跟着模块卡走（`syncLayout()` 量真实卡片宽度写回 `--rail-w`），但**封顶 34% 视口宽**：不然 1280 那档卡片 612px 宽，右栏跟着变 612，画面直接挤没。
这个上限只在"左画面 + 右栏"两列的贴合布局里生效，单列模式不写假数字。

900 高以下右栏那条滚动条去不掉：三张卡的最小内容高度约 400 px，
而那一行只剩 300 px —— 是装不下，不是没做。

**视口小于 915px 时右栏改成"1 + 2"**：`当前目标` 独占一整行，`性能指标` 和 `操作日志` 并排。
量到的宽度：914 档右栏 866px → 目标卡 866、另两张各 425；900 档 852 → 852 / 418 + 418。
915 档保持原样（三张都 426，两列），390 手机档也保持原样（三张竖排各 358）——
新规则写在 `@media (max-width: 760px)` 之前，靠后面的单列规则覆盖掉。
`check_layout` 里对着这条的断言是"914 及以下，右栏第一张卡的宽度必须等于右栏宽度"，
基线跑过一次是红的（914 那档报 425 vs 866），改完才转绿。

**操作日志卡的高度：试过"下限=指标卡高度"，又回退了。**
V1（现在这版）= `min-height: auto`，日志吃掉右栏剩下的全部高度，目标表一长它就塌。
14 条日志实测日志卡：2560→156px、1920→116、1600→98、1440/1280→67（指标卡恒为 178.7px），
右栏只有 1440/1280 两档竖滚。
V2 = `min-height: var(--metrics-h)`（`syncLayout()` 量指标卡真实渲染高度写进来）：
五档日志卡全部齐平成 179px，但日志塌不下去之后右栏就装不下了 ——
1920 那档内容 187+179+179+24 = 569px vs 右栏只有 510px，
**3 条 / 8 条 / 14 条日志三种情况下五档全部竖滚**。
他看完两版数字选了 V1：日志高度自适应。V2 的代码和断言已经删干净
（`--metrics-h`、`logFloor` 全仓 0 处命中），别再叠第三版。
顺带记一句：他截图那一屏（≤914 的 1+2 布局）两张卡本来就等高 —— 339 / 339，
是 grid 同一行 `align-items: stretch` 给的，量了 761 / 860 / 914 三档都一样。

**两张表（当前目标 / 模型组）的排布规则**：**表头和数值一律居中**（数字列右对齐过，表头左、
数值右，在 44~48px 的窄列里看着像两排没对齐的字），行内不许换行，
列宽写死 px 加 `min-width`，**首列（来源／名称）和末列（操作）钉在两边**，
中间那几列挤不下就横向滚 —— 滚动条只出现在表格容器里，这是设计内的，`check_layout` 不报它。
**表格容器的左右内边距是 0**，留白落在首末格自己的 `padding` 上：容器一旦留内边距，
钉住列和卡片边缘之间就剩一条缝，横向滚的时候内容会从缝里露出来。
操作列的按钮（启用／停用、锁定／已锁定）形状和底色与未激活态一致，只用描边和文字色区分状态
（原来是白底反色，块头太大）。
`border-collapse` 下 sticky 格子的 `border` 不跟着格子走，所以分界线用 `inset box-shadow` 画。

钉住这件事同样是真滚一遍量的（把 `.table-scroll` 的 `scrollLeft` 推到底，再量首末格离容器内沿多远）。
目标表平时是空的（没相机就没目标），所以探针会调页面自己的 `renderTargets()` 喂三条合成目标，
走真渲染路径，那三列数字和那颗按钮才量得到：

| 视口 | 右栏／模块卡宽 | 目标表横滚 | 模型表横滚 | 钉住后偏移 |
| --- | --- | --- | --- | --- |
| 1920×1080 | 458 / 458 | 不用滚 | 36 px | 0, 0 |
| 1600×1050 | 509 / 509 | 不用滚 | 不用滚 | — |
| 1440×900 | 456 / 456 | 不用滚 | 46 px | 0, 0 |
| 1280×800 | 435 / 612 | 6 px | 不用滚 | 0, 0 |
| 900×800 | 单列 | 不用滚 | 76 px | 0, 0 |
| 390×844 | 351 / 351 | 66 px | 136 px | 0, 0 |

12 个「视口 × 表」组合里：可换行格子 0、**没居中的格子 0**、表头截断 0、按钮被裁 0。
后两条是新加的断言 —— 行不许换行之后，列宽不够首先牺牲的就是表头，
而 `td` 带 ellipsis 会把"按钮缺半截"伪装成正常截断。
两处都是这么抓出来的：「可驱动激光」56px → 64px（权重列让 8px），操作列 72px → 84px
（"不可驱动"要 60px）。反证：把操作列改回 66px，断言立刻报 `按钮被裁=['不可驱动']`。
**操作列后来收到 70px**：把"不可驱动"这个状态从按钮改成灰色短横 `—` + 悬停说明
（`app.js` 里 `!target.targetable` 那一支），最宽的按钮就只剩"已锁定"49px，
加格子左边 4px 和末格的 `--card-pad` 16px → 69，取 70。
目标表 `min-width` 随之 422 → 400，并把「类别」列改成 `width: auto`
（`table-layout: fixed` 下富余空间按比例摊给每一列，写死 px 的列会被一起放大；
留一列 auto 吃掉富余，其余列才是真固定的）。
另外修了探针一处算法：横向钉住的那条断言原来拿容器右沿比，
没扣掉竖向滚动条占掉的 8px，390 那档误报"末列离右沿 8px"。

**表格首末格的内缩去掉了，列宽跟着重新量一轮（2026-09-21 定）。**
`td/th:first-child { padding-left: --card-pad }` 和末格那条对称规则删掉 —— 表格文字自己贴卡片边，
比标题文字靠左一个 16px。`tools/check_console.py` 里"表格首格／表格末格 = --card-pad"两条期望
和随之失效的 `TABLE_EDGE` 一起删，只留"表格外层 = 0"（钉住列和卡片边缘之间不许有缝），自检重新全绿。
列宽 50/45 在窄档是真裁字的：`c-op 45px` 装不下 49px 的"已锁定"，`c-src 50px` 装不下"mosquito"，
1280 差 3px、915 差 4px、390 差 7px；宽屏不犯，因为容器比 `min-width: 398px` 宽，
富余按比例摊给了每一列（这回没留 auto 列，`c-cls` 又写回 76px 了）。
改成 **57 / 50** 之后 10 档视口 9 档干净。390 剩 2~3px：容器 350 < 398，
Chrome 把列放大 398/369 ≈ 1.079 倍（实测操作格 54、来源格 61），格子还要吃掉左右各 4px 内衬
→ 按钮只剩 46px 可用。要治得 57/63，代价是手机上横滚更远，他选接受，
`check_layout.py` 里用 `CLIP_OK_ON = {"390x844"}` 只豁免这两条断言，同视口别的毛病照报。
断言现在带"格/衬"两个数，报出来的就是上面这套算式，不是猜的。

**滚动条统一改成 8px、轨道透明、无箭头**（`::-webkit-scrollbar` 那一套）。
模块卡的内容包在 `.card-body` 里，**卡片不滚、只有内容区滚**，标题栏钉住不动 ——
以前整张卡 `overflow-y: auto`，滚一下把"使用说明"标题也推走了。
这条不是靠眼睛认的：`tools/check_layout.py` 会把每张能滚的 `.card-body` 真的滚到底，
再量标题的 y 有没有跟着动（实测 1920/1600 有 3 张可滚、1440/1280 有 5 张，位移全是 0px）。

有个坑要记着：Chrome 121+ 也认标准属性 `scrollbar-width` / `scrollbar-color`，
但只要它们不是 `auto`，浏览器就切到标准渲染，**整套 `::-webkit-scrollbar` 伪元素失效**
—— 实测同时写 `scrollbar-width: thin` 时滚动条还是 15px。所以标准属性包在
`@supports not selector(::-webkit-scrollbar)` 里只给 Firefox 用。
`tools/check_layout.py` 量的就是这个厚度（`rect.height - clientHeight - 上下边框`），
改前改后分别是 15 和 8，判据门槛留在 5px，不贴着滚动条宽度写。

卡片贴边这件事有一条断言守着：`.card` 不留左右内边距，标题栏和表格自己用
`--card-pad` 留白，其余子元素用 `margin-inline` 缩进，**横向不许出现负外边距**。
以前是"卡片留 16、子元素 -16"，两处数字得手动对齐，我因此踩了两次横向滚动条
（右栏一次、模块行一次），肉眼还看不出来。`css_pairing()` 把样式表摊平后逐条查这三件事，
并且自带反证（把标题栏改回负边距写法必须报错）—— 上一版这条断言是假的，
它按整个 `@media` 块做集合相减，真滚动条还在屏幕上的时候它报了 OK。

**激光（`app/laser.py`）默认是关的，且没有驱动。** `laser.backend` 可选
`none`（默认）/ `sim`（模拟，用来验收交互）。两条写进代码的规矩：

1. 界面绝不假装硬件在线 —— 没驱动时状态灯是灰的，所有命令返回拒绝。
2. 出光要同时满足 驱动就绪 + 已解锁 + 有锁定目标，且有最长停留和心跳失联自动收光。

像素 → 云台角度的标定**还没有做**，`fire` 只把归一化坐标交给驱动，
页面上会一直挂着这条提示。

---

## V0.6 多路模型

同一个画面可以同时跑几路检测器。**每路一套 Detector + Tracks + MotionPredictor**，
对外合并成一张目标表，每条带 `source`。

为什么不共用一套 Tracks：ByteTrack 的 `track_id` 只在单个跟踪器内唯一，
两路都从 1 开始编号，合并就会把"蚊子#1"和"人#1"当成同一个目标 ——
轨迹串味、预测拿错历史、激光锁错东西。这条由 `tools/check_console.py` 断言钉着。

### 配置

`ai.models` 缺失时退化成单路，**旧的扁平 `ai` 段一个字不用改**，
并且那一路自动继承 `targetable = true`（激光本来就是围着它做的）。

```json
"ai": {
  "models": [
    { "key": "fly", "label": "飞虫",
      "model_path": "weights/mosquito_synth1280.pt", "targetable": true },

    { "key": "person", "label": "人",
      "model_path": "weights/yolo11n.pt",
      "classes": [0], "imgsz": 640, "conf": 0.45,
      "detect_every": 5, "track_enabled": true, "predict_enabled": false }
  ]
}
```

每路可覆盖 `model_path / device / imgsz / conf / iou / quantize / detect_every /
max_det / track_enabled / tracker / trail_frames / classes`；没写的沿用 `ai` 段的值。
`key` 必须唯一，重复会在启动时报错。

`classes` 是类别白名单 —— 这是"拿现成 COCO 权重只认人"的关键，
不传就等于把 80 个类全开出来。

### 实测成本（RTX 4080 SUPER，同一帧 960×720，30 次取均）

| 配置 | 帧均推理 | 上限 |
|---|---|---|
| 1 路 飞虫@1280 | 9.1 ms | 109 fps |
| 2 路 + 人@640 每 5 帧 | 12.5 ms | 80 fps |
| 3 路 全部每帧 | 24.0 ms | 42 fps |

显存：3 路合计 **已用 0.07 GB / 占用 0.23 GB**。

**结论和直觉相反：多路模型不是瓶颈，摄像头才是。** 手机流本身只有 20–25 fps，
三路全开还剩一倍余量。真正决定负担的是每路的 `detect_every` ——
人不需 25 fps，蚊子需要，所以这个参数按路给。

### `targetable`：默认不可见

**只有显式标了 `targetable: true` 的那几路，框才会进激光的候选。**
默认 False。通用框架里"任何模型都能触发发射"是不可接受的默认值 ——
接上"人"这个模型的那天，就是激光可能对着人的那天。

页面上对应两件事：不可驱动那几行的"锁定"按钮是灰的并写着「不可驱动」；
`_lock_target` 在后端也会拒绝。两条都有断言。

### 采集会按路分目录

`auto_labels` 打开时写 `labels_auto/<key>/<frame>.txt`，
`annotations.jsonl` 每条检测带 `source`。

原因：A 路的 class 0 是蚊子、B 路的 class 0 是人，
混进同一个 txt 就成了一份谁都解释不了的标签。
真要合并成一份训练集，必须是一次显式决定。

**已知局限**：会话的 `metadata.json` 里记录的权重名和类别表只取第一路。
多路时这份元数据不足以还原全部来源。

---

## V0.4-A Goal

Answer one question with a number, not a feeling:

> How well can the mosquito's position 50 / 100 / 150 ms from now be predicted?

Pipeline:

```
ByteTrack → Track ID → Trajectory Buffer → least-squares fit → velocity
                                          → constant-velocity extrapolation
                                          → predicted position
```

Deliberately not used yet: Kalman, LSTM, optical flow, physics. Those all need
numbers this project does not have yet (process noise, training data).

### Measured

Two synthetic clips generated from the sample image (the model
only produces real detections when the input is a real photo),
150 frames at 30 fps. Both motion laws are defined **in
seconds**, so `--clip-frames` changes only the duration — an
earlier version tied the wave period to the clip length, which
meant changing the sample length silently changed the motion
speed and made two runs incomparable.

Control — constant 120 px/s pan:

| horizon | CV error px (avg / p50) | hold-still px | gain |
| --- | --- | --- | --- |
| 50 ms | 6.5 / 3.9 | 8.9 / 7.7 | +27% |
| 100 ms | 8.4 / 4.6 | 12.9 / 11.8 | +35% |
| 150 ms | 10.6 / 5.4 | 19.0 / 19.3 | +44% |

The CV gain *grows* with the horizon while the baseline falls
behind proportionally: that is what a correct model looks like.

Hard case — sinusoidal motion, fixed 1.5 s period, plus drift.
Gain at a 50 ms horizon as the fit window changes:

| fit window | CV error px | hold-still px | gain |
| --- | --- | --- | --- |
| 4 points | 11.7 | 20.7 | **+43.6%** |
| 6 points (default) | 15.3 | 21.8 | +29.6% |
| 8 points | 19.9 | 23.8 | +16.7% |
| 12 points | 30.5 | 30.6 | +0.5% |
| 15 points | 38.9 | 37.3 | **-4.1%** |

**The fit window is the whole ballgame.** It crosses over near
12 points: a window longer than a quarter of the motion's
period averages the velocity away and lags in phase, and past
that the "prediction" is worse than refusing to predict.
4 points measures best but is only a 4-sample fit against 2
free parameters, so it is the noisiest choice on real jittery
detections — the default stays at 6 until there is mosquito
data to decide with.

Reproduce either row with:

```
.venv-ai\Scripts\python.exe -m tools.check_detector --predict --motion pan  --clip-frames 150
.venv-ai\Scripts\python.exe -m tools.check_detector --predict --motion wave --clip-frames 150 --window 6
```

Two caveats on those numbers:

- Error is measured against the raw detection centre, so it includes the box's own
  jitter. It is an upper bound.
- Roughly a third of the submitted predictions could not be scored because the ID
  disappeared before the horizon landed. The table only measures predictions whose
  target survived, so real coverage is worse than the sample count suggests.

Neither number means anything for mosquitoes yet — that needs real captured data,
which is what `R` is for.

---

## V0.3 Goal

Verify:

YOLO
    ↓
ByteTrack
    ↓
Mosquito ID
    ↓
Trajectory

Measure:

ID continuity — does one target keep one ID across frames

ID churn — how many IDs exist versus how many objects are on screen

track() cost over detect() cost

---

## V0.2 Goal

Verify:

Camera
    ↓
OpenCV
    ↓
YOLO
    ↓
Detection
    ↓
Bounding Box

Measure separately:

read() time — camera and network

detect() time — model and GPU

FPS — what you actually get

---

## Important: There Is No Mosquito Class

A pretrained YOLO detects the 80 COCO classes.

Mosquito is not one of them.

So V0.2 uses weights/yolo11n.pt as a pipeline test:

anything the model knows about (person, car, laptop, ...)
gets a box, which proves the wiring, timing and rendering
are correct.

The detection code does not care which weights you use.

Once you have your own model, set:

"ai": { "model_path": "weights/mosquito.pt" }

and nothing else changes.

Training that model is the next real milestone.

---

## Requirements

Windows 10 / Windows 11

Python 3.10+

Redmi K40

IP Webcam Android App

PC and phone connected to the same Wi-Fi network.

For V0.2 detection:

NVIDIA GPU (this project was verified on RTX 4080 SUPER)

---

## Project Structure

visual-sense/
├── app/
│   ├── __init__.py
│   ├── server.py        V0.5 入口：起管线线程 + Web 服务
│   ├── pipeline.py      主循环（后台线程）
│   ├── runtime.py       config → 对象，运行时开关
│   ├── hub.py           管线 ↔ Web 的唯一交界处
│   ├── laser.py         激光状态机 + 驱动接缝
│   ├── web.py           FastAPI 端点
│   ├── render.py        离线渲染 + HUD 文字（check_detector 用）
│   ├── camera.py        USB / IP Webcam capture
│   ├── detector.py      YOLO wrapper, timing, states
│   ├── tracks.py        per-ID trajectory store
│   ├── motion.py        least-squares fit, velocity, prediction
│   ├── recorder.py      V0.4-B frame / track capture
│   ├── draw.py          bounding box, trail and prediction rendering
│   └── metrics.py       FPS / read / infer statistics
├── tools/
│   ├── selftest.py        one-command self-test runner (see Self-test)
│   ├── check_console.py   laser safety invariants + UI copy parity + CSS 不变量
│   ├── check_layout.py    真实渲染几何：自带一份控制台，逐视口量滚动条
│   ├── check_detector.py  offline detection / tracking / prediction check
│   ├── measure_stream.py  live stream latency and sharpness measurement
│   ├── label_frames.py    drag-box annotator, writes YOLO labels
│   └── build_dataset.py   capture sessions → YOLO training layout
├── web/
│   ├── index.html       单页控制台（无框架 / 无 CDN / 无构建）
│   ├── app.css
│   └── app.js
├── config/
│   └── config.json
├── dataset/             采集会话，按 R 生成，不进版本库
│   └── 20260919_235435/
│       ├── images/
│       ├── labels/
│       ├── annotations.jsonl
│       └── metadata.json
├── requirements.txt     V0.1 runtime (opencv, numpy)
├── requirements-ai.txt  V0.2 detection (ultralytics)
├── run.bat              V0.1, uses .venv
├── run-ai.bat           V0.2, uses .venv-ai
└── README.md

Two environments on purpose:

.venv       stays tiny, V0.1 always runs
.venv-ai    torch + CUDA + ultralytics (~3 GB)

torch is not in requirements-ai.txt because it must come
from the PyTorch CUDA wheel index — see run-ai.bat.

---

## Step 1

Install IP Webcam on Redmi K40.

Start the camera server.

The application should display something similar to:

http://192.0.2.120:8080

---

## Step 2

Open the following URL on the Windows PC:

http://192.0.2.120:8080

Replace the IP address with the actual IP address shown by the phone.

If the IP Webcam page opens, the phone network connection is working.

---

## Step 3

Edit:

config/config.json

Change:

"camera_url": "http://192.0.2.120:8080/video"

to the actual address.

For example:

"camera_url": "http://192.0.2.125:8080/video"

To use the phone camera over Wi-Fi also set:

"camera_mode": "network"

---

## Step 4 (V0.1, no AI)

Start:

run.bat

The first startup will create:

.venv/

and install the required packages.

---

## Step 5 (V0.2, with detection)

Start:

run-ai.bat

The first startup will create:

.venv-ai/

download torch + torchvision for CUDA, then install ultralytics.

The first model load downloads weights/yolo11n.pt
(about 5 MB) if it is not there yet.

If torch fails to install, edit TORCH_INDEX at the top of
run-ai.bat and try cu126 instead of cu130.

---

## Configuration

config/config.json

    camera_mode        "usb" or "network"

    camera_index       USB camera index, used when mode is usb

    camera_url         MJPEG URL, used when mode is network

    window_width       display width

    window_height      display height

    metrics_window     samples used for FPS / timing averages

    reconnect_interval seconds between reconnect attempts

    buffer_size        OpenCV capture buffer size

ai:

    enabled            load the model at startup

    model_path         weights file, relative to project root

    device             auto / cpu / cuda / 0

    imgsz              inference input size, 640 is a good start

    conf               minimum confidence

    iou                NMS threshold

    quantize           null = FP32, "fp16" = half precision (GPU)

    detect_every       run inference every N frames

    max_det            maximum boxes per frame

    track_enabled      V0.3: assign IDs and draw trails

    tracker            bytetrack.yaml (default) or another
                       ultralytics tracker config

    trail_frames       how many past centres to keep per ID

    predict_enabled    V0.4: fit velocity and draw the predicted point

    predict_horizon_ms how far ahead to extrapolate

    predict_window     points used for the least-squares fit.
                       Must stay shorter than the target's motion
                       period — see the sweep table above

    predict_min_samples  fewer points than this and no prediction

    predict_max_gap_s   a trail gap longer than this is not fitted

record:

    dataset_dir        where sessions go (default "dataset")

    stride             save every Nth inferred frame;
                       at 20 fps, 5 means 4 images/second

    max_frames         per-session cap, so a forgotten
                       recording cannot fill the disk

    jpeg_quality       90 keeps the pixels honest for labelling;
                       go lower only if disk is tight

    auto_labels        also write the model's boxes into labels_auto/
                       (off by default: labels/ is human truth only)

    save_preview       also write annotated copies

detect_every is the latency / throughput trade-off knob.

On a 30 FPS stream, detect_every = 2 halves GPU load and
keeps the last boxes on screen between inferences.

---

## Controls

Q or ESC

Exit the program.

D

Toggle AI at runtime.

Useful for A/B comparison: watch FPS with AI off,
toggle the **AI 检测** chip, watch FPS again with the same stream.

T

Toggle tracking at runtime.

Same idea one level down: with tracking off you get boxes,
with it on you get IDs and trails. The tracker state and the
stored trails are cleared on every toggle, otherwise the
first frames after switching back would inherit stale motion
history from the previous run.

P

Toggle prediction at runtime.

Nothing needs clearing when you flip it: the fit is recomputed
from the trajectory every frame, so switching off just stops
drawing and switching back on gives an immediate result.

R

Toggle frame capture.

While recording, a red `REC : 123 / 3000  4.0 fps  31 s  stride 5` line
appears in the HUD. Press it again to close the session; exiting with it
still open also closes it, so a session never ends up with a half-written
`metadata.json`.

---

## Data Collection (V0.4-B)

Each capture start opens a timestamped session under `dataset/`:

```
dataset/20260919_235435/
├── images/000000.jpg        原始帧，不带任何叠加
├── labels/000000.txt        人工真值（由标注工具写）
├── labels_auto/000000.txt   模型预标注（可选，auto_labels）
├── preview/000000.jpg       带框图（可选，save_preview）
├── annotations.jsonl        每帧一行，含 track_id
└── metadata.json            这次会话的来龙去脉
```

`labels/` 和 `labels_auto/` 分开是有意的：`labels/` 只属于人工真值。
混在一个目录里，"标了多少帧"就没法统计，
而且一次误触就可能把人工结果盖掉。
模型的原话本来就在 `annotations.jsonl` 里，所以没有丢信息。

会话目录在**第一帧落盘时**才创建，不是按 R 时。
连按 R 开/停曾经一次产生 16 个会话、其中 7 个 0 帧，
空目录既污染 `dataset/2026*` 的通配，
也要人工去分辨哪些是真的没采到。

`labels/*.txt` is the standard `class x_center y_center width height`,
normalised. Empty files are written on purpose — those are the negative
samples the human confirmed as empty.

**A missing label file is not a negative sample.** Frames nobody looked at
are skipped by `build_dataset`; only an explicitly empty `labels/xxx.txt`
counts as "confirmed: no mosquito here". Turning unlabelled frames into
backgrounds teaches the model to miss exactly the thing you care about.

`annotations.jsonl` exists because `labels.txt` cannot carry enough
information to be re-derived later:

```json
{"file": "images/000000.jpg", "frame": 5, "timestamp": 1789833275.190115,
 "size": [960, 720],
 "detections": [{"track_id": 3, "class_name": "person", "confidence": 0.8821,
                 "bbox_px": [223, 409, 344, 860],
                 "bbox_norm": [0.191146, 0.527778, 0.126042, 0.626389]}]}
```

Frame-to-frame correspondence, absolute time and the tracking ID are what
make velocity, acceleration and prediction error recomputable from disk.
Without them, re-running the analysis means re-running detection.

Recording only stores frames where inference actually ran, so an image and
its labels can never come from different frames when `detect_every > 1`.

Cost: writing one 960x720 JPEG plus its label and annotation line measured
3.2 ms on average, and loop FPS was unchanged (86.4 before, 86.0 during).

**The auto-labels are not training data yet.** They come from COCO weights,
which have no mosquito class, and `metadata.json` says so explicitly. What
this captures today is frames plus whatever the placeholder model believed.
The mosquito boxes still have to be drawn by hand, or the model has to be
retrained, before `labels/` means anything.

---

## Performance Metrics

The console shows:

Resolution

FPS

Frame Interval

Average read() time

Minimum read() time

Maximum read() time

Camera connection state

AI model name

AI detect() time, average with min ~ max

AI object count, current and average

AI device / imgsz / conf / detect_every

AI tracking: tracker name, live ID count, total trails,
trail length

AI prediction: horizon, how many tracks have enough points
(`ok 9 / 11`), fit window, minimum samples

Capture (only while recording): frames saved / cap, effective
capture FPS, elapsed seconds, stride

---

## Architecture

Redmi K40
    ↓
PhoneCamera / IP Webcam
    ↓
HTTP/MJPEG
    ↓
OpenCV VideoCapture
    ↓
Camera
    ↓            ↘
    ↓             read time → Metrics
    ↓
Detector (YOLO, every detect_every frames)
    ↓
detect() → Detection list        (tracking off)
    ↓
track()  → Track list with ID    (tracking on)
    ↓             ↘
    ↓              infer time → Metrics
    ↓
Tracks (id → recent centres)
    ↓
MotionPredictor (least-squares over the last N points)
    ↓
TrackState: smoothed x,y + vx,vy + predicted x,y
    ↓
draw_trails → draw_detections → draw_predictions
    ↓
HUD
    ↓
Offline render check

---

## Offline Verification

No camera, no window, no phone needed:

.venv-ai\Scripts\python.exe -m tools.check_detector

.venv-ai\Scripts\python.exe -m tools.check_detector --track

It runs the model on the sample images shipped with
ultralytics and writes:

runs/check/bus_boxes.png    boxes and trails only
runs/check/bus_hud.png      boxes + HUD + window resize

plus a per-image table of class, confidence,
coordinates and inference time.

With --track there is no video to hand it, so the tool
synthesises one: it pans the sample image sideways and runs
tracking over it, then prints an ID continuity table —

    ID 1   出现  60 帧  [0~59]  最大断裂 0 帧

which is the only way to see whether one target keeps one ID.
Static images cannot show that. Note the caveat: the seam
created by panning makes objects re-enter from the other side,
so a duplicated target legitimately gets a second ID.

`--predict` adds the error back-test and switches the
synthesised clip from `pan` to `wave` (sinusoidal). That is
deliberate: a constant-velocity clip is a gift to a
constant-velocity model, and the gain it produces means
nothing.

.venv-ai\Scripts\python.exe -m tools.check_detector --predict

    --horizons 50,100,150   which lead times to score
    --window 6             fit window, sweep this to retune
    --min-samples 4

Give it a real clip to measure tracking properly:

.venv-ai\Scripts\python.exe -m tools.check_detector --predict --video runs/live/clip.mp4

Use it to test new weights before pointing them at the camera:

.venv-ai\Scripts\python.exe -m tools.check_detector --model weights/mosquito.pt

---

## Self-test

```
.venv\Scripts\python.exe -m tools.selftest             # 11 项，含 4 项要 GPU/模型
.venv\Scripts\python.exe -m tools.selftest --stream    # 再加线上手机流实测
selftest.bat                                           # 同一件事，双击版
```

退出码 0 = 没有 FAIL。SKIP 不算失败但会列出来。

**项目里没有 pytest，也没有 `tests/`。** 这一层跑的是已经建成断言的那几项，
其余是跑通检查 —— 表上全绿只说明没崩，不说明行为对。所以每项还带一个 marker：
**退出码 0 但输出里没有预期的那行字，一律记 FAIL**，
免得一条静默跳过的检查冒充成一条通过的检查。

| 项 | 判什么 |
| --- | --- |
| 编译 app/ + tools/ | 跑通 |
| 模块导入（无 torch） | 跑通：轻环境不碰 torch 也能起 V0.1 |
| Detector 缺 torch 时降级 | 断言：只提示、`detect()` 返回空表，不抛异常 |
| P 传递滑窗自检 | 断言 6 条：滑窗 vs 整段拟合 vs 吸附模型框（见 Labelling 那张表） |
| `build_dataset --dry-run` | 断言：每条标签的几何合法，坏行报到文件和行号；无可用数据时非零退出 |
| `config.json` 可解析 | 跑通 |
| torch / ultralytics | 跑通：CUDA 可见性 |
| Yolo 加载 + 一帧检测/跟踪 | 断言：`load()` 成功，`track()` 返回列表 |
| **ID 持续性断言** | 合成 pan 片段画面是刚性平移，真值位移 4.00 px/帧。核对每个形状稳定的长命 ID：逐帧位移对得上真值（±1 px）、零断裂、单帧跳变 ≤12 px，且这样的目标 ≥2 个 |
| **预测增益断言** | wave 片段上 50/100 ms 至少打赢"原地不动"基线 10%，150 ms 也要赢；每个时距样本 ≥100 条；误差必须随时距单调上升 |
| **激光安全性质 + 界面文案** | 断言：未连接/未解锁/无目标三种情况下 `fire()` 必须被拒且状态不变；超 `max_dwell`、目标丢失、心跳失联三条路径都必须自动收光；急停在任何状态都能收光。中英两套文案键集必须相等，且没有定义了的死键 |

第 11 项不看界面、不需要 GPU、不抢 8060 端口（服务在跑就顺手查一次 `/api/snapshot` 的字段，没跑就跳过）。
激光那几条是**安全边界**而不是功能：接上硬件之前，界面上看不出它锁不锁得住，只有真去按才知道。

第 9 项的形状过滤不是可有可无的：`np.roll` 平移会让贴边的目标被画面边缘切开，
它的框在平移中不断变形，框心根本不跟着画面走（实测逐帧只挪 1.62 px）。
把这类目标一起判，自检就会把正常现象报成跟丢。
同理这项只在 60 帧上成立 —— 150 帧的平移会把所有目标都推到边缘切开，
那时没有可核对的对象，自检会 FAIL 并说明是这个前提没了，而不是跟踪器坏了。

第 10 项刻意用 wave：匀速平移对匀速外推是送分题，误差好看得没有意义。

---

## Stream Measurement

`tools/measure_stream.py` measures the camera side against the
same `app.camera` / `app.detector` code the window uses:

.venv-ai\Scripts\python.exe -m tools.measure_stream --seconds 15

It reports delivered FPS, read() and detect() latency, and a
sharpness figure (variance of Laplacian) for the whole frame
and for the centre region, plus how many frames are sharp.

Sharpness is scene dependent — a blank wall scores low even in
perfect focus. Use it to detect focus hunting (a wide spread
between p50 and max), not to judge absolute quality.

---

## Labelling

```
.venv\Scripts\python.exe -m tools.label_frames --list      # 进度，不开窗口
.venv\Scripts\python.exe -m tools.label_frames             # 开始标
```

Runs on the lightweight `.venv` — labelling needs no torch.

| key | |
| --- | --- |
| 左键拖拽 | 画框 |
| `A` / `D` | 上一帧 / 下一帧 |
| `1 2 3` | 切类别 |
| `U` / `X` | 撤销最后一个框 / 删鼠标所在的框 |
| `N` | 这帧确认没有目标（写空文件，真负样本） |
| `P` | 按 track_id 把本帧的框沿滑窗拟合传递到相邻帧 |
| `M` | 显示/隐藏模型原框做对照 |
| 滚轮 / 右键拖 / `0` | 缩放 / 平移 / 复位 |
| `S` | 保存 |
| `Q` / `ESC` | 退出（自动保存当前帧） |

Mosquitoes are a dozen pixels, so wheel-zoom and pan are not optional —
without them the boxes cannot be placed accurately.

**`P` is the reason this exists instead of labelImg.** You draw one box;
the tool finds which tracked object it corresponds to, fits a straight line
through a **sliding window centred on the target frame** (default 4 frames,
`--propagate-window`), and moves the box along that fit into the neighbouring
frames — up to 30 in each direction. Three properties matter:

- the human box keeps its size and shape; only the offset is transferred.
- it transfers along the fit, not by copying the model box at the target
  frame — ByteTrack jitters by more than a mosquito is wide, and snapping
  would copy that jitter straight into the ground truth.
- it only writes frames that have **no human label yet**, and only inside the
  ID's observed time span — extrapolating a line past the last sighting is
  fabricating data, so those frames are skipped and counted.

The window size is a real trade-off, not a preference, so it is pinned by a
headless self-check: `check_propagate_window()` runs on every start (7 ms, no
window, no dataset) and `--verbose` prints the table. It fits a synthetic
150-frame elliptical track — always turning, never straight — carrying
±0.9 px jitter, and measures two things with the same definitions as the
original sweep: **偏差** is how far the transferred box sits from the
tracker's own observation, **粗糙度** is the median second difference, i.e.
how much jitter survives into the label.

| 传递方式 | 偏差 | 粗糙度 |
| --- | --- | --- |
| 逐帧吸附到模型框（旧做法） | 0.00 px | **1.65 px** |
| 整段轨迹一条直线（上一版） | **40.84 px** | 0.00 px |
| 滑窗 3 帧 | 0.55 px | 0.59 px |
| 滑窗 4 帧（默认） | 0.67 px | 0.43 px |
| 滑窗 6 帧 | 0.90 px | 0.37 px |
| 滑窗 20 帧 | 6.27 px | 0.33 px |

Both extremes are wrong. Snapping is unbiased *because* it is the observation
— it saves the jitter, which is the one thing a human-drawn box must not
inherit. One line over the whole track is maximally smooth and 40 px off on a
turn, because a straight line through a curve is not a fit, it is a
compromise. The default costs 0.67 px of bias to remove 1.22 px of jitter;
widening to 20 buys 0.1 px of smoothness for another 5.6 px of bias.

The self-check asserts all four of those relationships, so reverting
`smoothed()` to snapping, or the window to a whole-span fit, fails at startup
with the numbers printed — both were verified to fail. Re-run the sweep on
real mosquito footage when it exists: a mosquito's turn is sharper than this
ellipse, and the right window follows the motion period, not the other way
round. (The first sweep, on a 1.5 s sinusoid since deleted, gave the same
shape with a sharper turn — 40.21 px whole-span bias, 1.72 px at window 4.)

Where the geometry is pure functions (`to_yolo`, `from_yolo`, `View`,
`find_track`, `transfer`, `TrackBook`, `Session`) and was checked headlessly:
YOLO round-trip within 0.0002 px, zoom keeps the point under the cursor fixed,
transfer preserves box size exactly, and unmatchable boxes are refused with a
stated reason rather than guessed.
The console has been driven end to end over HTTP and rendered in headless Chrome by
`tools/check_layout.py`, which boots its own server instance with the camera source
disabled so two runs are comparable, and judges a container by whether a scrollbar is
actually painted (`rect.height - clientHeight - borders >= 8`), not by `scrollWidth`.
That retracts an earlier claim here: **1280×800 does render two horizontal scrollbars**
(target table, help card) **and 390 px one** (top-bar chips) — both predate the current
change and are listed as open defects above; 1920/1600/1440 are clean.
One layout rule is now pinned by an assertion: `css_pairing()` in
`tools/check_console.py` flattens the stylesheet, resolves the cascade per viewport
mode, and forbids negative inline margins anywhere in a card — the trick that made a
card header bleed to the edge, and the reason two scrollbars appeared that I could not
see. It carries its own negative test (restoring the old bleed must make it fail),
because the first version of that assertion was vacuous. Everything beyond that —
density, alignment, whether a panel looks right — is still judged by eye.

Boxes are clipped to the frame **as you draw them**, so what is on screen is
what gets saved.

---

## Building a training set

Once sessions are labelled:

```
.venv-ai\Scripts\python.exe -m tools.build_dataset --dry-run
.venv-ai\Scripts\python.exe -m tools.build_dataset --names mosquito --keep-classes 0
```

produces

```
dataset_yolo/
├── images/train/  images/val/
├── labels/train/  labels/val/
└── mosquito.yaml        # path / train / val / nc / names
```

```
yolo detect train data=dataset_yolo/mosquito.yaml model=weights/yolo11n.pt epochs=100 imgsz=640
```

**The train/val split is per session, not per image.** Frames inside one session
are consecutive shots of the same scene; splitting per image would put the
validation set next door to training frames and inflate mAP enough to be
actively misleading. With fewer than two sessions the tool falls back to a
per-image split and says so loudly, because the resulting score cannot be
trusted.

Other things it does:

- `--labels prefer|human|auto` picks the label source. `prefer` (default)
  takes `labels/` and falls back to `labels_auto/`, **and says so loudly** —
  training on model guesses as if they were truth teaches the model its own
  mistakes. Once everything is labelled by hand, use `--labels human`
- `--keep-classes` / `--names` remap source class ids to the training class list,
  so COCO-numbered pre-labels can be turned into a single-class mosquito set
- every label line is re-validated: field count, numeric, box inside [0,1],
  non-degenerate. Geometry is checked **before** the class filter, otherwise a
  malformed box that happens to be a dropped class is counted as a normal drop
  and a corrupt dataset looks healthy
- `--drop-empty` removes background frames; by default they stay as negatives,
  and the tool warns when more than half the set is empty
- output filenames are prefixed with the session, so two sessions that both
  contain `000000.jpg` cannot silently overwrite each other
- `--dry-run` validates and reports without writing; `--link` hardlinks instead
  of copying; `--seed` makes the split reproducible

Verified on three generated sessions (36 frames, 27 with boxes, 9 empty, one
deliberately corrupt): split was 24/12 with the validation set being exactly one
whole session, and the bad box was reported down to file and line number.

---

## Small targets: the measured scale curve

Public mosquito data exists but it is macro data. The Mendeley set
(`external/mosquito_yolo`, 1160 images / 646 mosquito boxes after import) trained fine
— P 0.921 / mAP50 0.913 on its own val — and was still useless on this project's
operating point, which the scale curve below makes visible.

The curve: take 60 single-mosquito val images, rescale the mosquito so its box's short
side is T pixels inside a 960×720 frame, and count a hit at IoU ≥ 0.3.

| 目标短边 | 原尺寸(~150px) | 64px | 48px | 32px | 24px | 16px | 12px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mosquito_ext`（外部集直训） | 80.0% | 13.3% | 8.3% | 1.7% | 0% | 0% | 0% |
| `mosquito_synth`（合成小目标） | 51.7% | 88.3% | 83.3% | 61.7% | 40.0% | 23.3% | 1.7% |

`tools/synth_small.py` builds the second one's training data: it cuts 131 usable mosquito
silhouettes out of the **train split only** (the scale curve then tests on mosquitoes the
cutouts never saw), mattes them by border-median + Otsu + keep-the-component-touching-box-centre,
and pastes them at 8–48 px onto his own captured frames (85%) and the set's background images
(15%), with rotation, exposure matching, mask-only optical blur and a JPEG re-encode.
2000 images / 2478 boxes. Trained: val P 0.444 / R 0.365 / mAP50 0.331 — that val is a
*different background pool* than train, so it is the honest generalisation number, not a
regression against 0.913.

Two things to read carefully:

- **`synth` is worse at 原尺寸 (51.7% vs 80%)**, and that is expected, not a defect: it never
  saw a mosquito bigger than 48 px. A model trained for one scale band is only good in it.
- **推理分辨率比模型更值钱。** Same `synth` weights, same images, only `imgsz` changed:

| imgsz | 32px | 24px | 16px | 12px | 单帧 |
| --- | --- | --- | --- | --- | --- |
| 640 | 61.7% | 40.0% | 23.3% | 1.7% | 7.9 ms |
| 960 | 83.3% | 68.3% | 41.7% | 26.7% | 7.5 ms |
| 1280 | 85.0% | 85.0% | 65.0% | 50.0% | 10.1 ms |

**Read the next two tables before using this one.** It is measured at a fixed conf of 0.25, and
the matched-false-positive re-measurement below shows most of its gain was noise, not eyesight.

A 960×720 frame letterboxed to 640 turns a 15 px mosquito into 10 px of tensor, so inference
resolution matters — **but it is not free, and the first reading of this table was wrong.**
The same weights re-measured after training at 1280:

| 推理 imgsz | 模型 | 32px | 24px | 16px | 12px | 自己 40 帧的误报 |
| --- | --- | --- | --- | --- | --- | --- |
| 640 | 640 训 | 61.7% | 40.0% | 23.3% | 1.7% | 0 框 |
| 640 | 1280 训 | 26.7% | 13.3% | 0.0% | 0.0% | 0 框 |
| 1280 | 640 训 | 85.0% | 85.0% | 65.0% | 50.0% | **70 框 / 20 张** |
| 1280 | 1280 训 | 58.3% | 51.7% | 35.0% | 25.0% | 0 框 |

Two corrections that came out of that one table:

- **"0 false positives on my own frames" was only true at imgsz 640.** At 1280 the 640-trained
  model paints 70 boxes on 40 frames that contain no mosquito at all. So "+42 points at 16 px for
  +2 ms" was bought together with a noise machine, and is not a free win.
- **A hit rate measured at a fixed `conf` compares aggressiveness, not eyesight.** The 640-trained
  model scores 65% at 16 px partly because it fires on everything. `tools/scale_curve.py --match-fp`
  exists because of this: it raises each model's conf to the point where it produces zero false
  boxes on the captured frames, and only then compares hit rates.

And the training-resolution result is the opposite of what the val mAP suggests: training at 1280
raised val mAP50 from 0.331 to 0.401 and mAP50-95 from 0.124 to 0.180, **with recall unchanged at
0.365** — it tightened the boxes without teaching it to see more — yet at every small target on
the curve it is *worse* than the 640-trained model, at both inference sizes. Val mAP on a
composited set is not the number to make this decision with.

**More silhouettes: `tools.import_cutouts.py`.** The first synth set had only 131 cutouts, which
is a real ceiling on variety, so it pulls macro photos from Mosquito Alert (BioStudies
`S-BIAD249`, 40,978 images, CC BY — **attribution required if the model ever ships**) and keeps
just the silhouettes: those images have species labels and no boxes, so they cannot train a
detector directly, but they are a mosquito *shape* library. 1200 downloaded → **309 usable**
(25.7%), 8 species, silhouette short side p50 259 px. Two API quirks worth knowing before
touching it: the listing endpoint caps `length` at 1000 and returns an *empty list* rather than an
error above that, and the listing is ordered by upload date, so sampling takes evenly spaced
pages and then round-robins across species folders — otherwise the whole pool is one 2014 shoot.

The 33%→25.7% pass rate is the interesting number: picking "the largest non-background blob" is
the wrong prior on citizen macro photos, because **in 856 of 1200 photos the largest blob is a
fingertip, a bright lens disc, or bokeh** — the mosquito is next to it, not in it. What separates
them is measurable: real silhouettes have fill ratio (alpha area / bbox area) 0.08–0.33 and do not
touch the frame edge; the wrong blobs sit at 0.52–0.79 and usually do touch it. So
`MAX_FILL = 0.35` plus a border-touch rejection is what makes the pool usable. A filter that
passes 60/60 is not a filter — that was the first version, and the rendered sheet showed fingers.

`synth_small` now draws from both sources (131 + 309 = 440). Training-resolution and pool-size
effects are separated by a 2×2: {131, 440} × {imgsz 640, 1280}, all evaluated on the same
composited curve.

One Windows trap recorded where it can be found: `batch=16 @ imgsz=1280` does not report CUDA OOM.
The allocator failure kills the dataloader workers and torch raises
`RuntimeError: DataLoader worker exited unexpectedly` — the message points at the data loader and
the real problem is VRAM. `batch=8` fits in 7.5 GB of the 16 GB card.

False positives and the matched operating point — **this is where the first conclusion died.**
Measured at a fixed conf ≥ 0.25, then re-measured with each model's conf raised until it produces
zero false boxes on the 40 captured frames (`tools/scale_curve.py --match-fp`):

| 推理 imgsz | 模型 | 对齐用 conf | 32px | 24px | 16px | 12px |
| --- | --- | --- | --- | --- | --- | --- |
| 640 | 640 训 | 0.25（本就 0 假框） | 61.7% | 40.0% | 23.3% | 1.7% |
| 640 | 1280 训 | 0.25（本就 0 假框） | 26.7% | 13.3% | 0.0% | 0.0% |
| 1280 | 640 训 | **0.61**（要屏蔽 70 个假框） | 3.3% | 6.7% | **0.0%** | 0.0% |
| 1280 | 1280 训 | 0.25（本就 0 假框） | 58.3% | 51.7% | **35.0%** | **25.0%** |

The 640-trained model's "65% at 16 px @1280" was **entirely the false-positive flood**: on 40
frames containing no mosquito at all it drew 70 boxes across 20 images. Silence those and it must
run at conf 0.61, and then it sees nothing below 32 px.

What is actually left after correcting for it:

- **Train at the resolution you will deploy at.** 640/640 → 23.3% at 16 px; 1280/1280 → 35.0% at
  16 px and 25.0% at 12 px, with zero false positives to begin with. Mismatched pairs are the two
  bad corners of the table, and each looks like a win if you only read one row.
- **A fixed-conf curve measures aggressiveness, not eyesight.** That claim killed two of my own
  conclusions in a row: first "1280 inference is a free +42 points", then "training at 1280 is
  worse for small targets". Both were artifacts of comparing at conf 0.25.

**Final comparison** — all three trained at imgsz 1280, all measured on the same external test
(Mendeley val split, mosquitoes the cutout pool never saw), each at its own zero-false-positive
conf:

| 模型 | 剪影池 | 合成集切分 | 32px | 24px | 16px | 12px | 误报 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mosquito_synth131_1280` | 131 | 按背景种类 | 58.3% | 51.7% | 35.0% | 25.0% | 0 |
| `mosquito_synth2_1280` | 440 | 按背景种类 | 56.7% | 45.0% | 18.3% | 15.0% | 0 |
| **`mosquito_synth1280`** | 440 | **按底图分组** | **85.0%** | **63.3%** | 33.3% | 18.3% | 0 |

The middle row looked like "expanding the cutout pool 131→440 hurts". It doesn't — **the split was
the problem, and it hurt the model, not just the metric.** Splitting by background *kind* meant
train was the `_wall` session alone: 1700 images composited onto just **40** background photos,
with all 571 external backgrounds locked out of training. Group-splitting by background *identity*
gives train 32 wall + 456 external = **488 distinct backgrounds**, and val keeps the same mix, just
different photos. Same pool, same code, same seed: val mAP50 went 0.05 → 0.644.

So: pool size bought 32 px (58.3% → 85.0%) and 24 px (51.7% → 63.3%); at 16 px everything lands in
a band 33-35% and no configuration broke through it. **16 px is currently the wall**, and it is not
a data-volume wall — three different pools and two splits all stop at the same height.

What that means for `ai.imgsz`: raising it to 1280 with a 1280-trained model is the best measured
pair (33.3% at 16 px vs 23.3% for 640/640) at ~8.6 ms/frame, still inside the 30 FPS budget. But
this model has still never seen a real mosquito — every number above is composited. Its correct
use today is **pre-labelling** (`labels_auto/` in `tools.label_frames`), not the live window.

None of this is a real mosquito yet. The curve is composited, and a composite keeps the source
photo's optics; a 15 px mosquito on a wall through a phone lens over MJPEG may be softer than
anything measured here. Weight names carry provenance on purpose — `mosquito_ext` is trained on
public data as-is, `mosquito_synth1280` is the current best (440 silhouettes, group split, trained
and deployed at 1280), `mosquito_synth131_1280` keeps the smaller-pool baseline that row 1 of the
table above refers to, and each has a matching `runs/<name>/` directory verified by md5 against the
weight. `weights/mosquito.pt` stays reserved for a model trained on frames captured by `R`.

---

## Next: label what the recorder captured

Capture works, so the blocking item is now the labels
themselves — a mosquito dataset from this camera, in this room,
under this light:

1. Point the camera at wherever mosquitoes actually sit, hit the 采集 chip,
   capture a few hundred frames.
2. `tools.label_frames`: draw one box per target, press `P` to spread it
   across the neighbouring frames by track id.
3. `tools.build_dataset --labels human` → `mosquito.yaml`.
4. Train `weights/mosquito.pt`, point `ai.model_path` at it,
   and re-run the same back-test on real motion.

The `track_id` and `timestamp` already in `annotations.jsonl` mean
steps after training — velocity, acceleration, prediction error —
can be recomputed from disk without running detection again.

## Later: V0.5

Decide the model from data, not taste:

prediction error at 50 / 100 / 150 / 200 ms

Constant Velocity vs Kalman vs whatever the residuals demand

---

## Later: V0.6

Hardware control:

Predicted Position
    ↓
Controller
    ↓
Actuator

---

## Important

V0.1 through V0.4 intentionally do NOT include:

Kalman or learned motion models

Hardware control

Mosquito-specific training data

V0.4 proves the fit is arithmetically correct and that the fit
window matters more than the choice of model — at 15 points a
constant-velocity prediction is worse than refusing to
predict. It does not prove the system can see a mosquito, and
the gains quoted above are on synthetic motion, not insects.
