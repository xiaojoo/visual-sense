# VisualSense

小飞虫实时检测控制台：手机摄像头当网络相机 → 本机 GPU 跑 YOLO → 浏览器里看画面、目标、
指标、采集和激光模块。局域网内的手机和平板都能打开，界面中文优先、可切英文。

现状：取流 / 检测 / 跟踪 / 预测 / 采集 / 控制台都能跑；**真虫精度还不行**，激光没有硬件。

## 功能

| 模块 | 做什么 |
| --- | --- |
| 取流 | USB 相机或局域网 MJPEG（IP Webcam / PhoneCamera 一类），断线自动重连 |
| 检测 | Ultralytics YOLO，**多路模型并发**：每路自己的 `imgsz` / `conf` / 每 N 帧 / 类别白名单 |
| 跟踪 | ByteTrack 稳定 ID，每路一套（`track_id` 只在单路内唯一），画轨迹拖尾 |
| 预测 | 轨迹最小二乘拟合速度 + 匀速外推，画预测点 |
| 采集 | 按帧存图 + YOLO 预标签（按模型分目录）+ `annotations.jsonl`；配离线拖框标注器 |
| 控制台 | 浏览器界面：实时画面、目标表、性能指标、操作日志、中英切换 |
| 激光 | 驱赶/灭杀状态机（连接 / 解锁 / 发射 / 急停 / 上锁），带最长停留、冷却、心跳失联自动收光。**只有模拟驱动** |

只有标了 `targetable` 的那一路模型能驱动激光，默认是关的。

## 快速开始

```
git clone https://github.com/xiaojoo/visual-sense.git
cd visual-sense
copy config\config.example.json config\config.json     # 填相机地址
run-ai.bat                                             # 带检测；run.bat 是不要 torch 的基线
```

然后浏览器开 `http://127.0.0.1:8060/`，手机/平板开打印出来的局域网地址。

- `run.bat` 建 `.venv`（opencv + fastapi + uvicorn），没有 torch 时检测会显示"未安装"，其余功能照用。
- `run-ai.bat` 建 `.venv-ai`（torch + ultralytics），首次会下载约 3 GB。
- 相机地址填成 `"camera_url": "http://<手机IP>:8080/video"`，`camera_mode` 保持 `network`。
- 权重不在仓库里。`ai.model_path` 指到你自己的 `.pt`，或者先把 `ai.enabled` 设为 `false`。

## 界面

- **顶部 chips**：AI 检测 / 跟踪 / 预测 / 采集四个开关，状态灯在标题栏。
- **实时画面**：MJPEG 流 + 目标框叠加，点框 = 把这个目标交给激光；右上"全屏"。
- **当前目标**：来源、ID、类别、尺寸、置信、速度、操作。表按内容长，最少 4 行、最多 20 行，
  超了在表内滚；它每多长一行，下面的操作日志就缩一行，缩到只剩标题栏为止。
  没拿到稳定 ID 的目标不能锁。
- **模型组**：每一路的参数和状态，可以单独启停。
- **激光驱赶**：连接驱动 → 解锁 → 发射；急停任何状态都能收光。
- **采集与标注**：开录、看已存帧数、下一步的标注命令直接可复制。

## 自检

```
.venv\Scripts\python.exe -m tools.selftest          # 11 项：编译、导入、降级路径、ID 持续性、激光安全性质
.venv\Scripts\python.exe -m tools.check_layout      # 起一份自带控制台，6 档视口量真实渲染出来的几何
```

`check_layout` 判的是"有没有真的画出滚动条 / 标题栏滚不滚得动"，不是 `scrollWidth` 那种猜测。

## 缺点 / 没做的

- **真虫精度未达标。** 只在 4 帧真虫上评过分：俯视姿态 36px 命中（IoU 0.58），侧视姿态 26 / 32 / 37px 在 conf 0.01 时**全部漏检**。这是姿态墙，不是分辨率墙 —— 现有训练池里的虫子全是同一个形状。合成集上 16px 的命中率是 33%。
- **没有真硬件。** 激光只有 `sim` 驱动，`laser.enabled` 默认 `false`；像素 → 云台角度的标定没做，界面上会一直挂着这条提示。
- **推流会漏线程（未修）。** 相机长时间离线时，每个连过的客户端会永久占住一个 worker 线程，攒够整站无响应 —— 实测 7 条连接 + 40 分钟无帧即可复现（进程还在，但 `/`、`/api/health` 全超时）。
- **没有认证。** 局域网裸 HTTP，同网任何设备打开就能发命令。要放到不可信网络之前先自己加反代鉴权。
- **多路采集的溯源不全。** 会话的 `metadata.json` 只记第一路的权重和类别表，`labels_auto/<key>/` 和 `annotations.jsonl` 才是逐路的。
- **布局遗留。** 390 手机档"已锁定"按钮和来源列各缺 2~3px；1440×900 及以下右栏要竖滚（日志卡最小 67px 放不下）。
- **只测过 Windows + Chrome/Edge。** Firefox 走标准滚动条样式那条分支；没在 Linux 上验证过。
- 手机推流的编码上限约 20–25 fps，**瓶颈是相机不是模型**（3 路模型并发实测 24.0 ms、显存 0.07 GB allocated / 0.23 GB reserved）。

## 许可

CC0 1.0 Universal，见 `LICENSE`。复制、修改、分发、商用都不必打招呼。

逐版本的验收数据、量法和踩过的坑记在 [`docs/notes.md`](docs/notes.md)。
