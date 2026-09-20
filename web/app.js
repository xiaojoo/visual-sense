/* VisualSense 控制台前端。
 *
 * 刻意不用任何框架和 CDN：
 * 这是一个局域网内的设备控制台，外网断了也得能用，
 * 而且多一套构建就多一个坏掉的地方。
 */

const I18N = {
  zh: {
    "top.stop": "停止服务",
    "pill.camera": "摄像头", "pill.ai": "检测", "pill.laser": "激光",
    "stage.title": "实时画面", "stage.full": "全屏", "stage.waiting": "等待画面…", "stage.fpsplot": "FPS 曲线",
    "quick.ai": "AI 检测", "quick.track": "跟踪", "quick.predict": "预测", "quick.record": "采集",
    "targets.title": "当前目标", "targets.class": "类别", "targets.size": "尺寸",
    "targets.conf": "置信", "targets.speed": "速度", "targets.none": "画面里没有目标",
    "targets.op": "操作",
    "targets.lock": "锁定", "targets.unlock": "已锁定", "targets.untracked": "未跟踪",
    "metrics.title": "性能指标", "metrics.read": "读取", "metrics.infer": "推理",
    "metrics.interval": "帧间隔", "metrics.avgdet": "平均目标", "metrics.frames": "累计帧",
    "log.title": "操作日志", "log.clear": "清空",
    "mod.camera.title": "摄像头检测",
    "mod.camera.lead": "画面来源、模型与推理参数。",
    "mod.camera.source": "数据源", "mod.camera.resolution": "分辨率",
    "mod.camera.device": "设备", "mod.camera.tracks": "活跃轨迹",
    "mod.laser.title": "激光驱赶 / 灭杀",
    "mod.laser.lead": "出光需要同时满足：驱动就绪、已解锁、有锁定目标。",
    "mod.laser.connect": "连接驱动", "mod.laser.arm": "解锁", "mod.laser.fire": "发射",
    "mod.laser.stop": "急停收光", "mod.laser.disarm": "上锁", "mod.laser.reset": "故障复位",
    "mod.laser.backend": "驱动", "mod.laser.locked": "锁定目标", "mod.laser.dwell": "最长停留",
    "mod.laser.cooldown": "冷却间隔", "mod.laser.power": "功率",
    "mod.laser.safety": "安全：本模块按可伤眼等级设计。发射前确认视场内无人、无宠物、无反光面。失联或超时会自动收光。",
    "mod.laser.off": "配置里 laser.enabled = false，模块未启用。",
    "mod.laser.sim": "当前是模拟驱动：状态机、按钮、日志全是真的，光不是。",
    "mod.laser.nocal": "像素→云台角度尚未标定，发射只把归一化坐标交给驱动。",
    "mod.record.title": "采集与标注",
    "mod.record.lead": "按帧存图给 tools/label_frames.py 做真值。",
    "mod.record.session": "会话", "mod.record.saved": "已存帧", "mod.record.time": "时长",
    "mod.record.stride": "步长", "mod.record.dropped": "退化框丢弃",
    "mod.record.next": "下一步：python -m tools.label_frames --session dataset/<会话名>",
    "mod.help.title": "使用说明",
    "help.1t": "接上画面", "help.1d": "手机装 PhoneCamera，和这台电脑同一个 WiFi，config 里填 camera_url。",
    "help.2t": "开检测", "help.2d": "点下面的「AI 检测」chip。第一次点会加载模型，画面会停几秒。",
    "help.3t": "开跟踪和预测", "help.3d": "跟踪给每个目标一个稳定 ID，预测才有的算。两个都开才看得到速度。",
    "help.4t": "攒真值", "help.4d": "开「采集」录一段，回来看见真虫就标，标完喂 build_dataset。",
    "help.5t": "激光", "help.5d": "先在 config 里把 laser.enabled 打开并选 backend，否则这里永远是灰色。",
    "help.klang": "切换界面语言", "help.klock": "锁定该目标给激光", "help.kstop": "任何状态都收光",
        "targets.source": "来源", "targets.notarget": "不可驱动",
    "mod.models.title": "模型组",
    "mod.models.lead": "同一个画面跑多路检测。每路自带跟踪和预测，因为 track_id 只在单路内唯一。",
    "mm.name": "名称", "mm.weights": "权重", "mm.every": "每N帧", "mm.ms": "推理",
    "mm.n": "目标", "mm.laser": "可驱动激光", "mm.op": "操作", "mm.on": "启用", "mm.off": "停用",
    "help.6t": "多路模型", "help.6d": "在 config 的 ai.models 里列几路就跑几路。每路的「每N帧」是能不能负担得起的关键 —— 人不需 25 fps，蚊子需要。",
"unit.s": "s", "unit.ms": "ms", "unit.px": "px",
    "msg.stopped": "服务已停止，可以关掉这个页面。",
    "state.connected": "已连接",
  },
  en: {
    "top.stop": "Shut down",
    "pill.camera": "Camera", "pill.ai": "Detect", "pill.laser": "Laser",
    "stage.title": "Live view", "stage.full": "Fullscreen", "stage.waiting": "Waiting for frames…", "stage.fpsplot": "FPS",
    "quick.ai": "AI detect", "quick.track": "Track", "quick.predict": "Predict", "quick.record": "Capture",
    "targets.title": "Targets", "targets.class": "Class", "targets.size": "Size",
    "targets.conf": "Conf", "targets.speed": "Speed", "targets.none": "No target in view",
    "targets.op": "Op",
    "targets.lock": "Lock", "targets.unlock": "Locked", "targets.untracked": "untracked",
    "metrics.title": "Performance", "metrics.read": "Read", "metrics.infer": "Infer",
    "metrics.interval": "Interval", "metrics.avgdet": "Avg targets", "metrics.frames": "Frames",
    "log.title": "Activity log", "log.clear": "Clear",
    "mod.camera.title": "Camera detection",
    "mod.camera.lead": "Stream source, weights and inference parameters.",
    "mod.camera.source": "Source", "mod.camera.resolution": "Resolution",
    "mod.camera.device": "Device", "mod.camera.tracks": "Active tracks",
    "mod.laser.title": "Laser repel / kill",
    "mod.laser.lead": "Emission needs all three: driver ready, armed, and a locked target.",
    "mod.laser.connect": "Connect", "mod.laser.arm": "Arm", "mod.laser.fire": "Fire",
    "mod.laser.stop": "E-stop", "mod.laser.disarm": "Safe", "mod.laser.reset": "Reset fault",
    "mod.laser.backend": "Driver", "mod.laser.locked": "Locked", "mod.laser.dwell": "Max dwell",
    "mod.laser.cooldown": "Cooldown", "mod.laser.power": "Power",
    "mod.laser.safety": "Safety: treated as eye-hazard class. Confirm no person, pet or reflective surface in the field before firing. Auto-extinguishes on timeout or lost heartbeat.",
    "mod.laser.off": "laser.enabled is false in config — module inactive.",
    "mod.laser.sim": "Simulated driver: state machine, buttons and log are real, the beam is not.",
    "mod.laser.nocal": "Pixel→pan/tilt not calibrated; fire only hands the driver normalised coords.",
    "mod.record.title": "Capture & label",
    "mod.record.lead": "Frames saved for ground truth via tools/label_frames.py.",
    "mod.record.session": "Session", "mod.record.saved": "Frames", "mod.record.time": "Elapsed",
    "mod.record.stride": "Stride", "mod.record.dropped": "Degenerate boxes dropped",
    "mod.record.next": "Next: python -m tools.label_frames --session dataset/<name>",
    "mod.help.title": "How to use",
    "help.1t": "Get a picture", "help.1d": "Install PhoneCamera on the phone, same Wi-Fi, set camera_url in config.",
    "help.2t": "Turn on detection", "help.2d": "Tap the AI chip. First tap loads the model and stalls the view for a few seconds.",
    "help.3t": "Track and predict", "help.3d": "Tracking gives each target a stable id, which prediction needs. Enable both to see speed.",
    "help.4t": "Collect truth", "help.4d": "Record a clip with Capture, label the real insects later, then feed build_dataset.",
    "help.5t": "Laser", "help.5d": "Set laser.enabled and pick a backend in config first, or this card stays grey.",
    "help.klang": "Switch language", "help.klock": "Lock this target for the laser", "help.kstop": "Extinguish from any state",
        "targets.source": "Source", "targets.notarget": "not targetable",
    "mod.models.title": "Model group",
    "mod.models.lead": "Several detectors on one frame. Each keeps its own tracker and predictor, because track_id is only unique inside one tracker.",
    "mm.name": "Name", "mm.weights": "Weights", "mm.every": "Every N", "mm.ms": "Infer",
    "mm.n": "N", "mm.laser": "Can aim", "mm.op": "Op", "mm.on": "Enable", "mm.off": "Disable",
    "help.6t": "Several models", "help.6d": "List them under ai.models in config. Per-model “every N frames” is what keeps it affordable — people do not need 25 fps, mosquitoes do.",
"unit.s": "s", "unit.ms": "ms", "unit.px": "px",
    "msg.stopped": "Server stopped. You can close this page.",
    "state.connected": "online",
  },
};

let LANG = localStorage.getItem("vs.lang") || "zh";
const t = (key) => (I18N[LANG] && I18N[LANG][key]) || I18N.zh[key] || key;

const $ = (id) => document.getElementById(id);
const state = { snapshot: null, busy: false };

/* ---------- i18n ---------- */

function applyLang() {
  document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
  document.documentElement.dataset.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  $("lang-zh").classList.toggle("on", LANG === "zh");
  $("lang-en").classList.toggle("on", LANG !== "zh");
  document.title = LANG === "zh" ? "VisualSense 控制台" : "VisualSense console";
  if (state.snapshot) render(state.snapshot);
}

function setLang(next) {
  LANG = next;
  localStorage.setItem("vs.lang", next);
  applyLang();
}

/* ---------- 通信 ---------- */

async function command(name, args) {
  if (state.busy) return null;
  state.busy = true;
  $("quick").querySelectorAll(".chip").forEach((c) => (c.disabled = true));

  try {
    const response = await fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, args: args || {} }),
    });
    const data = await response.json();

    toast(data.message || "", data.ok ? "ok" : "err");
    refreshSoon();

    return data;
  } catch (error) {
    toast("连不上服务：" + error, "err");
    return null;
  } finally {
    state.busy = false;
    $("quick").querySelectorAll(".chip").forEach((c) => (c.disabled = false));
  }
}

function toast(message, kind) {
  if (!message) return;
  const el = $("toast");
  el.textContent = message;
  el.className = "toast " + (kind || "");
  el.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => (el.hidden = true), 3600);
}

let refreshTimer = null;

function refreshSoon() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(poll, 120);
}

async function poll() {
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    const data = await response.json();

    state.snapshot = data;
    render(data);
    renderEvents(data.events || []);
    drawSpark(data.history || []);
  } catch (error) {
    if ($("stream").src) $("video-empty").hidden = false;
  }
}

function renderEvents(events) {
  const list = $("log");

  // 整段重建，不按 seq 追加：日志会去重改写已有条目的计数，
  // 追加式渲染永远看不到改写后的值，而且顺序一错就整页只剩一行。
  list.textContent = "";

  for (const item of events) {
    const li = document.createElement("li");
    const text = item.text || "";

    if (/失败|拒绝|未知|错误|fault/i.test(text)) li.className = "err";
    else if (/已开启|已连接|成功|已锁定|已收到|开始/.test(text)) li.className = "ok";

    li.innerHTML = `<time>${new Date(item.at * 1000).toTimeString().slice(0, 8)}</time><span></span>`;
    li.querySelector("span").textContent =
      text + (item.count > 1 ? `  ×${item.count}` : "");
    list.appendChild(li);
  }
}

/* ---------- 渲染 ---------- */

function render(data) {
  const camera = data.camera || {};
  const ai = data.ai || {};
  const metrics = data.metrics || {};
  const record = data.record || {};
  const laser = data.laser || {};
  const targets = data.targets || [];

  pill("camera", camera.connected ? "on" : "err", camera.connected ? t("state.connected") : "OFF");
  pill("ai", ai.any_ready ? "on" : (ai.models || []).some((m) => m.state === "loading") ? "warn" : "",
       `${ai.running || 0}/${ai.count || 0} ` + (ai.models || []).map((m) => m.key).join("+"));
  pill("laser", laser.state === "firing" ? "err" : laser.state === "armed" ? "warn" : laser.state === "idle" ? "on" : "", (laser.state || "offline").toUpperCase());

  $("uptime").textContent = clock(data.uptime_s || 0);

  $("stream-res").textContent = (camera.resolution || []).join("×") || "—";
  $("stream-age").textContent = ((data.stream || {}).frame_age_s ?? 0).toFixed(2) + " s";
  $("video-empty").hidden = !!camera.connected;

  // 摄像头断了之后，滚动窗口里的统计还是最后一次在线时的值。
  // 画面明明黑着却显示 24 FPS，比不显示更糟 —— 它会让人以为还在跑。
  const live = !!camera.connected && ((data.stream || {}).frame_age_s ?? 99) < 1.5;

  $("m-fps").innerHTML = fmt(metrics.fps, 1) + `<small>fps</small>`;
  $("m-read").innerHTML = fmt(metrics.read_ms, 1) + `<small>${t("unit.ms")} · ${fmt(metrics.read_min_ms, 0)}–${fmt(metrics.read_max_ms, 0)}</small>`;
  $("m-infer").innerHTML = fmt(metrics.infer_ms, 1) + `<small>${t("unit.ms")} · ${fmt(metrics.infer_max_ms, 0)} max</small>`;
  $("m-interval").innerHTML = fmt(metrics.frame_interval_ms, 1) + `<small>${t("unit.ms")}</small>`;
  $("m-avgdet").textContent = fmt(metrics.avg_detections, 2);
  $("m-frames").textContent = (metrics.frames || 0).toLocaleString();

  document.querySelector(".metrics").classList.toggle("dim", !live);

  const age = (data.stream || {}).frame_age_s ?? 0;
  const hint = $("metric-hint");

  if (!live) {
    hint.textContent = `摄像头未连接，以上停在最后一次出帧（${age.toFixed(1)} s 前）`;
    hint.style.color = "var(--danger)";
  } else if (metrics.fps > 0 && metrics.fps < 15) {
    hint.textContent = `FPS ${fmt(metrics.fps, 1)} — 模型组共 ${fmt(ai.infer_total_ms, 1)} ms，减路数或加大"每N帧"`;
    hint.style.color = "var(--warn)";
  } else {
    hint.textContent = "";
    hint.style.color = "";
  }

  chip("chip-ai", ai.any_ready);
  chip("chip-track", ai.any_tracking);
  chip("chip-predict", ai.any_predicting);
  chip("chip-record", record.recording);
  $("rec-badge").hidden = !record.recording;

  $("k-source").textContent = camera.mode || "—";
  $("k-res").textContent = (camera.resolution || []).join(" × ") || "—";
  $("k-device").textContent = (ai.models || [])[0]?.device || "—";
  $("k-tracks").textContent = ai.tracks ?? 0;
  $("k-camerr").textContent = camera.error || (ai.models || []).map((m) => m.error).filter(Boolean)[0] || "";

  renderModels(ai);

  $("r-session").textContent = record.session || "—";
  $("r-saved").textContent = record.saved ?? 0;
  $("r-time").textContent = fmt(record.seconds, 1) + " " + t("unit.s");
  $("r-stride").textContent = "1 / " + (record.stride ?? "—");
  $("r-dropped").textContent = record.dropped_boxes ?? 0;
  tag("rec-state", record.recording ? "err" : "", record.recording ? "REC" : "IDLE");

  renderLaser(laser, targets);
  renderTargets(targets);
}

function renderModels(ai) {
  const models = ai.models || [];

  tag("mm-state", ai.running ? (ai.running === ai.count ? "ok" : "warn") : "",
      `${ai.running || 0} / ${ai.count || 0}`);

  const body = $("mm-body");
  body.textContent = "";

  for (const model of models) {
    const row = document.createElement("tr");

    row.innerHTML =
      `<td>${esc(model.label || model.key)}</td>` +
      `<td class="mono-cell" title="${esc(model.model || "")}"></td>` +
      `<td>${model.imgsz}</td>` +
      `<td>${fmt(model.conf, 2)}</td>` +
      `<td>${model.detect_every}</td>` +
      `<td class="num">${fmt(model.infer_ms, 1)}</td>` +
      `<td class="num">${model.last_count}</td>` +
      `<td>${model.targetable ? '<i class="yes">是</i>' : '<i class="no">否</i>'}</td>` +
      `<td></td>`;

    // 权重名被省略号截断时，类别白名单还得看得见，所以用 JS 填文本而不是 innerHTML
    const wt = row.children[1];
    wt.textContent = model.model || "—";
    if (model.classes) {
      const em = document.createElement("em");
      em.textContent = `[${model.classes.join(",")}]`;
      wt.append(" ", em);
    }

    const cell = row.lastChild;
    const button = document.createElement("button");
    button.textContent = model.running ? t("mm.off") : t("mm.on");
    button.className = model.running ? "on" : "";
    button.disabled = model.state === "loading";
    button.onclick = () => command("toggle_ai", { key: model.key });
    cell.appendChild(button);
    body.appendChild(row);

    if (model.error) {
      const err = document.createElement("tr");
      err.innerHTML = `<td colspan="9" class="rowerr"></td>`;
      err.querySelector("td").textContent = model.error;
      body.appendChild(err);
    }
  }

  // 帧率预算写成人话：多一路就是多一次前向，这里给的是实测合计。
  const budget = $("mm-budget");
  if (!models.length) { budget.textContent = ""; return; }

  const total = ai.infer_total_ms || 0;
  budget.textContent =
    `合计推理 ${fmt(total, 1)} ms → 上限约 ${total > 0 ? Math.floor(1000 / total) : "∞"} fps。`
    + `加载中的那一路会先占住主循环。`;
}

function renderLaser(laser, targets) {
  tag("laser-state", laser.state === "firing" ? "err" : laser.state === "armed" ? "warn" : laser.state === "idle" ? "ok" : "",
      (laser.state || "offline").toUpperCase());

  const notes = [];
  if (!laser.enabled) notes.push(t("mod.laser.off"));
  if (laser.simulated && laser.state !== "offline") notes.push(t("mod.laser.sim"));
  if (laser.available) notes.push(t("mod.laser.nocal"));
  if (laser.error) notes.push(laser.error);

  const flag = $("laser-flag");
  flag.hidden = notes.length === 0;
  flag.textContent = notes.join(" · ");
  flag.className = "laser-flag" + (laser.simulated ? " sim" : "");

  $("l-backend").textContent = laser.backend + (laser.simulated ? " (sim)" : "");
  $("l-locked").textContent = laser.locked_id == null ? "—" : (laser.locked_source || "") + "#" + laser.locked_id;
  $("l-dwell").textContent = fmt(laser.max_dwell_s, 1) + " " + t("unit.s");
  $("l-cooldown").textContent = fmt(laser.cooldown_s, 1) + " " + t("unit.s");
  $("l-power").textContent = laser.power_w ? fmt(laser.power_w, 1) + " W" : "—";

  const hasTarget = targets.length > 0;

  $("lz-connect").disabled = !laser.enabled || laser.state !== "offline";
  $("lz-arm").disabled = laser.state !== "idle";
  $("lz-fire").disabled = laser.state !== "armed" || !hasTarget;
  $("lz-stop").disabled = laser.state !== "firing";
  $("lz-disarm").disabled = !["armed", "firing"].includes(laser.state);

  const extra = document.getElementById("lz-reset") || makeReset();
  extra.hidden = laser.state !== "fault";
}

function makeReset() {
  const button = document.createElement("button");
  button.id = "lz-reset";
  button.className = "btn warn";
  button.textContent = t("mod.laser.reset");
  button.onclick = () => command("laser", { action: "reset" });
  $("mod-laser").querySelector(".btn-row").appendChild(button);
  return button;
}

function renderTargets(targets) {
  $("target-count").textContent = targets.length;
  $("targets-empty").hidden = targets.length > 0;

  const body = $("target-body");
  body.textContent = "";

  const layer = $("targets");
  layer.textContent = "";

  for (const target of targets) {
    const color = palette(target.track_id);
    const tracked = target.track_id >= 0;

    const box = document.createElement("div");
    box.className = "tbox" + (target.locked ? " locked" : "") + (tracked ? "" : " untracked");
    box.style.cssText = `--tc:${color};left:${target.x * 100}%;top:${target.y * 100}%;width:${target.w * 100}%;height:${target.h * 100}%`;
    box.innerHTML = `<span>${tracked ? "#" + target.track_id + " " : ""}${esc(target.class)} ${fmt(target.conf, 2)}</span>`;
    if (tracked && target.targetable) box.onclick = () => lock(target);
    layer.appendChild(box);

    const row = document.createElement("tr");
    row.innerHTML =
      `<td>${esc(target.label || target.source || "—")}</td>` +
      `<td><i class="swatch" style="background:${color}"></i>${tracked ? target.track_id : "—"}</td>` +
      `<td>${esc(target.class)}</td>` +
      `<td class="num">${fmt(target.short_px, 0)}${t("unit.px")}</td>` +
      `<td class="num">${fmt(target.conf, 2)}</td>` +
      `<td class="num">${target.speed_px_s == null ? "—" : fmt(target.speed_px_s, 0)}</td>` +
      `<td></td>`;

    const cell = row.lastChild;
    const button = document.createElement("button");
    // 没拿到跟踪 ID 的目标不能锁：它下一帧就可能变成别的东西，
    // 锁上去等于让激光去追一个不存在的名字。
    // 不可驱动的那几路（比如"人"）压根不该出现在激光的候选里
    button.disabled = !tracked || !target.targetable;
    button.title = !target.targetable ? t("targets.notarget") : tracked ? "" : t("targets.untracked");
    button.textContent = !target.targetable
      ? t("targets.notarget")
      : tracked ? (target.locked ? t("targets.unlock") : t("targets.lock")) : t("targets.untracked");
    button.className = target.locked ? "on" : "";
    button.onclick = () => lock(target);
    cell.appendChild(button);

    body.appendChild(row);
  }
}

function lock(target) {
  command("laser", {
    action: "lock",
    track_id: target.locked ? null : target.track_id,
    source: target.source,
  });
}

function pill(name, kind, text) {
  const el = document.querySelector(`.pill[data-pill="${name}"]`);
  el.className = "pill " + (kind || "");
  $("pill-" + name).textContent = text;
}

function chip(id, on) {
  const el = $(id);
  el.parentElement.classList.toggle("on", !!on);
  el.textContent = on ? "ON" : "OFF";
}

function tag(id, kind, text) {
  const el = $(id);
  el.className = "tag " + (kind || "");
  el.textContent = text;
}

/* ---------- 杂项 ---------- */

function fmt(value, digits) {
  return value == null || Number.isNaN(value) ? "—" : Number(value).toFixed(digits);
}

function clock(seconds) {
  const total = Math.floor(seconds);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(Math.floor(total / 3600))}:${pad(Math.floor(total / 60) % 60)}:${pad(total % 60)}`;
}

function esc(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function palette(id) {
  const COLORS = ["#4c8dff", "#3ecf8e", "#f5a524", "#e06c9f", "#5ec8e0", "#b98cff", "#d9d94c", "#ff7a59"];
  return COLORS[Math.abs(Number(id) || 0) % COLORS.length];
}

function drawSpark(history) {
  const canvas = $("spark");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = 480, height = 46;

  if (canvas.width !== width * dpr) { canvas.width = width * dpr; canvas.height = height * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const points = history.map((row) => row[1]).slice(-90);

  if (points.length < 2) return;

  // 峰值贴着数据走，不用 max(20, …)：
  // 稳定 21 FPS 时那条线会顶在顶上，整块填充看着像坏了。
  const peak = Math.max(30, Math.max(...points) * 1.25);
  const step = width / (points.length - 1);

  ctx.beginPath();
  points.forEach((value, i) => {
    const y = height - 4 - (value / peak) * (height - 10);
    i ? ctx.lineTo(i * step, y) : ctx.moveTo(0, y);
  });
  ctx.strokeStyle = "#4c8dff";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.lineTo((points.length - 1) * step, height);
  ctx.lineTo(0, height);
  ctx.closePath();
  ctx.fillStyle = "rgba(76,141,255,.13)";
  ctx.fill();
}

/* ---------- 装配 ---------- */

document.querySelectorAll(".chip").forEach((chipEl) => {
  chipEl.onclick = () => command(chipEl.dataset.cmd);
});

$("lz-connect").onclick = () => command("laser", { action: "connect" });
$("lz-arm").onclick = () => command("laser", { action: "arm" });
$("lz-fire").onclick = () => command("laser", { action: "fire" });
$("lz-stop").onclick = () => command("laser", { action: "stop" });
$("lz-disarm").onclick = () => command("laser", { action: "disarm" });

$("btn-clear").onclick = () => command("clear_log");
$("btn-stop").onclick = async () => {
  if (!confirm(LANG === "zh" ? "确定停止服务？" : "Shut the server down?")) return;
  await command("stop");
  setTimeout(() => toast(t("msg.stopped"), "ok"), 400);
};

$("btn-fullscreen").onclick = () => {
  const wrap = $("video-wrap");
  document.fullscreenElement ? document.exitFullscreen() : wrap.requestFullscreen();
};

$("lang-zh").onclick = () => setLang("zh");
$("lang-en").onclick = () => setLang("en");

/* 激光心跳：解锁或出光期间，页面活着就得一直说。
   手机把标签切到后台时浏览器会节流定时器，
   所以心跳断了自动收光正是我们要的行为，不是 bug。 */
setInterval(() => {
  const laser = (state.snapshot || {}).laser || {};
  if (["armed", "firing"].includes(laser.state)) {
    fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "laser", args: { action: "heartbeat" } }),
    }).catch(() => {});
  }
}, 1000);

/* 三件"必须由真实渲染量出来"的事写回 CSS 变量：右栏宽度、表格行高、换行后的跨列。
   两边各写一个数字迟早会对不齐，而且模块卡是 auto-fit 出来的，宽度随视口变。
   模块行铺满整个布局宽度，它的列宽不依赖 --rail-w —— 不会来回震荡。 */
function syncLayout() {
  const modules = document.querySelector(".modules");
  const card = document.getElementById("mod-camera");
  const row = document.querySelector("#target-table th");
  if (!modules || !card || !row) return;

  const set = (name, value, unit = "px") => {
    const next = `${Math.round(value)}${unit}`;
    const current = document.documentElement.style.getPropertyValue(name);
    if (value > 0 && current !== next) document.documentElement.style.setProperty(name, next);
  };

  // 卡片一旦换行，列数变少、单卡变宽：1280 宽下模块只有 2 列，卡片 612px，
  // 右栏跟着变 612 就把画面挤扁。所以右栏最多吃掉 34% 视口宽。
  // 只在"左画面 + 右栏"两列的贴合布局里限宽 —— 单列模式下 --rail-w 根本没人用，
  // 但 390 宽的手机上 innerWidth*0.34 = 133px，写进去是个没意义的假数字。
  const twoCol = getComputedStyle(document.querySelector(".layout"))
    .gridTemplateColumns.trim().split(/\s+/).length > 1;
  const cap = twoCol ? innerWidth * 0.34 : Infinity;

  set("--rail-w", Math.min(card.getBoundingClientRect().width, cap));

  // 行高量表头：目标表按内容长，上下限用行数表达（4~20 行），所以要有真实行高
  set("--row-h", row.getBoundingClientRect().height);

  /* 换行之后最后一行常常差几张卡，右边空一大片 —— grid 不会自己把最后一行摊开。
     算出该有几列，让最后一张卡跨掉剩下的轨道：span = 列数 - (卡数-1) % 列数。
     5 张卡：4 列→跨 4，3 列→跨 2，2 列→跨 2，1 列→跨 1（不换行时也是 1）。

     列数不能去量 gridTemplateColumns：最后一张卡的跨列本身会逼 grid 多开轨道，
     量回来的列数再算出更大的跨列 —— 正反馈能把 1440 撑成 4 条轨道、横向溢出。
     所以用"容器宽 ÷ 最小卡宽"直接算，两边都只认 --card-min 这一个数。 */
  const style = getComputedStyle(modules);
  const box = modules.getBoundingClientRect().width;
  const gap = parseFloat(style.columnGap) || 0;
  const min = parseFloat(style.getPropertyValue("--card-min")) || 0;
  const cols = min > 0
    ? Math.max(1, Math.floor((box + gap) / (min + gap)))
    : style.gridTemplateColumns.trim().split(/\s+/).length;
  const cards = modules.querySelectorAll(".card").length;
  const rem = cards % cols;

  /* 列数够放下所有卡（宽屏）时绝不能跨：那时 auto-fit 会把空轨道收掉，
     五张卡本来就等宽，再跨一下就把最后一张撑成两倍宽了。
     只有真的换行、最后一行没坐满，才把剩下的轨道补给最后一张。 */
  const span = cols >= cards || rem === 0 ? 1 : 1 + (cols - rem);

  set("--last-span", span, "");
}

/* 不能用 load 事件：MJPEG 响应永不结束，window.load 在这页上根本不会触发
   （headless 截图卡死就是同一个原因）。
   ResizeObserver 盯模块容器 —— 字体加载完、视口变化、卡数变化都会回调。 */
function watchLayout() {
  const modules = document.querySelector(".modules");
  if (!modules) return;

  if (window.ResizeObserver) new ResizeObserver(syncLayout).observe(modules);
  addEventListener("resize", syncLayout);
  syncLayout();
}

/* MJPEG 断了不会自己回来：<img> 的 src 失败一次就停在碎图标上。
   服务端重启、手机息屏、WiFi 漫游都会掐掉这条长连接，
   所以出错后要重新挂一次 src，而不是等用户刷新页面。 */
let streamRetry = null;

function attachStream() {
  clearTimeout(streamRetry);
  streamRetry = null;

  const img = $("stream");
  img.onerror = () => { if (!streamRetry) streamRetry = setTimeout(attachStream, 1500); };
  img.onload = () => { clearTimeout(streamRetry); streamRetry = null; };
  img.src = "/stream.mjpg?t=" + Date.now();
}

applyLang();
watchLayout();
attachStream();
poll();
setInterval(poll, 600);
