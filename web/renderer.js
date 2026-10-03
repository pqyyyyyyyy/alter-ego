// 渲染核心：收后端消息 → 分发到各图层。
// 原则：前端不做任何业务判断，只把指令变成画面。

// ======================== 配置 ========================
const USE_MOCK    = true;    // 后端没好时用假数据；后端好了改成 false
const PLACEHOLDER = true;    // 素材没到时用色块；素材到了改成 false，并删掉 style.css 里的占位段
const DEBUG       = true;    // 左上角显示最近收到的消息，上板前改成 false

const WS_URL     = "ws://127.0.0.1:8000/ws";
const SPRITE_DIR = "../characters/example/sprites/";       // 人物素材：六张情绪立绘 + 三张嘴型
const BG_DIR     = "../characters/example/backgrounds/";   // 背景图

// 与 contracts.py 的 Emotion 保持一致
const EMOTIONS = ["neutral", "happy", "sad", "surprised", "angry", "shy"];
const FADE_MS  = 150;        // 与 style.css 里 #layer-char 的 transition 时长一致

// ======================== 状态 ========================
let currentState   = "IDLE";
let currentEmotion = null;
let currentUser    = null;
let fadeTimer      = null;

const $ = (id) => document.getElementById(id);

// ======================== 消息分发 ========================
function handle(msg) {
  if (!msg || typeof msg !== "object") return;

  switch (msg.type) {
    case "emotion":    setEmotion(msg.value);    break;
    case "speak":      speak(msg);               break;
    case "subtitle":   setSubtitle(msg.text);    break;
    case "background": setBackground(msg.value); break;
    case "user":       currentUser = msg;        break;   // 暂不渲染，先存着
    case "state":      currentState = msg.value; break;   // 暂不处理，后面做待机动效
    default: console.warn("未知消息类型", msg);
  }
  if (DEBUG) showDebug(msg);
}

// ======================== 各图层 ========================
function setEmotion(emotion) {
  if (!EMOTIONS.includes(emotion)) {
    console.warn("非法情绪值，兜底 neutral:", emotion);
    emotion = "neutral";
  }
  if (emotion === currentEmotion) return;    // 同一情绪不重复淡入淡出
  currentEmotion = emotion;

  const el = $("layer-char");
  clearTimeout(fadeTimer);                   // 连续快速切换时，只保留最后一次
  el.style.opacity = 0;                      // 淡出
  fadeTimer = setTimeout(() => {
    el.dataset.emotion = emotion;            // 占位模式靠它换色块颜色
    if (!PLACEHOLDER) el.src = SPRITE_DIR + emotion + ".png";
    el.style.opacity = 1;                    // 淡入
  }, FADE_MS);
}

function setSubtitle(text) {
  $("layer-subtitle").textContent = text || "";
}

function setBackground(file) {
  if (!file) return;
  if (PLACEHOLDER) { console.log("换背景（占位模式不加载图片）:", file); return; }
  $("layer-bg").style.backgroundImage = `url("${BG_DIR}${encodeURIComponent(file)}")`;
}

function speak(msg) {
  // 下一阶段实现：播音频 + 口型同步 + 回发 audio_ended
  setSubtitle(msg.text);
}

// ======================== 预加载 ========================
// 提前把六张立绘下载进缓存，避免第一次切换时闪一下
function preload() {
  if (PLACEHOLDER) return;
  for (const e of EMOTIONS) new Image().src = SPRITE_DIR + e + ".png";
}

// ======================== WebSocket ========================
let ws = null;

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen    = () => { console.log("WS 已连接"); send({type: "ready"}); };
  ws.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); }
    catch { console.warn("收到非 JSON 消息，已忽略:", e.data); return; }
    handle(msg);
  };
  ws.onerror   = () => console.error("WS 错误（后端没起？）");
  ws.onclose   = () => { console.log("WS 断开，2 秒后重连"); setTimeout(connect, 2000); };
}

// ======================== 调试面板 ========================
function showDebug(msg) {
  let el = $("debug");
  if (!el) {
    el = document.createElement("pre");
    el.id = "debug";
    document.body.appendChild(el);
  }
  el.textContent =
    `state:   ${currentState}\n` +
    `emotion: ${currentEmotion}\n` +
    `last:    ${JSON.stringify(msg)}`;
}

// ======================== 启动 ========================
setEmotion("neutral");
preload();
if (USE_MOCK) startMock(handle);
else          connect();
