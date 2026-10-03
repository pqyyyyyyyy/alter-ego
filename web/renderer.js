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

// ======================== 说话 + 口型同步 ========================
const MOUTH = {
  close: SPRITE_DIR + "mouth_close.png",
  half:  SPRITE_DIR + "mouth_half.png",
  open:  SPRITE_DIR + "mouth_open.png",
};

let currentAudio  = null;    // 正在播放的音频；新 speak 到来时用它打断旧的
let currentFinish = null;    // 当前这句的收尾函数，打断时调用它通知后端
let currentMouth  = null;

// 开口度 0~1 → 三档嘴型
function mouthLevel(v) {
  return v < 0.33 ? "close" : v < 0.66 ? "half" : "open";
}

function setMouth(level) {
  if (level === currentMouth) return;       // 没变就不动 DOM，省性能
  currentMouth = level;
  const el = $("layer-mouth");
  el.dataset.mouth = level;                 // 占位模式靠它改色块高度
  if (!PLACEHOLDER) el.src = MOUTH[level];
}

// 后端发的是 "/audio/xxx.wav" 这种相对后端的路径，要拼上后端地址；
// 完整 URL（http:、blob:）保持原样
function resolveAudioUrl(path) {
  return new URL(path, WS_URL.replace(/^ws/, "http")).href;
}

function stopSpeaking() {
  if (!currentAudio) return;
  const old = currentAudio;
  currentFinish("interrupted");             // 被打断也通知后端，并置空 currentAudio，旧回调不再生效
  old.pause();
  old.removeAttribute("src");
  old.load();                               // 释放音频资源
}

function speak(msg) {
  stopSpeaking();                           // 说话中又来新的一句：打断旧的
  setSubtitle(msg.text);
  if (!msg.audio) return;

  const audio  = new Audio(resolveAudioUrl(msg.audio));
  const frames = Array.isArray(msg.mouth) ? msg.mouth : [];
  currentAudio = audio;

  let finished = false;
  let watchdog = null;
  // 无论正常播完、被打断还是出错，都只通知后端一次，否则后端会一直卡在 SPEAKING。
  // 被打断时额外带 interrupted: true，后端不关心可以忽略这个字段
  function finish(reason) {
    if (finished || audio !== currentAudio) return;
    finished = true;
    currentAudio = currentFinish = null;
    clearTimeout(watchdog);
    setMouth("close");
    if (reason === "interrupted") {
      send({type: "audio_ended", interrupted: true});
    } else {
      send({type: "audio_ended"});
      if (reason !== "ended") console.warn("音频非正常结束:", reason);
    }
  }
  currentFinish = finish;

  // 用音频真实播放位置 currentTime 取开口度，而不是 setTimeout，才不会越说越错位
  function tick() {
    if (audio !== currentAudio || audio.paused || audio.ended) return;
    const t = audio.currentTime * 1000;     // 秒 → 毫秒
    let v = 0;
    for (const [ms, val] of frames) {
      if (ms <= t) v = val; else break;
    }
    setMouth(mouthLevel(v));
    requestAnimationFrame(tick);
  }

  audio.onplay  = () => requestAnimationFrame(tick);
  audio.onended = () => finish("ended");
  audio.onerror = () => finish("音频加载失败 " + audio.src);

  // 保险：超过预计时长 3 秒还没结束（卡住、网络断），强制收尾
  const limit = (msg.duration_ms || 30000) + 3000;
  watchdog = setTimeout(() => finish("超时"), limit);

  play(audio);
}

function play(audio) {
  audio.play().catch((err) => {
    if (err.name === "NotAllowedError") {
      // 浏览器自动播放限制：开发时点一下页面即可；板子 kiosk 启动参数会关掉这个限制
      console.warn("浏览器禁止自动播放，点击页面任意处后播放");
      showTip("点击屏幕以启用声音");
      document.addEventListener("click", () => {
        hideTip();
        if (audio === currentAudio) play(audio);
      }, {once: true});
    } else {
      console.error("播放失败", err);       // 其他错误交给 onerror / 看门狗收尾
    }
  });
}

function showTip(text) {
  let el = $("tip");
  if (!el) {
    el = document.createElement("div");
    el.id = "tip";
    document.body.appendChild(el);
  }
  el.textContent = text;
  el.hidden = false;
}

function hideTip() {
  const el = $("tip");
  if (el) el.hidden = true;
}

// ======================== 预加载 ========================
// 提前把立绘和嘴型下载进缓存，避免第一次切换时闪一下
function preload() {
  if (PLACEHOLDER) return;
  for (const e of EMOTIONS) new Image().src = SPRITE_DIR + e + ".png";
  for (const src of Object.values(MOUTH)) new Image().src = src;
}

// ======================== WebSocket ========================
let ws = null;

const sentLog = [];          // mock 模式下记录"本该发给后端"的消息，方便自测

function send(obj) {
  if (USE_MOCK) { sentLog.push(obj); console.log("→ 发给后端:", obj); return; }
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
    `last:    ${JSON.stringify(msg).slice(0, 120)}`;
}

// ======================== 启动 ========================
setEmotion("neutral");
setMouth("close");
preload();
if (USE_MOCK) { send({type: "ready"}); startMock(handle); }
else          connect();
