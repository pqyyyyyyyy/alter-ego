(() => {
  "use strict";

  const char = document.getElementById("layer-char");
  const subtitle = document.getElementById("layer-subtitle");
  const stateEl = document.getElementById("state");
  const connection = document.getElementById("connection");
  const bubble = document.getElementById("speech-bubble");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const send = form.querySelector("button");
  const replay = document.getElementById("replay-audio");
  const logToggle = document.getElementById("log-toggle");
  const logPanel = document.getElementById("dialog-log");
  const logList = document.getElementById("dialog-log-list");
  const logEmpty = document.getElementById("dialog-log-empty");
  const logClear = document.getElementById("log-clear");
  const perceptionStatus = document.getElementById("perception-status");
  const cameraToggle = document.getElementById("camera-toggle");
  const cameraFeedback = document.getElementById("camera-feedback");
  const recordToggle = document.getElementById("record-toggle");
  const recordCancel = document.getElementById("record-cancel");
  const voiceStatus = document.getElementById("voice-status");
  const cameraSwitchState = document.getElementById("camera-switch-state");
  const ttsToggle = document.getElementById("tts-toggle");
  const ttsSwitchState = document.getElementById("tts-switch-state");
  const ttsFast = document.getElementById("tts-fast");
  const ttsSync = document.getElementById("tts-sync");
  const ttsModeHelp = document.getElementById("tts-mode-help");
  const ttsFeedback = document.getElementById("tts-feedback");
  const ttsWarmup = document.getElementById("tts-warmup");
  const SPRITE_BASE = "/character/sprites/";
  let ws = null;
  let reconnectTimer = null;
  let animationTimer = null;
  let spriteRequest = 0;
  let displayedSprite = "";
  let lastSpriteName = "";
  let lastEmotion = "neutral";
  let initialSpriteRequest = null;
  const spriteCache = new Map();
  let currentAudio = null;
  let audioFallbackTimer = null;
  let speechId = 0;

  let voiceSession = null;
  let chatPending = false;
  let outputVoice = {configured: false, enabled: false, mode: "fast", status: "unknown"};
  let outputPending = false;
  let warmupTimer = null;
  let historyAudio = null;
  const audioCache = new Map();
  const replyEntries = new Map();

  let activeSpeech = null;
  let clientId = null;
  let isOwner = false;
  let protocolMode = "unknown";
  let detectedFaces = 0;
  let detectedUser = "";
  let cameraUnavailable = false;
  let cameraUsers = null;
  let cameraEnabled = null;
  let cameraControllable = false;
  let cameraStatus = "loading";
  let cameraPending = false;
  let cameraTarget = null;
  const completed = new Set();

  function normalizeFaceCount(value) {
    const count = Number(value);
    if (!Number.isFinite(count) || count <= 0) return 0;
    return Math.min(2, Math.floor(count));
  }

  function normalizeUser(value) {
    return value === "A" || value === "B" ? value : "";
  }

  function renderPerceptionStatus() {
    if (!perceptionStatus) return;
    const text = cameraEnabled === false ? "摄像头关闭" : `识别到 ${detectedFaces} 人 用户：[${detectedUser || "空"}]` +
      (cameraUnavailable ? " 摄像头未就绪" : "");
    if (perceptionStatus.textContent !== text) perceptionStatus.textContent = text;
  }

  function updatePresence(message) {
    const previousEnabled = cameraEnabled;
    const previouslyUnavailable = cameraUnavailable;
    if (typeof message?.camera_enabled === "boolean") cameraEnabled = message.camera_enabled;
    if (typeof message?.camera_controllable === "boolean") cameraControllable = message.camera_controllable;
    if (typeof message?.camera_status === "string") cameraStatus = message.camera_status;
    cameraUnavailable = message?.available === false;
    if (Array.isArray(message?.visible_users)) {
      cameraUsers = [...new Set(message.visible_users.filter((name) => normalizeUser(name)))].sort();
      detectedUser = cameraUsers.join("、");
    }
    if (message && message.value === false) {
      detectedFaces = 0;
      detectedUser = "";
    } else if (message) {
      detectedFaces = normalizeFaceCount(message.faces ?? message.n_faces);
    }
    if (!cameraPending && typeof message?.camera_enabled === "boolean" &&
        (previousEnabled !== cameraEnabled || previouslyUnavailable !== cameraUnavailable)) {
      delete cameraFeedback.dataset.kind;
      cameraFeedback.textContent = cameraEnabled === false ? "摄像头已关闭，角色保持唤醒" : "";
    }
    renderPerceptionStatus();
    renderCameraControls();
  }

  function updateUser(message) {
    if (cameraUsers === null) detectedUser = normalizeUser(message && (message.name ?? message.user));
    renderPerceptionStatus();
  }

  renderPerceptionStatus();

  function renderCameraControls() {
    cameraToggle.disabled = cameraPending || !cameraControllable || cameraEnabled === null || !ws || ws.readyState !== 1;
    cameraToggle.setAttribute("aria-pressed", String(cameraEnabled === true));
    cameraToggle.setAttribute("aria-checked", String(cameraEnabled === true));
    cameraToggle.setAttribute("aria-busy", String(cameraPending));
    cameraSwitchState.textContent = cameraPending ? "…" : cameraEnabled ? "ON" : "OFF";
    if (!cameraPending && cameraEnabled && cameraUnavailable && cameraControllable && cameraFeedback.dataset.kind !== "error") {
      cameraFeedback.textContent = cameraStatus === "loading" ? "摄像头启动中，请稍候…" : "摄像头未就绪，可关闭后重新开启";
    }
  }

  cameraToggle.addEventListener("click", async () => {
    if (cameraToggle.disabled) return;
    const socket = ws;
    cameraPending = true;
    cameraTarget = !cameraEnabled;
    cameraFeedback.textContent = cameraTarget ? "正在开启摄像头…" : "正在停止采集…";
    delete cameraFeedback.dataset.kind;
    renderCameraControls();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 25000);
    try {
      const response = await fetch("/camera/control", {method: "POST",
        headers: {"Content-Type": "application/json", "X-Alter-Ego-Camera": "1"},
        body: JSON.stringify({enabled: cameraTarget}), signal: controller.signal});
      const result = await response.json();
      if (ws !== socket || socket.readyState !== 1) return;
      if (!response.ok || !result.ok || typeof result.camera?.enabled !== "boolean") throw new Error("camera_control_failed");
      // WebSocket is the ordered source of shared state; the POST confirms the operation.
      cameraFeedback.textContent = result.camera.enabled ? "已开启，请先让 A、再让 B 正脸入镜登记" : "摄像头已关闭，角色保持唤醒";
    } catch (error) {
      if (ws !== socket || socket.readyState !== 1) return;
      cameraFeedback.textContent = "切换失败或超时，请以检测状态为准后重试";
      cameraFeedback.dataset.kind = "error";
    } finally {
      clearTimeout(timeout);
      cameraPending = false;
      renderCameraControls();
    }
  });
  renderCameraControls();

  function outputWarming() { return outputVoice.status === "warming" || outputPending; }

  function stopHistoryAudio() {
    if (historyAudio) { historyAudio.pause(); historyAudio.onended = historyAudio.onerror = null; }
    historyAudio = null;
  }

  function renderOutputControls() {
    const connected = !!ws && ws.readyState === 1 && !!clientId;
    ttsToggle.disabled = !connected || !outputVoice.configured || outputPending ||
      outputVoice.status === "warming" || (chatPending || activeSpeech) && !outputVoice.enabled || !!voiceSession;
    ttsToggle.setAttribute("aria-checked", String(outputVoice.enabled));
    ttsToggle.setAttribute("aria-busy", String(outputWarming()));
    ttsSwitchState.textContent = outputWarming() ? "…" : outputVoice.enabled ? "ON" : "OFF";
    for (const [button, mode] of [[ttsFast, "fast"], [ttsSync, "sync"]]) {
      button.setAttribute("aria-pressed", String(outputVoice.mode === mode));
      button.disabled = !connected || outputWarming() || chatPending || !!activeSpeech || !!voiceSession;
    }
    ttsModeHelp.textContent = outputVoice.mode === "fast" ? "先显示文字，语音生成后自动播放" : "文字与语音生成完毕后一起呈现";
    if (!outputVoice.configured && outputVoice.status !== "unknown") ttsFeedback.textContent = "尚未接入语音模块，仍可打字和录音";
    updateReplayControls();
  }

  function updateReplayControls() {
    for (const [key, entry] of replyEntries) {
      const cached = audioCache.has(key);
      entry.body.disabled = !cached || !outputVoice.enabled || outputWarming() || !!voiceSession ||
        chatPending && activeSpeech?.speech_id !== key || !!activeSpeech && activeSpeech.speech_id !== key;
      entry.body.setAttribute("aria-label", cached ? `重播七海语音：${entry.text}` : entry.text);
    }
  }

  function updateOutputVoice(message) {
    const wasWarming = outputVoice.status === "warming";
    outputVoice = {configured: message.configured === true, enabled: message.enabled === true,
      mode: message.mode === "sync" ? "sync" : "fast", status: message.status || "off"};
    clearTimeout(warmupTimer);
    if (outputVoice.status === "warming") {
      cancelRecording("语音模块预热中，录音已终止");
      ttsWarmup.hidden = false; ttsWarmup.textContent = "正在预热语音模块";
      delete ttsWarmup.dataset.kind;
    } else if (wasWarming && outputVoice.status === "ready") {
      ttsWarmup.hidden = false; ttsWarmup.textContent = "预热成功";
      delete ttsWarmup.dataset.kind;
      warmupTimer = setTimeout(() => { ttsWarmup.hidden = true; }, 3000);
      ttsFeedback.textContent = "";
    } else if (outputVoice.status === "failed") {
      ttsWarmup.hidden = false; ttsWarmup.textContent = "预热失败，已关闭语音；仍可文字对话";
      ttsWarmup.dataset.kind = "error";
      ttsFeedback.textContent = "请检查板端语音服务状态后重新开启";
    } else { ttsWarmup.hidden = true; }
    if (!outputVoice.enabled) {
      stopHistoryAudio();
      if (activeSpeech && currentAudio) finishSpeech(speechId, "cancelled", "voice_off");
    }
    voiceControls(voiceSession);
    renderOutputControls();
  }

  async function changeOutput(enabled, mode) {
    if (outputPending) return;
    const socket = ws;
    if (!socket || socket.readyState !== 1) return;
    outputPending = true; voiceControls(voiceSession); renderOutputControls();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch("/voice/control", {method: "POST",
        headers: {"Content-Type": "application/json", "X-Alter-Ego-Voice": "1"},
        body: JSON.stringify({enabled, mode}), signal: controller.signal});
      const result = await response.json();
      if (ws !== socket || socket.readyState !== 1) return;
      if (!response.ok || !result.ok) throw new Error(result.error || "voice_control_failed");
      // WS is authoritative. The POST snapshot can be older than a completed warmup.
      ttsFeedback.textContent = "";
    } catch (error) {
      if (ws !== socket || socket.readyState !== 1) return;
      ttsFeedback.textContent = error.message === "busy" ? "请等当前对话或录音结束后再切换" : "语音切换失败，请以开关状态为准";
    } finally {
      clearTimeout(timeout); outputPending = false; voiceControls(voiceSession); renderOutputControls();
    }
  }

  ttsToggle.addEventListener("click", () => {
    if (!ttsToggle.disabled) return changeOutput(!outputVoice.enabled, outputVoice.mode);
  });
  ttsFast.addEventListener("click", () => { if (!ttsFast.disabled) return changeOutput(outputVoice.enabled, "fast"); });
  ttsSync.addEventListener("click", () => { if (!ttsSync.disabled) return changeOutput(outputVoice.enabled, "sync"); });

  function setVoiceStatus(text, kind = "") {
    if (!voiceStatus) return;
    voiceStatus.textContent = text;
    if (kind) voiceStatus.dataset.kind = kind;
    else delete voiceStatus.dataset.kind;
  }

  function voiceControls(session) {
    const phase = session?.phase || "idle";
    recordToggle.textContent = {idle: "开始录音", starting: "等待麦克风…",
      recording: "停止并发送", finishing: "识别最后一段…", ready: "发送识别文字"}[phase];
    recordToggle.disabled = outputWarming() || chatPending || !!activeSpeech || ["starting", "finishing"].includes(phase);
    recordToggle.setAttribute("aria-pressed", String(phase === "recording"));
    recordCancel.hidden = !session;
    input.readOnly = outputWarming() || !!session && phase !== "ready";
    send.disabled = outputWarming() || chatPending || !!activeSpeech || !!session && phase !== "ready";
    renderOutputControls();
  }

  function closeMicrophone(session) {
    session.capturing = false;
    if (session.processor) { session.processor.onaudioprocess = null; session.processor.disconnect(); }
    if (session.source) session.source.disconnect();
    if (session.stream) session.stream.getTracks().forEach((track) => track.stop());
    if (session.context && session.context.state !== "closed") session.context.close().catch(() => {});
    clearTimeout(session.limitTimer);
  }

  function cancelRecording(message = "已终止，本次录音未发送") {
    const session = voiceSession;
    if (!session) return;
    voiceSession = null; // Late permissions/ASR responses cannot alter a newer recording.
    closeMicrophone(session);
    if (session.request) session.request.abort();
    input.value = session.original;
    session.chunks = []; session.pre = [];
    voiceControls(null);
    setVoiceStatus(message);
  }

  function pcmBytes(chunks, rate) {
    const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const samples = new Float32Array(length);
    let offset = 0;
    for (const chunk of chunks) { samples.set(chunk, offset); offset += chunk.length; }
    const count = Math.floor(length * 16000 / rate);
    const buffer = new ArrayBuffer(count * 2);
    const view = new DataView(buffer);
    // Average source samples when downsampling; explicitly encode little-endian.
    for (let i = 0; i < count; i++) {
      const start = Math.floor(i * rate / 16000);
      const end = Math.min(length, Math.max(start + 1, Math.floor((i + 1) * rate / 16000)));
      let sum = 0;
      for (let j = start; j < end; j++) sum += samples[j];
      const sample = Math.max(-1, Math.min(1, sum / (end - start)));
      view.setInt16(i * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
    }
    return buffer;
  }

  const voiceErrors = {asr_not_enabled: "板端语音识别未启用，仍可打字发送",
    busy: "后端正忙，请等当前回复结束后重试", text_length: "识别文字过长，请缩短后发送"};

  function queueSentence(session) {
    if (!session.chunks.length || session.voiced < .10) {
      session.chunks = []; session.voiced = 0; session.silence = 0; return;
    }
    const chunks = session.chunks;
    session.chunks = []; session.voiced = 0; session.silence = 0; session.pre = [];
    // One ASR at a time; a pause can start a preview while capture continues.
    session.pending = session.pending.then(async () => {
      if (voiceSession !== session || session.failed) return;
      const controller = new AbortController(); session.request = controller;
      const timeout = setTimeout(() => controller.abort(), 120000);
      setVoiceStatus("正在识别，文字会自动显示…", "working");
      try {
        const response = await fetch("/debug/transcribe", {method: "POST",
          headers: {"Content-Type": "application/octet-stream"},
          body: pcmBytes(chunks, session.rate), signal: controller.signal});
        const result = await response.json();
        if (voiceSession !== session) return;
        if (!response.ok || !result.ok) throw new Error(result.error || result.status || "asr_failed");
        const text = typeof result.text === "string" ? result.text.trim() : "";
        if (text) { session.text += text; input.value = session.text; }
        setVoiceStatus(text ? "文字已显示；可继续说，或按停止并发送" : "未识别到文字，可继续说话", text ? "success" : "");
      } catch (error) {
        if (voiceSession !== session) return;
        session.failed = true;
        closeMicrophone(session); session.phase = "ready";
        voiceControls(session);
        setVoiceStatus(voiceErrors[error.message] || "识别失败或超时，文字已保留，可修改后发送", "error");
      } finally { clearTimeout(timeout); if (session.request === controller) session.request = null; }
    });
  }

  function captureSamples(session, samples) {
    if (voiceSession !== session || !session.capturing) return;
    const frame = new Float32Array(samples);
    const seconds = frame.length / session.rate;
    let energy = 0;
    for (const sample of frame) energy += sample * sample;
    const voiced = Math.sqrt(energy / frame.length) >= 150 / 32768;
    if (!session.chunks.length) {
      session.pre.push(frame);
      while (session.pre.length > Math.ceil(.2 / seconds)) session.pre.shift();
      if (voiced) { session.chunks = session.pre; session.pre = []; session.voiced = seconds; }
    } else {
      session.chunks.push(frame);
      if (voiced) { session.voiced += seconds; session.silence = 0; }
      else session.silence += seconds;
      if (session.silence >= .8) queueSentence(session);
    }
  }

  async function startRecording() {
    if (voiceSession || chatPending || outputWarming() || activeSpeech) return;
    stopHistoryAudio();
    if (!ws || ws.readyState !== 1 || !clientId) {
      setVoiceStatus("请先等待网页连接后端", "error"); return;
    }
    const media = globalThis.navigator && globalThis.navigator.mediaDevices;
    const AudioContextCtor = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!media?.getUserMedia || !AudioContextCtor) {
      setVoiceStatus("录音需使用 localhost 页面并允许麦克风，请勿用板卡 IP 的 HTTP 页面", "error");
      return;
    }
    const session = {phase: "starting", original: input.value, text: "", pending: Promise.resolve(),
      chunks: [], pre: [], voiced: 0, silence: 0, failed: false, capturing: false};
    voiceSession = session; voiceControls(session);
    setVoiceStatus("请允许浏览器访问麦克风", "working");
    try {
      session.stream = await media.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}});
      if (voiceSession !== session) { closeMicrophone(session); return; }
      session.context = new AudioContextCtor();
      await session.context.resume();
      if (voiceSession !== session) { closeMicrophone(session); return; }
      session.rate = session.context.sampleRate;
      session.source = session.context.createMediaStreamSource(session.stream);
      // Firefox 122 on the board supports this local Web Audio capture path.
      // The output buffer stays silent; the microphone is never monitored.
      session.processor = session.context.createScriptProcessor(2048, 1, 1);
      session.processor.onaudioprocess = (event) => captureSamples(session, event.inputBuffer.getChannelData(0));
      session.source.connect(session.processor); session.processor.connect(session.context.destination);
      session.capturing = true; session.phase = "recording";
      input.value = ""; voiceControls(session);
      session.limitTimer = setTimeout(() => finishRecording(session, false), 28000);
      setVoiceStatus("正在录音；停顿后显示文字，再按一次停止并发送（最长28秒）", "working");
    } catch (error) {
      if (voiceSession !== session) return;
      cancelRecording();
      setVoiceStatus(error.name === "NotFoundError" ? "未找到麦克风，请检查连接" : "无法访问麦克风，请检查权限或设备占用", "error");
    }
  }

  async function finishRecording(session, commit = true) {
    if (voiceSession !== session || session.phase === "finishing") return;
    const wasReady = session.phase === "ready";
    closeMicrophone(session);
    session.phase = "finishing"; voiceControls(session);
    queueSentence(session);
    await session.pending;
    if (voiceSession !== session) return;
    session.phase = "ready"; voiceControls(session);
    if (session.failed && !wasReady) return;
    if (!input.value.trim()) {
      setVoiceStatus("没有识别到语音，可终止后重试", "error"); return;
    }
    if (!commit) { setVoiceStatus("已到录音时限，文字已保留，请点击发送", "success"); return; }
    voiceSession = null; voiceControls(null);
    setVoiceStatus("文字已显示，正在发送…", "working");
    const ok = await sendChat(input.value.trim(), true);
    setVoiceStatus(ok ? "已发送，录音结束" : "发送失败，文字已保留，可重试", ok ? "success" : "error");
  }

  recordToggle.addEventListener("click", () => voiceSession ? finishRecording(voiceSession) : startRecording());
  recordCancel.addEventListener("click", () => cancelRecording());

  const emotionLabels = {
    neutral: "平静", happy: "开心", surprised: "惊讶", shy: "害羞",
    unhappy: "低落", assertive: "认真",
  };

  function setLogVisibility(open) {
    logPanel.hidden = !open;
    logToggle.setAttribute("aria-expanded", String(open));
    if (open) logList.scrollTop = logList.scrollHeight;
  }

  function appendLog(role, text, emotion = "", sprite = "", key = "") {
    if (typeof text !== "string" || !text.trim()) return;
    const entry = document.createElement("article");
    entry.className = `dialog-entry ${role}`;
    const avatar = role === "character" && sprite ? document.createElement("img") : document.createElement("div");
    if (role === "character" && sprite) {
      avatar.className = "dialog-entry-avatar";
      avatar.src = sprite;
      avatar.alt = `角色表情：${emotionLabels[emotion] || emotion || "当前"}`;
      avatar.loading = "lazy";
    } else {
      avatar.className = "dialog-entry-avatar-placeholder";
      avatar.textContent = role === "user" ? "你" : "角色";
      avatar.setAttribute("aria-hidden", "true");
    }
    const content = document.createElement("div");
    content.className = "dialog-entry-content";
    const meta = document.createElement("div");
    meta.className = "dialog-entry-meta";
    meta.textContent = role === "user" ? "你" : "七海千秋";
    if (role === "character" && emotion) {
      const emotionEl = document.createElement("span");
      emotionEl.className = "dialog-entry-emotion";
      emotionEl.textContent = `· ${emotionLabels[emotion] || emotion}`;
      meta.appendChild(emotionEl);
    }
    const body = document.createElement(role === "character" ? "button" : "div");
    body.className = "dialog-entry-text";
    body.textContent = text;
    const hint = document.createElement("div");
    hint.className = "dialog-audio-hint";
    if (role === "character") {
      body.type = "button"; body.disabled = true;
      body.addEventListener("click", () => replayHistory(key));
    }
    content.append(meta, body);
    if (role === "character") content.appendChild(hint);
    entry.append(avatar, content);
    logList.appendChild(entry);
    logEmpty.hidden = true;
    if (!logPanel.hidden) logList.scrollTop = logList.scrollHeight;
    return {body, hint, text};
  }

  function showReply(msg, preview = false) {
    subtitle.textContent = msg.text || "";
    const key = msg.speech_id;
    let entry = key && replyEntries.get(key);
    if (!entry) {
      entry = appendLog("character", msg.text || msg.text_ja || "", lastEmotion,
        lastSpriteName ? spriteCandidates(lastSpriteName)[0] : displayedSprite, key);
      if (entry && key) replyEntries.set(key, entry);
    }
    if (!entry) return;
    if (preview) { entry.hint.textContent = "语音生成中…"; return; }
    if (key && typeof msg.audio === "string" && msg.audio) {
      audioCache.set(key, msg.audio);
      entry.hint.textContent = "点击回复重听";
      while (audioCache.size > 10) {
        const oldest = audioCache.keys().next().value;
        audioCache.delete(oldest);
        const oldEntry = replyEntries.get(oldest);
        if (oldEntry) oldEntry.hint.textContent = "语音已移出最近十条";
      }
    } else { entry.hint.textContent = msg.tts_error ? "语音生成失败，文字已保留" : ""; }
    updateReplayControls();
  }

  function replayHistory(key) {
    const entry = replyEntries.get(key);
    if (!entry || entry.body.disabled || !audioCache.has(key)) return;
    if (activeSpeech?.speech_id === key) { playSpeechAudio(); return; }
    stopHistoryAudio();
    const audio = new Audio(audioCache.get(key));
    historyAudio = audio;
    // History replay is local: never ACK a completed or unrelated backend turn.
    audio.onended = () => { if (historyAudio === audio) { historyAudio = null; entry.hint.textContent = "点击回复重听"; } };
    audio.onerror = () => { if (historyAudio === audio) { historyAudio = null; entry.hint.textContent = "重播失败，请再试一次"; } };
    entry.hint.textContent = "正在重播…";
    audio.play().catch(() => { if (historyAudio === audio) { historyAudio = null; entry.hint.textContent = "播放受阻，请再点击一次"; } });
  }

  function initialSpriteFor(state, data) {
    const animations = data && data.animations;
    const emotions = data && data.emotions;
    if (state === "IDLE" && animations && Array.isArray(animations.idle?.frames) && animations.idle.frames[0]) {
      return animations.idle.frames[0];
    }
    const neutral = emotions && emotions.neutral;
    return Array.isArray(neutral) ? neutral[0] : (typeof neutral === "string" ? neutral : "");
  }

  async function loadInitialSprite(state) {
    if (initialSpriteRequest || displayedSprite) return;
    initialSpriteRequest = fetch("/debug/sprites", {headers: {Accept: "application/json"}})
      .then((response) => response.ok ? response.json() : null)
      .then((data) => {
        const name = initialSpriteFor(state, data);
        // If a live emotion arrived while the sprite manifest was loading, do
        // not let the late initial request overwrite that newer expression.
        if (name && !displayedSprite && !lastSpriteName) { lastSpriteName = name; return setSprite(name); }
        return null;
      })
      .catch(() => null);
  }

  function sendMessage(message) {
    if (!ws || ws.readyState !== 1) return false;
    try { ws.send(JSON.stringify(message)); return true; } catch (err) { return false; }
  }

  function clearSpeech() {
    clearTimeout(audioFallbackTimer); audioFallbackTimer = null;
    if (currentAudio) { currentAudio.pause(); currentAudio.onended = currentAudio.onerror = currentAudio.onplaying = currentAudio.onloadedmetadata = null; }
    currentAudio = null; activeSpeech = null; ++speechId; replay.hidden = true;
    voiceControls(voiceSession);
  }

  function finishSpeech(id, result = "ended", reason = "onended") {
    if (id !== speechId || !activeSpeech) return;
    const msg = activeSpeech;
    if (!msg.speech_id || !completed.has(msg.speech_id)) {
      const report = {type: "audio_ended"};
      if (msg.speech_id) Object.assign(report, {turn_id: msg.turn_id, speech_id: msg.speech_id, client_id: clientId, result, reason});
      sendMessage(report);
      if (msg.speech_id) { completed.add(msg.speech_id); if (completed.size > 256) completed.delete(completed.values().next().value); }
    }
    clearSpeech();
    if (bubble.textContent === "点击播放语音") bubble.textContent = "";
  }

  function playSpeechAudio() {
    if (!currentAudio || !activeSpeech || !isOwner) return;
    clearTimeout(audioFallbackTimer);
    const audio = currentAudio; const id = speechId;
    audioFallbackTimer = setTimeout(() => finishSpeech(id, "timeout", "load_or_play_timeout"), 600000);
    audio.play().then(() => {
      if (id !== speechId || audio !== currentAudio) return;
      replay.hidden = true;
      if (bubble.textContent === "点击播放语音") bubble.textContent = "";
    }).catch((err) => {
      if (id !== speechId || audio !== currentAudio) return;
      clearTimeout(audioFallbackTimer);
      replay.hidden = false; bubble.textContent = "点击播放语音";
      audioFallbackTimer = setTimeout(() => finishSpeech(id, "timeout", "autoplay_wait"), 30000);
    });
  }

  replay.addEventListener("click", playSpeechAudio);

  function spriteUrl(name) {
    return SPRITE_BASE + encodeURIComponent(name).replace(/%2F/g, "/");
  }

  function spriteCandidates(name) {
    const original = spriteUrl(name);
    if (!/\.png$/i.test(name)) return [original];
    return [original.replace(/\.png$/i, ".webp"), original];
  }

  function loadSprite(url) {
    if (!spriteCache.has(url)) {
      const pending = new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = async () => {
          try {
            if (img.decode) await img.decode();
            resolve(img);
          } catch (err) { reject(err); }
        };
        img.onerror = () => reject(new Error(`立绘加载失败: ${url}`));
        img.src = url;
      });
      spriteCache.set(url, pending);
      // 失败的资源允许在下一次指令到来时重试。
      pending.catch(() => spriteCache.delete(url));
    }
    return spriteCache.get(url);
  }

  async function setSprite(name) {
    if (!name) return;
    lastSpriteName = name;
    const request = ++spriteRequest;
    const candidates = spriteCandidates(name);
    if (candidates.includes(displayedSprite)) return;
    let lastError = null;
    for (const url of candidates) {
      try {
        await loadSprite(url);
        // 旧图保留到新图解码完成；迟到的旧请求不能覆盖最新表情。
        if (request !== spriteRequest) return;
        char.src = url;
        displayedSprite = url;
        char.classList.add("ready");
        if (bubble.textContent === "立绘加载失败") bubble.textContent = "";
        return;
      } catch (err) {
        lastError = err;
      }
    }
    if (request === spriteRequest) bubble.textContent = "立绘加载失败";
    console.error(lastError);
  }

  function stopAnimation() {
    if (animationTimer !== null) clearInterval(animationTimer);
    animationTimer = null;
    ++spriteRequest;
  }

  function playAnimation(msg) {
    stopAnimation();
    const frames = Array.isArray(msg.frames) ? msg.frames : [];
    if (!frames.length) return;
    let i = 0;
    setSprite(frames[i]);
    animationTimer = setInterval(() => {
      i += 1;
      if (i >= frames.length) {
        if (!msg.loop) { clearInterval(animationTimer); animationTimer = null; return; }
        i = 0;
      }
      setSprite(frames[i]);
    }, Number(msg.interval_ms) || 1000);
  }

  function handle(msg) {
    if (!msg || typeof msg !== "object" || Array.isArray(msg)) return;
    switch (msg.type) {
      case "state":
        stopAnimation();
        stateEl.textContent = msg.value || "—";
        loadInitialSprite(msg.value || "IDLE");
        break;
      case "emotion":
        stopAnimation();
        lastEmotion = typeof msg.value === "string" ? msg.value : "neutral";
        lastSpriteName = msg.sprite || lastSpriteName;
        setSprite(msg.sprite);
        break;
      case "animation": playAnimation(msg); break;
      case "hello":
        protocolMode = "v1";
        clientId = msg.client_id; isOwner = msg.playback_owner === true;
        loadInitialSprite("IDLE");
        if (!isOwner) clearSpeech();
        break;
      case "sync":
        clearSpeech(); stopAnimation(); stateEl.textContent = msg.state || "—";
        if (typeof msg.playback_owner === "boolean") isOwner = msg.playback_owner;
        if (msg.sprite) setSprite(msg.sprite);
        else loadInitialSprite(msg.state || "IDLE");
        break;
      case "stop":
        if (isOwner && activeSpeech && msg.speech_id === activeSpeech.speech_id &&
            msg.turn_id === activeSpeech.turn_id &&
            (!msg.playback_client_id || msg.playback_client_id === clientId)) {
          if (currentAudio) currentAudio.pause();
          finishSpeech(speechId, "cancelled", "server_stop");
        }
        break;
      case "voice_output": updateOutputVoice(msg); break;
      case "reply_preview":
        if (msg.speech_id && !completed.has(msg.speech_id) && typeof msg.text === "string") showReply(msg, true);
        break;
      case "speak": {
        if (protocolMode === "unknown") {
          protocolMode = msg.speech_id || msg.turn_id || msg.playback_client_id ? "v1" : "legacy";
          isOwner = protocolMode === "legacy";
        }
        if (msg.speech_id && (completed.has(msg.speech_id) || activeSpeech?.speech_id === msg.speech_id)) return;
        if (typeof msg.text !== "string" && msg.text != null) return;
        showReply(msg);
        stopHistoryAudio();
        if (msg.tts_error) ttsFeedback.textContent = "本次语音生成失败，文字已保留";
        if (activeSpeech?.speech_id) finishSpeech(speechId, "cancelled", "replaced"); else clearSpeech();
        if (!isOwner || (msg.playback_client_id && msg.playback_client_id !== clientId)) return;
        activeSpeech = msg; const id = ++speechId;
        voiceControls(voiceSession);
        if (outputVoice.status !== "unknown" && !outputVoice.enabled && msg.audio) {
          finishSpeech(id, "cancelled", "voice_off");
          break;
        }
        if (typeof msg.audio === "string" && msg.audio) {
          currentAudio = new Audio(msg.audio);
          currentAudio.onended = () => finishSpeech(id);
          let startedReported = false;
          currentAudio.onplaying = () => {
            if (id !== speechId || !activeSpeech || startedReported) return;
            startedReported = true;
            if (msg.speech_id) sendMessage({type: "audio_started", turn_id: msg.turn_id, speech_id: msg.speech_id, client_id: clientId});
          };
          currentAudio.onerror = () => { if (id !== speechId) return; bubble.textContent = "语音播放失败"; finishSpeech(id, "failed", "audio_error"); };
          currentAudio.onloadedmetadata = () => {
            if (id !== speechId || !currentAudio) return;
            const seconds = currentAudio.duration;
            if (Number.isFinite(seconds) && seconds > 0) {
              clearTimeout(audioFallbackTimer);
              audioFallbackTimer = setTimeout(() => finishSpeech(id, "timeout", "duration_overrun"), (seconds + 30) * 1000);
            }
          };
          playSpeechAudio();
        } else if (msg.audio != null && typeof msg.audio !== "string") {
          finishSpeech(id, "failed", "invalid_audio");
        } else {
          const duration = Number(msg.duration_ms);
          audioFallbackTimer = setTimeout(() => finishSpeech(id, "no_audio", "subtitle_only"), Number.isFinite(duration) && duration > 0 ? Math.min(duration, 30000) : 500);
        }
        break;
      }
      case "presence": updatePresence(msg); break;
      case "user": updateUser(msg); bubble.textContent = msg.name ? `你好，${msg.name}` : "检测到你了"; break;
      case "background": document.body.dataset.background = msg.value || ""; break;
      default: break;
    }
  }

  function connect() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/ws`);
    const socket = ws;
    ws.onopen = () => {
      if (ws !== socket) return;
      connection.textContent = "已连接"; connection.className = "online";
      renderCameraControls();
      renderOutputControls();
      sendMessage({type: "ready"});
    };
    ws.onmessage = (event) => {
      if (ws !== socket) return;
      try { handle(JSON.parse(event.data)); } catch (err) { console.error(err); }
    };
    ws.onclose = () => {
      if (ws !== socket) return;
      cameraControllable = false;
      if (cameraUsers !== null) updatePresence({value: false, faces: 0, visible_users: [], available: false});
      renderCameraControls();
      cameraFeedback.textContent = "连接中断，摄像头状态待重新确认";
      stopHistoryAudio(); clearTimeout(warmupTimer); ttsWarmup.hidden = true;
      outputVoice = {configured: false, enabled: false, mode: outputVoice.mode, status: "unknown"};
      cancelRecording("连接中断，本次录音已终止");
      stopAnimation();
      clearSpeech(); clientId = null; isOwner = false; protocolMode = "unknown";
      renderOutputControls();
      connection.textContent = "未连接，重试中…"; connection.className = "offline";
      clearTimeout(reconnectTimer); reconnectTimer = setTimeout(connect, 2000);
    };
  }

  async function sendChat(text, fromVoice = false) {
    if (!text || chatPending || outputWarming() || activeSpeech) return false;
    stopHistoryAudio();
    appendLog("user", text);
    chatPending = true; voiceControls(voiceSession);
    try {
      const body = new URLSearchParams({text});
      const response = await fetch(`/debug/chat?${body}` , {method: "POST"});
      const result = await response.json();
      if (!response.ok || !result.ok) {
        bubble.textContent = voiceErrors[result.error] || result.error || "对话失败";
        const entry = replyEntries.get(result.speech_id);
        if (entry && !audioCache.has(result.speech_id)) entry.hint.textContent = "语音未完成，文字已保留";
        return false;
      }
      if (!fromVoice && input.value.trim() === text) input.value = "";
      return true;
    } catch (err) {
      bubble.textContent = "后端未启动";
      console.error(err);
      return false;
    } finally { chatPending = false; voiceControls(voiceSession); }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (voiceSession) {
      if (voiceSession.phase === "ready") await finishRecording(voiceSession);
      return;
    }
    await sendChat(input.value.trim());
  });

  globalThis.addEventListener?.("pagehide", () => { cancelRecording(); stopHistoryAudio(); clearSpeech(); });

  logToggle.addEventListener("click", () => setLogVisibility(logPanel.hidden));
  logClear.addEventListener("click", () => {
    logList.replaceChildren();
    audioCache.clear(); replyEntries.clear(); stopHistoryAudio();
    logEmpty.hidden = false;
  });

  voiceControls(null);
  connect();
})();
