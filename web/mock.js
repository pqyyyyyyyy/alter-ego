// 假数据发生器 —— 自测用，模拟后端推消息。
// 消息格式与后端约定完全一致，见任务书「后端会发给你的消息」。
//
// 测试音频在浏览器里现场合成（一个字一声"嘟"），不需要任何音频文件，
// 也就不会误提交 .wav 进仓库。

// 合成一段"说话"音频，返回 speak 消息所需的 audio / duration_ms / mouth
function makeTestSpeech(text) {
  const RATE = 16000;
  const SYL_MS = 220;                          // 每个字 220ms：160ms 发声 + 60ms 停顿
  const VOICE_MS = 160;
  const STEP_MS = 40;                          // mouth 时间轴每 40ms 一个点
  const chars = [...text.replace(/[，。！？…、\s]/g, "")];
  const durationMs = chars.length * SYL_MS;
  const n = Math.round(RATE * durationMs / 1000);
  const pcm = new Int16Array(n);

  // 每个字的音量包络：先张后合（0 → 1 → 0），模拟开口闭口
  const envelope = (ms) => {
    const p = ms % SYL_MS;
    return p < VOICE_MS ? Math.sin(Math.PI * p / VOICE_MS) : 0;
  };
  for (let i = 0; i < n; i++) {
    const ms = i * 1000 / RATE;
    const pitch = 220 + 40 * Math.floor(ms / SYL_MS) % 120;   // 每个字音高略不同
    pcm[i] = envelope(ms) * 0.3 * 32767 * Math.sin(2 * Math.PI * pitch * i / RATE);
  }

  const mouth = [];
  for (let ms = 0; ms < durationMs; ms += STEP_MS) {
    mouth.push([ms, Math.round(envelope(ms) * 100) / 100]);
  }
  mouth.push([durationMs, 0]);

  return {audio: URL.createObjectURL(wavBlob(pcm, RATE)), duration_ms: durationMs, mouth};
}

// 16bit 单声道 PCM → WAV 文件
function wavBlob(pcm, rate) {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const str = (off, s) => [...s].forEach((c, i) => v.setUint8(off + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, pcm.length * 2, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], {type: "audio/wav"});
}

function speakMsg(text) {
  return {type: "speak", text, ...makeTestSpeech(text)};
}

function startMock(handler) {
  // [等待毫秒, 消息]：等待时间是"发完这条之后隔多久发下一条"
  const s1 = speakMsg("欸，你来啦！今天天气不错呢。");
  const s2 = speakMsg("唔，有点不好意思呢。");
  const s3 = speakMsg("等一下，我还没说完——");
  const s4 = speakMsg("好吧，你先说。");

  const seq = [
    [1000, {type: "state",      value: "NOTICED"}],
    [1000, {type: "user",       name: "小明", known: true}],
    [500,  {type: "emotion",    value: "happy"}],
    [500,  {type: "state",      value: "SPEAKING"}],
    [s1.duration_ms + 800, s1],
    [500,  {type: "emotion",    value: "shy"}],
    [s2.duration_ms + 800, s2],
    // 打断测试：s3 说到一半，s4 到来，s3 应立即停止
    [500,  {type: "emotion",    value: "surprised"}],
    [Math.round(s3.duration_ms / 2), s3],
    [s4.duration_ms + 800, s4],
    // 坏音频测试：地址不存在，页面应照样回发 audio_ended，不卡住
    [1500, {type: "speak", text: "（坏音频测试）", audio: "blob:not-exist", duration_ms: 1000, mouth: []}],
    [500,  {type: "emotion",    value: "unknown_value"}],   // 非法情绪值，应兜底 neutral
    [500,  {type: "background", value: "room_day.png"}],
    [3000, {type: "state",      value: "IDLE"}],
  ];

  let i = 0;
  function next() {
    const [wait, msg] = seq[i++ % seq.length];
    handler(msg);
    setTimeout(next, wait);
  }
  setTimeout(next, 500);
}
