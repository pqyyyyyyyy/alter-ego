const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function renderer(options = {}) {
  const nodes = new Map();
  const listeners = new Map();
  function element() {
    const classes = new Set();
    const ownListeners = new Map();
    const node = {
      textContent: "", value: "", src: "", style: {}, dataset: {}, hidden: true, children: [], className: "", attributes: {},
      classList: {add: (v) => classes.add(v), remove: (v) => classes.delete(v),
                  contains: (v) => classes.has(v)},
      addEventListener(type, fn) {ownListeners.set(type, fn); if (!listeners.has(type)) listeners.set(type, fn);},
      setAttribute(key, value) { this.attributes[key] = value; },
      append(...values) {this.children.push(...values);},
      appendChild(value) {this.children.push(value);},
      replaceChildren(...values) {this.children = values;},
      querySelector() {return this.submitButton || (this.submitButton = {});},
    };
    node._listeners = ownListeners;
    return node;
  }
  const images = [];
  const intervals = new Map();
  const timeouts = new Map();
  const timeoutDurations = new Map();
  const audio = [];
  const sent = [];
  const fetches = [];
  const requests = [];
  const microphones = [];
  const processors = [];
  const contexts = [];
  let blockNextAudio = false;
  let socket;
  let timer = 0;
  const context = {
    document: {getElementById: (id) => {
      if (!nodes.has(id)) nodes.set(id, element());
      return nodes.get(id);
    }, createElement: () => element(), body: {dataset: {}}},
    location: {protocol: "http:", host: "localhost"},
    AbortController,
    URLSearchParams,
    navigator: {mediaDevices: {getUserMedia: async () => {
      if (options.permission) return options.permission;
      const track = {stop() {this.stopped = true;}};
      microphones.push(track);
      return {getTracks: () => [track]};
    }}},
    AudioContext: class {
      constructor() {this.sampleRate = 48000; this.state = "running"; contexts.push(this);}
      async resume() {}
      async close() {this.state = "closed";}
      createMediaStreamSource() {return {connect() {}, disconnect() {}};}
      createScriptProcessor() {
        const processor = {connect() {}, disconnect() {this.disconnected = true;}};
        processors.push(processor); return processor;
      }
    },
    fetch: (url, config) => {
      fetches.push(url);
      requests.push({url, config});
      if (options.fetch) return options.fetch(url, config);
      if (url === "/debug/transcribe") return Promise.resolve({ok: true, json: async () => ({ok: true, text: "你好。"})});
      if (url.startsWith("/debug/chat?")) return Promise.resolve({ok: true, json: async () => ({ok: true})});
      return Promise.resolve({ok: true, json: async () => ({
        animations: {idle: {frames: ["tired-1.png"]}},
        emotions: {neutral: ["usual.png"]},
      })});
    },
    Image: class {constructor() {images.push(this);} decode() {return Promise.resolve();}},
    WebSocket: class {constructor() {socket = this; this.readyState = 1;} send(value) {sent.push(JSON.parse(value));}},
    Audio: class {
      constructor(src) {this.src = src; this.attempts = 0; audio.push(this);}
      play() {
        this.attempts++;
        const blocked = this.reject || blockNextAudio;
        blockNextAudio = false;
        return blocked ? Promise.reject(new Error("blocked")) : Promise.resolve();
      }
      pause() {this.paused = true;}
    },
    setInterval: (fn) => {intervals.set(++timer, fn); return timer;},
    clearInterval: (id) => intervals.delete(id),
    setTimeout: (fn, ms) => {timeouts.set(++timer, fn); timeoutDurations.set(timer, ms); return timer;},
    clearTimeout: (id) => {timeouts.delete(id); timeoutDurations.delete(id);},
    console: {error() {}},
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../web/renderer.js"), "utf8"), context);
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  return {
    nodes, images, intervals, timeouts, timeoutDurations, audio, sent, fetches, requests, microphones, processors, contexts, listeners, settle,
    samples: (value, count = 4800) => processors.at(-1).onaudioprocess({inputBuffer: {
      getChannelData: () => new Float32Array(count).fill(value),
    }}),
    close: () => {socket.readyState = 3; socket.onclose();},
    reconnect: () => {
      const [id, fn] = [...timeouts.entries()][0];
      timeouts.delete(id); fn(); socket.onopen();
    },
    owner: () => socket.onmessage({data: JSON.stringify({type:'hello',client_id:'A',playback_owner:true})}),
    raw: (data) => socket.onmessage({data}),
    blockNextAudio: () => {blockNextAudio = true;},
    send: (msg) => socket.onmessage({data: JSON.stringify(msg)}),
    click: (id) => nodes.get(id)._listeners.get("click")(),
    load: async (name) => {
      const img = images.find((i) => i.src.endsWith(encodeURIComponent(name)));
      assert.ok(img, `Missing request for ${name}`);
      await img.onload();
      await settle();
    },
  };
}

test("initial sync loads the idle sprite before the first conversation", async () => {
  const r = renderer();
  r.send({type: "sync", state: "IDLE", playback_owner: true});
  await r.settle();
  await r.load("tired-1.webp");
  assert.equal(r.fetches[0], "/debug/sprites");
  assert.equal(r.nodes.get("layer-char").classList.contains("ready"), true);
});

test("character name comes from the loaded persona manifest", async () => {
  const r = renderer({fetch: () => Promise.resolve({ok: true, json: async () => ({
    character: "示例角色", emotions: {}, animations: {},
  })})});
  r.send({type: "sync", state: "IDLE", playback_owner: true});
  await r.settle();
  assert.equal(r.nodes.get("brand-character").textContent, "示例角色");
  r.send({type: "speak", speech_id: "fixture", turn_id: "turn", text: "你好"});
  assert.equal(r.nodes.get("dialog-log-list").children[0].children[1].children[0].textContent, "示例角色");
});

test("legacy state message also requests an initial sprite", async () => {
  const r = renderer();
  r.send({type: "state", value: "NOTICED"});
  await r.settle();
  await r.load("usual.webp");
  assert.equal(r.fetches[0], "/debug/sprites");
  assert.equal(r.nodes.get("layer-char").classList.contains("ready"), true);
});

test("dialog log starts collapsed and records both sides with the reply emotion sprite", async () => {
  const r = renderer();
  const panel = r.nodes.get("dialog-log");
  const input = r.nodes.get("chat-input");
  input.value = "你好";
  const submit = r.nodes.get("chat-form")._listeners.get("submit");
  await submit({preventDefault() {}});
  r.send({type: "emotion", value: "happy", sprite: "happy.png"});
  r.send({type: "speak", text: "你好呀"});
  const list = r.nodes.get("dialog-log-list");
  assert.equal(panel.hidden, true);
  assert.equal(list.children.length, 2);
  assert.equal(list.children[0].className, "dialog-entry user");
  assert.equal(list.children[1].className, "dialog-entry character");
  assert.equal(list.children[1].children[0].src, "/character/sprites/happy.webp");
  assert.match(list.children[1].children[1].children[0].children[0].textContent, /开心/);
  r.click("log-toggle");
  assert.equal(panel.hidden, false);
  r.click("log-clear");
  assert.equal(list.children.length, 0);
});

test("presence and A/B identity messages update the compact status marker", async () => {
  const r = renderer();
  const status = r.nodes.get("perception-status");
  assert.equal(status.textContent, "识别到 0 人 用户：[空]");
  r.send({type: "presence", value: true, faces: 1});
  assert.equal(status.textContent, "识别到 1 人 用户：[空]");
  r.send({type: "user", name: "A", known: true});
  assert.equal(status.textContent, "识别到 1 人 用户：[A]");
  r.send({type: "presence", value: true, faces: 2, visible_users: ["A", "B"]});
  assert.equal(status.textContent, "识别到 2 人 用户：[A、B]");
  r.send({type: "presence", value: false, faces: 0});
  assert.equal(status.textContent, "识别到 0 人 用户：[空]");
});

test("camera identities clear on unknown faces, failure and websocket loss", () => {
  const r = renderer();
  const status = r.nodes.get("perception-status");
  r.send({type: "presence", value: true, faces: 2, visible_users: ["B", "A"]});
  r.send({type: "user", name: "B"});
  assert.equal(status.textContent, "识别到 2 人 用户：[A、B]");
  r.send({type: "presence", value: true, faces: 1, visible_users: []});
  assert.equal(status.textContent, "识别到 1 人 用户：[空]");
  r.send({type: "presence", value: false, faces: 0, visible_users: [], available: false});
  assert.match(status.textContent, /摄像头未就绪/);
  r.send({type: "presence", value: true, faces: 1, visible_users: ["A"], available: true});
  assert.equal(status.textContent, "识别到 1 人 用户：[A]");
  r.close();
  assert.equal(status.textContent, "识别到 0 人 用户：[空] 摄像头未就绪");
});

test("six emotion commands replace the sprite without hiding the previous image", async () => {
  const r = renderer();
  const char = r.nodes.get("layer-char");
  for (const name of ["usual.png", "soft smile.png", "curious.png", "shy-annoyed.png", "confusing.png", "assertive-level1.png"]) {
    const previous = char.src;
    r.send({type: "emotion", sprite: name});
    assert.equal(char.src, previous);
    if (previous) assert.equal(char.classList.contains("ready"), true);
    await r.load(name.replace(/\.png$/, ".webp"));
    assert.equal(char.src, "/character/sprites/" + encodeURIComponent(name.replace(/\.png$/, ".webp")));
    assert.equal(char.classList.contains("ready"), true);
  }
});

test("camera switch confirms via shared presence, clears identities and reopens capture", async () => {
  let r;
  r = renderer({fetch: async (url, config) => {
    assert.equal(url, "/camera/control");
    assert.equal(config.headers["X-Alter-Ego-Camera"], "1");
    const {enabled} = JSON.parse(config.body);
    r.send({type: "presence", camera_controllable: true, camera_enabled: enabled,
      camera_status: enabled ? "loading" : "disabled", available: false, faces: 0, visible_users: []});
    return {ok: true, json: async () => ({ok: true, camera: {enabled}})};
  }});
  const button = r.nodes.get("camera-toggle");
  assert.equal(button.disabled, true);
  r.send({type: "presence", camera_controllable: true, camera_enabled: true,
    camera_status: "ready", available: true, faces: 2, visible_users: ["A", "B"]});
  assert.equal(button.disabled, false);
  assert.equal(r.nodes.get("camera-switch-state").textContent, "ON");
  assert.equal(button.attributes["aria-checked"], "true");
  await r.click("camera-toggle");
  assert.equal(r.nodes.get("perception-status").textContent, "摄像头关闭");
  assert.equal(r.nodes.get("camera-switch-state").textContent, "OFF");
  assert.equal(button.attributes["aria-checked"], "false");
  assert.equal(button.attributes["aria-pressed"], "false");
  // A late identity event must not expose stale A/B while disabled.
  r.send({type: "user", name: "A"});
  assert.equal(r.nodes.get("perception-status").textContent, "摄像头关闭");
  await r.click("camera-toggle");
  assert.equal(r.nodes.get("camera-switch-state").textContent, "ON");
  assert.match(r.nodes.get("perception-status").textContent, /识别到 0 人 用户：\[空\]/);
  assert.match(r.nodes.get("camera-feedback").textContent, /启动中/);
});

test("camera switch blocks repeat clicks and failure never pretends capture is off", async () => {
  let finish;
  const r = renderer({fetch: () => new Promise(resolve => {finish = resolve;})});
  r.send({type: "presence", camera_controllable: true, camera_enabled: true,
    available: true, faces: 1, visible_users: ["A"]});
  const switching = r.click("camera-toggle");
  assert.equal(r.nodes.get("camera-toggle").disabled, true);
  assert.equal(r.nodes.get("camera-switch-state").textContent, "…");
  await r.click("camera-toggle");
  assert.equal(r.requests.length, 1);
  finish({ok: false, json: async () => ({ok: false})});
  await switching;
  assert.equal(r.nodes.get("camera-toggle").disabled, false);
  assert.equal(r.nodes.get("perception-status").textContent, "识别到 1 人 用户：[A]");
  assert.equal(r.nodes.get("camera-feedback").dataset.kind, "error");
});

test("camera disabled snapshot survives refresh and disconnect blocks controls", async () => {
  let finish;
  const r = renderer({fetch: () => new Promise(resolve => {finish = resolve;})});
  r.send({type: "presence", camera_controllable: true, camera_enabled: false,
    available: false, faces: 0, visible_users: []});
  assert.equal(r.nodes.get("perception-status").textContent, "摄像头关闭");
  const switching = r.click("camera-toggle");
  r.close();
  finish({ok: true, json: async () => ({ok: true, camera: {enabled: true}})});
  await switching;
  assert.equal(r.nodes.get("camera-toggle").disabled, true);
  assert.equal(r.nodes.get("perception-status").textContent, "摄像头关闭");
  assert.match(r.nodes.get("camera-feedback").textContent, /连接中断/);
});

test("camera loading hint clears on readiness and another window can switch it off", () => {
  const r = renderer();
  r.send({type: "presence", camera_controllable: true, camera_enabled: true,
    camera_status: "loading", available: false, faces: 0, visible_users: []});
  assert.match(r.nodes.get("camera-feedback").textContent, /启动中/);
  r.send({type: "presence", camera_controllable: true, camera_enabled: true,
    camera_status: "ready", available: true, faces: 1, visible_users: ["A"]});
  assert.equal(r.nodes.get("camera-feedback").textContent, "");
  r.send({type: "presence", camera_controllable: true, camera_enabled: false,
    camera_status: "disabled", available: false, faces: 0, visible_users: []});
  assert.equal(r.nodes.get("camera-switch-state").textContent, "OFF");
  assert.match(r.nodes.get("camera-feedback").textContent, /保持唤醒/);
});

test("emotion stops a looping animation and rejects its late image load", async () => {
  const r = renderer();
  r.send({type: "animation", frames: ["idle.png", "idle2.png"], loop: true});
  assert.equal(r.intervals.size, 1);
  r.send({type: "emotion", sprite: "happy.png"});
  assert.equal(r.intervals.size, 0);
  await r.load("happy.webp");
  await r.load("idle.webp");
  assert.equal(r.nodes.get("layer-char").src, "/character/sprites/happy.webp");
});

test("a state change cancels the previous animation and pending frame", async () => {
  const r = renderer();
  r.send({type: "emotion", sprite: "usual.png"});
  await r.load("usual.webp");
  r.send({type: "animation", frames: ["thinking.png"], loop: true});
  r.send({type: "state", value: "SPEAKING"});
  await r.load("thinking.webp");
  assert.equal(r.intervals.size, 0);
  assert.equal(r.nodes.get("layer-char").src, "/character/sprites/usual.webp");
});

test("newest sprite wins when requests complete out of order", async () => {
  const r = renderer();
  r.send({type: "emotion", sprite: "slow.png"});
  r.send({type: "emotion", sprite: "fast.png"});
  await r.load("fast.webp");
  await r.load("slow.webp");
  assert.equal(r.nodes.get("layer-char").src, "/character/sprites/fast.webp");
});

test("failed image preserves the visible sprite and can be retried", async () => {
  const r = renderer();
  r.send({type: "emotion", sprite: "usual.png"});
  await r.load("usual.webp");
  r.send({type: "emotion", sprite: "missing.png"});
  r.images.at(-1).onerror();
  await r.settle();
  assert.ok(r.images.at(-1).src.endsWith("missing.png"));
  r.images.at(-1).onerror();
  await r.settle();
  assert.equal(r.nodes.get("layer-char").src, "/character/sprites/usual.webp");
  assert.equal(r.nodes.get("layer-char").classList.contains("ready"), true);
  assert.equal(r.nodes.get("speech-bubble").textContent, "立绘加载失败");
  r.send({type: "emotion", sprite: "missing.png"});
  assert.equal(r.images.filter((i) => i.src.endsWith("missing.webp")).length, 2);
  r.images.at(-1).onerror();
  await r.settle();
  await r.images.at(-1).onload();
  await r.settle();
  assert.equal(r.nodes.get("speech-bubble").textContent, "");
});

test("speech does not start a mouth animation", () => {
  const r = renderer();
  r.send({type: "speak", text: "字幕", mouth: [[0, 1]], duration_ms: 100});
  assert.equal(r.intervals.size, 0);
  assert.equal(r.nodes.get("layer-subtitle").textContent, "字幕");
});

test("MP3 speech plays, waits for onended and acknowledges exactly once", async () => {
  const r = renderer();
  r.send({type: "speak", text: "你好", audio: "data:audio/mpeg;base64,SUQz"});
  await r.settle();
  assert.equal(r.audio[0].src, "data:audio/mpeg;base64,SUQz");
  assert.equal(r.audio[0].attempts, 1);
  assert.equal(r.nodes.get("replay-audio").hidden, true);
  assert.equal(r.sent.filter((s) => s.type === "audio_ended").length, 0);
  const ended = r.audio[0].onended;
  ended(); ended();
  assert.equal(r.sent.filter((s) => s.type === "audio_ended").length, 1);
});

test("autoplay rejection exposes user-driven replay", async () => {
  const r = renderer();
  r.blockNextAudio();
  r.send({type: "speak", audio: "data:audio/mpeg;base64,SUQz"});
  await r.settle();
  assert.equal(r.nodes.get("replay-audio").hidden, false);
  assert.equal(r.nodes.get("speech-bubble").textContent, "点击播放语音");
  r.audio[0].reject = false;
  r.click("replay-audio");
  await r.settle();
  assert.equal(r.audio[0].attempts, 2);
  assert.equal(r.nodes.get("replay-audio").hidden, true);
  r.audio[0].onended();
  assert.equal(r.sent.filter((s) => s.type === "audio_ended").length, 1);
});

test("stale speech completion cannot end a newer reply", async () => {
  const r = renderer();
  r.send({type: "speak", audio: "first"});
  const oldEnded = r.audio[0].onended;
  r.send({type: "speak", audio: "second"});
  oldEnded();
  assert.equal(r.sent.filter((s) => s.type === "audio_ended").length, 0);
  r.audio[1].onended();
  assert.equal(r.sent.filter((s) => s.type === "audio_ended").length, 1);
});


test("v1 owner IDs, duplicates and replaced speech acknowledgement", async () => {
  const r=renderer();r.send({type:'hello',client_id:'A',playback_owner:true});
  r.send({type:'speak',turn_id:'t1',speech_id:'s1',playback_client_id:'A',audio:'data:audio/wav;base64,AA'});
  const oldEnded=r.audio[0].onended;
  r.send({type:'speak',turn_id:'t1',speech_id:'s1',audio:'duplicate'});assert.equal(r.audio.length,1);
  r.send({type:'speak',turn_id:'t2',speech_id:'s2',audio:'data:audio/mp3;base64,AA'});
  oldEnded();assert.equal(r.sent.length,1);assert.equal(r.sent[0].speech_id,'s1');assert.equal(r.sent[0].result,'cancelled');
  r.audio[1].onended();assert.equal(r.sent[1].speech_id,'s2');assert.equal(r.sent[1].result,'ended');
  r.send({type:'speak',turn_id:'t2',speech_id:'s2',audio:'duplicate'});assert.equal(r.audio.length,2);
});

test("observer never plays or acknowledges and sync never replays", async () => {
  const r=renderer();r.send({type:'hello',client_id:'B',playback_owner:false});
  r.send({type:'speak',turn_id:'t',speech_id:'s',playback_client_id:'A',text:'字幕',audio:'a.wav'});
  assert.equal(r.audio.length,0);assert.equal(r.sent.length,0);assert.equal(r.nodes.get('layer-subtitle').textContent,'字幕');
  r.send({type:'sync',state:'SPEAKING',playback_owner:true});assert.equal(r.audio.length,0);
});

test("disconnect removes audio callbacks and timers, stale end cannot send", async () => {
  const r=renderer();r.owner();r.send({type:'speak',turn_id:'t',speech_id:'s',audio:'a.wav'});
  const end=r.audio[0].onended;r.close();end();assert.equal(r.sent.length,0);assert.equal(r.audio[0].onended,null);
  assert.equal(r.timeouts.size,1); // only reconnect remains
});

test("bad JSON, null and arrays do not crash; audio errors report reasons", async () => {
  const r=renderer();r.owner();for(const raw of ['bad','null','[]','42'])r.raw(raw);
  r.send({type:'speak',turn_id:'t',speech_id:'s',audio:'invalid.wav'});r.audio[0].onerror();
  assert.equal(r.sent[0].result,'failed');assert.equal(r.sent[0].reason,'audio_error');
});

test("long audio waits for onended instead of subtitle length", async () => {
  const r=renderer();r.owner();r.send({type:'speak',turn_id:'t',speech_id:'s',text:'短',audio:'long.wav'});
  r.audio[0].duration=180;r.audio[0].onloadedmetadata();await r.settle();assert.equal(r.sent.length,0);
  r.audio[0].onended();assert.equal(r.sent[0].result,'ended');
});

test("v1 autoplay timeout and subtitle-only have explicit outcomes", async () => {
  const r=renderer();r.owner();r.blockNextAudio();r.send({type:'speak',turn_id:'t',speech_id:'s',audio:'a.mp3'});await r.settle();
  const timer=[...r.timeouts.values()][0];timer();assert.equal(r.sent[0].reason,'autoplay_wait');
  r.send({type:'speak',turn_id:'t2',speech_id:'s2',text:'无语音'});[...r.timeouts.values()][0]();assert.equal(r.sent[1].result,'no_audio');
});

test("legacy reconnect plays fresh speech and never replays the old audio", () => {
  const r=renderer();r.send({type:'speak',audio:'old.wav'});r.close();r.reconnect();
  r.send({type:'speak',audio:'fresh.wav'});
  assert.equal(r.audio.length,2);assert.equal(r.audio[0].paused,true);
  assert.equal(r.audio[1].attempts,1);
});

test("v1 reconnect requires hello owner and observer remains silent", () => {
  const r=renderer();r.owner();r.send({type:'speak',turn_id:'t',speech_id:'s',audio:'old.wav'});
  r.close();r.reconnect();r.send({type:'speak',turn_id:'t2',speech_id:'s2',audio:'new.wav'});
  assert.equal(r.audio.length,1);
  r.send({type:'hello',client_id:'B',playback_owner:false});
  r.send({type:'speak',turn_id:'t3',speech_id:'s3',audio:'observer.wav'});
  assert.equal(r.audio.length,1);
  r.send({type:'hello',client_id:'B',playback_owner:true});
  r.send({type:'speak',turn_id:'t4',speech_id:'s4',playback_client_id:'B',audio:'owner.wav'});
  assert.equal(r.audio.length,2);
});

test("server stop pauses before ACK; old stop cannot cancel a fresh sentence", () => {
  const r=renderer();r.owner();
  const speech={type:'speak',turn_id:'t',speech_id:'s',playback_client_id:'A',audio:'a.wav'};
  r.send(speech);r.send({type:'stop',turn_id:'wrong',speech_id:'s'});
  assert.equal(r.audio[0].paused,undefined);
  r.send({type:'stop',turn_id:'t',speech_id:'s',playback_client_id:'A'});
  assert.equal(r.audio[0].paused,true);assert.equal(r.sent[0].result,'cancelled');
  assert.equal(r.sent[0].reason,'server_stop');
  r.send({...speech,turn_id:'t2',speech_id:'s2'});
  r.send({type:'stop',turn_id:'t',speech_id:'s'});
  assert.equal(r.audio[1].paused,undefined);assert.equal(r.sent.length,1);
});

test("a pause previews local ASR without sending a dialog; second click sends once", async () => {
  const r = renderer(); r.owner();
  await r.click("record-toggle");
  assert.equal(r.nodes.get("record-toggle").textContent, "停止并发送");
  assert.equal(r.nodes.get("record-cancel").hidden, false);
  r.samples(.05); r.samples(.05);
  for (let i = 0; i < 9; i++) r.samples(0);
  await r.settle();
  assert.equal(r.nodes.get("chat-input").value, "你好。");
  assert.equal(r.fetches.filter(url => url.startsWith("/debug/chat?")).length, 0);
  const pcm = r.requests.find(row => row.url === "/debug/transcribe").config.body;
  const sample = new DataView(pcm).getInt16(0, true);
  assert.equal(sample, Math.round(.05 * 32767));
  await r.click("record-toggle");
  assert.equal(r.fetches.filter(url => url.startsWith("/debug/chat?")).length, 1);
  assert.equal(r.nodes.get("chat-input").value, "你好。");
  assert.equal(r.nodes.get("dialog-log-list").children.length, 1);
  assert.equal(r.microphones[0].stopped, true);
  assert.equal(r.contexts[0].state, "closed");
});

test("stop flushes the final sentence and waits for ASR before committing", async () => {
  let resolveASR;
  const r = renderer({fetch: (url) => url === "/debug/transcribe"
    ? new Promise(resolve => {resolveASR = resolve;})
    : Promise.resolve({ok: true, json: async () => ({ok: true})})});
  r.owner(); await r.click("record-toggle");
  r.samples(.05); r.samples(.05);
  const finish = r.click("record-toggle"); await r.settle();
  assert.equal(r.fetches.filter(url => url.startsWith("/debug/chat?")).length, 0);
  assert.equal(r.nodes.get("record-toggle").disabled, true);
  resolveASR({ok: true, json: async () => ({ok: true, text: "最后一句"})});
  await finish;
  assert.ok(r.fetches.some(url => url.startsWith("/debug/chat?")));
  assert.equal(r.nodes.get("chat-input").value, "最后一句");
});

test("terminate during ASR aborts upload and ignores a late result", async () => {
  let resolveASR;
  const r = renderer({fetch: (url) => url === "/debug/transcribe"
    ? new Promise(resolve => {resolveASR = resolve;})
    : Promise.resolve({ok: true, json: async () => ({ok: true})})});
  r.nodes.get("chat-input").value = "原草稿";
  r.owner(); await r.click("record-toggle");
  r.samples(.05); r.samples(.05);
  const finish = r.click("record-toggle"); await r.settle();
  r.click("record-cancel");
  assert.equal(r.requests.find(row => row.url === "/debug/transcribe").config.signal.aborted, true);
  resolveASR({ok: true, json: async () => ({ok: true, text: "迟到结果"})});
  await finish;
  assert.equal(r.nodes.get("chat-input").value, "原草稿");
  assert.equal(r.fetches.filter(url => url.startsWith("/debug/chat?")).length, 0);
});

test("terminate while browser permission is pending stops the late microphone", async () => {
  let grant;
  const permission = new Promise(resolve => {grant = resolve;});
  const r = renderer({permission}); r.owner();
  const start = r.click("record-toggle");
  r.click("record-cancel");
  const track = {stop() {this.stopped = true;}};
  grant({getTracks: () => [track]}); await start;
  assert.equal(track.stopped, true);
  assert.equal(r.processors.length, 0);
});

test("silence produces no ASR/dialog and recording timeout requires explicit send", async () => {
  const r = renderer(); r.owner(); await r.click("record-toggle");
  for (let i = 0; i < 9; i++) r.samples(0);
  await r.click("record-toggle");
  assert.equal(r.fetches.filter(url => url === "/debug/transcribe").length, 0);
  assert.match(r.nodes.get("voice-status").textContent, /没有识别/);
  r.click("record-cancel"); await r.click("record-toggle");
  r.samples(.05); r.samples(.05);
  await [...r.timeouts.values()][0]();
  assert.equal(r.nodes.get("record-toggle").textContent, "发送识别文字");
  assert.equal(r.fetches.filter(url => url.startsWith("/debug/chat?")).length, 0);
  await r.click("record-toggle");
  assert.equal(r.fetches.filter(url => url.startsWith("/debug/chat?")).length, 1);
});

test("ASR unavailable restores editable text without automatic conversation", async () => {
  const r = renderer({fetch: () => Promise.resolve({ok: true, json: async () => ({ok: false, error: "asr_not_enabled"})})});
  r.owner(); await r.click("record-toggle"); r.samples(.05); r.samples(.05);
  await r.click("record-toggle");
  assert.match(r.nodes.get("voice-status").textContent, /未启用/);
  assert.equal(r.nodes.get("chat-input").readOnly, false);
  assert.equal(r.fetches.filter(url => url.startsWith("/debug/chat?")).length, 0);
});

test("failed text chat keeps the original draft", async () => {
  const r = renderer({fetch: () => Promise.resolve({ok: true, json: async () => ({ok: false, error: "busy"})})});
  r.nodes.get("chat-input").value = "保留草稿";
  await r.nodes.get("chat-form")._listeners.get("submit")({preventDefault() {}});
  assert.equal(r.nodes.get("chat-input").value, "保留草稿");
});

test("voice warmup blocks text and microphone until READY, then hides success after three seconds", async () => {
  const r = renderer(); r.owner();
  r.send({type: "voice_output", configured: true, enabled: true, mode: "fast", status: "warming"});
  assert.equal(r.nodes.get("tts-warmup").textContent, "正在预热语音模块");
  assert.equal(r.nodes.get("tts-warmup").hidden, false);
  assert.equal(r.nodes.get("record-toggle").disabled, true);
  assert.equal(r.nodes.get("chat-form").submitButton.disabled, true);
  assert.equal(r.nodes.get("chat-input").readOnly, true);
  r.nodes.get("chat-input").value = "测试";
  await r.nodes.get("chat-form")._listeners.get("submit")({preventDefault() {}});
  await r.click("record-toggle");
  assert.equal(r.microphones.length, 0);
  assert.equal(r.fetches.filter(url => url.startsWith("/debug/chat?")).length, 0);
  r.send({type: "voice_output", configured: true, enabled: true, mode: "fast", status: "ready"});
  assert.equal(r.nodes.get("tts-switch-state").textContent, "ON");
  assert.equal(r.nodes.get("tts-toggle").attributes["aria-checked"], "true");
  assert.equal(r.nodes.get("tts-warmup").textContent, "预热成功");
  assert.equal(r.nodes.get("record-toggle").disabled, false);
  assert.equal(r.nodes.get("chat-form").submitButton.disabled, false);
  const timer = [...r.timeoutDurations].find(([id, duration]) => duration === 3000)[0];
  r.timeouts.get(timer)();
  assert.equal(r.nodes.get("tts-warmup").hidden, true);
});

test("failed warmup switches OFF and releases text input", () => {
  const r = renderer(); r.owner();
  r.send({type: "voice_output", configured: true, enabled: true, status: "warming"});
  r.send({type: "voice_output", configured: true, enabled: false, status: "failed"});
  assert.equal(r.nodes.get("tts-switch-state").textContent, "OFF");
  assert.match(r.nodes.get("tts-warmup").textContent, /预热失败/);
  assert.equal(r.nodes.get("chat-form").submitButton.disabled, false);
  assert.equal(r.nodes.get("record-toggle").disabled, false);
});

test("voice modes and switch follow ordered server state, not an older POST response", async () => {
  let r;
  r = renderer({fetch: async (url, config) => {
    if (url !== "/voice/control") return {ok: true, json: async () => ({})};
    assert.equal(config.headers["X-Alter-Ego-Voice"], "1");
    const {enabled, mode} = JSON.parse(config.body);
    r.send({type: "voice_output", configured: true, enabled, mode, status: enabled ? "ready" : "off"});
    return {ok: true, json: async () => ({ok: true, voice: {enabled, mode: "fast", status: "warming"}})};
  }});
  r.owner(); r.send({type: "voice_output", configured: true, enabled: true, mode: "fast", status: "ready"});
  await r.click("tts-sync");
  assert.equal(r.nodes.get("tts-sync").attributes["aria-pressed"], "true");
  assert.equal(r.nodes.get("tts-fast").attributes["aria-pressed"], "false");
  assert.match(r.nodes.get("tts-mode-help").textContent, /一起/);
  await r.click("tts-toggle");
  assert.equal(r.nodes.get("tts-switch-state").textContent, "OFF");
});

test("fast text preview and final audio share one log entry and replay emits no new ACK", async () => {
  const r = renderer(); r.owner();
  r.send({type: "voice_output", configured: true, enabled: true, mode: "fast", status: "ready"});
  const msg = {turn_id: "t", speech_id: "s", playback_client_id: "A", text: "你好"};
  r.send({type: "reply_preview", ...msg});
  assert.equal(r.nodes.get("layer-subtitle").textContent, "你好");
  assert.equal(r.audio.length, 0);
  assert.equal(r.sent.length, 0);
  const list = r.nodes.get("dialog-log-list");
  assert.equal(list.children.length, 1);
  r.send({type: "speak", ...msg, audio: "data:audio/wav;base64,test"});
  assert.equal(list.children.length, 1);
  r.audio.at(-1).onplaying(); r.audio.at(-1).onended();
  assert.equal(r.sent.length, 2);
  const body = list.children[0].children[1].children[1];
  assert.equal(body.disabled, false);
  body._listeners.get("click")(); await r.settle();
  assert.equal(r.audio.length, 2);
  r.audio.at(-1).onended();
  assert.equal(r.sent.length, 2);
});

test("only the last ten generated replies can replay, voice OFF stops and blocks replay", () => {
  const r = renderer(); r.owner();
  r.send({type: "voice_output", configured: true, enabled: true, status: "ready"});
  for (let i = 0; i < 11; i++) {
    r.send({type: "speak", turn_id: "t"+i, speech_id: "s"+i, playback_client_id: "A",
      text: "回复"+i, audio: "data:audio/wav;base64,"+i});
    r.audio.at(-1).onended();
  }
  const entries = r.nodes.get("dialog-log-list").children;
  assert.equal(entries.length, 11);
  assert.equal(entries[0].children[1].children[1].disabled, true);
  assert.match(entries[0].children[1].children[2].textContent, /移出/);
  for (const entry of entries.slice(1)) assert.equal(entry.children[1].children[1].disabled, false);
  entries[1].children[1].children[1]._listeners.get("click")();
  const replayAudio = r.audio.at(-1);
  r.send({type: "voice_output", configured: true, enabled: false, status: "off"});
  assert.equal(replayAudio.paused, true);
  assert.equal(entries[1].children[1].children[1].disabled, true);
  r.click("log-clear"); assert.equal(r.nodes.get("dialog-log-list").children.length, 0);
});

test("voice OFF mutes active playback exactly once and refuses late audio", () => {
  const r = renderer(); r.owner();
  r.send({type: "voice_output", configured: true, enabled: true, status: "ready"});
  r.send({type: "speak", turn_id: "t", speech_id: "s", text: "test", audio: "data:audio/wav;base64,x"});
  const audio = r.audio.at(-1);
  r.send({type: "voice_output", configured: true, enabled: false, status: "off"});
  assert.equal(audio.paused, true);
  assert.equal(r.sent.filter(row => row.type === "audio_ended").length, 1);
  assert.equal(r.sent.at(-1).result, "cancelled");
  r.send({type: "speak", turn_id: "t2", speech_id: "s2", text: "late", audio: "data:audio/wav;base64,x"});
  assert.equal(r.audio.length, 1);
  assert.equal(r.sent.at(-1).reason, "voice_off");
});
