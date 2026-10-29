const nameEl = document.getElementById("character-name");
const titleEl = document.getElementById("conversation-title");
const initialEl = document.getElementById("portrait-initial");
const languageEl = document.getElementById("language");
const portrait = document.getElementById("portrait");
const placeholder = document.getElementById("portrait-placeholder");
const statusEl = document.getElementById("service-status");
const messages = document.getElementById("messages");
const empty = document.getElementById("empty");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const send = document.getElementById("send");
const reset = document.getElementById("reset");
let characterName = "角色";
let ready = false;

function showStatus(text, error = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", error);
}

function showPortrait(url) {
  if (!url) {
    portrait.hidden = true;
    placeholder.hidden = false;
    return;
  }
  portrait.onload = () => {
    portrait.hidden = false;
    placeholder.hidden = true;
  };
  portrait.onerror = () => {
    portrait.hidden = true;
    placeholder.hidden = false;
  };
  portrait.src = `${url}?t=${Date.now()}`;
}

function addMessage(who, body, subtitle = "") {
  empty.hidden = true;
  const item = document.createElement("div");
  item.className = `message ${who === "你" ? "user" : "character"}`;
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who;
  const text = document.createElement("p");
  text.className = "body";
  text.textContent = body;
  item.append(label, text);
  if (subtitle && subtitle !== body) {
    const sub = document.createElement("span");
    sub.className = "subtitle";
    sub.textContent = subtitle;
    item.append(sub);
  }
  messages.append(item);
  messages.scrollTop = messages.scrollHeight;
}

async function responseJson(response) {
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "请求失败");
  return body;
}

async function loadStatus() {
  try {
    const data = await responseJson(await fetch("/api/status"));
    characterName = data.character;
    nameEl.textContent = characterName;
    titleEl.textContent = characterName;
    initialEl.textContent = characterName[0] || "角";
    languageEl.textContent = data.language === "ja" ? "日本語" : "中文";
    showPortrait(data.sprite_url);
    ready = data.ready;
    send.disabled = !ready;
    showStatus(data.issue || `${data.provider} 已设置 · 首次提问时验证连接 · ${data.local_rewrite ? "本地口吻处理已启用" : "显示云端原文"}`, !ready);
  } catch (error) {
    ready = false;
    send.disabled = true;
    showStatus(`无法连接服务：${error.message}`, true);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || !ready || send.disabled) return;
  input.value = "";
  addMessage("你", text);
  send.disabled = true;
  showStatus("正在等待回答");
  try {
    const data = await responseJson(await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text}),
    }));
    addMessage(characterName, data.reply, data.subtitle);
    showPortrait(data.sprite_url);
    showStatus(data.rewrite_note || "对话已连接");
  } catch (error) {
    showStatus(error.message, true);
  } finally {
    send.disabled = false;
    input.focus();
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

reset.addEventListener("click", async () => {
  try {
    await responseJson(await fetch("/api/reset", {method: "POST"}));
    messages.replaceChildren(empty);
    empty.hidden = false;
    showStatus("已开始新对话");
    input.focus();
  } catch (error) {
    showStatus(error.message, true);
  }
});

loadStatus();
