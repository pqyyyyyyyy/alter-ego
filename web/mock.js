// 假数据发生器 —— 自测用，模拟后端推消息。
// 消息格式与后端约定完全一致，见任务书「后端会发给你的消息」。
function startMock(handler) {
  const seq = [
    {type: "state",      value: "NOTICED"},
    {type: "user",       name: "小明", known: true},
    {type: "emotion",    value: "happy"},
    {type: "subtitle",   text: "欸，你来啦！"},
    {type: "emotion",    value: "shy"},
    {type: "subtitle",   text: "唔…有点不好意思呢。"},
    {type: "emotion",    value: "surprised"},
    {type: "subtitle",   text: "诶？真的吗！"},
    {type: "emotion",    value: "sad"},
    {type: "subtitle",   text: "这样啊……"},
    {type: "emotion",    value: "angry"},
    {type: "subtitle",   text: "哼，不理你了。"},
    {type: "emotion",    value: "unknown_value"},   // 故意发非法值，应兜底成 neutral
    {type: "subtitle",   text: "（非法情绪值 → 兜底 neutral）"},
    {type: "background", value: "room_day.png"},
    {type: "state",      value: "IDLE"},
  ];
  let i = 0;
  setInterval(() => { handler(seq[i++ % seq.length]); }, 1000);
}
