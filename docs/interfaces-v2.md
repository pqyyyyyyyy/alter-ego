# P13 服务协议 aeg-ws-v2

本版落实 P00 的 interfaces-v1 拟新增字段，并加入按 ID 停音确认。P01–P07 模块的 v1 文档是历史设计依据；服务/前端以本文件为准。负责人将本文件转给语音、前端和 B01/B03；本对话不自动发送消息。

## 输入与生命周期

`POST /debug/chat?text=...`、`POST /debug/say?text=...`、PCM 切句均调用同一 Lifecycle。单活跃轮次，待处理对话队列为 0；忙时拒绝，不保存麦克风回授。ASR 在实际识别前占用轮次，不能与手动对话重叠。say 的输入须为日语，skip_rewrite 也必须过日文门禁；user 参数只改本轮称呼，不作为身份键。

HTTP `status=playback, ok=true` 表示本轮结果已发送、等待播放确认，**不代表音频播放完成**。`degraded, ok=true` 表示预设/字幕降级已发送；error/tts_error 说明原因。`busy/no_client/timeout/discarded/failed/empty/closed/invalid/unavailable` 均 ok=false。PCM 接口 `accepted_pcm` 仅表示接收切句输入，识别/播放是异步过程。

稳定注册 ID 作为 session 键；显示姓名允许相同。访客每次到达、离开后下一次输入建立独立会话；最多 16 项，闲置 900 秒，历史最多 6 轮。请求开始快照贯穿整轮；切换身份/离开使旧结果失效。已运行的云端轮次可能写入其原有会话，但不会写新会话，也不会发送迟到的字幕/音频。

## WebSocket

连接建立先接收：

```json
{"type":"hello","protocol":"aeg-ws-v2","client_id":"server-assigned","playback_owner":true}
{"type":"sync","state":"IDLE","playback_owner":true}
```

首个连接拥有播放权，其余只观察。owner 断开后现存首个 observer 收到新 hello 获得播放权；不重播旧句。握手完成前服务不把连接加入广播池。服务使用连接绑定 ID 校验来源，忽略用户 JSON 中伪造的 client_id。

```json
{"type":"speak","turn_id":"t","speech_id":"s","playback_client_id":"owner","text":"中文字幕或日文回落","text_ja":"日文","audio":"data:audio/wav;base64,...","duration_ms":1000}
{"type":"audio_started","turn_id":"t","speech_id":"s","client_id":"owner"}
{"type":"audio_ended","turn_id":"t","speech_id":"s","client_id":"owner","result":"ended","reason":"onended"}
```

audio_ended 接受 ended/failed/timeout/cancelled/no_audio；ID、连接、阶段必须匹配，旧 ID、重复 ACK、observer、无 ID 的 legacy ACK 不释放当前轮次。

超时/LEFT/身份变化/人数变化/退出发送：

```json
{"type":"stop","turn_id":"t","speech_id":"s","playback_client_id":"owner","reason":"left"}
```

renderer 只停止匹配的当前句，先 pause，再发 cancelled/server_stop。服务在 stopping 阶段继续关闭输入 gate，收到对应结束确认后释放。3 秒未收到停止确认会关闭 owner 的 WS，renderer onclose 停音，再释放；owner 断连可恢复新连接。网络隔离时服务无法证明远端扬声器已静音，这一路径只是软件恢复策略，必须在 H 阶段测试物理回授风险。

renderer 可以与无 hello、无轮次 ID 的 legacy 服务兼容，每次重连重新判定协议；有 v1/v2 ID 的 speak 必须先取得 owner。v1 服务若缺 stop，服务端释放播放超时的旧问题仍存在，不能混用旧部署。

## 调度、超时、退出

默认整轮模型/ASR/TTS响应等待 120 秒，播放等待 240 秒，停止确认 3 秒。模型等待超时返回 timeout，但保持 quarantine 与 gate；真实作业完成前拒绝任何新轮次。底层线程不会被假取消或强杀。TTS 适配 async WAV/MP3；旧同步 synthesize 返回编码 WAV 与采样率，在线 edge/HTTP 仍可选。

用 scripts/run_service.py 启动时，父进程监督子服务。quarantine 的真实作业总寿命超过 180 秒，或服务无法响应超过 180 秒，终止其拥有的整个服务进程并退出 75，**不自动重启**。正常退出等待实际作业并关闭设备、请求 Session；监督器给 10 秒退出宽限后终止卡住的子服务。直接 uvicorn 运行没有这个硬退出保证，/health.supervised=false。HTTP 模型服务等外部进程不归此监督器所有，不会被停止；外部服务的队列/资源恢复须 B03 另验。

gate 默认覆盖整轮；busy 时暂停昂贵身份提取、存在检测限制为 1fps，空闲最多 5fps。baseline 只解除身份提取/存在检测预算，仍维持单轮次和安全 gate。相同输入与观测可供 B06 对比，PC 不证明板端节省。

## 观测与健康

`GET /debug/events` 返回有界、只含阶段/ID/结果的服务进程单调时钟事件，不记录原始 PCM、私有正文、密钥。cloud/rewrite/ASR 起止由真实调用边界产生；tts_end 在底层实际返回后产生；WAV 时长从帧数计算，MP3 时长未解析则缺失。统计保留失败/缺失分母。

audio_started/audio_ended 是服务器收到浏览器 onplaying/onended 回报的时刻，backend=ws_receipt，包含传输延迟；不是实听首音。PCM vad_end 是控制器将完整切句交给统一轮次的时刻，不是硬件采样时钟。浏览器、PC、板端时钟不能相减。缓存最多 50000 条事件，长测需及时导出，跨断片统计会标缺失。样本分别保留 trace_complete（轨迹端点有效）、playback_complete（ended 回报）、failed/degraded；complete 仍表示成功且轨迹完整。云端预设等降级会记录匿名分类，统计单列 n_degraded。

health.components 区分 disabled/loading/loaded/ready/running/degraded/failed；客户端构造后 loaded/connection_not_checked 不当作连通性。tts_loaded 与 tts_ready 分开；历史字段保留以兼容旧诊断。ALTER_EGO_DEVICE_ID 必须按实际设备填写；fixture 默认为 pc-fixture，正式未配置则为 unspecified。clock_id 为每次服务进程独立生成，跨重启也不可直接相减。

ASR 默认关闭，ALTER_EGO_ASR=1 且指定已有本地模型才加载，不下载。只接 PCM，H03 再连接 ALSA 源。感知默认关闭；fixture 明确启用时仅使用 fake 感知事件、fake ASR、fake 云端与合成提示音，绝不打开设备或读取真实用户库。fixture 控制路由在正式模式返回 404，服务启动器只绑定 127.0.0.1。
