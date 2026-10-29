# P00 前置接口约定 v1（执行副本）
日期：2026-10-08；只冻结文档，不修改 app.py/contracts.py。原机HEAD仅来自收到的包，未访问原机。

## 现有协议（源码核对）
四个公开状态 IDLE/NOTICED/DIALOG/SPEAKING；emotion 使用六情绪及 sprite；animation 使用 frames/interval_ms/loop；speak 使用 text/audio；客户端 audio_ended 当前无服务端标识，多个浏览器都可能发送。当前 WebSocket 不提供安全的轮次仲裁。
ASR 为整句 feed(bytes)/poll()/reset()；输入 16000Hz mono S16LE。合成后统一以浏览器可播放URL或 data URI（WAV/MP3）发送，不能把一次完整合成叫流式首包。contracts 的同步 TTSEngine 与实现的异步方法有差异，P13适配，不改变本次公共接口。

## 模拟新增字段（P13 尚未接入）
|方向/事件|字段|约束|
|---|---|---|
|服务端 hello|client_id, playback_owner|服务端指定一个唯一播放者；其他浏览器只观察|
|服务端 speak|turn_id,speech_id,playback_client_id,text,audio|ID服务端生成，每轮唯一；只有owner播放|
|客户端 audio_started|turn_id,speech_id,client_id,client_elapsed_ms|必须绑定连接身份，客户端声明不能代替认证|
|客户端 audio_ended|turn_id,speech_id,client_id,result,reason|result=ended/failed/timeout/cancelled/no_audio；旧ID、重复、非owner无效|
|服务端 sync|state,playback_owner|重连获取当前状态，不重播旧句|
|内部输入|source=voice/chat/say; user_id; arrival_id|不以显示名字作会话键|
|内部观测|schema,device_id,clock_id,turn_id,speech_id,event,mono_s,wall_time|不同设备/时钟绝不直接相减|

## 仲裁状态与错误生命周期
|内部阶段|公开State|语音gate|重身份提取|退出|
|---|---|---|---|---|
|idle/present|IDLE/NOTICED|开|正常|接收新轮|
|model（含云端、改写、TTS）|DIALOG|关|策略模式暂停；baseline不暂停|实际作业结束后才能释放|
|playback|SPEAKING|关|暂停/低频存在检测|匹配owner+IDs的确认、失败、断连、离开、超时|
|quarantine（超时/取消但模型未停）|DIALOG|关|同model|底层真正完成后释放；迟到结果丢弃|

默认语音/手动输入忙时全部拒绝，明确busy；队列容量0（<=1），不缓存语音回声。未配置前端则返回no_client并释放，不记播放成功。模型超时默认30秒、播放超时默认120秒；由P13定期调用expire，模型阶段超时只标作废并隔离，不假取消底层线程。断连/离开同理；故障释放幂等。
baseline只关闭资源协调（恢复视觉预算），仍保留单活跃轮次、语音防回声、ID隔离等安全条件；B06需固定相同负载。

## fake样例
```json
{"type":"speak","turn_id":"t1","speech_id":"s1","playback_client_id":"browser-A","text":"示例字幕","audio":"data:audio/wav;base64,..."}
```
```json
{"type":"audio_ended","turn_id":"t1","speech_id":"s1","client_id":"browser-A","result":"ended","reason":"onended"}
```
legacy客户端仅用于原服务回归，不允许它在v1集成中解除gate。服务端验证连接身份，不能信任JSON自报client_id。

## 轻量依赖和复验
P01/P05 标准库+pytest；P02 numpy用于已有ASR；P03/P04 PyYAML/janome/requests；P07 numpy；P06 node内置test无需npm。完整现有测试因导入服务需fastapi；P06浏览器演示另需uvicorn/websockets。
实际独立环境见 task-results/P00-prerequisite/evidence/environment.json；不安装 requirements.txt，不下载GPU/人脸/ASR/TTS模型。
本次源码副本基线：112 passed,1 skipped；Node 9 passed。私有Nanami测试skip，不冒充实测。
# P13 状态更新

本文件为 P00/P01–P07 的历史接口基线。已集成的服务协议、停止确认、健康状态和进程监督见 [interfaces-v2.md](interfaces-v2.md)；部署时服务与 renderer 必须一起更新。
