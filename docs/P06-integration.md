# P06 前端与P13接入
保留现有视觉和9项回归，无mouth动画、无框架。legacy服务无ID时保留audio_ended兼容；v1需hello指定client_id/playback_owner、speak指定服务端turn/speech及playback_client_id。只有owner实际播放/回报；server仍必须验证连接身份。sync只同步state，重连不重播旧句。
播放结果明确ended/failed/timeout/cancelled/no_audio和reason；重复speech_id去重（最多256），旧回调Promise/timer不结束新句。WS readyState非OPEN不send；断连暂停、清回调timer，2秒重连仅单timer。媒体长度优先metadata+30s；无metadata上限10分钟防无限等待，不按字幕字数早结束。autoplay显示点击按钮，等30s失败回报；可重试播放。音频实际codec/超长/隐私素材需浏览器/目标屏幕验收。
新增字段仍是模拟新增，app/contracts未改，旧app不能强制多客户端安全。PC截图用自制几何占位，最终屏幕H01待测。
