# P05 观测口径及集成挂点
schema=aeg-events-v1。关联device_id/clock_id/turn_id/speech_id；monotonic计时，wall_time仅时间戳。每个进程重启换clock_id。浏览器clock单独，不与板端直接相减；浏览器client_elapsed_ms可写入ACK，板端ACK延迟单列，不等于首音。
事件：turn_start/VAD结束、ASR/云端/改写/TTS开始结束、WS发送、audio_started/ended/failed、state/gate、拒绝/超时/resources。实际听到首音需H06测量，onplaying只代表浏览器播放启动；未测到不能填实测首音。
ASR_RTF=识别秒/输入音频秒，TTS_RTF=合成秒/输出音频秒。turn_sec=turn_start到audio_ended；stop_to_first_audio_sec只在同一时钟已正确记录两端点时定义。WS发送不是听到首音；整句不是流式首包。
缺失输出null，负时差/重复事件/多个speech_id给错误；样本乱序按事件名取端点。汇总保留失败/不完整样本数及每个指标缺失分母；p95至少20个观察样本。比较必须相同device/version/backend/cold/input_manifest/seed/n_planned，不能丢失败样本。
白名单允许CPU%、RSS、threads、queue_depth、npu_watts；异常NPU写null/missing，不用0W证明节能。默认不收原始语音/完整个人文本/密钥。
P13接入：所有任务通过EventLog.add挂点，模型线程起止由实际线程记录，客户端时钟另组；本次仅虚构fixture，不采真实性能/CPU/NPU。
