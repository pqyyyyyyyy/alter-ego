# P02 接入及边界
默认：16k/mono/S16LE，20ms，200ms前置，60ms起始确认，400ms结束静音，100ms最短有效语音，最长30s，RMS 500（待H03校准，不是识别精度）。任意奇数块保留字节，EOF不完整帧报错；停止直接清空，不提交半句。注入vad(frame)->bool。
InputController在同一事件循环操作，音频源可替换，ALSA线程用call_soon_threadsafe送feed。队列1，超过丢弃并可观测，不积压；单个to_thread执行reset/feed/poll/reset，sink(text,sequence)为快速同步回调，由P13提交统一调度。不加载真实模型，不依赖麦克风。
每次gate变化epoch递增，丢弃队列及切句缓存；正在识别旧结果不提交。stop等待实际worker结束，不假装杀线程；识别器永久阻塞需H03进程级监督，不能启动第二个线程重叠识别。start在旧worker未结束时拒绝。能量阈值与模拟波形不能证明真实自然语音切句/CER。
