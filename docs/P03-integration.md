# P03 集成约定
独立SessionDialogs(factory)包装现有DialogOrchestrator，无需修改dialog公共API。factory每次返回新Orchestrator，max_turns保留6，不能返回单例。稳定user_id从perception身份传入，不用显示姓名；手动名字仅为snapshot.user_name展示，不能授权到其他档案。
注册用户用user:<id>；每次访客arrive生成独立guest:<uuid>，离开删除；容量16，空闲900s，LRU淘汰不在途项。不存磁盘/不承诺C4持久记忆。
P13在reserve时取得snapshot，通过to_thread调用reply(snapshot,text,ctx)。识别换人/leave递增epoch；结果为None必须丢弃TTS/字幕，不发送后来用户。旧内容仅留在原用户独立历史（reset会换对象）；统一P01保证实际模型串行。底层云端未完成时继续gate，绝不凭None返回逻辑提前启动新作业。
