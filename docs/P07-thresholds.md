# P07 阈值评估及真实采集要求
输入CSV：pair_id,similarity,label(match/nonmatch),subject_a,subject_b,distance,lighting,pose,device_id,model_version,split。pair_id唯一；拒绝未知标签/非有限/范围外相似度，任一正负分母缺失则不能算FAR/FRR。
match=同一人真匹配；nonmatch=不同人/未注册测试者与注册模板非匹配。不无限复制访客作真负、不让训练采集对泄漏到测试。FAR=nonmatch中score>=threshold比例；FRR=match中score<threshold比例，报实际分子分母。脚本只扫描，不自动选阈值；0.40未改。
真实H04至少多参与者、远近/姿态/光照、多会话和多人遮挡；采样计划、人数/配对数、授权、拆分方法由负责人填。仅传脱敏相似度表，不入库原图/512向量。合成向量和fixture仅证明计算与边界，不能证明识别率。
异常：5点关键点缺失跳过特征；512维/finite/非零验证失败跳过；投票平手及多人不明确时按访客；LEFT清身份。注册/删除持久化失败必须保留可重试的内存状态并抛可观测异常。详细回归见test_perception_pipeline.py、test_identity_store.py、test_perception_boundaries_v1.py。
