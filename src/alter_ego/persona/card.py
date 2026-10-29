"""人格卡：角色的说话方式规格。

设计意图 —— 人格卡是可替换的，换一个 YAML 就换一个角色。
这是「形象与引擎解耦」的实现：仓库里不含任何特定 IP 的人格，
用户自带角色包，自负授权责任。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..contracts import EMOTIONS


@dataclass
class RewriteParams:
    """改写的推理参数。默认值来自 PC 阶段实测（见 docs/实验数据.md）。"""
    temperature: float = 0.6      # v1 用 0.7 偏高导致编造，降到 0.6 更稳
    top_p: float = 0.9
    max_tokens: int = 48          # 实测平均输出 26 token，48 够用
    retry_on_invalid: int = 1     # 一致性自评失败后的重生成次数


def _parse_emotions(raw: dict) -> dict[str, list[str]]:
    """情绪映射归一成列表。

    兼容两种写法：
        happy: happy.png              → ["happy.png"]
        happy: [soft smile, cute]     → ["soft smile", "cute"]
    不带扩展名的自动补 .png —— YAML 里写文件名时带不带后缀都对。
    """
    out: dict[str, list[str]] = {}
    for emo, v in raw.items():
        items = [v] if isinstance(v, str) else list(v or [])
        out[str(emo)] = [
            s if str(s).lower().endswith(".png") else f"{s}.png"
            for s in items if s
        ]
    return out


def _parse_animations(raw: dict) -> dict[str, SpriteAnimation]:
    out: dict[str, SpriteAnimation] = {}
    for key, v in raw.items():
        if isinstance(v, (str, list)):          # 只给了帧列表的简写
            v = {"frames": [v] if isinstance(v, str) else v}
        frames = v.get("frames", []) or []
        out[str(key)] = SpriteAnimation(
            frames=[f if str(f).lower().endswith(".png") else f"{f}.png"
                    for f in frames],
            interval_ms=int(v.get("interval_ms", 1000)),
            loop=bool(v.get("loop", False)),
        )
    return out


@dataclass
class SpriteAnimation:
    """按状态播放的立绘序列。

    与情绪立绘的区别：情绪是**一个时刻的状态**（云端选一个，前端换一张图），
    动画是**一段时间的过程**（前端按 interval 逐帧播）。
    两者走不同的 WebSocket 消息，互不干扰。
    """
    frames: list[str] = field(default_factory=list)
    interval_ms: int = 1000
    loop: bool = False

    def to_dict(self) -> dict:
        """发给前端的形式。前端不做判断，只按这个播。"""
        return {"frames": self.frames, "interval_ms": self.interval_ms,
                "loop": self.loop}


@dataclass
class PersonaProfile:
    """角色设定：身份、背景、关系、知识边界。

    ⚠️ 这部分**只进云端主对话模型的 system prompt**，不进本地改写模型。

    原因：本地改写模型是 0.5B，注意力容量有限。实测教训 ——
    v1 把改写规则放在 prompt 中段，被 200 token 的示例冲淡，
    出现编造信息与人称翻转。往前面再堆几百 token 背景，改写质量必退步。
    而且前缀变长会加重冷启动（板子上已约 3 秒）。

    职责划分：云端决定「说什么」（需要背景），本地决定「怎么说」（只需风格）。
    """
    identity: str = ""                                  # 一句话身份
    background: str = ""                                # 背景故事，可多行
    personality: str = ""                               # 性格内核（非说话风格）
    relationship: str = ""                              # 与用户的关系设定
    knows: list[str] = field(default_factory=list)      # 熟悉的领域
    unknown: list[str] = field(default_factory=list)    # 明确不知道的事
    taboo: list[str] = field(default_factory=list)      # 不愿谈的话题
    # 角色原台词，给云端当语气参照。日文角色就写日文原句。
    #
    # 与 examples.jsonl（few-shot 池）的区别：池子只被**中文路径**使用
    # （rewriter._build_prefix 的 pool.render()），日文角色走 rewrite_ja()
    # 根本不碰池子。所以日文角色的语气参照必须放在这里。
    #
    # 量级 5-10 条就够 —— 它不是训练数据，只是给云端看「她这么说话」。
    # 任务书里「30-50 条」那个数字是给中文 few-shot 模型学风格映射用的。
    speech_samples: list[str] = field(default_factory=list)
    # 全部 provider 都失败时的预设回复。板子上网络抖动是常态，
    # 没有预设回复演示时会尴尬地卡住。日文角色写日文。
    fallback_reply: str = ""

    def is_empty(self) -> bool:
        return not any([
            self.identity, self.background, self.personality,
            self.relationship, self.knows, self.unknown, self.taboo,
            self.speech_samples,
        ])


@dataclass
class PersonaCard:
    name: str
    tone: str
    # 输出语言。"ja" 走规则引擎做语尾变换（确定性 100%），
    # "zh" 走模型改写。见 docs/实验数据.md 的选型对照。
    language: str = "zh"
    sentence_endings: list[str] = field(default_factory=list)
    fillers: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    self_reference: str = "我"
    address_user: str = "你"
    # 情绪 → 候选立绘列表。一个情绪可以有多张图，本地轮换，视觉上不单调。
    # 这是「不靠扩枚举提升丰富度」的实现 —— 枚举要小（云端才挑得准），
    # 表现力靠本地选图。YAML 里写单个字符串也兼容，load 时统一成列表。
    emotions: dict[str, list[str]] = field(default_factory=dict)
    # 状态动画。**不占情绪枚举** —— state 和 emotion 是两条独立的 WebSocket
    # 消息，所以「待机时睡着」「唤醒时打哈欠」这类时序表现不必塞进 Emotion。
    animations: dict[str, SpriteAnimation] = field(default_factory=dict)
    params: RewriteParams = field(default_factory=RewriteParams)
    profile: PersonaProfile = field(default_factory=PersonaProfile)

    @classmethod
    def load(cls, path: str | Path) -> PersonaCard:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        style = d.get("style", {}) or {}
        rw = d.get("rewrite", {}) or {}
        pf = d.get("profile", {}) or {}

        card = cls(
            name=d.get("name", "未命名"),
            tone=style.get("tone", ""),
            language=d.get("language", "zh"),
            sentence_endings=style.get("sentence_endings", []) or [],
            fillers=style.get("fillers", []) or [],
            forbidden=style.get("forbidden", []) or [],
            self_reference=style.get("self_reference", "我"),
            address_user=style.get("address_user", "你"),
            emotions=_parse_emotions(d.get("emotions", {}) or {}),
            animations=_parse_animations(d.get("animations", {}) or {}),
            params=RewriteParams(
                temperature=rw.get("temperature", 0.6),
                top_p=rw.get("top_p", 0.9),
                max_tokens=rw.get("max_tokens", 48),
                retry_on_invalid=rw.get("retry_on_invalid", 1),
            ),
            profile=PersonaProfile(
                identity=pf.get("identity", ""),
                background=pf.get("background", ""),
                personality=pf.get("personality", ""),
                relationship=pf.get("relationship", ""),
                knows=pf.get("knows", []) or [],
                unknown=pf.get("unknown", []) or [],
                taboo=pf.get("taboo", []) or [],
                speech_samples=pf.get("speech_samples", []) or [],
                fallback_reply=pf.get("fallback_reply", ""),
            ),
        )
        return card

    def output_spec(self) -> str:
        """云端的输出格式规格。按 language 分支。

        与 system_prompt 拆开，是因为两条语言路径的规格根本不同：
          · ja：云端出**敬体**日文 + 中文字幕 + 情绪，本地规则引擎做语尾变换
          · zh：云端出中性中文 + 情绪，本地 0.5B 模型做口吻改写

        **`ja` 必须是敬体，这是整条链路成立的前提。** 规则引擎只认 です・ます；
        给它常体会静默失败（门禁不命中但仍拼终助词），表现为「改写没生效」。
        """
        enum = " / ".join(EMOTIONS)
        if self.language == "ja":
            return (
                "\n【输出格式】\n"
                "只输出一个 JSON 对象，不要任何额外说明，不要代码围栏：\n"
                '{"ja": "日语回答", "zh": "中文翻译", "emotion": "情绪"}\n\n'
                "三个字段的要求：\n"
                "- ja：用日语回答。**必须用敬体（です・ます体），不要用常体。**\n"
                "  你的语尾会由后续模块按角色设定改写，所以这里只需标准敬体。\n"
                "- zh：ja 的中文翻译。意思要严格对应，不要自己另写一句。\n"
                f"- emotion：只能是这六个英文词之一：{enum}\n"
            )
        return (
            "\n【输出格式】\n"
            "只输出一个 JSON 对象，不要任何额外说明，不要代码围栏：\n"
            '{"text": "回答内容", "emotion": "情绪"}\n\n'
            "- text：用中文回答，语气中性即可，口吻会由后续模块改写。\n"
            f"- emotion：只能是这六个英文词之一：{enum}\n"
        )

    def system_prompt(self, user_name: str | None = None) -> str:
        """渲染云端主对话模型的 system prompt。

        只有这里用 profile —— 本地改写模型永远看不到这些内容。

        ⚠️ 这段必须对同一角色逐字节恒定（`user_name` 除外），
        感知上下文等易变信息应拼进 user message 而不是这里 ——
        前缀一变云端的 prompt 缓存失效（本地实测省 92% prompt 处理耗时）。
        """
        p = self.profile
        parts = [f"你是{self.name}。"]
        if p.identity:
            parts.append(p.identity)
        if p.background:
            parts.append(f"\n【你的经历】\n{p.background}")
        if p.personality:
            parts.append(f"\n【你的性格】\n{p.personality}")
        if p.relationship:
            who = user_name or "对方"
            parts.append(f"\n【你和{who}的关系】\n{p.relationship}")
        if p.knows:
            parts.append(f"\n你熟悉：{'、'.join(p.knows)}")
        if p.unknown:
            # 知识边界对沉浸感很关键 —— 角色不该知道世界观之外的事
            parts.append(
                f"你不知道这些，被问到就如实表示不了解："
                f"{'、'.join(p.unknown)}"
            )
        if p.taboo:
            parts.append(f"你不愿谈：{'、'.join(p.taboo)}，会自然地回避或转移话题。")
        if p.speech_samples:
            # 「不要照搬」是实测过的失败模式：日文选型实验里 1B 有 11/20 条
            # 直接复述示例内容（见 docs/实验数据.md）。云端模型能力强得多，
            # 但这句约束成本为零，加上更稳。
            samples = "\n".join(f"- {s}" for s in p.speech_samples)
            parts.append(
                f"\n【你的说话风格参考】\n{samples}\n"
                "这些是你说过的话。模仿这种语气和用词习惯，但不要照搬内容。"
            )
        parts.append(
            "\n用第一人称自然对话。回答简短口语化，一两句话为宜 —— "
            "你的回答会被转成语音播放，长篇大论不适合听。"
        )
        parts.append(self.output_spec())
        return "\n".join(parts)

    def validate(self, sprite_dir: str | Path | None = None) -> list[str]:
        """返回警告列表（不抛异常 —— 缺字段降级运行比启动失败好）。

        传 `sprite_dir` 会额外检查文件是否真的存在。**这个检查很重要** ——
        映射里写错文件名不会报错，只会让该情绪静默兜底成 neutral，
        表现为「某个表情从来不出现」，而你会先去怀疑情绪判定。
        """
        warns: list[str] = []
        missing = [e for e in EMOTIONS if e not in self.emotions]
        if missing:
            warns.append(f"人格卡缺少情绪立绘映射: {', '.join(missing)}")
        extra = [e for e in self.emotions if e not in EMOTIONS]
        if extra:
            warns.append(
                f"人格卡里有枚举外的情绪 {', '.join(extra)} —— "
                "云端永远不会输出它们，这些立绘不会被用到"
            )
        if not self.sentence_endings:
            warns.append("未指定句尾语气词，改写风格会偏弱")
        if self.language == "ja" and not self.profile.speech_samples:
            # 日文角色不走 few-shot 池（rewrite_ja 不碰 pool），
            # 语气参照只能来自 speech_samples。缺了不报错，
            # 只会让云端输出通用日语，角色感丢失。
            warns.append(
                "日文角色未填 profile.speech_samples —— 云端会输出通用日语，"
                "角色感丢失。日文路径不使用 examples.jsonl，语气参照只能放这里"
            )
        if not self.profile.fallback_reply:
            warns.append("未填 profile.fallback_reply，云端全失败时会没话可说")

        if sprite_dir is not None:
            d = Path(sprite_dir)
            if not d.is_dir():
                warns.append(f"立绘目录不存在: {d}")
            else:
                refs = {f for files in self.emotions.values() for f in files}
                refs |= {f for a in self.animations.values() for f in a.frames}
                absent = sorted(f for f in refs if not (d / f).exists())
                if absent:
                    warns.append(f"立绘文件缺失: {', '.join(absent)}")
        return warns

    def style_prompt(self) -> str:
        """渲染成 prompt 片段。

        注意：此输出进入 prompt 的「固定前缀」区，必须对同一角色保持稳定，
        否则 llama.cpp 的前缀缓存失效 —— 实测缓存能省 92% 的 prompt 处理耗时。
        """
        lines = []
        if self.tone:
            lines.append(f"- {self.tone}")
        if self.sentence_endings:
            lines.append(f"- 句尾爱用 {' / '.join(self.sentence_endings)}")
        if self.fillers:
            lines.append(f"- 犹豫时开头说 {' / '.join(self.fillers)}")
        return "\n".join(lines)

    def rules_prompt(self) -> str:
        """改写规则。置于示例之后、原句之前。

        位置是实测结论：小模型对靠近 prompt 结尾的指令明显更敏感。
        v1 把规则放在中段，被 200 token 的示例冲淡，出现编造信息和人称翻转。
        """
        rules = [
            "1. 只改说话方式，不改意思",
            f"2.「{self.address_user}」和「{self.self_reference}」不能互换",
            "3. 不添加原句里没有的信息",
            "4. 语气词只放句尾，不能放句首",
            "5. 不要照搬上面示例的内容",
        ]
        if self.forbidden:
            rules.append(f"6. 禁止出现：{'、'.join(self.forbidden)}")
        return "\n".join(rules)

    def sprites_for(self, emotion: str) -> list[str]:
        """情绪 → 候选立绘列表。未映射的情绪兜底到 neutral。"""
        got = self.emotions.get(emotion)
        if got:
            return got
        return self.emotions.get("neutral") or ["neutral.png"]

    def sprite_for(self, emotion: str, index: int = 0) -> str:
        """情绪 → 一张立绘文件名。

        `index` 由调用方递增以轮换候选图（同一情绪连续出现时不总是同一张）。
        取模而非越界兜底 —— 越界返回 neutral 会让情绪静默丢失。
        """
        cands = self.sprites_for(emotion)
        return cands[index % len(cands)]

    def animation(self, key: str) -> SpriteAnimation | None:
        return self.animations.get(key)
