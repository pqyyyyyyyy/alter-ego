"""日语敬体→常体变换引擎（自研）。

## 为什么用规则而不是模型

实测（见 docs/实验数据.md）：
  Qwen2.5-0.5B  日文语尾命中 15%，实际可用率极低
  Sarashina-0.5B 语尾命中 75%，但**人工判定可用率仅 35%** —— 混英文 6 条、抄示例 3 条、日语崩坏 2 条
  Sarashina-1B   语尾命中 60%，**可用率仅 10%** —— 20 条里 11 条直接复述 few-shot 示例

三轮 prompt 调整（v1→v2→v2.1）未能突破 35%。而失败模式有共同特征：
**它们都不是"理解"问题，是生成不可控**。

日语敬体→常体恰好是**机械的形态变换**，查表加词形还原即可完成，
确定性 100%：不会混英文、不会抄示例、不会语法崩坏。

## 分工

  规则引擎（本文件）  → 语尾变换，确定性
  小模型             → 只做情绪分类（Sarashina-1B 实测 20/20 准确）
  云端               → 内容生成

模型输出从约 30 token 降到 3 token，板子上改写耗时从 2.4-3.0 s 降到 0.3-0.4 s。

## 依赖

janome —— 纯 Python 日语分词器，无 C 扩展，aarch64 可直接 pip 安装。
用它拿动词基本形（如 分かり → 分かる），这是 ます形还原的关键。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from janome.tokenizer import Token

# ==================== 语尾风格配置 ====================


@dataclass
class EndingStyle:
    """角色的语尾偏好。由人格卡的 style.sentence_endings 驱动。"""

    # 陈述句句尾：です/ます 体转成常体后追加什么
    declarative: list[str] = field(default_factory=lambda: ["よ", "ね"])
    # 疑问句句尾
    interrogative: list[str] = field(default_factory=lambda: ["かな", "の"])
    # 意志/劝诱（ましょう）
    volitional: list[str] = field(default_factory=lambda: ["か"])
    # 犹豫时的句首填充词
    fillers: list[str] = field(default_factory=lambda: ["うーん", "えっと"])
    # 是否使用「だ」。女性角色通常省略（晴れだね → 晴れね 也可，但保留 だね 更中性）
    use_da: bool = True

    @classmethod
    def from_card(cls, endings: list[str], fillers: list[str]) -> EndingStyle:
        """从人格卡的语尾列表推断各类句型该用什么。

        人格卡只给一个扁平的语尾列表（如 [かな, だよ, ね, よ]），
        这里按日语语法把它们分派到陈述/疑问/意志三类。
        """
        if not endings:
            return cls(fillers=fillers or ["うーん", "えっと"])

        # 「かも」是七海人格卡里的不确定语气词，属于陈述句语尾，
        # 不能因为它既不是疑问词、又不在旧的白名单里而被静默丢弃。
        # 同时接受常见的连用写法「かもね」。
        interro = [e for e in endings if e in ("かな", "の", "か", "かしら")]
        decl = [e for e in endings
                if e not in interro and e in (
                    "よ", "ね", "な", "だよ", "だね", "さ", "かも", "かもね",
                )]
        # 归一化：だよ/だね 里的 だ 由变换逻辑补，这里只留终助词部分
        decl = [e[1:] if e.startswith("だ") and len(e) > 1 else e for e in decl]
        decl = list(dict.fromkeys(decl)) or ["よ"]

        return cls(
            declarative=decl,
            interrogative=interro or ["かな"],
            volitional=["か"],
            fillers=fillers or ["うーん", "えっと"],
        )


# ==================== 词形变换表 ====================

# 一段动词、サ変、カ変 的 ます形还原无需五段规则，janome 的 base_form 直接可用。
# 这里处理的是助动词层面的变换。

# 形容词的敬体→常体
_ADJ_POLITE = [
    (r"くありませんでした$", "くなかった"),
    (r"くありません$", "くない"),
    (r"かったです$", "かった"),
    (r"いです$", "い"),
]

# 名词/形容动词 + です
_DESU = [
    (r"ではありませんでした$", "じゃなかった"),
    (r"ではありません$", "じゃない"),
    (r"じゃありませんでした$", "じゃなかった"),
    (r"じゃありません$", "じゃない"),
    (r"でした$", "だった"),
    (r"でしょう$", "だろう"),
    (r"です$", "だ"),
]

# 句末礼貌表达的直接替换（优先级最高，先匹配这些）
_FIXED = [
    # 「〜てください」的 て 已在动词里，只删 ください。
    # 曾误写成 (r"ください$", "て") 导致「取ってください」→「取っててよ」
    (r"てください$", "て"),
    (r"でください$", "で"),
    (r"下さい$", ""),
    (r"ありがとうございます$", "ありがとう"),
    (r"すみませんでした$", "ごめんね"),
    (r"すみません$", "ごめん"),
    (r"おはようございます$", "おはよう"),
    (r"こんにちは$", "こんにちは"),
    (r"いただけますか$", "くれる"),
    (r"いただけませんか$", "くれない"),
]


@lru_cache(maxsize=1)
def _tokenizer():
    """janome Tokenizer 初始化约 1-2 秒，缓存复用。"""
    from janome.tokenizer import Tokenizer
    return Tokenizer()


def _is_question(text: str) -> bool:
    return text.rstrip().endswith(("か", "か。", "か？", "？", "?")) or "ですか" in text


def _strip_terminal(text: str) -> tuple[str, str]:
    """剥离句末标点，返回 (正文, 标点)。"""
    m = re.search(r"([。．\.！!？\?…]+)\s*$", text)
    if m:
        return text[: m.start()], m.group(1)
    return text, ""


# ==================== ます形还原 ====================
# janome 把「分かりません」切成 分かり(動詞,基本形=分かる) + ませ(助動詞) + ん
# 所以只要识别出尾部的助动词串，再用动词基本形重建即可。

def _plain_from_masu(stem_base: str, aux: str) -> str | None:
    """由动词基本形与助动词串推出常体形。

    aux 是 ます 之后的助动词拼接，如 "ます" "ません" "ました" "ましょう"。
    返回 None 表示不认识这个模式，交由上层降级处理。
    """
    base = stem_base
    if aux in ("ます",):
        return base                                    # 待ちます → 待つ
    if aux in ("ません",):
        return _negative(base)                         # 分かりません → 分からない
    if aux in ("ました",):
        return _past(base)                             # 行きました → 行った
    if aux in ("ませんでした",):
        neg = _negative(base)
        return neg[:-1] + "かった" if neg.endswith("い") else neg
    if aux in ("ましょう", "ましょ"):
        return _volitional(base)                       # しましょう → しよう
    if aux in ("ますか",):
        return base
    return None


_GODAN_NEG = {"う": "わ", "く": "か", "ぐ": "が", "す": "さ", "つ": "た",
              "ぬ": "な", "ぶ": "ば", "む": "ま", "る": "ら"}
_GODAN_PAST = {"う": "った", "つ": "った", "る": "った",
               "く": "いた", "ぐ": "いだ",
               "す": "した",
               "ぬ": "んだ", "ぶ": "んだ", "む": "んだ"}
_ICHIDAN_HINT = ("見る", "食べる", "寝る", "起きる", "出る", "着る", "いる",
                 "できる", "覚える", "考える", "教える", "答える", "疲れる",
                 "休める", "飲める", "忘れる", "分かれる", "入れる", "晴れる")


def _is_ichidan(base: str) -> bool:
    """判断一段动词。启发式：janome 不直接给活用型，用词尾+常用词表近似。"""
    if base in _ICHIDAN_HINT:
        return True
    if not base.endswith("る") or len(base) < 2:
        return False
    # 一段动词倒数第二个假名在 え/い 段
    return base[-2] in "えけげせぜてでねへべめれいきぎしじちぢにひびみり"


def _negative(base: str) -> str:
    if base == "する":
        return "しない"
    if base in ("来る", "くる"):
        return "来ない"
    if base == "ある":
        return "ない"
    if _is_ichidan(base):
        return base[:-1] + "ない"
    tail = base[-1]
    if tail in _GODAN_NEG:
        return base[:-1] + _GODAN_NEG[tail] + "ない"
    return base + "ない"


def _past(base: str) -> str:
    if base == "する":
        return "した"
    if base in ("来る", "くる"):
        return "来た"
    if base == "行く":
        return "行った"            # 不规则
    if _is_ichidan(base):
        return base[:-1] + "た"
    tail = base[-1]
    if tail in _GODAN_PAST:
        return base[:-1] + _GODAN_PAST[tail]
    return base + "た"


def _volitional(base: str) -> str:
    if base == "する":
        return "しよう"
    if base in ("来る", "くる"):
        return "来よう"
    if _is_ichidan(base):
        return base[:-1] + "よう"
    tail = base[-1]
    vol = {"う": "おう", "く": "こう", "ぐ": "ごう", "す": "そう", "つ": "とう",
           "ぬ": "のう", "ぶ": "ぼう", "む": "もう", "る": "ろう"}
    if tail in vol:
        return base[:-1] + vol[tail]
    return base + "よう"


# ==================== 主变换 ====================

# 助动词表面形。注意必须包含「まし」——janome 把「ました」切成
# まし(助動詞) + た(助動詞)，漏了它会导致过去形全部识别失败。
_AUX_SET = {"ます", "まし", "ませ", "ましょ", "ん", "た", "でし", "でした",
            "う", "です", "だ", "か"}


@dataclass
class Result:
    """变换结果。coverage 用于统计规则命中率。"""
    text: str
    changed: bool              # 是否发生了变换
    kind: str                  # masu / desu / adj / fixed / none
    is_question: bool


def to_plain(text: str) -> Result:
    """敬体 → 常体。只做形态变换，不加语尾。"""
    body, punct = _strip_terminal(text)
    if not body:
        return Result(text, False, "none", False)

    q = _is_question(body) or punct in ("？", "?")
    # 疑问句的「か」在转常体时通常脱落
    body_nq = re.sub(r"か$", "", body) if q else body

    # 先剥离原文自带的终助词（ね/よ/な/わ）——
    # 否则「面白いですね」会被当成不认识的模式，最后变成「面白いですねよ」。
    # 剥掉之后走正常变换，角色语尾由 apply_style 统一追加。
    body_nq = re.sub(r"(ね|よ|な|わ|ねえ|よね)$", "", body_nq) or body_nq

    # 1. 固定表达优先
    for pat, rep in _FIXED:
        if re.search(pat, body_nq):
            return Result(re.sub(pat, rep, body_nq) + punct, True, "fixed", q)

    # 2. ます形 —— 交给 janome
    # 「まし」必须在列：ました 是 ま+し+た，不含 ます/ませ/ましょ 任何一个，
    # 漏掉它会让所有过去肯定形走不到变换逻辑（实测 6/28 全栽在这）
    if re.search(r"(ます|まし|ませ|ましょ)", body_nq):
        out = _convert_masu(body_nq)
        if out is not None:
            return Result(out + punct, True, "masu", q)

    # 3. 形容词 + です
    for pat, rep in _ADJ_POLITE:
        if re.search(pat, body_nq):
            return Result(re.sub(pat, rep, body_nq) + punct, True, "adj", q)

    # 4. 名词/形容动词 + です
    for pat, rep in _DESU:
        if re.search(pat, body_nq):
            return Result(re.sub(pat, rep, body_nq) + punct, True, "desu", q)

    return Result(body_nq + punct, False, "none", q)


# ==================== 情绪关键词兜底 ====================
# 情绪主路径是云端模型的结构化输出 —— 它最清楚自己的情绪意图，
# 且零额外成本。本表仅用于断网降级时的本地兜底。
#
# 为何不用本地小模型做分类：Sarashina-1B 实测仅 7/12（58%），
# 且失败模式是全部倒向 neutral（「会えてうれしいです」也判 neutral）。
# 日语情绪标记词相当显性，规则覆盖率反而更高，且省掉 0.22 s 推理。

_EMO_RULES: list[tuple[str, tuple[str, ...]]] = [
    # shy 置顶：「褒め」「照れ」是比泛泛道谢更强的信号。
    # 实测「ほめてくれてありがとう」若让 happy 的「ありがと」先命中会判错。
    ("shy", ("ほめ", "褒め", "照れ", "恥ずかし", "はずかし", "緊張")),
    ("happy", ("うれしい", "嬉しい", "楽しい", "たのしい", "よかった", "最高",
               "ありがと", "おいしい", "好き", "すごい", "やった", "いい天気")),
    # assertive 必须在 unhappy 之前。「違う」「だめ」这类否定既可能是不悦，
    # 也可能是据理力争，而七海在学级裁判里的「それは違う！」是后者 ——
    # 指出错误是主张，不是生气。放在前面让更具体的信号先命中。
    ("assertive", ("違う", "ちがう", "そうじゃない", "待って", "まって",
                   "聞いて", "きいて", "絶対", "ぜったい", "はずだ", "に決まって",
                   "思うよ", "だと思う", "確か")),
    # unhappy 取代原先的 angry 与 sad。两者的关键词都归到这里 ——
    # 素材里没有哭也没有怒，只有「不悦」这一档，映射必须如实反映可表现的范围。
    # 代价是「疲れた」和「ひどい」会得到同一个表情，但那比指向一个不存在的立绘好。
    ("unhappy", ("疲れ", "つかれ", "さびしい", "寂しい", "悲しい", "かなしい",
                 "残念", "つらい", "大変", "しんどい", "ごめん", "すみません",
                 "我慢できない", "我慢できません", "ひどい", "だめ", "ダメ",
                 "許せない", "いやだ", "嫌だ", "ふざけ")),
    ("surprised", ("どうして", "なんで", "なぜ", "本当", "ほんとう", "まさか",
                   "えっ", "え、", "びっくり", "意外", "知らなかった")),
]


def guess_emotion(text: str) -> str:
    """关键词兜底情绪判定。无命中返回 neutral。

    疑问句优先判 surprised 之前先查 happy/sad —— 「楽しいですか？」
    这类句子的情绪主体是形容词而非疑问语气。
    """
    for emo, kws in _EMO_RULES:
        if any(k in text for k in kws):
            return emo
    if text.rstrip().endswith(("？", "?", "か", "か。")):
        return "surprised"
    return "neutral"


def _normalize_aux(aux: str) -> str:
    """把 janome 切碎的助动词串还原成规范形。

    janome 的切分方式：
        ました      → まし + た
        ません      → ませ + ん
        ませんでした → ませ + ん + でし + た
        ましょう    → ましょ + う
    """
    a = aux.replace(" ", "")
    if a.startswith("ませ") and "でし" in a:
        return "ませんでした"
    if a.startswith("ませ"):
        return "ません"
    if a.startswith("ましょ"):
        return "ましょう"
    if a.startswith("まし"):
        return "ました"
    if a.startswith("ます"):
        return "ます"
    return a


def _convert_masu(body: str) -> str | None:
    """用 janome 定位动词 + 助动词串，重建常体。"""
    tokens: list[Token] = list(_tokenizer().tokenize(body))
    if not tokens:
        return None

    # 从尾部收集助动词串
    i = len(tokens) - 1
    aux_parts: list[str] = []
    while i >= 0:
        pos = tokens[i].part_of_speech.split(",")[0]
        if pos in ("助動詞", "動詞") and tokens[i].surface in _AUX_SET:
            aux_parts.insert(0, tokens[i].surface)
            i -= 1
        elif pos == "助詞" and tokens[i].surface == "か" and aux_parts:
            i -= 1          # 疑问助词已在上层剥离
        else:
            break

    if i < 0 or not aux_parts:
        return None

    verb = tokens[i]
    if verb.part_of_speech.split(",")[0] != "動詞":
        return None

    # 复合动词：「降り始めました」切成 降り(自立) + 始め(非自立) + まし + た。
    # 活用发生在最后那个动词上（始める），前面的连用形原样保留，
    # 所以 verb 取 tokens[i] 即可，但要把更前面的动词也算进前缀。
    # 这里 i 已指向「始め」，prefix 会自然包含「降り」，无需额外处理。

    # 归一化助动词串。janome 把「ました」切成 まし+た、
    # 「ませんでした」切成 ませ+ん+でし+た，需要还原成规范形。
    aux = "".join(aux_parts)
    aux = _normalize_aux(aux)

    plain = _plain_from_masu(verb.base_form, aux)
    if plain is None:
        return None

    prefix = "".join(t.surface for t in tokens[:i])
    return prefix + plain


def apply_style(result: Result, style: EndingStyle) -> str:
    """在常体之上追加角色语尾。"""
    body, punct = _strip_terminal(result.text)
    if not body:
        return result.text

    if result.is_question:
        tail = style.interrogative[0] if style.interrogative else "かな"
        # 常体疑问句不需要「だ」：危険だ → 危険かな
        body = re.sub(r"だ$", "", body)
        out = body + tail
        punct = punct or "？"
        return out + ("？" if punct in ("。", "") else punct)

    tail = style.declarative[0] if style.declarative else "よ"
    if not style.use_da:
        body = re.sub(r"だ$", "", body)
    out = body + tail
    return out + (punct or "。")


def rewrite(text: str, style: EndingStyle) -> tuple[str, str]:
    """对外入口：敬体日语 → 角色口吻。返回 (结果, 命中的规则类型)。"""
    r = to_plain(text)
    return apply_style(r, style), r.kind
