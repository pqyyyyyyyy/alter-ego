"""Conservative Japanese output heuristic, not translation or expert validation."""
import re
KANJI_ONLY = frozenset(('了解','承知','感謝','大丈夫','問題無'))
SAFE_JA = 'すみません。今は答えられません。'
def japanese_error(text, max_chars=2000):
    if not isinstance(text,str): return 'non_string'
    text=text.strip()
    if not text:return 'empty'
    if len(text)>max_chars:return 'too_long'
    if any(0xD800<=ord(c)<=0xDFFF or (ord(c)<32 and c not in '\n\t') for c in text):
        return 'invalid_unicode_or_control'
    if re.search(r'[ぁ-ゖァ-ヺ]',text):return ''
    if text.rstrip('。！？.!? ') in KANJI_ONLY:return ''
    return 'language_unconfirmed'
