"""自动转写初稿不能覆盖人工校对，也不能替代人工批准。"""
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/transcribe_voice_dataset.py"
SPEC = importlib.util.spec_from_file_location("transcribe_voice_dataset", SCRIPT)
transcription = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transcription)


def test_draft_fills_blank_but_never_approves():
    row = {"text": "", "approved": False}
    result = transcription.merge_draft(row, {"text": "こんにちは", "error": ""})
    assert result["text"] == "こんにちは"
    assert result["approved"] is False
    assert row["text"] == ""


def test_human_edits_and_explicit_clear_survive():
    for row in ({"text": "手動修正", "approved": True},
                {"text": "", "text_edited": True, "approved": False},
                {"text": "未批准但已修改", "approved": False}):
        result = transcription.merge_draft(row, {"text": "ASR", "error": ""})
        assert result["text"] == row["text"]
        assert result["approved"] == row["approved"]


def test_failed_transcription_preserves_previous_draft():
    row = {"text": "旧初稿", "approved": False, "asr": {"text": "旧初稿"}}
    assert transcription.merge_draft(row, {"error": "decode failed"}) == row


def test_model_identity_changes_when_weights_change(tmp_path):
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (tmp_path / name).write_bytes(b"first")
    initial = transcription.model_identity(tmp_path)
    (tmp_path / "model.bin").write_bytes(b"second")
    assert transcription.model_identity(tmp_path) != initial
