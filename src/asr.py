"""Transcribe a sample wav and cache the text.

Reads audio from the active split. Writes one UTF-8 file per wav under
cache/transcripts/, named ``<sample_id>__<wav stem>.txt``. On train, the gold
``audio_text`` field can stand in for the latest request so planning can be
scored apart from speech errors.
"""

from __future__ import annotations

from pathlib import Path


def cache_path(cache_dir: Path, sample_id: str, wav_path: str) -> Path:
    stem = Path(wav_path).stem
    return cache_dir / f"{sample_id}__{stem}.txt"


def read_cache(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def write_cache(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def transcribe(sample: dict, wav_path: str, settings: dict) -> str:
    """Return the transcript for ``wav_path``.

    Order: on-disk cache, then train ``audio_text`` when ``use_gold_transcript``
    is set and this wav is the latest request. Loading PhoWhisper is the next
    step; until then a missing cache raises ``NotImplementedError``.
    """
    cache_dir = Path(settings["cache_dir"])
    cached = read_cache(cache_path(cache_dir, sample["id"], wav_path))
    if cached is not None:
        return cached

    if settings.get("use_gold_transcript") and wav_path == sample.get("audio"):
        gold = sample.get("audio_text")
        if isinstance(gold, str):
            return gold

    raise NotImplementedError(
        f"no transcript for {sample['id']} ({wav_path}). "
        "Cache one under cache/transcripts/ or set use_gold_transcript on train."
    )
