"""Transcribe a sample wav from a cached text file.

Reads one UTF-8 file per wav under cache/transcripts/, named
``<sample_id>__<wav stem>.txt``. On train, the gold ``audio_text`` field can
stand in for the latest request so planning can be scored apart from speech errors.
"""

from __future__ import annotations

from pathlib import Path


def cache_path(cache_dir: Path, sample_id: str, wav_path: str) -> Path:
    """Build the cache file path for one wav.

    Input:
        cache_dir (Path): directory such as cache/transcripts.
        sample_id (str): sample id, such as ``public_0002``.
        wav_path (str): audio path relative to the split, such as ``audio/public_0002/public_0002.wav``.

    Output:
        Path: ``<cache_dir>/<sample_id>__<wav stem>.txt``.
    """
    stem = Path(wav_path).stem
    return cache_dir / f"{sample_id}__{stem}.txt"


def read_cache(path: Path) -> str | None:
    """Read a cached transcript if it exists.

    Input:
        path (Path): file returned by ``cache_path``.

    Output:
        str: transcript text when the file exists.
        None: when the file is missing.
    """
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def transcribe(sample: dict, wav_path: str, settings: dict) -> str:
    """Return the transcript for one wav.

    Tries the on-disk cache first. On train, uses ``audio_text`` when
    ``use_gold_transcript`` is set and ``wav_path`` is the latest request.
    A missing cache raises ``NotImplementedError`` until PhoWhisper is wired in.

    Input:
        sample (dict): one JSONL row. Uses ``id``, ``audio``, and maybe ``audio_text``.
        wav_path (str): wav to transcribe, relative to the split directory.
        settings (dict): must include ``cache_dir`` (str). ``use_gold_transcript`` (bool) is optional.

    Output:
        str: the transcript.
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
