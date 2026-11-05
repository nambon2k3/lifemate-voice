"""Transcribe a sample wav from a cached text file.

Reads audio from the active split. A cache miss is sent to the ASR server
on vLLM. Writes one UTF-8 file per wav under
cache/transcripts/, named ``<sample_id>__<wav stem>.txt``. On train, the gold
``audio_text`` field can stand in for the latest request so planning can be
scored apart from speech errors.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASR_URL = "http://localhost:2043/v1/chat/completions"
_ASR_TEXT_TAG = "<asr_text>"


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


def write_cache(path: Path, text: str) -> None:
    """Store a transcript so the next run skips the speech model.

    Input:
        path (Path): destination from ``cache_path``. Parent directories are created.
        text (str): transcript to save as UTF-8.

    Output:
        None. Writes ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def audio_file(wav_path: str, settings: dict) -> Path:
    """Resolve a split-relative wav to a file on disk.

    Input:
        wav_path (str): path relative to the split, such as
        ``audio/public_0002/public_0002.wav``. An existing absolute path is kept.
        settings (dict): ``kit_root`` and ``split`` locate the split directory.

    Output:
        Path: the wav file. It may not exist yet; the caller checks.
    """
    path = Path(wav_path)
    if path.is_file():
        return path
    kit_root = REPO_ROOT / settings.get("kit_root", "train+public_test")
    return kit_root / settings.get("split", "") / wav_path


def parse_transcript(raw: str) -> str:
    """Pull the transcript out of an ASR completion.

    Splits on ``<asr_text>`` and keeps the text after the tag.

    Input:
        raw (str): ``choices[0].message.content`` from the ASR server.

    Output:
        str: transcript text, stripped. Empty when the server returned nothing.
    """
    text = "" if raw is None else str(raw)
    if _ASR_TEXT_TAG in text:
        text = text.split(_ASR_TEXT_TAG, 1)[1]
    return text.strip()


def call_asr(audio_path: Path, settings: dict) -> str:
    """Send one wav to the Qwen3-ASR server and return its text.

    Input:
        audio_path (Path): local wav file.
        settings (dict): optional ``asr_url`` (str, default
        ``http://localhost:2043/v1/chat/completions``) and ``asr_language``
        (str, default ``Vietnamese``). An empty ``asr_language`` leaves
        language detection to the server.

    Output:
        str: transcript text.
    """
    mime_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
    audio_base64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
    content: list[dict] = [
        {
            "type": "audio_url",
            "audio_url": {"url": f"data:{mime_type};base64,{audio_base64}"},
        }
    ]
    language = settings.get("asr_language", "Vietnamese")
    if language:
        content.append(
            {"type": "text", "text": f"Please transcribe the audio in {language}."}
        )
    response = requests.post(
        settings.get("asr_url", DEFAULT_ASR_URL),
        json={
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 4096,
        },
        timeout=300,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    return parse_transcript(raw)


def transcribe(sample: dict, wav_path: str, settings: dict) -> str:
    """Return the transcript for one wav.

    Tries the on-disk cache first. On train, uses ``audio_text`` when
    ``use_gold_transcript`` is set and ``wav_path`` is the latest request.
    Otherwise calls the ASR server and writes the cache.

    Input:
        sample (dict): one JSONL row. Uses ``id``, ``audio``, and maybe ``audio_text``.
        wav_path (str): wav to transcribe, relative to the split directory.
        settings (dict): must include ``cache_dir`` (str). ``use_gold_transcript``
        (bool), ``asr_url`` (str), and ``asr_language`` (str) are optional.

    Output:
        str: the transcript.
    """
    cache_dir = Path(settings["cache_dir"])
    path = cache_path(cache_dir, sample["id"], wav_path)
    cached = read_cache(path)
    if cached is not None:
        return cached

    if settings.get("use_gold_transcript") and wav_path == sample.get("audio"):
        gold = sample.get("audio_text")
        if isinstance(gold, str):
            return gold

    audio_path = audio_file(wav_path, settings)
    if not audio_path.is_file():
        raise FileNotFoundError(f"missing wav for {sample['id']}: {audio_path}")
    text = call_asr(audio_path, settings)
    write_cache(path, text)
    return text
