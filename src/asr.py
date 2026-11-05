"""Transcribe a sample wav and cache the text.

Reads audio from the active split. A cache miss is sent to the ASR server
on vLLM. Writes one UTF-8 file per wav under
cache/transcripts/, named ``<sample_id>__<wav stem>.txt``. On train, the gold
``audio_text`` field can stand in for the latest request so planning can be
scored apart from speech errors.
"""

from __future__ import annotations

from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASR_URL = "http://localhost:2043/v1/audio/transcriptions"
_ASR_TEXT_TAG = "<asr_text>"
_LANG_CODES = {"vietnamese": "vi", "vi": "vi"}


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
    """Send one wav to the ASR server and return its text.

    PhoWhisper is served with the OpenAI transcription route, not chat
    completions. The decoder length cap is 448 tokens.

    Input:
        audio_path (Path): local wav file.
        settings (dict): optional         ``asr_url`` (str, default
        ``http://localhost:2043/v1/audio/transcriptions``), ``asr_served_model``
        (str, the vLLM served name, default ``qwen3-asr``), and
        ``asr_language`` (str, default ``Vietnamese``, sent as ``vi``).

    Output:
        str: transcript text.
    """
    language = str(settings.get("asr_language", "Vietnamese") or "")
    form = {
        "model": str(settings.get("asr_served_model") or "qwen3-asr"),
        "response_format": "json",
        "temperature": "0",
    }
    code = _LANG_CODES.get(language.strip().lower())
    if code:
        form["language"] = code
    with audio_path.open("rb") as handle:
        response = requests.post(
            settings.get("asr_url", DEFAULT_ASR_URL),
            data=form,
            files={"file": (audio_path.name, handle, "audio/wav")},
            timeout=300,
        )
    response.raise_for_status()
    body = response.json()
    raw = body.get("text") if isinstance(body, dict) else body
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
        (bool), ``asr_url`` (str), ``asr_served_model`` (str), and
        ``asr_language`` (str) are optional.

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
