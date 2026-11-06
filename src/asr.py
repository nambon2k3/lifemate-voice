"""Transcribe a sample wav from a cached text file.

Reads audio from the active split. A cache miss is sent to the ASR server
on vLLM. Writes one UTF-8 file per wav under
cache/transcripts/, named ``<sample_id>__<wav stem>.txt``. On train, the gold
``audio_text`` field can stand in for the latest request so planning can be
scored apart from speech errors.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

import requests

from env import env_value

REPO_ROOT = Path(__file__).resolve().parents[1]
_ASR_TEXT_TAG = "<asr_text>"


_DIGIT_WORD = frozenset({
    "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín",
    "mươi", "mười", "trăm", "ngàn", "nghìn", "linh", "lẻ", "mốt", "lăm",
})
_CHO_SO = ("cho", "số")
_MIN_PHONE_WORDS = 8


def asr_instruction(language: str) -> str:
    """Build the digit-word context sent with a second transcription pass.

    The list names every digit, and says ``không`` is the digit 0. No contact
    names are included. ``language`` is sent as its own field.

    Input:
        language (str): language name, such as ``Vietnamese``. Unused here.

    Output:
        str: the context string.
    """
    del language
    return "không một hai ba bốn năm sáu bảy tám chín. không là chữ số 0."


def asr_language_code(language: str) -> str:
    """Map a language name to the code the transcription endpoint accepts.

    Input:
        language (str): name or code, such as ``Vietnamese``.

    Output:
        str: code such as ``vi``. An unknown value is returned unchanged.
    """
    return {"vietnamese": "vi"}.get(language.strip().casefold(), language.strip())


def served_model(settings: dict) -> str:
    """Return the model name the ASR server actually exposes.

    ``asr_model`` in settings is the old checkpoint id. The request must use
    the served name, such as ``qwen3-asr``.

    Input:
        settings (dict): optional ``asr_served_model``.

    Output:
        str: the model name to send.
    """
    name = settings.get("asr_served_model")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return env_value("ASR_SERVED_MODEL", "qwen3-asr")


def transcriptions_url(settings: dict) -> str:
    """Point a chat-completions URL at the audio transcriptions endpoint.

    Input:
        settings (dict): optional ``asr_url``.

    Output:
        str: URL ending in ``/audio/transcriptions``.
    """
    url = settings.get("asr_url") or env_value("ASR_URL", "http://localhost:2043/v1/chat/completions")
    trimmed = url.rstrip("/")
    suffix = "/chat/completions"
    if trimmed.endswith(suffix):
        trimmed = trimmed[: -len(suffix)]
    if trimmed.endswith("/audio/transcriptions"):
        return trimmed
    return trimmed + "/audio/transcriptions"


def _norm_tokens(text: str) -> list[str]:
    """Split a transcript into comparable tokens.

    Input:
        text (str): one transcript line.

    Output:
        list[str]: lowercased tokens with punctuation removed.
    """
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.casefold())
    return cleaned.split()


def _digit_span(tokens: list[str]) -> tuple[int, int] | None:
    """Locate the digit-word run that follows the first ``cho số``.

    Input:
        tokens (list[str]): normalized transcript tokens.

    Output:
        tuple[int, int] | None: start and end of the digit run. ``None`` when
        ``cho số`` is absent.
    """
    for index in range(len(tokens) - 1):
        if tuple(tokens[index:index + 2]) == _CHO_SO:
            start = index + 2
            end = start
            while end < len(tokens) and tokens[end] in _DIGIT_WORD:
                end += 1
            return start, end
    return None


def recover_spoken_zeros(base: str, biased: str) -> str | None:
    """Keep a second transcript when it only inserts ``không`` after ``cho số``.

    The text before ``cho số`` and the text after the digit run must match.
    Dropping or changing any other syllable is rejected.

    Input:
        base (str): first transcript, with the digit zeros missing.
        biased (str): second transcript, decoded with the digit-word context.

    Output:
        str | None: ``biased`` when the only new syllables are ``không``.
        ``None`` when the rest of the sentence changed or nothing was added.
    """
    base_tokens = _norm_tokens(base)
    biased_tokens = _norm_tokens(biased)
    base_span = _digit_span(base_tokens)
    biased_span = _digit_span(biased_tokens)
    if base_span is None or biased_span is None:
        return None
    base_start, base_end = base_span
    biased_start, biased_end = biased_span
    if base_tokens[:base_start] != biased_tokens[:biased_start]:
        return None
    if base_tokens[base_end:] != biased_tokens[biased_end:]:
        return None
    base_run = base_tokens[base_start:base_end]
    biased_run = biased_tokens[biased_start:biased_end]
    if "không" in base_run or len(base_run) < _MIN_PHONE_WORDS:
        return None
    if [token for token in biased_run if token != "không"] != base_run:
        return None
    if biased_run == base_run:
        return None
    return biased


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

    The file may also contain the gold sentence on the second line. Only the
    first line is the transcript the planner should see.

    Input:
        path (Path): file returned by ``cache_path``.

    Output:
        str: transcript text when the file exists.
        None: when the file is missing.
    """
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return ""
    return lines[0]


def write_cache(path: Path, text: str, gold: str | None = None) -> None:
    """Store a transcript so the next run skips the speech model.

    Input:
        path (Path): destination from ``cache_path``. Parent directories are created.
        text (str): transcript to save as UTF-8.
        gold (str | None): the correct sentence, written on the next line when
        the sample has ``audio_text``. Omitted for splits that have no label.

    Output:
        None. Writes ``path``. Line 1 is the transcript. Line 2 is ``gold``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    asr_line = " ".join(text.split())
    lines = [asr_line]
    if isinstance(gold, str) and gold.strip():
        lines.append(" ".join(gold.split()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    kit_root = REPO_ROOT / (settings.get("kit_root") or env_value("KIT_ROOT", "train+public_test"))
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


def _post_transcription(audio_path: Path, settings: dict, prompt: str | None) -> str:
    """Send one wav to the transcriptions endpoint.

    Input:
        audio_path (Path): local wav file.
        settings (dict): optional ``asr_url``, ``asr_language``, and ``asr_served_model``.
        prompt (str | None): digit-word context. ``None`` leaves the first pass unbiased.

    Output:
        str: transcript text.
    """
    if "asr_language" in settings:
        language = settings["asr_language"]
    else:
        language = env_value("ASR_LANGUAGE", "Vietnamese")
    mime_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
    data = {
        "model": served_model(settings),
        "temperature": "0",
    }
    if language:
        data["language"] = asr_language_code(language)
    if prompt:
        data["prompt"] = prompt
    with audio_path.open("rb") as handle:
        response = requests.post(
            transcriptions_url(settings),
            files={"file": (audio_path.name, handle, mime_type)},
            data=data,
            timeout=300,
        )
    response.raise_for_status()
    return parse_transcript(response.json().get("text") or "")


def call_asr(audio_path: Path, settings: dict) -> str:
    """Send one wav to the Qwen3-ASR server and return its text.

    The first pass has no digit context. When ``cho số`` is followed by a
    digit-word run that contains no ``không``, a second pass is decoded with
    that context. The second text is kept only when the rest of the sentence
    is unchanged and the only new syllables are ``không``.

    Input:
        audio_path (Path): local wav file.
        settings (dict): optional ``asr_url`` (str, default ``ASR_URL`` or
        ``http://localhost:2043/v1/chat/completions``), ``asr_language``
        (str, default ``ASR_LANGUAGE`` or ``Vietnamese``), and ``asr_served_model``.
        An empty ``asr_language`` leaves language detection to the server.

    Output:
        str: transcript text.
    """
    if "asr_language" in settings:
        language = settings["asr_language"]
    else:
        language = env_value("ASR_LANGUAGE", "Vietnamese")
    base = _post_transcription(audio_path, settings, None)
    span = _digit_span(_norm_tokens(base))
    if span is None:
        return base
    run = _norm_tokens(base)[span[0]:span[1]]
    if "không" in run or len(run) < _MIN_PHONE_WORDS:
        return base
    biased = _post_transcription(audio_path, settings, asr_instruction(language))
    return recover_spoken_zeros(base, biased) or base


def transcribe(sample: dict, wav_path: str, settings: dict) -> str:
    """Return the transcript for one wav.

    On train, returns ``audio_text`` when ``use_gold_transcript`` is set and
    ``wav_path`` is the latest request. That path does not read or write the
    ASR cache. Otherwise tries the on-disk cache, then the ASR server. The
    gold sentence, when the sample has one, is stored on the second cache line.

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
    gold = sample.get("audio_text")
    gold = gold if isinstance(gold, str) else None
    if settings.get("use_gold_transcript") and wav_path == sample.get("audio") and gold is not None:
        return gold
    cached = read_cache(path)
    if cached is not None:
        if gold and path.is_file() and len(path.read_text(encoding="utf-8").splitlines()) < 2:
            write_cache(path, cached, gold)
        return cached

    audio_path = audio_file(wav_path, settings)
    if not audio_path.is_file():
        raise FileNotFoundError(f"missing wav for {sample['id']}: {audio_path}")
    text = call_asr(audio_path, settings)
    write_cache(path, text, gold)
    return text
