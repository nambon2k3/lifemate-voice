"""Align transcript tokens to anchors the sample already gives us.

One syllable distance covers every near miss: tone, the n/ng/nh coda, the
d/gi/r onset, and one vowel letter. A token changes only when that distance
to exactly one anchor is at most one. Anchors are the user's given name and
a verb licensed by an installed tool plus a cue already in the sentence.
No contact list, no per-pair rewrite, no tool call.
"""

from __future__ import annotations

import re

_FUNCTION = frozenset({
    "à", "ạ", "ơi", "nhé", "nha", "nhá", "đi", "cho", "của", "với", "và", "hay",
    "rồi", "thì", "mà", "nếu", "để", "giúp", "dùm", "giùm", "hộ", "lại", "tiếp",
    "cũng", "sẽ", "đã", "đang", "vừa", "các", "những", "cái", "này", "đó", "kia",
    "mới", "bây", "mấy", "ừ", "ờ", "từ", "về", "ở", "ra", "vào", "lên", "xuống", "qua",
})
_NUMBER = frozenset({
    "không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu",
    "bảy", "bẩy", "tám", "chín", "mười", "mươi", "trăm", "ngàn", "nghìn",
    "triệu", "tỷ",
})
_VOCATIVE = frozenset({
    "em", "anh", "chị", "cô", "chú", "bác", "ông", "bà", "bạn", "cho",
})
_ONSETS = ("gi", "tr", "ch", "kh", "gh", "ng", "nh", "ph", "th", "qu")
_ONSET_CLASS = {"d": "z", "r": "z", "gi": "z", "tr": "tr", "ch": "tr", "s": "s", "x": "s", "k": "k", "c": "k"}
_CODA_CLASS = {"n": "n", "ng": "n", "nh": "n"}
_GLIDE = frozenset({"", "i", "y", "o", "u"})
_FOLD = str.maketrans({
    "á": "a", "à": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ấ": "a", "ầ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a", "â": "a",
    "ắ": "a", "ằ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a", "ă": "a",
    "é": "e", "è": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ế": "e", "ề": "e", "ể": "e", "ễ": "e", "ệ": "e", "ê": "e",
    "í": "i", "ì": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ó": "o", "ò": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ố": "o", "ồ": "o", "ổ": "o", "ỗ": "o", "ộ": "o", "ô": "o",
    "ớ": "o", "ờ": "o", "ở": "o", "ỡ": "o", "ợ": "o", "ơ": "o",
    "ú": "u", "ù": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ứ": "u", "ừ": "u", "ử": "u", "ữ": "u", "ự": "u", "ư": "u",
    "ý": "y", "ỳ": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
    "đ": "d",
})


def is_function(token: str) -> bool:
    """Return whether a token is a grammatical word, not a name or a verb.

    Input:
        token (str): one word from the transcript.

    Output:
        bool: True for particles such as ``cho``, ``thì``, or ``tiếp``.
    """
    return token.lower() in _FUNCTION


def is_number_word(token: str) -> bool:
    """Return whether a token reads a number.

    Input:
        token (str): one word, or a digit string.

    Output:
        bool: True for ``năm``, ``ngàn``, or ``50``.
    """
    return token.lower() in _NUMBER or token.isdigit()


def syllable_distance(heard: str, anchor: str) -> int:
    """Count phonological steps between two Vietnamese syllables.

    Tone marks are ignored. Onsets d/gi/r match, and so do tr/ch and s/x.
    Codas n/ng/nh match. An empty coda against a glide, or one extra vowel
    letter, is one step. Any other onset or coda change is one step by itself,
    so two such changes do not count as a near miss.

    Input:
        heard (str): the token from ASR.
        anchor (str): the context form, such as a profile name or ``dời``.

    Output:
        int: 0 when the syllables are the same under those classes, 1 when a
        single step separates them, 2 or more otherwise.
    """
    left = _parts(heard)
    right = _parts(anchor)
    if left is None or right is None:
        return 9
    score = 0 if left[0] == right[0] else 1
    if left[1] != right[1]:
        score += 1 if _one_edit(left[1], right[1]) else 2
    if left[2] != right[2]:
        same_nasal = _CODA_CLASS.get(left[2], left[2]) == _CODA_CLASS.get(right[2], right[2])
        glide = left[2] in _GLIDE and right[2] in _GLIDE
        if not same_nasal:
            score += 1 if glide else 2
    return score


def correct_slots(transcript: str, sample: dict) -> str:
    """Replace a token when one sample anchor is a single step away.

    Input:
        transcript (str): ASR line for the latest request.
        sample (dict): one JSONL row. Uses ``context.user_profile.name`` and
        ``context.installed_apps``. No tool is called.

    Output:
        str: the same sentence. A person-position token may become the user's
        given name. A non-time token may become ``dời`` when a calendar tool
        is installed and the sentence already says ``muộn`` or ``sớm``, or
        ``nạp`` when mobile top-up is installed and the sentence already has
        a thousand-amount with ``cho số`` or ``điện thoại cho``. A word that
        is two or more steps away stays, including ``đặt`` against ``nạp``.
    """
    given = _given_name(sample)
    targets = _verb_targets(sample, transcript)
    pieces = re.split(r"(\s+)", transcript)
    indexes = [index for index, piece in enumerate(pieces) if piece.strip()]
    for position, index in enumerate(indexes):
        token = pieces[index]
        if is_function(token) or is_number_word(token):
            continue
        previous = pieces[indexes[position - 1]] if position else ""
        following = pieces[indexes[position + 1]] if position + 1 < len(indexes) else ""
        if given and _name_slot(previous, following) and _near(token, given):
            pieces[index] = given
            continue
        if _is_clock(token, previous, following):
            continue
        pieces[index] = _nearest(token, targets)
    return "".join(pieces)


def _near(token: str, anchor: str) -> bool:
    """Return whether a token should be written as this anchor.

    Input:
        token (str): the ASR word.
        anchor (str): one candidate form.

    Output:
        bool: True when the syllable distance is at most 1 and the folded
        spellings still differ.
    """
    return _fold(token) != _fold(anchor) and syllable_distance(token, anchor) <= 1


def _nearest(token: str, anchors: list[str]) -> str:
    """Return the unique anchor within one step, or the token itself.

    Input:
        token (str): the ASR word.
        anchors (list[str]): verb lemmas licensed for this sentence.

    Output:
        str: the anchor, cased like ``token``, when exactly one anchor is
        within one step. ``token`` when none or more than one are that close.
    """
    found = [anchor for anchor in anchors if _near(token, anchor)]
    if len(found) != 1:
        return token
    anchor = found[0]
    if token[:1].isupper():
        return anchor[:1].upper() + anchor[1:]
    return anchor


def _name_slot(previous: str, following: str) -> bool:
    """Return whether this token sits where a person's name is spoken.

    Input:
        previous (str): the word before the token. Empty at the start.
        following (str): the word after the token. Empty at the end.

    Output:
        bool: True after a vocative or ``cho``, and before a number, a
        function word, or the end of the sentence. A following content word
        such as ``hệ`` keeps the token, so ``liên hệ`` stays.
    """
    if previous.lower() not in _VOCATIVE:
        return False
    if not following:
        return True
    return is_function(following) or is_number_word(following)


def _is_clock(token: str, previous: str, following: str) -> bool:
    """Return whether ``giờ`` here is a clock word rather than a verb.

    Input:
        token (str): the candidate.
        previous (str): the word before it.
        following (str): the word after it.

    Output:
        bool: True for ``bây giờ``, ``mấy giờ``, ``một giờ``, or ``giờ mới``.
    """
    if _fold(token) != "gio":
        return False
    if is_number_word(previous) or _fold(previous) in {"bay", "may"}:
        return True
    return following.lower() in {"mới", "này", "đó", "kia"}


def _verb_targets(sample: dict, text: str) -> list[str]:
    """List verb lemmas the installed apps and the sentence both support.

    Input:
        sample (dict): one JSONL row.
        text (str): the whole transcript. Cues are read from it. State is not.

    Output:
        list[str]: ``dời`` when a calendar tool is installed and the sentence
        says ``muộn`` or ``sớm``. ``nạp`` when mobile top-up is installed and
        a thousand-amount is tied to ``cho số`` or ``điện thoại cho``.
    """
    tools = _installed_tools(sample)
    low = text.lower()
    targets = []
    if any(name.startswith("calendar.") for name in tools) and re.search(r"\b(?:muộn|sớm)\b", low):
        targets.append("dời")
    topup = "ewallet.topup_mobile" in tools
    amount = re.search(r"\b(?:ngàn|nghìn)\b", low) is not None
    cue = "cho số" in low or "điện thoại cho" in low
    if topup and amount and cue:
        targets.append("nạp")
    return targets


def _installed_tools(sample: dict) -> set[str]:
    """Collect tool names listed on the sample.

    Input:
        sample (dict): one JSONL row. ``context.installed_apps`` maps an app
        id to ``{"tools": [...]}``.

    Output:
        set[str]: tool names. Empty when the sample has no installed apps.
    """
    apps = (sample.get("context") or {}).get("installed_apps") or {}
    found: set[str] = set()
    if not isinstance(apps, dict):
        return found
    for spec in apps.values():
        if not isinstance(spec, dict):
            continue
        for name in spec.get("tools") or []:
            if isinstance(name, str):
                found.add(name)
    return found


def _given_name(sample: dict) -> str:
    """Read the user's given name from the profile.

    Input:
        sample (dict): one JSONL row. ``context.user_profile.name`` is a
        string such as ``Em Linh``.

    Output:
        str: the last word of that name. Empty when the profile has no name.
    """
    profile = (sample.get("context") or {}).get("user_profile") or {}
    parts = str(profile.get("name") or "").split()
    if not parts or len(parts[-1]) < 2:
        return ""
    return parts[-1]


def _fold(token: str) -> str:
    """Lowercase a token and drop tone marks and vowel hats.

    Input:
        token (str): one word.

    Output:
        str: the folded letters. Empty when ``token`` is empty.
    """
    return token.lower().translate(_FOLD)


def _parts(token: str) -> tuple[str, str, str] | None:
    """Split one syllable into onset class, nucleus, and coda.

    Input:
        token (str): one word.

    Output:
        tuple[str, str, str]: onset class, nucleus, coda.
        None: when the token has no nucleus.
    """
    folded = _fold(token)
    onset = ""
    rest = folded
    for prefix in _ONSETS:
        if rest.startswith(prefix) and len(rest) > len(prefix):
            onset = prefix
            rest = rest[len(prefix):]
            break
    else:
        if not rest:
            return None
        onset, rest = rest[:1], rest[1:]
    coda = ""
    for ending in ("ng", "nh", "ch", "n", "m", "p", "t", "c"):
        if rest.endswith(ending) and len(rest) > len(ending):
            coda = ending
            rest = rest[:-len(ending)]
            break
    if not coda and len(rest) > 1 and rest[-1] in "iyou":
        coda = "i" if rest[-1] in "iy" else rest[-1]
        rest = rest[:-1]
    if not rest:
        return None
    return _ONSET_CLASS.get(onset, onset), rest, coda


def _one_edit(left: str, right: str) -> bool:
    """Return whether two short strings differ by one insert, delete, or swap.

    Input:
        left (str): a nucleus.
        right (str): a nucleus.

    Output:
        bool: True when the edit distance is 1.
    """
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    for index, char in enumerate(left):
        if char != right[index]:
            return left[index:] == right[index + 1 :] or left[index + 1 :] == right[index + 1 :]
    return True
