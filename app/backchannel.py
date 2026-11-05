"""Lexical backchannel detector: is this (short) user utterance just an acknowledgement while the agent talks?

The turn head's backchannel class is not reliable enough on its own, so a segmented utterance is also checked
against a dictionary of English backchannel phrases. "yeah okay" -> True; "okay so what about Tuesday" -> False.
Only meaningful while the agent is speaking: when it is silent, "yeah" is an answer."""
from __future__ import annotations

import re

PHRASES = {
    # vocalisations
    "mm", "mhm", "mmhmm", "mmhm", "hmm", "hm", "uhhuh", "uhuh", "huh", "aha", "ah", "oh", "ooh", "uh", "um", "uhm", "mmm",
    "yeah", "yep", "yup", "yes", "ya", "yea", "okay", "ok", "kay", "right", "sure", "alright", "all right",   # not "no"/"nah": over us they object
    "i see", "got it", "gotcha", "true", "exactly", "indeed", "really", "interesting", "cool", "nice", "great", "good",
    "wow", "oh wow", "oh okay", "oh right", "oh yeah", "oh i see", "i know", "makes sense", "fair enough", "of course",
    "go on", "keep going", "carry on", "uh huh", "mm hmm", "mm hm", "totally", "absolutely", "for sure", "fine", "perfect",
    "sounds good", "no way", "oh no", "oh really", "is that so", "i understand", "understood", "correct", "yes yes",
}
MAX_TOKENS = 5                    # longer than this is content, whatever the words
_PHRASE_TOKENS = {tuple(p.split()) for p in PHRASES}
_MAX_PHRASE = max(len(p) for p in _PHRASE_TOKENS)


def normalize(text: str) -> list[str]:
    text = text.lower().replace("-", "").replace("'", "")
    return re.findall(r"[a-z]+", text)


def is_backchannel(text: str) -> bool:
    toks = normalize(text)
    if not toks or len(toks) > MAX_TOKENS:
        return False
    # can the token sequence be tiled with dictionary phrases? (small DP)
    ok = [True] + [False] * len(toks)
    for i in range(1, len(toks) + 1):
        for k in range(1, min(_MAX_PHRASE, i) + 1):
            if ok[i - k] and tuple(toks[i - k:i]) in _PHRASE_TOKENS:
                ok[i] = True
                break
    return ok[len(toks)]
