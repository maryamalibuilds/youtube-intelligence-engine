"""Full comment-cleaning pipeline built on top of normalizer.expand().

clean_comment() is the single entry point the rest of the system calls. It runs
a deterministic, order-sensitive sequence so results are reproducible:

    raw -> strip html/urls -> demojize -> lowercase -> expand slang
        -> (optional) spell-correct -> collapse whitespace/repeats

Each step is toggleable so you can ablate it in the evaluation section of the
report (e.g. "sentiment F1 with vs. without acronym expansion").
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

from .normalizer import expand, spell_correct

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG = re.compile(r"<[^>]+>")
_HANDLE = re.compile(r"@\w+")
_REPEAT = re.compile(r"(.)\1{2,}")          # "soooo" -> "soo"
_MULTISPACE = re.compile(r"\s+")
_NON_PRINT = re.compile(r"[​-‏‪-‮]")  # zero-width / RTL marks


@dataclass
class CleanConfig:
    strip_urls: bool = True
    strip_handles: bool = False     # keep @mentions; NER may want them
    demojize: bool = True
    lower: bool = True
    expand_slang: bool = True
    spellcheck: bool = False        # off by default (slow); enable for eval
    dampen_repeats: bool = True


DEFAULT = CleanConfig()


def _demojize(text: str) -> str:
    try:
        import emoji

        return emoji.demojize(text, delimiters=(" ", " "))
    except Exception:
        return text


def clean_comment(text: str, cfg: CleanConfig = DEFAULT) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = _NON_PRINT.sub("", text)
    text = _HTML_TAG.sub(" ", text)
    if cfg.strip_urls:
        text = _URL.sub(" ", text)
    if cfg.strip_handles:
        text = _HANDLE.sub(" ", text)
    if cfg.demojize:
        text = _demojize(text)
    if cfg.dampen_repeats:
        text = _REPEAT.sub(r"\1\1", text)
    if cfg.lower:
        text = text.lower()
    if cfg.expand_slang:
        text = expand(text)
    if cfg.spellcheck:
        text = spell_correct(text)
    return _MULTISPACE.sub(" ", text).strip()


def clean_batch(comments, cfg: CleanConfig = DEFAULT):
    return [clean_comment(c, cfg) for c in comments]
