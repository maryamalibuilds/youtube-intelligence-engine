"""Text normalization for noisy YouTube comments.

Loads the two project dictionaries and expands contractions + internet slang,
then optionally spell-corrects. This is the foundation step every downstream
model (NER, sentiment, topics, RAG) depends on, so it is fully implemented and
unit-tested (see tests/test_normalizer.py).

Dictionary formats
------------------
- contractions.txt : ``key:value`` per line  (e.g. ``don't:do not``)
- acronyms.csv     : ``key,value`` per line, CSV-quoted where a field contains
                     a comma (e.g. ``a3,"anyplace, anywhere, anytime"``)

Both are matched case-insensitively. Multi-word keys (``fo shizzle``,
``i c``) are handled with a phrase pass before single-token replacement.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

from config import ACRONYMS_FILE, CONTRACTIONS_FILE

# A token is a run of word-ish chars, but social text needs apostrophes,
# leetspeak punctuation and digits to survive tokenization so that keys like
# ``don't`` or ``h4x0r`` can match. We split on whitespace and peel trailing
# punctuation off each token instead of using \w+.
_TRAILING_PUNCT = re.compile(r"^([^\w$@<>|^*]*)(.*?)([^\w$@<>|^*]*)$")


def _load_contractions(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if key:
                mapping[key.lower()] = val
    return mapping


def _load_acronyms(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            key, val = row[0].strip(), row[1].strip()
            if key:
                mapping[key.lower()] = val
    return mapping


@lru_cache(maxsize=1)
def load_dictionaries() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (contractions, acronyms). Cached so files are read once."""
    contractions = _load_contractions(CONTRACTIONS_FILE)
    acronyms = _load_acronyms(ACRONYMS_FILE)
    return contractions, acronyms


@lru_cache(maxsize=1)
def _merged_maps() -> Tuple[Dict[str, str], Dict[str, str], "re.Pattern[str]"]:
    """Build single- and multi-word lookup tables + a phrase regex.

    Contractions take priority over acronyms on key collisions because they are
    grammatical expansions (e.g. ``it's`` -> ``it is``) rather than slang.
    """
    contractions, acronyms = load_dictionaries()
    single: Dict[str, str] = {}
    multi: Dict[str, str] = {}
    for src in (acronyms, contractions):  # contractions applied last = win
        for key, val in src.items():
            (multi if " " in key else single)[key] = val

    # Longest phrases first so "fo shizzle" wins over a hypothetical "fo".
    phrases = sorted(multi, key=len, reverse=True)
    phrase_re = (
        re.compile(r"(?<!\w)(" + "|".join(re.escape(p) for p in phrases) + r")(?!\w)", re.I)
        if phrases
        else re.compile(r"(?!x)x")  # never matches
    )
    return single, multi, phrase_re


def expand(text: str) -> str:
    """Expand contractions + slang/acronyms in ``text`` (case-insensitive)."""
    if not text:
        return text
    single, multi, phrase_re = _merged_maps()

    # 1) Multi-word phrases first.
    text = phrase_re.sub(lambda m: multi[m.group(1).lower()], text)

    # 2) Token-by-token for single-word keys, preserving surrounding punctuation.
    out = []
    for tok in text.split():
        lead, core, trail = _TRAILING_PUNCT.match(tok).groups()
        repl = single.get(core.lower())
        out.append(f"{lead}{repl if repl is not None else core}{trail}")
    return " ".join(out)


# --- Optional spell correction (lazy: only import symspell if used) -----------
_SYM = None


def _get_symspell():
    global _SYM
    if _SYM is None:
        from importlib import resources

        from symspellpy import SymSpell, Verbosity  # noqa: F401

        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        with resources.path("symspellpy", "frequency_dictionary_en_82_765.txt") as p:
            sym.load_dictionary(str(p), term_index=0, count_index=1)
        _SYM = sym
    return _SYM


def spell_correct(text: str) -> str:
    """Best-effort spell correction. Returns input unchanged if symspell absent."""
    try:
        from symspellpy import Verbosity

        sym = _get_symspell()
    except Exception:
        return text
    out = []
    for tok in text.split():
        if not tok.isalpha() or len(tok) < 4:
            out.append(tok)
            continue
        sug = sym.lookup(tok.lower(), Verbosity.TOP, max_edit_distance=2)
        out.append(sug[0].term if sug else tok)
    return " ".join(out)


if __name__ == "__main__":
    samples = [
        "idk why ppl h8 this, it's gr8 2moro",
        "y'all gonna luv this fo shizzle, c u l8r",
        "this vid sux but the editing is awsm tbh",
    ]
    for s in samples:
        print(f"IN : {s}\nOUT: {expand(s)}\n")
