"""Unit tests for the dictionary-driven normalizer.

Run:  pytest -q
These exercise the two project dictionaries (contractions.txt + acronyms.csv)
on realistic noisy comment fragments.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess.cleaner import clean_comment
from src.preprocess.normalizer import expand, load_dictionaries


def test_dictionaries_load():
    contractions, acronyms = load_dictionaries()
    assert contractions["don't"] == "do not"
    assert contractions["y'all"] == "you all"
    assert acronyms["idk"] == "I don't know"
    assert acronyms["gr8"] == "great"
    # CSV-quoted multi-comma value must parse intact.
    assert acronyms["a3"] == "anyplace, anywhere, anytime"


def test_expand_single_tokens():
    out = expand("idk why ppl h8 this").lower()
    assert "i don't know" in out
    assert "hate" in out          # h8 -> hate
    assert "idk" not in out


def test_expand_preserves_punctuation():
    out = expand("it's gr8!")
    assert "great" in out.lower()
    assert out.strip().endswith("!")


def test_expand_multiword_phrase():
    out = expand("this is awesome fo shizzle").lower()
    assert "for sure" in out       # "fo shizzle" multi-word key


def test_clean_comment_full_pipeline():
    raw = "OMG y'all this vid is soooo gr8 😍 https://x.com idk why ppl h8 it"
    out = clean_comment(raw)
    assert "http" not in out          # url stripped
    assert "great" in out             # gr8 expanded
    assert "you all" in out           # y'all expanded
    assert "soo" in out and "soooo" not in out   # repeats dampened
    assert out == out.lower()         # lowercased


def test_empty_input():
    assert clean_comment("") == ""
    assert expand("") == ""
