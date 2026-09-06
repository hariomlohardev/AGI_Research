"""Tests for file_entropy — Week 2 Day 3."""

import math
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from file_entropy import file_entropy

ENGLISH = (
    "Entropy is the mathematical definition of surprise. A fair coin "
    "toss is maximally uncertain, while a coin that almost always lands "
    "heads barely surprises you at all. Language sits in between: "
    "letters follow habits and patterns, so English text is far from "
    "random, yet far from perfectly predictable either."
)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _random_text(n=2000, seed=0):
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(alphabet) for _ in range(n))


def test_repetitive_ab_is_one_bit(tmp_path):
    p = _write(tmp_path, "rep.txt", "ab" * 500)
    assert file_entropy(p, "char") == pytest.approx(1.0, rel=1e-9)


def test_entropy_ordering_repetitive_lt_english_lt_random(tmp_path):
    rep = _write(tmp_path, "rep.txt", "ab" * 500)
    eng = _write(tmp_path, "eng.txt", ENGLISH)
    rnd = _write(tmp_path, "rnd.txt", _random_text())
    h_rep = file_entropy(rep, "char")
    h_eng = file_entropy(eng, "char")
    h_rnd = file_entropy(rnd, "char")
    assert h_rep < h_eng < h_rnd


def test_word_level_runs_and_is_nonnegative(tmp_path):
    p = _write(tmp_path, "eng.txt", ENGLISH)
    h = file_entropy(p, "word")
    assert math.isfinite(h) and h >= 0


def test_word_level_matches_hand_computation(tmp_path):
    # words of "a a a b": [a, a, a, b] => -(0.75*log2(0.75)+0.25*log2(0.25))
    p = _write(tmp_path, "toy.txt", "a a a b")
    expected_word = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert file_entropy(p, "word") == pytest.approx(expected_word, rel=1e-9)


def test_raises_on_empty_file(tmp_path):
    p = _write(tmp_path, "empty.txt", "")
    with pytest.raises(ValueError):
        file_entropy(p, "char")


def test_raises_on_bad_level(tmp_path):
    p = _write(tmp_path, "eng.txt", ENGLISH)
    with pytest.raises(ValueError):
        file_entropy(p, "byte")
