"""Tests for entropy_of_text — Week 2 Day 3."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from text_entropy import entropy_of_text


def test_constant_text_has_zero_entropy_char_level():
    assert entropy_of_text("a" * 100) == pytest.approx(0.0)


def test_uniform_chars_give_log2_alphabet():
    # "abcd" repeated: 4 equiprobable chars -> exactly 2 bits
    assert entropy_of_text("abcd" * 25) == pytest.approx(2.0)


def test_entropy_rises_with_unpredictability():
    constant = entropy_of_text("a" * 77)
    skewed = entropy_of_text("a" * 70 + "bcdefgh")
    uniform = entropy_of_text("abcdefgh" * 10)  # 8 equiprobable chars -> 3 bits
    assert constant == pytest.approx(0.0)
    assert uniform == pytest.approx(3.0)
    assert constant < skewed < uniform


def test_word_level_known_value():
    # "hello" 2/3, "world" 1/3 -> H ~= 0.9183 bits
    assert entropy_of_text("hello world hello", level="word") == pytest.approx(
        0.9183, abs=1e-3
    )


def test_repeated_word_has_zero_entropy():
    assert entropy_of_text("yes yes yes yes", level="word") == pytest.approx(0.0)


def test_char_and_word_both_run_on_english():
    text = "the cat sat on the mat and the dog sat too"
    h_char = entropy_of_text(text, level="char")
    h_word = entropy_of_text(text, level="word")
    assert h_char > 0
    assert h_word > 0


def test_raises_on_empty_or_bad_level():
    with pytest.raises(ValueError):
        entropy_of_text("")
    with pytest.raises(ValueError):
        entropy_of_text("hello", level="sentence")
