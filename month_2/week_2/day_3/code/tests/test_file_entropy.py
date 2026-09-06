"""Tests for file_entropy.py."""

import math
import os
import tempfile
import pytest

from file_entropy import compute_file_entropy, entropy_from_counts


def test_entropy_from_counts():
    counts = {'a': 3, 'b': 1}
    # total 4, p_a=0.75, p_b=0.25
    expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert entropy_from_counts(counts) == pytest.approx(expected, abs=1e-9)


def test_entropy_from_counts_empty():
    assert entropy_from_counts({}) == 0.0


def test_entropy_from_counts_single():
    counts = {'x': 5}
    assert entropy_from_counts(counts) == 0.0


def test_compute_file_entropy_char():
    content = "aaab"  # a:3, b:1
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(content)
        fname = f.name
    try:
        ent = compute_file_entropy(fname, unit='char')
        expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        assert ent == pytest.approx(expected, abs=1e-9)
    finally:
        os.unlink(fname)


def test_compute_file_entropy_word():
    content = "hello world hello"  # words: hello(2), world(1)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(content)
        fname = f.name
    try:
        ent = compute_file_entropy(fname, unit='word')
        # total 3 words, p_hello=2/3, p_world=1/3
        expected = -((2/3) * math.log2(2/3) + (1/3) * math.log2(1/3))
        assert ent == pytest.approx(expected, abs=1e-9)
    finally:
        os.unlink(fname)


def test_compute_file_entropy_newline():
    content = "a\nb\nb"  # chars: a, \n, b, \n, b -> a:1, \n:2, b:2
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(content)
        fname = f.name
    try:
        ent = compute_file_entropy(fname, unit='char')
        total = 5
        pa = 1/5
        pn = 2/5
        pb = 2/5
        expected = -(pa * math.log2(pa) + pn * math.log2(pn) + pb * math.log2(pb))
        assert ent == pytest.approx(expected, abs=1e-9)
    finally:
        os.unlink(fname)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        compute_file_entropy("nonexistent.txt", unit='char')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])