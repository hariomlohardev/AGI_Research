"""file_entropy — Week 2 Day 3: exact Shannon entropy of a text file.

Treats each character (or word) as a random variable with probabilities
estimated from its frequency in the file, then applies entropy().

Compression link — leave your own explanation here (coding_problems.md
Problem 2): a lower-entropy file needs fewer bits per symbol on average
(Shannon's source coding theorem), so it compresses better.
# TODO(student): write 1-2 sentences on why lower entropy => more compressible.
"""

from __future__ import annotations
from typing import Sequence
import random
import tempfile
from collections import Counter
from pathlib import Path

from entropy import entropy


def pdf(data:Sequence[float]):
    len_data = len(data)
    value_count = Counter(data)

    return [count /len_data for count in value_count.values()]

def file_entropy(path: str | Path, level: str = "char") -> float:
    """Shannon entropy (bits) of a text file's token distribution.

    Args:
        path: text file to analyse (read as UTF-8).
        level: "char" treats each character as an outcome; "word" splits
            on whitespace and treats each word as an outcome.

    Returns:
        Entropy in bits, estimated from within-file token frequencies.

    Raises:
        ValueError: if the file yields no tokens, or level is not
            "char"/"word". (A missing file raises FileNotFoundError.)
    """

    if level not in ("char", "word"):
        raise ValueError(f"Given level is {level}, but this function only takes 'word' or 'char'")



    
    try :
        with open(path , encoding='utf-8') as file:
            file_content = file.read().strip()
            if len(file_content) == 0:
                raise ValueError("The file has no tokens")
            words = file_content.split(" ")
            if level == "word":
               pdf_words = pdf(words)
               return entropy(pdf_words)

            elif level== "char":
                chars = []
                for word in words:
                    chars += list(word)
                pdf_chars = pdf(chars)
                return entropy(pdf_chars)

    except FileNotFoundError:
        raise FileNotFoundError(f"File is not found on the given file path {path}")


if __name__ == "__main__":
    # Required experiments — keep these runnable (coding_problems.md §3):
    # 1) ordering: repetitive < varied English < pseudo-random
    repetitive = "ab" * 500
    english = (
        "Entropy is the mathematical definition of surprise. A fair coin "
        "toss is maximally uncertain, while a coin that almost always lands "
        "heads barely surprises you at all. Language sits in between: "
        "letters follow habits and patterns, so English text is far from "
        "random, yet far from perfectly predictable either."
    )
    rng = random.Random(0)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    random_text = "".join(rng.choice(alphabet) for _ in range(2000))
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for name, text in (("repetitive", repetitive), ("english", english), ("random", random_text)):
            p = Path(tmp) / f"{name}.txt"
            p.write_text(text, encoding="utf-8")
            paths[name] = p
        for name, p in paths.items():
            print(f"{name:>12}: {file_entropy(p, 'char'):.4f} bits/char")
    # 2) char vs word level on the same English file
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "english.txt"
        p.write_text(english, encoding="utf-8")
        print(f"char-level: {file_entropy(p, 'char'):.4f} bits/char")
        print(f"word-level: {file_entropy(p, 'word'):.4f} bits/word")
