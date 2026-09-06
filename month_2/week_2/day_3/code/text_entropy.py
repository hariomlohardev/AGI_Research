"""text_entropy — Week 2 Day 3: exact Shannon entropy of a text.

Treats each character (or word) as an outcome whose probability is its relative
frequency in the text itself, then delegates to entropy().
"""

from __future__ import annotations

from entropy import entropy


def entropy_of_text(text: str, level: str = "char") -> float:
    """Exact Shannon entropy of a text at character or word level.

    Args:
        text: the text to measure.
        level: "char" (P(c) = count(c)/len(text)) or "word" (split on
            whitespace first, P(w) = count(w)/n_words).

    Returns:
        Entropy in bits.

    Raises:
        ValueError: if text is empty or level is not "char"/"word".
        NotImplementedError: stub — implement this (Problem 2).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Compression thought experiment — leave your answer here (coding_problems.md
# Problem 3).  Tests do not grade this, but /done will ask you about it.
# ---------------------------------------------------------------------------
# TODO(student): Why is a file with lower entropy more compressible?
#   (Hint: entropy is the expected number of bits per symbol under an optimal
#   encoding — a skewed distribution lets frequent symbols use short codes.
#   Name the connection: what does a compressor exploit that entropy measures?)
#   Write your 2-3 sentence answer below:


if __name__ == "__main__":
    # Required experiments — keep these runnable (coding_problems.md Problem 2):
    # 1) repetitive vs varied vs random text: print char-level entropy of each,
    #    confirming it rises with genuine unpredictability.
    # 2) same English paragraph at char level vs word level: print both, note
    #    which is higher (practice question 4 — the explanation is for /done).
    pass
