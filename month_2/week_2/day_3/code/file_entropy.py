"""Compute Shannon entropy of a text file.

This script reads a text file and computes its Shannon entropy in bits,
treating either each character or each whitespace-separated word as a symbol.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from typing import Dict, List, Tuple, Union


def entropy_from_counts(counts: Dict[Union[str, int], int]) -> float:
    """Compute entropy from symbol counts.

    Args:
        counts: mapping from symbol to its frequency (non-negative ints).

    Returns:
        Shannon entropy in bits.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def compute_file_entropy(filename: str, unit: str = "char") -> float:
    """Compute entropy of a file.

    Args:
        filename: path to the text file.
        unit: either "char" (default) or "word".

    Returns:
        Entropy in bits.
    """
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    if unit == "char":
        symbols: List[str] = list(text)
    elif unit == "word":
        symbols = text.split()  # splits on any whitespace
    else:
        raise ValueError("unit must be 'char' or 'word'")

    counts = Counter(symbols)
    return entropy_from_counts(counts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Shannon entropy of a text file."
    )
    parser.add_argument("filename", help="Path to the text file")
    parser.add_argument(
        "--unit",
        choices=["char", "word"],
        default="char",
        help="Treat each character or each word as a symbol (default: char)",
    )
    args = parser.parse_args()

    try:
        ent = compute_file_entropy(args.filename, args.unit)
        print(f"{ent:.6f}")
    except FileNotFoundError:
        print(f"Error: file not found: {args.filename}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()