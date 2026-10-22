"""Day 20 — train/test splitting by hand (stub, see Problem 3)."""

from __future__ import annotations

import math
import random

def stratified_split(
    X: list[list[float]],
    y: list[int],
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[list[float]], list[list[float]], list[int], list[int]]:
    """Split into train/test preserving each class's share (stratified).

    Shuffles each class with a seeded local RNG (`random.Random(seed)`),
    then takes `test_ratio` from each class into the test set.

    Raises:
        ValueError: on empty input, length mismatch, a `test_ratio`
            outside (0, 1), or a class too small to stratify.
    """
    # 1. Validation checks
    if not X or not y:
        raise ValueError("Input data X or y cannot be empty.")
        
    if len(X) != len(y):
        raise ValueError("Length mismatch between X and y.")
        
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be strictly between 0 and 1.")

    # 2. Group indices by class label
    class_indices = {}
    for idx, label in enumerate(y):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # 3. Process each class split independently
    train_indices = []
    test_indices = []
    
    # Initialize isolated local random number generator
    rng = random.Random(seed)

    for label, indices in class_indices.items():
        # A class must have at least 2 samples to be stratified split into train and test
        if len(indices) < 2:
            raise ValueError(f"Class {label} has fewer than 2 samples and cannot be stratified.")
            
        # Shuffle indices locally for reproducibility
        rng.shuffle(indices)
        
        # Calculate split boundary (standard round down approach, ensuring at least 1 item if ratio permits)
        test_size = math.floor(len(indices) * test_ratio)
        if test_size == 0 and len(indices) >= 2:
            # Fallback constraint: ensure at least 1 sample goes to test if the math rounds to zero
            test_size = 1
        if len(indices) - test_size == 0:
            # Fallback constraint: ensure at least 1 sample remains in train
            test_size = len(indices) - 1
            
        # Distribute indices
        test_indices.extend(indices[:test_size])
        train_indices.extend(indices[test_size:])

    # 4. Construct the final datasets based on calculated indices
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test
