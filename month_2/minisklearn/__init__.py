"""`minisklearn` — classical ML from scratch (Month 2 capstone package).

Scaffolding for Week 4 Day 1: this file is intentionally almost empty. Your
Problem 3 is to wire up the explicit re-exports so users get the clean API:

    from .linear_model import LinearRegression, LogisticRegression
    from .tree import DecisionTreeClassifier
    from .ensemble import RandomForestClassifier
    from .cluster import KMeans
    from .metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

(See `month_2/week_4/day_1/coding_problems.md`. Tests import submodules
directly AND check these top-level names.)
"""

from .linear_model import LinearRegression, LogisticRegression
from .tree import DecisionTreeClassifier
from .ensemble import RandomForestClassifier
from .cluster import KMeans
from .svm import LinearSVMGD
from .metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
