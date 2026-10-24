# Day 23 flagged questions

## Flag 1 (logged 2026-10-24 via `/done` escape hatch)

**Question (Q2, second half):** For each of the two diagnosed causes
(too-tiny learning rate on unscaled features; exact ID3 splits fragmenting
continuous inputs), explain in one line why that cause pushes *recall*
down specifically, rather than some other metric.

**Learner's stated reason:** "i dont understand it" — could name both
causes from the blog post but could not connect either cause to the
recall-specific symptom (missed survivors / false negatives vs. the
~38%-positive Titanic split).

**Status:** Open — revisit via `/spaced-review`: why does an underfit,
majority-leaning classifier lose recall first (conservative positive
predictions → false negatives), and why does ID3 fragmentation of
continuous features end in under-predicted positives?

**Status update (2026-10-24 review pass):** revisited — the ID3→recall chain
answered correctly first ask (fragmented leaves memorize training points;
unseen survivors fall into died-majority leaves → false negatives). Record
kept for history.
