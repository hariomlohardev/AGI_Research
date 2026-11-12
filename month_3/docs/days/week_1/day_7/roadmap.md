# Month 3 · Week 1 Day 7 — Roadmap: diagnose a real training run

**Time:** 5–6 hours (split across study days if needed). **Mode:** guided. **Tags:** `debugging`, `training-dynamics`.

## Order of work (with breaks between blocks)

| # | Block | Minutes | Output |
|---|---|---|---|
| 1 | Recall: BatchNorm backward dependency + two histogram plots, no captions | 15–20 | Spoken explanation from memory |
| 2 | Read: training-debugging recipe + roadmap §Day 7 scope (ordinary run, two bug branches, bug-report evidence) | 45–60 | Closed tabs; predicted signatures for both bugs |
| 3 | Verified video (linked in `learn.md`) | scoped sections only | Curve-reading and bug-vs-noise judgment watched |
| 4 | Run: ordinary training branch, describe the curve | 30–45 | Loss/accuracy/grad-norm log + honest description |
| 5 | Implement: Problems 1–2 (frozen check, alignment pairs) + Bug report A | 60–75 | `True` on frozen snapshots; written report |
| 6 | Implement: Problem 3 (curve summary) + Bug report B + memorization sentence | 60–75 | Flags match hand values; written report |
| 7 | Evidence + conclusion (Problem 5), kernel restart, Run All in smoke mode | 30–45 | Training log, two bug reports, `run_smoke` passes |

## Done when

- One readable training log exists for the ordinary run, described without claiming every fluctuation is a bug.
- Two short bug reports exist (symptom, evidence, cause, fix, verification): the omitted update and the label misalignment.
- The conclusion answers why tiny-batch memorization alone cannot catch every data bug.

## Practice questions (also the `/done` quiz pool)

1. Which signal distinguishes no learning from a logging mistake?
2. How would you investigate rising validation loss alongside falling train loss?
3. Why is freezing one layer not guaranteed to make training fail?
4. What additional check caught the label error that the tiny-batch check could miss?

## Smoke evidence (seed 11, CPU)

`run_smoke` asserts: the ordinary run finishes with finite losses and a correctly shaped log; `params_unchanged` returns `True` on the frozen branch and `False` on a perturbed copy; `alignment_report` returns exact indexed pairs on the fixture; `curve_summary` matches hand values including the rising-val/falling-train flags. Fresh-kernel execution must stop at no stub — every stub implemented before `/done`.
