# Day 15 — Support Vector Machines & the Kernel Trick (Week 3 Day 3, Month 2)

**Topic:** maximum-margin classification from scratch — hinge loss,
subgradient descent, support vectors, polynomial kernel
**Mode:** guided · **Language:** python · **Skill level:** intermediate

> Labels are `{-1, +1}` today (the SVM convention), not the `{0, 1}` of
> Days 13–14 — the hinge formula needs symmetric labels. All data is
> synthetic and generated inline (seeded `random`) — see
> `coding_problems.md`. No external dataset is fetched.

## Why this matters

Days 13–14 trained by maximising likelihood (MSE, BCE). Today flips the
objective: maximise the *margin* — the distance from the decision boundary
to the nearest point of each class. Same `fit`/`predict` interface, entirely
different geometry, and it sets up the week's second half: Day 16's trees
reuse Week 2's entropy the way today reuses Week 2 Day 2's regularisation
(the `l2` term is the same MAP tug wearing a margin hat). The kernel trick
is worth learning even though pure SVMs are rarer now — "features beat
algorithms" returns on Week 4 Day 2 when encoding choices matter more than
model choice, and `minisklearn` gains a margin-loss model for its
comparison table.

## Watch (in this order)

1. **Support Vector Machines Part 1 (of 3): Main Ideas!!!** — StatQuest
   with Josh Starmer — https://www.youtube.com/watch?v=efR1C6CvhmE
   The core geometric intuition everything today rests on: why maximising
   the margin generalises better than any old separator, and why only the
   support vectors determine the boundary. Watch before touching the
   linear-SVM code — it motivates the margin plot in the `__main__` demo.
2. **Support Vector Machines Part 2: The Polynomial Kernel (Part 2 of 3)**
   — StatQuest with Josh Starmer —
   https://www.youtube.com/watch?v=Toet3EiSFcM
   The kernel-trick installment: adding polynomial features makes
   non-separable data separable, then the trick computes the dot products
   without ever building the expanded vectors. Maps one-to-one onto the
   `polynomial_kernel` demo and the hand-built `x²+y²` rescue — same idea,
   done explicitly first so the trick lands.
3. **Support Vector Machines Part 3: The Radial (RBF) Kernel (Part 3 of
   3)** — StatQuest with Josh Starmer —
   https://www.youtube.com/watch?v=Qc5IyLW_hns
   Stretch goal, not required for the tasks: the RBF kernel,
   infinite-dimensional feature spaces, and the gamma intuition. Worth it
   because RBF is the kernel you'll meet most often in libraries — this
   prevents polynomial-only tunnel vision.
4. **Support Vector Machines: Data Science Concepts** (optional) —
   ritvikmath — https://www.youtube.com/watch?v=iEQ0e-WLgkQ
   A more mathematical treatment of the optimisation setup than
   StatQuest's intuition-first style — watch after the core if you want to
   see where the minimised objective comes from before implementing
   subgradient descent on it.

(All four verified live at generation time. Gaps the videos don't cover —
worked by hand in Math §2–§3 and the `l2` note below: the hinge loss as an
explicit formula, the piecewise subgradient update rule, and how the
regularisation knob appears in the update. Our `l2` plays the role of
`1/(2·C)` in the textbook `C`-formulation: larger `l2` = smaller `C` =
wider margin, more violations tolerated. Try `l2=1.0` vs `l2=0.001` on the
separable blobs and compare the support-vector counts.)

## Math — work through by hand

1. **The margin picture.** Write the hyperplane `w·x + b = 0`; explain why
   scaling `(w, b)` doesn't change the boundary but does change raw scores,
   and why the SVM fixes the scale by demanding `y_i·(w·x_i + b) >= 1`.
   Sketch two blobs with the boundary, the two margin lines, and circle the
   points touching them — those are the support vectors (quota: draw this
   from memory by `/done`).
2. **Hinge case analysis.** Write `max(0, 1 − y·(w·x+b))` and evaluate three
   cases by hand: comfortable margin (loss exactly `0`), on-the-line, and
   confidently wrong — then state precisely why cross-entropy can never do
   the first one (practice question 1 previews this).
3. **The subgradient rule.** From the case analysis, derive who pulls on the
   weights: violators contribute `−y_i·x_i`, comfortable points contribute
   nothing. Note the kink at margin exactly 1 and the convention the code
   adopts (contributes nothing) — quota: state the rule without notes.
4. **Kernel arithmetic by hand.** Compute `(x·y + 1)²` for `x = [1, 2]`,
   `y = [3, 4]` (dot `11`, result `144`) and confirm the ordering
   `K(a, neighbour) > K(a, far)`. Then explain in one paragraph what the
   trick avoids computing — the expanded vectors themselves.

## Code — see `coding_problems.md`

`hinge_loss` → `hinge_subgradients` → `LinearSVMGD`
(`fit`/`decision_function`/`predict` + `l2`) → `polynomial_kernel` →
`add_radius_feature` (the hand-built rescue for the ring).

## Practice questions (source plan)

1. Why does the hinge loss equal exactly 0 for points that are correctly
   classified with a comfortable margin, unlike cross-entropy which is
   never exactly 0?
2. What are "support vectors," and why does the SVM only care about them
   and not the rest of the training data?
3. In your own words, what problem does the kernel trick solve, and why is
   it computationally clever (what does it let you avoid computing
   directly)?
4. Code check: on your ring-shaped non-separable dataset, manually engineer
   a new feature like `x² + y²` (distance from center) and show that a
   *linear* separator now works in this new feature space — this is the
   kernel trick's idea, done by hand instead of automatically.
