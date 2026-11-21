# Month 3 · Week 2 Day 4 — AlexNet ideas, at a size you can train

## 1. What to learn today

AlexNet was the first deep CNN that won big (ImageNet 2012). You will not
rebuild it. You will steal its *ideas* and fit them onto 32×32 CIFAR photos.

New terms, defined here:

- **adaptation:** taking an idea from a paper and changing its sizes and
  settings to fit your own data. The opposite of blind copying.
- **replication:** rebuilding the paper's exact setup. Not today's goal —
  your photos are 32×32, theirs were 224×224.
- **ReLU:** the function `max(0, x)`. Negative inputs become 0, positive
  inputs pass through unchanged. The nonlinearity AlexNet used instead of
  tanh.
- **saturation:** when a unit's output stops changing even as its input
  grows (tanh flattens near −1 and +1). A saturated unit barely learns.
- **dead unit:** a ReLU unit whose input is always negative, so it always
  outputs 0 and its gradient is 0 forever. ReLU avoids saturation on the
  positive side but can still die on the negative side.
- **dropout:** during training, randomly switching off a fraction of units
  (you met it on Day 4). AlexNet used it in its big classifier head.
- **LRN (local response normalization):** a 2012 trick where each activation
  was divided by its neighbors' activity. History, not a tool — modern nets
  dropped it. Today you learn *why it was tried*, not how to use it.
- **classifier head:** the fully connected layers at the end of a CNN that
  turn the last feature map into 10 scores. Often holds most of the weights.
- **adaptation table:** your three-column record — original idea, your
  change, reason for the change. The day's written proof that you adapted,
  not copied.

## 2. Why learn this

Three reasons. First, ReLU + stacked convolution + pooling + dropout is the
skeleton inside nearly every vision model since 2012 — learn it once here
and you will recognize it in VGG tomorrow and ResNet the day after. Second,
this is your first *paper-to-code* day: research is mostly reading an idea
and reshaping it to fit your problem, and that reshaping is the skill being
graded. Third, the LRN story teaches historical honesty — a famous paper can
contain an idea that later dies, and "it was in AlexNet" is never proof that
it helps your model.

## 3. Where it helps in the journey to AI / AGI research

Week 2 is one arc: Day 3 built the pipeline, today fits the first classic
architecture onto it, Day 5 (VGG) tests small stacked kernels, Day 6
(ResNet) adds skip connections. Each day reuses the same CIFAR setup, so
only the *architecture idea* changes — that is controlled comparison, the
same discipline behind every serious ML paper. Reading papers for
transferable ideas (not for numbers to copy) is also how Months 4–5 will
consume the attention and sequence literature.

## 4. About the day's code

You reuse your Day 11 CIFAR pipeline and training loop unchanged: same
45k/5k split, same seed, same train-only stats, same deterministic check
transform, sealed test set. The only new code is the model itself — a
compact AlexNet-inspired net for 32×32 inputs — plus the checks around it:
a layer/shape table with parameter counts, a tiny-batch fit, one bounded
run of at most 10 epochs (or a predeclared time cap), a confusion matrix,
and a saved best checkpoint. You also pick the shared comparison budget
with tomorrow's VGG run *today*, before either runs.

## 5. Mission = things we must GET from the code / result

- ✅ Recall written down: your most common CIFAR validation error + one
  limitation of the Day 11 OFF-vs-ON comparison.
- ✅ A compact AlexNet-inspired model with sensible 32×32 spatial sizes
  (no 224×224 geometry copied blindly).
- ✅ A layer/shape table with a counted parameter total.
- ✅ One bounded run (≤10 epochs or your time cap): run table with train
  loss, check loss, check accuracy, time, and a saved best checkpoint.
- ✅ A confusion matrix with full class order, compared against your Day 11
  error notes.
- ✅ An adaptation table: original idea → your change → reason, grounded
  in the paper (ReLU, conv structure, dropout, LRN's historical role).
- ✅ An honest note: what the short run does and does not prove. Losing to
  the simple CNN under a small budget settles nothing about full training.

## 6. Understand-before-code = things to know BEFORE coding

1. **ReLU vs tanh slopes.** Away from their kinks, tanh's slope shrinks
   toward 0 for large |x| (that is saturation — the gradient vanishes and
   learning stalls), while ReLU's slope is exactly 1 for all x > 0 (no
   positive-side saturation). But a ReLU unit with always-negative input
   has slope 0 forever — a dead unit. So ReLU is not a universal gradient
   fix; it trades one failure for a milder one.
2. **Why a big head dominates the weights.** One fully connected layer
   from F inputs to C outputs holds F × C weights plus C biases. Example:
   a 4096-wide head on a 256×5×5 = 6400 map holds 6400 × 4096 ≈ 26M
   weights in a single layer — more than all conv layers combined. Shrink
   the map (pooling) or shrink the head before blaming the convolutions.
3. **The LRN story in one paragraph.** AlexNet divided each activation by
   its neighbors to punish lone spikes ("lateral inhibition" from biology).
   Later work showed BatchNorm and better inits cover that ground, and LRN
   vanished from modern nets. Lesson: a famous paper's trick is a
   historical fact, not evidence it helps *your* model.
4. **Sizing formula.** Output = floor((input + 2×pad − (kernel−1) − 1) /
   stride) + 1 per spatial side. Check every block on paper before coding —
   32×32 leaves no room for 11×11 stride-4 openers.
5. **Shared budget with Day 5.** Agree *today* on the epochs-or-time cap
   both days will use, and write it down. A comparison is only fair if the
   budgets match.
6. **Old rules still hold.** Train-only stats, deterministic check, sealed
   test set, same split and seed as Day 11, save the best-checkpoint with
   its config.

## 7. Extras

- Primary reading (60–90 min, then close it): AlexNet sections 1–4 —
  ReLU, conv structure, dropout, LRN's historical role.
- Optional, only if all required work is finished: one LRN on/off
  comparison, at most one extra matched run inside the day's budget. It is
  not a completion gate and nobody needs LRN to win.

## 8. Success matrix

| Question for today   | What success looks like                                                     |
|----------------------|-----------------------------------------------------------------------------|
| Learn a topic?       | You can name which AlexNet ideas transfer across sizes and which needed resizing, with reasons. |
| Learn an algorithm?  | A compact conv net runs end to end on 32×32 with counted shapes and params. |
| Code runs?           | Shapes table + tiny-batch fit + one bounded run all recorded, checkpoint saved. |
| Speed?               | Run fits the ≤10-epoch / time cap you predeclared; total day 4–5 hours.      |

## 9. Restrictions = what you CAN use and what you CANNOT

CAN use: your Day 11 pipeline and training loop, `torch` layers /
optimizers / losses, torchvision CIFAR-10 + transforms, sklearn accuracy
and confusion-matrix helpers, matplotlib, the paper and official docs.

CANNOT: copy the ImageNet 224×224 geometry (your photos are 32×32);
treat LRN as required work; touch the test set for any picking decision;
fit normalization stats on check or test; claim an architecture wins from
one short run or one seed; paste a full tutorial solution — the final code
is yours.
