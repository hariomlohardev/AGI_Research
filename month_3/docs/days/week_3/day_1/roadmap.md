# Roadmap — Month 3 Week 3 Day 1: RNN cell and gradients through time

## 1. What to learn today

- What a recurrent cell is: one small net reused at every time step with the same weights.
- What hidden state is: the memory vector passed from step to step. Example: h_3 carries a summary of steps 1 to 3.
- The one-step Jacobian: dh_t/dh_(t-1) = diag(1-h_t²) W_hh. It says how a nudge at step t-1 moves step t.
- Why a bounded tanh slope alone does not prove vanishing. Weight size and direction matter too.
- How to check your cell against nn.RNNCell. Match the bias rule first, or the numbers differ on bookkeeping, not math.

New words, defined where they first appear:

- **Recurrent cell** = a function that reads one input plus the old memory and writes new memory: h_t = f(x_t, h_(t-1)). Example: the same cell runs at step 1, step 2, step 3.
- **Hidden state** = the memory vector h carried across steps. Example: with batch-first input shaped (B,T,D), the states form (B,T,H).
- **Jacobian** = the matrix of all first-order partial derivatives. Example: dh_t/dh_(t-1) tells how each entry of h_t moves when h_(t-1) wiggles.
- **Contractive** = a setup that shrinks signals each step, so old steps fade fast. Example: tiny recurrent weights make h fade toward zero.
- **Detach** = cutting a tensor off the gradient graph. Example: a detached hidden state stops gradients from flowing back past it.
- **Gradient norm plot** = a curve of gradient size versus time step. Example: it shows whether early steps still get learning signal.

## 2. Why learn this

Every sequence model reuses one cell across time. If you cannot see the gradient path through 5 steps, you cannot debug 50. Today builds the eye for it.

## 3. Where it helps in the journey to AI / AGI research

Copy task, LSTM, GRU, and later attention all ask one question: how does signal survive across steps? The Jacobian product is the shared math behind all of them. Learn it on the smallest cell first.

## 4. About the day's code

You write one RNN cell with raw tensor ops and nn.Parameter weights. You unroll a short batch-first sequence. You copy weights to and from nn.RNNCell and compare outputs and gradients on a tiny fixed sequence. You put the loss on the final state only, keep some hidden gradients, and plot their size over time. You compare a shrinking setup against a different recurrent-weight size and report what actually happens.

## 5. Mission = things we must GET FROM the code / result

- A working cell: a_t = W_xh x_t + W_hh h_(t-1) + b, h_t = tanh(a_t). Batch-first. Every axis labeled.
- Parity with nn.RNNCell on a tiny fixed sequence: outputs, final hidden state, input gradients, parameter gradients — all inside a stated tolerance.
- Two gradient-norm plots over time, with a note on where the loss sits and how the steps are indexed.
- A contractive-versus-other-scale comparison with an honest report. A flat curve is data, not failure.
- Correct notation work: tanh'(a) = 1-tanh²(a) at a = 0, 1, 2. Never plug h = 2 into a formula where h is a tanh output (h lives between -1 and 1).

## 6. Understand-before-code = things to know BEFORE coding

- Recall first (write it down): the Jacobian of a linear map, and J = I + J_F from ResNet day.
- Column-vector math: a_t = W_xh x_t + W_hh h_(t-1) + b and h_t = tanh(a_t). Then dh_t/dh_(t-1) = diag(1-h_t²) W_hh.
- Across time these Jacobians multiply. Weight size, activation slope, and direction all count. A tanh slope of at most one alone proves nothing about every net.
- Bias rule: match nn.RNNCell's bias convention before comparing numbers.
- Detach warning: detaching the hidden state at every step destroys the path you came to study. Reset state between independent sequences, never inside one.
- Read the PyTorch nn.RNNCell docs for the exact equation and shapes. One source is enough — then close it and build.

## 7. Extras

- Optional: finite-difference check on a few parameters with a tiny float64 example. Skip it if the core work needs the time. Do not swap the implementation for an autodiff wrapper.

## 8. Success matrix = what success means for that day

| Goal | Good looks like |
|---|---|
| Recurrence idea | Can say why one shared weight sums its gradient contributions over time |
| Cell | Runs batch-first with labeled axes; matches nn.RNNCell inside tolerance |
| Gradient paths | Two norm plots exist; loss place and indexing explained |
| Honesty | The scale comparison reports what actually happened; no forced vanishing curve |
| Notation | tanh' computed at preactivations a = 0, 1, 2; never h = 2 |

## 9. Restrictions = what you CAN use and what you CANNOT use

CAN:
- Raw tensor ops, nn.Parameter, autograd, and your own training loop shape.
- nn.RNNCell as a reference for parity only.
- matplotlib for the two gradient plots.
- A shorter smoke sequence for checks, then the fixed diagnostic sequence.

CANNOT:
- Put nn.RNN or nn.RNNCell inside your own cell — the custom equations are yours to write.
- Detach the hidden state inside a sequence you measure.
- Claim vanishing (or no vanishing) from a single initialization.
- Call a correct output shape proof of correct data layout.
- Copy a full tutorial solution — docs are open, final code is yours.
