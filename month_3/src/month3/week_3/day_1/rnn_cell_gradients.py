# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🔁 Day 16 — RNN Cell and Gradients Through Time ⏳🧠
#
# ## 🎯 Goal
#
# Build ONE recurrent cell yourself, with raw tensor ops — then SEE gradients flow back through time. By the end you can say, from your own plots: how far back does learning signal reach, and what controls it?
#
# ## 🧠 Quick background (read this, then close it and build)
#
# - A **recurrent cell** is one small net reused at every time step with the SAME weights: `h_t = tanh(W_xh x_t + W_hh h_(t-1) + b)`.
# - **Hidden state** `h` is the memory vector passed step to step. Batch-first input `(B, T, D)` gives states `(B, T, H)`.
# - The one-step **Jacobian** is `dh_t/dh_(t-1) = diag(1 - h_t²) · W_hh`. Across steps these MULTIPLY — that product is the whole story of vanishing / exploding gradients.
# - ⚠️ Notation trap: `tanh'(a) = 1 - tanh²(a)` takes the PREACTIVATION `a`. Never plug `h = 2` into a formula where `h` is a tanh output (outputs live in [-1, 1]).
# - `nn.RNNCell` is your REFERENCE for parity checks only — your cell must be your own equations.
#
# ## 🪜 Step-by-step work (the empty cells below are ALL yours)
#
# - **Step 0 — Recall:** the Jacobian of a linear map, `J = I + J_F` from ResNet day, and `tanh'(a)` at `a = 0, 1, 2`.
# - **Step 1 — Your cell:** raw ops + `nn.Parameter`, unroll a short batch-first sequence, label every axis.
# - **Step 2 — Reference parity:** copy weights to/from `nn.RNNCell` (match the bias rule FIRST), compare outputs, final hidden, input grads, param grads on a tiny FIXED sequence, inside a tolerance YOU state.
# - **Step 3 — Gradient paths:** loss on the FINAL state only, keep some hidden grads, plot their norms over time, explain loss place + indexing.
# - **Step 4 — Init comparison:** a shrinking (contractive) setup vs a different recurrent-weight size. Report what ACTUALLY happens — a flat curve is data, not failure.
# - **Step 5 — Results:** tolerances, both plots, one honest paragraph.
# - **Step 6 — Your tests:** write your own checks. They must pass before handoff.
#
# ## ✅ Final result checklist
#
# - [ ] Cell runs batch-first with labeled axes
# - [ ] Parity vs `nn.RNNCell` inside a stated tolerance (outputs + grads)
# - [ ] Two gradient-norm plots + loss-place/indexing note
# - [ ] Init comparison reported honestly (no forced vanishing curve)
# - [ ] `tanh'` done at preactivations, never `h = 2`
# - [ ] My own tests pass in this notebook
#
# ## 🚧 Restrictions
#
# YOUR equations only — no `nn.RNN`/`nn.RNNCell` inside your cell. No detach inside a measured sequence (reset BETWEEN sequences). No claiming vanishing from one init. A right shape is not proof of a right layout.
#
# ## 🏁 How to finish
#
# 1. Work in Colab for speed (this notebook has no local-only paths).
# 2. Download the finished `.ipynb` back OVER this repo copy.
# 3. Say **handoff done** — sync and checks run after that, never before. 💪

# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# %% [markdown]
# ## Step 0 — Recall ✍️
#
# Jacobian of a linear map. `J = I + J_F`. `tanh'(a) = 1 - tanh²(a)` at `a = 0, 1, 2`. (Your scratch space — show the notation is clean.)

# %%
"""

T(x) = Ax is simply the constant matrix A itself.

Because the partial derivatives of a linear function are constant everywhere,

the Jacobian does not depend on the input point x."""
"""

In a residual block, the output is defined as y = x + F(x), where x is the input and F(x) is the residual function

J = I + J_F
tanh'(a) = 1 - tanh²(a)

a = 0 → 1.0, a = 1 → ≈0.42, a = 2 → ≈0.07

never plug h = 2, h lives in [-1, 1]

"""

# %% [markdown]
# ## Step 1 — Your cell 🧱
#
# Raw ops + `nn.Parameter`. Unroll a short batch-first sequence. Label every axis: which is batch, which is time, which is features.

# %%
D = 3 #— each input step is 3 numbers.
H = 5 #— each hidden state is 5 numbers.
B = 2
T = 4 #sequences, 4 steps each.

# %%
W_ih_x = nn.Parameter(torch.randn(D, H) * 0.5)
W_hh_h = nn.Parameter(torch.randn(H, H) * 0.5)
b_ih   = nn.Parameter(torch.randn(H,) * 0.5)
b_hh   = nn.Parameter(torch.randn(H,) * 0.5)


# %%
def cell(x_t, h_prev):

  "tanh(x_t @ W_xh + h_prev @ W_hh + b)"

  return torch.tanh(((x_t @ W_ih_x) + b_ih) +((h_prev @ W_hh_h) + b_hh))


# %%
x = torch.randn(B, T, D)

# %%
h_prev = torch.zeros(B, H)
states = []

# %%
for i in range(T):
  x_t = x[:, i, :]
  h_t = cell(x_t , h_prev)
  states.append(h_t)
  h_prev = h_t

states = torch.stack(states, dim=1)  # (B, T, H)

# %%
states.shape

# %%
states

# %%
all(states.view(-1) < 1) , all(-1 < states.view(-1))

# %%
# in (2,4,3) -> out (2,4,5): batch 2, time 4, memory 5.

# %% [markdown]
# ## Step 2 — Reference parity 🔍
#
# Match the `nn.RNNCell` bias rule FIRST, then copy weights both ways. Tiny FIXED sequence. Compare outputs, final hidden, input grads, param grads. State your tolerance.

# %%
ref = nn.RNNCell(input_size=D, hidden_size=H)

# %%
ref.weight_hh.data = W_hh_h.t().detach().clone()
ref.weight_ih.data = W_ih_x.t().detach().clone()
ref.bias_hh.data = b_hh.detach().clone()
ref.bias_ih.data = b_ih.detach().clone()

# %%

# %%
h_ref = torch.zeros(B,H)

# %%
h_ref.shape

# %%
W_ih_x.shape

# %%
ref_states = []


for i in range(T):
  x_t = x[: , i , :]
  h_t = ref(x_t , h_ref)
  ref_states.append(h_t)
  h_ref = h_t
ref_states = torch.stack(ref_states , dim = 1)

# %%
ref_states

# %%
(states - ref_states).max().abs()

# %%
(states[: , -1 , :] - ref_states[: , -1 , :]).max().abs()


# %%
x.requires_grad = True

# %%
x

# %%
x = torch.rand(B , T , D)
x_my = x.detach().clone().requires_grad_()
x_ref = x.detach().clone().requires_grad_()

my_states = []
ref_states = []

# my things
h_my = torch.zeros(B, H)

for i in range(T):
  x_t = x_my[:, i, :]
  h_t = cell(x_t , h_my)
  my_states.append(h_t)
  h_my = h_t

my_states = torch.stack(my_states, dim=1)  # (B, T, H)



## RNNs
h_ref = torch.zeros(B,H)

for i in range(T):
  x_t = x_ref[: , i , :]
  h_t = ref(x_t , h_ref)
  ref_states.append(h_t)
  h_ref = h_t
ref_states = torch.stack(ref_states , dim = 1)


print((h_my - h_ref).abs().max())


my_loss = h_my.sum()
ref_loss = h_ref.sum()

my_loss.backward()
ref_loss.backward()


print((x_my.grad  - x_ref.grad).abs().max())

# %%
# Parameter gradient parity

pairs = [
    (W_ih_x.grad, ref.weight_ih.grad.T, "W_ih_x"),
    (W_hh_h.grad, ref.weight_hh.grad.T, "W_hh_h"),
    (b_ih.grad, ref.bias_ih.grad, "b_ih"),
    (b_hh.grad, ref.bias_hh.grad, "b_hh")
]

for mine, theirs, name in pairs:
    diff = (mine - theirs).abs().max().item()
    print(name, "max diff:", diff)
    assert diff < 1e-5, f"{name} gradient mismatch"

print("PASS: All parameter gradients match")

# %% [markdown]
# ## Step 3 — Gradient paths 📉
#
# Loss on the FINAL state only. Keep selected hidden grads. Plot norms over time (plot 1). Explain where the loss sits and how steps are indexed.

# %%
x = torch.rand(B , T , D)

my_states = []

h_my = torch.zeros(B, H)

for i in range(T):
  x_t = x[:, i, :]
  h_t = cell(x_t , h_my)
  h_t.retain_grad()
  my_states.append(h_t)
  h_my = h_t
  h = h_t

states_stack = torch.stack(my_states, dim=1)  # (B, T, H)


loss = h.sum()

loss.backward()

norms = []

for state in my_states:
  norms.append(state.grad.norm().item())

# %%
norms

# %%
plt.plot(norms)

# %% [markdown]
# ## Step 4 — Init comparison ⚖️
#
# Contractive setup vs a different recurrent-weight size, same input. Second gradient plot. Report what ACTUALLY happens — flat is data, not failure.

# %%
with torch.no_grad():
    W_hh_h.mul_(1.5)

# %%
x = torch.rand(B , T , D)

my_states = []

h_my = torch.zeros(B, H)

for i in range(T):
  x_t = x[:, i, :]
  h_t = cell(x_t , h_my)
  h_t.retain_grad()
  my_states.append(h_t)
  h_my = h_t
  h = h_t

states_stack = torch.stack(my_states, dim=1)  # (B, T, H)


loss = h.sum()

loss.backward()

norms = []

for state in my_states:
  norms.append(state.grad.norm().item())

# %%
norms

# %%
plt.plot(norms)

# %% [markdown]
# ## Step 5 — Results 📝
#
# Tolerances, both plots, one honest paragraph: how far back does signal reach here, and what controls it?

# %%
"""
- tol: 1e-5.
- Parity diffs: outputs(0), final hidden (0), input grads(0), each param(0). i used two bias same as in the ref so my diff is 0
- Plot 1 norms: your 1.12 -> 3.16.
- Plot 2 norms: your 4.51 -> 3.16.
- Both plots with axes: X = step, Y = grad size, note loss at last step.
- One honest paragraph: Run B with W_hh times 1.5 gave [1.1278197765350342, 1.5338034629821777, 2.982121229171753, 3.1622776985168457] to [4.5132670402526855, 4.2675299644470215, 3.701479434967041, 3.1622776985168457]


The loss is calculated on the final hidden state only (step 3). On the X-axis, step 0 is the earliest and step 3 is the last; Y represents gradient size. In the base run, gradients reach all four steps, decreasing from 3.16 at the final step to 1.12 at the earliest step, but remain above zero. With W_hh multiplied by 1.5, the earliest gradient increases to 4.51, showing that a larger recurrent weight increased early gradients. Gradient flow depends on recurrent weight size, the tanh slope (maximum 1), and the product of Jacobians. One run is not enough to claim vanishing gradients.

"""

# %% [markdown]
# ## Step 6 — Your tests ✅
#
# Write your own checks (shapes, parity, grad flow). They must pass in this notebook before handoff.

# %%
states_stack

# %%

# Step 6 — My own checks

# Shapes
assert x.shape == (B, T, D)
assert states.shape == (B, T, H)
assert states[:, -1, :].shape == (B, H)
print("PASS: Shapes")


# Input gradient
assert x_my.grad is not None
assert x_my.grad.shape == x_my.shape
print("PASS: Input gradients")


# Gradient flow
assert len(norms) == T
assert all(n > 0 for n in norms)
assert abs(norms[-1] - 3.16) < 0.01
print("PASS: Gradient flow")

print("ALL CHECKS PASSED!")


# %%
