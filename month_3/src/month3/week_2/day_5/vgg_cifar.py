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
# # 📓 Day 13 — VGG-style blocks, at a size you can train
#
# ---
# 🎯 **GOAL — what you must build (all by yourself):**
# - A compact VGG-style CNN for 32×32 CIFAR photos: small 3×3 layers stacked deep, a ReLU between each pair 🧱
# - One bounded run on the SHARED budget (10 epochs, seed 42, aug ON, SGD lr 0.01) — the budget does not move 🏃
# - A receptive-field count per stage + a counted parameter total, both worked on paper first 📐
# - A three-way comparison table (simple CNN vs AlexNet-inspired vs VGG-style) with every row labeled controlled or contextual 📊
# - An honest note: what the three rows do and do NOT prove ⚖️
#
# 📚 **QUICK BACKGROUND — read it, then close it and build:**
# - 🏛️ VGG = the 2014 family that stacked many small 3×3 layers instead of few big ones. Same view, fewer weights, more bends.
# - 👁️ Receptive field = the patch of the photo one unit sees. One 3×3 sees 3×3. Stack a second 3×3 after it (stride 1) and its units see 5×5 of the photo. A third sees 7×7. Each layer adds 2.
# - 🌀 Nonlinear composition = the ReLU sitting between the stacked layers. Two 3×3s with a ReLU between them can draw bendier boundaries than one flat 5×5 with the same view. Equal view is NOT equal power.
# - 🧮 Parameter math: two 3×3 layers at equal width C hold 2 × C × C × 9 = 18C² weights (bias aside). One 5×5 holds 25C². Example at C = 16: 4,608 vs 6,400.
# - ⚠️ The width warning: 18-vs-25 needs equal widths in and out. Your Day-12 net widened 18 → 36, so blanket claims are banned — calculate your own real numbers.
# - 🤝 Controlled = only the model differs (same split, seed, aug, epochs, LR). Contextual = an old unmatched number quoted for background (Day 11's 15-epoch runs are context, never head-to-head).
#
# 🛠️ **WORK — do it in this order:**
# 1. ✏️ Recall: one AlexNet adaptation from Day 12 + what yesterday's budget lets you claim.
# 2. 🔀 Data: rebuild your Day-12 loaders unchanged. Copy the shared budget into the notebook: 45k/5k, seed 42, aug ON, 10 epochs, SGD lr 0.01 / momentum 0.9 / wd 5e-4.
# 3. 🧱 Model: VGG-style stacked 3×3 convs with ReLU, one or two pool blocks, a modest head. Note your widths and why.
# 4. 📐 Shapes: layer/shape table with a counted parameter total + receptive field per stage. Tiny-batch fit + initial loss near ln(10) ≈ 2.30.
# 5. 🏃 Bounded run: 10 epochs on the shared budget, nothing more. Save the best checkpoint with its config.
# 6. 📊 Results: run table + confusion matrix with full class order + three-way comparison (controlled/contextual labels) + honest note.
# 7. 🧪 Your tests: checks you wrote yourself — they must pass before handoff.
#
# 📦 **FINAL RESULT — admin wants THIS from you:**
# - ✅ Recall written down
# - ✅ VGG-style net with sensible 32×32 sizes
# - ✅ Receptive-field counts + parameter total
# - ✅ One shared-budget run + best checkpoint
# - ✅ Three-way table with controlled/contextual labels + honest note
#
# 🚫 **NOT ALLOWED:**
# - ❌ Changing the budget to chase a winner (no extra epochs, no new LR)
# - ❌ Copying VGG-16/19 widths or the 224×224 geometry
# - ❌ Day-11 numbers presented as head-to-head (context label only)
# - ❌ Test set used for ANY picking decision
# - ❌ Claiming an architecture wins from one short run or one seed
# - ❌ Copying a full tutorial solution — docs are open, final code is yours
#
# 🚀 **HOW TO FINISH:** work in Colab → download the finished `.ipynb` back over this repo copy → say `handoff done` → show your result to admin. Result first, writing second. ✨
# ---
#

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# %% [markdown]
# ---
# ## Your workspace — the model, the counts, the results,
# and the tests are yours
# Steps 1 and 4 carry over your own Day-12 code
# (loaders, loop, checkpoint save) so the budget stays fixed.
# No read-only cells. No hidden answers.
# ---

# %% [markdown]
# ## Step 0 — Recall
# One AlexNet adaptation from Day 12, plus what yesterday's budget lets you claim. Your own words.

# %%
"""Dropout and the budget is 10  epochs and lr is 0.01"""

# %% [markdown]
# ## Step 1 — Data (same pipeline, fixed budget)
# Rebuild your Day 12 loaders unchanged. Copy the shared budget into the notebook: 45k/5k split, seed 42, aug ON, 10 epochs, SGD lr 0.01 / momentum 0.9 / wd 5e-4.

# %%
AUGMENTATION = True
## HYPER PARAMS
BATCH_SIZE = 128
NUM_WORKERS = 0   # 0 first — safe on Windows/notebooks
HIDDEN = 64
DROPOUT_P = 0.5

SEED = 42

### TRAIN
LR = 0.01
WEIGHT_D = 5e-4
MOMENTUM = 0.9


## CIFAR-10 things
PHOTO_SIZE = 32

CONV_K = 3
CONV_S = 1
CONV_P = 1

POOL_K = 2
POOL_S = 2
POOL_P = 0

CONV1_OUT = (PHOTO_SIZE + 2*CONV_P - CONV_K) // CONV_S + 1
POOL1     = (CONV1_OUT + 2*POOL_P - POOL_K) // POOL_S + 1
CONV2_OUT = (POOL1 + 2*CONV_P - CONV_K) // CONV_S + 1
POOL2     = (CONV2_OUT + 2*POOL_P - POOL_K) // POOL_S + 1


# %%
from google.colab import drive
drive.mount('/content/drive')


# %%
def load_cifar(augmentation = False  ,seed = 42 , batch_size = 64 ,num_workers = 0, train_count = 45000 , val_count = 5000 ):
  ROOT = "/content/drive/MyDrive/data"

  mean = (0.4914, 0.4822, 0.4465) #— red, green, blue
  std = (0.2470, 0.2435, 0.2616) #— red, green, blue

  if augmentation:
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # cut a 32×32 window out of a 40×40 padded photo (random spot) if
        transforms.RandomHorizontalFlip(),     # coin flip: 50% chance mirror left-right
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
  else:
    train_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

  val_tfm = transforms.Compose([
      transforms.ToTensor(),                 # same ending, no random steps
      transforms.Normalize(mean, std),
  ])

  train_set = datasets.CIFAR10(ROOT, train=True, download=False, transform=train_tfm)
  val_set = datasets.CIFAR10(ROOT, train=True, download=False, transform=val_tfm)

  g = torch.Generator().manual_seed(seed)
  perm = torch.randperm(50000, generator=g).tolist()

  val_idx = perm[:val_count]       # first 5000 → check
  train_idx = perm[50000 - train_count:]     # rest 45000 → train


  train_data = Subset(train_set, train_idx)
  val_data = Subset(val_set, val_idx)


  train_loader  = DataLoader(train_data, batch_size=batch_size ,shuffle=True ,num_workers=num_workers )
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  test_set = datasets.CIFAR10(ROOT, train=False, download=False, transform=val_tfm)
  test_set

  return train_loader, val_loader , test_set

# %%
train_loader , val_loader , test_set = load_cifar(AUGMENTATION , SEED , BATCH_SIZE , NUM_WORKERS )


# %% [markdown]
# ## Step 2 — Model (VGG-style stacked 3×3)
# Stacked 3×3 convs with ReLU between them, one or two pool blocks, a modest head. Note your widths and why.

# %%

class VGG(nn.Module):
  def __init__(self):
    super().__init__()
    self.nn = nn.Sequential(
        ## Conv1 setup
        nn.Conv2d(in_channels=3 , out_channels=12 , kernel_size=3 , stride=1 , padding=1), # -- 32x32
        nn.ReLU(),
        nn.Conv2d(in_channels=12 , out_channels=18 , kernel_size=3 , stride=1 , padding=1), # -- 32x32
        nn.ReLU(),
        nn.MaxPool2d(2), # -- 32x32 -->> 16 x 16

        ## Conv2 setup
        nn.Conv2d(in_channels=18 , out_channels=16 , kernel_size=3 , stride=1 , padding=1), # -- 16x16
        nn.ReLU(),
        nn.Conv2d(in_channels=16 , out_channels=36 , kernel_size=3 , stride=1 , padding=1), # -- 16x16
        nn.ReLU(),
        nn.MaxPool2d(2), # -- 16 x 16 -->> 8 x 8

        ## flattern
        nn.Flatten(),

        ## fully connected layer
        nn.Linear(in_features= 36 * 8 * 8 , out_features=HIDDEN),
        nn.ReLU(),
        nn.Dropout(p = DROPOUT_P),


        ## final layer
        nn.Linear(HIDDEN , 10),
    )

  def forward(self , x):
    if x.dim() != 4 or x.shape[1] != 3:
      raise ValueError(f"expected (N,3,32,32), got {tuple(x.shape)}")
    return self.nn(x)

# %% [markdown]
# ## Step 3 — Shapes, params, receptive field
# Layer/shape table with a counted parameter total + receptive field per stage (5 after two 3×3s, 7 after three). Tiny-batch memorize check + initial loss near ln(10).

# %%
"""

parameters
- conv 3→12: 12·3·9 + 12 = 336
- conv 12→18: 18·12·9 + 18 = 1,962
- conv 18→16: 16·18·9 + 16 = 2,608
- conv 16→36: 36·16·9 + 36 = 5,220
- fc 2304→64: 147,520 · final 64→10: 650

Conv subtotal 10,126 + head 148,170 = 158,296



- Start: 1 (a single pixel)
- conv 3×3 → 3, conv 3×3 → 5 (each stride-1 3×3 adds 2)
- pool 2×2 → 10 (pooling doubles everything so far)
- conv 3×3 → 12, conv 3×3 → 14
- pool 2×2 → 28
"""

torch.manual_seed(SEED)
xb, yb = next(iter(train_loader))
print(xb.shape , yb.shape)
xb, yb = xb[:32], yb[:32]
print(xb.shape , yb.shape)

model = VGG()

opt = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()

for s in range(200):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    opt.step()
    with torch.no_grad():
        acc = (model(xb).argmax(1) == yb).float().mean().item()
        print(acc)
    if acc >= 1.0:
        print(f"memorized in {s+1} steps ✅")
        break

# %% [markdown]
# ## Step 4 — Bounded run (shared budget, 10 epochs)
# Same optimizer discipline as Day 12. Save the best-checkpoint with its config. The budget does not move.

# %%
model = VGG()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr = LR, weight_decay=WEIGHT_D, momentum=MOMENTUM)

# %%
import copy
import time
EPOCHS = 10


best_state = None
best = 0
final_train_loss = 0
final_val_loss = 0
best_train_loss = 0
best_val_loss = 0
best_epoch = None

start = time.perf_counter()

best_conf_mtx = None

def train(epoch_no):
  model.train()
  loss_sum = 0
  n_seen = 0
  for j ,(xs ,ys) in enumerate(train_loader):

    ## Zeroing the grads
    opt.zero_grad()

    # prediction based on featurs
    logits = model(xs)
    #loss and backward
    loss = loss_fn(logits , ys)
    loss.backward()

    # optimzer step
    opt.step()

    loss_sum += loss.item() * len(ys)
    n_seen += len(ys)
  current_epoch_loss = loss_sum / n_seen
  print(f"Loss of epoch no. {epoch_no} : {current_epoch_loss:.4f}")
  return current_epoch_loss

def eval():
  was_training = model.training
  model.eval()
  correct, total = 0, 0
  all_true = []
  all_pred = []
  val_loss_sum = 0

  with torch.no_grad():
      for xb, yb in val_loader:
        logits = model(xb)
        pred = logits.argmax(dim=1)   # biggest score per row = guess
        all_pred.extend(pred.tolist())
        all_true.extend(yb.tolist())
        val_loss_sum += loss_fn(logits, yb).item() * len(yb)
        correct += (pred == yb).sum().item()
        total += len(yb)

  conf_mtx = confusion_matrix(all_true, all_pred, labels=list(range(10))).tolist()
  val_loss = val_loss_sum / total
  acc = correct / total
  model.train(was_training)
  return acc , val_loss , conf_mtx

for i in range(EPOCHS):
  final_train_loss = train(i)
  acc, final_val_loss , conf_mtx = eval()
  print(f"accuray of epoch :{acc} \n")
  if acc > best:
    best = acc
    best_conf_mtx = conf_mtx
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = i
    best_train_loss = final_train_loss
    best_val_loss = final_val_loss



end = time.perf_counter()

total_time = f"{end - start:.6f}"

# %%

current_time = time.ctime()
checkpoint = f"alexnet_v1_ep{best_epoch}_acc{best:.4f}_{current_time}.pth"


config = {
    "run": "aug_on" if AUGMENTATION else "aug_off",
    "seed": SEED,
    "split": "45k/5k",
    "model": {"c1": 18, "c2": 36, "hidden": HIDDEN},
    "augmentation": AUGMENTATION,
    "optimizer": {"name": opt.__class__.__name__, "lr": LR, "momentum": MOMENTUM, "weight_decay": WEIGHT_D},
    "epochs": EPOCHS,
    "batch": BATCH_SIZE,
    "best_epoch": best_epoch,
    "best_val_acc": best,
    "final_train_loss": final_train_loss,
    "final_val_loss": final_val_loss,
    "best_train_loss": best_train_loss,
    "best_val_loss": best_val_loss,
    "time_secs": total_time,
    "checkpoint": checkpoint,
    "confusion_matrix" : best_conf_mtx
}

model_dict = {
    "state_dict": best_state,
    "config" : config
}
torch.save(model_dict, checkpoint)

# %%
from google.colab import files

files.download(checkpoint)


# %%
config

# %% [markdown]
# ## Step 5 — Results
# Run table + confusion matrix with full class order + three-way comparison table with controlled/contextual labels + honest note.

# %%
"""
epochs          : 10
Best Trian loss : 0.8874
Best check loss : 0.8419
check acc       : 0.7092
time            : 842.05 sec

Confssion Matrix
[[409, 8, 58, 8, 5, 1, 1, 9, 26, 5],
 [23, 408, 7, 8, 0, 0, 2, 1, 20, 14],
 [28, 0, 383, 17, 14, 16, 8, 7, 5, 0],
 [14, 0, 91, 268, 11, 50, 20, 17, 4, 3],
 [15, 2, 119, 26, 256, 7, 33, 32, 3, 0],
 [10, 0, 79, 92, 13, 261, 10, 29, 2, 1],
 [7, 0, 95, 29, 12, 4, 364, 2, 4, 0],
 [5, 0, 46, 20, 19, 16, 3, 404, 3, 2],
 [41, 10, 15, 4, 3, 2, 2, 3, 427, 4],
 [20, 60, 11, 4, 4, 1, 2, 13, 14, 366]]






┌─────────────────────────────┬────────────┬────────────┬───────────┬──────────┬────────┬─────────────────────────────────┐
│             net             │ train loss │ check loss │ check acc │  params  │  time  │           race label            │
├─────────────────────────────┼────────────┼────────────┼───────────┼──────────┼────────┼─────────────────────────────────┤
│ simple CNN (Day 11, aug ON) │ 1.90       │ —          │ 0.6100    │ 138010   │ 983.0s │ contextual (15 epochs, lr 0.05) │
├─────────────────────────────┼────────────┼────────────┼───────────┼──────────┼────────┼─────────────────────────────────┤
│ AlexNet-inspired (Day 12)   │ 1.2326     │ 0.9931     │ 0.6488    │ ~154,500 │ 540.9s │ controlled                      │
├─────────────────────────────┼────────────┼────────────┼───────────┼──────────┼────────┼─────────────────────────────────┤
│ VGG-style (today)           │ 0.8874     │ 0.8419     │ 0.7092    │ 158,296  │ 842.0s │ controlled                      │
└─────────────────────────────┴────────────┴────────────┴───────────┴──────────┴────────┴─────────────────────────────────┘







HONEST GAP :


1. What happened (just numbers, no adjectives):
"VGG: best check 0.7092 acc at ep 9, 842.05s. AlexNet-inspired (Day 12): 0.6488 acc, 10 epochs , Day 11 simple CNN: OFF 0.6484, ON 0.6100, 15 epochs each. Todays net is better then yesterday one"

2. Why it's not a fair fight (your confounds, name them):
"when compaired to Day 12 (AlexNet) this has some model structure changes that is introducing two conv layer before each max pool and the parameteres are slightly increased not too much but approx 4,000 all other remains same "

3. What it proves and what would settle it:
"This suggests using two or more conv layers with relu non-linearity (activation fn) is better than one conv layer and also using two 3x3 conv layer instead of one 5x5 make the model bit better and also reduces parameter counts and same for using the 3x3 three times rather than one 7x7 , i would suggest this VGG over Alexnet "

"""


# %% [markdown]
# ## Step 6 — Your tests
# Checks you wrote yourself (shapes, guards, best-saved, tiny fit). They must pass before handoff.

# %%
def smoke_test():
  model = VGG()
  assert next(iter(train_loader))[0][0].shape == torch.Size([3,32,32])


  try:
      model(torch.randn(2, 3, 32))   # missing an axis on purpose
      raise AssertionError("should have refused 3D input")
  except ValueError:
      pass


  assert best_state is not None
  assert conf_mtx is not None


  ## BATCH OVER FIT
  xb, yb = next(iter(train_loader))
  xb, yb = xb[:32], yb[:32]



  opt = torch.optim.SGD(model.parameters(), lr=0.05)
  loss_fn = nn.CrossEntropyLoss()

  for s in range(200):
      model.train()
      opt.zero_grad()
      loss = loss_fn(model(xb), yb)
      loss.backward()
      opt.step()
      with torch.no_grad():
          acc = (model(xb).argmax(1) == yb).float().mean().item()
      if acc >= 1.0:
          print(f"memorized in {s+1} steps ✅")
          break
  assert acc == 1

  print("all test passed")

# %%
smoke_test()

# %%
