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
# # 📓 Day 12 — AlexNet ideas, at a size you can train
#
# ---
# 🎯 **GOAL — what you must build (all by yourself):**
# - A compact AlexNet-inspired CNN for 32×32 CIFAR photos — ideas stolen, sizes resized 🔁
# - One bounded run (≤10 epochs or your predeclared time cap) with a run table + confusion matrix + saved best checkpoint 🏃
# - An adaptation table: original idea → your change → reason (grounded in the paper) 📝
# - An honest note: what the short run does and does NOT prove ⚖️
#
# 📚 **QUICK BACKGROUND — read this, then close it and build:**
# - 🏆 AlexNet = the first deep CNN that won big (ImageNet 2012). Its ideas: ReLU instead of tanh, stacked conv + pooling blocks, dropout in the big classifier head. LRN was its neighbor-normalization trick — dead history now. Know the story, don't use the trick.
# - 📐 ReLU = `max(0, x)`. Slope 1 for x > 0 (no positive-side saturation like tanh), slope 0 for x < 0 (a unit stuck there is a dead unit — ReLU is better, not magic).
# - 🧠 Big head = big weights. One fc layer from F inputs to C outputs holds F × C + C numbers. A 4096-wide head can outweigh all conv layers — shrink the map or the head first.
# - 📏 Sizing formula per side: floor((input + 2×pad − (kernel−1) − 1) / stride) + 1. Check every block on paper — 32×32 photos leave no room for 11×11 stride-4 openers.
# - 🔁 Reuse your Day 11 pipeline + training loop unchanged: same 45k/5k split, same seed, train-only stats, deterministic check, sealed test set. Only the model is new.
# - 🤝 Pick the shared comparison budget with tomorrow's VGG run TODAY and write it down — fair comparisons need matching budgets.
#
# 🛠️ **WORK — do it in this order:**
# 1. ✏️ Recall: your most common CIFAR validation error + one limitation of the Day 11 OFF-vs-ON comparison.
# 2. 🔀 Data: rebuild your Day 11 loaders (same split, same seed, train-only stats, deterministic check, sealed test).
# 3. 🧱 Model: compact AlexNet-inspired net for 32×32. Write the adaptation table as you go (idea → change → reason).
# 4. 📐 Shapes: layer/shape table with a counted parameter total. Tiny-batch fit + initial loss near ln(10) ≈ 2.30.
# 5. 🏃 Bounded run: ≤10 epochs or your time cap. Save the best-checkpoint with its config.
# 6. 📊 Results: run table (train loss, check loss, check acc, time) + confusion matrix with full class order + honest note.
# 7. 🧪 Your tests: checks you wrote yourself — they must pass before handoff.
#
# 📦 **FINAL RESULT — admin wants THIS from you:**
# - ✅ Recall written down
# - ✅ Compact model with sensible 32×32 sizes (no 224×224 copying)
# - ✅ Layer/shape table + parameter count
# - ✅ Bounded run table + confusion matrix + saved best checkpoint
# - ✅ Adaptation table grounded in the paper (ReLU, conv structure, dropout, LRN's role)
# - ✅ Honest note (losing small-budget settles nothing)
#
# 🚫 **NOT ALLOWED:**
# - ❌ Copying the ImageNet 224×224 geometry
# - ❌ LRN as required work (optional extra only, ≤1 matched run)
# - ❌ Test set used for ANY picking decision
# - ❌ Stats fitted on check or test
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
# ## ✏️ Your workspace — everything below is yours
# Write every function and every test yourself. No read-only cells. No hidden answers.
# ---

# %% [markdown]
# ## Step 0 — Recall
# Your most common CIFAR validation error, plus one limitation of the Day 11 OFF-vs-ON comparison. Your own words.

# %%
"""
Recall:
- Most common CIFAR validation error I saw: Day 11 kept only overall accuracy, so I have no measured most-common error. My Step 5 confusion matrix on this day will be my first measured one.
- One limitation of my Day 11 OFF-vs-ON comparison: we train on the lower number of epochs
"""

# %% [markdown]
# ## Step 1 — Data (reuse your Day 11 pipeline)
# Same 45k/5k split, same seed, train-only stats, deterministic check transform, sealed test set. Prove train and check do not share one mutable transform.

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
# ## Step 2 — Model (compact AlexNet-inspired, 32×32)
# ReLU, stacked conv + pooling blocks, a modest head. Write the adaptation table as comments: original idea → your change → reason.

# %%
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=18, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=18, out_channels=36, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    self.fc1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(36 * 8 * 8, HIDDEN),
        nn.ReLU(),
        nn.Dropout(p = DROPOUT_P)
    )
    self.fc2 = nn.Linear(HIDDEN, 10)


  def forward(self , x):
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"expected (N,3,32,32), got {tuple(x.shape)}")
    z = self.conv1(x)
    z = self.conv2(z)
    z = self.fc1(z)
    return self.fc2(z)


# %% [markdown]
# ## Step 3 — Shapes, params, tiny fit
# Layer/shape table with a counted parameter total. Tiny-batch memorize check + initial loss near ln(10).

# %%
model = Model()

# %%

xb, yb = next(iter(train_loader))
print(xb.shape , yb.shape)
xb, yb = xb[:32], yb[:32]
print(xb.shape , yb.shape)


opt = torch.optim.SGD(model.parameters(), lr=0.1)
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
# ## Step 4 — Bounded run (≤10 epochs or your time cap)
# Same optimizer discipline as Day 11. Save the best check checkpoint with its config. Write the Day 5 shared budget down before you start.

# %%
model = Model()
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
config

# %%
from google.colab import files

files.download(checkpoint)


# %% [markdown]
# ## Step 5 — Results
# Run table (train loss, check loss, check acc, time) + confusion matrix with full class order + honest note + finished adaptation table.

# %%
"""
epochs          : 10
Best Trian loss : 1.2326
Best check loss : 0.9931
check acc       : 0.6488
time            : 540.90 sec

Confssion Matrix
[[387, 16, 17, 7, 2, 6, 3, 11, 52, 29],
  [11, 398, 0, 1, 2, 0, 2, 2, 12, 55],
  [55, 5, 197, 38, 24, 46, 46, 51, 11, 5],
  [14, 5, 18, 186, 13, 105, 59, 43, 9, 26],
  [25, 6, 44, 34, 206, 14, 51, 99, 6, 8],
  [11, 9, 13, 96, 15, 263, 14, 59, 8, 9],
  [7, 10, 29, 36, 8, 10, 394, 6, 3, 14],
  [9, 4, 5, 17, 11, 27, 4, 414, 4, 23],
  [39, 24, 4, 7, 1, 5, 3, 4, 397, 27],
  [13, 59, 2, 0, 0, 0, 1, 7, 11, 402]]




HONEST GAP :


1. What happened (just numbers, no adjectives):
"AlexNet-inspired: best check 0.6488 at ep 9, 540.9s. Day 11 simple CNN: OFF 0.6484, ON 0.6100, 15 epochs each. The new net scored bit higher."

2. Why it's not a fair fight (your confounds, name them):
"Different budget (10 vs 15 epochs), different learning rate (0.01 vs 0.05) — two variables changed, not one. Day 11's numbers are context, not a controlled baseline."



3. What it proves and what would settle it:
"Tied Day 11 OFF (0.6488 vs 0.6484) on fewer epochs with a smaller LR. Proves nothing about the architecture yet. The VGG run on the shared budget below is the actual comparison."


SHARED BUDGET (Day 13 VGG): same 45k/5k split, seed 42, aug ON, 10 epochs, SGD lr 0.01, momentum 0.9, wd 5e-4.


# IDEA | CHANGE | REASON (Step 2) | VERDICT (Step 5)
# dropout 0.5 in head | added after fc1 ReLU | head holds most weights | KEPT — train/val gap stayed small
# LRN | dropped | dead history, no modern evidence | CONFIRMED — nothing in the run missed it
# 2 blocks, not 5 | CIFAR is 10 classes at 32x32 | REVISIT? — val plateaued early, depth may help later

"""


# %% [markdown]
# ## Step 6 — Your tests
# Checks you wrote yourself (shapes, guards, best-saved, tiny fit). They must pass before handoff.

# %%
def smoke_test():
  assert next(iter(train_loader))[0][0].shape == torch.Size([3,32,32])


  try:
      model(torch.randn(2, 3, 32))   # missing an axis on purpose
      raise AssertionError("should have refused 3D input")
  except ValueError:
      pass   # refused loudly — correct


  assert best_state is not None


  xb, yb = next(iter(train_loader))
  xb, yb = xb[:32], yb[:32]


  opt = torch.optim.SGD(model.parameters(), lr=0.1)
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
