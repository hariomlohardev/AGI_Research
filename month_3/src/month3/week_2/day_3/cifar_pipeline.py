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
# # 📓 Day 11 — CIFAR-10 pipeline + augmentation test
#
# ---
# 🎯 **GOAL — what you must build (all by yourself):**
# - A clean CIFAR-10 photo pipeline you can reuse for AlexNet / VGG / ResNet days 🔁
# - One fair test: same model, same split, same seed — augmentation OFF vs ON ⚖️
# - Test data stays sealed the whole day 🔒 — you pick everything with validation only.
#
# 📚 **QUICK BACKGROUND — read this, then close it and build:**
# - 🖼️ CIFAR-10 = 60,000 small color photos (32x32). 10 groups: plane, car, bird, cat, deer, dog, frog, horse, ship, truck. 50,000 train + 10,000 test.
# - ✂️ Split = cut the 50,000 train into 45,000 train + 5,000 check. Check = validation = the small part you use to pick the model. Test = the final part you open only at the very end.
# - 📏 Normalization = calm the numbers so training is stable. Formula: `(pixel - mean) / std`. Compute mean/std from train 45k ONLY — or use fixed numbers and write down where they came from. Never fit them on check or test.
# - 🌾 Augmentation = extra fake copies for training only. Example: flip a dog photo left-right, or cut it with a small shift (crop). The model sees more versions and learns the dog, not the exact spot. Use RandomCrop(32, padding=4) + RandomHorizontalFlip in train. Check uses clean deterministic steps only.
# - 👁️ Denormalize = undo normalization for YOUR eyes: `picture * std + mean`. The model sees normalized numbers, you need normal colors for plots.
# - 🔢 Logits = raw scores before softmax. Softmax turns one row of scores into weights that sum to 1.
# - 🧪 Tiny-batch fit = take ~32 photos, no augmentation, and force the model to memorize them. This checks your plumbing learns. It says NOTHING about new data — but if it fails, your code is broken, fix it first.
# - 📊 Baseline = the simple score to beat. Here: initial loss near ln(10) ≈ 2.30 for 10 groups, plus the tiny-batch fit.
#
# 🛠️ **WORK — do it in this order:**
# 1. 🔢 Fix the split: make 45k/5k index lists from ONE seed. Reuse the SAME lists for both runs. Write the seed down.
# 2. 🔀 Build two separate dataset objects: train (random crop + flip ➕ ToTensor ➕ Normalize) and check (ToTensor ➕ Normalize only). Prove they do NOT share one mutable transform — changing train must never touch check.
# 3. 🖼️ Decode and show: one train batch + one check batch, labels on, denormalized for display. Plot ~30 train photos across all 10 classes and write which pairs look alike (a guess, not a measured score).
# 4. 🧪 Tiny-batch check: memorize a small fixed no-augmentation set. Record steps + accuracy. Also check initial logits/loss near ln(10).
# 5. 🏃🏃 Two bounded runs, max 15 epochs each: (A) augmentation OFF, (B) augmentation ON. Same split, same seed, same optimizer (SGD lr=0.05, momentum=0.9, weight_decay=5e-4 is a fine default). Set a wall-time cap BEFORE running — if a run stops early, compare the same finished epochs and write both times down.
# 6. 📊 After each run record: seed, changed variable, epochs/updates done, train loss, check loss, check accuracy, time, checkpoint path. Save the BEST check checkpoint for each run.
# 7. 🧩 Check confusion matrix with full class order for each run. Compare your errors with your first guess from step 3.
# 8. 📝 Honest note: which won, by how much, and what the gap does NOT prove. A ~1% gap can be noise — say so. Never claim a win from one short run or one seed.
#
# 📦 **FINAL RESULT — admin wants THIS from you:**
# - ✅ 45k/5k indices fixed, reused for both runs, seed written down
# - ✅ Train has crop + flip, check is deterministic, separation proven
# - ✅ Normalization is train-only or fixed-with-source, never fitted on held-out
# - ✅ Decoded train + check batches shown and labeled
# - ✅ Tiny-batch fit passes, initial loss checked
# - ✅ Two 15-epoch runs with run table + confusion matrices + 2 saved best checkpoints
# - ✅ Honest note (winner, gap, why small gaps are not proof, one limitation)
#
# 🚫 **NOT ALLOWED:**
# - ❌ One shared dataset object with two views where changing the transform hits both
# - ❌ Random augmentation in check or test
# - ❌ Mean/std fitted on check or test
# - ❌ Test set used for ANY picking decision
# - ❌ Claiming augmentation wins from one short run or one seed
# - ❌ Copying a full tutorial solution — docs are open, final code is yours
#
# 🚀 **HOW TO FINISH:** work in Colab → download the finished `.ipynb` back over this repo copy → say `handoff done` → show your result to admin. Result first, writing second. If the result is correct, the day is good. ✨
# ---
#

# %%
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# %%
from google.colab import drive
drive.mount('/content/drive')


# %% [markdown]
# ---
# ## ✏️ Your workspace — everything below is yours
# Write every function and every test yourself. No read-only cells. No hidden answers.
# ---
#

# %%
AUGMENTATION = True
## HYPER PARAMS
BATCH_SIZE = 128
NUM_WORKERS = 0   # 0 first — safe on Windows/notebooks
HIDDEN = 128

SEED = 42

### TRAIN
LR = 0.05
WEIGHT_D = 5e-4
MOMENTUM = 0.9

EPOCHS = 15

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


## MODEL THINGS
C1 = 32
C2 = 16

# %%
mean = (0.4914, 0.4822, 0.4465) #— red, green, blue
std = (0.2470, 0.2435, 0.2616) #— red, green, blue


# %%
# # Your functions here (model, transforms, loaders, train/eval, helpers...)
ROOT = "/content/drive/MyDrive/data"  # lives on your Drive forever

if AUGMENTATION:
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

g = torch.Generator().manual_seed(SEED)   # your seed — write it down
perm = torch.randperm(50000, generator=g).tolist()

val_idx = perm[:5000]       # first 5000 → check
train_idx = perm[5000:]     # rest 45000 → train


train_data = Subset(train_set, train_idx)   # same indices, train clothes
val_data = Subset(val_set, val_idx)          # same indices, check clothes


train_loader  = DataLoader(train_data, batch_size=BATCH_SIZE ,shuffle=True ,num_workers=NUM_WORKERS )
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_set = datasets.CIFAR10(ROOT, train=False, download=False, transform=val_tfm)  # 🔒 sealed — load it, never pick with it



# %%
## MODEL >>>>>>>

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=C1, kernel_size=CONV_K, stride=CONV_S, padding=CONV_P),
        nn.ReLU(),
        nn.MaxPool2d(POOL_K),

    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(C1 , C2, kernel_size=CONV_K, stride=CONV_S, padding=CONV_P),
        nn.ReLU(),
        nn.MaxPool2d(POOL_K)
    )
    self.fc1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(C2 * POOL2 * POOL2, HIDDEN),
        nn.ReLU()
    )
    self.fc2 = nn.Linear(HIDDEN, 10)               # 128 → 10 answers (was missing)


  def forward(self , x):
    z = self.conv1(x)
    z = self.conv2(z)
    z = self.fc1(z)
    return self.fc2(z)


# %%
model = Model()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr = LR, weight_decay=WEIGHT_D ,momentum=MOMENTUM )

# %%

import copy
import time

best_state = None
best = 0
final_train_loss = 0
best_epoch = None

start = time.perf_counter()



def train(epoch_no):
  model.train()
  history = []
  for j ,(xs ,ys) in enumerate(train_loader):

    ## Zeroing the grads
    opt.zero_grad()

    # prediction based on featurs
    logits = model(xs)
    probs = torch.softmax(logits, dim=1)   # (N,10) rows now sum to 1

    #loss and backward
    loss = loss_fn(probs , ys)
    loss.backward()

    # optimzer step
    opt.step()

    history.append(loss.item())
  current_epoch_loss = sum(history)/len(history)
  print(f"Loss of epoch no. {epoch_no} : {current_epoch_loss:.4f}")
  return current_epoch_loss

def eval():
  model.eval()
  correct, total = 0, 0
  with torch.no_grad():
      for xb, yb in val_loader:
        pred = model(xb).argmax(dim=1)   # biggest score per row = guess
        correct += (pred == yb).sum().item()
        total += len(yb)
  acc = correct / total
  return acc

for i in range(EPOCHS):
  final_train_loss = train(i)
  acc = eval()
  print(f"accuray of epoch :{acc} \n")
  if acc > best:
    best = acc
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = i


end = time.perf_counter()

total_time = f"{end - start:.6f}"

# %%

current_time = time.ctime()
checkpoint  = "best_aug_" + ("on" if AUGMENTATION else "off") +str(current_time) +".pth"


config = {
    "run": "aug_on" if AUGMENTATION else "aug_off",          # which arm: aug_off / aug_on
    "seed": SEED,                # your seed
    "split": "45k/5k",         # train/check sizes
    "model": {"c1": C1, "c2": C2, "hidden": HIDDEN},
    "augmentation": AUGMENTATION,     # the ONE changed variable
    "optimizer": {"name": opt.__class__.__name__, "lr": LR, "momentum": MOMENTUM, "weight_decay": WEIGHT_D},
    "epochs": EPOCHS,
    "batch": BATCH_SIZE,
    "best_epoch": best_epoch,           # fill after training
    "best_val_acc": best,       # fill after training
    "final_train_loss": final_train_loss,   # fill after training
    "time_secs": total_time,          # fill after training
    "checkpoint": checkpoint,
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

# %%
# Your tests here (checks you wrote yourself — they must pass before handoff)
def tests():
  assert mean == (0.4914, 0.4822, 0.4465), "mean are not correct fix it"
  assert std == (0.2470, 0.2435, 0.2616), "std are not correct fix it"
  out = model(torch.randn(2, 3, 32, 32))
  assert tuple(out.shape) == (2, 10) and torch.isfinite(out).all()




  assert best_state is not None, "best never saved!"


  # grab ONE tiny fixed batch (take once, reuse it every step)
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

tests()

# %%
# Your runs + results here (tables, plots, checkpoints, honest note)


"""
┌─────┬────────────────┬────────────┬─────────┐
│ arm │    best acc    │ final loss │  time   │
├─────┼────────────────┼────────────┼─────────┤
│ OFF │ 0.6484 (ep 14) │ 1.75       │ ~14m20s │
├─────┼────────────────┼────────────┼─────────┤
│ ON  │ 0.6100 (ep 14) │ 1.90       │ ~16m23s │
└─────┴────────────────┴────────────┴─────────┘

plots are not recorded

checkpoints are two
"C:\\Users\\hario\\Downloads\\best_aug_onThu Nov 19 14_18_38 2026.pth"
"C:\\Users\\hario\\Downloads\\best_aug_offThu Nov 19 12_42_52 2026.pth"

honest note is we run this on an lower epoch no. of 15 so we see more accuracy in the OFF arm and lower final loss and low time too than hte ON arm which takes more time and worse here but
we can't surely say if off is better or not because its an small run on short epoch no. so its too small to surely say something
"""
