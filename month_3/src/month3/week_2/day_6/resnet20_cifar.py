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
#     name: python3
# ---

# %% [markdown]
# # 📓 Day 14 — ResNet-20: the skip that lets nets go deep
#
# ---
# 🎯 **GOAL — what you must build (all by yourself):**
# - A BasicBlock (two 3×3 convs + a skip that adds x back) and a full CIFAR ResNet-20: 6n+2 with n = 3 🧱
# - Three stages (16 / 32 / 64 channels, three blocks each) + global average pooling + 10-class head 🏗️
# - Shortcuts the paper's way (option A: stride subsample + zero padding) — or a learned projection you label as YOUR deviation 🏷️
# - A counted parameter total near 0.27M + a shape table where both branches match at every addition 📐
# - A timing pilot, a wall-time estimate, and a max duration — decided BEFORE the real run ⏱️
# - One bounded run (≤ 2 GPU-hours) + a checkpoint that reloads exactly on a fixed validation batch 💾
# - Curves against the chance baseline (10% acc, loss ≈ 2.30) + validation metrics + a status label 📊
#
# 📚 **QUICK BACKGROUND — read it, then close it and build:**
# - 🧪 Residual block = the net learns a correction F(x) and adds the input back: y = x + F(x). If the correction is zero, x flows through untouched.
# - 🛣️ Shortcut (skip) = the wire carrying x around the correction layers to the addition. In code: `out = x + self.correction(x)`.
# - 📉 Jacobian line: for y = x + F(x), J = I + J_F. The identity I is a direct path that HELPS — but cancellation is possible, so it promises nothing.
# - ⚠️ ReLU warning: the original block puts a ReLU AFTER the addition, so the block derivative includes that activation mask. Never write "skips guarantee gradients never vanish."
# - 🏗️ The count: 6n+2 with n = 3 → 1 first conv + 3 stages × 3 blocks × 2 convs = 19 conv layers + head = 20.
# - 📋 Paper recipe (reference, not your pass mark): SGD momentum 0.9, wd 1e-4, batch 128, LR 0.1 with drops at 32k/48k steps, 64k endpoint, no dropout. 0.27M params, 91.25% accuracy. Your shorter schedule is an adaptation — record every deviation.
# - 🔀 Option A at stage changes: shrink space with stride 2, fill new channels with ZEROS. No learned weights in the shortcut.
# - 🧭 Still-learning = loss falling, above chance, curves moving right, just slow. Probably-broken = stuck at chance, NaN, or the tiny batch won't memorize.
#
# 🛠️ **WORK — do it in this order:**
# 1. ✏️ Recall: Jacobian of a linear map + why a depth comparison must control more than the dataset name.
# 2. 🔀 Data: same CIFAR pipeline as all week. Write down preprocessing — future-you needs it to reuse the checkpoint.
# 3. 🧱 Model: BasicBlock, then the three stages. Check branch shapes match at every addition.
# 4. 📐 Shapes: layer/shape table + counted total near 0.27M + forward/backward + TINY-BATCH MEMORIZATION. No memorize → no real run.
# 5. ⏱️ Budget: recipe card (every setting) + timing pilot after warmup + wall-time estimate + max duration. Cap: 2 GPU-hours, one diagnosed restart only.
# 6. 🏃 Bounded run: train, watch curves vs chance, save a resumable checkpoint with config + preprocessing.
# 7. 📊 Results: reload check on a fixed val batch + validation metrics + status label ("validated development checkpoint" or "candidate teacher; needs additional training/assessment") + honest note.
# 8. 🧪 Your tests: checks you wrote yourself — they must pass before handoff.
#
# 📦 **FINAL RESULT — admin wants THIS from you:**
# - ✅ Recall written down (linear-map Jacobian + controlled depth comparison)
# - ✅ ResNet-20 with correct wiring and matching branch shapes
# - ✅ Shape table + param total near 0.27M
# - ✅ Tiny-batch memorized BEFORE the run
# - ✅ Recipe card + timing pilot + bounded run inside the cap
# - ✅ Checkpoint that reloads exactly + validation metrics + status label
#
# 🚫 **NOT ALLOWED:**
# - ❌ Swapping in a stock ImageNet ResNet-18 (different architecture)
# - ❌ Calling a compressed schedule a replication of the paper protocol
# - ❌ Picking anything on the test set (validation only)
# - ❌ Claiming "research-grade" from architecture or accuracy alone
# - ❌ Training past the 2 GPU-hour cap or starting undeclared extra runs
# - ❌ Copying a full tutorial solution — docs are open, final code is yours
#
# 🚀 **HOW TO FINISH:** work in Colab → download the finished `.ipynb` back over this repo copy → say `handoff done` → show your result to admin. Result first, writing second. ✨
# ---
#

# %%
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F


# %%

# %% [markdown]
# ---
# ## Your workspace — the block, the budget, the checkpoint, and the tests are yours
#

# %% [markdown]
# ## Step 0 — Recall
# Jacobian of a linear map (write it, don't quote it), plus what a depth comparison must control besides the dataset name.

# %%
"""
What a Jacobian is : A jacobian is an matric of first order partial derivative of multivariable , vector functions
y = Wx → J = W : wigling x wigels y W times becase the J = W
y = x + F(x) → J = I + J_F : I or the Identity matrix just make the flow fo gradints to go backwards if J_F is smaler and smaller in each layers

— why "helps, not guarantees." The block's derivative is a sum: I + J_F. Sums can cancel. If the correction's derivative happens to be ≈ −I at some point, then I + (−I) ≈ 0 — the direct path and the correction path wipe each other out and (almost) nothing flows back. Rare, but possible, so no guarantees. Second hole: the original block puts a ReLU after the add. A ReLU's backward pass is an on/off switch — 1 if the unit fired, 0 if it didn't. A dead unit blocks the gradient completely, skip included. So the skip helps enormously in practice, but two things can still kill the signal: cancellation and dead ReLUs.

— the depth-race controls. Picture the claim: "my 20-layer net beats my 10-layer net, both trained on CIFAR." Same dataset alone proves nothing about depth, because the deeper net might have won for boring reasons: it trained for more steps, it saw augmented data the other didn't, it got a luckier seed, or someone picked the winner on the test set. So a fair depth race must fix all of that: same split, same update budget, same augmentation, same seed discipline, winner chosen on validation. Only then does the gap say something about depth.
"""

# %%

# %% [markdown]
# ## Step 1 — Data (same pipeline, written-down preprocessing)
# Same CIFAR loaders as all week. Record normalization and augmentation on the recipe card — the checkpoint is useless without them.

# %%
AUGMENTATION = True
## HYPER PARAMS
BATCH_SIZE = 128
NUM_WORKERS = 2
HIDDEN = 64
DROPOUT_P = 0.5

SEED = 42

### TRAIN
LR = 0.01
WEIGHT_D = 5e-4
MOMENTUM = 0.9

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


# %%

# %% [markdown]
# ## Step 2 — Model (BasicBlock + ResNet-20)
# Two 3×3 convs + skip per block; stages 16/32/64 with three blocks each; option A (or YOUR labeled projection) at stage changes; global average pooling; 10-class head.

# %%
class ResNetBlock(nn.Module):
  def __init__(self, in_channels , out_channels , stride = 1) -> None:
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride

    self.conv1 = nn.Conv2d(in_channels , out_channels , kernel_size=3 , stride=stride, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)

    self.conv2 = nn.Conv2d(out_channels , out_channels , kernel_size=3 , stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.relu = nn.ReLU()

  def forward(self, x):
    n1 = self.relu(self.bn1(self.conv1(x)))
    n2 = self.bn2(self.conv2(n1))

    if x.shape != n2.shape:
      small = x[:, :, ::self.stride, ::self.stride]
      shortcut = F.pad(small, (0, 0, 0, 0, 0, self.out_channels - self.in_channels))
      return self.relu(n2 + shortcut)

    else:
      return self.relu(n2 + x)


# %%
"""The count: 6n+2 with n = 3 → 1 first conv + 3 stages × 3 blocks × 2 convs = 19 conv layers + head = 20."""


# %%
class ResNet(nn.Module):
  def __init__(self , n ):
    super().__init__()
    self.n = n

    self.start = nn.Sequential(
        nn.Conv2d(3, 16 ,kernel_size=3 ,padding=1 ,stride=1),
        nn.BatchNorm2d(16),
        nn.ReLU())

    self.stages = []
    last_out = 16
    for i in range(self.n):
      for j in range(self.n):
        local_strde = 0
        if i == 0 :
          local_strde = 1
        elif j == 0 :
          local_strde = 2
        else:
          local_strde = 1

        self.stages += [ResNetBlock(last_out , local_strde * last_out , stride = local_strde)]
        last_out = local_strde * last_out


    self.nn = self.stages + [nn.AdaptiveAvgPool2d(1) , nn.Flatten() , nn.Linear(64 , 10)]
    self.seq = nn.Sequential(*self.nn)

  def forward(self , x):
    if x.dim() != 4 or x.shape[1] != 3:
      raise ValueError(f"expected (N,3,32,32), got {tuple(x.shape)}")
    return self.seq(self.start(x))

# %% [markdown]
# ## Step 3 — Shapes, params, branch match
# Layer/shape table + counted total near 0.27M + forward/backward + tiny-batch memorization. The tiny batch MUST memorize before Step 5.

# %%
model = ResNet(n = 3)
X = next(iter(train_loader))[0]

# %%

model.eval()
with torch.no_grad():
    x = torch.randn(2, 3, 32, 32)
    print("in   ", tuple(x.shape))
    x = model.start(x)
    print("stem ", tuple(x.shape))
    for i, block in enumerate(model.stages):
        x = block(x)
        print(f"blk {i} ", tuple(x.shape))
    # pool / flatten / head live at the end of model.seq:
    for m in list(model.seq)[len(model.stages):]:
        x = m(x)
        print("tail ", tuple(x.shape))

# %%
sum(p.numel() for p in model.parameters())

# %%

model.train()
xb, yb = next(iter(train_loader))
loss = nn.CrossEntropyLoss()(model(xb), yb)
loss.backward()
dead = [n for n, p in model.named_parameters() if p.grad is None]
print("loss:", float(loss.item()), "| params without grad:", dead)


# %%
torch.manual_seed(SEED)

xb, yb = next(iter(train_loader))
xb, yb = xb[:32], yb[:32]

model = ResNet(n=3)

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
# ## Step 4 — Recipe card + timing pilot
# Every setting written down (init, preprocessing, optimizer, schedule, deviations). Time steps after warmup, estimate wall time, set the max duration.

# %%

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# %%

import time

model = ResNet(n =3 ).to(device)
model.train()
model.to(device=device)
it = iter(train_loader)

# warmup: timing lies while kernels spin up, discard these
for _ in range(10):
    xb, yb = next(it)
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    opt.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    opt.step()

# pilot: time 50 real steps
t0 = time.perf_counter()
for _ in range(50):
    try:
        xb, yb = next(it)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
    except StopIteration:
        it = iter(train_loader)
        xb, yb = next(it)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
    opt.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    opt.step()
t1 = time.perf_counter()

per_step = (t1 - t0) / 50
steps_per_epoch = len(train_loader)
print(f"sec/step: {per_step:.3f} | steps/epoch: {steps_per_epoch}")
print(f"est per epoch: {per_step * steps_per_epoch:.1f}s")


# %%
"""
ON CPU
sec/step: 1.013 | steps/epoch: 352
est per epoch: 356.5s


ON GPU (cuda)
sec/step: 0.062 | steps/epoch: 352
est per epoch: 22.0s
"""

"""
Estimated time :
(22* 100 + 5 * 60 ) / 60 = 41.7 minutes

"""

# %%
"""
- Seed / split: 42, 45k/5k, same CIFAR loaders as all week
- Preprocessing: normalize with the   mean = (0.4914, 0.4822, 0.4465)std = (0.2470, 0.2435, 0.2616) + RandomCrop(32, padding=4)/HorizontalFlip
- Init: PyTorch defaults
- Model: ResNet-20 (n=3), option-A shortcuts, no dropout
- Optimizer: SGD, lr 0.01, momentum 0.9, weight_decay 1e-4
- Schedule: 80 epochs, MultiStepLR milestones [40, 60], gamma 0.1, batch 128
- Budget: pilot 22s/epoch → est ~30 min + overhead, max 2 GPU-hours
- Deviations from the paper: ~28k steps vs 64k; lr 0.01 vs 0.1; epochs 80 vs ~164. Same architecture, honest adaptation.
- Selection rule: best by validation accuracy, test set stays sealed
"""

# %% [markdown]
# ## Step 5 — Bounded run + reload check
# One run inside the 2 GPU-hour cap. Curves vs chance, best checkpoint with config, reload reproduces a fixed validation batch exactly.

# %%
torch.manual_seed(SEED)


model = ResNet(n=3).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr = LR, weight_decay=1e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 60], gamma=0.1)

# %%
import copy
import time
EPOCHS = 80


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
  correct = 0
  for j ,(xs ,ys) in enumerate(train_loader):
    xs = xs.to(device, non_blocking=True)
    ys = ys.to(device, non_blocking=True)

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
    correct += (logits.argmax(1) == ys).sum().item()
    n_seen += len(ys)
  current_epoch_loss = loss_sum / n_seen
  current_epoch_acc = correct / n_seen
  print(f"Loss of epoch no. {epoch_no} : {current_epoch_loss:.4f}")
  print(f"accuray of train: {current_epoch_acc:.4f}")
  return current_epoch_loss ,current_epoch_acc

def eval():
  was_training = model.training
  model.eval()
  correct, total = 0, 0
  all_true = []
  all_pred = []
  val_loss_sum = 0

  with torch.no_grad():
      for xb, yb in val_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        pred = logits.argmax(dim=1)
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
  final_train_loss, current_epoch_acc = train(i)
  acc, final_val_loss , conf_mtx = eval()
  print(f"accuray of val :{acc} \n")
  if acc > best:
    best = acc
    best_conf_mtx = conf_mtx
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = i
    best_train_loss = final_train_loss
    best_val_loss = final_val_loss
  scheduler.step()



end = time.perf_counter()

total_time = f"{end - start:.6f}"

# %%

current_time = time.ctime()
checkpoint = f"ResNet20_ep{best_epoch}_acc{best:.4f}_{current_time}.pth"


config = {
    "run": "aug_on" if AUGMENTATION else "aug_off",
    "seed": SEED,
    "split": "45k/5k",
    "augmentation": AUGMENTATION,
    "optimizer": {"name": opt.__class__.__name__, "lr": LR, "momentum": MOMENTUM, "weight_decay": 1e-4},
    "model": {
        "arch": "resnet20",
        "n": 3,
        "stages": "16/32/64",
        "blocks": "3 each",
        "shortcut": "option-A-zero-pad",
        "head": "avgpool + 10-class"
    },
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
# ## Step 6 — Results
# Validation metrics + status label (validated development checkpoint OR candidate teacher needing more) + honest note: still-learning or diagnosed, and what Month 7 needs besides the .pth.

# %%
"""
## Step 6 - Results

Best epoch = 62. Val acc = 0.8974. Train acc = 0.9604. Train loss = 0.1193.
Last epoch = 79. Val acc = 0.8958. Train acc = 0.9616. Train loss = 0.1152.

Val = check split. The 5k images I do not train on. Example: train on 45k, check on 5k.
Chance baseline = dumb guess. 10% acc, loss 2.30. Example: my 89.7% beats 10%, so it learned.

Status label = what the file is good for.
My label = validated development checkpoint. Not a teacher yet. Paper ref is 91.25%. I am at 89.7%.

Pick rule = I pick by val only. Not by train. Not by test. Test stays sealed.
"""

# %%
"""
Honest note:
My run is still-learning. Still-learning = loss falls and stays above chance, just slow.
My loss fell 1.55 to 0.11. Train 41% to 96%. Val 46% to 89.7%.

Flat part = after epoch 40, train goes 93% to 96%, val stays 89% to 89.7%.
That means more epochs help train more than val now.

What Month 7 needs besides the .pth:
.pth = weights only. Not enough alone.
Needs config, norm numbers (0.4914, 0.4822, 0.4465 / 0.2470, 0.2435, 0.2616), class order 0-9, 45k/5k split IDs, seed 42, SGD lr/momentum/wd + schedule, preprocessing + code revision.
"""


# %% [markdown]
# ## Step 7 — Your tests
# Checks you wrote yourself (branch shapes, shortcut option, count, reload, tiny fit). They must pass before handoff.

# %%
def smoke_test():
  model = ResNet(n = 3)
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
