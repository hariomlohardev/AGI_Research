# Month 3 · Week 2 Day 3 — Roadmap: CIFAR-10 pipeline + augmentation test

**Time:** 6–7 hours. **Compute:** laptop for checks, Colab/Kaggle GPU for the two 15-epoch runs. Set a wall-time cap before you start. If a run stops early, compare the same finished epochs and write down both times.

## 1. What to learn today

- Build a clean CIFAR-10 photo pipeline once, reuse it for AlexNet / VGG / ResNet days.
- Run one fair test: same model, same split, same seed, same optimizer — only augmentation on vs off.
- Keep test data sealed. Pick with validation only.

New words:
- CIFAR-10 = 60,000 small color photos. Size 32x32. 10 groups. 50,000 train, 10,000 test.
- split = cut train into 45,000 train + 5,000 check. Check = validation. Validation = small part to pick the model. Test = final part, open only at end.
- normalization = shift pictures so mean is 0. Formula: `(pixel - mean) / std`.
- augmentation = extra fake copies by crop + flip, only in train, to teach the model.
- denormalize = undo normalization for your eyes: `picture * std + mean`.
- logits = raw scores before softmax. Softmax turns them into weights that sum to 1.
- baseline = simple score to beat. Here: initial loss near ln(10) and a tiny-batch fit.

## 2. Why learn this

MNIST was gray digits. CIFAR is real photos with color, blur, and look-alike animals.
If the pipeline leaks here, every later vision result is fake.

## 3. Where it helps in the journey

This is the base for all vision work in Months 3, 4, and 7.
Same split rule, same train-only stats, same sealed test, same run table.
Month 4 attention needs the same clean-data habit. Month 7 needs your saved CIFAR checkpoint + exact preprocessing.

## 4. About the day's code

You work in `cifar_pipeline.ipynb` in Colab. Then you download it back and say handoff done.

You implement exactly two things:
1. `TinyCifarCNN` forward — small 3-channel CNN.
2. `build_transforms` — train vs validation transforms.

Harness ships complete: CIFAR loading, separate train/val objects, train/eval loops, memorize check, denormalize, comparison printer, smoke runner.

## 5. Mission — things we must GET FROM the code

- Correct 45k/5k indices from a fixed seed. Same indices for both runs.
- Train object uses random crop + flip. Validation object uses clean deterministic steps only.
- Proof the two objects do NOT share one mutable transform.
- Mean/std from train 45k only, or fixed numbers with source written down. Never fit on validation or test.
- One decoded train batch + one decoded validation batch, both labeled, denormalized for display.
- Tiny-batch fit with no augmentation: model must memorize a small fixed set.
- Two bounded runs: augmentation off vs on. Same split, same seed, same optimizer, max 15 epochs each.
- Run table: seed, changed variable, epochs/updates, train loss, validation loss, validation accuracy, time, checkpoint path.
- Validation confusion matrix with full class order. Save best validation checkpoint for each run.
- Test set stays sealed. No test numbers today.

## 6. Understand-before-code

- Why two dataset objects, not two Subset views of one object? Changing one shared transform changes both. That leaks augmentation into validation.
- Why train-only stats? Fitting on held-out data shows the model the answers early.
- Why denormalize to show? Model sees normalized numbers. Your eyes need normal colors.
- Why tiny-batch fit first? It checks the plumbing learns. It says nothing about new data.
- Why same seed and same budget? Else you compare luck, not augmentation.
- Why short runs can fool you? Augmentation makes training harder at first. Judge at the full 15-epoch budget, not epoch 2.
- Why small gaps mean little? 1% can be noise. Repeat seeds before you claim a win.

## 7. Extras

- Plot 30 train photos across all 10 classes. Write which pairs look alike. This is a guess, not a measured human score.
- Compare your errors with your first guess after the runs.
- Keep the official test set closed until Week 4.

## 8. Success matrix — what success means today

Pass = pipeline is clean + comparison is fair, not a winner:
- [ ] 45k/5k indices fixed and reused for both runs.
- [ ] Train has random crop + flip. Validation is deterministic. Separation is proven.
- [ ] Normalization stats are train-only or fixed with source.
- [ ] Tiny-batch fit passes. Initial logits/loss near ln(10) is checked.
- [ ] Two runs finished at the same budget with run table + confusion matrix + saved best checkpoints.
- [ ] Honest note: which won, by how much, and why a small gap is not proof.
- [ ] `run_smoke` passes on a fresh kernel with synthetic data only.

Fail = leaky transform, fitted stats on held-out data, test used for picking, invented numbers, or passing because the curve looks nice.

## 9. Restrictions — what you CAN and CANNOT use

CAN:
- torchvision CIFAR10, transforms, DataLoader, `num_workers=0` first.
- torch layers, loss, optimizer, autograd. nn for the small CNN.
- pandas run table, sklearn accuracy/confusion_matrix, matplotlib plots.
- Your Week 1 training loop pattern, copied with source note.

CANNOT:
- One shared dataset object with two Subset views where changing transform hits both.
- Random augmentation in validation or test.
- Mean/std fitted on validation or test.
- Test set for any picking decision.
- Claiming augmentation wins from one short run or one seed.
- Copying a full tutorial solution. Docs are open, final code is yours.
