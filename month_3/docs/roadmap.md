# Month 3 — PyTorch, CNNs & Sequence Models
## Detailed Roadmap V5 — Notebook-First Experiments for an AI Research Foundation

**Revised:** 23 October 2026<br>
**Starting point:** Month 2 is complete; this is the next month, not a restart.<br>
**Basis:** preserves the substantive corrections in month-3-detailed-revised-v4.md, which used your revised V3 rather than the old Month 3 plan. V5 changes the working format and library boundaries; Month 2 is already complete. The Month 4 research track and later-month dependencies remain the destination.<br>
**Schedule:** 28 day labels: 24 study days and 4 protected rest days. A day is a unit of work, not a deadline.<br>
**Purpose:** build the experimental and mathematical foundation for your long-term AI/AGI research-scientist goal: independently implement, verify, train, evaluate and reload small models; ask answerable questions; distinguish evidence from speculation; and carry these habits into attention research in Month 4.
**Working format:** all required new executable coursework lives in Jupyter/Colab .ipynb notebooks. Use established libraries for routine data work. Implement the mechanisms that each exercise explicitly asks you to understand. Notebook format still uses Python; it changes how you organize and inspect experiments, not the language.

---

## 1. What success means

By the end, you should be able to do these things with documentation available but without copying a complete tutorial solution:

1. Build a PyTorch training and validation loop from a blank notebook code cell; explain the input, target and output shapes.
2. Check a hand-written backward pass against a separate autograd reference and locate an intentionally introduced error.
3. Run a controlled model comparison with fixed data splits, a stated budget, a baseline and an honest interpretation.
4. Train an image classifier, inspect its failures and load its saved checkpoint after a notebook kernel/runtime restart.
5. Implement small RNN, LSTM and GRU cells; verify their equations and compare their behaviour on a precisely defined memory task.
6. Train a character language model with correct next-character targets and held-out loss; generate text without accidentally using training mode or future targets.
7. Carry a reusable data/training/checkpoint pattern into Months 4–5, and preserve a documented CIFAR-10 checkpoint for Month 7.

**This is a foundation in practical model development and small-scale experimentation.** It is not a claim that one month makes you ready for production deployment or frontier-scale research. A reproducible, well-explained small result is a successful outcome.

### What V5 improves over V4

- Required code artifacts are .ipynb notebooks, including training, evaluation, copy-task experiments and separate inference notebooks. No mandatory script conversion or packaging project.
- NumPy, pandas, torchvision, scikit-learn and plotting libraries handle standard operations; you do not rebuild data tooling to prove that you understand a neural network.
- A task-by-task boundary identifies what you implement, what libraries may do, and how you verify the result. Manual mechanisms remain bounded teaching exercises.
- Clean-kernel execution, explicit experiment state and saved artifacts replace dependence on cells run in a lucky order.
- Daily notebook destinations and a research evidence template make the existing workload easier to execute. These replace previous organization tasks; they are not another week of coursework.
- There is no promise of a perfect score or research readiness from a file format. Quality means correct mechanisms, controlled comparisons, reproducible evidence and honest interpretation.

### Corrections retained from V3 → V4

- Completion depends on correctness and evidence, rather than forcing an expected experimental winner.
- Train/validation/test separation and text-window leakage prevention are explicit.
- Tensor shapes and batched operations receive focused practice before Month 4.
- ResNet implementation, accuracy aspirations and compute limits are separate requirements.
- Residual/LSTM gradient explanations, BatchNorm edge cases and manual-gradient verification are corrected.
- Language-model target alignment, hidden-state handling, clipping and quantitative evaluation are specified.
- Historical architecture work is bounded; expensive optional experiments cannot silently consume the month.
- The final report contains two related case studies instead of an unsupported CNN-versus-RNN ranking across unrelated datasets.
- An inference-only notebook exercised after a kernel restart, plus an unfamiliar-input exercise, tests practical independence.
- Rest remains rest. Catch-up gets an extra study day instead of quietly replacing every recovery day.

### A brief prerequisite check — no repeat of Month 2

Spend no more than 30–45 minutes before Day 1 recalling the chain rule, matrix multiplication, gradient descent, cross-entropy, train/test separation and overfitting. Locate your Month 1 autograd implementation and Month 2 metrics/classifiers. Use them as references. If your scalar autograd engine cannot handle vectors, compare scalar operations it actually supports; extending that engine is not a prerequisite for this month.

A forgotten formula is a small review item. An inability to independently run or explain your earlier training code is a reason to repair that specific gap before adding new architecture complexity.

## 2. Workload, compute and working habits

You have the whole day available. Use that freedom for focused work, breaks and careful debugging; do not turn every available hour into another required topic.

| Week | Day | Core work | Focused hours |
|---|---|---|---|
| 1 | 1 | Tensors, shapes and autograd | 5–6 |
| 1 | 2 | Module, Dataset, DataLoader and training loops | 5–6 |
| 1 | 3 | GPU timing and checkpoint recovery | 4–5 |
| 1 | 4 | Dropout, BatchNorm and diagnostic histograms | 6–7 |
| 1 | 5 | Manual backward pass with a separate reference | 6–8 |
| 1 | 6 | Protected rest | 0 |
| 1 | 7 | Training diagnostics and controlled bug exercises | 5–6 |
| 1 | 8 | Independent MNIST project | 5–6 |
| 2 | 1 | Convolution mechanics and correctness | 5–6 |
| 2 | 2 | CNN baseline and tensor-shape workshop | 5–6 |
| 2 | 3 | CIFAR-10 pipeline and augmentation comparison | 6–7 |
| 2 | 4 | AlexNet ideas, small implementation | 4–5 |
| 2 | 5 | VGG blocks, receptive fields and a bounded comparison | 4–5 |
| 2 | 6 | ResNet-20 implementation and budgeted training | 6–8 |
| 2 | 7 | Protected rest | 0 |
| 3 | 1 | RNN cell and gradients through time | 5–6 |
| 3 | 2 | A precisely defined copy task | 5–6 |
| 3 | 3 | LSTM cell, gradient paths and original paper | 6–8 |
| 3 | 4 | Protected rest | 0 |
| 3 | 5 | GRU cell and controlled comparison | 4–5 |
| 3 | 6 | Correct character-language-model pipeline | 6–8 |
| 3 | 7 | Retrieval, independent rebuild and cleanup | 4–5 |
| 4 | 1 | Budgeted tuning on validation loss | 5–6 |
| 4 | 2 | Final evaluation, sampling and failure analysis | 4–5 |
| 4 | 3 | Separate inference notebooks, clean-kernel checks | 4–5 |
| 4 | 4 | Two-case-study research report | 4–5 |
| 4 | 5 | Reproduction, archive and Month 4 readiness | 4–5 |
| 4 | 6 | Protected rest | 0 |

**Total listed focused work: 117–145 hours.** These are planning estimates, not measured completion times. Allow another 10–25 hours if setup, debugging or integration takes longer: approximately 127–170 hours overall. GPU wall time is recorded separately and may overlap with reading or documentation. Do not count an unattended training hour as an hour of focused learning.

The section called Week 1 has eight day labels; these are learning blocks, not four literal calendar weeks. Start with the 28-day sequence and add study/buffer days when needed. A 30–35-day finish is acceptable if the difficult exercises need it.

**Suggested ordinary study day:** 15–20 minutes retrieval; 60–90 minutes selected reading and derivation; 2 hours implementation; 90 minutes checks/experiments; 30–45 minutes notes and explanation. Take breaks between blocks. On an 8-hour estimated task, splitting the work over two study days is allowed.

**Compute:** use your laptop for small checks; use an available Colab or Kaggle GPU for larger training runs. Check your account's current limits and hardware. Free GPU availability, model and runtime duration are not guaranteed. Colab's published free-runtime maximum is up to 12 hours subject to availability and usage, not a promise of a 12-hour T4 session. Keep a backup environment and durable checkpoints. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

### Core, optional and catch-up rules

- Complete each day's core build, checks, questions and evidence. Optional work starts only when those are complete and time remains.
- One main learning resource per topic is usually enough. Read/watch only the relevant section, then close it and implement. Do not watch several complete courses for the same day.
- Keep the manual-backprop and LSTM days challenging; simplify surrounding experiments instead of removing their central learning goal.
- On study days, use a 30–45 minute troubleshooting block before changing approach. Make a minimal example, consult the primary reference and record what you learned. Do not keep rerunning unchanged settings.
- If a core correctness check still fails, mark it unresolved and schedule a repair block. A budget cap is permission to stop work today, not to declare broken code correct.
- Keep all four rest days free of coursework. Insert a catch-up day before/after a rest day if necessary. Optional historical ablations are the first things to drop.
- Required seeds: one recorded seed for most learning runs. Repeat a headline comparison with a second seed if its conclusion matters and budget permits; otherwise label it a single-run observation. Do not make significance claims from one run.

### 2.1 Notebook-first workflow

Use JupyterLab locally or a Colab notebook with durable storage. Keep the required code in .ipynb files. Markdown notes, JSON configs, CSV result tables, image files and model checkpoints retain their natural formats; putting large datasets or weights inside a notebook is unnecessary. This roadmap specifies notebooks you will build during study; it does not supply completed solutions.

**Set up once, then reuse:** in the setup notebook record the Python executable/version, relevant package versions, device and project root. Use the notebook's active kernel for installation; an explicit %pip cell is acceptable when packages are missing. Follow the official installation guidance for a compatible torch/torchvision pair and your hardware. Do not blindly upgrade a working Colab environment. Save the working versions and restart the kernel if installation requires it. This fits inside Week 1 setup time.

**Default tools:** Python, NumPy, pandas, matplotlib, PyTorch, torchvision, scikit-learn and Jupyter/Colab. Pillow is suitable for image loading; seaborn, tqdm and TensorBoard are optional conveniences. Add a package only for a concrete need. Learning several logging tools or installing every possible data library is not a goal.

Each experiment section uses this order:

1. **Question and prediction — Markdown:** one question, a predicted direction or explanation, and what observation could challenge it. State the baseline, changed variable, controls and budget before running.
2. **Setup and configuration — code:** imports, explicit project/data/output paths, device, seed, dataset/split IDs, model and optimizer settings, execution mode and unique run ID. Keep expensive execution separate from definitions.
3. **Inspect the data — code and plots:** actual transformed examples, shapes/dtypes, ranges, class/vocabulary mapping and split boundaries. Use a small table for metadata and summaries.
4. **Mechanism — Markdown plus code:** equations with axes, your model/functions, and the tiny correctness checks required for this day. Explain any library operation whose behavior affects your hypothesis.
5. **Run — code:** create a fresh model, optimizer, scheduler, loaders and metric history for a new independent run. Restore all relevant state only for an explicitly labeled continuation.
6. **Analyze — code and Markdown:** tidy run table, labeled figures, comparison against baseline, uncertainties and at least one alternative explanation. Save metrics/config/checkpoint references to disk.
7. **Conclude — Markdown:** answer the day's questions, distinguish a bug from a model limitation, and choose one justified next step. A contrary or inconclusive result can be scientifically useful.

### 2.2 What you build and what libraries do

The decision rule is: **implement the mechanism under study; use tested tools for surrounding work; verify the interface between them.** Using libraries well is part of research skill. Repeating Month 2 utilities in basic Python does not automatically deepen understanding.

| Task | Build or reason through yourself | Use libraries for | Minimum check |
|---|---|---|---|
| Tabular loading and exploration | Choose columns, identify leakage/missingness, justify the split | pandas readers, selection, summaries and grouping | Check schema, row counts, label alignment and duplicates relevant to the split |
| Tabular preprocessing | Decide what may be fitted and on which data | scikit-learn Pipeline/ColumnTransformer, imputation, scaling, encoding | Fit on training only; transform held-out data using the fitted objects |
| Image data | Choose split IDs and train/evaluation policies | torchvision datasets/transforms, Dataset wrappers and DataLoader | Inspect transformed batches; train and validation must not share a mutable augmentation setting |
| Dataset mechanics | Write one tiny custom Dataset in Week 1 Day 2 to understand __len__/__getitem__ | TensorDataset for ordinary in-memory tensor pairs; built-in datasets afterward | Shapes, target alignment and one manually inspected batch |
| Training and evaluation | Write train_epoch/evaluate, mode changes, loss weighting, checkpoint selection and experiment controls | nn.Module, standard losses, optimizers, autograd, DataLoader | Weight-update, finite-value, tiny-batch-fit and evaluation checks |
| Ordinary MLPs/CNNs/ResNet | Compose the architecture and explain axes, residual/shortcut wiring and parameter counts | nn.Linear, nn.Conv2d, pooling, activations, BatchNorm, Dropout | Shape checks, intended registered parameters and the day's diagnostics |
| Manual backprop day | Derive and code the specified forward/backward arithmetic | Tensor operations; autograd only in a separate reference branch | Compare intermediate and parameter gradients for the same function |
| Convolution mechanics day | Hand example and one small NumPy loop implementation | Image I/O/display; PyTorch convolution only as the reference | Numerical parity on hand, padding and stride cases; use Conv2d for subsequent training |
| Recurrent-cell days | Implement RNN/LSTM/GRU equations, state transitions and gate layout | Tensor matmul/nonlinearities, autograd and optimizers; library cells as references | Match parameters/conventions and compare required outputs/gradients |
| Copy task and text windows | Define task semantics, recall mask, split boundaries, shifted targets and hidden-state policy | Vectorized NumPy/PyTorch indexing, random generation and DataLoader | Decode tiny examples; confirm masking and boundary behavior |
| Practical character model | Assemble Embedding → LSTM → Linear and control its training/evaluation | nn.LSTM for longer runs after custom-cell verification | Same data contract and held-out metrics; library recurrence does not remove target-alignment checks |
| Metrics and reporting | Choose metrics, denominator, aggregation, comparison and interpretation | scikit-learn classification metrics, pandas result tables, matplotlib/seaborn plots | Hand-check a few predictions once; ensure class order and token/example weighting are correct |

A high-level Trainer, AutoML system or downloaded whole-model implementation must not replace the required training-loop, custom-cell or architecture exercise. This is a scope choice for learning those mechanisms, not a general ban on these tools in later research. Standard layers and optimizers are encouraged here. For ResNet, compose the requested CIFAR ResNet-20; a stock ImageNet ResNet-18 is a different architecture.

For table operations and plots, consult the relevant [pandas introductory examples](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html). For standard vision data use [torchvision datasets](https://docs.pytorch.org/vision/stable/datasets.html). For train-fitted transformations and leakage prevention use [scikit-learn's common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html). These are references for the existing exercises, not extra courses.

### 2.3 Notebook correctness is part of experiment correctness

- **Top-to-bottom execution:** imports, configuration, function/class definitions, data preparation, checks, execution and analysis must have a clear dependency order. Do not rely on a variable left over from a deleted cell or another notebook. Avoid %run chains that execute a training notebook as a side effect of inference.
- **Clean-kernel evidence:** before marking a notebook milestone complete, restart the kernel/runtime and Run All in its documented lightweight mode. A kernel restart gives a fresh interpreter; an environment recreation checks installation separately. Clearing displayed outputs alone does not reset state.
- **Execution modes:** default to a small smoke run for training notebooks. A configuration cell explicitly selects smoke, train, resume or saved-results analysis where applicable. Smoke mode uses a small fixed subset and a few updates; train uses the declared full budget; resume loads the recorded state; analysis reads saved artifacts. Label mode and dataset size in every output. Running analysis is not evidence of rerunning training, and smoke metrics are not final metrics.
- **No accidental long runs:** definitions and inspection cells must not start training. Full training requires an explicit mode/config change. A missing required checkpoint in inference or analysis must produce a clear error, not silently retrain a model or substitute random weights. Do not use stale outputs as evidence that the current code passed.
- **Fresh state for comparisons:** set the recorded seed at the start of each independent run, then create all run state anew. Do not reseed every batch or optimizer step. Do not carry optimizer momentum, accumulated gradients, recurrent hidden state, old histories or BatchNorm statistics from one independent configuration into another. For a paired comparison, specify what is matched and retain a consistent initialization policy.
- **Changed class definitions:** rerunning a class cell does not update an existing instance. Recreate the model and optimizer after a definition change, then rerun the relevant checks. Prefer state_dict plus architecture/config information over serializing a whole notebook-defined model object. Preserve the exact notebook revision used for a saved result.
- **Data interfaces:** use pandas for tabular data, metadata and result tables, not as a container for all image tensors. Make array-to-tensor dtype conversions explicit: commonly float32 features and int64 class/index targets, except float64 gradient-check exercises. Keep differentiable model operations in PyTorch. For offline plots/metrics, detach tensors and move them to CPU before converting to NumPy; do not accidentally sever the training graph.
- **Windows-friendly loading:** start with num_workers=0. Use library batching instead of hand-written sample batching. Multiprocessing/performance tuning is optional once correctness is established; it must not force a script-based rewrite this month.
- **Durable, identifiable evidence:** save config, split IDs, metrics, selected checkpoint and run ID outside the notebook, and link them from its conclusion. Use pathlib and an explicit root rather than assuming a particular current directory. Notebook outputs are for inspection; saved artifacts allow recovery when a runtime disappears.

**Optional automation:** after manual restart-and-run checks work, nbclient can execute a notebook programmatically. It is not a required extra framework or a replacement for checking results. [nbclient documentation](https://nbclient.readthedocs.io/en/latest/)

**Four recurring research questions:** What mechanism am I learning? Which library behavior must I verify? What exactly does this comparison isolate? Could someone reconstruct the claim from my saved evidence? Answer briefly in the notebook conclusion rather than creating another report each day.

## 3. Rules for trustworthy experiments

### 3.1 Data and evaluation

1. Split data before fitting preprocessing or selecting settings. Fit scalers, imputers and learned preprocessing only on training data. Reuse the fitted transformation on validation/test data.
2. For MNIST and CIFAR-10, reserve a reproducible validation subset from the official training set. Suggested sizes: 55k/5k for MNIST and 45k/5k for CIFAR-10. Keep official test sets sealed until final evaluation in Week 4.
3. Use separate dataset wrappers/transforms for train and validation indices. Random image augmentation belongs only in training. Validation/test use deterministic preprocessing.
4. Select architecture, learning rate, epoch/checkpoint and other settings using validation results. Test data is for final reporting after decisions are frozen. [Evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html)
5. For text, partition raw text into fixed, disjoint contiguous regions, then construct windows within each region. No input or target window crosses a split boundary. Freeze the vocabulary from the training region; define an unknown-character policy and report any held-out unknowns.
6. Predefine a comparison's question and what is held equal: data, optimizer settings where appropriate, updates/examples processed, or elapsed training time. Equal epochs across different models are not equal FLOPs. Equal wall time answers a practical budget question, not a pure architecture question.
7. Record contrary or inconclusive results. A working RNN at a long gap is evidence, not a reason to sabotage it until it fails.

### 3.2 Five debugging checks, with their limits

These checks are inspired by [Karpathy's training recipe](https://karpathy.github.io/2019/04/25/recipe/). Run them when introducing a new pipeline or model family; do not repeat expensive versions after every cosmetic edit.

| Check | How to use it | What it does not prove |
|---|---|---|
| Inspect exact model inputs | Decode/plot the actual transformed batch, inspect targets, dtype, range and shape | Attractive raw samples do not validate preprocessing |
| Establish an initial-loss baseline | For approximately uniform K-class probabilities, mean CE is approximately ln(K); inspect logits if it differs | It is not a universal requirement for regression or every initialization; balanced labels are not needed for the uniform-prediction calculation |
| Memorize a fixed tiny batch | Use 5–10 examples, enough capacity, no random augmentation, zero dropout/weight decay for this diagnostic; choose consistent BatchNorm behaviour | Some tasks have irreducible loss or conflicting targets; low training loss does not prove generalization or correct labels |
| Test input usefulness | Compare against a trained input-independent/constant baseline using held-out metrics, or document a trained-model input ablation as a different diagnostic | A random untrained model need not prefer real inputs; zeroing inputs at inference is not the same experiment as training a baseline |
| Check sample independence | For a sample-separable model and loss from sample i only, gradients w.r.t. other input samples should be zero | BatchNorm in training mode and intentionally cross-sample objectives are exceptions; shared parameter gradients are not “leakage” |

Additional routine checks: finite loss/gradients; optimizer receives intended parameters; gradients are cleared between updates; labels are in range; loss reduction is correct; inference uses evaluation mode and disables gradient recording. Evaluation mode and disabling gradients perform different jobs.

For classification use raw logits with CrossEntropyLoss and integer class targets in the ordinary single-label case. Do not apply softmax first. For regression choose a suitable regression loss and baseline. [CrossEntropyLoss reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

### 3.3 Evidence and checkpoint contract

Use the notebook structure in Section 6, with vision and sequence experiments clearly separated. Write small named functions in code cells and keep one authoritative definition per notebook. Reuse a checked, version-labeled function cell when needed; do not create a framework or require imports between notebooks. Record the source revision when copying a small definition into a separate inference notebook, and verify fixed-input parity after any change. Each experiment records:

- Run ID, question, prediction, changed variable and fixed controls.
- Dataset/version, split indices or boundaries, preprocessing, seed and model configuration.
- Optimizer, learning-rate schedule, batch size, updates/epochs, stopping rule and compute budget.
- Train/validation metrics, duration, parameter count, checkpoint location and hardware/software versions.
- Observed outcome, one alternative explanation, limitations and next action.

A resumable checkpoint stores model and optimizer state, epoch/global step and scheduler state if used. For a controlled continuation comparison also restore relevant Python/NumPy/PyTorch CPU/CUDA RNG state, data-order/generator state and sampler position as needed. Save scaler state if mixed precision is introduced later. Prefer epoch-boundary resume initially; mid-epoch replay is harder. Save preprocessing, class/vocabulary mappings and architecture config alongside the weights. [Checkpoint reference](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

A smooth-looking loss curve alone does not prove exact resume. Compare a fixed next batch/update in a controlled environment. Exact bitwise reproducibility across devices, library versions and nondeterministic kernels is not promised. [Reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)

---

## WEEK 1 — PyTorch foundations you can reuse

**Week goal:** a verified training pipeline, shape literacy, correct gradients, diagnostic habits and one independently assembled MNIST classifier.

### Day 1 — Tensors, shape reasoning and autograd

**Time:** 5–6 focused hours. **Goal:** explain what every axis means and verify gradients instead of treating backward as a black box.

**Recall first:** derive the chain rule for a three-function composition; explain how a shared node receives contributions from two downstream branches.

**Primary resource:** [PyTorch tensor tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html), followed by the autograd material linked in the resource list. Read the sections you need, not the whole documentation site.

**Derive and predict:** for scalar z = xy + x², derive both gradients. Then reuse x in two branches and predict their summed contribution. Predict shapes for (B,D) @ (D,H), adding a bias of shape (H,), and elementwise multiplication by a mask of shape (B,1).

**Build and verify:**
1. Create floating-point tensors and integer index tensors; inspect shape, dtype and device. Practice slicing, broadcasting and reductions with explicit dimensions.
2. Compare the scalar expression using paper, your earlier autograd engine and PyTorch. For vector expressions compare paper/PyTorch if your old engine is scalar-only.
3. Demonstrate leaf versus non-leaf tensors, parameter-gradient accumulation across backward calls and clearing gradients. Use a fresh forward graph for each ordinary training step.
4. Show a graph reused without retaining it raises an error in a case where saved intermediates are needed. Explain why retain_graph is not the normal training-loop fix.
5. Create a small shape notebook containing predicted shapes, actual shapes and one deliberately wrong broadcasting example.

**Questions:**
1. Why do repeated uses of a tensor add gradient contributions?
2. Why can an intermediate require gradients while its .grad is not retained by default?
3. Why can broadcasting produce a mathematically valid result with the wrong intended shape?
4. What is the difference between clearing accumulated gradients and retaining a computation graph?

**Done when / evidence:** scalar gradients match; the shared-node case passes; you can explain and correct the broadcasting bug; save the gradient comparison and shape table.

**Optional:** compute a second derivative with create_graph=True and explain what extra graph must exist. Do not let this displace core shape practice.

### Day 2 — Module, Dataset, DataLoader and a complete training loop

**Time:** 5–6 hours. **Goal:** build the reusable skeleton used by the remaining months.

**Recall first:** explain gradient accumulation and identify the batch axis in yesterday's examples.

**Read:** the Module, Dataset/DataLoader and optimization sections of [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).

**Derive:** for independent per-example gradient estimates with covariance Σ, the mean of B estimates has covariance Σ/B. Discuss why correlation, sampling without replacement and changed optimization dynamics complicate “double the batch, halve the variance.” This variance identity does not require a CLT approximation.

**Build and verify:**
1. Generate a fixed small regression dataset using NumPy/PyTorch array operations, make a reproducible split and wrap it in one tiny custom Dataset to understand indexing. Then show the equivalent TensorDataset for ordinary tensor pairs; use the simpler library wrapper in later exercises when it fits. Start DataLoader with num_workers=0, especially on Windows/notebooks. Do not hand-code a batching framework.
2. Implement a two-layer nn.Module and train it with MSE. Write separate train_epoch and evaluate functions, with model.train(), model.eval() and no-grad evaluation in the correct places.
3. Inspect registered parameters and verify one optimizer step changes intended weights. Report an example-weighted epoch loss rather than an unweighted mean of unequal batch means.
4. Compare against a constant predictor fitted on training targets. The required regression baseline is not ln(K).
5. Run the sample-independence check on the MLP without BatchNorm: loss from one sample, gradients w.r.t. all batch inputs.
6. Make one bounded ordered-versus-shuffled comparison, using a target-sorted training set, the same initialization and update budget. Record either outcome; shuffling need not produce a visible advantage here.

**Questions:**
1. What does model(x) do beyond a direct forward(x) call?
2. Why do validation batches normally avoid random augmentation and gradient recording?
3. Why are parameters shared across examples compatible with zero cross-sample input gradients?
4. Why is a larger batch not automatically a better use of a fixed compute budget?

**Done when / evidence:** the complete loop runs, weights update, held-out MSE and baseline are logged, sample independence passes and your shuffling observation is documented. Save the loop as named function cells for reuse, with a short explanation of its inputs, outputs and loss denominator. Restart the kernel and run the notebook in smoke mode before closing it.

### Day 3 — GPU setup, honest timing and checkpoint recovery

**Time:** 4–5 hours. **Goal:** know when a GPU helps and recover work after a runtime reset.

**Recall first:** explain the regression baseline and the difference between evaluation mode and disabling gradients.

**Read:** [CUDA timing/semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) and the checkpoint reference in Section 3.3. Check current Colab/Kaggle limits in your own account.

**Build and verify:**
1. Confirm GPU access; record device name, memory and software versions. Keep a CPU fallback and a backup notebook environment.
2. Benchmark CPU and GPU matrix multiplication at three manageable sizes, using the same dtype. Warm up; repeat measurements; use CUDA events or synchronization around GPU wall-clock timing. Report median timings.
3. Measure compute-only timing separately from a transfer-inclusive timing. Confirm numerical agreement within a stated tolerance. Do not require a GPU win at a particular size.
4. Save your small model using the checkpoint contract. Copy the checkpoint to durable storage; restart the runtime; rebuild from configuration and reload.
5. On a fixed batch, compare predictions before/after reload in evaluation mode. For resume, compare a controlled next update with and without interruption, restoring relevant state. Start at an epoch boundary.

**Questions:**
1. Why is unsynchronized GPU wall-clock timing misleading?
2. When can transfer and launch overhead dominate the arithmetic?
3. What optimizer information would be lost by saving weights alone?
4. Why does a similar loss after restart not prove the exact same training trajectory?

**Done when / evidence:** a correctly measured timing table and a successful durable save/reload/controlled-resume check. Explain any remaining numerical differences rather than claiming universal bitwise identity.

### Day 4 — Dropout, BatchNorm and useful histograms

**Time:** 6–7 hours. **Goal:** distinguish regularization, normalization and initialization effects.

**Recall first:** explain held-out error and why a lower training loss need not mean a better model.

**Read:** the Dropout/BatchNorm references and the relevant activation/gradient section of Karpathy's neural-network course. Use one small tabular classification dataset, preferably the Month 2 dataset, with train-fitted preprocessing and a fixed validation split. Use pandas for inspection and scikit-learn preprocessing/Pipeline for imputation, scaling or encoding as needed; reuse your Month 2 choices rather than rebuilding their algorithms. Convert the resulting features/labels to the intended PyTorch dtypes.

**Derive:** inverted dropout keeps E[mx/(1-p)] = x when m is Bernoulli(1-p); this preserves the activation's expectation, not its variance or the expectation of an entire nonlinear network. Explain training batch statistics versus BatchNorm's running statistics at evaluation.

**Build and verify:**
1. Create four small MLP variants: baseline, dropout only, BatchNorm only and both. Keep their width, data and update budget fixed. These four short runs are the budget; do not launch a large search.
2. Record training objective and deterministic train-subset/validation metrics. Dropout-active training loss and dropout-disabled validation loss are not directly identical conditions.
3. Inspect a four-layer tanh MLP's preactivations, activations and gradient norms/histograms. Start with reasonable initialization, then deliberately increase weight scale to create saturation. Compare to a normalization variant using the same input batch.
4. Try BatchNorm1d in training mode on input shaped (1,C), record the error; then use evaluation mode with established running statistics. Explain why (1,C,L) with L>1 can have enough values per channel and is a different case. Batch size 1 does not universally break all BatchNorm uses.

**Questions:**
1. Which comparison isolates dropout's effect, and which isolates BatchNorm's?
2. Why might regularization fail to improve validation performance on this dataset?
3. What do activation histograms reveal that a final accuracy number does not?
4. Why does model.eval() matter even inside a no-grad context?

**Done when / evidence:** the four-run table, labeled histograms, a train/eval explanation and an honest conclusion. A smaller generalization gap or visible improvement is not mandatory.

**Optional:** inspect a high-dropout setting on one short run. No extra full hyperparameter search.

### Day 5 — Manual backward pass, with independent verification

**Time:** 6–8 hours; split across study days if needed. **Goal:** derive and implement gradients through a small MLP with BatchNorm.

**Recall first:** derive a linear-layer gradient and the tanh derivative in terms of its input and output.

**Resource:** Karpathy's “Becoming a Backprop Ninja,” linked through his course. Watch targeted sections as needed; pause to derive each step yourself.

**Exact scope:** a tiny floating-point input X, Linear → training-mode BatchNorm → tanh → Linear → softmax cross-entropy. An embedding layer is optional and not a hidden prerequisite. Use double precision on CPU for these small numerical checks. Implement BatchNorm's forward formula explicitly with a stated population variance (divide by batch size) and epsilon, so both branches compute the same function.

**Derive and build:**
1. Write shapes and derive gradients for mean cross-entropy/logits, second linear layer, tanh, BatchNorm's scale/shift and normalization, then first linear layer and X.
2. Separate the manual and reference calculations. The manual backward function uses tensor arithmetic only: no backward, autograd.grad or functional autodiff inside it.
3. The reference branch is allowed to use autograd. Use autograd.grad for selected intermediates or retain_grad before backward; ordinary non-leaf .grad is not populated automatically.
4. Compare every corresponding intermediate and parameter gradient. Record maximum absolute error and an allclose-style check, for example atol=1e-8 and rtol=1e-6 for this small float64 exercise. Investigate scale-dependent differences; do not silently loosen tolerances to hide a bug.
5. Deliberately omit a mean/variance dependency in the manual BatchNorm backward calculation, identify the first wrong intermediate and repair it.

**Questions:**
1. Where does division by batch size enter the mean-loss derivative?
2. Why does broadcasting a bias in the forward pass require a sum in its backward pass?
3. Why can changing one example affect another example through training-mode BatchNorm?
4. Why must the manual and autograd branches use the same variance definition and epsilon?

**Done when / evidence:** the complete gradient table passes and the injected error is located and corrected. If incomplete, log the failing operation and schedule an extra study block; do not mark it passed because six hours elapsed.

### Day 6 — Protected rest

**Time:** no coursework. No recall quiz, paper, experiment or cleanup obligation. If Day 5 is unfinished, put its remaining work on an additional study day. Rest does not depend on obtaining a perfect result.

**Done when:** you have taken a real break from the roadmap.

### Day 7 — Diagnose a real training run

**Time:** 5–6 hours. **Goal:** use evidence to distinguish implementation bugs from ordinary optimization/generalization behaviour.

**Recall first:** explain the BatchNorm backward dependency and interpret two of your histogram plots without reading their captions.

**Build and verify:**
1. Reuse the tabular pipeline, a modest MLP and the relevant checks from Section 3.2. Do not add every regularizer automatically; choose and record a baseline first.
2. Log train/validation loss and accuracy, learning rate and a gradient norm to TensorBoard or a simple equivalent plot/log. Record model mode for measured metrics.
3. Complete one ordinary run and describe its learning curve without claiming every fluctuation indicates a bug.
4. In a separate short diagnostic branch, omit the optimizer update; verify parameter snapshots remain unchanged. Repair it.
5. In another short branch, introduce a label/input alignment error and inspect indexed sample-label pairs. Explain why a sufficiently expressive model can memorize a fixed tiny batch with shuffled labels: tiny-batch overfitting alone cannot catch every data bug.

**Questions:**
1. Which signal distinguishes no learning from a logging mistake?
2. How would you investigate rising validation loss alongside falling train loss?
3. Why is freezing one layer not guaranteed to make training fail?
4. What additional check caught the label error that the tiny-batch check could miss?

**Done when / evidence:** one readable training log and two short bug reports: symptom, evidence, cause, fix and verification. A plain text report is enough.

### Day 8 — Independent MNIST classifier

**Time:** 5–6 hours. **Goal:** assemble a complete small project from your utilities without following a full tutorial line by line.

**Recall first:** outline the training loop and its evaluation branch from memory.

**Build and verify:**
1. Inspect 20–30 examples, labels and class counts. Record potential confusions as hypotheses, not a measured human error rate.
2. Create the 55k/5k training/validation split and keep the official test set unopened. Persist split indices and deterministic validation transforms.
3. Build an MLP baseline. Use your debugging checks, train under a preset budget and select the checkpoint by validation performance. Do not add BatchNorm/dropout unless you can explain the choice.
4. Produce a validation confusion matrix and inspect ten mistakes. Explain at least two error patterns.
5. Save model configuration, preprocessing and class mapping. Restart the kernel, reconstruct the model using saved configuration and load its state_dict without retraining. Compare evaluation predictions on the same fixed validation examples. Use torchvision MNIST and DataLoader for loading; your model, training loop and explanation are the learning work.

**Questions:**
1. What did data inspection change about your expectations?
2. Why is guessing a human error rate from 30 images not a valid benchmark?
3. What information besides weights is required to classify a new image correctly?
4. Which debugging step did you run independently, and which still required notes?

**Done when / evidence:** a reloadable MNIST model, validation metrics, confusion matrix, error notes and a short README. Test evaluation is reserved for Week 4. Record achieved accuracy; do not fabricate or chase a number just to pass the day.

---
## WEEK 2 — Convolutions, fair comparisons and a reusable ResNet

**Week goal:** understand convolution numerically, train small vision models with a consistent evaluation protocol, and preserve a verified CIFAR-10 model. Architecture history supports the learning; it does not require several expensive replications.

### Day 1 — Convolution mechanics and numerical correctness

**Time:** 5–6 hours. **Goal:** connect sliding-window arithmetic, output shapes and parameter sharing.

**Recall first:** explain the difference between a training target, a validation decision and a final test result. Predict the shape of a batch of RGB images.

**Read/watch:** one explanation of convolution, then the [PyTorch Conv2d definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) for the precise operation and output-size convention.

**Derive:** for one spatial dimension, output = floor((input + 2p − d(k−1) − 1)/s) + 1. Start with dilation d=1. Work a 4×4 input and 2×2 kernel by hand. CNN libraries usually implement cross-correlation, without flipping the kernel; mathematical convolution flips it. Name the convention used in your implementation.

**Build and verify:**
1. Implement a single-channel, two-dimensional cross-correlation in NumPy with explicit loops, configurable stride and symmetric zero padding. Dilation support is optional; understand its role in the formula.
2. Check the hand-worked example, one padded example and one stride-2 example against a separate PyTorch reference with identical weights and no bias.
3. Apply an edge kernel and blur kernel to a grayscale image; inspect ranges and display scaling. Do not rely only on “it looks right” for correctness.
4. Explain how input/output channels extend the operation. Count parameters as Cout × (Cin × kh × kw + 1) when each output channel has a bias.

**Questions:**
1. Why can two correctly written implementations disagree if only one flips the kernel?
2. How do you derive the output formula from the last legal window position?
3. What boundary information changes when zero padding is introduced?
4. Which part of parameter sharing makes convolution cheaper than a fully connected map on an image?

**Done when / evidence:** three numerical comparisons pass within a recorded tolerance, output dimensions match and your before/after images are labeled. Save the small implementation and its checks in the convolution notebook. This deliberate loop-based exercise ends here; use nn.Conv2d in the remaining CNN training work.

**Optional:** design a diagonal-edge kernel and test it on synthetic diagonal and non-diagonal patterns. A failed design with a diagnosis is useful; there is no required visual winner.

### Day 2 — A CNN baseline and the Month 4 tensor workshop

**Time:** 5–6 hours. **Goal:** implement a CNN and become comfortable changing tensor layouts without changing their meaning.

**Recall first:** derive one convolution output shape and count its weights without opening code.

**Build and verify — CNN:**
1. Build a small Conv2d → activation → pooling stack with a classification head. Use the exact MNIST split and preprocessing contract from Week 1.
2. Print/check the shape after every block using one batch. Run a small memorization check, then a bounded training run.
3. Compare with the MLP using validation accuracy, parameter count and measured training time. State the budget used. Neither a CNN accuracy win nor fewer parameters is required by definition; it depends on the architectures chosen.
4. Explain pooling as local aggregation that can reduce sensitivity to some translations. It does not guarantee exact translation invariance; stride, boundaries and the classifier head matter.

**Build and verify — tensor workshop:**
1. Start with an ordered tensor shaped (B,T,D). Split D=H×Dh, rearrange to (B,H,T,Dh), then invert the transformation and confirm exact recovery of the original values.
2. Explain why a reshape alone does not exchange the time and head axes. Practice reshape, transpose/permute and making a contiguous tensor when a view operation requires it.
3. For Q shaped (B,H,T,Dh) and K shaped (B,H,S,Dh), predict and verify Q @ K.transpose(-2,-1) has shape (B,H,T,S). No attention model is required yet.
4. Apply softmax over the last axis and check that each row sums to one. Introduce a wrong-axis version and explain why the resulting shape can still look correct.

**Questions:**
1. Why is a correct output shape necessary but insufficient for correct data layout?
2. Which axis represents candidate keys in the last tensor exercise?
3. Give an example where pooling discards information needed by a classifier.
4. What can you conclude if the CNN has fewer parameters but worse validation accuracy under today's budget?

**Done when / evidence:** the CNN run is documented, the split/merge round-trip succeeds and the wrong-axis bug is explained. Save the shape worksheet for Month 4.

**Optional:** replace max pooling with average pooling for one matched-budget comparison. Predict first, then accept either outcome.

### Day 3 — CIFAR-10 data pipeline and an augmentation experiment

**Time:** 6–7 hours. **Goal:** run a controlled experiment on a harder dataset and make it reproducible.

**Recall first:** state the augmentation/evaluation split rule and the difference between equal epochs and equal compute.

**Inspect and prepare:** plot 30 training images across all ten classes, record ambiguous pairs and establish 45k/5k train/validation indices. Use only training data to compute any dataset normalization statistics, or use documented fixed statistics and state their origin. Do not fit them on held-out data.

**Build and verify:**
1. Adapt your small CNN to three-channel inputs and ten logits. Inspect the exact normalized batch and a correctly denormalized display.
2. Use random crop with padding and horizontal flip for the augmentation condition; deterministic conversion/normalization for validation. Use torchvision CIFAR10 with standard torchvision transforms and distinct dataset instances/wrappers for training and validation. Two Subset objects around the same mutable dataset do not isolate its transform: changing that shared transform would affect both. Reuse split indices while keeping augmentation separate.
3. Run relevant diagnostics, including initial logits/loss and a no-augmentation tiny-batch fit.
4. Compare augmentation off/on using the same split, initialization seed, optimizer and maximum 15 epochs each. Set a shared wall-time safety cap before running; if one run is interrupted, compare a common completed epoch/update budget and report both durations.
5. Record train/validation loss, validation accuracy and a confusion matrix. Use a pandas run table, scikit-learn accuracy/confusion-matrix functions and matplotlib/seaborn for the display; specify the full class order. Compare observed errors with your initial hypotheses. Save the best validation checkpoint for each condition.

**Questions:**
1. Which flips preserve a plausible CIFAR label, and why would the same policy be questionable for some digit images?
2. Why must the validation subset not inherit the training augmentation transform?
3. Why might augmentation help final generalization but hurt performance in a very short run?
4. What would you need to repeat before claiming a small accuracy difference is reliable?

**Done when / evidence:** two controlled run records, a saved checkpoint, actual metrics and an interpretation that acknowledges the budget. Neither augmentation improvement nor a smaller loss gap is a required outcome. Keep the official test set sealed.

### Day 4 — AlexNet ideas, implemented at a manageable scale

**Time:** 4–5 hours. **Goal:** identify transferable architectural ideas and distinguish adaptation from replication.

**Recall first:** describe the most common CIFAR validation error and one limitation of yesterday's comparison.

**Read:** [AlexNet, sections 1–4](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). Limit initial reading/notes to approximately 60–90 minutes. Focus on ReLU, convolutional structure, dropout and the historical role of local response normalization (LRN).

**Derive:** compare the derivatives of ReLU and tanh away from their special points. Explain why ReLU avoids positive-side saturation but can still have inactive units; do not call it a universal gradient fix.

**Build and verify:**
1. Implement a compact AlexNet-inspired model for 32×32 inputs. Choose kernels/strides that preserve sensible spatial sizes; do not copy the ImageNet input geometry blindly.
2. Record a layer/shape table and parameter count. Reuse your CIFAR pipeline and training loop.
3. Run one budgeted short training experiment, at most 10 epochs or a predeclared time cap. Choose the shared budget with tomorrow's VGG-style experiment now.
4. Write a short table: original idea, your adaptation, and the reason for the change. Label the result “AlexNet-inspired on CIFAR-10,” not a reproduction of ImageNet accuracy.

**Questions:**
1. Which two design choices transfer across input sizes, and which two need resizing or reconsideration?
2. Why can a very large fully connected head dominate parameter count?
3. What did LRN try to achieve, and why does its historical use not establish that it will help your model?
4. Why does a more complicated model losing under a small budget not settle its full-training potential?

**Done when / evidence:** a checked architecture, one run, validation metrics and an adaptation note grounded in the paper.

**Optional:** one LRN on/off comparison, only if the required work is finished and at most one additional matched run fits the day's budget. It is not a completion gate.

### Day 5 — VGG-style blocks and the limits of architecture comparisons

**Time:** 4–5 hours. **Goal:** connect receptive field, parameter count and nonlinear composition.

**Recall first:** explain one AlexNet adaptation and identify what the previous day's budget permits you to claim.

**Derive:** at stride 1 and dilation 1, two 3×3 convolutions have a 5×5 theoretical receptive field; three have 7×7. For equal input/intermediate/output width C and ignoring bias, two 3×3 layers use 18C² weights versus 25C² for one 5×5. This parameter advantage is conditional on channel widths; calculate it explicitly when widths differ. Equal receptive field does not mean equal functions.

**Build and verify:**
1. Implement a small VGG-style network using stacked 3×3 convolutions with nonlinearities. Keep the overall size appropriate to the available GPU.
2. Verify shapes and parameter counts; train using the same CIFAR split and the comparison budget selected yesterday.
3. Compare validation loss/accuracy, parameter count, updates completed and wall time against AlexNet-inspired and the simple CNN where budgets permit.
4. Write which aspects are controlled and which differ. If you reused an unmatched earlier result, label it contextual rather than a controlled head-to-head.

**Questions:**
1. What function-class difference does an intermediate ReLU introduce?
2. Why does changing channel width invalidate a blanket “two 3×3s always use fewer parameters” claim?
3. How can you increase receptive field without adding the same number of parameters as a large dense kernel?
4. If your accuracy difference is small, which additional evidence would be more useful than declaring a winner?

**Done when / evidence:** correct receptive-field/parameter calculations, one model run and a comparison with explicit limitations. You do not need to make VGG beat AlexNet.

**Optional:** plan a plain-versus-residual depth experiment on paper. Execution is optional, not a second hidden full training project.

### Day 6 — ResNet-20, a documented recipe and an honest checkpoint

**Time:** 6–8 focused hours. **Goal:** implement the CIFAR architecture correctly, measure training cost and preserve a usable model. A full paper-level accuracy replication may require additional GPU time.

**Recall first:** derive the Jacobian of a linear map and explain why a depth comparison should control more than the dataset name.

**Read:** [ResNet, Section 4.2](https://arxiv.org/pdf/1512.03385). The paper's ResNet-20 has approximately 0.27M parameters and reports 8.75% test error (91.25% accuracy). This is a reference result, not a one-day pass condition.

**Derive carefully:** for y=x+F(x), J = I+J_F. The identity term supplies a direct path; it does not guarantee a lower bound on the total gradient because cancellation is possible. In an original post-activation block, a ReLU follows the addition, so the full block derivative also includes its activation mask. Avoid the statement “skip connections guarantee gradients never vanish.”

**Build and verify:**
1. Implement the CIFAR 6n+2 architecture with n=3: initial 3×3 convolution; three stages with 16/32/64 channels and three two-convolution residual blocks per stage; global average pooling; ten-class linear head.
2. Document downsampling at stage transitions and the shortcut choice. For the paper's CIFAR option A use spatial subsampling and zero channel padding, not an unexplained projection. If you choose a learned projection, label that deviation.
3. Check shapes through both same-size and downsampling blocks, parameter registration/count, a forward/backward pass and tiny-batch memorization. Confirm that both branches have identical shapes at addition.
4. Use a recorded recipe. The paper reference uses SGD, momentum 0.9, weight decay 1e-4, batch size 128, initial LR 0.1, LR reductions at 32k and 48k steps and a 64k-step endpoint; no dropout. Record initialization, preprocessing and any deviation. A shorter schedule is a budgeted adaptation, not the same training protocol.
5. Run a short timing pilot after warmup. Estimate the full run's wall time from measured step time, including validation/checkpoint overhead. Decide a maximum duration before starting.
6. Default allocation: one main run, capped at 2 GPU-hours today. One corrected restart is allowed only for a diagnosed issue and inside that total cap. If the full chosen recipe exceeds the cap, save a resumable intermediate checkpoint and record the remaining schedule; do not silently compress it and call it a replication.
7. Inspect learning curves, compare against the chance/constant baseline and verify reload on a fixed validation batch. Save the checkpoint durably with its exact configuration and preprocessing.

**Questions:**
1. What does the identity path provide, and why is “guaranteed nonzero gradient” too strong?
2. What changes when the shortcut crosses a stage boundary?
3. Why does the same architecture with a different training schedule not constitute the same experiment?
4. What evidence separates “still undertrained” from “probably broken” in your run?
5. What would Month 7 need besides a .pth file to use this model correctly?

**Done when / evidence:** the implementation checks pass; a real budgeted run has learned a meaningful signal on training/validation data or a remaining failure is explicitly diagnosed; checkpoint reload reproduces fixed-batch predictions; record validation metrics and status. If it cannot learn a tiny batch, it is not complete—repair that before using the checkpoint.

**Accuracy and later-use status:** approximately 90%+ final CIFAR-10 test accuracy remains an aspiration for a stronger teacher, not a forced selection target. Select by validation metrics only. Mark the saved model either “validated development checkpoint” or “candidate teacher; needs additional training/assessment.” Week 4 Day 1 provides one predeclared continuation window if needed. A weaker honest checkpoint does not block tensor/attention study in Month 4; strengthen and reassess it before relying on it for Month 7 distillation. Do not claim “research-grade” from architecture or accuracy alone.

**Optional depth study:** compare matched-width plain and residual networks at two predefined depths with equal data/update budgets. At most four short runs; record the total compute cap before starting. Compare training and validation errors separately. Failure to reproduce degradation is a valid result. Do not keep increasing depth until a desired plot appears.

### Day 7 — Protected rest

**Time:** no coursework. ResNet is checkpointed before the break. Additional training belongs in a planned compute/study window, not in a day spent anxiously watching a runtime.

**Done when:** you have taken a real break and have a written next action if the checkpoint needs more work.

---
## WEEK 3 — Sequence models with valid tests

**Week goal:** verify recurrent cells, investigate memory limits without forcing failure, and build the exact kind of shifted-target text pipeline that Months 4–5 need.

### Day 1 — RNN cell and gradients through time

**Time:** 5–6 hours. **Goal:** implement recurrence and understand the full Jacobian that governs gradient propagation.

**Recall first:** explain J = I+J_F and why it is not a guarantee against all gradient problems.

**Derive:** using column-vector notation, a_t = W_xh x_t + W_hh h_(t-1) + b and h_t = tanh(a_t). Then dh_t/dh_(t-1) = diag(1−h_t²) W_hh. Across time, these Jacobians multiply. Weight singular values, activation derivatives and their directions all matter; tanh's bounded derivative alone does not prove that every RNN must vanish.

**Correct notation exercise:** compute tanh'(a)=1−tanh²(a) at preactivation a=0,1,2. Do not substitute h=2 into a formula where h is a tanh output: h lies between −1 and 1.

**Build and verify:**
1. Implement one cell with raw PyTorch tensor operations and nn.Parameter weights, then unroll a short sequence. Use batch-first input and label every dimension.
2. Copy matching parameters to/from a reference nn.RNNCell. Compare outputs, final hidden state, input gradients and parameter gradients on a tiny deterministic sequence. Match bias conventions explicitly.
3. Put a loss on the final state, retain selected intermediate hidden gradients and plot their norms over time. Explain the indexing and loss location.
4. Compare a deliberately contractive setup with a different recurrent-weight scale using the same input. Report what happens; if the original initialization shows no dramatic decay, that is not a correctness failure.

**Questions:**
1. Why does recurrent reuse of one parameter require summing its contributions over time?
2. Why is a final-state-only loss helpful when isolating long gradient paths?
3. How can gradients explode even though tanh' is at most one?
4. Why would detaching the hidden state at every step destroy this investigation?

**Done when / evidence:** cell/reference checks pass within a stated tolerance, two gradient plots are recorded and the observed behaviour is explained. A vanishing curve under every initialization is not required.

**Optional:** finite-difference check a few selected parameters on a tiny double-precision example. Avoid replacing the main implementation with an autodiff library wrapper.

### Day 2 — The copy task, defined precisely

**Time:** 5–6 hours. **Goal:** measure how recall performance changes with a delay while keeping data and metrics meaningful.

**Recall first:** derive the one-step RNN Jacobian and explain why a near-zero training loss is not evidence of held-out recall.

**Task specification:**
- Use eight content symbols, plus distinct BLANK and GO input tokens. Let memory length M=5 initially.
- Input: M content symbols, G BLANKs, one GO, then M BLANKs. During the final M positions the model must output the original M content symbols in order.
- Use an eight-class output head for the recall symbols. Compute cross-entropy only on the final M recall positions; do not reward predicting BLANK during the long delay.
- Create fixed, disjoint sets of unique content sequences for training, validation and final test, using a recorded seed; suggested sizes are 4096/1024/1024 from the 8^5 possible sequences. Reuse these splits across gap lengths and models.
- Reset recurrent state between independent sequences. Do not detach inside a full copy sequence: the gradient must be able to traverse its memory gap.
- Report recall-token accuracy and whole-sequence exact-match accuracy separately. The independent uniform baseline has expected token accuracy 1/8 and exact-match probability (1/8)^M.

**Build and verify:**
1. Print an encoded and decoded example, the loss mask/positions and the target. Confirm no original content is fed again during the recall phase.
2. Fit a tiny set of short-gap sequences first. Confirm the training and reference implementation agree on the task convention.
3. Predefine G in {5,10,15,20,30}, a shared update cap and a per-run time safety cap. Suggested starting budget: up to 500 updates per gap with a 10-minute cap each. If capped runs finish different updates, report that and avoid calling the final endpoints update-matched.
4. Train the vanilla RNN from a recorded initialization for each gap. Plot validation token and sequence accuracy versus gap, plus hidden-gradient norms for one fixed diagnostic sequence.
5. If no failure appears, report the tested range as solvable under this setup. An optional larger-gap probe must be declared separately; it is not permission to keep searching until a failure appears.

**Questions:**
1. Why would counting BLANK accuracy make a poor model look successful?
2. Why does a model with 90% token accuracy not necessarily have 90% exact-sequence accuracy?
3. What can a shrinking gradient plot suggest, and why does it not alone prove the cause of an accuracy failure?
4. If both training and validation accuracy remain low, what would you inspect before blaming memory length?

**Done when / evidence:** correct task encoding/loss positions, a successful short-gap tiny-set fit, the predefined gap sweep and honest plots. “RNN must fail at 20” is not a requirement.

### Day 3 — LSTM cell, memory paths and the original paper

**Time:** 6–8 hours; split if necessary. **Goal:** understand and verify a modern LSTM, while separating its equations from the historical original.

**Recall first:** explain your copy-task loss positions and summarize the RNN result without inventing a collapse point if none appeared.

**Read:** selected conceptual sections of [Hochreiter & Schmidhuber's 1997 LSTM paper](https://people.idsia.ch/~juergen/lstm1997-2024head.pdf), especially the constant error carousel; use the modern [PyTorch LSTM equations](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) for implementation. Time-box the first paper pass to about 60–90 minutes and list unresolved notation. Full mastery of every original derivation is not today's gate.

**Derive:** i_t and f_t are sigmoid input/forget gates; g_t is a tanh candidate; o_t is a sigmoid output gate. c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t and h_t = o_t ⊙ tanh(c_t). This is three sigmoid gates plus a candidate, not four independent sigmoid gates.

The direct cell-state path multiplies forget-gate values across time. With values near one, it can preserve signal much better; values consistently below one can still attenuate it. Other dependencies contribute to the full derivative. The additive update does not magically remove every repeated multiplication or all vanishing/exploding gradients. The modern forget gate was introduced after the original 1997 model.

**Build and verify:**
1. Implement a small LSTM cell from raw operations; no nn.LSTM/nn.LSTMCell inside the custom implementation. Autograd is allowed for training this cell: manual LSTM backward is not required.
2. Compare with nn.LSTMCell after copying identical parameters. Match the i/f/g/o gate order and bias convention; compare h, c, input gradients and parameter gradients on a tiny sequence.
3. Use the same copy-task split, loss and evaluation protocol. Choose a gap from yesterday before looking at LSTM results—one difficult for the RNN if available, otherwise the largest tested gap.
4. Compare under a stated update budget; record parameter counts and time because equal hidden width does not mean equal model size or compute. Start with one run and one diagnosed adjustment inside a fixed budget.
5. Plot hidden-state and cell-state gradient norms separately for a fixed sequence. Keep axes/scales and the diagnostic loss location consistent with the RNN comparison.

**Questions:**
1. What should the forget gate do when information must survive a long delay?
2. Why does a product of 0.9 values still decay over many steps?
3. Which part of your implementation differs historically from the 1997 formulation?
4. If the LSTM also fails, which implementation and training checks come before concluding that gating is useless?
5. Why is an equal-hidden-width comparison informative but not parameter-matched?

**Done when / evidence:** numerical reference checks pass, the comparison and gradient plots exist and your interpretation separates implementation evidence from performance. An LSTM victory is not mandatory. An inability to fit the short-gap tiny set remains an unresolved issue to repair.

### Day 4 — Protected rest

**Time:** no coursework or recall questions. Carry any unfinished derivation into an additional study block rather than making rest conditional on it.

**Done when:** you have had a real break from sequence-model work.

### Day 5 — GRU cell and a controlled three-model comparison

**Time:** 4–5 hours. **Goal:** implement a compact gated recurrence and compare it fairly with your earlier cells.

**Recall first:** write the LSTM state update and explain one limitation of your RNN/LSTM comparison.

**Read:** [PyTorch GRU equations and implementation note](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html). Choose this convention for numerical parity; some descriptions apply the reset gate before the recurrent matrix multiplication, whereas PyTorch applies it after the hidden affine transform in the candidate equation.

**Derive:** r and z are sigmoid reset/update gates; n is the candidate. In PyTorch's convention, n = tanh(W_in x + b_in + r ⊙ (W_hn h_prev + b_hn)); h = (1−z) ⊙ n + z ⊙ h_prev. Write the reset/update affine equations and all shapes before coding. A different update-gate convention is valid if consistently defined, but cannot be compared by blindly copying parameters.

**Build and verify:**
1. Implement the GRU cell and compare outputs/gradients with nn.GRUCell after aligning gate order, weights and biases.
2. Count parameters from actual shapes for all three cells, including biases and the output head. Do not infer speed from the number of named “matrices” alone; libraries may fuse operations.
3. Run the same selected copy-task gap and split as the LSTM experiment with the same stated update/time policy.
4. Make a table of parameter count, elapsed training time, validation token accuracy and exact-match accuracy for RNN/LSTM/GRU. Disclose whether hidden width or parameter count was matched.
5. If timing cells implemented in Python loops, compare those implementations consistently; do not present a fused nn.LSTM versus Python GRU timing as an architecture-only result.

**Questions:**
1. Why are reset-before-matmul and reset-after-matmul generally different?
2. How does the GRU combine old and candidate state?
3. When might a separate LSTM cell state be useful despite added parameters?
4. What evidence would justify a speed claim about this implementation on this hardware?

**Done when / evidence:** the custom cell passes reference checks and the comparison table reports the observed tradeoffs. Neither GRU speed superiority nor matching LSTM accuracy is required.

### Day 6 — A correct character-language-model pipeline

**Time:** 6–8 hours. **Goal:** train and evaluate next-character prediction with no target leakage, and generate text autoregressively.

**Recall first:** connect cross-entropy to next-character likelihood. Explain why copy-task recall positions were masked.

**Data:** Tiny Shakespeare from the original [Karpathy char-rnn data repository](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare). Record the source and a checksum of the downloaded file. Read a page and record structure, capitalization and punctuation before training.

**Build the data pipeline:**
1. Split the raw text into fixed contiguous regions, for example 80% training, 10% validation and 10% final test. Record exact character offsets. Build vocabulary from training text, with an explicit unknown-character policy; log unknown rates.
2. Build each (input,target) pair from a length T+1 substring within one split: input is the first T IDs, target the final T IDs. Confirm by decoding a short example and checking the one-character shift.
3. Make training and validation windows entirely within their own regions. Validation uses fixed positions; use a documented coverage/stride policy and a token-weighted aggregate loss. Do not randomly split already-overlapping windows.
4. Use input IDs (B,T), embedding output (B,T,E), recurrent output (B,T,H) and logits (B,T,V). For ordinary class-index cross-entropy, reshape logits to (B×T,V) and targets to (B×T), or permute to the supported class-axis layout. Verify the ordering matches; shape alone is not enough.

**Build the model and training loop:**
1. Use Embedding → a modest one-layer nn.LSTM → Linear as the default practical model. Your custom cell is already verified; fused library recurrence is appropriate for longer training. Record vocabulary, embedding/hidden sizes and parameter count.
2. Start with independently sampled fixed-length chunks and reset hidden state per chunk. This is the required simple version. If you later carry state, preserve stream continuity and detach both h and c between truncated segments; never carry state across shuffled unrelated chunks.
3. Train with a recorded optimizer/LR and finite checks. Clip global gradient norm after backward and before optimizer.step; choose and log a threshold such as 1.0 along with the pre-clipping norm. Detailed optimizer/AMP work remains in Month 5.
4. Run a short tiny-batch diagnostic with regularization disabled, then a budgeted full run. Keep a fixed maximum duration and evaluate at consistent intervals. Save best-validation and last-resume checkpoints separately.
5. Compare validation mean cross-entropy with an input-independent unigram baseline fitted from training counts, with documented smoothing. Report perplexity as exp(mean CE) when using natural-log CE; compare only under the same vocabulary, split and evaluation convention.

**Generate:** switch to evaluation mode, disable gradients and prime the hidden state with a seed prompt. Then feed back one generated character at a time. The generation path must not receive true future characters. Implement greedy decoding first; for positive temperature T_s, sample from softmax(logits/T_s). Use a generation seed for comparable stochastic examples.

**Questions:**
1. Why would predicting each input character itself yield misleadingly good loss?
2. Why does splitting overlapping windows at random leak information?
3. Why is teacher-forced validation different from free-running generation?
4. What does detaching hidden state change, and when is resetting it the right choice?
5. Why should validation loss determine model selection rather than the nicest sampled passage?

**Done when / evidence:** decoded alignment examples, split boundaries, baseline/model validation loss, a checkpoint, a generation sample and named failure observations. The model must show learning under a correct pipeline; “looks like Shakespeare” alone is not validation. If it cannot learn a fixed tiny batch, fix that before tuning.

**Optional:** explore temperature on a few fixed prompts. Top-k, top-p and more sampling methods remain in Month 5; do not expand this day into that month's syllabus.

### Day 7 — Retrieval, a small independent rebuild and cleanup

**Time:** 4–5 hours. **Goal:** prove that you can reconstruct and explain the pipeline rather than only execute existing notebooks.

**Recall first:** from a blank page, write RNN/LSTM/GRU state equations, the language-model input/target shapes and your train/validation split rule. Check afterwards and repair each error.

**Build and verify:**
1. Without copying the existing implementation, rebuild a tiny text-window function from a toy string and verify its decoded input/target pairs. Documentation is allowed.
2. Write a compact training/evaluation-loop skeleton from memory; check it against your working version and annotate missing steps.
3. Re-run the recurrent-cell numerical checks and inspect one saved training log. Cleanup must not silently change equations, splits or results.
4. Organize sequence notebooks, configs, figures and run records. Restart and Run All for the small rebuilt-window/cell-check sections using the documented lightweight mode. If using Git, commit the relevant notebook revision; public publishing is optional.
5. Make a short backlog: correctness issues first, optional extensions last. Bring at most one unresolved non-blocking performance question into the capstone; fix core data/gradient failures before tuning.

**Questions:**
1. Which equation or implementation step did you genuinely reconstruct incorrectly?
2. Which comparison supports your strongest conclusion, and what is its main limitation?
3. Why can tokenization differences make two perplexity numbers incomparable?
4. What will you reuse in Month 4, and what must change when recurrence becomes attention?

**Done when / evidence:** corrected retrieval sheet, working independently rebuilt window function, readable project structure and an honest issue list. Memorizing the exact numerical results of every run is not required; know where their evidence lives.

---
## WEEK 4 — Tune, evaluate, use and explain your models

**Week goal:** turn working experiments into independently usable artifacts, without adding an unrelated technology stack or relying on cherry-picked samples.

### Day 1 — Budgeted tuning with a valid selection rule

**Time:** 5–6 hours. **Goal:** select a language-model configuration using validation evidence and a fixed budget.

**Recall first:** reconstruct the shifted-target example and explain why final test results cannot choose a learning rate.

**Plan before running:** use at most five configurations, including a rerun of the baseline under the same budget. Default: 20–25 minutes of training per configuration, with validation overhead recorded separately. This is a fixed-wall-time practical comparison. Record updates/tokens processed, since models with different throughput will see different amounts of data.

**Suggested five-run design:** baseline; lower learning rate; higher learning rate; one larger hidden size; one weight-decay change. State exact values based on your baseline before starting. Change one intended factor per comparison where possible. The proposal is a manageable design, not a promise these settings are optimal. If your single-layer nn.LSTM has dropout set on the module, remember that built-in recurrent-module dropout is between layers and has no such inter-layer effect with only one layer; do not “tune” an inactive setting.

**Build and verify:**
1. Reuse frozen splits, vocabulary, context length, evaluation windows and a consistent seed policy. Confirm the baseline is correct before spending the search budget.
2. Store the predeclared configurations in a short list/dictionary table and run them with your own checked training functions. Create fresh run state for each independent configuration. Use pandas to collect validation mean CE, runtime, tokens/updates and checkpoint path; no AutoML system is needed for five controlled settings. Save the best validation checkpoint within each run.
3. Choose by the declared validation metric. The existing baseline may remain best. If differences are tiny, say so; do not manufacture a “clearly better model.”
4. Write what one hypothetical next experiment would test, without automatically running it. A repeat-seed check of a headline result takes priority over a speculative sixth configuration when extra budget is available.
5. Freeze selected model configurations, preprocessing and reporting metrics before tomorrow's final evaluation.

**ResNet continuation, if needed:** one separate, predeclared window of up to 2 additional GPU-hours may continue the checkpoint from Week 2. Preserve its original schedule/optimizer state. Record it as continuation, not an independent attempt. If it competes with the LSTM runs for resources, schedule a buffer day. Do not sacrifice rest or exceed free-service limits. Select using validation data only; if the desired teacher quality remains unmet, record the outstanding work for before Month 7.

**Questions:**
1. Why is the baseline a legitimate winner?
2. What question does an equal-wall-time comparison answer, and what does it leave confounded?
3. Why might the lowest single validation reading be noisy?
4. What evidence motivates your hypothetical next experiment?

**Done when / evidence:** a table of the completed predeclared configurations, a documented selection decision and frozen final-evaluation settings. If the resource budget only permits three runs, report three; do not invent the remaining results or expand the budget silently.

### Day 2 — Final metrics, samples and failure analysis

**Time:** 4–5 hours. **Goal:** produce an honest final evaluation for a predeclared model shortlist.

**Recall first:** name the selected configuration, its selection metric and the largest unresolved uncertainty.

**Build and verify:**
1. Freeze the shortlist before opening test data: normally the final MNIST model, the simple CIFAR baseline, ResNet-20 and the selected character LSTM. Evaluate additional saved variants only if declared in advance and useful to the report. Do not use their test results to restart model selection.
2. Run deterministic preprocessing, evaluation mode and no-grad/inference mode. Use scikit-learn for classification accuracy/confusion matrices and retain per-example predictions with their IDs/targets. Hand-check a few examples and the class order; do not rebuild the Month 2 metric library. Use pandas for the result table and plotting libraries for figures.
3. Evaluate the LSTM and the training-fitted unigram baseline on the same test region, vocabulary and window/coverage policy. Report mean CE and derived perplexity. Record any unknown-character rate.
4. If reporting final copy-task performance, use the held-out sequence set and the same recall-only metrics. Keep validation-gap sweeps clearly labeled as development results.
5. Generate equal-length passages for a fixed small prompt set using greedy decoding and a few positive temperatures, for example 0.5, 1.0 and 1.3. Keep prompts, output lengths and seed policy fixed; retain all samples from the declared set.
6. Inspect errors and samples. Record observed repetition, malformed structure, invented words, capitalization or long-context inconsistency where they actually occur. Do not require exactly three distinct failures if the sample set does not support that.

**Questions:**
1. Why can a fluent-looking passage coexist with worse held-out cross-entropy?
2. Why cannot you directly compare this character perplexity with a BPE-token GPT perplexity in Month 5?
3. What would you report if test performance is lower than validation performance?
4. Why is an apparently confident prediction on an unfamiliar image not proof of calibrated confidence?

**Done when / evidence:** final metric tables, stored evaluation settings, a fixed sample set and concrete error notes. Final test results are reported as observed. If an implementation error invalidates an evaluation, document the correction and rerun; do not disguise performance tuning as a bug fix.

### Day 3 — Use saved models in separate inference notebooks

**Time:** 4–5 hours. **Goal:** use new inputs after a kernel restart, with no training-session state and no retraining.

**Recall first:** list every item needed to reconstruct your selected model and preprocess its input correctly.

**Build and verify:**
1. Create vision/06_predict_images.ipynb with a small configuration cell for image path, checkpoint/config paths and device. Use Pillow/torchvision for decoding and deterministic preprocessing. Define the exact selected architecture, load its state_dict and saved class mapping, and show the original image, model input and prediction. Do not rebuild image decoders or a file-browser interface.
2. Create sequences/04_generate.ipynb with configuration for checkpoint, prompt, output length, seed, temperature and decoding mode. Reconstruct the selected model and load its vocabulary/configuration. Implement the autoregressive generation loop yourself; library tensor sampling is appropriate. Handle unknown prompt characters explicitly. Greedy decoding is a separate mode, not division by zero temperature.
3. Keep only required imports, definitions, preprocessing, loading, checks and inference in these notebooks. Do not execute a training notebook through %run/import tricks. Small copied model definitions must identify their training-notebook source revision and pass the same fixed-input parity check. Save the exact definition version used for each checkpoint.
4. Restart each kernel and Run All. On the small models, support CPU loading/inference where practical. Compare fixed evaluation logits/predictions against saved reference outputs with a stated tolerance. For sampling, a seed alone does not guarantee identical generated text across devices or library versions; use fixed-input logits as the primary reconstruction check.
5. Test five to ten newly drawn digit images, or another small set of appropriate new inputs. Inspect polarity, crop, aspect ratio, size and scaling. Save preprocessing examples and outputs. Treat the personally selected set as a qualitative transfer check, not a representative accuracy benchmark.
6. Add understandable errors for missing checkpoints, invalid paths/images and unsupported text symbols. A notebook with a configuration cell and displayed results is sufficient. No script, web app or dashboard is required.

**Questions:**
1. Why can identical weights produce different predictions when preprocessing changes?
2. Which variables or objects did the first clean-kernel run reveal were missing, and how did you remove that dependency?
3. Why might a model trained on centered MNIST digits struggle with a phone photo of handwriting?
4. Which failure is a preprocessing bug, which is a distribution/model limitation, and what evidence distinguishes them?

**Done when / evidence:** both saved inference notebooks pass Restart + Run All using the README instructions, without training. Fixed-input parity passes, configuration and source revisions are recorded, and new-input observations are saved. This is the practical independence demonstration.

### Day 4 — Write a report with two defensible case studies

**Time:** 4–5 hours. **Goal:** explain what your evidence supports without turning unrelated metrics into a false architecture contest.

**Recall first:** explain inductive bias in your own words and identify one result that contradicted or weakened your initial prediction.

**Write a concise report with these sections:**
1. **Question and summary:** what you tried to learn, and the main observed results.
2. **Setup:** datasets, split policies, models, optimizer/budget, hardware, seed count and selection rule. Link exact run records instead of burying the reader in every detail.
3. **Vision case study:** the controlled augmentation or small-CNN comparison, plus the ResNet result with its training-budget status. State when architectures were not compute-matched.
4. **Sequence case study:** copy-task memory behaviour and the language-model baseline/held-out evaluation. Show token versus exact-sequence metrics where relevant.
5. **Failure analysis:** one real bug, one ordinary model limitation, and the new-input inference observations. Explain how you distinguished them.
6. **Limitations:** toy datasets, small number of seeds, hardware/time caps, unknown characters if any and incomplete paper replication where applicable.
7. **Next steps:** one actionable experiment per case study, chosen from evidence rather than an arbitrary list of advanced topics.

Include at least two readable figures/tables with labeled axes, metrics and splits. Present the CNN and sequence results as related learning case studies. CIFAR classification accuracy and Shakespeare perplexity cannot establish a head-to-head CNN-versus-RNN winner.

**Questions:**
1. Which conclusion is directly supported by a controlled comparison?
2. Which statement is only a hypothesis or interpretation?
3. What would a reader need to reproduce your main figure?
4. Which limitation would most likely change your conclusion if investigated?

**Done when / evidence:** the report is complete, references real runs and includes limitations. Save report.ipynb with Markdown explanations and figures generated from the saved result tables. Its analysis cells should read saved evidence, not rerun all training. A Markdown/HTML export is optional for sharing; keep the notebook and its linked evidence as the local source. Publishing to a repo/blog is a useful optional portfolio step after review; public posting is not a prerequisite for scientific correctness or for progressing to Month 4.

**Optional:** if you specifically want an architecture head-to-head, design a shared-task comparison with matched splits and a declared budget. Do not add it retroactively merely to justify a CNN-versus-RNN title.

### Day 5 — Reproduction, archive and readiness for Month 4

**Time:** 4–5 hours. **Goal:** ensure that the month's outputs are recoverable and that your remaining gaps are visible.

**Recall first:** explain the sequence-logit shape and trace the tensor split/merge operations that Month 4 will use.

**Build and verify:**
1. Follow only your README: restart kernels and Run All in the documented smoke/evaluation modes for one short training run, one evaluation and both inference notebooks. Confirm that inference loads saved weights without training. Verify the recorded environment instructions in your chosen runtime; if you did not recreate the environment, say so. This checks notebook execution and artifact recovery, not a repeat of every expensive experiment.
2. Record package versions and necessary installation instructions. Include model configs, split definitions, preprocessing, metric definitions and checkpoints or durable download locations.
3. Archive the ResNet checkpoint and its manifest. Include architecture/shortcut option, class ordering, input normalization, model/optimizer/scheduler state where relevant, completed steps, validation-selected status, final test metric if evaluated, hardware and known limitations.
4. Verify the archive by loading it and comparing fixed-input logits. Store a checksum and a small fixed reference input/output fixture or its reproducible source/index. Keep large checkpoints out of ordinary Git history when unsuitable; document their storage location.
5. Complete the readiness checklist below. Repair any core correctness failure before marking the month complete. Missing optional LRN/depth experiments and an unfinished 90% aspiration do not fail the month.
6. Make a short Month 4 handoff note: reusable notebook/function cells with source revisions, verified tensor conventions, data contracts, open research questions and any checkpoint work needed before Month 7. Keep the notebook-first workflow for learning attention; extracting mature utilities into modules later is optional, not a prerequisite to begin Month 4.

**Questions:**
1. What can a stranger reproduce using only the README and archived assets?
2. What exact evidence shows the archived ResNet can be reconstructed?
3. Which model-development habit will prevent the most likely attention implementation bug?
4. What remains unfinished, and is it a correctness blocker, a performance aspiration or optional work?

**Done when / evidence:** the reproduction checklist passes, the archive is verified and the Month 4 handoff note identifies actual readiness and remaining work. A public GitHub push is optional; a usable saved project is required.

### Day 6 — Protected rest

**Time:** no coursework. Finish the core readiness check before declaring the month complete, or schedule a repair day; do not turn this into a hidden repo-cleanup day.

**Done when:** you have taken a real break before beginning Month 4.

---

## 4. End-of-month readiness checklist

Use evidence, not confidence alone. Documentation is allowed; unexplained copying of the complete solution is not evidence of independent skill.

| Capability | Required evidence | If not yet ready |
|---|---|---|
| Training pipeline | Restart-and-Run-All smoke training/evaluation with your own checked loop and documented mode | Repair loop/data handling before Month 4 |
| Tensor literacy | Correct (B,T,D) split/merge, batched matmul and softmax-axis checks | Repeat the focused shape workshop |
| Backprop understanding | Manual MLP/BatchNorm gradient comparison passes and an injected error is explained | Repair the failing operation; do not redo the entire month |
| Evaluation discipline | Frozen splits, validation-based selection, final test reporting and no overlapping text-window leakage | Correct the protocol and label any invalidated result |
| Debugging independence | Two diagnosed bugs plus an explained learning curve | Do one focused diagnostic exercise |
| Vision learning | Trained small CNN and a verified ResNet implementation/checkpoint with honest quality status | Fix correctness now; schedule additional teacher training separately if needed |
| Sequence mechanics | RNN/LSTM/GRU cell parity checks and a correctly evaluated copy-task comparison | Repair equation/layout/task errors before using the conclusion |
| Language modeling | Correct shifted targets, held-out CE against a baseline and autoregressive generation | Repair the pipeline before starting a larger model |
| Practical use | Two inference-only notebooks pass after kernel restart using saved checkpoints, with fixed-input parity and new-input notes | Remove hidden notebook dependencies |
| Communication | Research report notebook, saved configs/run tables, exact notebook revisions and verified archive | Finish the minimum documentation needed to reproduce the work |

**Strong performance targets:** improve on meaningful baselines; aim for a strong ResNet teacher and a language model that uses context. These guide investigation. They do not justify repeatedly changing a test-driven experiment until the headline number looks good.

### What carries forward

| Destination | Reuse from V5 | Boundary to remember |
|---|---|---|
| Month 4 attention | Batched matmul, explicit axes, module registration, residual reasoning and gradient checks | Correct shapes alone do not prove causal masking or KV-cache correctness |
| Month 4 MiniGPT data | Dataset/DataLoader pattern, shifted targets, fixed splits and token-weighted loss | BPE changes vocabulary and token counts; do not compare raw perplexities across tokenizations |
| Month 5 training | Logging, validation selection, checkpoint recovery, clipping and fixed-budget comparisons | AMP, accumulation and advanced schedules add state and correctness checks; build them incrementally |
| Month 7 vision/distillation | CIFAR model, exact preprocessing, class mapping, baseline metrics and checkpoint manifest | A saved model is not automatically an effective teacher; verify quality and freeze/evaluate it appropriately before distillation |

The existing later-month files are context, not infallible instructions. Carry V5's experimental and notebook-execution rules forward even where later prose predicts a mandatory winner. In particular, equal epochs/tokens are not equal compute across different model sizes, and a small-model demonstration does not establish a general scaling law. For a rough transformer compute-matching exercise in Month 5, account for both parameter count and processed tokens, then disclose the approximation.

## 5. How to answer the practice questions

Write the answer before running the related experiment where possible. A complete answer usually has: the mechanism or equation, the relevant assumptions, a prediction, and the observation that would support or challenge it. Afterward add what actually happened. A correct negative result is acceptable; a failed implementation check is not.

Use a short retrieval cycle: at each study day's warm-up recall yesterday's central idea plus one earlier-week concept. At the end of each week reconstruct one small component without copying it. Revisit concepts you missed after a few study days. Do not turn review into memorizing every metric or rereading the whole roadmap.

### Quick correction anchors

- Uniform predictions give mean classification CE ln(K), independent of label balance. Arbitrary initialized logits need not be uniform.
- A variance-of-the-mean argument needs assumptions about independence/covariance; it is not simply “because CLT.”
- Dropout preserves an activation expectation under the stated scaling; it does not preserve everything downstream.
- BatchNorm train-mode coupling is intentional. Batch size one alone does not specify whether sufficient values per channel exist.
- Non-leaf gradients require explicit retention or an appropriate autograd query for inspection.
- Numerical gradient agreement requires matching the forward function, conventions, parameters and precision.
- An identity shortcut gives a direct path, not an unconditional gradient guarantee.
- LSTM memory depends on gates and optimization; the additive state update is not immunity from vanishing gradients.
- A correct tiny-batch fit checks part of the learning machinery; it cannot certify clean labels or generalization.
- A fixed experiment budget can produce an inconclusive result. Report the limitation rather than expanding until the desired result appears.
- Equal hidden size, parameter count, updates, wall time and FLOPs are different controls. State which one you used.
- A pretrained model is useful in many applications, but implementing/training small models yourself serves this roadmap's immediate learning goal. Transfer learning can be a later extension instead of another required topic now.

## 6. Notebook deliverables and day-to-notebook map

Use one project folder with the following small collection. Each notebook has multiple clearly titled sections; a day does not require another duplicated project. All paths below are suggested deliverables you create during learning, not files supplied by this roadmap.

    month3/
      00_setup.ipynb
      foundations/
        01_tensors_autograd.ipynb
        02_training_pipeline.ipynb
        03_regularization_diagnostics.ipynb
        04_manual_backprop.ipynb
      vision/
        01_mnist.ipynb
        02_convolution_shapes.ipynb
        03_cifar_architectures.ipynb
        04_resnet20.ipynb
        05_evaluate.ipynb
        06_predict_images.ipynb
      sequences/
        01_cells_copy_task.ipynb
        02_text_pipeline_training.ipynb
        03_tuning_evaluation.ipynb
        04_generate.ipynb
      report.ipynb
      configs/                  # JSON configs and environment version records
      splits/                   # Image indices and raw-text region boundaries
      results/                  # CSV run tables, figures, predictions, check summaries
      checkpoints/              # Weights/state or manifest pointing to durable storage
      README.md
      checkpoint_manifest.md
      month4_handoff.md

**Keep responsibilities clear:** foundations/02 is the canonical teaching version of the training loop; task notebooks contain the small verified adaptation they actually execute and name its source revision. Do not depend on importing an already-run notebook. Keep one definition of each helper/class inside a notebook. If a bug is fixed in a shared copied cell, update affected copies and rerun their small checks. A later module extraction can reduce duplication, but it is not required work this month.

| Day | Notebook and section | Library help / your learning focus |
|---|---|---|
| W1 D1 | foundations/01_tensors_autograd — tensor and gradient worksheet | PyTorch operations; you predict axes and explain gradients |
| W1 D2 | foundations/02_training_pipeline — dataset, loop and baseline | One custom Dataset, then TensorDataset/DataLoader; you own train/evaluate |
| W1 D3 | 00_setup plus foundations/02 — timing and recovery | Library timing/checkpoint facilities; you verify resumed state |
| W1 D4 | foundations/03_regularization_diagnostics — four-run comparison | pandas, sklearn preprocessing and plots; you explain/control the experiment |
| W1 D5 | foundations/04_manual_backprop — derivations and parity | Tensor arithmetic for your manual branch; separate autograd reference |
| W1 D6 | Rest | No notebook work |
| W1 D7 | foundations/03_regularization_diagnostics — injected bugs | Plot/log with libraries; you diagnose and document causes |
| W1 D8 | vision/01_mnist — independent MLP | torchvision data; you compose/train/check the model |
| W2 D1 | vision/02_convolution_shapes — small manual convolution | NumPy loops for the mechanism; library image I/O and separate Conv2d reference |
| W2 D2 | vision/01_mnist — CNN comparison; vision/02 — tensor workshop | Standard layers; you compose the CNN and verify axis semantics |
| W2 D3 | vision/03_cifar_architectures — pipeline and augmentation | torchvision, sklearn metrics and pandas; you control the comparison |
| W2 D4 | vision/03_cifar_architectures — AlexNet-inspired model | Library layers; you implement the bounded adapted architecture |
| W2 D5 | vision/03_cifar_architectures — VGG blocks | Library layers; you explain receptive fields and comparison limits |
| W2 D6 | vision/04_resnet20 — architecture, checks and budgeted run | Library primitives; you implement CIFAR residual structure and record recipe |
| W2 D7 | Rest | No notebook work |
| W3 D1 | sequences/01_cells_copy_task — RNN | Your cell equations; library reference and autograd |
| W3 D2 | sequences/01_cells_copy_task — copy-task generator and recall mask | Vectorized data generation; you define/test task and metrics |
| W3 D3 | sequences/01_cells_copy_task — LSTM | Your gates/state update; reference parity and paper interpretation |
| W3 D4 | Rest | No notebook work |
| W3 D5 | sequences/01_cells_copy_task — GRU and matched comparison | Your GRU convention; pandas/plots for evidence |
| W3 D6 | sequences/02_text_pipeline_training — text windows and language model | nn.LSTM for practical training; you own shifts, splits, loss and state policy |
| W3 D7 | sequences/02 — retrieval/rebuild section; sequences/01 — checks | Independently rebuild tiny windows in a fresh cell; inspect and repair evidence |
| W4 D1 | sequences/03_tuning_evaluation — frozen search plan and run table | Your config loop, pandas results; optional declared ResNet continuation stays in vision/04 |
| W4 D2 | vision/05_evaluate and sequences/03 — final evaluation | Standard metrics/plots; you freeze decisions, weight losses and analyze failures |
| W4 D3 | vision/06_predict_images and sequences/04_generate | Library decoding/transforms; you reconstruct models and implement text generation |
| W4 D4 | report.ipynb — two case studies | Read saved result tables, regenerate figures, write supported claims |
| W4 D5 | Relevant smoke/evaluation/inference notebooks plus README/manifest/handoff | Clean-kernel execution, saved artifact checks and gap decisions |
| W4 D6 | Rest | No notebook work |

For multi-day notebooks, select the day's small experiment using explicit configuration and named functions. Run All in smoke mode exercises the required definitions/checks cheaply; it must not silently launch every historical training run. Keep full-run outputs in results/ with distinct run IDs so a later smoke check cannot overwrite reported metrics. Inference notebooks default to loading a selected saved checkpoint; report.ipynb defaults to analyzing saved results.

**Minimum run-table columns:** run_id, notebook_revision, execution_mode, task, dataset/split_id, seed, model/config_id, changed_variable, budget, completed_updates/epochs, train_metric, validation_metric, elapsed_time, checkpoint_path, status and notes. Add metric-specific fields rather than squeezing unrelated quantities into one unlabeled score. Store final test metrics in a clearly labeled final-evaluation table after selection is frozen.

**Minimum README:** environment/kernel and working versions; project-root setup; notebook order; what each execution mode does; exact steps to run a smoke check or load inference; location of datasets/checkpoints/configs; known limitations; and which results were reproduced versus only loaded for analysis.

**Minimum notebook completion record:** date, notebook revision, mode, run IDs, numerical checks passed/failed, clean-kernel result, conclusion and open issue. Mark a required notebook complete only when its core checks pass. Expensive training need not be repeated merely to refresh a presentation cell.

Large data/checkpoints can live elsewhere with verified locations in the manifest. Save a checkpoint together with the information needed to interpret it. No required deliverable in this plan is a .py script; existing earlier-month files can remain as references without conversion.

## 7. Primary learning resources

Use these as references, not an extra reading syllabus. Read the relevant sections for the assigned day. Documentation can change; consult the version matching your installed environment.

| Resource | Use it for |
|---|---|
| [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Tensors, autograd, datasets, modules and optimization |
| [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) | Graphs, gradients and gradient modes |
| [Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | Practical diagnostic habits and incremental experiments |
| [Karpathy: Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html) | Relevant makemore activation and Backprop Ninja lessons |
| [BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) | Statistics, modes and supported shapes |
| [Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html) | Training scaling and evaluation behaviour |
| [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | Logits, class axis, target formats and reductions |
| [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) | Asynchronous execution and correct timing |
| [Saving/loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | Checkpoint recovery and inference |
| [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | Seeds, deterministic settings and practical limits |
| [Evaluation and data splitting](https://scikit-learn.org/stable/modules/cross_validation.html) | Validation selection and test-set isolation |
| [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) | Selected original architecture ideas |
| [ResNet paper](https://arxiv.org/pdf/1512.03385) | CIFAR architecture and reference training protocol |
| [Original LSTM paper, author-hosted copy](https://people.idsia.ch/~juergen/lstm1997-2024head.pdf) | The 1997 constant-error-carousel argument |
| [Learning to Forget](https://sferics.idsia.ch/pub/juergen/FgGates-NC.pdf) | Historical introduction of the forget gate; optional context |
| [LSTM reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) | Modern equations, shapes and dropout semantics |
| [GRU reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) | Gate conventions and the reset-gate implementation difference |
| [Tiny Shakespeare data](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) | Reproducible text source |
| [Colab FAQ](https://research.google.com/colaboratory/faq.html) | Resource availability and session constraints |

**Final completion standard:** you can independently run, check, explain, save and reuse what you built; your recorded claims match your evidence; and you know which unresolved items are correctness problems versus optional improvements. Proceed to Month 4 on that basis.
