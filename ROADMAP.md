# Roadmap

Where zeroth-grad stands, and what comes next. Every milestone has a target you can check by running a command. The method and current results are in the [white paper](docs/whitepaper.md).

## Where it stands

| Edit | Synthetic zoo (held-out) | Real data (8×8 digits) |
| --- | --- | --- |
| Inject | 0.997 on an unseen skill; old skill unchanged. Rank-k: 0.935–0.980 on slab, wedge, and XOR skills | not yet evaluated |
| Compress | 0.984, equal to the closed-form collapse | not yet evaluated |
| Merge, 2 checkpoints | 0.984 against a 0.985 ceiling | 0.994 (2-way) |
| Merge, many checkpoints | 0.977 at 6 skills into width 32 | 0.971 at 10 digits into width 64, equal to the ensemble |
| Merge over capacity | — | widened: equal to the ensemble (0.972 sparse, 0.967 dense); folded at fixed width: 0.950 at width 32, 0.945 dense at width 64 |
| Transformers (2 layers, geometric) | a hidden symmetry is undone exactly; averaging then matches the unmoved merge | — |

What is established: when the parents' live units fit in the width, merging is exact (white paper, Section 4.5). Past capacity, widening restores exactness, and folding trades a little accuracy for width. Sinkhorn alignment scales to width 8,192 within 0.12% of the optimal cost. The core pipeline reproduces in under a minute on a laptop CPU.

What is not: MLP depth beyond two hidden layers, convolutions, independently initialized parents (transformers from independent initializations still show a barrier after alignment), a learned compiler for transformers, and a learned compiler that matters for merging at capacity.

## Milestones

### 1. Rigor

Make every number in the README something CI can catch regressing.

- [ ] Report means with 95% intervals across at least 5 compiler seeds and 5 detector seeds.
- [ ] A `--quick` mode for `eval` and `digits-eval` that finishes in under 30 seconds, plus a CI job that fails when a headline metric drops by more than 0.01.
- [ ] Property tests for every edit: equivariance under random hidden permutations, no gradient into checkpoint tensors, and exact function preservation where the white paper claims it.
- [ ] An exactness check for Section 4.5: on collision-free merges, measure the maximum logit difference between each merged head and its parent.

**Done when:** README tables carry intervals, and a deliberately broken alignment cost fails CI.

### 2. Over capacity: make the learned part matter

Today the learned compiler adds little at capacity. Its job should be the cases where geometry leaves a choice.

- [x] **Shared-slot reading.** Positively parallel units (cosine at least 0.999) share one slot, and both parents read it at full strength (`share_slots`).
- [x] **Compress-then-merge.** Activation folding (`compile fold`) refits each parent onto fewer units before the merge.
- [x] **Widening.** `--expand` adds dead units on collisions until the merge is collision-free, and so exact.
- [x] **Declared ownership.** `--declare-heads` replaces readout-norm ownership, which fails for dense parents.
- [ ] **Learned collision choice.** Train the compiler's merge heads on deliberately over-capacity folds, where the geometric prior is known to fall short.
- [ ] Train a digits-sized compiler and compare it with the untrained prior.

**Done when:** the width-32 ten-digit merge rises from 0.891 to at least 0.95, against a 0.972 ensemble. *Met by geometry, not by learning:* folding at 0.98 agreement gives 0.950 at width 32, and widening gives 0.972 at width 48. The learned items above remain open.

### 3. Bigger real data

- [ ] MNIST and Fashion-MNIST at 28×28 (784 inputs), with ten sparse one-vs-rest detectors merged into one classifier.
- [ ] Frozen pretrained embeddings (for example CIFAR-10 features from a public encoder) as inputs to small MLP heads, so the merge runs on realistic features without training a large model.
- [ ] Real-data inject: install a held-out digit from 50 labeled examples, and compare against fine-tuning on the same examples. The rank-k inject (`compile inject --rank k --examples`) is the candidate method.

**Done when:** the MNIST ten-detector merge is within 0.01 of its ensemble, at a width no larger than the detectors' total live units plus 25%.

### 4. Depth and architecture

The encoder, the alignment, and the same-parent rule are written for exactly two hidden layers.

- [ ] Generalize to $L$ hidden layers. Tokens per layer, messages through each $W_\ell$, layerwise alignment, and the same-parent rule on every matrix.
- [ ] Residual MLPs, where the residual stream ties the permutation across layers.
- [ ] Convolutional channels as tokens, for small convnets on digits and MNIST.
- [x] Transformer symmetries: residual permutation, scale, and (RMSNorm) rotation; head permutation; per-head Q/K and V/O orthogonal maps; MLP permutation (`transformer.py`).
- [x] Transformer weight matching over those symmetries, then averaging or task arithmetic (`transformer_align.py`).
- [ ] Same-parent gating for attention heads and transformer MLP units, and a compiler trained on a transformer zoo.

**Done when:** a 4-hidden-layer MLP zoo reaches the same lossless-at-capacity merge as the two-layer one, and a small convnet merge beats aligned averaging.

### 5. Independent initializations

Every zoo so far shares one initialization.

- [ ] Train detectors from different initializations and measure the merge with interference alignment, legacy alignment, and activation-based matching in the style of Git Re-Basin.
- [ ] Close the transformer barrier: midpoints of independently trained transformers reach 0.34 (LayerNorm) and 0.64 (RMSNorm, with rotations) after alignment, against endpoints at 1.0.
- [ ] Test whether sparsity alone, without a shared initialization, is enough for exact packing. Section 4.5 suggests it should be, since disjoint live sets do not depend on the initialization.

**Done when:** the ten-digit merge from independent initializations is measured and reported, whatever the result.

### 6. Usable by other people

- [ ] A Python API: `zeroth_grad.merge(paths, out=...)`, `zeroth_grad.inject(...)`, and `zeroth_grad.compress(...)`, with docstrings and a short tutorial notebook.
- [ ] Interop with plain PyTorch `state_dict`s and Hugging Face `safetensors` layouts for MLP heads, not only this repo's key names.
- [ ] GPU support and batched evaluation, including sparse or low-rank costs for Sinkhorn alignment past width 8,192.
- [ ] A PyPI release with versioned compiler checkpoints.

**Done when:** someone can merge two of their own MLP heads with `pip install zeroth-grad` and three lines of Python.

## Open questions

- Is a merge at capacity always the ensemble of its parents? Section 4.5 proves this under hard gates. With soft ownership, how far from exact can a merge drift, and does it matter in practice?
- How much width does a family of skills need when the detectors are trained jointly for mergeability, for example with a shared sparsity budget?
- Can feature sharing across parents beat the ensemble on width, by packing ten detectors into fewer units than their live units add up to, without losing accuracy?
- Where does a learned weight-space compiler beat geometry plus a few gradient steps on the edited model? The honest bar is a small fine-tuning budget, not zero.

## Non-goals for now

- Large language models. The transformer path exists to test the symmetries and alignment on small models. The questions above are cheaper to answer on small networks first, and the answers decide whether scaling is worth attempting.
- Claims of state of the art. Results are reported against the strongest simple baselines the repo can run, not against published numbers on other setups.
