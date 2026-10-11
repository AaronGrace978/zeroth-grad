# Zeroth-Grad: Compiling Checkpoint Edits in Weight Space

**Aaron Grace** · October 2026 · Code, data, and figures: [github.com/AaronGrace978/zeroth-grad](https://github.com/AaronGrace978/zeroth-grad)

*Technical report, draft 1. The implementation and experiments were developed with AI coding assistants. Every number below is produced by the commands in Section 9 and lands in `runs/metrics.json` or `runs/digits_metrics.json`.*

---

## Abstract

Editing a trained network usually means training it again. This report describes a *weight-space compiler*: a map from one checkpoint to another that runs as a single forward pass and never applies an optimizer to the network being edited. The compiler reads the weights of a small multilayer perceptron as a set of hidden-unit tokens, processes them with a permutation-equivariant encoder, and predicts residual corrections to three geometric edits. **Inject** installs a new skill on a chosen output head. **Compress** removes duplicated hidden units. **Merge** combines checkpoints trained on different skills into one network of the same width.

Two components make merging work: a per-head readout-ownership gate, and a Hungarian alignment cost that measures interference between hidden units instead of geometric distance. On a synthetic zoo of halfspace detectors, merging is lossless up to the source models' accuracy (0.984 against a ceiling of 0.985), where uniform averaging after alignment reaches 0.887. Six single-skill checkpoints fold into one 32-wide network at 0.977. On scikit-learn's 8×8 handwritten digits, ten one-vs-rest detectors merge without training into one 64-wide network. Read as a 10-way classifier, it scores 0.971 on held-out images: exactly the accuracy of the ten detectors run separately, and about 0.001 below a same-size network trained on all ten labels. The governing constraint is capacity. A merge is lossless when the parents' live hidden units fit in the width, and a group-sparsity penalty during detector training cuts the units needed from about 123 to about 45.

---

## 1. Introduction

A trained checkpoint is usually treated as the output of an optimizer: to change what it computes, you run the optimizer again. That is expensive, it needs data for the new behavior, and it can damage behavior you wanted to keep. This project asks a different question. Can a network learn to *edit other networks' weights directly*, in one pass, the way a compiler rewrites a program?

Three properties make the question tractable for small MLPs:

1. **Hidden units are a set.** Permuting the hidden units of a layer, together with the matching rows and columns of the adjacent weight matrices, leaves the function unchanged. An editor should be equivariant to these permutations, so that relabeling a checkpoint relabels the edit and changes nothing else.
2. **Many edits have a geometric core.** Duplicated units can be collapsed exactly, a linear feature can be written onto unused units, and two networks' units can be matched before mixing. A learned model only needs to predict a correction on top.
3. **The loss is functional.** The compiler is trained by evaluating the *edited* network on data and backpropagating into the compiler's parameters. The checkpoint tensors stay constants.

### Contributions

- A permutation-equivariant compiler for two-hidden-layer MLPs that performs inject, compress, and merge in one forward pass, trained with a functional loss on a zoo of checkpoints (Section 4).
- **Per-head readout ownership** for merging: the rule "keep a connection only when both endpoints come from the same parent", extended to the output layer through a gate with a learned temperature. On the synthetic zoo it closes the entire gap between the previous stitch (0.874) and the source ceiling (0.985) (Section 4.4).
- An **interference-based alignment cost** that sends live units to free slots before letting any two collide. On real images the previous, geometric cost collapsed ten detectors into about ten slots (0.401 accuracy); the interference cost reaches 0.971 (Section 4.4, Section 5.6).
- A sufficient condition under which a merge exactly reproduces each parent's heads, and experiments that trace accuracy against that capacity condition on synthetic and real data (Sections 4.5 and 5).

---

## 2. Background and related work

**Permutation symmetry and alignment.** Entezari et al. (2022) conjectured that, once permutation symmetry is accounted for, independently trained networks lie in one linearly connected basin. Git Re-Basin (Ainsworth et al., 2023) aligns hidden units by weight or activation matching before averaging. Optimal-transport fusion (Singh & Jaggi, 2020) uses soft alignments. All three use the Hungarian method (Kuhn, 1955) or a relaxation of it. This project reuses Hungarian matching, but changes what the cost measures (Section 4.4).

**Merging without training.** Model soups (Wortsman et al., 2022) average fine-tuned checkpoints of one task. Fisher-weighted averaging (Matena & Raffel, 2022), task arithmetic (Ilharco et al., 2023), and TIES-merging (Yadav et al., 2023) combine task-specific updates. ZipIt! (Stoica et al., 2024) merges models trained on *different* tasks without retraining, allowing features to be shared within one model; it is the closest neighbor to the merge studied here. Those methods operate on large networks. This report studies small networks, but it makes the merge exact under a stated condition and measures where that condition breaks.

**Learning on weights.** HyperNetworks (Ha et al., 2017) generate weights from another network. Deep weight-space networks (Navon et al., 2023) and neural functionals (Zhou et al., 2023a; 2023b) build permutation-equivariant layers over a network's weights for tasks such as predicting generalization or editing implicit neural representations. The compiler here belongs to that family. Its output is a full edited checkpoint, and its training signal is the edited network's own loss.

**Function-preserving surgery and editing.** Net2Net (Chen et al., 2016) widens a network by duplicating units and splitting their outgoing weights, which leaves the function unchanged. The compress edit inverts that operation. Knowledge-editing methods such as ROME (Meng et al., 2022) write a targeted rank-one change into a large model. The inject edit is a rank-one write too, onto unused units of a small one.

---

## 3. Setting

A checkpoint is a two-hidden-layer MLP

$$
f(x) = W_3\,\sigma\!\big(W_2\,\sigma(W_1 x + b_1) + b_2\big) + b_3,\qquad \sigma = \mathrm{ReLU},
$$

with $W_1 \in \mathbb{R}^{H_1\times d}$, $W_2 \in \mathbb{R}^{H_2\times H_1}$, and $W_3 \in \mathbb{R}^{K\times H_2}$. Each of the $K$ outputs is the logit of one binary skill, so several skills can be true of one input, and a checkpoint can hold several skills at once.

A hidden-1 unit $i$ is **live** if its outgoing column $W_2[:,i]$ is nonzero, and **dead** otherwise. A hidden-2 unit $j$ is live if $W_3[:,j]$ is nonzero. A dead unit cannot affect $f$. **Spare** units are dead units pinned at zero during training, so that later edits have room.

For permutations $P_1$ and $P_2$, the map

$$
(W_1, b_1, W_2, b_2, W_3) \mapsto (P_1W_1,\; P_1b_1,\; P_2W_2P_1^\top,\; P_2b_2,\; W_3P_2^\top)
$$

leaves $f$ unchanged. Every component below is either equivariant or invariant to it.

---

## 4. Method

### 4.1 The equivariant encoder

Each hidden-1 unit $i$ becomes a token from its incoming row and bias, $t^{(1)}_i = \mathrm{Linear}([W_1[i,:], b_1[i]])$. Each hidden-2 unit $j$ becomes a token from its bias and readout column, $t^{(2)}_j = \mathrm{Linear}([b_2[j], W_3[:,j]])$. Two rounds of messages pass through $W_2$:

$$
m^{(1)} = W_2^\top t^{(2)},\qquad m^{(2)} = W_2\, t^{(1)},\qquad t^{(\ell)} \leftarrow \mathrm{Linear}([t^{(\ell)}, m^{(\ell)}]).
$$

Both sides of a round read the previous tokens. A permutation of hidden-1 units therefore permutes $t^{(1)}$ and leaves $t^{(2)}$ unchanged, and the same holds the other way round. A task embedding, the target feature direction, and the target head condition both layers through FiLM (Perez et al., 2018). Each layer then runs a small Transformer encoder without positional encodings, followed by cross-attention between the two layers. The compiler has about 205,000 parameters ($d_{\text{model}} = 64$, 4 heads, depth 2). Every residual head is zero-initialized, so an untrained compiler applies the geometric edit unchanged.

### 4.2 Inject

Given a feature direction $\phi$ and a target head $k$, inject writes a rank-one circuit onto spare units:

$$
\Delta W_1 = a\,\phi^\top,\qquad \Delta W_2 = c\,s^\top,\qquad W_3[k,:] = r.
$$

The per-unit scales $a$, $s$, $c$, $r$ are a learned global magnitude times a per-unit modulation $1 + \tfrac12\tanh(\cdot)$ predicted from the tokens, masked to spare units. With $s$ and $c$ normalized by the number of spares, the installed margin does not depend on how many spares share the write. Other heads are untouched, so existing skills are preserved exactly. $\phi$ is either a known direction or the Fisher direction $\mathrm{normalize}(\mathbb{E}[x\mid y{=}1] - \mathbb{E}[x\mid y{=}0])$ estimated from labeled examples.

### 4.3 Compress

Units whose incoming rows (with bias) have cosine at least 0.999 form a cluster. The compiler keeps up to a budget of clusters, ranked by a learned score plus a detached duplicate-count prior. Each kept cluster keeps one survivor, and the survivor's outgoing column is replaced by the sum of the cluster's outgoing columns. On exact duplicates this inverts Net2Net widening exactly. A straight-through softmax over cluster members carries gradient to the learned score.

### 4.4 Merge

**Alignment by interference.** Before mixing, the hidden units of checkpoint $B$ are permuted to line up with those of $A$, one layer at a time, by the Hungarian method. With incoming features $u_i = [W_1[i,:], b_1[i]]$, incoming cosine $\cos_{ij}$, and outgoing norms $o^A_i$ and $o^B_j$, the cost of putting $A$'s unit $i$ and $B$'s unit $j$ in one slot is

$$
C_{ij} = o^A_i\, o^B_j\,(1-\cos_{ij}) \;+\; 10^{-3}\, G_{ij},
$$

where $G_{ij}$ is magnitude-weighted cosine distance, which only breaks ties. The first term is zero when either unit is dead, and when the two units compute the same direction. The earlier cost was $G_{ij} + 5\,o^A_io^B_j(1-\cos_{ij})$. Under that cost, pairing a live unit with an empty slot still costs about half the live unit's norm, so two live units with incoming cosine above roughly 0.2 were cheaper to stack than to separate. In 16 input dimensions that rarely happens. In 64 dimensions, with a shared initialization, cosines of 0.3 to 0.5 are routine, and the old cost stacked units that should have been kept apart (Section 5.6).

**Node gates and the same-parent rule.** Each aligned slot gets a gate $\lambda_1$ (hidden-1) or $\lambda_2$ (hidden-2), the weight on $A$. Its prior is the outgoing-norm ratio $o^A/(o^A + o^B)$, plus a learned residual in logit space. Incoming weights and biases mix by the gate. A connection is kept only to the extent that both of its endpoints come from the same parent:

$$
W_2 = (\lambda_2\lambda_1^\top)\odot W_2^A + \big((1-\lambda_2)(1-\lambda_1)^\top\big)\odot W_2^B .
$$

**Per-head readout ownership.** Output heads have a parent too. Head $k$ gets an ownership gate

$$
\lambda_3[k] = \mathrm{sigmoid}\!\Big(\tau \log\frac{\lVert W_3^A[k,:]\rVert}{\lVert W_3^B[k,:]\rVert} + \rho_k\Big),
$$

and the same-parent rule extends to the readout:

$$
W_3 = (\lambda_3\lambda_2^\top)\odot W_3^A + \big((1-\lambda_3)(1-\lambda_2)^\top\big)\odot W_3^B,\qquad b_3 = \lambda_3 b_3^A + (1-\lambda_3) b_3^B .
$$

$\tau$ is a learned temperature, initialized at 4, so the prior equals the readout-norm ratio raised to the fourth power. The residual $\rho_k$ comes from a zero-initialized linear head on per-head tokens: hidden-2 tokens pooled along each head's normalized readout row, which makes them invariant to hidden-2 order. Without this gate, every merged head reads every hidden unit. A skill head then picks up the other parent's untrained readout weights, which stay near their initialization values but multiply that parent's trained, large activations.

**Folding.** `merge_many` merges any number of checkpoints left to right, $\mathrm{merge}(\mathrm{merge}(c_0, c_1), c_2)$, and so on.

### 4.5 When a merge is exact

**Proposition.** Let $A$ and $B$ be checkpoints of the same shape whose dead units have zero outgoing weights. Suppose that after alignment the live hidden-1 sets $S^A_1$ and $S^B_1$ are disjoint, and so are the live hidden-2 sets $S^A_2$ and $S^B_2$. Take hard gates: $\lambda_1 = 1$ on $S^A_1$ and $0$ elsewhere, $\lambda_2 = 1$ on $S^A_2$ and $0$ elsewhere, and $\lambda_3[k] = 1$ exactly for the heads owned by $A$. Then every head owned by $A$ in the merged network computes $A$'s logit exactly, and every other head computes $B$'s.

*Proof sketch.* For $j \in S^A_2$ and any hidden-1 slot $i$, the merged entry $W_2[j,i]$ equals $W^A_2[j,i]$ if $i \in S^A_1$ and $0$ otherwise. On dead $i$, $W^A_2[j,i] = 0$ anyway. So every hidden-2 unit in $S^A_2$ computes exactly what it computes in $A$. For a head $k$ owned by $A$, $W_3[k,j] = W^A_3[k,j]$ on $S^A_2$ and $0$ elsewhere, which is $A$'s readout, because $A$'s dead hidden-2 columns are zero; $b_3[k] = b_3^A[k]$. The argument for $B$ is symmetric. $\square$

Two consequences follow. First, an exact merge exists only when the live units fit: $|S^A_\ell| + |S^B_\ell| \le H_\ell$ in both layers, or, for a fold of $n$ checkpoints, $\sum_m |S^{(m)}_\ell| \le H_\ell$. Second, the interference term of the alignment cost is zero exactly on collision-free assignments, so the Hungarian method finds one whenever capacity allows, unless the $10^{-3}$ tie-break term overrides it for units with very small outgoing norms. The norm-ratio priors give gates within $10^{-8}$ of 0 or 1 on live–dead pairs. Ownership is soft, so the merged heads track their parents closely but not to machine precision. Under these conditions, a merge packs the ensemble of its parents into one set of weights.

### 4.6 Training the compiler

Each step samples one edit: inject (50%), compress (25%), or merge (25%). The edit is applied to a training checkpoint, and the edited network's binary cross-entropy on 256 fresh inputs is backpropagated into the compiler.

- **Inject** is scored on the new skill and the source skill.
- **Compress** runs on a network in which every live unit has been split into four equal copies and shuffled. It is scored on the source skill.
- **Merge** pairs two checkpoints with different skills and is scored on both skills.

A small penalty on each residual head keeps the geometric prior as the default. The optimizer is Adam: learning rate $10^{-3}$ for the network and $2\times10^{-2}$ for global scales, including $\tau$, with gradient-norm clipping at 5. Training takes 400 steps and about 7 seconds on 4 CPU cores.

---

## 5. Experiments

### 5.1 Synthetic zoo

Inputs are $x \sim \mathcal{N}(0, I_{16})$. Eight orthonormal directions $v_k$ define skills $y = \mathbf{1}[v_k^\top x > 0]$. Each checkpoint is a $16 \to 32 \to 32 \to 8$ MLP (1,864 parameters), trained from one shared initialization with AdamW for 300 steps. Four units per layer are pinned at zero as spares. Afterward, any unit whose removal costs at most 0.01 accuracy is zeroed (carved). Skills 0–5 have checkpoints; skills 6 and 7 are held out as inject targets. Seeds 0–4 of each skill are training checkpoints, and seeds 5–6 are held out. A carved single-skill checkpoint keeps 4–9 live hidden-1 units and 1–7 live hidden-2 units. Accuracy is sign agreement on 2,500–3,000 fresh samples. Every result below is on held-out checkpoints, after training the compiler with seed 0.

![Held-out results on the synthetic zoo](heldout_results.png)

### 5.2 Inject

24 edits: 12 held-out checkpoints, each injected with skills 6 and 7.

| | new skill | old skill |
| --- | ---: | ---: |
| checkpoint before the edit | 0.503 | 0.983 |
| untrained compiler (geometric prior) | 0.807 | 0.983 |
| trained compiler | **0.997** | 0.983 |
| 200 steps of Adam on the target, both skills (oracle) | 0.902 | — |

Inject is where training matters most: the learned scales and biases take the rank-one write from 0.807 to 0.997. The old skill is unchanged to three decimals, because inject never touches its head.

### 5.3 Compress

12 held-out checkpoints; every live unit is split into four copies, and the units are shuffled.

| | accuracy |
| --- | ---: |
| original checkpoint | 0.984 |
| compiler | 0.984 |
| closed-form cluster collapse | 0.984 |
| magnitude pruning to the same width | 0.643 |

The compiler equals the closed-form collapse: on exact duplicates the geometry already solves the problem, and nothing is left to learn. Magnitude pruning keeps the largest fragments of a few large units and drops entire smaller ones.

### 5.4 Merging two checkpoints

30 held-out pairs with distinct skills that share a seed; accuracy is the mean over the two skills.

| | accuracy |
| --- | ---: |
| each source checkpoint on its own skill (ceiling) | 0.985 |
| compiler | **0.984** |
| untrained compiler | 0.984 |
| norm-gate stitch without head ownership | 0.874 |
| uniform average, no alignment | 0.846 |
| uniform average after alignment | 0.887 |
| compiler, partner's hidden units permuted | 0.984 |
| uniform average, partner's hidden units permuted | 0.861 |

Alignment places the two parents' live units in different slots in all 30 pairs, so the whole loss of the stitch without ownership is readout crosstalk. Permuting the partner's hidden units does not change the compiler's output.

**Ownership temperature.** Fixing $\tau$ and removing the residual isolates the role of sharpness. The four rows below use the same alignment and node gates; only the head gate changes.

| head gate | mean accuracy | worst pair |
| --- | ---: | ---: |
| readout-norm ratio ($\tau = 1$) | 0.950 | 0.834 |
| $\tau = 2$ | 0.977 | 0.893 |
| $\tau = 4$ (initialization) | 0.984 | 0.977 |
| hard ownership (oracle) | 0.985 | 0.980 |

Training moves $\tau$ from 4.0 to about 5.2.

### 5.5 Merging k checkpoints

Every k-subset of distinct held-out skills sharing a seed is folded into one 32-wide network; accuracy is the mean over the k skills.

| k | subsets | compiler | untrained compiler | aligned uniform soup | ceiling |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 30 | 0.984 | 0.984 | 0.887 | 0.985 |
| 3 | 40 | 0.984 | 0.983 | 0.660 | 0.985 |
| 4 | 30 | 0.984 | 0.982 | 0.591 | 0.985 |
| 5 | 12 | 0.984 | 0.979 | 0.565 | 0.985 |
| 6 | 2 | 0.977 | 0.968 | 0.549 | 0.985 |

At $k = 6$ the parents need 35 and 37 live hidden-1 units (one 6-subset per held-out seed), more than the 32 slots, so collisions are forced. A few 5-subsets already need 32 or 33. That is where the learned temperature separates the compiler from its initialization. Compiler training seeds 0, 1, and 2 give 0.984 on pairs and 0.977–0.978 at $k = 6$. Only two 6-subsets exist, so the $k = 6$ row is a small sample.

### 5.6 Real data: ten digit detectors, one classifier

**Setup.** scikit-learn's 8×8 digits (Pedregosa et al., 2011; from the UCI optical-digits collection) has 1,797 images of 64 pixels. They are shuffled with a fixed seed and split into 1,200 training and 597 test images, then standardized with training statistics. Each checkpoint is a $64 \to 64 \to 64 \to 10$ MLP (8,970 parameters) trained from one shared initialization to detect one digit on its own logit. Training uses balanced one-vs-rest BCE for 400 steps of AdamW, plus a group-sparsity penalty (Yuan & Lin, 2006) of $3\times10^{-3}$ times the sum of hidden-1 outgoing column norms and the absolute readout into the detector's head. The checkpoint is then carved at a 0.005 drop in training balanced accuracy. Three detector seeds give three sets of ten. The merge uses the untrained compiler: alignment, norm gates, and readout-norm ownership at $\tau = 4$. The halfspace compiler is sized for 16 inputs, and the digits results need no learned residual. The merged network is read as a 10-way classifier, taking the argmax over heads.

![Digits results](digits_results.png)

**Ten-way accuracy**, mean over 3 seeds (per seed in parentheses):

| | 10-way test accuracy |
| --- | ---: |
| ten detectors merged into one 64-wide network | **0.971** (0.972, 0.970, 0.972) |
| ensemble of the ten separate detectors (ceiling) | 0.971 (0.972, 0.970, 0.972) |
| one 64-wide network trained on all ten labels | 0.972 (0.975, 0.972, 0.970) |
| merge with the legacy alignment cost | 0.401 (0.353, 0.370, 0.481) |
| aligned uniform soup | 0.112 |

The merge equals the ensemble on every seed, and at every subset size from 2 to 10 digits (up to 20 sampled subsets per size and seed). Against the jointly trained reference, it is 0.003 lower, 0.002 lower, and 0.002 higher on the three seeds.

**Capacity.** The same pipeline at other widths:

| width | live hidden-1 units, ten detectors | merged | ensemble |
| ---: | ---: | ---: | ---: |
| 32 | 48 | 0.891 | 0.972 |
| 48 | 43 | 0.970 | 0.971 |
| 64 | 45 | 0.971 | 0.971 |
| 128 | 45 | 0.972 | 0.972 |
| 64, no sparsity penalty | 123 | 0.633 | 0.967 |

Accuracy follows the condition in Section 4.5. Once the live units fit, the merge matches the ensemble; at width 32 they do not, and accuracy falls to 0.891. The sparsity penalty does not hurt the ensemble (0.971 with it, 0.967 without) and makes the detectors about 2.7 times more compact. Without it, the two alignment costs are close (0.633 interference, 0.651 legacy), because collisions are forced under either.

---

## 6. Discussion

**What the learned part contributes.** It contributes most on inject (0.807 → 0.997), where the geometry fixes the direction of the write but not the scale. It contributes nothing measurable on compress, where the closed form is exact. On merge it contributes a little at capacity, and more once collisions begin ($k = 6$: 0.968 → 0.977). Most of the merge result comes from structure: alignment, the same-parent rule, and ownership. On the evidence here, the learned compiler is useful mainly where the geometry leaves a continuous choice open (scales, biases, or which units collide), and should be judged there.

**Why merging works here.** Two facts do the work. The checkpoints are sparse, with a handful of live units each, and they share an initialization. Sparsity creates room, the interference cost uses that room, and ownership keeps the readouts apart. Under those conditions a merge is a packing problem, and Section 4.5 says when the packing is exact. Feature sharing, where two parents read the same unit at full strength, is not exploited yet. Section 4.4's cost lets identical units share a slot for free, but the same-parent rule halves what each parent reads from a shared slot, so shared units are not yet a capacity gain.

**Merging versus ensembling.** At capacity, the merged network *is* the ensemble, computed in one forward pass with one set of weights the size of a single parent. The cost is width; the saving is everything else: one checkpoint, one forward pass, and no routing logic. It is also the reason a merge can match a jointly trained network: given enough room, ten good detectors make a good classifier.

---

## 7. Limitations

- **Scale.** The networks have two hidden layers and at most 128 units, and the inputs have 16 or 64 dimensions. Nothing here shows that the approach survives deep networks, convolutions, or attention.
- **Shared initialization and sparsity.** Every zoo is trained from one shared initialization, and the merge results depend on sparse parents. Dense or independently initialized checkpoints are untested, apart from the no-sparsity ablation, where the merge degrades to 0.633.
- **The ownership assumption.** Ownership is inferred from readout norms. If one parent's untrained readout row were as large as the other parent's trained row for the same head, the gate would settle at 0.5. A unit test pins that behavior down.
- **Disjoint skills.** Merges combine checkpoints with different output heads. Merging checkpoints of the *same* task, where both parents own a head, is the soups setting and is not studied here.
- **Statistics.** Results are means over held-out checkpoints, with three compiler or detector seeds for the headline numbers. No confidence intervals are reported, and the $k = 6$ synthetic row rests on two subsets.
- **No trained compiler for digits.** The digits merge uses the untrained compiler; inject and compress on real data are not evaluated.

---

## 8. Future work

The roadmap in [`ROADMAP.md`](../ROADMAP.md) sets out the next steps in order, each with a measurable target. The central questions:

1. **Over capacity.** Can a learned compiler choose collisions, or share features across parents, well enough to close the gap at width 32 on digits (0.891 against 0.972)?
2. **Depth and architecture.** Extend the encoder, the alignment, and the same-parent rule to $L$ layers, to residual MLPs, and to convolutional channels.
3. **Independent initializations.** Measure how much of the result survives without a shared initialization, where Git Re-Basin-style alignment becomes necessary rather than convenient.
4. **Real-data inject.** Train a compiler on a digits zoo, and test whether a skill installed from a few labeled examples matches fine-tuning.

---

## 9. Reproducibility

All runs were on CPU (4 cores) with Python 3.12 and PyTorch 2.14 (CPU build). Wall-clock times are approximate.

```bash
pip install -e ".[dev,digits,plot]"

python -m zeroth_grad populate --out checkpoints          # ~11 s, synthetic zoo
python -m zeroth_grad train --checkpoints checkpoints \
    --out runs/compiler.pt --steps 400                     # ~7 s
python -m zeroth_grad eval --compiler runs/compiler.pt \
    --checkpoints checkpoints                              # ~8 s  -> runs/metrics.json

python -m zeroth_grad digits-populate --out digits         # ~15 s, 3 seeds x 10 detectors
python -m zeroth_grad digits-eval --checkpoints digits     # ~18 s -> runs/digits_metrics.json
python -m zeroth_grad digits-populate --out digits_w32 --hidden 32   # capacity sweep
python -m zeroth_grad digits-populate --out digits_dense --sparsity 0

python scripts/plot_results.py                             # redraws both figures
pytest                                                     # 26 tests
```

The width and sparsity rows of the capacity table come from running `digits-populate` and then `digits-eval --checkpoints <dir>` on each directory. Section 5.4's ownership-temperature ablation passes `mix_checkpoints` a fixed head gate, `sigmoid(tau * head_owner_logit(A, B))`, for each $\tau$.

---

## References

- Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2023). Git Re-Basin: Merging Models modulo Permutation Symmetries. *ICLR*.
- Chen, T., Goodfellow, I., & Shlens, J. (2016). Net2Net: Accelerating Learning via Knowledge Transfer. *ICLR*.
- Entezari, R., Sedghi, H., Saukh, O., & Neyshabur, B. (2022). The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks. *ICLR*.
- Ha, D., Dai, A., & Le, Q. V. (2017). HyperNetworks. *ICLR*.
- Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., & Farhadi, A. (2023). Editing Models with Task Arithmetic. *ICLR*.
- Kuhn, H. W. (1955). The Hungarian Method for the Assignment Problem. *Naval Research Logistics Quarterly*, 2(1–2), 83–97.
- Matena, M., & Raffel, C. (2022). Merging Models with Fisher-Weighted Averaging. *NeurIPS*.
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS*.
- Navon, A., Shamsian, A., Achituve, I., Fetaya, E., Chechik, G., & Maron, H. (2023). Equivariant Architectures for Learning in Deep Weight Spaces. *ICML*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI*.
- Singh, S. P., & Jaggi, M. (2020). Model Fusion via Optimal Transport. *NeurIPS*.
- Stoica, G., Bolya, D., Bjorner, J., Ramesh, P., Hearn, T., & Hoffman, J. (2024). ZipIt! Merging Models from Different Tasks without Training. *ICLR*.
- Wortsman, M., et al. (2022). Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy without Increasing Inference Time. *ICML*.
- Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023). TIES-Merging: Resolving Interference When Merging Models. *NeurIPS*.
- Yuan, M., & Lin, Y. (2006). Model Selection and Estimation in Regression with Grouped Variables. *Journal of the Royal Statistical Society: Series B*, 68(1), 49–67.
- Zhou, A., Yang, K., Burns, K., Cardace, A., Jiang, Y., Sokota, S., Kolter, J. Z., & Finn, C. (2023a). Permutation Equivariant Neural Functionals. *NeurIPS*.
- Zhou, A., Yang, K., Jiang, Y., Burns, K., Xu, W., Sokota, S., Kolter, J. Z., & Finn, C. (2023b). Neural Functional Transformers. *NeurIPS*.
