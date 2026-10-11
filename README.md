# zeroth-grad

A weight-space compiler. It reads a small MLP checkpoint (`.safetensors` or `.pt`) and writes an edited checkpoint in one forward pass. Compress, inject, and merge never run an optimizer on the network being edited.

The compiler itself is trained. Each step samples a checkpoint from a zoo of models trained on synthetic halfspaces, applies an edit, and backpropagates a functional loss into the compiler only. At apply time the checkpoint tensors are constants.

![A checkpoint goes into the weight compiler, which compresses, injects, or merges it](docs/weight-compiler-diagram.jpg)

The [extensions](#extensions) scale alignment with Sinkhorn optimal transport, widen merges on demand, fold dense parents, inject rank-k non-linear skills, and align and merge small transformers. The method, results, and limitations are written up in the [white paper](docs/whitepaper.md). What comes next is in the [roadmap](ROADMAP.md).

## Edit families

Hidden units are a set. Permuting a layer permutes that layer's tokens and leaves the other layer's tokens in place, because messages between layers are products with the weight matrix (`W @ features`), not a flattened parameter vector.

**Inject.** Install an unseen halfspace on a chosen output head. Spare neurons (outgoing norm zero) are overwritten with a rank-1 circuit

```
ΔW1 = a φᵀ,   ΔW2 = c sᵀ,   ΔW3[head] = r
```

with scales `a`, `s`, `c`, `r` predicted by the compiler. Old heads are left alone, so the skill already in the checkpoint stays put. `φ` is the concept direction, or `normalize(mean(x | y=1) - mean(x | y=0))` estimated from a labeled batch.

**Compress.** Exact duplicate neurons (incoming cosine at least 0.999) are one cluster. The compiler keeps one neuron per cluster, up to the pre-duplication width, and adds the dropped copies' outgoing weights onto the survivor. A learned residual breaks ties; on exact copies the collapse is geometric. Magnitude pruning keeps the largest fragments and drops the rest, which is the baseline this edit beats.

**Merge.** Hidden units of checkpoint B are matched to A with the Hungarian algorithm, or with Sinkhorn optimal transport for wide layers (see [Extensions](#extensions)). The cost of putting two units in one slot is their interference, `|out_A| · |out_B| · (1 − cos)` on incoming weights (row plus bias), with magnitude-weighted cosine distance as a tie breaker. A live unit next to a dead one costs nothing, so detectors spread into free slots before any two collide, and two copies of one direction still share a slot. Per-neuron gates mix the endpoints, and a connection is kept only when both endpoints come from the same parent. A random hidden permutation of B is undone by the match, so the merged function does not depend on that permutation.

Output heads have a parent too. Each head gets an ownership gate

```
λ3[k] = sigmoid(τ · log(|w3_A[k]| / |w3_B[k]|) + residual[k])
```

and the same-parent rule covers the readout: head k reads a hidden unit only when both come from the same parent, and `b3[k]` follows the owner. `τ` is learned (initialized at 4, so the prior is the readout-norm ratio raised to the fourth power), and the residual is read from hidden-2 tokens pooled along each head's readout direction. Without this gate, a skill head also reads the other parent's untrained readout weights, and that crosstalk was the whole merge gap. Ownership assumes a parent's untrained readout rows are smaller than the other parent's trained row for the same head, which holds when untrained rows stay near their initialization scale.

`merge_many` folds any number of checkpoints left to right, `merge(merge(c0, c1), c2)`.

## Models

Inputs are `x ~ N(0, I_16)`. Eight fixed orthonormal directions define the skills `y = 1[v_k · x > 0]`. The MLP is `16 → 32 → 32 → 8`: one logit per skill, so two skills can be true of one checkpoint at the same time. A single shared logit cannot score well on two different halfspaces of the same `x`.

The zoo is trained from one shared initialization. Skills 0–5 are the training skills. Skills 6–7 are held out and are the inject targets. The last two seeds of each training skill are held out as well. Four hidden units in each layer are pinned at zero during student training, and any further neuron whose removal does not change skill accuracy is zeroed afterward. Those spares are where an inject lands.

## Results

Held-out checkpoints, CPU, after `populate --steps 300` and `train --steps 400`. Accuracy is mean sign agreement on 2,500–3,000 fresh samples. The full dump is written to `runs/metrics.json`. `python scripts/plot_results.py` redraws the figure below from that file.

![Held-out accuracies for inject, compress, and merge](docs/heldout_results.png)

**Inject** (24 edits: 12 held-out models × skills 6 and 7):

| | new skill | old skill |
| --- | ---: | ---: |
| checkpoint before the edit | 0.503 | 0.983 |
| untrained compiler (geometric prior only) | 0.807 | 0.983 |
| trained compiler | 0.997 | 0.983 |
| 200-step Adam on the target (oracle) | 0.902 | — |

**Compress** (12 held-out models, every live neuron split into four equal copies, shuffled):

| | skill accuracy |
| --- | ---: |
| original dense checkpoint | 0.984 |
| compiler | 0.984 |
| closed-form cluster collapse | 0.984 |
| magnitude pruning at the same width | 0.643 |

**Merge** (30 held-out pairs, mean accuracy on the two source skills):

| | accuracy |
| --- | ---: |
| each source checkpoint on its own skill (ceiling) | 0.985 |
| compiler | 0.984 |
| untrained compiler (alignment, norm gates, head ownership) | 0.984 |
| norm-gate stitch without head ownership | 0.874 |
| uniform average, no alignment | 0.846 |
| uniform average after alignment | 0.887 |
| compiler, partner hidden units permuted | 0.984 |
| uniform average, partner hidden units permuted | 0.861 |

The merge is lossless up to the ceiling, and permutation does not move it. Without head ownership the same stitch scores 0.874: alignment places the two parents' live units in different slots in all 30 pairs, and the remaining loss is readout crosstalk.

**Merge k checkpoints** (every k-subset of distinct held-out skills sharing a seed, folded into one 32-wide network, mean accuracy on the k skills):

| k | subsets | compiler | untrained compiler | aligned uniform soup | ceiling |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 30 | 0.984 | 0.984 | 0.887 | 0.985 |
| 3 | 40 | 0.984 | 0.983 | 0.660 | 0.985 |
| 4 | 30 | 0.984 | 0.982 | 0.591 | 0.985 |
| 5 | 12 | 0.984 | 0.979 | 0.565 | 0.985 |
| 6 | 2 | 0.977 | 0.968 | 0.549 | 0.985 |

Six single-skill checkpoints fit in one network at the original width. The trained ownership temperature (4.0 to 5.2 over training) is what separates the compiler from its initialization as k grows. Compiler training seeds 0, 1, and 2 give 0.984 on pairs and 0.977–0.978 at k = 6.

## Real data: ten digit detectors, one classifier

The halfspace zoo is synthetic, so `digits-populate` builds a second zoo on scikit-learn's 8×8 handwritten digits (1,797 images, 64 pixels; 1,200 train, 597 test). Each checkpoint is a `64 → 64 → 64 → 10` MLP trained from a shared initialization to detect one digit on its own logit, with balanced one-vs-rest BCE and a group-sparsity penalty on outgoing weights, then carved like the halfspace zoo. `digits-eval` merges the ten detectors of a seed into one network and reads it as a 10-way classifier (argmax over heads). Nothing is trained during the merge, and the compiler is the untrained geometric prior (alignment, norm gates, head ownership), because the halfspace compiler is sized for 16 inputs.

![Ten merged digit detectors match their ensemble and a jointly trained network](docs/digits_results.png)

Mean over 3 detector seeds, 10-way test accuracy:

| | accuracy |
| --- | ---: |
| ten detectors merged into one 64-wide network | 0.971 |
| ensemble of the ten separate detectors (ceiling) | 0.971 |
| one 64-wide network trained on all ten labels | 0.972 |
| merge with the legacy alignment cost | 0.401 |
| aligned uniform soup | 0.112 |

The merge equals the ensemble at every k from 2 to 10 digits, and it is about 0.001 below a network trained on all ten labels at once. The legacy alignment cost charged a live unit half its norm for sitting next to an empty slot, so it stacked live units whose incoming cosine was above about 0.2. In 64 dimensions that is common, and the ten detectors collapsed into a handful of slots.

The limit is capacity: the merge is lossless when the detectors' live units fit in the width.

| width | live hidden-1 units, all ten detectors | merged | ensemble |
| ---: | ---: | ---: | ---: |
| 32 | 48 | 0.891 | 0.972 |
| 48 | 43 | 0.970 | 0.971 |
| 64 | 45 | 0.971 | 0.971 |
| 128 | 45 | 0.972 | 0.972 |
| 64, no sparsity penalty | 123 | 0.633 | 0.967 |

The sparsity penalty is what makes the detectors mergeable: it cuts the units they need from about 123 to about 45, and the ensemble of sparse detectors is no worse (0.971 against 0.967 without the penalty). Without it, the two alignment costs are close (0.633 interference, 0.651 legacy), since collisions are forced either way. The next section removes that limit.

## Extensions

Five additions, each measured by `python -m zeroth_grad extensions`, which writes `runs/extensions_metrics.json`. `python scripts/plot_results.py` draws the figure below from it.

![Alignment scaling, capacity, rank-k inject, and transformer merging](docs/extensions_results.png)

### Alignment that scales: Sinkhorn optimal transport

The Hungarian solve is O(N³). `--solver sinkhorn` solves the same assignment problem as entropic optimal transport (`solvers.py`). It runs log-domain Sinkhorn with epsilon scaling (ε from 1 down to 10⁻⁵ of the mean cost, 16 stages), then rounds each of the last three transport plans greedily over each row's top candidates and keeps the cheapest. A few leftover rows are solved exactly, and a 2-opt pass repairs rows not sitting at their row minimum. `--solver auto` uses Hungarian below width 2048 and Sinkhorn from there up.

Interference cost between a sparse 64-input layer (40% live units) and a noisy, permuted copy of it, wall-clock seconds on 4 CPU cores:

| width | Hungarian | Sinkhorn | Sinkhorn cost above optimal |
| ---: | ---: | ---: | ---: |
| 256 | 0.003 | 0.011 | 0.00% |
| 1024 | 0.075 | 0.066 | 0.12% |
| 2048 | 0.441 | 0.279 | 0.05% |
| 4096 | 2.64 | 1.75 | 0.04% |
| 8192 | 19.5 | 8.5 | 0.09% |

Sinkhorn is faster from width 1024 up, and its lead grows with width. The merge benchmarks are unchanged under it: 0.984 on halfspace pairs and 0.971 on the ten-digit merge, the same as with Hungarian.

### Dynamic capacity: widening, shared slots, declared heads

Three changes let a merge go over capacity without colliding (`capacity.py`, `compiler.py`):

- **Widening on collisions** (`--expand`). After alignment, a slot where two live units with different directions meet is a collision. The merge pads both parents with dead units, one per collision, and aligns again, until no collisions remain. Dead units are free in the interference cost, so the colliding units move apart.
- **Shared slots.** Two units whose incoming rows are positively parallel (cosine at least 0.999) compute the same feature up to scale, since `relu(c·z) = c·relu(z)`. They share one slot exactly: the partner's outgoing weights are rescaled onto the reference unit, and both parents read that slot at full strength. The same-parent rule used to halve what each parent read from a shared slot. On by default; `--no-share` disables it.
- **Declared ownership** (`--declare-heads`). Readout-norm ownership fails for dense parents, whose untrained readout rows are as large as their trained ones. Declared ownership reads each parent's `skills` metadata and gives it hard ownership of those heads, with a straight-through gate during training.

**Dense parents: activation folding.** Literal superposition (two unrelated ReLU features in one neuron) is not function-preserving, so dense parents are compacted approximately instead. `compile fold` (`fold.py`) refits a parent onto fewer units of its own. It picks hidden-2 units by greedy orthogonal matching pursuit against the owned logits, then hidden-1 units against the pre-activations of the kept hidden-2 units, weighted by readout importance. It refits by class-balanced weighted least squares and stops when balanced sign agreement with the original reaches `--agreement`. Readout rows for heads the parent does not own are zeroed. The fold also reports the numerical rank (SVD) of the layer it is fitting, which is a floor on the units any linear refit needs.

Ten-digit merges, mean over 3 detector seeds, 10-way test accuracy. "Fold 0.98" is the plain merge after folding every detector at 0.98 agreement, with no widening.

| zoo | merge | + declared heads | + widening (width h1/h2) | fold 0.98, original width | ensemble |
| --- | ---: | ---: | ---: | ---: | ---: |
| sparse, width 32 | 0.891 | 0.905 | 0.972 (48/37) | 0.950 | 0.972 |
| dense, width 64 (≈123 live units) | 0.633 | 0.840 | 0.967 (123/66) | 0.945 | 0.967 |
| sparse, width 64 | 0.971 | 0.971 | 0.971 (64/64) | 0.956 | 0.971 |

Widening makes every merge equal its ensemble, at the cost of width. Folding keeps the width fixed and costs a little accuracy: the width-32 ten-digit merge goes from 0.891 to 0.950, and the dense merge from 0.633 to 0.945. The two combine: folding at 0.98 and then widening gives 0.961 at width 35 on the width-32 zoo. Folding the dense detectors at 0.995 needs about 93 units, and widening then gives 0.965 at width 93.

### Rank-k inject

The rank-1 inject writes one direction, so it can only install halfspaces. `compile inject --rank k --examples batch.pt` installs a skill that depends on a k-dimensional subspace, in closed form (`lowrank.py`):

1. Estimate the subspace `U` from labeled examples by sliced average variance estimation: whitened class means plus differences of class covariances, top k eigenvectors.
2. Build a dictionary of ReLU ridge features on the projected inputs `Uᵀx` (directions times quantile thresholds), select up to `--units` of them greedily, and fit the readout by least squares followed by a few Newton (IRLS) steps of logistic regression.
3. Write the circuit onto spare units: `ΔW1 = Θ Uᵀ` (rank k), a ± pair of hidden-2 units that passes the fitted logit through the second ReLU, and the target head reads that pair. If there are too few spares, the network is widened with dead units first.

Old heads are untouched, so the old skill is unchanged (0.984 throughout). 12 held-out halfspace checkpoints, target head 6, 2,048 labeled examples (512 in parentheses), new-skill accuracy:

| skill | rank | before | rank-1 compiler | rank-k inject | rank-k, known subspace | 200-step Adam on the examples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| halfspace `v6·x > 0` | 1 | 0.503 | 0.966 | 0.980 (0.955) | 0.999 | 0.900 |
| slab `|v6·x| < 0.67` | 1 | 0.491 | 0.498 | 0.973 (0.950) | 0.991 | 0.579 |
| wedge (two halfspaces, AND) | 2 | 0.496 | 0.828 | 0.952 (0.910) | 0.987 | 0.798 |
| XOR `(v6·x)(v7·x) > 0` | 2 | 0.512 | 0.502 | 0.935 (0.868) | 0.988 | 0.563 |

The slab and XOR are invisible to a rank-1 write: their class means coincide. The gap to the known-subspace column is the cost of estimating `U` from examples. The Adam baseline is weak partly because the spares are dead and get no gradient. 16 fresh units in place of the spares give the same rank-k numbers.

### Transformers

`transformer.py` defines a small pre-norm decoder (2 layers, `d_model` 32, 4 heads of width 8, MLP 64, LayerNorm or RMSNorm) and every symmetry it has:

- **Residual stream.** One permutation of the residual coordinates, applied to every read (embedding, norm gains, Q/K/V and MLP inputs) and every write (attention output, MLP output, embeddings). A positive scale of every write is exact under either norm. Under RMSNorm, with the norm gains folded into the next weights, any orthogonal rotation of the stream is exact too.
- **Attention.** Heads are a set (permute them as units). Inside a head, `W_Q` and `W_K` can take any orthogonal map `R` together (`Rᵀ R = I` cancels in `QKᵀ`), and so can `W_V` and `W_O`.
- **MLP.** Hidden units permute as before.

`transformer_align.py` aligns one transformer to another by coordinate descent over these groups, in the style of Git Re-Basin weight matching. Each round matches the residual stream (assignment on read/write features, or Procrustes when rotations are allowed), fits a least-squares scale, matches heads by their QK and OV circuits, aligns each head's internals by Procrustes, and matches MLP units. `merge_transformers` then averages, or adds task vectors onto a shared base.

Benchmark (`transformer_bench.py`): a base model is trained on two token tasks, "copy the previous digit" and "running maximum". Two fine-tunes each specialize on one task. The second fine-tune is then moved by a random exact symmetry: residual permutation, head permutation, Q/K and V/O rotations, MLP permutation, and under RMSNorm a residual rotation with a scale of 1.7. Mean accuracy over both tasks:

| norm | fine-tunes alone | average | task arithmetic | moved partner, average, unaligned | moved partner, average, aligned | partner distance before → after |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LayerNorm | 0.974 / 0.793 | 0.991 | 0.896 | 0.162 | 0.991 | 1.31 → 0.000 |
| RMSNorm | 0.989 / 0.870 | 0.998 | 0.952 | 0.334 | 0.998 | 1.69 → 0.000 |

Alignment recovers the hidden symmetry exactly, so the merge of a moved partner equals the merge of the unmoved one. Between two models trained from independent initializations, the midpoint is still poor (endpoints 1.0; midpoint 0.21 unaligned, 0.34 aligned under LayerNorm; 0.21, 0.42 with permutations only, and 0.64 with rotations under RMSNorm). Alignment removes symmetry, not the loss barrier between independently trained basins. The transformer path is geometric only: the learned compiler does not process transformer weights.

## Commands

```bash
python -m zeroth_grad populate --out checkpoints
python -m zeroth_grad train --checkpoints checkpoints --out runs/compiler.pt
python -m zeroth_grad eval --compiler runs/compiler.pt --checkpoints checkpoints

python -m zeroth_grad widen --ckpt checkpoints/heldout_single_s0_seed5.safetensors \
    --out /tmp/wide.safetensors --copies 4 --shuffle
python -m zeroth_grad compile compress --ckpt /tmp/wide.safetensors --out /tmp/slim.safetensors

python -m zeroth_grad compile inject --ckpt checkpoints/heldout_single_s0_seed5.safetensors \
    --concepts checkpoints/concepts.pt --concept-index 6 --head 6 --out /tmp/injected.safetensors

python -m zeroth_grad compile merge \
    --ckpt-a checkpoints/heldout_single_s0_seed5.safetensors \
    --ckpt-b checkpoints/heldout_single_s1_seed5.safetensors \
    --out /tmp/merged.safetensors

python -m zeroth_grad compile merge --ckpts checkpoints/heldout_single_s{0,1,2,3}_seed5.safetensors \
    --out /tmp/merged4.safetensors

python -m zeroth_grad digits-populate --out digits
python -m zeroth_grad digits-eval --checkpoints digits
python -m zeroth_grad compile merge --prior --ckpts digits/digit{0,1,2,3,4,5,6,7,8,9}_seed0.safetensors \
    --out /tmp/ten_way.safetensors

python -m zeroth_grad digits-populate --out digits_w32 --hidden 32
python -m zeroth_grad digits-populate --out digits_dense --sparsity 0
python -m zeroth_grad compile merge --prior --declare-heads --expand --solver auto \
    --ckpts digits_dense/digit{0,1,2,3,4,5,6,7,8,9}_seed0.safetensors --out /tmp/ten_way_wide.safetensors
python -m zeroth_grad compile fold --ckpt digits_dense/digit3_seed0.safetensors \
    --examples digits_x.pt --agreement 0.98 --out /tmp/digit3_folded.safetensors
python -m zeroth_grad compile inject --rank 2 --examples xor_batch.pt --head 6 \
    --ckpt checkpoints/heldout_single_s0_seed5.safetensors --out /tmp/xor.safetensors

python -m zeroth_grad extensions        # -> runs/extensions_metrics.json
python scripts/plot_results.py
```

`compile merge` folds `--ckpts` (or `--ckpt-a` followed by one or more `--ckpt-b`) left to right, and the output's `skills` metadata is the union of its parents'. `--prior` merges with an untrained compiler sized to the checkpoints, for shapes the trained compiler was not built for. `compile inject` also accepts `--examples batch.pt` (`{"x", "y"}`) or `--phi direction.pt`. `--solver` picks `hungarian`, `sinkhorn`, or `auto` alignment. `--expand` widens a merge instead of letting live units collide, `--declare-heads` gives each parent hard ownership of the heads in its `skills` metadata, and `--no-share` turns off exact slot sharing for parallel units. `compile fold` needs a calibration batch (`--examples`, a tensor or `{"x": ...}`, labels unused) and takes `--agreement` and `--heads`. `compile inject --rank k` needs labeled `--examples`, and `--units` sets the circuit size (default: the spares). `compile` runs under `torch.no_grad()` and does not require gradients on the loaded tensors.

`checkpoints/`, `digits/`, `digits_w32/`, `digits_dense/`, and `runs/` are local artifacts. `extensions` reads the three digit zoos and skips any that are missing. Tests run with `pip install -e ".[dev]"` and `pytest`. The digits benchmark needs scikit-learn (`pip install -e ".[digits]"`; its test is skipped without it), and the figures need matplotlib (`pip install -e ".[plot]"`).

## Neighbors

Git Re-Basin aligns hidden units before averaging. Task arithmetic edits a checkpoint by adding a task vector. Deep weight-space networks and neural functional transformers process a network's weights with permutation-equivariant layers. Those methods are the neighborhood. This repository is the checkpoint-to-checkpoint map: one forward pass that compresses, injects, or merges, trained by the edited model's loss and not by a training loop on that model.

## Layout

`src/zeroth_grad/` holds the MLP and functional forward (`mlp.py`), synthetic skills (`data.py`), checkpoint I/O (`io.py`), alignment and collision counting (`align.py`), Hungarian and Sinkhorn assignment solvers (`solvers.py`), the edit algebra (`edit.py`), widening and shared slots (`capacity.py`), the equivariant compiler (`compiler.py`), activation folding (`fold.py`), rank-k inject (`lowrank.py`), the zoo (`zoo.py`), functional training (`train.py`), held-out eval including the k-way merge curve (`evaluate.py`), the digits benchmark (`digits.py`), the transformer model and its symmetries (`transformer.py`), transformer alignment and merging (`transformer_align.py`), the transformer benchmark (`transformer_bench.py`), the extensions benchmark (`extensions.py`), and the CLI (`cli.py`).
