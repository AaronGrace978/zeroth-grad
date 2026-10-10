# zeroth-grad

A weight-space compiler. It reads a small MLP checkpoint (`.safetensors` or `.pt`) and writes an edited checkpoint in one forward pass. Compress, inject, and merge never run an optimizer on the network being edited.

The compiler itself is trained. Each step samples a checkpoint from a zoo of models trained on synthetic halfspaces, applies an edit, and backpropagates a functional loss into the compiler only. At apply time the checkpoint tensors are constants.

![A checkpoint goes into the weight compiler, which compresses, injects, or merges it](docs/weight-compiler-diagram.jpg)

## Edit families

Hidden units are a set. Permuting a layer permutes that layer's tokens and leaves the other layer's tokens in place, because messages between layers are products with the weight matrix (`W @ features`), not a flattened parameter vector.

**Inject.** Install an unseen halfspace on a chosen output head. Spare neurons (outgoing norm zero) are overwritten with a rank-1 circuit

```
ΔW1 = a φᵀ,   ΔW2 = c sᵀ,   ΔW3[head] = r
```

with scales `a`, `s`, `c`, `r` predicted by the compiler. Old heads are left alone, so the skill already in the checkpoint stays put. `φ` is the concept direction, or `normalize(mean(x | y=1) - mean(x | y=0))` estimated from a labeled batch.

**Compress.** Exact duplicate neurons (incoming cosine at least 0.999) are one cluster. The compiler keeps one neuron per cluster, up to the pre-duplication width, and adds the dropped copies' outgoing weights onto the survivor. A learned residual breaks ties; on exact copies the collapse is geometric. Magnitude pruning keeps the largest fragments and drops the rest, which is the baseline this edit beats.

**Merge.** Hidden units of checkpoint B are matched to A with the Hungarian algorithm. The cost is magnitude-weighted cosine distance on incoming weights (row plus bias), plus a penalty for pairing two live neurons whose incoming directions disagree. Per-neuron gates mix the endpoints, and a connection is kept only when both endpoints come from the same parent. A random hidden permutation of B is undone by the match, so the merged function does not depend on that permutation.

## Models

Inputs are `x ~ N(0, I_16)`. Eight fixed orthonormal directions define the skills `y = 1[v_k · x > 0]`. The MLP is `16 → 32 → 32 → 8`: one logit per skill, so two skills can be true of one checkpoint at the same time. A single shared logit cannot score well on two different halfspaces of the same `x`.

The zoo is trained from one shared initialization. Skills 0–5 are the training skills. Skills 6–7 are held out and are the inject targets. The last two seeds of each training skill are held out as well. Four hidden units in each layer are pinned at zero during student training, and any further neuron whose removal does not change skill accuracy is zeroed afterward. Those spares are where an inject lands.

## Results

Held-out checkpoints, CPU, after `populate --steps 300` and `train --steps 400`. Accuracy is mean sign agreement on 2,500–3,000 fresh samples. The full dump is written to `runs/metrics.json`. `python scripts/plot_results.py` redraws the figure below.

![Held-out accuracies for inject, compress, and merge](docs/heldout_results.png)

**Inject** (24 edits: 12 held-out models × skills 6 and 7):

| | new skill | old skill |
| --- | ---: | ---: |
| checkpoint before the edit | 0.503 | 0.983 |
| untrained compiler (geometric prior only) | 0.807 | 0.983 |
| trained compiler | 0.998 | 0.983 |
| 200-step Adam on the target (oracle) | 0.904 | — |

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
| compiler | 0.874 |
| norm-gate stitch (the compiler's initialization) | 0.874 |
| uniform average, no alignment | 0.846 |
| uniform average after alignment | 0.886 |
| compiler, partner hidden units permuted | 0.874 |
| uniform average, partner hidden units permuted | 0.861 |

Permutation does not move the compiler's score. Aligned uniform soup is the strongest of these averages; the learned gates stay on the norm-gate stitch they were initialized to.

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
```

`compile inject` also accepts `--examples batch.pt` (`{"x", "y"}`) or `--phi direction.pt`. `compile` runs under `torch.no_grad()` and does not require gradients on the loaded tensors.

`checkpoints/` and `runs/` are local artifacts.

## Neighbors

Git Re-Basin aligns hidden units before averaging. Task arithmetic edits a checkpoint by adding a task vector. Deep weight-space networks and neural functional transformers process a network's weights with permutation-equivariant layers. Those methods are the neighborhood. This repository is the checkpoint-to-checkpoint map: one forward pass that compresses, injects, or merges, trained by the edited model's loss and not by a training loop on that model.

## Layout

`src/zeroth_grad/` holds the MLP and functional forward (`mlp.py`), synthetic skills (`data.py`), checkpoint I/O (`io.py`), Hungarian alignment (`align.py`), the edit algebra (`edit.py`), the equivariant compiler (`compiler.py`), the zoo (`zoo.py`), functional training (`train.py`), held-out eval (`evaluate.py`), and the CLI (`cli.py`).
