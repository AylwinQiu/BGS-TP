# BGS-TP

**Block Gauss-Seidel Target Propagation** derives per-layer targets by inverting a reversible network, then trains each layer with local gradient updates instead of end-to-end backpropagation. Forward states and targets are recomputed after every sub-update so that the next layer uses the latest weights. This standalone project includes BGS-TP, a BP baseline using the same architecture, single-sample overfitting diagnostics, and plotting tools.

## Running the project

Install Pixi first. The environment configuration supports Linux x86_64 with Python 3.12 and uses the CPU by default. Dependencies are downloaded on the first installation.

```bash
cd BGS-TP
pixi install
pixi run smoke                 # Small CPU checks; no dataset download

# Short training run; CIFAR-10 is downloaded to data/ on first use
pixi run train --depth 6 --epochs 1 --train-samples 128 --test-samples 64

# Original setup: 18 layers, 3 sequential updates per batch, LR decays to 0.2x
pixi run train --depth 18 --epochs 300 --layer-select round-robin \
  --layers-per-step 3 --lr-decay-to-frac 0.2
pixi run bp --depth 18 --epochs 300
```

For an NVIDIA GPU, select the separate CUDA 12.6 environment. A compatible driver is required:

```bash
pixi run -e cuda train --depth 18 --layer-select round-robin \
  --layers-per-step 3 --lr-decay-to-frac 0.2
pixi run -e cuda bp --depth 18
```

The program selects CUDA or CPU automatically; use `--device cpu` or `--device cuda` to choose explicitly. Set `--data-dir` for the CIFAR-10 cache and `--output-dir` for generated outputs; both default to directories within the project. Training writes `outputs/logs/*.jsonl` and `outputs/plots/*.png`. Runs with the same name overwrite previous outputs; use `--run-name` to distinguish experiments. Periodic evaluation samples 3,000 examples per split by default, while the final evaluation processes the complete selected splits in batches.

```bash
# Compare BP, Jacobi, and GS on the same samples and initial weights
# Generate logs and a plot with a logarithmic MSE axis
pixi run overfit --method all --n-samples 1 --steps 3000
# Or run one method with --method bp, jacobi, or gs

pixi run compare \
  --run outputs/logs/bp_baseline_depth18.jsonl:BP \
  --run outputs/logs/bgstp_depth18_round-robin_decay0.2_sweep3.jsonl:BGS-TP \
  --out outputs/plots/comparison.png
```

All entry points support `--help`. To run Python directly, use `python -m src.train`, `python -m src.train --method bp`, `python -m src.overfit`, or `python -m src.plotting` from the project root.

## Method and task

CIFAR-10 images are scaled to `[0, 1]` and flattened in CHW order into 3,072-dimensional vectors. The first 1,536 values provide context; the remaining 1,536 are zeroed out and predicted using MSE as the metric. A “half-image” here means half of the flattened vector, not a spatial left/right split. Classification labels are unused.

`RevNet` uses independent linear layers to update two states alternately: even-indexed layers apply `x1 += f(x2)`, and odd-indexed layers apply `x2 += f(x1)`. The function `f` uses LeakyReLU and clips its output when the L2 norm exceeds 10. Each update is inverted by subtraction. The terminal target keeps the current `x1` and replaces `x2` with the ground-truth pixels, then inversion propagates targets to earlier layers. The local loss is the MSE between concatenated state pairs, with backpropagation restricted to the current layer.

- **BP**: Backpropagates through the entire network and updates all layers at each step, providing a baseline for fitting capacity.
- **BGS-TP**: Selects distinct layers for each batch and updates them sequentially, recomputing inputs and targets for every update. `--layers-per-step 1` gives single-layer GS; `random` selects layers randomly, while `round-robin` cycles through them.
- **Jacobi diagnostic**: Updates adjacent layers using two alternating grouping schemes, sharing a stale target snapshot within each step to examine the effects of outdated targets.

Early experiments explored Forward-Forward/goodness objectives and coupling functions based on tanh, unbounded LeakyReLU, and LayerNorm. Pixel regression with ground-truth targets was then adopted to evaluate target propagation in isolation. Single-sample tests exposed a fitting bottleneck in the old LayerNorm configuration, motivating norm clipping that limits only magnitude. For deeper networks, round-robin selection, learning-rate decay, and small blocks of sequential updates help address training fluctuations and insufficient updates per layer.

## Previous results

The following approximate results come from the original experiments (300 epochs). They provide context and were not regenerated during this code reorganization.

| Depth | Method | Train MSE | Test MSE |
| --- | --- | --- | --- |
| 6 | BP | 0.0056 | 0.0121 |
| 6 | Jacobi | 0.0391 | 0.0407 |
| 6 | GS, random single-layer updates | 0.0118 | 0.0124 |
| 18 | BP | 0.0027 | 0.0119 |
| 18 | GS, random single-layer updates | 0.0133 | 0.0136 |
| 18 | Round-robin single-layer updates + LR decay to 0.2x | 0.0129 | 0.0132 |
| 18 | Round-robin + LR decay + sweep=3 | 0.0112 | 0.0115 |

BGS-TP still has higher training error than BP; lower test error in one experiment does not establish a general advantage in generalization. Earlier 36-layer experiments showed more pronounced fluctuations. A full sweep-size study, comparisons across multiple random seeds, and evaluation on other tasks remain open.

Each local update builds a gradient graph only for the active layer, but this implementation still stores states and targets for all layers, so total GPU memory use is not independent of depth. Repeated forward passes and target inversion also add computation; equal step counts across methods do not imply equal compute budgets. The current design requires layers to have separate parameters. Floating-point addition and subtraction are not guaranteed to be bit-exact inverses: training uses ordinary subtraction to derive new targets, and inverse checks allow numerical tolerance. The original TwoSum numerical appendix is not retained as a training dependency.

## Files

| File | Purpose |
| --- | --- |
| `src/revnet.py` | Reversible coupling network, norm clipping, and target inversion |
| `src/training.py` | Shared update logic and batched evaluation |
| `src/train.py` | BGS-TP / BP training entry point |
| `src/data.py` | Standalone CIFAR-10 loading and masking task |
| `src/overfit.py` | Overfitting diagnostics and comparison plots |
| `src/plotting.py` | Training curves and JSONL comparisons, including legacy logs |
| `tests/test_training.py` | Checks for invertibility, local updates, data layout, and fitting |

Documentation is consolidated in this README. Historical reports, run logs, and duplicate images are not bundled with the project. Datasets, environments, and generated experiment outputs are excluded by `.gitignore`.
