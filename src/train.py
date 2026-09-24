"""Train BGS-TP or the end-to-end BP baseline on masked CIFAR-10 pixels."""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import HALF_DIM, ROOT, load_task_data
from .plotting import plot_histories
from .revnet import RevNet
from .training import Trainer, evaluate


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def add_common_args(parser):
    parser.add_argument("--depth", type=positive_int, default=6)
    parser.add_argument("--lr", type=positive_float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def build_run_name(args):
    if args.method == "bp":
        return f"bp_baseline_depth{args.depth}"
    name = f"bgstp_depth{args.depth}_{args.layer_select}"
    if args.lr_decay_to_frac is not None:
        name += f"_decay{args.lr_decay_to_frac}"
    if args.layers_per_step != 1:
        name += f"_sweep{args.layers_per_step}"
    return name


def train(args):
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    model = RevNet(args.depth, HALF_DIM).to(device)
    trainer = Trainer(model, args.method, args.lr, args.layer_select,
                      args.layers_per_step, args.lr_decay_to_frac)
    train_x, test_x = load_task_data(args.data_dir, args.train_samples, args.test_samples)
    steps_per_epoch = math.ceil(len(train_x) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    eval_every = max(1, round(args.eval_every_epochs * steps_per_epoch))
    name = args.run_name or build_run_name(args)
    log_path = args.output_dir / "logs" / f"{name}.jsonl"
    plot_path = args.output_dir / "plots" / f"{name}.png"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name}: device={device}, train={len(train_x)}, test={len(test_x)}, "
          f"steps={total_steps}, params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    history, start = [], time.monotonic()
    with log_path.open("w") as log:
        for step in range(1, total_steps + 1):
            indices = torch.randint(len(train_x), (args.batch_size,))
            loss = trainer.step(train_x[indices].to(device), step, total_steps)
            if step % eval_every == 0 or step == total_steps:
                # The final record uses the complete selected train/test splits.
                n = None if step == total_steps else args.eval_samples
                entry = {"step": step, "epoch": step * args.batch_size / len(train_x),
                         "train_mse": evaluate(model, train_x, n, args.batch_size),
                         "test_mse": evaluate(model, test_x, n, args.batch_size)}
                history.append(entry)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                plot_histories({name: history}, plot_path)
                print(f"step {step}/{total_steps} | loss={loss:.6f} | "
                      f"train={entry['train_mse']:.6f} test={entry['test_mse']:.6f} | "
                      f"{time.monotonic() - start:.1f}s", flush=True)
    print(f"wrote {log_path}\nwrote {plot_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--method", choices=["bgstp", "bp"], default="bgstp")
    parser.add_argument("--epochs", type=positive_int, default=300)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--layer-select", choices=["random", "round-robin"], default="random")
    parser.add_argument("--layers-per-step", type=positive_int, default=1)
    parser.add_argument("--lr-decay-to-frac", type=positive_float)
    parser.add_argument("--eval-every-epochs", type=positive_float, default=1.0)
    parser.add_argument("--eval-samples", type=positive_int, default=3000)
    parser.add_argument("--train-samples", type=positive_int, help="limit the training split for a short run")
    parser.add_argument("--test-samples", type=positive_int, help="limit the test split for a short run")
    parser.add_argument("--run-name", help="log/plot basename; same-name runs overwrite previous outputs")
    args = parser.parse_args()
    if args.depth < 2 or args.layers_per_step > args.depth:
        parser.error("depth must be >= 2 and layers-per-step must not exceed depth")
    if args.lr_decay_to_frac is not None and args.lr_decay_to_frac > 1:
        parser.error("lr-decay-to-frac must be in (0, 1]")
    if args.method == "bp" and (args.lr_decay_to_frac is not None or
                                args.layers_per_step != 1 or args.layer_select != "random"):
        parser.error("layer selection, block sweeps and LR decay are BGS-TP options")
    if args.run_name and Path(args.run_name).name != args.run_name:
        parser.error("run-name must be a filename, not a path")
    train(args)


if __name__ == "__main__":
    main()
