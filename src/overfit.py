"""Fit a fixed CIFAR-10 subset with BP, grouped Jacobi and/or BGS-TP."""
import argparse
import json

import torch

from .data import HALF_DIM, load_task_data
from .plotting import plot_histories
from .revnet import RevNet
from .train import add_common_args, positive_int, resolve_device
from .training import Trainer, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--method", choices=["all", "bp", "jacobi", "gs"], default="all")
    parser.add_argument("--n-samples", type=positive_int, default=1)
    parser.add_argument("--steps", type=positive_int, default=3000)
    parser.add_argument("--eval-every", type=positive_int, default=20)
    args = parser.parse_args()
    if args.depth < 2:
        parser.error("depth must be >= 2")
    device = resolve_device(args.device)
    train_x, _ = load_task_data(args.data_dir, test_samples=1)
    if args.n_samples > len(train_x):
        parser.error("n-samples exceeds the training split size")
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(train_x))[:args.n_samples]
    fixed = train_x[indices].to(device)
    print(f"device={device}, fixed sample indices={indices.tolist()}", flush=True)
    methods = ["bp", "jacobi", "gs"] if args.method == "all" else [args.method]
    runs = {}
    suffix = f"depth{args.depth}_n{args.n_samples}"
    for method in methods:
        torch.manual_seed(args.seed)
        model = RevNet(args.depth, HALF_DIM).to(device)
        trainer = Trainer(model, "bgstp" if method == "gs" else method, args.lr)
        history = []
        log_path = args.output_dir / "logs" / f"overfit_{method}_{suffix}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            for step in range(1, args.steps + 1):
                i = (step - 1) % len(fixed)
                trainer.step(fixed[i:i + 1], step, args.steps)
                if step == 1 or step % args.eval_every == 0 or step == args.steps:
                    entry = {"step": step, "train_mse": evaluate(model, fixed)}
                    history.append(entry)
                    log.write(json.dumps(entry) + "\n")
                    log.flush()
                    print(f"{method} step {step}: mse={entry['train_mse']:.8f}", flush=True)
        runs[method] = history
    path = args.output_dir / "plots" / f"overfit_{args.method}_{suffix}.png"
    plot_histories(runs, path, log_scale=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
