#!/usr/bin/env python3
"""Manual test: write scalars in a loop to verify live chart updates.

Usage:
    # Terminal 1: start the server
    cd /home/alex/serenityboard
    python3 -m uvicorn serenityboard.server.app:app --factory --port 6006
    # (use: create_app("./test_logdir") — see below for factory usage)

    # Or use the helper:
    python3 scripts/write_loop.py --serve &
    python3 scripts/write_loop.py

    # Open browser: http://localhost:6006
    # Select the run, select loss/train and loss/val tags
    # Watch the charts update live
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serenityboard as sb


def main():
    parser = argparse.ArgumentParser(description="SerenityBoard write loop test")
    parser.add_argument(
        "--logdir", default="./test_logdir", help="Log directory (default: ./test_logdir)"
    )
    parser.add_argument(
        "--run-name", default="demo_run", help="Run name (default: demo_run)"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of steps (default: 1000)"
    )
    parser.add_argument(
        "--interval", type=float, default=0.1, help="Seconds between writes (default: 0.1)"
    )
    parser.add_argument(
        "--serve", action="store_true", help="Start the server instead of writing"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Delete logdir before starting"
    )
    parser.add_argument(
        "--port", type=int, default=6006, help="Server port (default: 6006)"
    )
    args = parser.parse_args()

    logdir = os.path.abspath(args.logdir)

    if args.clean and os.path.isdir(logdir):
        shutil.rmtree(logdir)
        print(f"Cleaned {logdir}")

    if args.serve:
        import uvicorn
        from serenityboard.server.app import create_app

        app = create_app(logdir)
        print(f"Serving SerenityBoard at http://localhost:{args.port}")
        print(f"Watching logdir: {logdir}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return

    # ── Write loop ────────────────────────────────────────────────

    os.makedirs(logdir, exist_ok=True)

    print(f"Writing to {logdir}/{args.run_name}/board.db")
    print(f"Steps: {args.steps}, interval: {args.interval}s")
    print("Start the server in another terminal:")
    print(f"  python3 scripts/write_loop.py --serve --logdir {logdir}")
    print(f"Then open http://localhost:6006")
    print()

    with sb.SummaryWriter(
        logdir=logdir,
        run_name=args.run_name,
        hparams={"lr": 1e-4, "batch_size": 32, "model": "resnet50"},
    ) as writer:
        for step in range(args.steps):
            # Simulated training loss (decaying with noise)
            train_loss = 2.0 * math.exp(-step / 200) + 0.1 * math.sin(step / 10) + 0.05
            val_loss = 2.0 * math.exp(-step / 250) + 0.15 * math.sin(step / 15) + 0.08

            # Learning rate with warmup + cosine decay
            if step < 50:
                lr = 1e-4 * (step / 50)
            else:
                lr = 1e-4 * 0.5 * (1 + math.cos(math.pi * (step - 50) / (args.steps - 50)))

            # Accuracy (rising)
            accuracy = 1.0 - math.exp(-step / 300) + 0.02 * math.sin(step / 20)
            accuracy = max(0, min(1, accuracy))

            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("loss/val", val_loss, step)
            writer.add_scalar("lr", lr, step)
            writer.add_scalar("metrics/accuracy", accuracy, step)

            if step % 100 == 0:
                writer.add_text("log", f"Step {step}: loss={train_loss:.4f}", step)
                print(f"  step {step:4d}  loss={train_loss:.4f}  acc={accuracy:.4f}")

            time.sleep(args.interval)

        writer.add_hparams(
            {"lr": 1e-4, "batch_size": 32, "model": "resnet50"},
            {"final_loss": train_loss, "final_accuracy": accuracy},
        )

    print(f"\nDone! {args.steps} steps written.")


if __name__ == "__main__":
    main()
