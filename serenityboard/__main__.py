"""SerenityBoard CLI entry point."""
from __future__ import annotations

import argparse
import sys

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="serenityboard",
        description="SerenityBoard — Training dashboard for PyTorch trainers.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the SerenityBoard server")
    serve_parser.add_argument("--logdir", required=True, help="Root log directory")
    serve_parser.add_argument("--port", type=int, default=6006, help="Server port")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Server host")

    # migrate
    migrate_parser = subparsers.add_parser("migrate", help="Migrate TensorBoard logs")
    migrate_parser.add_argument("--from", dest="from_dir", required=True, help="TensorBoard logdir")
    migrate_parser.add_argument("--to", dest="to_dir", required=True, help="SerenityBoard logdir")

    args = parser.parse_args(argv)

    if args.command == "serve":
        _serve(args)
    elif args.command == "migrate":
        _migrate(args)
    else:
        parser.print_help()
        sys.exit(1)


def _serve(args) -> None:
    import uvicorn
    from serenityboard.server.app import create_app

    app = create_app(args.logdir)
    uvicorn.run(app, host=args.host, port=args.port)


def _migrate(args) -> None:
    """Migrate TensorBoard event files to SerenityBoard format."""
    try:
        from tensorboard.compat.proto import event_pb2  # noqa: F401
    except ImportError:
        print(
            "Migration requires tensorboard. Install with: pip install tensorboard",
            file=sys.stderr,
        )
        sys.exit(1)

    import glob
    import os

    import serenityboard as sb

    from_dir = args.from_dir
    to_dir = args.to_dir

    if not os.path.isdir(from_dir):
        print(f"Source directory not found: {from_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(to_dir, exist_ok=True)

    # Find all event files
    event_files = glob.glob(os.path.join(from_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        print(f"No TensorBoard event files found in {from_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(event_files)} event file(s)")

    for event_file in event_files:
        # Derive run name from relative directory
        rel_dir = os.path.relpath(os.path.dirname(event_file), from_dir)
        run_name = rel_dir if rel_dir != "." else os.path.basename(event_file).split(".")[0]

        print(f"  Migrating: {rel_dir} -> {run_name}")

        writer = sb.SummaryWriter(logdir=to_dir, run_name=run_name)
        count = 0

        try:
            from tensorboard.backend.event_processing import event_file_loader
            loader = event_file_loader.EventFileLoader(event_file)

            for event in loader.Load():
                if event.HasField("summary"):
                    for value in event.summary.value:
                        if value.HasField("simple_value"):
                            writer.add_scalar(value.tag, value.simple_value, event.step)
                            count += 1
                        elif value.HasField("tensor"):
                            # Skip complex tensor data in V1
                            pass
        except Exception as e:
            print(f"    Error: {e}")
        finally:
            writer.close()

        print(f"    Migrated {count} scalar(s)")

    print("Migration complete.")


if __name__ == "__main__":
    main()
