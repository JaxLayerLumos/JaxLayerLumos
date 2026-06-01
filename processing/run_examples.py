from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Iterable


DEFAULT_TIMEOUT_SECONDS = 600


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root()
    default_output_dir = Path(tempfile.gettempdir()) / "jaxlayerlumos-example-runs"

    parser = argparse.ArgumentParser(
        description="Execute example notebooks without changing repository files by default."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Execute notebooks and write executed copies to --output-dir. This is the default.",
    )
    mode.add_argument(
        "--inplace",
        action="store_true",
        help="Execute notebooks and overwrite the notebook files with fresh outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory for executed notebooks in check mode. Default: {default_output_dir}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout in seconds per notebook. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help="Kernel name used to execute notebooks. Default: python3",
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help=f"Notebook paths to execute. Default: all notebooks under {repo_root / 'examples'}",
    )
    return parser.parse_args()


def discover_notebooks(paths: Iterable[Path]) -> list[Path]:
    notebooks = list(paths)
    if notebooks:
        return [path.resolve() for path in notebooks]

    examples_dir = get_repo_root() / "examples"
    return sorted(examples_dir.glob("*.ipynb"))


def execute_notebook(
    notebook_path: Path,
    output_path: Path,
    *,
    kernel_name: str,
    timeout: int,
) -> None:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise SystemExit(
            "Missing notebook execution dependencies. Install them with "
            "`pip install .[dev]`; use `pip install .[dev,examples]` "
            "to run all example notebooks."
        ) from exc

    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        kernel_name=kernel_name,
        timeout=timeout,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)


def main() -> int:
    args = parse_args()
    notebooks = discover_notebooks(args.notebooks)

    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1

    check_mode = not args.inplace
    for index, notebook_path in enumerate(notebooks, start=1):
        if not notebook_path.exists():
            print(f"[{index}/{len(notebooks)}] MISSING {notebook_path}", file=sys.stderr)
            return 1

        output_path = notebook_path if args.inplace else args.output_dir / notebook_path.name
        print(f"[{index}/{len(notebooks)}] Executing {notebook_path}")

        try:
            execute_notebook(
                notebook_path,
                output_path,
                kernel_name=args.kernel_name,
                timeout=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - report notebook path clearly.
            print(f"[{index}/{len(notebooks)}] FAILED {notebook_path}: {exc}", file=sys.stderr)
            return 1
        else:
            destination = "in place" if args.inplace else output_path
            print(f"[{index}/{len(notebooks)}] OK -> {destination}")

    if check_mode:
        print(f"\nExecuted notebooks written to: {args.output_dir}")
    else:
        print("\nNotebook outputs refreshed in place.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
