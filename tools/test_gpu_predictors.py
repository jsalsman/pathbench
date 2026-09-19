#!/usr/bin/env python3
"""Install a CUDA-enabled PathBench environment and smoke-test ArtP and DArtP.

Stepwise installation (each step skips components that are already suitable):

1. Install the Ubuntu prerequisites listed in README.md if they are missing.
2. Build the pinned espeak-ng commit unless the installation marker confirms it.
3. Ensure an NVIDIA driver is installed and ``nvidia-smi --list-gpus`` works.
   Host driver installation is intentionally left to the machine or cloud provider.
4. From the PathBench checkout, run ``python3 tools/test_gpu_predictors.py``.
5. This script then reuses or creates ``tools/gpu_venv``; installs PyTorch,
   torchaudio, PathBench, and test dependencies only when its import/version
   checks fail; verifies CUDA access; and runs the focused ArtP and DArtP tests.
6. Download the English n-gram model described in README.md to exercise DArtP;
   without it, pytest reports the DArtP test as skipped.

Run with ``--help`` to select another Python, CUDA/PyTorch version, or venv.
Allow at least 12 GB system RAM and 8 GB GPU VRAM (16 GB RAM and 12--16 GB
VRAM recommended). Google Colab NVIDIA GPU runtimes are supported when
``nvidia-smi`` works; T4, L4, and A100 assignments meet the recommendation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


class CommandError(RuntimeError):
    """A child command failed and its exit status should be preserved."""

    def __init__(self, message: str, returncode: int) -> None:
        super().__init__(message)
        self.returncode = returncode


def run(
    command: list[str], description: str, *, cwd: Path | None = None
) -> None:
    """Run a visible command and turn a nonzero status into a useful error."""
    print(f"\n==> {description}", flush=True)
    print("+ " + " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, cwd=cwd)
    except subprocess.CalledProcessError as error:
        raise CommandError(
            f"{description} failed with exit status {error.returncode}. "
            f"Review the command output above: {' '.join(command)}",
            error.returncode,
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a CUDA-enabled PathBench virtual environment, verify GPU "
            "access, and run the ArtP and DArtP smoke tests."
        )
    )
    parser.add_argument(
        "--python", default=os.environ.get("PYTHON", "python3"),
        help="Python 3.10-3.12 executable (default: PYTHON or python3)",
    )
    parser.add_argument(
        "--pytorch-version", default=os.environ.get("PYTORCH_VERSION", "2.6.0"),
        help="matching PyTorch and torchaudio version (default: 2.6.0)",
    )
    parser.add_argument(
        "--cuda-version", default=os.environ.get("CUDA_VERSION", "12.4"),
        help="CUDA wheel version, such as 12.4 (default: 12.4)",
    )
    parser.add_argument(
        "--venv", type=Path,
        default=Path(os.environ.get("VENV", SCRIPT_DIR / "gpu_venv")),
        help="virtual-environment destination (default: tools/gpu_venv)",
    )
    return parser.parse_args()


def require_program(program: str, explanation: str) -> str:
    path = shutil.which(program)
    if path is None:
        raise RuntimeError(f"Required program '{program}' was not found. {explanation}")
    return path


def python_succeeds(python: Path | str, code: str) -> bool:
    """Return whether an environment already satisfies a Python-side check."""
    return subprocess.run(
        [str(python), "-c", code], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def main() -> int:
    args = parse_args()
    venv = args.venv.expanduser().resolve()
    try:
        python = require_program(
            args.python, "Set --python to a Python 3.10-3.12 executable."
        )
        nvidia_smi = require_program(
            "nvidia-smi", "Install an NVIDIA driver and expose the GPU to this environment."
        )
        require_program(
            "espeak-ng", "Install the pinned version described in README.md first."
        )

        version_check = subprocess.run(
            [
                python, "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}'); "
                "raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)",
            ],
            text=True, capture_output=True,
        )
        if version_check.returncode != 0:
            detected = version_check.stdout.strip() or "unknown"
            raise RuntimeError(
                f"Python 3.10-3.12 is required, but '{python}' reports {detected}."
            )

        run([nvidia_smi, "--list-gpus"], "Checking the NVIDIA driver and visible GPUs")
        if not (venv / "bin" / "python").is_file():
            run([python, "-m", "venv", str(venv)], "Creating the GPU virtual environment")
        else:
            print(f"\n==> Reusing existing virtual environment: {venv}")
        venv_python = venv / "bin" / "python"
        cuda_tag = f"cu{args.cuda_version.replace('.', '')}"

        torch_check = (
            "import torch, torchaudio; "
            f"assert torch.__version__.split('+')[0] == '{args.pytorch_version}'; "
            f"assert torchaudio.__version__.split('+')[0] == '{args.pytorch_version}'; "
            f"assert (torch.version.cuda or '').replace('.', '') == '{cuda_tag[2:]}'"
        )
        if not python_succeeds(venv_python, torch_check):
            run([
                str(venv_python), "-m", "pip", "install",
                "--force-reinstall",
                f"torch=={args.pytorch_version}", f"torchaudio=={args.pytorch_version}",
                "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}",
            ], "Installing missing or mismatched CUDA-enabled PyTorch packages")
        else:
            print("\n==> Matching CUDA-enabled PyTorch packages are already installed")

        project_check = (
            "from pathlib import Path; import pathbench, pytest, kenlm; "
            f"assert Path(pathbench.__file__).resolve().is_relative_to(Path({str(REPO_ROOT)!r}))"
        )
        if not python_succeeds(venv_python, project_check):
            run([
                str(venv_python), "-m", "pip", "install", "-e", f"{REPO_ROOT}[all]",
                "pytest", "kenlm",
            ], "Installing missing PathBench or test dependencies")
        else:
            print("\n==> PathBench and test dependencies are already installed")
        run([
            str(venv_python), "-c",
            "import torch; "
            "print(f'PyTorch: {torch.__version__}'); "
            "print(f'PyTorch CUDA runtime: {torch.version.cuda}'); "
            "assert torch.cuda.is_available(), "
            "'CUDA build installed, but no GPU is available; check the driver and container access'; "
            "print(f'GPU: {torch.cuda.get_device_name(0)}')",
        ], "Verifying that PyTorch can use the GPU")

        language_models = [
            REPO_ROOT / "lms" / "wiki_en_token.arpa",
            REPO_ROOT / "lms" / "wiki_en_token.arpa.bin",
        ]
        if not any(path.is_file() for path in language_models):
            print(
                "\nWarning: the English n-gram model is absent. The DArtP test "
                "will be reported as skipped; see README.md's N-gram models section.",
                file=sys.stderr,
            )

        run([
            str(venv_python), "-m", "pytest",
            "tests/test_evaluators.py::TestEvaluatorMethods::test_articulatory_precision",
            "tests/test_evaluators.py::TestEvaluatorMethods::test_artp_double_asr", "-v",
        ], "Running the ArtP and DArtP smoke tests", cwd=REPO_ROOT)
    except CommandError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return error.returncode
    except RuntimeError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
