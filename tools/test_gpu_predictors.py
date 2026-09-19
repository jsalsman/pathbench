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
import hashlib
import http.client
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_NAMES = ("wiki_en_token.arpa.bin", "wiki_en_token.arpa")
# The versioned record URL is deliberately not the mutable ``latest`` link.
LANGUAGE_MODEL_URL = (
    "https://zenodo.org/records/18738598/files/wiki_en_token.arpa.bin?download=1"
)
# SHA-256 published for the English binary in Zenodo record 18738598.
LANGUAGE_MODEL_SHA256 = "8c5f43d9758f1af5b36740b45957d78690a7e712686270981d4f8db2262e74f7"


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
        "--cuda-version", default=os.environ.get("PATHBENCH_CUDA_VERSION", "12.4"),
        help="CUDA wheel version, such as 12.4 (default: 12.4)",
    )
    parser.add_argument(
        "--venv", type=Path,
        default=Path(os.environ.get("VENV", SCRIPT_DIR / "gpu_venv")),
        help="virtual-environment destination (default: tools/gpu_venv)",
    )
    parser.add_argument(
        "--download-language-model", action="store_true",
        help="download and install the English DArtP model (large; opt in)",
    )
    parser.add_argument(
        "--language-model-url",
        default=os.environ.get("PATHBENCH_LANGUAGE_MODEL_URL", LANGUAGE_MODEL_URL),
        help="model/archive URL (PATHBENCH_LANGUAGE_MODEL_URL)",
    )
    parser.add_argument(
        "--language-model-sha256",
        default=os.environ.get("PATHBENCH_LANGUAGE_MODEL_SHA256"),
        help="SHA-256 for a custom URL (PATHBENCH_LANGUAGE_MODEL_SHA256)",
    )
    parser.add_argument(
        "--language-model-cache", type=Path,
        default=Path(os.environ.get("PATHBENCH_LANGUAGE_MODEL_CACHE", "~/.cache/pathbench")),
        help="download cache (PATHBENCH_LANGUAGE_MODEL_CACHE; default: ~/.cache/pathbench)",
    )
    parser.add_argument(
        "--force-language-model-download", action="store_true",
        help="discard a cached artifact and download it again",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> bool:
    path = Path(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def _extract_model(archive: Path, destination: Path) -> Path:
    """Extract only the preferred English model, after checking every path."""
    members: dict[str, object]
    opener: object
    if zipfile.is_zipfile(archive):
        opener = zipfile.ZipFile(archive)
        members = {item.filename: item for item in opener.infolist()}
    elif tarfile.is_tarfile(archive):
        opener = tarfile.open(archive)
        members = {item.name: item for item in opener.getmembers()}
    else:
        # A direct ARPA or KenLM binary needs no extraction.
        target = destination / next(
            (name for name in MODEL_NAMES if name in archive.name), MODEL_NAMES[0]
        )
        shutil.copyfile(archive, target)
        return target
    with opener:
        if any(not _safe_member(name) for name in members):
            raise RuntimeError("Language-model archive contains an unsafe path")
        selected = next(
            (name for wanted in MODEL_NAMES for name in members if Path(name).name == wanted),
            None,
        )
        if selected is None:
            raise RuntimeError("Archive contains neither wiki_en_token.arpa.bin nor wiki_en_token.arpa")
        target = destination / Path(selected).name
        source = opener.open(members[selected]) if isinstance(opener, zipfile.ZipFile) else opener.extractfile(members[selected])
        if source is None:
            raise RuntimeError("Selected language-model archive member is not a file")
        with source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
        return target


def install_language_model(
    url: str, expected_sha256: str, cache_dir: Path, *, force: bool = False
) -> Path:
    """Fetch a verified model artifact into the cache and install its model."""
    if not expected_sha256 or len(expected_sha256) != 64:
        raise RuntimeError("A pinned 64-character SHA-256 is required for the language model")
    expected_sha256 = expected_sha256.lower()
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    source_name = source_name if source_name else "download"
    artifact = cache_dir / f"language-model-{expected_sha256}-{source_name}"
    if force:
        artifact.unlink(missing_ok=True)
    if artifact.exists() and sha256(artifact) != expected_sha256:
        artifact.unlink()
    if not artifact.exists():
        temporary = Path(tempfile.mkstemp(prefix=".language-model-", dir=cache_dir)[1])
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PathBench/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                total = 0
                last_report = time.monotonic()
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    total += len(chunk)
                    if time.monotonic() - last_report >= 5:
                        print(f"Downloaded {total:,} bytes...", flush=True)
                        last_report = time.monotonic()
            print(f"Downloaded {total:,} bytes.", flush=True)
            actual = sha256(temporary)
            if actual != expected_sha256:
                raise RuntimeError(
                    f"Language-model SHA-256 mismatch (expected {expected_sha256}, got {actual})"
                )
            os.replace(temporary, artifact)
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            raise RuntimeError(f"Language-model download failed: {error}") from error
        finally:
            temporary.unlink(missing_ok=True)
    else:
        print(f"Reusing verified language-model download: {artifact}")

    models_dir = REPO_ROOT / "lms"
    models_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".model-", dir=models_dir))
    try:
        extracted = _extract_model(artifact, staging)
        if extracted.stat().st_size == 0:
            raise RuntimeError("Downloaded language model is empty")
        installed = models_dir / extracted.name
        os.replace(extracted, installed)
        return installed
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def installed_language_model(expected_sha256: str | None = None) -> Path | None:
    """Return the preferred installed model, rejecting empty/known-bad files."""
    for name in MODEL_NAMES:
        path = REPO_ROOT / "lms" / name
        if not path.is_file() or path.stat().st_size == 0:
            continue
        # The built-in artifact is the compiled model itself, so its digest also
        # authenticates an already-installed copy. Archive digests do not.
        if expected_sha256 and name == "wiki_en_token.arpa.bin":
            if sha256(path) != expected_sha256.lower():
                continue
        return path
    return None


def validate_language_model(venv_python: Path, model: Path) -> None:
    """Have the target environment parse the model before expensive tests start."""
    run([
        str(venv_python), "-c",
        "import kenlm, pathlib, sys; p=pathlib.Path(sys.argv[1]); "
        "assert p.stat().st_size, 'model is empty'; kenlm.Model(str(p))",
        str(model),
    ], "Validating the English KenLM model")


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

        expected_digest = args.language_model_sha256
        if args.language_model_url == LANGUAGE_MODEL_URL and not expected_digest:
            expected_digest = LANGUAGE_MODEL_SHA256
        known_installed_digest = (
            expected_digest if args.language_model_url == LANGUAGE_MODEL_URL else None
        )
        model = installed_language_model(known_installed_digest)
        if model is None and args.download_language_model:
            model = install_language_model(
                args.language_model_url, expected_digest or "",
                args.language_model_cache, force=args.force_language_model_download,
            )
        elif model is None:
            print(
                "\nWarning: the English n-gram model is absent. The DArtP test "
                "will be reported as skipped; see README.md's N-gram models section.",
                file=sys.stderr,
            )
        if model is not None:
            validate_language_model(venv_python, model)

        pytest_environment = os.environ.copy()
        pytest_environment["MPLBACKEND"] = "Agg"
        print("\n==> Running the ArtP and DArtP smoke tests", flush=True)
        command = [
            str(venv_python), "-m", "pytest",
            "tests/test_evaluators.py::TestEvaluatorMethods::test_articulatory_precision",
            "tests/test_evaluators.py::TestEvaluatorMethods::test_artp_double_asr", "-v",
        ]
        print("+ " + " ".join(command), flush=True)
        try:
            subprocess.run(command, check=True, cwd=REPO_ROOT, env=pytest_environment)
        except subprocess.CalledProcessError as error:
            raise CommandError(
                f"Running the ArtP and DArtP smoke tests failed with exit status "
                f"{error.returncode}. Review the command output above: {' '.join(command)}",
                error.returncode,
            ) from error
    except CommandError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return error.returncode
    except RuntimeError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
