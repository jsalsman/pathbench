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
import io
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
import binascii
import re
import struct
import zlib

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_NAMES = ("wiki_en_token.arpa.bin", "wiki_en_token.arpa")
# The versioned record URL is deliberately not the mutable ``latest`` link.
LANGUAGE_MODEL_URL = (
    "https://zenodo.org/api/records/18738598/files/lms.zip/content"
)
# The MD5 describes the complete archive.  Range extraction cannot verify it.
LANGUAGE_MODEL_ARCHIVE_SIZE = 35_017_940_434
LANGUAGE_MODEL_ARCHIVE_MD5 = "01d62027902e93270e5f0d00806c473c"
LANGUAGE_MODEL_MEMBER = "lms/wiki_en_token.arpa.bin"
LANGUAGE_MODEL_SIZE = 14_600_342_241
LANGUAGE_MODEL_COMPRESSED_SIZE = 8_582_666_912
LANGUAGE_MODEL_CRC32 = 0x5AFB90EF
# Independently calculated from the extracted English model (not the ZIP bytes).
LANGUAGE_MODEL_SHA256 = "8c5f43d9758f1af5b36740b45957d78690a7e712686270981d4f8db2262e74f7"
USER_AGENT = "PathBench GPU predictor model installer/1.0"
RANGE_BLOCK_SIZE = 1024 * 1024
RANGE_RETRIES = 4


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


class HTTPRangeClient:
    """Strict, retrying HTTP byte-range client for a pinned immutable object."""

    def __init__(self, url: str, size: int, *, retries: int = RANGE_RETRIES) -> None:
        self.url = url
        self.size = size
        self.retries = retries

    def _open(self, start: int, end: int):
        if start < 0 or end < start or end >= self.size:
            raise RuntimeError(f"Invalid archive byte range {start}-{end}")
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
        )
        response = urllib.request.urlopen(request, timeout=60)
        status = getattr(response, "status", None) or response.getcode()
        if status != 206:
            response.close()
            raise RuntimeError(
                f"Range server returned HTTP {status}, not 206; refusing a possible "
                f"full {self.size:,}-byte archive response"
            )
        value = response.headers.get("Content-Range")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value or "")
        if not match:
            response.close()
            raise RuntimeError(f"Malformed or missing Content-Range: {value!r}")
        reported = tuple(map(int, match.groups()))
        if reported != (start, end, self.size):
            response.close()
            raise RuntimeError(
                f"Unexpected Content-Range {value!r}; expected bytes "
                f"{start}-{end}/{self.size}"
            )
        return response

    @staticmethod
    def _transient(error: BaseException) -> bool:
        if isinstance(error, RuntimeError):
            return False
        if isinstance(error, urllib.error.HTTPError):
            return error.code in {408, 425, 429, 500, 502, 503, 504}
        return isinstance(error, (OSError, http.client.HTTPException,
                                  urllib.error.URLError))

    def read(self, start: int, end: int) -> bytes:
        for attempt in range(self.retries + 1):
            try:
                with self._open(start, end) as response:
                    data = response.read(end - start + 1)
                    if response.read(1) or len(data) != end - start + 1:
                        raise OSError("truncated or overlong HTTP range response")
                    return data
            except Exception as error:
                if not self._transient(error) or attempt == self.retries:
                    raise RuntimeError(f"HTTP range request failed: {error}") from error
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")

    def chunks(self, start: int, end: int, chunk_size: int = 4 * 1024 * 1024):
        """Yield an exact interval, resuming interrupted responses with a new range."""
        position = start
        failures = 0
        while position <= end:
            response = None
            try:
                response = self._open(position, end)
                while position <= end:
                    chunk = response.read(min(chunk_size, end - position + 1))
                    if not chunk:
                        raise OSError("truncated HTTP range response")
                    position += len(chunk)
                    yield chunk
                if response.read(1):
                    raise OSError("overlong HTTP range response")
                failures = 0
            except Exception as error:
                if not self._transient(error) or failures >= self.retries:
                    raise RuntimeError(f"HTTP range download failed: {error}") from error
                time.sleep(min(2 ** failures, 8))
                failures += 1
            finally:
                if response is not None:
                    response.close()


class BufferedHTTPRangeReader(io.RawIOBase):
    """Seekable range-backed file with block coalescing for ZIP metadata reads."""

    def __init__(self, client: HTTPRangeClient, block_size: int = RANGE_BLOCK_SIZE):
        self.client = client
        self.block_size = block_size
        self.position = 0
        self.cache: dict[int, bytes] = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        positions = {os.SEEK_SET: offset, os.SEEK_CUR: self.position + offset,
                     os.SEEK_END: self.client.size + offset}
        if whence not in positions or positions[whence] < 0:
            raise ValueError("invalid seek")
        self.position = positions[whence]
        return self.position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.client.size - self.position
        size = min(size, self.client.size - self.position)
        output = bytearray()
        while size > 0:
            block = self.position // self.block_size
            if block not in self.cache:
                start = block * self.block_size
                end = min(start + self.block_size, self.client.size) - 1
                self.cache[block] = self.client.read(start, end)
            data = self.cache[block]
            within = self.position - block * self.block_size
            take = min(size, len(data) - within)
            output.extend(data[within:within + take])
            self.position += take
            size -= take
        return bytes(output)


def _zip64_values(extra: bytes) -> list[int]:
    position = 0
    while position + 4 <= len(extra):
        kind, length = struct.unpack_from("<HH", extra, position)
        value = extra[position + 4:position + 4 + length]
        if len(value) != length:
            raise RuntimeError("Truncated ZIP extra field")
        if kind == 1:
            if length % 8:
                raise RuntimeError("Malformed ZIP64 extra field")
            return list(struct.unpack(f"<{length // 8}Q", value))
        position += 4 + length
    return []


def inspect_zenodo_member(client: HTTPRangeClient) -> tuple[zipfile.ZipInfo, int, int]:
    """Read ZIP/ZIP64 metadata remotely and return the raw DEFLATE interval."""
    reader = BufferedHTTPRangeReader(client)
    try:
        with zipfile.ZipFile(reader) as archive:
            info = archive.getinfo(LANGUAGE_MODEL_MEMBER)
    except (KeyError, zipfile.BadZipFile, OSError) as error:
        raise RuntimeError(f"Cannot inspect Zenodo ZIP metadata: {error}") from error
    if not _safe_member(info.filename) or info.filename != LANGUAGE_MODEL_MEMBER:
        raise RuntimeError(f"Unsafe or unexpected ZIP member name: {info.filename!r}")
    expected = (zipfile.ZIP_DEFLATED, LANGUAGE_MODEL_SIZE,
                LANGUAGE_MODEL_COMPRESSED_SIZE, LANGUAGE_MODEL_CRC32)
    actual = (info.compress_type, info.file_size, info.compress_size, info.CRC)
    if actual != expected:
        raise RuntimeError(f"Zenodo member metadata changed: {actual!r} != {expected!r}")
    if info.flag_bits & 1:
        raise RuntimeError("Encrypted ZIP members are unsupported")

    fixed = client.read(info.header_offset, info.header_offset + 29)
    signature, _version, flags, method, _time, _date, crc, compressed, expanded, name_len, extra_len = \
        struct.unpack("<IHHHHHIIIHH", fixed)
    if signature != 0x04034B50:
        raise RuntimeError("Invalid ZIP local-header signature")
    variable = client.read(info.header_offset + 30,
                           info.header_offset + 29 + name_len + extra_len)
    raw_name, extra = variable[:name_len], variable[name_len:]
    encoding = "utf-8" if flags & 0x800 else "cp437"
    try:
        local_name = raw_name.decode(encoding)
    except UnicodeDecodeError as error:
        raise RuntimeError("Invalid ZIP local-header filename") from error
    if local_name != info.filename or method != info.compress_type or flags != info.flag_bits:
        raise RuntimeError("ZIP local and central headers are inconsistent")
    if method != zipfile.ZIP_DEFLATED or flags & 1:
        raise RuntimeError("Unsupported or encrypted ZIP member")
    if not flags & 8:
        values = iter(_zip64_values(extra))
        local_expanded = next(values, None) if expanded == 0xFFFFFFFF else expanded
        local_compressed = next(values, None) if compressed == 0xFFFFFFFF else compressed
        if (crc, local_compressed, local_expanded) != (info.CRC, info.compress_size, info.file_size):
            raise RuntimeError("ZIP local and central size/CRC metadata are inconsistent")
    elif crc not in (0, info.CRC) or compressed not in (0, 0xFFFFFFFF, info.compress_size) \
            or expanded not in (0, 0xFFFFFFFF, info.file_size):
        raise RuntimeError("ZIP data-descriptor placeholders are inconsistent")
    start = info.header_offset + 30 + name_len + extra_len
    return info, start, start + info.compress_size - 1


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


def _check_model_space(directory: Path, required_size: int = LANGUAGE_MODEL_SIZE) -> None:
    """Require room for the staged model plus a conservative one-GiB margin."""
    available = shutil.disk_usage(directory).free
    required = required_size + 1024 ** 3
    if available < required:
        raise RuntimeError(
            f"Insufficient disk space: {available:,} bytes free; {required:,} required "
            "for the temporary expanded model and safety margin"
        )


def _expand_deflate_range(
    client: HTTPRangeClient, start: int, end: int, destination: Path, *,
    compressed_size: int = LANGUAGE_MODEL_COMPRESSED_SIZE,
    expanded_size: int = LANGUAGE_MODEL_SIZE,
    expected_crc: int = LANGUAGE_MODEL_CRC32,
    expected_sha256: str = LANGUAGE_MODEL_SHA256,
) -> None:
    """Download and authenticate one raw-DEFLATE ZIP member."""
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    transferred = expanded = crc = 0
    digest = hashlib.sha256()
    last_report = time.monotonic()
    with destination.open("wb") as output:
        for chunk in client.chunks(start, end):
            transferred += len(chunk)
            if transferred > compressed_size:
                raise RuntimeError("Compressed member exceeds its advertised size")
            data = inflater.decompress(chunk, expanded_size - expanded + 1)
            if inflater.unconsumed_tail:
                raise RuntimeError("Expanded model exceeds its advertised size")
            expanded += len(data)
            if expanded > expanded_size:
                raise RuntimeError("Expanded model exceeds its advertised size")
            output.write(data)
            crc = binascii.crc32(data, crc)
            digest.update(data)
            if time.monotonic() - last_report >= 5:
                print(f"Transferred {transferred:,}/{compressed_size:,}; expanded "
                      f"{expanded:,}/{expanded_size:,} bytes...", flush=True)
                last_report = time.monotonic()
        tail = inflater.flush()
        expanded += len(tail)
        if expanded > expanded_size:
            raise RuntimeError("Expanded model exceeds its advertised size")
        output.write(tail)
        crc = binascii.crc32(tail, crc)
        digest.update(tail)
    print(f"Transferred {transferred:,}/{compressed_size:,}; expanded "
          f"{expanded:,}/{expanded_size:,} bytes.", flush=True)
    if transferred != compressed_size:
        raise RuntimeError(f"Compressed size mismatch: {transferred:,} != {compressed_size:,}")
    if not inflater.eof or inflater.unused_data:
        raise RuntimeError("Truncated or trailing compressed member data")
    if expanded != expanded_size:
        raise RuntimeError(f"Expanded size mismatch: {expanded:,} != {expanded_size:,}")
    if crc & 0xFFFFFFFF != expected_crc:
        raise RuntimeError(f"Model CRC-32 mismatch: {crc & 0xFFFFFFFF:08x}")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256.lower():
        raise RuntimeError(
            f"Model SHA-256 mismatch (expected {expected_sha256}, got {actual_sha256})"
        )


def install_zenodo_language_model(
    *, force: bool = False, validator=None,
) -> Path:
    """Range-extract, validate, and atomically install the pinned ZIP member."""
    models_dir = REPO_ROOT / "lms"
    models_dir.mkdir(parents=True, exist_ok=True)
    installed = models_dir / Path(LANGUAGE_MODEL_MEMBER).name
    if not force and installed.is_file() and installed.stat().st_size == LANGUAGE_MODEL_SIZE \
            and sha256(installed) == LANGUAGE_MODEL_SHA256:
        print(f"Reusing verified installed language model: {installed}")
        return installed
    print(
        "The Zenodo range download will transfer approximately 8.0 GiB and install "
        "a 13.6 GiB English language model. Additional temporary space and a 1 GiB "
        "safety margin are required.", flush=True,
    )
    _check_model_space(models_dir)
    client = HTTPRangeClient(LANGUAGE_MODEL_URL, LANGUAGE_MODEL_ARCHIVE_SIZE)
    _info, start, end = inspect_zenodo_member(client)
    temporary = Path(tempfile.mkstemp(prefix=".wiki-en-", dir=models_dir)[1])
    try:
        _expand_deflate_range(client, start, end, temporary)
        if validator is not None:
            validator(temporary)
        os.replace(temporary, installed)
        return installed
    finally:
        temporary.unlink(missing_ok=True)


def install_language_model(
    url: str, expected_sha256: str, cache_dir: Path, *, force: bool = False,
    validator=None,
) -> Path:
    """Install a custom standalone file/archive with a user-supplied digest."""
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
        if validator is not None:
            validator(extracted)
        installed = models_dir / extracted.name
        if installed.name == "wiki_en_token.arpa":
            # The evaluator prefers the binary whenever it exists. Removing a
            # stale binary ensures it actually consumes the verified ARPA we
            # are about to install and validate.
            (models_dir / "wiki_en_token.arpa.bin").unlink(missing_ok=True)
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
                # The evaluator always prefers this binary when it exists. Do
                # not fall back to an ARPA that the smoke test would not use;
                # make the opted-in preparation path replace the bad binary.
                return None
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


def prepare_language_model(
    *,
    download: bool,
    url: str,
    expected_sha256: str,
    cache_dir: Path,
    force: bool,
    installed_sha256: str | None,
    validator=None,
) -> Path | None:
    """Reuse an installed model, unless an opted-in forced refresh was requested."""
    model = installed_language_model(installed_sha256)
    if download and (model is None or force):
        if url == LANGUAGE_MODEL_URL:
            return install_zenodo_language_model(force=force, validator=validator)
        kwargs = {"force": force}
        if validator is not None:
            kwargs["validator"] = validator
        return install_language_model(url, expected_sha256, cache_dir, **kwargs)
    return model


def installed_model_digest(
    *, download: bool, url: str, expected_sha256: str | None,
) -> str | None:
    """Return a digest only when this run is managing the pinned artifact.

    Manually installed models are supported and need not be byte-for-byte
    identical to the project's default Zenodo binary. They are checked by
    KenLM's parser instead of against the download artifact's digest.
    """
    if download and url == LANGUAGE_MODEL_URL:
        return expected_sha256
    return None


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
        known_installed_digest = installed_model_digest(
            download=args.download_language_model,
            url=args.language_model_url,
            expected_sha256=expected_digest,
        )
        model = prepare_language_model(
            download=args.download_language_model,
            url=args.language_model_url,
            expected_sha256=expected_digest or "",
            cache_dir=args.language_model_cache,
            force=args.force_language_model_download,
            installed_sha256=known_installed_digest,
            validator=lambda path: validate_language_model(venv_python, path),
        )
        if model is None:
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
