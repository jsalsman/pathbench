"""Small, network-free checks shared by the ArtP/DArtP Colab notebook.

Keep downloading and model inference in the notebook and GPU installer; this module
only makes integrity, capacity, numeric, and pytest-result checks testable locally.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

BLUE_RECORDINGS = {
    "BLUE_japanese10.wav": "2f67bec80deee5757978763f603430eec6d24aa461b10026d7c0a4ddc9cafa6f",
    "BLUE_english32.wav": "3c23c4a3947b2d0546103355dbb4d8fc34306ae6f4d868aa1c5a22dfe9ac7e3b",
    "BLUE_english33.wav": "a15ad6aa18a4d0a609726ad80e8cb9602eb6d608635ab24fd93acf9d57ca6b85",
    "BLUE_english34.wav": "6aefd9e4d80ffc3647e9f0ed5904e40d6961dafbd7560365b3a7d90a1992407e",
}
BLUE_LICENSE = "CC0-1.0 (public domain dedication)"
BLUE_PROVENANCE_URL = "https://github.com/Bartelds/neural-acoustic-distance"
TRANSCRIPT = "blue"
ARTP_EXPECTED = 0.0695
DARTP_EXPECTED = 0.4405
SCORE_TOLERANCE = 0.00005  # equivalent to unittest assertAlmostEqual(..., places=4)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_blue_recordings(data_dir: Path) -> dict[str, str]:
    """Return verified fixture digests, raising on a missing/changed recording."""
    actual = {name: file_sha256(data_dir / name) for name in BLUE_RECORDINGS}
    mismatches = {name: value for name, value in actual.items()
                  if value != BLUE_RECORDINGS[name]}
    if mismatches:
        raise RuntimeError(f"BLUE recording SHA-256 mismatch: {mismatches}")
    return actual


def disk_space(path: Path, expanded_bytes: int, margin_bytes: int = 1024**3) -> tuple[int, int]:
    """Return required and available bytes; fail before an expensive download."""
    required = expanded_bytes + margin_bytes
    available = shutil.disk_usage(path).free
    if available < required:
        raise RuntimeError(f"Insufficient disk: {required:,} required, {available:,} available")
    return required, available


def assert_score(name: str, score: float | None, expected: float, tolerance: float) -> float:
    if score is None:
        raise AssertionError(f"{name} skipped/returned None")
    if abs(float(score) - expected) > tolerance:
        raise AssertionError(f"{name}={score}; expected {expected} ± {tolerance}")
    return float(score)


def assert_junit_no_skips(report: Path) -> None:
    """Explicitly reject a focused regression run containing skips or failures."""
    root = ET.parse(report).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {key: sum(int(s.get(key, "0")) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if totals["tests"] < 2 or any(totals[key] for key in ("failures", "errors", "skipped")):
        raise AssertionError(f"Focused regressions did not fully pass: {totals}")
