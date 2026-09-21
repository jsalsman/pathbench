"""Network-free tests for notebook bootstrap checks."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import colab_bootstrap as bootstrap


def test_verify_blue_recordings(tmp_path):
    expected = {}
    for name in bootstrap.BLUE_RECORDINGS:
        value = name.encode()
        (tmp_path / name).write_bytes(value)
        expected[name] = hashlib.sha256(value).hexdigest()
    original = bootstrap.BLUE_RECORDINGS
    bootstrap.BLUE_RECORDINGS = expected
    try:
        assert bootstrap.verify_blue_recordings(tmp_path) == expected
        (tmp_path / next(iter(expected))).write_bytes(b"changed")
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            bootstrap.verify_blue_recordings(tmp_path)
    finally:
        bootstrap.BLUE_RECORDINGS = original


def test_disk_space_reports_and_rejects(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap.shutil, "disk_usage", lambda _: SimpleNamespace(free=200))
    assert bootstrap.disk_space(tmp_path, 100, 50) == (150, 200)
    with pytest.raises(RuntimeError, match="Insufficient disk"):
        bootstrap.disk_space(tmp_path, 151, 50)


def test_score_requires_value_and_tolerance():
    assert bootstrap.assert_score("ArtP", 0.06951, 0.0695, 0.00005) == 0.06951
    with pytest.raises(AssertionError, match="returned None"):
        bootstrap.assert_score("ArtP", None, 0.0695, 0.00005)
    with pytest.raises(AssertionError, match="expected"):
        bootstrap.assert_score("DArtP", 0.5, 0.4405, 0.00005)


def test_junit_explicitly_rejects_skips(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text('<testsuite tests="2" failures="0" errors="0" skipped="0"/>')
    bootstrap.assert_junit_no_skips(report)
    report.write_text('<testsuite tests="2" failures="0" errors="0" skipped="1"/>')
    with pytest.raises(AssertionError, match="skipped"):
        bootstrap.assert_junit_no_skips(report)
