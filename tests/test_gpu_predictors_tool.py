"""Network-free tests for the opt-in GPU smoke-test model downloader."""

from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import urllib.error
import zipfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "test_gpu_predictors", Path(__file__).parents[1] / "tools/test_gpu_predictors.py"
)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def zip_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return output.getvalue()


def configure(monkeypatch, tmp_path: Path, payload: bytes):
    calls = []
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(
        tool.urllib.request, "urlopen",
        lambda *_args, **_kwargs: calls.append(True) or Response(payload),
    )
    return calls, hashlib.sha256(payload).hexdigest()


def test_successful_download_prefers_binary_and_reuses_cache(monkeypatch, tmp_path):
    payload = zip_bytes({"models/wiki_en_token.arpa": b"text", "wiki_en_token.arpa.bin": b"binary"})
    calls, digest = configure(monkeypatch, tmp_path, payload)
    first = tool.install_language_model("https://example/model.zip", digest, tmp_path / "cache")
    second = tool.install_language_model("https://example/model.zip", digest, tmp_path / "cache")
    assert first.read_bytes() == second.read_bytes() == b"binary"
    assert len(calls) == 1


def test_checksum_mismatch_removes_partial(monkeypatch, tmp_path):
    calls, _ = configure(monkeypatch, tmp_path, b"not expected")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        tool.install_language_model("https://example/model", "0" * 64, tmp_path / "cache")
    assert calls and not list((tmp_path / "cache").iterdir())


def test_interrupted_download_removes_partial(monkeypatch, tmp_path):
    class Interrupted(Response):
        def read(self, *_args):
            raise OSError("connection reset")

    monkeypatch.setattr(tool.urllib.request, "urlopen", lambda *_a, **_k: Interrupted(b"x"))
    with pytest.raises(RuntimeError, match="download failed"):
        tool.install_language_model("https://example/model", "0" * 64, tmp_path / "cache")
    assert not list((tmp_path / "cache").iterdir())


def test_http_failure(monkeypatch, tmp_path):
    def fail(*_args, **_kwargs):
        raise urllib.error.HTTPError("url", 503, "unavailable", {}, None)

    monkeypatch.setattr(tool.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="503"):
        tool.install_language_model("https://example/model", "0" * 64, tmp_path / "cache")


def test_safe_archive_extracts_only_model(tmp_path):
    archive = tmp_path / "models.zip"
    archive.write_bytes(zip_bytes({"docs/readme": b"no", "nested/wiki_en_token.arpa": b"yes"}))
    output = tmp_path / "out"
    output.mkdir()
    assert tool._extract_model(archive, output).read_bytes() == b"yes"
    assert sorted(path.name for path in output.iterdir()) == ["wiki_en_token.arpa"]


@pytest.mark.parametrize("name", ["../wiki_en_token.arpa", "/tmp/wiki_en_token.arpa", "C:\\..\\wiki_en_token.arpa"])
def test_malicious_archive_path_is_rejected(tmp_path, name):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(zip_bytes({name: b"bad"}))
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(RuntimeError, match="unsafe"):
        tool._extract_model(archive, output)


def test_corrupt_kenlm_data_is_reported(monkeypatch, tmp_path):
    model = tmp_path / "wiki_en_token.arpa.bin"
    model.write_bytes(b"corrupt")
    monkeypatch.setattr(tool.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(
        tool.subprocess.CalledProcessError(1, "python")
    ))
    with pytest.raises(tool.CommandError, match="Validating the English KenLM model failed"):
        tool.validate_language_model(Path("python"), model)


def test_already_installed_verified_model(monkeypatch, tmp_path):
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    model = tmp_path / "lms" / "wiki_en_token.arpa.bin"
    model.parent.mkdir()
    model.write_bytes(b"model")
    assert tool.installed_language_model(hashlib.sha256(b"model").hexdigest()) == model
    assert tool.installed_language_model("0" * 64) is None


def test_forced_download_replaces_an_installed_model(monkeypatch, tmp_path):
    installed = tmp_path / "lms" / "wiki_en_token.arpa.bin"
    refreshed = tmp_path / "refreshed" / "wiki_en_token.arpa.bin"
    monkeypatch.setattr(tool, "installed_language_model", lambda _digest: installed)
    calls = []

    def install(url, digest, cache, *, force):
        calls.append((url, digest, cache, force))
        return refreshed

    monkeypatch.setattr(tool, "install_language_model", install)
    result = tool.prepare_language_model(
        download=True,
        url="https://example/custom-model",
        expected_sha256="1" * 64,
        cache_dir=tmp_path / "cache",
        force=True,
        installed_sha256=None,
    )

    assert result == refreshed
    assert calls == [
        ("https://example/custom-model", "1" * 64, tmp_path / "cache", True)
    ]


def test_force_without_download_remains_opted_out(monkeypatch, tmp_path):
    installed = tmp_path / "lms" / "wiki_en_token.arpa.bin"
    monkeypatch.setattr(tool, "installed_language_model", lambda _digest: installed)
    monkeypatch.setattr(
        tool,
        "install_language_model",
        lambda *_args, **_kwargs: pytest.fail("download must remain opt-in"),
    )

    assert tool.prepare_language_model(
        download=False,
        url="https://example/custom-model",
        expected_sha256="1" * 64,
        cache_dir=tmp_path / "cache",
        force=True,
        installed_sha256=None,
    ) == installed


def test_download_remains_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tool"])
    assert tool.parse_args().download_language_model is False
