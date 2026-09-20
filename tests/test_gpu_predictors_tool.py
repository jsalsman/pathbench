"""Network-free tests for the opt-in GPU smoke-test model downloader."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import urllib.error
import zipfile
import zlib

import pytest


SPEC = importlib.util.spec_from_file_location(
    "test_gpu_predictors", Path(__file__).parents[1] / "tools/test_gpu_predictors.py"
)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def test_builtin_language_model_checksum_matches_published_value():
    assert tool.LANGUAGE_MODEL_SHA256 == (
        "d786eec55174c696c0bf3327928ff496684f482194ba3c6ebdf4311acb823d00"
    )


class Response(io.BytesIO):
    status = 206

    def __init__(self, value=b"", headers=None, status=206):
        super().__init__(value)
        self.headers = headers or {}
        self.status = status

    def getcode(self):
        return self.status

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


def test_installing_arpa_removes_stale_preferred_binary(monkeypatch, tmp_path):
    payload = zip_bytes({"models/wiki_en_token.arpa": b"replacement arpa"})
    _calls, digest = configure(monkeypatch, tmp_path, payload)
    stale_binary = tmp_path / "repo" / "lms" / "wiki_en_token.arpa.bin"
    stale_binary.parent.mkdir(parents=True)
    stale_binary.write_bytes(b"stale binary")

    installed = tool.install_language_model(
        "https://example/model.zip", digest, tmp_path / "cache",
    )

    assert installed.name == "wiki_en_token.arpa"
    assert installed.read_bytes() == b"replacement arpa"
    assert not stale_binary.exists()


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


def test_rejected_preferred_binary_blocks_arpa_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    models = tmp_path / "lms"
    models.mkdir()
    (models / "wiki_en_token.arpa.bin").write_bytes(b"rejected binary")
    (models / "wiki_en_token.arpa").write_bytes(b"otherwise valid arpa")

    assert tool.installed_language_model("0" * 64) is None


def test_rejected_preferred_binary_is_replaced(monkeypatch, tmp_path):
    rejected = tmp_path / "lms" / "wiki_en_token.arpa.bin"
    fallback = tmp_path / "lms" / "wiki_en_token.arpa"
    replacement = tmp_path / "replacement" / "wiki_en_token.arpa.bin"
    rejected.parent.mkdir()
    rejected.write_bytes(b"rejected binary")
    fallback.write_bytes(b"otherwise valid arpa")
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        tool,
        "install_zenodo_language_model",
        lambda **_kwargs: replacement,
    )

    assert tool.prepare_language_model(
        download=True,
        url=tool.LANGUAGE_MODEL_URL,
        expected_sha256="0" * 64,
        cache_dir=tmp_path / "cache",
        force=False,
        installed_sha256="0" * 64,
    ) == replacement


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


def test_manual_binary_is_not_compared_with_download_digest():
    assert tool.installed_model_digest(
        download=False,
        url=tool.LANGUAGE_MODEL_URL,
        expected_sha256=tool.LANGUAGE_MODEL_SHA256,
    ) is None


def test_managed_builtin_download_verifies_installed_binary():
    assert tool.installed_model_digest(
        download=True,
        url=tool.LANGUAGE_MODEL_URL,
        expected_sha256=tool.LANGUAGE_MODEL_SHA256,
    ) == tool.LANGUAGE_MODEL_SHA256


def test_download_remains_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tool"])
    assert tool.parse_args().download_language_model is False


def test_range_reader_validates_content_range_and_coalesces(monkeypatch):
    payload = bytes(range(256)) * 10000
    calls = []

    def open_range(request, **_kwargs):
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        calls.append((start, end))
        return Response(payload[start:end + 1], {
            "Content-Range": f"bytes {start}-{end}/{len(payload)}"
        })

    monkeypatch.setattr(tool.urllib.request, "urlopen", open_range)
    reader = tool.BufferedHTTPRangeReader(
        tool.HTTPRangeClient("https://example/archive", len(payload)), 1024 * 1024
    )
    reader.seek(17)
    assert reader.read(5) == payload[17:22]
    reader.seek(800_000)
    assert reader.read(5) == payload[800_000:800_005]
    assert len(calls) == 1


@pytest.mark.parametrize("header", [None, "bytes 0-8/10", "nonsense"])
def test_malformed_content_range_is_rejected(monkeypatch, header):
    monkeypatch.setattr(tool.urllib.request, "urlopen", lambda *_a, **_k: Response(
        b"0123456789", {"Content-Range": header} if header else {}
    ))
    with pytest.raises(RuntimeError, match="Content-Range"):
        tool.HTTPRangeClient("https://example/archive", 10).read(0, 9)


def test_http_200_range_response_is_rejected(monkeypatch):
    monkeypatch.setattr(tool.urllib.request, "urlopen", lambda *_a, **_k: Response(
        b"whole archive", status=200
    ))
    with pytest.raises(RuntimeError, match="HTTP 200"):
        tool.HTTPRangeClient("https://example/archive", 100).read(0, 9)


def _compressed_case(data: bytes):
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    compressed = compressor.compress(data) + compressor.flush()

    class Client:
        def chunks(self, _start, _end):
            yield compressed

    return Client(), compressed


def test_raw_deflate_member_download(tmp_path):
    data = b"ordinary ZIP member" * 100
    client, compressed = _compressed_case(data)
    output = tmp_path / "model"
    tool._expand_deflate_range(
        client, 0, len(compressed) - 1, output,
        compressed_size=len(compressed), expanded_size=len(data),
        expected_crc=zlib.crc32(data), expected_sha256=hashlib.sha256(data).hexdigest(),
    )
    assert output.read_bytes() == data


@pytest.mark.parametrize("failure", ["truncated", "overlong", "crc", "sha"])
def test_corrupt_member_downloads_are_rejected(tmp_path, failure):
    data = b"model data" * 100
    client, compressed = _compressed_case(data)
    kwargs = dict(compressed_size=len(compressed), expanded_size=len(data),
                  expected_crc=zlib.crc32(data),
                  expected_sha256=hashlib.sha256(data).hexdigest())
    if failure == "truncated":
        kwargs["compressed_size"] += 1
    elif failure == "overlong":
        kwargs["expanded_size"] -= 1
    elif failure == "crc":
        kwargs["expected_crc"] ^= 1
    else:
        kwargs["expected_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        tool._expand_deflate_range(client, 0, len(compressed) - 1,
                                   tmp_path / failure, **kwargs)


def test_zip64_extra_supports_large_sizes():
    large_expanded, large_compressed = 6_000_000_000, 5_000_000_000
    extra = b"\x01\x00\x10\x00" + large_expanded.to_bytes(8, "little") \
        + large_compressed.to_bytes(8, "little")
    assert tool._zip64_values(extra) == [large_expanded, large_compressed]


def test_zip_metadata_derives_member_bounds_and_checks_local_header(monkeypatch):
    data = b"member" * 100
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(tool.LANGUAGE_MODEL_MEMBER, data)
    payload = bytearray(output.getvalue())

    class MemoryClient:
        size = len(payload)

        def read(self, start, end):
            return bytes(payload[start:end + 1])

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        expected = archive.getinfo(tool.LANGUAGE_MODEL_MEMBER)
    monkeypatch.setattr(tool, "LANGUAGE_MODEL_SIZE", len(data))
    monkeypatch.setattr(tool, "LANGUAGE_MODEL_COMPRESSED_SIZE", expected.compress_size)
    monkeypatch.setattr(tool, "LANGUAGE_MODEL_CRC32", expected.CRC)
    info, start, end = tool.inspect_zenodo_member(MemoryClient())
    assert end - start + 1 == info.compress_size

    payload[info.header_offset + 30] ^= 1
    with pytest.raises(RuntimeError, match="local and central"):
        tool.inspect_zenodo_member(MemoryClient())


def test_insufficient_disk_space(monkeypatch, tmp_path):
    usage = type("Usage", (), {"free": 1})()
    monkeypatch.setattr(tool.shutil, "disk_usage", lambda _path: usage)
    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        tool._check_model_space(tmp_path)


def test_retry_exhaustion(monkeypatch):
    calls = []
    monkeypatch.setattr(tool.time, "sleep", lambda _seconds: None)

    def fail(*_args, **_kwargs):
        calls.append(1)
        raise urllib.error.URLError("interrupted")

    monkeypatch.setattr(tool.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="range request failed"):
        tool.HTTPRangeClient("https://example/archive", 10, retries=2).read(0, 9)
    assert len(calls) == 3


@pytest.mark.skipif(not os.environ.get("PATHBENCH_LIVE_ZENODO_METADATA"),
                    reason="set PATHBENCH_LIVE_ZENODO_METADATA=1 for live range check")
def test_live_zenodo_metadata():
    client = tool.HTTPRangeClient(tool.LANGUAGE_MODEL_URL,
                                  tool.LANGUAGE_MODEL_ARCHIVE_SIZE)
    info, start, end = tool.inspect_zenodo_member(client)
    assert info.filename == tool.LANGUAGE_MODEL_MEMBER
    assert end - start + 1 == tool.LANGUAGE_MODEL_COMPRESSED_SIZE
    assert (start, end) == (18_177_078_384, 26_759_745_295)
