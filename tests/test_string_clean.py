"""Tests for text cleaning and the pinned phonemizer API."""

import importlib.util
import os
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "pathbench_string_clean", Path(__file__).parents[1] / "pathbench" / "string_clean.py"
)
assert _SPEC and _SPEC.loader
_STRING_CLEAN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STRING_CLEAN)
cached_phonemize = _STRING_CLEAN.cached_phonemize


def test_phonemizer_fork_exposes_expected_api():
    """The PyPI fork must retain the imports used by ``string_clean``."""
    from phonemizer.phonemize import phonemize
    from phonemizer.separator import Separator

    assert callable(phonemize)
    assert Separator(phone=" ", word="|").phone == " "


def test_phonemizer_reuses_only_matching_espeak_configuration(monkeypatch):
    """Reuse an espeak backend without leaking state across languages.

    This is the minimal sequence for the cache added by ``phonemizer-fork``
    3.3.2: upstream/``phonemizer-fork`` 3.3.1 constructs three backends for
    these calls, while 3.3.2 constructs two and reuses the original English
    backend after the intervening Italian call.  Checking every result guards
    against a stale backend being reused for a different configuration.

    The fake backend keeps this Python cache regression independent of the
    separately opt-in espeak-ng revision test below.
    """
    phonemize_module = importlib.import_module("phonemizer.phonemize")

    created_languages = []

    class FakeEspeakBackend:
        def __init__(self, language, **_kwargs):
            self.language = language
            created_languages.append(language)

        def phonemize(self, text, **_kwargs):
            return [f"{self.language}:{item}" for item in text]

    monkeypatch.setitem(phonemize_module.BACKENDS, "espeak", FakeEspeakBackend)
    backend_cache = getattr(phonemize_module, "_PHONEMIZER_CACHE", None)
    if backend_cache is not None:
        backend_cache.clear()
    cached_phonemize.cache_clear()

    assert cached_phonemize("first", "en-us") == "en-us:first"
    assert cached_phonemize("second", "it") == "it:second"
    assert cached_phonemize("third", "en-us") == "en-us:third"
    assert created_languages == ["en-us", "it"]


@pytest.mark.skipif(
    os.environ.get("PATHBENCH_TEST_PINNED_ESPEAK") != "1",
    reason="requires the README-pinned espeak-ng commit",
)
def test_pinned_espeak_preserves_language_specific_ipa():
    """The documented espeak-ng revision uses an Italian tap, unlike English."""
    cached_phonemize.cache_clear()

    assert cached_phonemize("Roma", "it").strip() == "ɾ o m a"
    assert cached_phonemize("Roma", "en-us").strip() == "ɹ oʊ m ə"
