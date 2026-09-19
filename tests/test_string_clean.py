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


@pytest.mark.skipif(
    os.environ.get("PATHBENCH_TEST_PINNED_ESPEAK") != "1",
    reason="requires the README-pinned espeak-ng commit",
)
def test_pinned_espeak_preserves_language_specific_ipa():
    """The documented espeak-ng revision uses an Italian tap, unlike English."""
    cached_phonemize.cache_clear()

    assert cached_phonemize("Roma", "it").strip() == "ɾ o m a"
    assert cached_phonemize("Roma", "en-us").strip() == "ɹ oʊ m ə"
