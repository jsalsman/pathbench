# PathBench

<p align="center"><img src="assets/leonberger_transparent.png" width="150" /></p>

[![Unit Tests](https://github.com/karkirowle/pathbench/actions/workflows/tests.yml/badge.svg)](https://github.com/karkirowle/pathbench/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/path-benc/badge/?version=latest)](https://path-benc.readthedocs.io/en/latest/)
![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue)

PathBench is a benchmark designed to evaluate pathological speech assessment systems.

## Results

Speaker-level Pearson Correlation Coefficients (PCC) results can be found below.
Currently ArtP is the best reference-based, and DArtP is the best reference-free method.

* **CSV** (canonical, in this repo): [`results/results_table.csv`](results/results_table.csv) (RTX 3090) and [`results/results_table_2080ti.csv`](results/results_table_2080ti.csv) (RTX 2080 Ti). The two are the same benchmark on different GPUs; small differences (≤0.01) are due to GPU floating-point non-determinism.
* **Google docs** [Google Docs](https://docs.google.com/spreadsheets/d/1ri-y_bHPgED3jJLuonChuwpaSP_3ddrx/edit?usp=sharing&ouid=112094007551573667400&rtpof=true&sd=true)


## Usage guide

There are several use cases for PathBench:

* [I want to evaluate my newly developed predictor](#i-want-to-evaluate-my-newly-developed-predictor)

* [I want to use the predictors developed by you](#i-want-to-use-the-predictors-developed-by-you)

* [I want to contribute a new predictor to this repository, how do I do that?](#i-want-to-contribute-a-new-predictor-to-this-repository-how-do-i-do-that)

* [I want to reproduce your research](#i-want-to-reproduce-your-research)

### I want to evaluate my newly developed predictor

No install needed beyond numpy. Provide a CSV of predicted speaker scores and compare against the PathBench ground truth.


**Single dataset:**
```bash
python scripts/evaluate_from_csv.py \
    --predictions results/datasets/copas/pathological/word/balanced/dummy_scores.csv \
    --ground-truth datasets/copas/pathological/word/balanced/spk2score
```

**Full benchmark** — evaluate one evaluator across all datasets. Place your CSVs in a results directory that mirrors the dataset structure, with each CSV named `<evaluator>.csv`. Dummy score files are provided as a worked example:

```
results/datasets/
  copas/pathological/word/balanced/dummy_scores.csv
  torgo/pathological/utterances/balanced/dummy_scores.csv
  youtube/dummy_scores.csv
```

Then run:
```bash
python scripts/evaluate_from_csv.py \
    --results-dir results/datasets/ \
    --datasets-root datasets/ \
    --evaluator dummy_scores
```

This prints a table with the Pearson correlation for each dataset and the mean across all datasets.

Expected CSV format (speaker IDs must match the ground truth exactly):
```
speaker_id,score
C16,15.61
C17,16.74
```

All speaker IDs in the CSV must match the ground truth exactly — the script exits with an error if any are missing from either side.

If it works well, you might want to consider opening a pull request for your evaluator and incorporating into a codebase along the contribution
guidelines, as this is the only way we can ensure that you are not "cheating", i.e., using resources that are not allowed for your evaluator.

### I want to use the predictors developed by you

Follow the steps in the [Installation](#installation) section. Then you can run inference on individual audio files. For example, to score a single utterance with ArtP:

```python
from pathbench import ArticulatoryPrecisionEvaluator

evaluator = ArticulatoryPrecisionEvaluator()
score = evaluator.score("utt1", "/path/to/audio.wav", transcription="the cat sat on the mat", language="en")
print(f"ArtP score: {score}")
```

ArtP is reference-based: it force-aligns the phonemes in the supplied transcription
with the audio, so the transcription and its language are required. DArtP is
reference-free: a language-specific ASR model and n-gram language model first
produce a transcription, which is then scored by the same phonetic model:

```python
from pathbench import ArtPDoubleASREvaluator

evaluator = ArtPDoubleASREvaluator(language="en-us")
score = evaluator.score("utt1", "/path/to/audio.wav")
print(f"DArtP score: {score}")
```

DArtP currently supports `en`/`en-us`, `es`, `nl`, `it`, and `cmn`. It also
requires the corresponding file from the [n-gram model download](#n-gram-models)
to be placed in `lms/`; without it, scoring returns `None`. Run commands from
the repository root because DArtP resolves `lms/` relative to the working
directory.

### I want to contribute a new predictor to this repository, how do I do that?

See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide.


### I want to reproduce your research

1. Follow the steps in the [Installation](#installation) section.
2. Follow the steps in the [Downloads](#downloads) section.
3. Follow the steps in the [Testing](#testing) section.
4. Run the evaluation script on the dataset(s) you want to evaluate. For example, to evaluate on the YouTube dataset:

```bash
python scripts/evaluate_spk2score.py datasets/youtube
```

To also write per-evaluator score CSVs, pass `--results-dir`:

```bash
python scripts/evaluate_spk2score.py datasets/youtube --results-dir results/datasets/
```

You can evaluate multiple datasets in a single run:

```bash
python scripts/evaluate_spk2score.py \
    datasets/copas/pathological/word/balanced \
    datasets/torgo/pathological/utterances/balanced \
    datasets/youtube
```

Results are written to the `results_11/` directory as timestamped text files containing per-evaluator Pearson correlations and a summary table.


## Installation

We are continuously trying to make the installation easier for your use case.

### Complete GPU installation (install missing components only)

The following Ubuntu procedure is safe to re-run: it installs only absent apt
packages, builds the pinned `espeak-ng` unless its commit marker matches,
clones PathBench only when the checkout is absent, and lets the GPU helper reuse
an existing virtual environment and matching Python packages.

```bash
# 1. Install missing build prerequisites.
packages=(git python3 python3-venv build-essential cmake libfftw3-dev liblapack-dev)
missing=()
for package in "${packages[@]}"; do
  dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "ok installed" \
    || missing+=("$package")
done
if ((${#missing[@]})); then
  sudo apt-get update -qq
  sudo apt-get install -y "${missing[@]}"
fi

# 2. Build the reproducible phonemizer backend unless the pinned commit is installed.
espeak_ng_commit=2ea41210
espeak_ng_marker=/usr/local/share/pathbench/espeak-ng-commit
if ! command -v espeak-ng >/dev/null \
    || [[ ! -r "$espeak_ng_marker" ]] \
    || [[ "$(cat "$espeak_ng_marker")" != "$espeak_ng_commit" ]]; then
  if { test -d /tmp/espeak-ng/.git \
      || git clone https://github.com/espeak-ng/espeak-ng.git /tmp/espeak-ng; } \
      && git -C /tmp/espeak-ng fetch origin "$espeak_ng_commit" \
      && git -C /tmp/espeak-ng checkout --detach "$espeak_ng_commit" \
      && cmake -S /tmp/espeak-ng -B /tmp/espeak-ng/build \
        -DUSE_ASYNC=OFF -DBUILD_SHARED_LIBS=ON \
      && cmake --build /tmp/espeak-ng/build -j"$(nproc)" \
      && sudo cmake --install /tmp/espeak-ng/build \
      && sudo ldconfig \
      && sudo install -d "$(dirname "$espeak_ng_marker")"; then
    printf '%s\n' "$espeak_ng_commit" \
      | sudo tee "$espeak_ng_marker" >/dev/null
  else
    echo "Failed to install pinned espeak-ng; commit marker was not written." >&2
    exit 1
  fi
fi

# 3. Reuse the current checkout, or clone to a stable absolute destination.
if pathbench_root=$(git rev-parse --show-toplevel 2>/dev/null) \
    && test -f "$pathbench_root/tools/test_gpu_predictors.py"; then
  : # Already anywhere inside a PathBench checkout.
else
  pathbench_root=${PATHBENCH_ROOT:-"$PWD/pathbench"}
  test -d "$pathbench_root/.git" \
    || git clone https://github.com/karkirowle/pathbench.git "$pathbench_root"
fi
cd "$pathbench_root"
python3 tools/test_gpu_predictors.py
```

This procedure assumes that a working NVIDIA driver is already installed;
`nvidia-smi` must list the assigned GPU. Driver installation is host- and
cloud-specific and is deliberately not attempted by the script.

If you have the opportunity to start from a clean AWS/GCE instance, please do so and follow the make installation.

If you are working on a highly restricted HPC cluster, I would recommend starting from the singularity container [provided](https://github.com/karkirowle/pathbench/releases/download/v0.1.0/pathbench.sif).

Package installation is the recommended pathway when incorporating PathBench
into an existing environment. In that case, you are responsible for resolving
dependency conflicts.

All Python runtime dependencies are available from package indexes rather than
VCS URLs. In particular, `phonemizer-fork==3.3.2` (which installs the
`phonemizer` import package) and `pyctcdecode==0.5.0` use versioned PyPI
releases, so installing PathBench does not require GitHub credentials. The
system-level espeak-ng revision below remains separately pinned because its
language-specific IPA output is part of the metric definition.

### Package installation

**System dependencies** (not installable via pip — must be installed separately):
- `espeak-ng` at commit [`2ea41210`](https://github.com/espeak-ng/espeak-ng/commit/2ea41210) (post-1.52.0) — required by the phonemizer for grapheme-to-phoneme conversion. The exact commit matters: different espeak-ng versions produce different IPA symbols for some languages (e.g. Italian `ɾ` vs `r`), which affects phoneme-based metrics (PER, dPER, ArtP). Build from source:
  ```bash
  git clone https://github.com/espeak-ng/espeak-ng.git
  cd espeak-ng && git checkout 2ea41210
  cmake -B build -DUSE_ASYNC=OFF -DBUILD_SHARED_LIBS=ON
  cmake --build build -j$(nproc) && sudo cmake --install build
  ```
- PyTorch — install the CPU or CUDA build appropriate for your system by following
  [pytorch.org](https://pytorch.org/get-started/locally/) *before* installing PathBench.

**Option A — Install from a GitHub Release:**
```bash
pip install https://github.com/karkirowle/pathbench/releases/download/v0.1.0/pathbench-0.1.0-py3-none-any.whl
```

**Option B — Install directly from the repository:**
```bash
pip install git+https://github.com/karkirowle/pathbench.git
```

With optional dependencies (scripts, docs):
```bash
pip install "pathbench[scripts] @ git+https://github.com/karkirowle/pathbench.git"
```

### Make installation

The `make` installation route assumes the default setup of a standard Ubuntu
22.04 image (`ubuntu-2204-jammy`) and Python 3.10–3.12. It creates
`tools/venv`. The default is a CPU-only PyTorch installation, which works for
inference and tests but is slower than a supported GPU.

```bash
sudo apt-get update -qq
sudo apt install python3 python3-pip python3-venv build-essential cmake libfftw3-dev liblapack-dev -y
# Install espeak-ng from source (pinned commit for reproducible phonemization)
git clone https://github.com/espeak-ng/espeak-ng.git /tmp/espeak-ng
cd /tmp/espeak-ng && git checkout 2ea41210
cmake -B build -DUSE_ASYNC=OFF -DBUILD_SHARED_LIBS=ON
cmake --build build -j$(nproc) && sudo cmake --install build && sudo ldconfig
cd -
git clone https://github.com/karkirowle/pathbench.git
cd pathbench/tools && make
cd ..
source tools/venv/bin/activate
```

For a CUDA build, select a wheel index supported by the pinned PyTorch version.
For example, PyTorch 2.6.0 provides CUDA 12.4 wheels:

```bash
cd pathbench/tools
make CUDA_VERSION=12.4
```

You can select a particular interpreter with, for example,
`make PYTHON=python3.12`. Re-running `make` resumes after completed stages;
run `make clean` first to rebuild the environment with a different Python,
PyTorch, or CUDA selection.

### GPU installation and predictor smoke test

After installing the pinned `espeak-ng` build above, systems with an NVIDIA GPU
and driver can use the helper script to create a separate CUDA environment and
run the focused ArtP and DArtP tests:

```bash
python tools/test_gpu_predictors.py
```

The script checks for `nvidia-smi` and `espeak-ng`, creates
`tools/gpu_venv`, installs the CUDA 12.4 builds of PyTorch and torchaudio 2.6.0,
installs PathBench and its test dependencies, verifies that PyTorch can access
the GPU, and runs both predictor tests. It can be invoked from any directory.
The first run downloads the Python packages and model checkpoints and therefore
requires network access and several gigabytes of free disk space.

Override its defaults with command-line options (or the corresponding
`PYTHON`, `CUDA_VERSION`, `PYTORCH_VERSION`, and `VENV` environment variables)
when needed. The selected CUDA wheel must exist for the selected PyTorch release:

```bash
python tools/test_gpu_predictors.py --python python3.11 --cuda-version 12.6 \
  --pytorch-version 2.6.0 --venv /path/to/pathbench-gpu-venv
```

The script requires an NVIDIA driver compatible with the chosen CUDA wheel;
installing the wheel does not install a host GPU driver or the CUDA toolkit.
It does not download the DArtP n-gram model. Place the English model in `lms/`
as described under [N-gram models](#n-gram-models); otherwise the DArtP test is
reported as skipped. ArtP does not need that model.

For both tests together, allow **at least 12 GB of system RAM and 8 GB of GPU
VRAM**; **16 GB system RAM and 12–16 GB VRAM are recommended** to leave room
for both wav2vec2 models, the decoder, and transient activations. Any NVIDIA
CUDA GPU supported by the selected PyTorch wheel is acceptable; a T4 (16 GB),
L4 (24 GB), A10/A10G (24 GB), V100 (16/32 GB), or A100 works. Smaller 8 GB
cards may require closing other GPU processes and can run out of memory on
long audio. AMD ROCm GPUs, Apple GPUs, and CPU-only runtimes do not satisfy
this CUDA smoke test.

Google Colab GPUs can be used. Select a GPU runtime and confirm that
`nvidia-smi` works; the commonly assigned T4 and higher-memory L4/A100 options
meet the recommendation. Colab does not guarantee a particular GPU, RAM
amount, availability, or uninterrupted runtime, and its temporary filesystem
means the environment and downloaded checkpoints may need to be recreated in
a later session. If Colab assigns a smaller GPU or low-RAM runtime, inspect
`nvidia-smi` and available system memory before running the helper.

**Without sudo access:** A containerised environment such as Docker is recommended.

## Downloads

### Datasets

We are not allowed to share these datasets ourselves, however, all of them are relatively easily accesible. Please get your copy.

* [COPAS](https://taalmaterialen.ivdnt.org/download/tstc-corpus-pathologische-en-normale-spraak-copas/)

* [EasyCall](http://neurolab.unife.it/easycallcorpus/)

* [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)

* [NeuroVoz](https://zenodo.org/records/10777657)

* [UASpeech](https://speechtechnology.web.illinois.edu/uaspeech/)

* [Oral Cancer - YouTube](https://zenodo.org/records/18738598)

* [MDSC (Mandarin Dysarthric Speech Corpus)](https://www.aishelltech.com/AISHELL_6B)

After downloading the datasets, repoint the `wav.scp` files to your local dataset root. We do not provide a script for this, but you can use a regex replacement such as:

```bash
find datasets/ -name "wav.scp" -exec sed -i 's|/data/group1/z40484r/datasets|/path/to/your/datasets|g' {} +
```

**EasyCall fix:** Some EasyCall audio files have a stray space in their filename (`m13 _` instead of `m13_`). Rename them before running the benchmark:

```bash
find /path/to/your/datasets/easycall/EasyCall/m13 -name "m13 _*" -exec bash -c 'mv "$1" "${1//m13 _/m13_}"' _ {} \;
```

### N-gram models

The n-gram models required by DArtP are included in the
[Oral Cancer - YouTube](https://zenodo.org/records/18738598) download. ArtP does
not require an n-gram model. Create `lms/` at the repository root and copy the
models there with these exact names:

| Language | Filename |
| --- | --- |
| English | `wiki_en_token.arpa` or `wiki_en_token.arpa.bin` |
| Dutch | `wiki_nl_token.arpa` or `wiki_nl_token.arpa.bin` |
| Spanish | `wiki_es_token.arpa.bin` |
| Italian | `wiki_it_token.arpa.bin` |
| Mandarin Chinese | `wiki_zh_token.arpa` or `wiki_zh_token.arpa.bin` |

## Testing

### Installation integrity

It is recommended that after this setup you run the unit tests below. If these pass you can be reasonably sure about installation integrity.

```bash
source tools/venv/bin/activate
python -m pytest tests/test_evaluators.py::TestEvaluatorMethods -v
```

All tests should pass. If all evaluator tests fail simultaneously, the reference audio file in `tests/data/test_audio.wav` may be corrupted — the `test_audio_integrity` test will confirm this.

To test only ArtP and DArtP after downloading the English n-gram model, run:

```bash
python -m pytest \
  tests/test_evaluators.py::TestEvaluatorMethods::test_articulatory_precision \
  tests/test_evaluators.py::TestEvaluatorMethods::test_artp_double_asr -v
```

The first run downloads the Hugging Face checkpoints used by the phonetic and
English ASR models and therefore requires network access and several gigabytes
of free disk space. The DArtP test is reported as skipped, rather than failed,
when neither English n-gram filename listed above exists.

> **Note:** During the NAD evaluator tests you will see a `Wav2Vec2Model LOAD REPORT` table listing several keys (e.g. `project_q`, `quantizer`) as **UNEXPECTED**. These warnings are harmless — the keys belong to pre-training heads that are not needed for feature extraction and can be safely ignored.

### Dataset integrity

```bash
python -m pytest tests/test_evaluators.py::TestDatasetIntegrity::test_audio_file_hashes -v
```

Share these hashes alongside your results so others can verify they are using the same data.

> **Note:** Different versions of UASpeech exist. A denoising step was applied to UASpeech in December 2020. PathBench uses the **denoised** version canonically — `datasets/uaspeech/` points at the `noisereduce` audio set. If you have the pre-denoising audio your hashes will not match.



# Citation

If you use PathBench in your research, please cite:

```bibtex
@misc{halpern2026pathbenchspeechintelligibilitybenchmark,
      title={PathBench: Speech Intelligibility Benchmark for Automatic Pathological Speech Assessment},
      author={Bence Mark Halpern and Thomas Tienkamp and Defne Abur and Tomoki Toda},
      year={2026},
      eprint={2603.08097},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.08097},
}
```

# License

The PathBench repository code is released under the MIT License. However, some individual evaluators include or derive from code released under more restrictive licenses. In particular, the `PraatSpeechRateEvaluator` in `pathbench/speech_rate.py` is based on the Praat Script Syllable Nuclei by Nivja de Jong and Ton Wempe, which is licensed under the GNU General Public License v3 (GPL-3.0).

We are not able to provide legal advice. If you believe there is a licensing concern with any component in this codebase, please [open an issue](https://github.com/karkirowle/pathbench/issues).

# Acknowledgements

Many parts were shamelessly copied from others libraries or reproduced after consultation with those people.
I would like to especially say thanks to  Martijn Bartelds and Parvaneh Janbakhshi.
-  WADA-SNR: https://gist.github.com/johnmeade/d8d2c67b87cda95cd253f55c21387e75
-  NAD: https://github.com/Bartelds/neural-acoustic-distance
-  CPP: https://github.com/satvik-dixit/CPP
-  Unit test audio from the [Speech Accent Archive](https://accent.gmu.edu/)

# Funding

This work is partly financed by the Dutch Research Council (NWO) under project number 019.232SG.011, and partly supported by JST CREST JPMJCR19A3, Japan.

## Author

Bence Mark Halpern, Nagoya University
