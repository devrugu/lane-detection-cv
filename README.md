# Lane Detection: Classical CV vs. Deep Learning

Comparing a traditional OpenCV lane-detection pipeline against a U-Net deep
learning baseline on the **TuSimple** benchmark, with an exponential-moving-
average (EMA) temporal-smoothing module proposed as an extension to the
classical pipeline. CS523 project.

There are two ways to use this repository:

- **Notebooks** — walk through the system stage by stage to understand and
  reproduce how it works.
- **Demo app** — a one-file Windows executable that runs the classical + EMA
  detector on your own images and videos, no Python required.

---

## Methods compared

1. **Classical CV pipeline** — perspective warp to a bird's-eye view, HLS color
   + Sobel-x thresholding, sliding-window pixel search, and degree-2 polynomial
   fitting of each lane.
2. **Classical CV + temporal smoothing** *(proposed)* — the same pipeline with
   an EMA applied to the polynomial coefficients across video frames, plus
   carry-forward when a lane is briefly lost. Reduces frame-to-frame jitter.
3. **U-Net baseline** — an encoder–decoder CNN trained on TuSimple, used as a
   deep-learning point of comparison.

### Results on the TuSimple test subset (100 frames, CPU)

| Method | Point accuracy (SOTA convention) | Matched-lane precision | Speed (avg FPS) |
|---|---|---|---|
| Classical CV | 0.25 | 0.99 | ~39 |
| Classical CV + EMA | — | — | ~36 |
| U-Net | 0.40 | 1.00 | ~1.5 |

The two accuracy columns measure different things: the SOTA convention counts
every ground-truth point (penalizing missed lanes), while matched-lane
precision measures geometric accuracy only on lanes that were actually
detected. The classical pipeline is highly precise on the lanes it finds but
detects fewer of them; the U-Net detects more lanes but runs ~25× slower on
CPU. Full numbers are in `results/tusimple_metrics_*.json` and
`results/fps_benchmark.json`.

---

## Repository layout

```
lane-detection-cv/
├── src/                  Core pipeline modules + scripts (see below)
├── notebooks/            Stage-by-stage exploration (01–08)
├── demo/                 Desktop demo app + build/release tooling
├── checkpoints/          Trained U-Net weights (unet_best.pth)
├── data/                 TuSimple data (downloaded locally; git-ignored)
├── results/              Figures, videos, metrics, cached predictions
├── report/               Report build artifacts
├── tests/                Unit tests
├── .github/workflows/    CI that builds the Windows demo exe
├── requirements.txt      Full research environment
└── .devcontainer/        Codespaces / VS Code dev container config
```

### Core pipeline modules (`src/`)

| Module | Role |
|---|---|
| `perspective.py` | `PerspectiveWarper` — camera ↔ bird's-eye transform |
| `thresholding.py` | `threshold_pipeline` — HLS color + Sobel-x binary mask |
| `sliding_window.py` | `sliding_window_search` — groups lane pixels left/right |
| `polyfit.py` | `fit_lane_polynomial`, `evaluate_polynomial` — degree-2 fit |
| `pipeline.py` | `ClassicalPipeline` — ties stages together; `draw_overlay` |
| `temporal.py` | `EMASmoother` — coefficient smoothing across frames |
| `tusimple_loader.py` | Parses TuSimple annotations into image paths + lanes |

### Scripts (`src/`)

| Script | Purpose |
|---|---|
| `check_env.py` | Verify the environment is set up correctly |
| `download_subset.py` | Fetch a small dev subset (≈100 frames) without the 21.6 GB zip |
| `download_whole_data.py` | Fetch the full test set, extract curated clips, delete the zip |
| `make_demo_videos.py` | Download full clips and assemble demo videos for the app |
| `check_clips.py` | List which clips have all 20 frames downloaded |
| `evaluate.py` | TuSimple metrics (SOTA accuracy + matched precision) |
| `reevaluate_both.py` | Re-compute metrics for classical and U-Net |
| `benchmark.py` | FPS benchmark for all three methods |
| `precompute_predictions.py` | Cache predictions for the comparison video |
| `make_comparison_video.py` | Render the 3-column comparison video |
| `make_pipeline_figure.py` | 6-panel figure of the pipeline stages |
| `make_summary_figure.py` | Headline accuracy/FPS/jitter figure |

---

## Getting started

### Option A — Codespaces / dev container (recommended)

The repo ships a dev container, so opening it in GitHub Codespaces (or locally
with the VS Code Dev Containers extension) installs everything automatically via
`requirements.txt`.

After it starts, sanity-check the environment:

```bash
python src/check_env.py
```

### Option B — local install

Requires Python 3.11.

```bash
git clone https://github.com/devrugu/lane-detection-cv.git
cd lane-detection-cv
pip install -r requirements.txt
python src/check_env.py
```

### Getting the data

The TuSimple data is not committed. Downloading needs a configured **Kaggle
CLI** (place your `kaggle.json` token in `~/.kaggle/`). Then fetch the
annotation file and a small working subset:

```bash
# Annotation file
kaggle datasets download manideep1108/tusimple \
    -f TUSimple/test_label.json -p data/tusimple --unzip

# Small subset for development (~100 frames, a few clips with all 20 frames)
python src/download_subset.py
```

For the full curated set, use `src/download_whole_data.py` instead.

---

## Using the notebooks (explore the system)

The notebooks in `notebooks/` follow the pipeline end to end. Run them in order;
each builds on the modules in `src/`.

| Notebook | What it covers |
|---|---|
| `01_explore_data.ipynb` | Inspect TuSimple images and annotations |
| `02_perspective_transform.ipynb` | Bird's-eye warp and calibration points |
| `03_thresholding.ipynb` | HLS color + Sobel thresholding |
| `04_sliding_window.ipynb` | Sliding-window lane-pixel search |
| `05_polynomial_fit.ipynb` | Fitting and evaluating lane polynomials |
| `06_temporal_smoothing.ipynb` | EMA smoothing across video frames |
| `07_evaluation.ipynb` | Metrics, comparisons, and figures |
| `08_unet_training.ipynb` | Training the U-Net baseline |

Launch with:

```bash
jupyter notebook        # or open them in VS Code / Codespaces
```

### Reproducing the figures and metrics

```bash
python src/benchmark.py              # FPS numbers -> results/fps_benchmark.json
python src/reevaluate_both.py        # accuracy metrics -> results/*.json
python src/make_pipeline_figure.py   # results/pipeline_stages.png
python src/make_summary_figure.py    # results/summary_comparison.png
python src/precompute_predictions.py # cache predictions
python src/make_comparison_video.py  # results/comparison_video.mp4
```

---

## Using the demo app (test the system)

The demo is a desktop app that runs the **classical CV + EMA** detector (not the
U-Net) on your own media. It has two modes:

- **Image mode** — load any image (`png`, `jpg`, `jpeg`, `bmp`, `webp`, `tif`);
  the original shows on the left, detected lanes on the right; save the result.
- **Video mode** — open a dashcam clip (`mp4`, `avi`, `mov`, `mkv`, `webm`);
  lanes are drawn on every frame, an openable/closable live plot shows the EMA
  coefficients `a`, `b`, `c` for both lanes, and you can export the annotated
  video.

> The perspective transform is calibrated for forward-facing dashcam footage at
> roughly 1280×720 (TuSimple geometry). Inputs are auto-resized to that working
> size, so footage from a very different camera angle may detect less reliably.

### Run it from source (for development)

```bash
pip install -r demo/requirements-demo.txt
python demo/app.py
```

(In a headless environment like Codespaces there's no display, so the window
won't appear — build the executable instead, below.)

### Build the Windows executable

PyInstaller can only build for the OS it runs on, so the Windows `.exe` is built
on a GitHub Actions Windows runner. You trigger it without leaving Codespaces:

```bash
./demo/release.sh            # auto-bumps the version
./demo/release.sh v1.0.0     # or name the version explicitly
```

This commits your current code, pushes a tag, and the
`.github/workflows/build-windows.yml` workflow builds `LaneDetectionDemo.exe`
and attaches it to the repo's **Releases** page in a few minutes. Because the
build always uses the tagged code, changing any algorithm and re-running
`release.sh` produces a fresh exe with the new logic.

You can also trigger a build without cutting a release from the repo's
**Actions** tab → *Build Windows demo* → *Run workflow*; the exe appears as a
downloadable artifact on that run.

To build a binary for your current OS locally (handy for testing):

```bash
./demo/build.sh
```

If the release step ever fails with a permissions error, enable write access at
repo **Settings → Actions → General → Workflow permissions → Read and write**.

### Generating test videos from TuSimple

Each TuSimple clip is 20 consecutive frames (~1 second). To build watchable demo
videos, download full clips and stitch them together:

```bash
python src/make_demo_videos.py
```

This writes both a native-speed and a slowed-down video per batch into
`data/tusimple/demo_videos/`. Feed any `demo_*.mp4` into the app's Video mode.

There is more detail in `demo/README.md`.

---

## Running on Windows

Copy the downloaded `LaneDetectionDemo.exe` to any Windows machine and
double-click it — no Python or install needed. On first launch, Windows
SmartScreen may warn about an unknown publisher (the exe isn't code-signed);
choose **More info → Run anyway**.

---

## Tests

```bash
pytest
```

---

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgments

Built on the [TuSimple Lane Detection benchmark](https://github.com/TuSimple/tusimple-benchmark).
