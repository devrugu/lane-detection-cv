# Lane Detection Demo (desktop app)

A self-contained Windows desktop app that runs the project's **classical CV +
EMA** lane detector on your own images and videos. No Python install needed on
the target machine — one `.exe`, double-click to run.

## What it does

**Image mode** — Load any image (`png`, `jpg`, `jpeg`, `bmp`, `webp`, `tif`).
The original shows on the left, detected lanes on the right. Save the result
as PNG or JPEG.

**Video mode** — Open a dashcam clip (`mp4`, `avi`, `mov`, `mkv`, `webm`). Lanes
are drawn on every frame using the classical pipeline with exponential-moving-
average temporal smoothing. A **Coefficient plot** button opens (and closes) a
live, dockable panel that plots the EMA coefficients `a`, `b`, `c` for both
lanes as the video plays. **Export video** renders the full annotated clip to
an `.mp4`.

> Note on calibration: the perspective transform is tuned for a forward-facing
> dashcam at roughly 1280×720 (the TuSimple geometry). Inputs are fitted to that
> working size automatically, so footage from a very different camera angle may
> detect less reliably — that's a property of the classical pipeline, not the app.

## Running from source (for development)

```bash
pip install -r demo/requirements-demo.txt
python demo/app.py
```

## Building the Windows .exe

You don't build on Windows by hand. The build runs on GitHub Actions on a
Windows runner, so you can trigger it from Codespaces.

**From Codespaces (recommended):**

```bash
./demo/release.sh            # auto-bumps version, or:
./demo/release.sh v1.0.0     # pick a version
```

This commits your current code, pushes a tag, and GitHub Actions builds
`LaneDetectionDemo.exe` and attaches it to the repo's **Releases** page in a few
minutes. Because the build always uses the code at that tag, changing any
algorithm and re-running `release.sh` gives you a fresh exe with the new logic —
no other steps.

**On-demand build without a release:** open the repo's **Actions** tab →
*Build Windows demo* → *Run workflow*. The exe appears as a downloadable
artifact on that run.

**Local build (produces a binary for your current OS, handy for testing):**

```bash
./demo/build.sh
```

## How it maps to the project

The app imports the existing pipeline from `src/` unchanged:
`PerspectiveWarper` → `threshold_pipeline` → `sliding_window_search` →
`fit_lane_polynomial`, wrapped by `EMASmoother` for video. The U-Net method is
intentionally not included — the demo showcases the classical + EMA contribution.