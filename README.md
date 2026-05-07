# lane-detection-cv
Comparing traditional CV and deep learning approaches for lane detection — CS523 project


This project compares a classical OpenCV-based lane detection pipeline against a U-Net deep learning baseline on the TuSimple Lane Detection benchmark. A temporal smoothing module is proposed as an extension to the classical pipeline to improve robustness across video frames.

## Methods compared

1. **Traditional CV pipeline** — perspective warp, HLS + Sobel thresholding, sliding window search, polynomial fitting
2. **Traditional CV + temporal smoothing** (proposed) — adds exponential moving average across frames
3. **U-Net deep learning baseline** — encoder-decoder CNN trained on TuSimple