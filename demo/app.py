"""Lane Detection Demo — desktop GUI.

Two modes:
  • Image  — load a still, see detected lanes side by side, save the result.
  • Video  — play a clip with lanes drawn each frame (classical CV + EMA),
             with an openable/closable live plot of the EMA coefficients,
             and export of the rendered video.

The heavy work runs in QThread workers so the window stays responsive.
"""

from __future__ import annotations

import os
import sys
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QAction, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QStackedWidget, QSlider, QStyle, QSizePolicy,
    QDockWidget, QMessageBox, QFrame,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core import ImageProcessor, VideoProcessor, FrameOutput

# ----------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------
INK = "#10151c"        # near-black background
PANEL = "#1a2230"      # raised panel
EDGE = "#2b3647"       # hairline borders
TEXT = "#e6ecf3"       # primary text
MUTED = "#8794a7"      # secondary text
LANE_L = "#00e5e5"     # left-lane cyan  (matches overlay)
LANE_R = "#ffb400"     # right-lane amber (matches overlay)
ACCENT = "#4ea1ff"     # interactive accent

STYLESHEET = f"""
QMainWindow, QWidget {{ background: {INK}; color: {TEXT};
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; font-size: 13px; }}
QLabel#title {{ font-size: 17px; font-weight: 600; letter-spacing: 0.3px; }}
QLabel#hint {{ color: {MUTED}; }}
QLabel#caption {{ color: {MUTED}; font-size: 11px; letter-spacing: 1px; }}
QPushButton {{ background: {PANEL}; border: 1px solid {EDGE}; border-radius: 8px;
    padding: 9px 16px; color: {TEXT}; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; border-color: {EDGE}; }}
QPushButton#primary {{ background: {ACCENT}; border: none; color: #061018;
    font-weight: 600; }}
QPushButton#primary:hover {{ background: #6cb3ff; }}
QPushButton#primary:disabled {{ background: {EDGE}; color: {MUTED}; }}
QFrame#view {{ background: #0a0e14; border: 1px solid {EDGE}; border-radius: 12px; }}
QLabel#imgview {{ background: transparent; }}
QSlider::groove:horizontal {{ height: 4px; background: {EDGE}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; height: 14px;
    margin: -6px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QDockWidget {{ titlebar-close-icon: none; color: {TEXT}; }}
QDockWidget::title {{ background: {PANEL}; padding: 7px 12px;
    border-bottom: 1px solid {EDGE}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {EDGE}; }}
"""


def rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Convert an HxWx3 RGB uint8 array to a QPixmap (copy is intentional)."""
    h, w = rgb.shape[:2]
    contiguous = np.ascontiguousarray(rgb)
    img = QImage(contiguous.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def fmt_coeffs(c: np.ndarray | None) -> str:
    if c is None:
        return "—"
    a, b, cc = c
    return f"a={a:+.2e}  b={b:+.3f}  c={cc:+.1f}"


# ----------------------------------------------------------------------------
# A QLabel that scales its pixmap to fit while keeping aspect ratio.
# ----------------------------------------------------------------------------
class ImageView(QLabel):
    def __init__(self, placeholder: str):
        super().__init__()
        self.setObjectName("imgview")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder
        self.setText(placeholder)
        self.setStyleSheet(f"color: {MUTED};")

    def set_image(self, rgb: np.ndarray | None):
        if rgb is None:
            self._pixmap = None
            self.setText(self._placeholder)
            return
        self._pixmap = rgb_to_qpixmap(rgb)
        self._rescale()

    def _rescale(self):
        if self._pixmap is None:
            return
        self.setPixmap(self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        self._rescale()
        super().resizeEvent(e)


# ----------------------------------------------------------------------------
# Live EMA coefficient plot (dockable, openable/closable during playback)
# ----------------------------------------------------------------------------
class CoeffPlot(QWidget):
    MAXLEN = 240  # ~ rolling window of frames

    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(4, 4), facecolor=INK)
        self.canvas = FigureCanvas(self.fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.canvas)

        self.axes = self.fig.subplots(3, 1, sharex=True)
        self.labels = ["a  (curvature)", "b  (slope)", "c  (offset)"]
        for ax, lab in zip(self.axes, self.labels):
            ax.set_facecolor("#0a0e14")
            ax.tick_params(colors=MUTED, labelsize=8)
            for s in ax.spines.values():
                s.set_color(EDGE)
            ax.set_ylabel(lab, color=TEXT, fontsize=8)
        self.axes[-1].set_xlabel("frame", color=MUTED, fontsize=8)
        self.fig.tight_layout(pad=0.8)

        self._x = deque(maxlen=self.MAXLEN)
        self._L = [deque(maxlen=self.MAXLEN) for _ in range(3)]
        self._R = [deque(maxlen=self.MAXLEN) for _ in range(3)]
        self._n = 0

    def reset(self):
        self._x.clear()
        for d in self._L + self._R:
            d.clear()
        self._n = 0
        for ax in self.axes:
            ax.cla()
        self.canvas.draw_idle()

    def push(self, left: np.ndarray | None, right: np.ndarray | None):
        self._n += 1
        self._x.append(self._n)
        for i in range(3):
            self._L[i].append(np.nan if left is None else float(left[i]))
            self._R[i].append(np.nan if right is None else float(right[i]))

    def redraw(self):
        if not self._x:
            return
        x = list(self._x)
        for i, ax in enumerate(self.axes):
            ax.cla()
            ax.set_facecolor("#0a0e14")
            ax.plot(x, list(self._L[i]), color=LANE_L, lw=1.6, label="left")
            ax.plot(x, list(self._R[i]), color=LANE_R, lw=1.6, label="right")
            ax.set_ylabel(self.labels[i], color=TEXT, fontsize=8)
            ax.tick_params(colors=MUTED, labelsize=8)
            for s in ax.spines.values():
                s.set_color(EDGE)
            ax.grid(True, color="#161d28", lw=0.6)
        self.axes[0].legend(loc="upper right", fontsize=7, framealpha=0.0,
                            labelcolor=TEXT, ncol=2)
        self.axes[-1].set_xlabel("frame", color=MUTED, fontsize=8)
        self.fig.tight_layout(pad=0.8)
        self.canvas.draw_idle()


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------
class ImageWorker(QThread):
    done = Signal(object)   # FrameOutput
    failed = Signal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            data = np.fromfile(self.path, dtype=np.uint8)  # unicode-path safe
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                self.failed.emit("That file could not be read as an image.")
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out = ImageProcessor().process(rgb)
            self.done.emit(out)
        except Exception as e:  # pragma: no cover
            self.failed.emit(str(e))


class VideoWorker(QThread):
    """Decodes a video, runs detection per frame, emits results paced by fps."""
    frame_ready = Signal(int, object)   # frame_index, FrameOutput
    progress = Signal(int, int)         # current, total
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, path: str, alpha: float = 0.25):
        super().__init__()
        self.path = path
        self.alpha = alpha
        self._paused = False
        self._stop = False

    def pause(self, p: bool):
        self._paused = p

    def stop(self):
        self._stop = True

    def run(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.failed.emit("That file could not be opened as a video.")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        delay_ms = max(int(1000.0 / fps), 1)
        proc = VideoProcessor(alpha=self.alpha)
        idx = 0
        try:
            while not self._stop:
                if self._paused:
                    self.msleep(30)
                    continue
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                out = proc.process(rgb)
                self.frame_ready.emit(idx, out)
                self.progress.emit(idx + 1, total)
                idx += 1
                self.msleep(delay_ms)
            self.finished_ok.emit()
        except Exception as e:  # pragma: no cover
            self.failed.emit(str(e))
        finally:
            cap.release()


class ExportWorker(QThread):
    """Re-runs detection over the whole video and writes an annotated mp4."""
    progress = Signal(int, int)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, src: str, dst: str, alpha: float = 0.25):
        super().__init__()
        self.src = src
        self.dst = dst
        self.alpha = alpha

    def run(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            self.failed.emit("Could not reopen the source video for export.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.dst, fourcc, fps, (w, h))
        proc = VideoProcessor(alpha=self.alpha)
        idx = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                out = proc.process(rgb)
                writer.write(cv2.cvtColor(out.overlay_rgb, cv2.COLOR_RGB2BGR))
                idx += 1
                self.progress.emit(idx, total)
            self.done.emit(self.dst)
        except Exception as e:  # pragma: no cover
            self.failed.emit(str(e))
        finally:
            cap.release()
            writer.release()


# ----------------------------------------------------------------------------
# Mode pages
# ----------------------------------------------------------------------------
IMG_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"
VID_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.webm)"


class ImageMode(QWidget):
    def __init__(self):
        super().__init__()
        self._orig: np.ndarray | None = None
        self._overlay: np.ndarray | None = None
        self._worker: ImageWorker | None = None

        cap = QLabel("IMAGE MODE")
        cap.setObjectName("caption")
        hint = QLabel("Load a road image. Lanes appear on the right.")
        hint.setObjectName("hint")

        self.left_view = ImageView("Original\n\nLoad an image to begin")
        self.right_view = ImageView("Detected lanes")
        for v in (self.left_view, self.right_view):
            frame = QFrame()
            frame.setObjectName("view")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(8, 8, 8, 8)
            fl.addWidget(v)
            v.frame = frame

        views = QHBoxLayout()
        views.setSpacing(14)
        views.addWidget(self.left_view.frame)
        views.addWidget(self.right_view.frame)

        self.btn_load = QPushButton(" Load image")
        self.btn_load.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.btn_load.clicked.connect(self.on_load)
        self.btn_save = QPushButton(" Save result")
        self.btn_save.setObjectName("primary")
        self.btn_save.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save.setEnabled(False)

        self.status = QLabel("")
        self.status.setObjectName("hint")

        bar = QHBoxLayout()
        bar.addWidget(self.btn_load)
        bar.addWidget(self.btn_save)
        bar.addStretch(1)
        bar.addWidget(self.status)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 18)
        lay.setSpacing(10)
        lay.addWidget(cap)
        lay.addWidget(hint)
        lay.addLayout(views, 1)
        lay.addLayout(bar)

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load image", "", IMG_FILTER)
        if not path:
            return
        data = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.warning(self, "Cannot read", "That file isn't a readable image.")
            return
        self._orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.left_view.set_image(self._orig)
        self.right_view.set_image(None)
        self.right_view.setText("Detecting…")
        self.btn_save.setEnabled(False)
        self.status.setText("Detecting lanes…")

        self._worker = ImageWorker(path)
        self._worker.done.connect(self.on_done)
        self._worker.failed.connect(self.on_fail)
        self._worker.start()

    def on_done(self, out: FrameOutput):
        self._overlay = out.overlay_rgb
        self.right_view.set_image(out.overlay_rgb)
        self.btn_save.setEnabled(True)
        bits = []
        bits.append("left ✓" if out.left_detected else "left ✗")
        bits.append("right ✓" if out.right_detected else "right ✗")
        self.status.setText("   ".join(bits) + f"     {fmt_coeffs(out.left_coeffs)}")

    def on_fail(self, msg: str):
        self.status.setText("")
        QMessageBox.warning(self, "Detection failed", msg)

    def on_save(self):
        if self._overlay is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", "lanes.png", "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        bgr = cv2.cvtColor(self._overlay, cv2.COLOR_RGB2BGR)
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, bgr)
        if ok:
            buf.tofile(path)
            self.status.setText(f"Saved to {os.path.basename(path)}")
        else:
            QMessageBox.warning(self, "Save failed", "Could not encode the image.")


class VideoMode(QWidget):
    coeffs_pushed = Signal()

    def __init__(self, coeff_plot: CoeffPlot):
        super().__init__()
        self._path: str | None = None
        self._worker: VideoWorker | None = None
        self._export: ExportWorker | None = None
        self._plot = coeff_plot
        self._playing = False
        self._plot_dirty = False

        cap = QLabel("VIDEO MODE")
        cap.setObjectName("caption")
        hint = QLabel("Play a dashcam clip with lanes drawn each frame "
                      "(classical CV + EMA smoothing).")
        hint.setObjectName("hint")

        self.view = ImageView("Open a video to begin")
        frame = QFrame()
        frame.setObjectName("view")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(8, 8, 8, 8)
        fl.addWidget(self.view)

        self.coeff_readout = QLabel("EMA coefficients will appear here")
        self.coeff_readout.setObjectName("hint")

        self.btn_open = QPushButton(" Open video")
        self.btn_open.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.btn_open.clicked.connect(self.on_open)

        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)

        self.btn_plot = QPushButton(" Coefficient plot")
        self.btn_plot.setCheckable(True)
        self.btn_plot.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))

        self.btn_export = QPushButton(" Export video")
        self.btn_export.setObjectName("primary")
        self.btn_export.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.btn_export.clicked.connect(self.on_export)
        self.btn_export.setEnabled(False)

        self.progress = QSlider(Qt.Horizontal)
        self.progress.setEnabled(False)

        self.status = QLabel("")
        self.status.setObjectName("hint")

        controls = QHBoxLayout()
        controls.addWidget(self.btn_open)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_plot)
        controls.addStretch(1)
        controls.addWidget(self.btn_export)

        botrow = QHBoxLayout()
        botrow.addWidget(self.coeff_readout, 1)
        botrow.addWidget(self.status)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 18)
        lay.setSpacing(10)
        lay.addWidget(cap)
        lay.addWidget(hint)
        lay.addWidget(frame, 1)
        lay.addWidget(self.progress)
        lay.addLayout(controls)
        lay.addLayout(botrow)

    # --- playback ---
    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", VID_FILTER)
        if not path:
            return
        self._stop_worker()
        self._path = path
        self._plot.reset()
        self.view.set_image(None)
        self.view.setText("Ready — press play")
        self.btn_play.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._playing = False
        self.status.setText(os.path.basename(path))

    def toggle_play(self):
        if self._path is None:
            return
        if self._worker is None:
            # start fresh
            self._plot.reset()
            self._worker = VideoWorker(self._path)
            self._worker.frame_ready.connect(self.on_frame)
            self._worker.progress.connect(self.on_progress)
            self._worker.finished_ok.connect(self.on_finished)
            self._worker.failed.connect(self.on_fail)
            self._worker.start()
            self._playing = True
        else:
            self._playing = not self._playing
            self._worker.pause(not self._playing)
        self.btn_play.setIcon(self.style().standardIcon(
            QStyle.SP_MediaPause if self._playing else QStyle.SP_MediaPlay))

    def on_frame(self, idx: int, out: FrameOutput):
        self.view.set_image(out.overlay_rgb)
        self._plot.push(out.left_coeffs, out.right_coeffs)
        fb = []
        if out.used_fallback_left:
            fb.append("L carry-fwd")
        if out.used_fallback_right:
            fb.append("R carry-fwd")
        tag = ("   [" + ", ".join(fb) + "]") if fb else ""
        self.coeff_readout.setText(
            f"L  {fmt_coeffs(out.left_coeffs)}      "
            f"R  {fmt_coeffs(out.right_coeffs)}{tag}")
        # Only redraw the plot if it's visible (cheap when hidden).
        if self.btn_plot.isChecked():
            self._plot.redraw()

    def on_progress(self, cur: int, total: int):
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(cur)

    def on_finished(self):
        self._playing = False
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._worker = None
        if self.btn_plot.isChecked():
            self._plot.redraw()
        self.status.setText("Playback complete")

    def on_fail(self, msg: str):
        self._worker = None
        self._playing = False
        QMessageBox.warning(self, "Video error", msg)

    def _stop_worker(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(1500)
            self._worker = None

    # --- export ---
    def on_export(self):
        if self._path is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export annotated video", "lanes_out.mp4", "MP4 (*.mp4)")
        if not path:
            return
        self.btn_export.setEnabled(False)
        self.status.setText("Exporting…  0%")
        self._export = ExportWorker(self._path, path)
        self._export.progress.connect(self.on_export_progress)
        self._export.done.connect(self.on_export_done)
        self._export.failed.connect(self.on_export_fail)
        self._export.start()

    def on_export_progress(self, cur: int, total: int):
        pct = int(100 * cur / total) if total else 0
        self.status.setText(f"Exporting…  {pct}%")

    def on_export_done(self, path: str):
        self.btn_export.setEnabled(True)
        self.status.setText(f"Exported to {os.path.basename(path)}")

    def on_export_fail(self, msg: str):
        self.btn_export.setEnabled(True)
        QMessageBox.warning(self, "Export failed", msg)


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lane Detection — Classical CV + EMA")
        self.resize(1180, 720)

        self.coeff_plot = CoeffPlot()
        self.dock = QDockWidget("Live EMA coefficients", self)
        self.dock.setWidget(self.coeff_plot)
        self.dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.hide()

        self.image_mode = ImageMode()
        self.video_mode = VideoMode(self.coeff_plot)

        # tie the video mode's plot toggle to the dock
        self.video_mode.btn_plot.toggled.connect(self.dock.setVisible)
        self.dock.visibilityChanged.connect(self._sync_plot_button)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.image_mode)
        self.stack.addWidget(self.video_mode)

        # --- header with mode switch ---
        title = QLabel("Lane Detection Demo")
        title.setObjectName("title")
        self.btn_img = QPushButton("Image")
        self.btn_vid = QPushButton("Video")
        self.btn_img.setCheckable(True)
        self.btn_vid.setCheckable(True)
        self.btn_img.setChecked(True)
        self.btn_img.clicked.connect(lambda: self.set_mode(0))
        self.btn_vid.clicked.connect(lambda: self.set_mode(1))

        header = QHBoxLayout()
        header.setContentsMargins(18, 12, 18, 0)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.btn_img)
        header.addWidget(self.btn_vid)

        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(header)
        rl.addWidget(self.stack, 1)
        self.setCentralWidget(root)

    def set_mode(self, i: int):
        self.stack.setCurrentIndex(i)
        self.btn_img.setChecked(i == 0)
        self.btn_vid.setChecked(i == 1)
        if i == 0:           # image mode has no coeff plot
            self.dock.hide()

    def _sync_plot_button(self, visible: bool):
        self.video_mode.btn_plot.setChecked(visible)

    def closeEvent(self, e):
        self.video_mode._stop_worker()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()