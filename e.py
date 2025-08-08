import sys
import time
import hashlib
import threading
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageFont, ImageDraw
import mss
import easyocr
from googletrans import Translator

from PyQt5.QtCore import Qt, QRect, QThread, pyqtSignal, QSize, QObject
from PyQt5.QtGui import QGuiApplication, QFont, QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QPlainTextEdit,
    QCheckBox,
    QMessageBox,
    QShortcut,
)
import os
import subprocess
import shutil

"""
Screen Translator Overlay – Fixed / Full Version
------------------------------------------------
- ROI picker works across multi-monitor setups; overlay hidden during pick.
- OCR coordinates mapped back to absolute (virtual desktop) space.
- Overlay covers the entire virtual desktop; label sizes adapt to content.
- ESC cancels selection; Ctrl+C quits app.

Requirements (pip):
  pip install pyqt5 mss pillow easyocr googletrans==4.0.0-rc1 numpy

Tip: EasyOCR's language codes used below: en, vi, ja, ko, ch_sim, ch_tra, ...
"""

# ----------------------------
# Defaults / Config
# ----------------------------
OVERLAY_SCRIPT = os.environ.get("OVERLAY_SCRIPT", "overlay.py")  # path to user's Tk overlay
DEFAULT_TARGET_LANG = "vi"
OCR_LANGS = [
    "en"
]

# ----------------------------
# Logging setup
# ----------------------------
class LogEmitter(QObject):
    sig = pyqtSignal(str)

class QtLogHandler(logging.Handler):
    def __init__(self, emitter: 'LogEmitter'):
        super().__init__()
        self.emitter = emitter

    def emit(self, record):
        try:
            msg = self.format(record)
            # Ensure GUI updates happen on the main thread
            self.emitter.sig.emit(msg)
        except Exception:
            pass

logger = logging.getLogger("OverlayTranslator")
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(
    "overlay_translator.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)

# ----------------------------
# Data types
# ----------------------------
@dataclass
class TextItem:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (absolute, virtual desktop)
    text: str
    conf: float

# ----------------------------
# UI helpers – ROI picker
# ----------------------------
class ScreenSelector(QWidget):
    region_selected = pyqtSignal(QRect)  # absolute (virtual desktop) rect

    def __init__(self, unified_geometry: QRect):
        super().__init__()
        self.setWindowTitle("Chọn vùng dịch")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(unified_geometry)
        self.start = None
        self.end = None

    def showEvent(self, _):
        # Fullscreen over the unified desktop span
        self.showFullScreen()
        self.raise_()
        self.activateWindow()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # Cancel selection
            self.close()

    def mousePressEvent(self, event):
        self.start = event.pos()
        self.end = self.start
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if self.start and self.end:
            x1 = min(self.start.x(), self.end.x())
            y1 = min(self.start.y(), self.end.y())
            x2 = max(self.start.x(), self.end.x())
            y2 = max(self.start.y(), self.end.y())
            # Map widget-local -> absolute virtual desktop coords
            g = self.geometry()
            rect = QRect(x1 + g.left(), y1 + g.top(), (x2 - x1), (y2 - y1))
            if rect.width() > 0 and rect.height() > 0:
                self.region_selected.emit(rect)
        self.close()

    def paintEvent(self, _):
        from PyQt5.QtGui import QPainter, QColor
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 60))
        if self.start and self.end:
            x1 = min(self.start.x(), self.end.x())
            y1 = min(self.start.y(), self.end.y())
            x2 = max(self.start.x(), self.end.x())
            y2 = max(self.start.y(), self.end.y())
            painter.fillRect(QRect(x1, y1, x2 - x1, y2 - y1), QColor(0, 0, 0, 0))
            painter.setPen(QColor(255, 255, 255, 200))
            painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))

# ----------------------------
# Worker – capture, OCR, translate
# ----------------------------
class OCRWorker(QThread):
    results_ready = pyqtSignal(list)  # List[(bbox_abs, translated_text)]
    log_msg = pyqtSignal(str)

    def __init__(
        self,
        roi: Optional[QRect],
        target_lang: str,
        interval_sec: float,
        min_conf: float,
        use_gpu: bool,
        ocr_langs: List[str],
    ):
        super().__init__()
        self._stop = threading.Event()
        self.roi = roi  # absolute coords (virtual desktop)
        self.target_lang = target_lang
        self.interval_sec = max(0.2, float(interval_sec))
        self.min_conf = float(min_conf)
        self.use_gpu = bool(use_gpu)
        self.ocr_langs = ocr_langs
        self.reader = None
        self.translator = Translator()
        self.last_hash = None

    def log(self, level: int, msg: str):
        logger.log(level, msg)
        self.log_msg.emit(msg)

    def stop(self):
        self._stop.set()

    @staticmethod
    def _bbox_from_poly(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (min(xs), min(ys), max(xs), max(ys))

    def preprocess(self, img: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        # Sharpen the image to enhance text edges for better OCR accuracy
        try:
            gray = gray.filter(ImageFilter.UnsharpMask(radius=1, percent=150))
        except Exception:
            pass
        return gray

    @lru_cache(maxsize=4096)
    def translate_cached(self, text: str, dest: str) -> str:
        if not text or not text.strip():
            return ""
        try:
            # One call is cheaper than detect()+translate(); let Google auto-detect
            res = self.translator.translate(text, src="auto", dest=dest)
            return res.text
        except Exception as e:
            self.log(logging.ERROR, f"Translate failed: {e}")
            return text

    def init_reader(self):
        try:
            self.reader = easyocr.Reader(self.ocr_langs, gpu=self.use_gpu)
            self.log(logging.INFO, f"EasyOCR initialized (gpu={self.use_gpu}) with langs={self.ocr_langs}")
        except Exception as e:
            self.log(logging.ERROR, f"Failed to init easyocr: {e}")
            raise

    def run(self):
        try:
            self.init_reader()
        except Exception:
            return

        with mss.mss() as sct:
            while not self._stop.is_set():
                time.sleep(self.interval_sec)
                # Determine capture region (absolute virtual desktop coords)
                try:
                    if self.roi and not self.roi.isNull():
                        mon = {
                            "left": int(self.roi.left()),
                            "top": int(self.roi.top()),
                            "width": int(self.roi.width()),
                            "height": int(self.roi.height()),
                        }
                        offset_x, offset_y = mon["left"], mon["top"]
                    else:
                        # Full virtual desktop is monitors[0]
                        monitor = sct.monitors[0]
                        mon = {
                            "left": monitor["left"],
                            "top": monitor["top"],
                            "width": monitor["width"],
                            "height": monitor["height"],
                        }
                        offset_x, offset_y = mon["left"], mon["top"]
                except Exception as e:
                    self.log(logging.ERROR, f"Get monitor failed: {e}")
                    continue

                try:
                    shot = sct.grab(mon)
                    img = Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)
                except Exception as e:
                    self.log(logging.ERROR, f"Screen grab failed: {e}")
                    continue

                pre = self.preprocess(img)

                # Skip identical frame
                try:
                    h = hashlib.md5(pre.tobytes()).hexdigest()
                    if h == self.last_hash:
                        self.log(logging.DEBUG, "Skip identical frame")
                        continue
                    self.last_hash = h
                except Exception as e:
                    self.log(logging.WARNING, f"Hash failed: {e}")

                # OCR
                try:
                    result = self.reader.readtext(np.array(pre), detail=1, paragraph=False)
                    self.log(logging.DEBUG, f"OCR found {len(result)} items")
                except Exception as e:
                    self.log(logging.ERROR, f"EasyOCR failed: {e}")
                    continue

                items: List[TextItem] = []
                try:
                    for poly, text, conf in result:
                        try:
                            conf_f = float(conf)
                        except Exception:
                            conf_f = 0.0
                        if conf_f < self.min_conf:
                            continue
                        x1, y1, x2, y2 = self._bbox_from_poly(poly)
                        # Map ROI-local -> absolute screen coords
                        x1 += offset_x
                        x2 += offset_x
                        y1 += offset_y
                        y2 += offset_y
                        t = (text or "").strip()
                        if len(t) < 2:
                            continue
                        items.append(TextItem((int(x1), int(y1), int(x2), int(y2)), t, conf_f))
                except Exception as e:
                    self.log(logging.WARNING, f"Parse OCR result failed: {e}")

                merged = self.merge_overlaps(items)

                out = []
                for it in merged:
                    translated = self.translate_cached(it.text, self.target_lang)
                    out.append((it.bbox, translated))

                self.results_ready.emit(out)

    @staticmethod
    def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / float(area_a + area_b - inter)

    def merge_overlaps(self, items: List[TextItem]) -> List[TextItem]:
        items = sorted(items, key=lambda x: (x.bbox[1], x.bbox[0]))
        merged: List[TextItem] = []
        for it in items:
            if not merged:
                merged.append(it)
                continue
            last = merged[-1]
            if self.iou(last.bbox, it.bbox) > 0.15:
                x1 = min(last.bbox[0], it.bbox[0])
                y1 = min(last.bbox[1], it.bbox[1])
                x2 = max(last.bbox[2], it.bbox[2])
                y2 = max(last.bbox[3], it.bbox[3])
                merged[-1] = TextItem(
                    (x1, y1, x2, y2), last.text + " " + it.text, max(last.conf, it.conf)
                )
            else:
                merged.append(it)
        return merged

# ----------------------------
# Overlay Window – draws translated labels
# ----------------------------
class OverlayWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Translator Overlay")
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Compute unified virtual desktop geometry
        vg = None
        for s in QGuiApplication.screens():
            vg = s.geometry() if vg is None else vg.united(s.geometry())
        self.virtual_geo = vg
        self.origin_x, self.origin_y = vg.left(), vg.top()
        self.setGeometry(vg)

        self.container = QWidget(self)
        self.setCentralWidget(self.container)
        self.vbox = QVBoxLayout(self.container)
        self.vbox.setContentsMargins(0, 0, 0, 0)

        self.labels: List[QLabel] = []
        self.font = QFont("Segoe UI", 14)

    def clear_labels(self):
        for lb in self.labels:
            lb.setParent(None)
            lb.deleteLater()
        self.labels.clear()

    def update_overlay(self, results: List[Tuple[Tuple[int, int, int, int], str]]):
        self.clear_labels()
        for (x1, y1, x2, y2), text in results:
            if not text or not text.strip():
                continue
            # Translate absolute (virtual) -> local window coords
            lx1 = x1 - self.origin_x
            ly1 = y1 - self.origin_y
            w = max(80, (x2 - x1))
            h = max(20, (y2 - y1))
            lb = QLabel(self)
            lb.setText(self.wrap_text(text, max_chars=max(8, w // 12)))
            lb.setWordWrap(True)
            lb.setFont(self.font)
            lb.setStyleSheet(
                """
                QLabel {
                    color: white;
                    background-color: rgba(0, 0, 0, 140);
                    border-radius: 8px;
                    padding: 6px 10px;
                }
                """
            )
            # Fix overlay height to original text height
            lb.setGeometry(int(lx1), int(ly1), int(w), int(h))
            lb.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            lb.show()
            self.labels.append(lb)

    @staticmethod
    def wrap_text(s: str, max_chars: int = 20) -> str:
        s = s.strip()
        if len(s) <= max_chars:
            return s
        words = s.split()
        lines = []
        cur = []
        n = 0
        for w in words:
            add = (1 if cur else 0) + len(w)
            if n + add > max_chars:
                lines.append(" ".join(cur))
                cur = [w]
                n = len(w)
            else:
                cur.append(w)
                n += add
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines)

# ----------------------------
# Main UI – controls + logging
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Translator Controller")
        self.setMinimumSize(QSize(680, 520))
        try:
            self.setWindowIcon(QIcon.fromTheme("translate"))
        except Exception:
            pass

        # Keep references
        self.selector: Optional[ScreenSelector] = None
        self.overlay = OverlayWindow()
        self.overlay.showFullScreen()

        # State
        self.roi: Optional[QRect] = None
        self.worker: Optional[OCRWorker] = None

        # ---- Controls ----
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        ctrl_box = QGroupBox("Điều khiển")
        grid = QGridLayout(ctrl_box)

        # Target language
        grid.addWidget(QLabel("Ngôn ngữ đích:"), 0, 0)
        self.cbo_lang = QComboBox()
        common_langs = [
            ("Tiếng Việt", "vi"), ("English", "en"), ("日本語", "ja"), ("한국어", "ko"), ("中文(简)", "zh-cn"), ("中文(繁)", "zh-tw"),
            ("Español", "es"), ("Français", "fr"), ("Deutsch", "de"), ("Português", "pt"), ("Русский", "ru"), ("ไทย", "th"),
        ]
        for name, code in common_langs:
            self.cbo_lang.addItem(f"{name} ({code})", code)
        idx = self.cbo_lang.findData(DEFAULT_TARGET_LANG)
        if idx >= 0:
            self.cbo_lang.setCurrentIndex(idx)
        grid.addWidget(self.cbo_lang, 0, 1)

        # Interval
        grid.addWidget(QLabel("Chu kỳ quét (s):"), 0, 2)
        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setRange(0.2, 5.0)
        self.spin_interval.setSingleStep(0.1)
        self.spin_interval.setValue(0.8)
        grid.addWidget(self.spin_interval, 0, 3)

        # Min confidence
        grid.addWidget(QLabel("Ngưỡng tin cậy OCR:"), 1, 0)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.5)
        grid.addWidget(self.spin_conf, 1, 1)

        # GPU toggle
        self.chk_gpu = QCheckBox("Dùng GPU (nếu có)")
        self.chk_gpu.setChecked(False)
        grid.addWidget(self.chk_gpu, 1, 2, 1, 2)

        # Buttons
        self.btn_pick = QPushButton("Chọn vùng (ROI)")
        self.btn_start = QPushButton("Bắt đầu")
        self.btn_stop = QPushButton("Dừng")
        self.btn_clear = QPushButton("Xóa chữ nổi")
        grid.addWidget(self.btn_pick, 2, 0)
        grid.addWidget(self.btn_start, 2, 1)
        grid.addWidget(self.btn_stop, 2, 2)
        grid.addWidget(self.btn_clear, 2, 3)

        # Option: use external Tk overlay backend
        self.chk_tk_backend = QCheckBox("Dùng overlay.py (Tk) để hiển thị")
        grid.addWidget(self.chk_tk_backend, 3, 0, 1, 2)

        layout.addWidget(ctrl_box)

        # Log panel
        log_box = QGroupBox("Debug Log")
        v = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        v.addWidget(self.log_view)
        layout.addWidget(log_box, 1)

        # Connect
        self.btn_pick.clicked.connect(self.pick_region)
        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop.clicked.connect(self.stop_worker)
        self.btn_clear.clicked.connect(self.clear_overlays)

        # Shortcuts: Ctrl+C to quit
        self.quit_sc = QShortcut(QKeySequence("Ctrl+C"), self)
        self.quit_sc.activated.connect(self.close)

        # Hook Python logging to UI
        self.log_emitter = LogEmitter()
        self.log_emitter.sig.connect(self.append_log)
        ui_handler = QtLogHandler(self.log_emitter)
        ui_handler.setLevel(logging.DEBUG)
        ui_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(ui_handler)

        logger.info("Ứng dụng khởi động")
        self.append_log("==== Sẵn sàng. Nhấn 'Bắt đầu' để chạy OCR + dịch. ====")

    # ---- Logging helpers ----
    def append_log(self, msg: str):
        self.log_view.appendPlainText(msg)
        self.log_view.ensureCursorVisible()

    # ---- ROI selection ----
    def unified_geometry(self) -> QRect:
        vg = None
        for s in QGuiApplication.screens():
            vg = s.geometry() if vg is None else vg.united(s.geometry())
        return vg

    def pick_region(self):
        # Hide overlay so it doesn't block clicks
        self.overlay.hide()

        # Close old selector if still alive
        if self.selector and self.selector.isVisible():
            self.selector.close()

        vg = self.unified_geometry()
        self.selector = ScreenSelector(vg)
        self.selector.setWindowModality(Qt.ApplicationModal)
        self.selector.region_selected.connect(self._on_region_selected)
        # When picker closes (even if canceled), show overlay again
        self.selector.destroyed.connect(lambda: self.overlay.showFullScreen())

        self.selector.show()
        self.selector.raise_()
        self.selector.activateWindow()

    def _on_region_selected(self, rect: QRect):
        self.roi = rect
        logger.info(
            f"ROI set to: x={rect.left()} y={rect.top()} w={rect.width()} h={rect.height()}"
        )
        self.append_log("Đã chọn vùng. Nhấn Bắt đầu để chạy.")
        if self.selector:
            self.selector.close()
            self.selector = None
        self.overlay.showFullScreen()

    # ---- Worker lifecycle ----
    def start_worker(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Đang chạy", "Worker đang chạy rồi.")
            return
        target_lang = self.cbo_lang.currentData() or DEFAULT_TARGET_LANG
        interval = self.spin_interval.value()
        min_conf = self.spin_conf.value()
        use_gpu = self.chk_gpu.isChecked()

        self.worker = OCRWorker(self.roi, target_lang, interval, min_conf, use_gpu, OCR_LANGS)
        self.worker.results_ready.connect(self.on_results)
        self.worker.log_msg.connect(self.append_log)
        try:
            self.worker.start()
            logger.info(
                f"Worker started | lang={target_lang} | interval={interval}s | conf>={min_conf} | gpu={use_gpu}"
            )
            self.append_log("Worker đã khởi động.")
        except Exception as e:
            logger.exception("Start worker failed")
            QMessageBox.critical(self, "Lỗi", f"Không thể khởi động worker: {e}")

    def stop_worker(self):
        if not self.worker:
            return
        try:
            self.worker.stop()
            self.worker.wait(3000)
            logger.info("Worker stopped")
            self.append_log("Worker đã dừng.")
        except Exception as e:
            logger.exception("Stop worker failed")
            QMessageBox.warning(self, "Cảnh báo", f"Dừng worker lỗi: {e}")
        finally:
            self.worker = None
            self.kill_external_overlays()

    def closeEvent(self, event):
        try:
            self.stop_worker()
            self.kill_external_overlays()
        finally:
            self.overlay.close()
            super().closeEvent(event)

    # ---- Results handling & external Tk overlay ----
    def clear_overlays(self):
        # Clear both internal and external overlays
        self.overlay.clear_labels()
        self.kill_external_overlays()

    def on_results(self, results: List[Tuple[Tuple[int, int, int, int], str]]):
        if self.chk_tk_backend.isChecked():
            # Use external Tk overlay: kill old, spawn new per item
            self.kill_external_overlays()
            for (x1, y1, x2, y2), text in results:
                if not text.strip():
                    continue
                try:
                    self.spawn_tk_overlay(text, x1, y1, x2, y2)
                except Exception as e:
                    logger.warning(f"Spawn Tk overlay failed: {e}")
        else:
            self.overlay.update_overlay(results)

    def kill_external_overlays(self):
        if not hasattr(self, "tk_procs"):
            self.tk_procs = []
        for p in list(self.tk_procs):
            try:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=0.2)
                    except Exception:
                        p.kill()
            except Exception:
                pass
        self.tk_procs = []

    def measure_text_height(self, text: str, font_path: str, size: int) -> int:
        if size < 1:
            return 0
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            font = ImageFont.load_default()
        img = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        return max(0, bbox[3] - bbox[1])

    def calc_font_size_for_height(self, text: str, target_h: int, font_path: str = "arial.ttf") -> int:
        target_h = max(8, int(target_h))
        s = 48
        h = self.measure_text_height(text, font_path, s)
        if h <= 0:
            return target_h
        s = max(6, min(400, int(s * (target_h / h))))
        # refine a couple of steps
        for _ in range(2):
            h = self.measure_text_height(text, font_path, s)
            if h == 0:
                break
            if abs(h - target_h) <= 1:
                break
            s = max(6, min(400, int(s * (target_h / h))))
        return int(s)

    def spawn_tk_overlay(self, text: str, x1: int, y1: int, x2: int, y2: int):
        # Compute font size so rendered height ~= original bbox height
        h = max(8, int(y2 - y1))
        size = self.calc_font_size_for_height(text, h)
        # Truncate very long strings to avoid command-line limits
        text_arg = text if len(text) <= 300 else text[:297] + "..."
        py = sys.executable
        script = OVERLAY_SCRIPT
        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen([py, script, text_arg, str(int(size)), str(int(x1)), str(int(y1))],
                                    creationflags=creationflags)
            if not hasattr(self, "tk_procs"):
                self.tk_procs = []
            self.tk_procs.append(proc)
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "Không tìm thấy overlay.py",
                f"Không tìm thấy script overlay.py tại: {script}."
                "Đặt overlay.py cùng thư mục .py hoặc đặt biến môi trường OVERLAY_SCRIPT."
            )
        except Exception as e:
            logger.exception(f"Failed to launch overlay.py: {e}")

# ----------------------------
# Entry
# ----------------------------

def main():
    # High-DPI awareness on Windows for crisp overlay
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
