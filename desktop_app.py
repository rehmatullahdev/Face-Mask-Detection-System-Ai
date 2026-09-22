

import sys
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np

from PIL import Image, ImageTk

from app.services.mask_detection_pipeline import MaskDetectionPipeline
from app.utils.drawing import draw_all_detections
from app import create_app
from app.services.logging_service import log_detection
import time

# ── Model paths ─────────────────────────────────────────────────────────────
RESOURCES  = ROOT / "app" / "resources" / "models"
FACE_MODEL = str(RESOURCES / "blaze_face_full_range.tflite")
MASK_MODEL = str(RESOURCES / "mask_detector.keras")

# ── Design tokens (mirrors web app palette) ──────────────────────────────────
BG         = "#f0f4f8"
CARD       = "#ffffff"
SURFACE    = "#f8fafc"
BLUE       = "#3b82f6"
BLUE_DARK  = "#2563eb"
SUCCESS    = "#10b981"
DANGER     = "#ef4444"
TEXT       = "#0f172a"
MUTED      = "#64748b"
BORDER     = "#e2e8f0"
FONT       = "Segoe UI"


# ─────────────────────────────────────────────────────────────────────────────
class RoundedButton(tk.Canvas):
    """A pill-shaped button drawn on a Canvas for a modern look."""

    def __init__(self, parent, text, command,
                 bg=BLUE, fg="white", hover_bg=BLUE_DARK,
                 enabled_bg=None, enabled_fg=None, enabled_hover=None,
                 font_size=10, height=38, radius=10, state="normal", **kwargs):
        super().__init__(parent, highlightthickness=0, **kwargs)
        self._text    = text
        self._command = command
        self._font    = (FONT, font_size, "bold")
        self._height  = height
        self._radius  = radius
        self._state   = state

        # Enabled colors — default to whatever is passed as bg/fg/hover
        self._en_bg    = enabled_bg    or bg
        self._en_fg    = enabled_fg    or fg
        self._en_hover = enabled_hover or hover_bg

        # Current active colors (start at whatever state requires)
        self._bg   = bg
        self._fg   = fg
        self._hover = hover_bg

        self.configure(height=height, cursor="hand2" if state == "normal" else "arrow")
        self.bind("<Configure>", self._draw)
        self.bind("<Button-1>",  self._on_click)
        self.bind("<Enter>",     self._on_enter)
        self.bind("<Leave>",     self._on_leave)

    def _draw(self, _=None, fill=None):
        if fill is None:
            fill = self._bg if self._state == "normal" else BORDER
        fg = self._fg if self._state == "normal" else MUTED
        self.delete("all")
        w, h, r = self.winfo_width(), self._height, self._radius
        self.create_polygon(
            r, 0,  w - r, 0,  w, 0,  w, r,
            w, h - r,  w, h,  w - r, h,
            r, h,  0, h,  0, h - r,
            0, r,  0, 0,
            fill=fill, smooth=True, outline=""
        )
        self.create_text(w // 2, h // 2, text=self._text,
                         fill=fg, font=self._font, anchor="center")

    def _on_enter(self, _):
        if self._state == "normal":
            self._draw(fill=self._hover)

    def _on_leave(self, _):
        if self._state == "normal":
            self._draw()

    def _on_click(self, _):
        if self._state == "normal" and self._command:
            self._command()

    def set_state(self, state):

        self._state = state
        if state == "normal":
            # Restore enabled color set
            self._bg    = self._en_bg
            self._fg    = self._en_fg
            self._hover = self._en_hover
        self.configure(cursor="hand2" if state == "normal" else "arrow")
        self._draw()


# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Face Mask Detector")
        self.geometry("1020x640")
        self.minsize(820, 540)
        self.configure(bg=BG)
        self.resizable(True, True)

        # State
        self._pipeline: MaskDetectionPipeline | None = None
        self._source_img: np.ndarray | None = None   # raw loaded image
        self._tk_img     = None                       # Tkinter photo (kept alive)

        self._flask_app = create_app()
        self._build()
        self._load_models_async()

    # ─── UI ──────────────────────────────────────────────────────────────────

    def _build(self):
        self._build_navbar()
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True, padx=14, pady=(8, 0))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        self._build_preview(content)
        self._build_sidebar(content)
        self._build_statusbar()

    # ── Navbar ────────────────────────────────────────────────────────────────
    def _build_navbar(self):
        bar = tk.Frame(self, bg=CARD, height=56)
        bar.pack(fill="x", padx=14, pady=(12, 0))
        bar.pack_propagate(False)

        # Logo badge
        logo = tk.Label(bar, text="😷", font=(FONT, 16),
                        bg=BLUE, fg="white", width=3)
        logo.pack(side="left", padx=(14, 10), pady=8)
        logo.configure(relief="flat")

        brand = tk.Frame(bar, bg=CARD)
        brand.pack(side="left", pady=8)
        tk.Label(brand, text="Face Mask Detector",
                 font=(FONT, 12, "bold"), bg=CARD, fg=TEXT).pack(anchor="w")
        tk.Label(brand, text="Upload an image to detect mask",
                 font=(FONT, 8), bg=CARD, fg=MUTED).pack(anchor="w")

        # Model status chip
        self._chip_var = tk.StringVar(value="⏳  Loading models…")
        self._chip = tk.Label(bar, textvariable=self._chip_var,
                              font=(FONT, 8, "bold"), bg="#eff6ff", fg=BLUE,
                              padx=12, pady=5, relief="flat")
        self._chip.pack(side="right", padx=14)

    # ── Preview canvas ────────────────────────────────────────────────────────
    def _build_preview(self, parent):
        outer = tk.Frame(parent, bg=CARD, relief="flat")
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        # Inner top bar
        top = tk.Frame(outer, bg=CARD)
        top.pack(fill="x", padx=14, pady=(10, 6))
        tk.Label(top, text="Preview", font=(FONT, 10, "bold"),
                 bg=CARD, fg=TEXT).pack(side="left")
        self._preview_label = tk.Label(top, text="No image loaded",
                                       font=(FONT, 8), bg=CARD, fg=MUTED)
        self._preview_label.pack(side="right")

        # Separator
        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", padx=14)

        # Canvas
        self._canvas = tk.Canvas(outer, bg="#0f172a", highlightthickness=0,
                                 cursor="hand2")
        self._canvas.pack(fill="both", expand=True, padx=0, pady=0)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind("<Button-1>",  lambda _: self._browse())
        self._draw_placeholder()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=CARD)
        side.grid(row=0, column=1, sticky="nsew", pady=(0, 8))

        tk.Label(side, text="Controls", font=(FONT, 11, "bold"),
                 bg=CARD, fg=TEXT).pack(anchor="w", padx=16, pady=(16, 2))
        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(2, 10))

        # Upload button
        self._btn_upload = RoundedButton(
            side, text="⬆   Upload Image",
            command=self._browse,
            bg=BLUE, hover_bg=BLUE_DARK,
            font_size=10, height=38
        )
        self._btn_upload.pack(fill="x", padx=16, pady=(0, 6))

        # Analyse button — starts disabled (grey), enabled colors are blue
        self._btn_analyse = RoundedButton(
            side, text="🔍  Analyse",
            command=self._run_detection_async,
            bg=BORDER, fg=MUTED, hover_bg=BORDER,
            enabled_bg=BLUE, enabled_fg="white", enabled_hover=BLUE_DARK,
            font_size=10, height=38, state="disabled"
        )
        self._btn_analyse.pack(fill="x", padx=16, pady=(0, 4))

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=10)

        # Stats
        tk.Label(side, text="Results", font=(FONT, 10, "bold"),
                 bg=CARD, fg=TEXT).pack(anchor="w", padx=16)

        stats = tk.Frame(side, bg=CARD)
        stats.pack(fill="x", padx=16, pady=(8, 0))
        stats.columnconfigure(0, weight=1)
        stats.columnconfigure(1, weight=1)

        self._lbl_faces   = self._stat_card(stats, "Faces",     "—",  TEXT,    0, 0)
        self._lbl_mask    = self._stat_card(stats, "Mask",      "—",  SUCCESS, 0, 1)
        self._lbl_nomask  = self._stat_card(stats, "No Mask",   "—",  DANGER,  1, 0, span=2)

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=10)

        # Detection list
        tk.Label(side, text="Detections", font=(FONT, 9, "bold"),
                 bg=CARD, fg=MUTED).pack(anchor="w", padx=16)

        list_frame = tk.Frame(side, bg=SURFACE, relief="flat")
        list_frame.pack(fill="both", expand=True, padx=16, pady=(4, 16))

        sb = tk.Scrollbar(list_frame, orient="vertical")
        self._det_list = tk.Listbox(
            list_frame,
            font=(FONT, 9),
            bg=SURFACE, relief="flat",
            selectbackground=BLUE, selectforeground="white",
            highlightthickness=1, highlightcolor=BORDER,
            activestyle="none",
            yscrollcommand=sb.set, bd=0
        )
        sb.configure(command=self._det_list.yview)
        sb.pack(side="right", fill="y")
        self._det_list.pack(fill="both", expand=True)

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        self._status_var = tk.StringVar(value="Ready — select an image to begin.")
        tk.Label(self, textvariable=self._status_var,
                 font=(FONT, 8), bg=BORDER, fg=MUTED,
                 anchor="w", padx=14, pady=4).pack(fill="x", side="bottom")

    # ─── Stat card helper ─────────────────────────────────────────────────────

    def _stat_card(self, parent, label, value, color, row, col, span=1):
        padx_right = 4 if col == 0 and span == 1 else 0
        padx_left  = 4 if col == 1 else 0
        frame = tk.Frame(parent, bg=SURFACE, relief="flat")
        frame.grid(row=row, column=col, columnspan=span,
                   padx=(padx_left, padx_right), pady=3, sticky="ew")
        val = tk.Label(frame, text=value, font=(FONT, 20, "bold"),
                       bg=SURFACE, fg=color)
        val.pack(pady=(8, 0))
        tk.Label(frame, text=label, font=(FONT, 8),
                 bg=SURFACE, fg=MUTED).pack(pady=(0, 8))
        return val

    # ─── Canvas helpers ───────────────────────────────────────────────────────

    def _draw_placeholder(self):
        self._canvas.delete("all")
        self._canvas.update_idletasks()
        w = self._canvas.winfo_width()  or 620
        h = self._canvas.winfo_height() or 420
        self._canvas.create_text(
            w // 2, h // 2 - 22,
            text="🖼", font=("Segoe UI Emoji", 44),
            fill="#334155", tags="ph"
        )
        self._canvas.create_text(
            w // 2, h // 2 + 28,
            text="Click to upload an image",
            font=(FONT, 11), fill="#64748b", tags="ph"
        )
        self._canvas.create_text(
            w // 2, h // 2 + 50,
            text="JPG · PNG · BMP · WEBP",
            font=(FONT, 8), fill="#94a3b8", tags="ph"
        )

    def _on_canvas_resize(self, _event=None):
        if self._source_img is not None:
            self._show_on_canvas(self._source_img)
        else:
            self._draw_placeholder()

    def _show_on_canvas(self, bgr_img: np.ndarray):
        """Fit-scale a BGR numpy image onto the dark canvas."""
        self._canvas.update_idletasks()
        cw = self._canvas.winfo_width()  or 620
        ch = self._canvas.winfo_height() or 420
        ih, iw = bgr_img.shape[:2]
        scale   = min(cw / iw, ch / ih)
        nw, nh  = int(iw * scale), int(ih * scale)

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)

        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, anchor="center", image=tk_img)
        self._tk_img = tk_img           # prevent garbage collection

    # ─── Model loading ────────────────────────────────────────────────────────

    def _load_models_async(self):
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            self._pipeline = MaskDetectionPipeline(FACE_MODEL, MASK_MODEL)
            self.after(0, self._on_models_ready)
        except Exception as exc:
            self.after(0, lambda: self._on_models_failed(exc))

    def _on_models_ready(self):
        self._chip.configure(text="✅  Models loaded", bg="#ecfdf5", fg=SUCCESS)
        self._chip_var.set("✅  Models ready")
        self._status("Ready — select an image to begin.")

    def _on_models_failed(self, exc):
        self._chip.configure(text="❌  Model error", bg="#fef2f2", fg=DANGER)
        self._chip_var.set("❌  Model error")
        self._status(f"Error loading models: {exc}")
        messagebox.showerror("Model Load Error",
                             f"Could not load detection models:\n\n{exc}")

    # ─── Image browsing ───────────────────────────────────────────────────────

    def _browse(self):
        if self._pipeline is None:
            messagebox.showinfo("Please wait",
                                "Models are still loading — try again in a moment.")
            return
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files",   "*.*")
            ]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read the selected file.")
            return

        self._source_img = img          # keep original for re-analyse
        self._show_on_canvas(img)
        self._preview_label.configure(text=Path(path).name)
        self._reset_results()
        self._btn_analyse.set_state("normal")   # restores blue via _en_bg
        self._status(f"Loaded: {Path(path).name}")

    # ─── Detection ────────────────────────────────────────────────────────────

    def _run_detection_async(self):
        if self._source_img is None or self._pipeline is None:
            return
        self._btn_analyse.set_state("disabled")
        self._btn_upload.set_state("disabled")
        self._status("Analysing image…")
        threading.Thread(target=self._detect, daemon=True).start()

    def _detect(self):
        img = self._source_img.copy()
        try:
            result = self._pipeline.process_image(img)
            detections = result["detections"]
            stats      = result["stats"]
            

            annotated  = draw_all_detections(img.copy(), detections)
            
            with_mask    = stats["with_mask"]
            without_mask = stats["without_mask"]
            
            # ── Log to Database ───────────────────────────────────────────────
            try:
                filename = f"desktop_{int(time.time())}.jpg"
                upload_dir = self._flask_app.config['UPLOADS_DIR']
                filepath = os.path.join(upload_dir, filename)
                cv2.imwrite(filepath, annotated)
                
                with self._flask_app.app_context():
                    log_detection(
                        with_mask=with_mask,
                        without_mask=without_mask,
                        source='desktop',
                        image_filename=filename
                    )
            except Exception as log_err:
                print(f"Log Error: {log_err}")

            self.after(0, lambda: self._show_results(
                annotated, detections, with_mask, without_mask
            ))
        except Exception as exc:
            self.after(0, lambda: self._on_detect_error(exc))

    def _show_results(self, annotated, detections, with_mask, without_mask):
        self._show_on_canvas(annotated)

        # Update stat cards
        self._lbl_faces .configure(text=str(len(detections)))
        self._lbl_mask  .configure(text=str(with_mask))
        self._lbl_nomask.configure(text=str(without_mask))

        # Update detection list
        self._det_list.delete(0, "end")
        if not detections:
            self._det_list.insert("end", "  ⚠   No faces detected")
        else:
            for i, det in enumerate(detections, 1):
                icon = "✅" if det["label"] == "Mask" else "❌"
                conf = det["confidence"] * 100
                self._det_list.insert(
                    "end",
                    f"  {icon}  Face {i}:  {det['label']}  ({conf:.1f}%)"
                )

        noun = "face" if len(detections) == 1 else "faces"
        self._status(
            f"Done — {len(detections)} {noun} detected  ·  "
            f"{with_mask} mask  ·  {without_mask} no mask"
        )
        self._btn_analyse.set_state("normal")
        self._btn_upload.set_state("normal")

    def _on_detect_error(self, exc):
        messagebox.showerror("Detection Error", str(exc))
        self._btn_analyse.set_state("normal")
        self._btn_upload.set_state("normal")
        self._status(f"Error: {exc}")

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _reset_results(self):
        for lbl in (self._lbl_faces, self._lbl_mask, self._lbl_nomask):
            lbl.configure(text="—")
        self._det_list.delete(0, "end")

    def _status(self, msg: str):
        self._status_var.set(msg)


if __name__ == "__main__":
    App().mainloop()
