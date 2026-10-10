from pathlib import Path
from queue import Empty
from time import monotonic
import tkinter as tk

import customtkinter as ctk
import numpy as np

from core.calibration import (
    CALIBRATION_STEPS,
    CalibrationSample,
    CalibrationSession,
    TrackingAdapter,
    load_profile,
    save_profile,
    screen_targets_in_canvas,
)


BACKGROUND = "#10151e"
SURFACE = "#192230"
ACCENT = "#66dec1"
TEXT = "#eff5fa"
MUTED = "#9eafc3"
STEP_LABELS = {
    "neutral_head": "Relax your head and look at the center",
    "top_left": "Look at the top-left dot",
    "top_right": "Look at the top-right dot",
    "bottom_right": "Look at the bottom-right dot",
    "bottom_left": "Look at the bottom-left dot",
    "center": "Look at the center dot",
}


class CalibrationWindow(ctk.CTk):
    def __init__(
        self,
        adapter=None,
        profile_path=None,
        camera_metadata=None,
        coordinate_space="tracker",
        auto_start=True,
        pipeline_errors=None,
    ):
        ctk.set_appearance_mode("dark")
        super().__init__(fg_color=BACKGROUND)
        self.title("Eye & head calibration")
        self.geometry("900x680+0+0")
        self.resizable(True, True)
        self.minsize(600, 540)
        self.adapter = adapter if adapter is not None else TrackingAdapter()
        self.profile_path = Path(profile_path) if profile_path else Path(__file__).resolve().parents[1] / "calibration_profile.json"
        self.camera_metadata = camera_metadata
        self.coordinate_space = coordinate_space
        self.pipeline_errors = pipeline_errors
        self._pipeline_error = None
        self.monitor = {"width": self.winfo_screenwidth(), "height": self.winfo_screenheight(), "x": 0, "y": 0}
        self.profile = load_profile(self.profile_path, self.monitor)
        if self.profile and (
            self.profile["mapping"]["coordinate_space"] != coordinate_space
            or (camera_metadata is not None and self.profile["camera"] != camera_metadata)
        ):
            self.profile = None
        self.session = None
        self.stage = None
        self.last_sample = None
        self._closed = False
        self._poll_id = None
        self._start_id = None
        self._fullscreen_id = None
        self._maximize_id = self.after(0, self._maximize_window)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda event: self.cancel_calibration())
        self._build_dashboard()
        self._refresh_dashboard()
        self._poll_id = self.after(40, self._poll_tracking)
        if auto_start and self.profile is None:
            self._start_id = self.after(350, self.start_calibration)

    def _maximize_window(self):
        self._maximize_id = None
        if not self._closed and self.stage is None:
            self.state("zoomed")

    def _build_dashboard(self):
        card = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=20)
        self.card = card
        card.pack(fill="both", expand=True, padx=28, pady=28)
        ctk.CTkLabel(card, text="EYE & HEAD CONTROL", text_color=ACCENT, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=28, pady=(26, 4))
        ctk.CTkLabel(card, text="Set up your calibration", font=("Segoe UI", 27, "bold"), text_color=TEXT).pack(anchor="w", padx=28)
        ctk.CTkLabel(
            card,
            text="Sit comfortably, face the camera, and keep your head still.\nFollow each dot with your eyes. Capture happens automatically.",
            justify="left", text_color=MUTED, font=("Segoe UI", 14),
        ).pack(anchor="w", padx=28, pady=(12, 20))
        self.summary = ctk.CTkLabel(card, text="", justify="left", anchor="w", font=("Segoe UI", 15), text_color=TEXT)
        self.summary.pack(fill="x", padx=28, pady=(0, 18))
        threshold_row = ctk.CTkFrame(card, fg_color="transparent")
        threshold_row.pack(fill="x", padx=28)
        ctk.CTkLabel(threshold_row, text="Head gesture threshold", font=("Segoe UI", 14)).pack(side="left")
        value = self.profile["gesture_settings"]["trigger_degrees"] if self.profile else 15
        self.threshold = ctk.CTkOptionMenu(
            threshold_row, values=["10°", "15°", "20°"], width=100,
            fg_color="#2b394d", button_color="#344960", button_hover_color="#426079",
        )
        self.threshold.set(f"{value:g}°")
        self.threshold.pack(side="right")
        self.tracking_label = ctk.CTkLabel(card, text="Waiting for tracking…", text_color=MUTED, font=("Segoe UI", 13))
        self.tracking_label.pack(anchor="w", padx=28, pady=(20, 4))
        self.message = ctk.CTkLabel(card, text="", wraplength=520, justify="left", anchor="w", text_color=MUTED)
        self.message.pack(fill="x", padx=28, pady=(0, 12))
        self.start_button = ctk.CTkButton(
            card, text="Start calibration", command=self.start_calibration,
            fg_color=ACCENT, hover_color="#43bea2", text_color=BACKGROUND,
            font=("Segoe UI", 15, "bold"), height=44,
        )
        self.start_button.pack(fill="x", padx=28, pady=(0, 12))
        ctk.CTkButton(card, text="Close", command=self.close, fg_color="transparent", border_width=1, border_color="#405169", height=34).pack(fill="x", padx=28, pady=(0, 24))

    def _refresh_dashboard(self):
        resolution = f"Primary display · {self.monitor['width']} × {self.monitor['height']}"
        if self.profile:
            self.summary.configure(text=f"{resolution}\n\n✓ Five eye positions captured\n✓ Neutral head position recorded\n✓ Calibration saved")
            self.start_button.configure(text="Redo calibration")
            self.message.configure(text="Your saved calibration is ready.", text_color=ACCENT)
        else:
            self.summary.configure(text=f"{resolution}\n\nSix short tests · About 25 seconds\nPress Esc at any time to cancel.")
            self.start_button.configure(text="Start calibration")
            self.message.configure(text="A complete calibration is needed before eye control can start.", text_color=MUTED)

    def start_calibration(self):
        if self._start_id is not None:
            self.after_cancel(self._start_id)
        self._start_id = None
        if self.stage is not None or self._closed or self._pipeline_error:
            return
        self.session = CalibrationSession(self.monitor)
        self.stage = tk.Toplevel(self, bg=BACKGROUND)
        self.stage.withdraw()
        self.stage.protocol("WM_DELETE_WINDOW", self.close)
        self.stage.bind("<Escape>", lambda event: self.cancel_calibration())
        self.canvas = tk.Canvas(self.stage, bg=BACKGROUND, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self._draw_stage())
        self.withdraw()
        self.stage.deiconify()
        self.stage.lift()
        self.stage.focus_force()
        self._fullscreen_id = self.after(100, self._enter_stage_fullscreen, self.stage)
        self.start_button.configure(state="disabled")
        self.threshold.configure(state="disabled")
        self._draw_stage()

    def _enter_stage_fullscreen(self, stage):
        self._fullscreen_id = None
        if stage is self.stage and not self._closed:
            stage.state("zoomed")
            stage.update_idletasks()
            stage.attributes("-fullscreen", True)
            stage.lift()

    def _draw_stage(self):
        if (
            self.stage is None or self.session is None or self.session.complete
            or self.stage.state() not in ("normal", "zoomed")
        ):
            return
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 100 or height < 100:
            return
        step = self.session.step
        targets = screen_targets_in_canvas(self.monitor, width, height)
        x, y = targets["center" if step == "neutral_head" else step]
        self.canvas.create_text(width / 2, 56, text=f"CALIBRATION  ·  {self.session.step_index + 1} / {len(CALIBRATION_STEPS)}", fill=MUTED, font=("Segoe UI", 14, "bold"))
        self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20, outline="#345c58", width=2)
        self.canvas.create_oval(x - 9, y - 9, x + 9, y + 9, fill=ACCENT, outline="")
        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=TEXT, outline="")
        text_y = height * (0.7 if step in ("neutral_head", "center") else 0.44)
        self.canvas.create_text(width / 2, text_y, text=STEP_LABELS[step], fill=TEXT, font=("Segoe UI", 23, "bold"), width=width - 100)
        self.canvas.create_text(width / 2, text_y + 48, text=self.session.status, fill=ACCENT, font=("Segoe UI", 15), width=width - 100)
        bar_width = min(280, width - 100)
        left, top = (width - bar_width) / 2, text_y + 82
        self.canvas.create_rectangle(left, top, left + bar_width, top + 5, fill="#29394a", outline="")
        self.canvas.create_rectangle(left, top, left + bar_width * self.session.progress, top + 5, fill=ACCENT, outline="")
        self.canvas.create_text(width / 2, height - 52, text="Keep your head still. Move only your eyes.   ·   Esc to cancel", fill=MUTED, font=("Segoe UI", 13))

    def _poll_tracking(self):
        if self._closed:
            return
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        if self.pipeline_errors is not None:
            try:
                error = self.pipeline_errors.get_nowait()
            except Empty:
                pass
            else:
                self._pipeline_error = str(error)
                if self._start_id is not None:
                    self.after_cancel(self._start_id)
                    self._start_id = None
                if self.stage is not None:
                    self._close_stage()
                self.start_button.configure(state="disabled")
                self.message.configure(text=self._pipeline_error, text_color="#f5b492")
        if self._pipeline_error:
            self.tracking_label.configure(text="Tracking stopped", text_color=MUTED)
            self._poll_id = self.after(40, self._poll_tracking)
            return
        now = monotonic()
        capture_visible = self.stage is not None and self.stage.state() in ("normal", "zoomed")
        if self.session is not None and not capture_visible:
            self.session.interrupt()
        for _ in range(180):
            try:
                sample = self.adapter.samples.get_nowait()
            except Empty:
                break
            if not isinstance(sample, CalibrationSample):
                continue
            self.last_sample = sample
            if self.session is not None and capture_visible:
                try:
                    self.session.accept(sample, now)
                except (TypeError, ValueError):
                    self.session.interrupt()
        sample = self.last_sample
        try:
            connected = sample is not None and (
                np.isfinite([sample.eye_x, sample.eye_y, sample.pitch, sample.yaw, sample.roll, sample.timestamp]).all()
                and -0.1 <= now - sample.timestamp <= 0.5
            )
        except (TypeError, ValueError):
            connected = False
        if not connected:
            self.tracking_label.configure(text="Waiting for tracking…", text_color=MUTED)
            if self.session is not None:
                self.session.interrupt()
        elif not sample.tracking_valid:
            self.tracking_label.configure(text="Face tracking unavailable", text_color=MUTED)
        elif not sample.eyes_open:
            self.tracking_label.configure(text="Open both eyes to continue", text_color=MUTED)
        else:
            self.tracking_label.configure(text="Tracking connected", text_color=ACCENT)
        if self.session is not None and self.session.complete:
            self._finish_calibration()
        else:
            self._draw_stage()
        self._poll_id = self.after(40, self._poll_tracking)

    def _finish_calibration(self):
        try:
            profile = self.session.build_profile(
                trigger_degrees=float(self.threshold.get().rstrip("°")),
                camera=self.camera_metadata,
                coordinate_space=self.coordinate_space,
            )
            save_profile(self.profile_path, profile)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._close_stage()
            self.message.configure(text=f"Calibration could not be saved. {error}", text_color="#f5b492")
            return
        self.profile = profile
        self._close_stage()
        self._refresh_dashboard()

    def _close_stage(self):
        if self._fullscreen_id is not None:
            self.after_cancel(self._fullscreen_id)
            self._fullscreen_id = None
        if self.stage is not None:
            self.stage.destroy()
            self.deiconify()
            self.state("zoomed")
        self.stage = None
        self.session = None
        self.start_button.configure(state="normal")
        self.threshold.configure(state="normal")
        self.focus_force()

    def cancel_calibration(self):
        if self.stage is not None:
            self._close_stage()
            self.message.configure(text="Calibration cancelled. You can start again when ready.", text_color=MUTED)

    def close(self):
        self._closed = True
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        if self._start_id is not None:
            self.after_cancel(self._start_id)
        if self._fullscreen_id is not None:
            self.after_cancel(self._fullscreen_id)
        if self._maximize_id is not None:
            self.after_cancel(self._maximize_id)
        self.destroy()
