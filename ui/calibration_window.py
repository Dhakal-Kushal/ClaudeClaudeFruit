import sys
from copy import deepcopy
from pathlib import Path
from queue import Empty
from time import monotonic
import tkinter as tk

import customtkinter as ctk
import numpy as np

from core.calibration import (
    CALIBRATION_STEPS,
    EYE_RELATIVE_COORDINATE_SPACE,
    CalibrationSample,
    CalibrationSession,
    TrackingAdapter,
    load_profile,
    save_profile,
    screen_targets_in_canvas,
    camera_compatibility_errors,
    profile_compatibility_errors,
    validate_camera_metadata,
)
from core.validation import ValidationSession


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


def _physical_screen_size():
    """Return (width, height) in physical pixels on Windows, else None."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except (AttributeError, OSError):
                pass
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except (AttributeError, OSError):
            pass
    return None


class CalibrationWindow(ctk.CTk):
    def __init__(
        self,
        adapter=None,
        profile_path=None,
        camera_metadata=None,
        coordinate_space=EYE_RELATIVE_COORDINATE_SPACE,
        auto_start=True,
        pipeline_errors=None,
        camera_metadata_provider=None,
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
        self.camera_metadata_provider = camera_metadata_provider
        self._camera_checked = False
        self._waiting_since = None
        self._start_requested = False
        self.phase = "waiting_camera"
        self.candidate_profile = None
        self._target_since = None
        self._drawn_step = None
        self._stage_ready = False
        self.coordinate_space = coordinate_space
        self.pipeline_errors = pipeline_errors
        self._pipeline_error = None
        physical = _physical_screen_size()
        if physical:
            self.monitor = {"width": physical[0], "height": physical[1], "x": 0, "y": 0}
        else:
            self.monitor = {"width": self.winfo_screenwidth(), "height": self.winfo_screenheight(), "x": 0, "y": 0}
        self.profile = load_profile(self.profile_path, self.monitor)
        if self.profile and (
            self.profile["mapping"]["coordinate_space"] != coordinate_space
            or (camera_metadata is not None and camera_compatibility_errors(self.profile["camera"], camera_metadata))
        ):
            self.profile = None
        self.session = None
        self.stage = None
        self.last_sample = None
        self._closed = False
        self.cancelled = False
        self._auto_start = auto_start
        self._poll_id = None
        self._start_id = None
        self._fullscreen_id = None
        self._maximize_id = None
        self.protocol("WM_DELETE_WINDOW", self._cancel_and_close)
        self.bind("<Escape>", lambda event: self.cancel_calibration())
        self._build_dashboard()
        self._refresh_dashboard()
        self._poll_id = self.after(40, self._poll_tracking)
        if auto_start:
            self.withdraw()
            self._start_id = self.after(0, self.start_calibration)
        else:
            self._maximize_id = self.after(0, self._maximize_window)

    def _maximize_window(self):
        self._maximize_id = None
        if not self._closed and self.stage is None:
            self.state("zoomed")

    def _build_dashboard(self):
        card = ctk.CTkScrollableFrame(self, fg_color=SURFACE, corner_radius=20)
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

        self.tracking_label = ctk.CTkLabel(card, text="Waiting for tracking…", text_color=MUTED, font=("Segoe UI", 13))
        self.tracking_label.pack(anchor="w", padx=28, pady=(20, 4))
        self.message = ctk.CTkLabel(card, text="", wraplength=520, justify="left", anchor="w", text_color=MUTED)
        self.message.pack(fill="x", padx=28, pady=(0, 12))
        self.start_button = ctk.CTkButton(
            card, text="Start calibration", command=self.start_calibration,
            fg_color=ACCENT, hover_color="#43bea2", text_color=BACKGROUND,
            font=("Segoe UI", 15, "bold"), height=44,
        )
        self.start_button.pack(fill="x", padx=28, pady=(0, 10))
        self.recalib_button = ctk.CTkButton(
            card, text="Redo calibration", command=self.start_calibration,
            fg_color="transparent", border_width=1, border_color="#405169",
            text_color=TEXT, hover_color="#273447", font=("Segoe UI", 14), height=38,
        )
        self.close_button = ctk.CTkButton(
            card, text="Exit", command=self._cancel_and_close,
            fg_color="transparent", text_color=MUTED, hover_color="#212d3d", height=30,
        )
        self.close_button.pack(fill="x", padx=28, pady=(0, 24))

    def _cancel_and_close(self):
        self.cancelled = True
        self.close()

    def _refresh_dashboard(self):
        resolution = f"Primary display · {self.monitor['width']} × {self.monitor['height']}"
        if self.profile:
            self.summary.configure(text=f"{resolution}\n\n✓ Five eye positions captured\n✓ Neutral head position recorded\n✓ Calibration saved")
            self.start_button.configure(text="Start tracking", command=self._use_saved_profile)
            self.recalib_button.pack(fill="x", padx=28, pady=(0, 10), before=self.close_button)
            self.close_button.configure(text="Exit")
            self.message.configure(text="Your saved calibration is ready. Click Start Tracking to begin.", text_color=ACCENT)
        else:
            self.summary.configure(text=f"{resolution}\n\nSix calibration steps, then eight validation targets\nReview accuracy before saving. Esc cancels.")
            self.start_button.configure(text="Start calibration", command=self.start_calibration)
            self.recalib_button.pack_forget()
            self.close_button.configure(text="Exit")
            self.message.configure(text="A complete calibration is needed before eye control can start.", text_color=MUTED)

    def start_calibration(self):
        if self._start_id is not None:
            self.after_cancel(self._start_id)
        self._start_id = None
        if self.stage is not None or self._closed or self._pipeline_error:
            return
        self._start_requested = True
        if not self._check_camera():
            return
        self._start_requested = False
        self.cancelled = False
        self.candidate_profile = None
        self.phase = "calibration"
        self._drawn_step = None
        self._stage_ready = False
        self.session = CalibrationSession(self.monitor)
        self.stage = tk.Toplevel(self, bg=BACKGROUND)
        self.stage.withdraw()
        self.stage.protocol("WM_DELETE_WINDOW", self._cancel_and_close)
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
        self._draw_stage()

    def _enter_stage_fullscreen(self, stage):
        self._fullscreen_id = None
        if stage is self.stage and not self._closed:
            stage.state("zoomed")
            stage.update_idletasks()
            stage.attributes("-fullscreen", True)
            stage.lift()
            self._stage_ready = True
            self._draw_stage()

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
        if self.phase == "validation":
            point = self.session.targets[step]
            x = (point[0] - self.monitor["x"]) * width / self.monitor["width"]
            y = (point[1] - self.monitor["y"]) * height / self.monitor["height"]
            label, count = "Look at the validation dot", len(self.session.targets)
        else:
            targets = screen_targets_in_canvas(self.monitor, width, height)
            x, y = targets["center" if step == "neutral_head" else step]
            label, count = STEP_LABELS[step], len(CALIBRATION_STEPS)
        self.canvas.create_text(width / 2, 56, text=f"{self.phase.upper()}  ·  {self.session.step_index + 1} / {count}", fill=MUTED, font=("Segoe UI", 14, "bold"))
        self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20, outline="#345c58", width=2)
        self.canvas.create_oval(x - 9, y - 9, x + 9, y + 9, fill=ACCENT, outline="")
        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=TEXT, outline="")
        if step in ("neutral_head", "center"):
            text_y = height * 0.15
        elif y < height / 2:
            text_y = height * 0.75
        else:
            text_y = height * 0.25
        self.canvas.create_text(width / 2, text_y, text=label, fill=TEXT, font=("Segoe UI", 23, "bold"), width=width - 100)
        self.canvas.create_text(width / 2, text_y + 48, text=self.session.status, fill=ACCENT, font=("Segoe UI", 15), width=width - 100)
        bar_width = min(280, width - 100)
        left, top = (width - bar_width) / 2, text_y + 82
        self.canvas.create_rectangle(left, top, left + bar_width, top + 5, fill="#29394a", outline="")
        self.canvas.create_rectangle(left, top, left + bar_width * self.session.progress, top + 5, fill=ACCENT, outline="")
        self.canvas.create_text(width / 2, height - 52, text="Keep your head still. Move only your eyes.   ·   Esc to cancel", fill=MUTED, font=("Segoe UI", 13))
        key = (self.phase, step, width, height)
        if self._stage_ready and self._drawn_step != key:
            self.session.interrupt()
            self._drawn_step = key
            self.canvas.update_idletasks()
            self._target_since = monotonic()

    def _show_error(self, message, *, fatal=False):
        self._close_stage()
        self._start_requested = False
        self.phase = "error"
        self.candidate_profile = None
        self.start_button.configure(text="Retry calibration", command=self.start_calibration,
                                    state="disabled" if fatal else "normal")
        self.recalib_button.pack_forget()
        self.message.configure(text=message, text_color="#f5b492")

    def _check_camera(self):
        if self._pipeline_error:
            return False
        if self.pipeline_errors is not None:
            try:
                error = self.pipeline_errors.get_nowait()
            except Empty:
                pass
            else:
                self._pipeline_error = str(error)
                self._show_error(self._pipeline_error, fatal=True)
                return False
        physical = _physical_screen_size()
        if physical and tuple(physical) != (self.monitor["width"], self.monitor["height"]):
            self._pipeline_error = "Display resolution changed. Restart calibration for the current display."
            self._show_error(self._pipeline_error, fatal=True)
            return False
        actual = self.camera_metadata_provider() if self.camera_metadata_provider else self.camera_metadata
        if actual is None:
            if self._waiting_since is None:
                self._waiting_since = monotonic()
            if monotonic() - self._waiting_since > 10:
                self._pipeline_error = "Camera startup timed out. Close this window and check the selected camera."
                self._show_error(self._pipeline_error, fatal=True)
            return False
        try:
            validate_camera_metadata(actual)
            if self._camera_checked:
                errors = camera_compatibility_errors(self.camera_metadata, actual)
                if errors:
                    raise ValueError("Camera settings changed: " + "; ".join(errors))
            else:
                self.camera_metadata = dict(actual)
                self._camera_checked = True
                if self.profile and profile_compatibility_errors(self.profile, self.monitor, actual):
                    self.profile = None
                    self._refresh_dashboard()
        except (TypeError, ValueError, KeyError) as error:
            self._pipeline_error = f"Camera compatibility check failed. {error}. Restart and recalibrate."
            self._show_error(self._pipeline_error, fatal=True)
            return False
        return True

    def _use_saved_profile(self):
        if self._check_camera() and self.profile:
            errors = profile_compatibility_errors(self.profile, self.monitor, self.camera_metadata)
            if errors:
                self._show_error("; ".join(errors))
            else:
                self.close()

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
                self._show_error(self._pipeline_error, fatal=True)
        if not self._pipeline_error and self._check_camera() and self._start_requested:
            self.start_calibration()
        if self._pipeline_error:
            self.tracking_label.configure(text="Tracking stopped", text_color=MUTED)
            self._poll_id = self.after(40, self._poll_tracking)
            return
        now = monotonic()
        capture_visible = self._stage_ready and self.stage is not None and self.stage.state() in ("normal", "zoomed")
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
                    if self._target_since is None or sample.timestamp < self._target_since:
                        continue
                    previous_step = self.session.step
                    self.session.accept(sample, now)
                    if self.session.step != previous_step:
                        break  # Show the next target before consuming more frames.
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
            if self.phase == "calibration":
                self._finish_calibration()
            elif self.phase == "validation":
                self._finish_validation()
        else:
            self._draw_stage()
        if not self._closed:
            self._poll_id = self.after(40, self._poll_tracking)

    def _finish_calibration(self):
        try:
            self.candidate_profile = self.session.build_profile(
                trigger_degrees=float(self.threshold.get().rstrip("°")),
                camera=self.camera_metadata,
                coordinate_space=self.coordinate_space,
            )
            self._start_validation()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._show_error(f"Calibration could not be validated. {error}")

    def _start_validation(self):
        self.session = ValidationSession(self.candidate_profile)
        self.phase = "validation"
        self._drawn_step = None
        self._draw_stage()

    def _finish_validation(self):
        report = self.session.build_report()
        self.candidate_profile["validation"] = report
        self._close_stage()
        self.phase = "results"
        lines = ["Independent validation: 8 targets (before Kalman smoothing)",
                 f"Median error: {report['median_error_px']:.0f} px ({report['median_error_percent_diagonal']:.2f}% of diagonal)",
                 f"90th percentile error: {report['p90_error_px']:.0f} px ({report['p90_error_percent_diagonal']:.2f}% of diagonal)",
                 f"Rejected frames: {report['rejected_sample_rate']:.1%}",
                 "Target     Bias X / Y (px)     RMS jitter (px)"]
        lines.extend(f"{index + 1}                 {item['bias_px'][0]:+.0f} / {item['bias_px'][1]:+.0f}                 {item['rms_jitter_px']:.0f}"
                     for index, item in enumerate(report["targets"]))
        self.summary.configure(text="\n".join(lines), font=("Segoe UI", 13))
        self.threshold.configure(state="disabled")
        self.message.configure(text="Smaller errors are better. Nothing has been saved yet. Use these results to decide whether to accept or recalibrate.", text_color=MUTED)
        self.start_button.configure(text="Use this calibration", command=self._accept_calibration, state="normal")
        self.recalib_button.configure(text="Retry calibration")
        self.recalib_button.pack(fill="x", padx=28, pady=(0, 10), before=self.close_button)
        self.close_button.configure(text="Cancel")

    def _accept_calibration(self):
        if self.phase != "results" or self.candidate_profile is None or not self._check_camera():
            return
        profile = deepcopy(self.candidate_profile)
        profile["validation"]["accepted"] = True
        try:
            errors = profile_compatibility_errors(profile, self.monitor, self.camera_metadata)
            if errors:
                raise ValueError("; ".join(errors))
            save_profile(self.profile_path, profile)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.message.configure(text=f"Calibration could not be saved. {error}", text_color="#f5b492")
            return
        self.profile = profile
        self.close()

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
        self._stage_ready = False
        self._target_since = None
        self.start_button.configure(state="normal")
        self.focus_force()

    def cancel_calibration(self):
        self.cancelled = True
        self.candidate_profile = None
        if getattr(self, "_auto_start", False):
            self.close()
        else:
            if self.stage is not None:
                self._close_stage()
                self.phase = "idle"
                self._refresh_dashboard()
                self.message.configure(text="Calibration cancelled. You can start again when ready.", text_color=MUTED)
            else:
                self.close()

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
