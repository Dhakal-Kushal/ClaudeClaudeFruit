from pathlib import Path
import customtkinter as ctk

BACKGROUND = "#10151e"
SURFACE = "#192230"
ACCENT = "#66dec1"
TEXT = "#eff5fa"
MUTED = "#9eafc3"


class StartupDialog(ctk.CTk):
    def __init__(self, profile=None):
        ctk.set_appearance_mode("dark")
        super().__init__(fg_color=BACKGROUND)
        self.title("Eye & Head Tracking")
        self.geometry("520x400")
        self.resizable(False, False)
        self.choice = "exit"

        monitor = profile.get("monitor", {}) if profile else {}
        w = monitor.get("width", self.winfo_screenwidth())
        h = monitor.get("height", self.winfo_screenheight())

        card = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=18)
        card.pack(fill="both", expand=True, padx=24, pady=24)

        ctk.CTkLabel(card, text="CALIBRATION PROFILE FOUND", text_color=ACCENT, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=24, pady=(22, 4))
        ctk.CTkLabel(card, text="Choose an Action", font=("Segoe UI", 24, "bold"), text_color=TEXT).pack(anchor="w", padx=24)
        ctk.CTkLabel(
            card,
            text=f"A complete profile is already saved for your display ({w} × {h}).\nWould you like to start tracking or redo the calibration?",
            justify="left", text_color=MUTED, font=("Segoe UI", 13),
        ).pack(anchor="w", padx=24, pady=(10, 20))

        self.track_button = ctk.CTkButton(
            card, text="Start Tracking", command=self._choose_track,
            fg_color=ACCENT, hover_color="#43bea2", text_color=BACKGROUND,
            font=("Segoe UI", 15, "bold"), height=42,
        )
        self.track_button.pack(fill="x", padx=24, pady=(0, 10))

        self.recalib_button = ctk.CTkButton(
            card, text="Redo Calibration", command=self._choose_calibrate,
            fg_color="transparent", border_width=1, border_color="#405169",
            text_color=TEXT, hover_color="#273447", font=("Segoe UI", 14), height=38,
        )
        self.recalib_button.pack(fill="x", padx=24, pady=(0, 10))

        ctk.CTkButton(
            card, text="Exit", command=self._choose_exit,
            fg_color="transparent", text_color=MUTED, hover_color="#212d3d", height=30,
        ).pack(fill="x", padx=24)

        self.protocol("WM_DELETE_WINDOW", self._choose_exit)
        self.bind("<Escape>", lambda event: self._choose_exit())
        self.bind("<Return>", lambda event: self._choose_track())

    def _choose_track(self):
        self.choice = "track"
        self.destroy()

    def _choose_calibrate(self):
        self.choice = "calibrate"
        self.destroy()

    def _choose_exit(self):
        self.choice = "exit"
        self.destroy()


def prompt_startup_choice(profile=None):
    dialog = StartupDialog(profile=profile)
    dialog.mainloop()
    return dialog.choice
