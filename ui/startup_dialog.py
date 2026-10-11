from pathlib import Path
import customtkinter as ctk

from core.camera import list_available_cameras

BACKGROUND = "#10151e"
SURFACE = "#192230"
ACCENT = "#66dec1"
TEXT = "#eff5fa"
MUTED = "#9eafc3"


class StartupDialog(ctk.CTk):
    def __init__(self, profile=None, default_camera=0, camera_list=None):
        ctk.set_appearance_mode("dark")
        super().__init__(fg_color=BACKGROUND)
        self.title("Eye & Head Tracking")
        self.geometry("520x500")
        self.resizable(False, False)
        self.choice = "exit"
        self.profile = profile

        # Default camera preference from saved profile if present
        if profile and isinstance(profile, dict) and "camera" in profile and "index" in profile["camera"]:
            default_camera = profile["camera"]["index"]
        self.camera_index = int(default_camera)

        # Populate camera list
        if camera_list is not None:
            self.cameras = list(camera_list)
        else:
            self.cameras = list_available_cameras()

        if not self.cameras:
            self.cameras = [{
                "index": self.camera_index,
                "name": f"Camera {self.camera_index}",
                "width": 0,
                "height": 0,
                "label": f"Camera {self.camera_index} (Default)",
            }]

        self._label_to_index = {c["label"]: c["index"] for c in self.cameras}
        self._index_to_label = {c["index"]: c["label"] for c in self.cameras}

        selected_label = self._index_to_label.get(self.camera_index, self.cameras[0]["label"])
        self.camera_index = self._label_to_index.get(selected_label, self.camera_index)

        card = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=18)
        card.pack(fill="both", expand=True, padx=24, pady=24)

        if profile:
            monitor = profile.get("monitor", {}) if isinstance(profile, dict) else {}
            w = monitor.get("width", self.winfo_screenwidth())
            h = monitor.get("height", self.winfo_screenheight())
            header_text = "CALIBRATION PROFILE FOUND"
            title_text = "Eye & Head Tracking"
            desc_text = (
                f"A complete profile is already saved for your display ({w} × {h}).\n"
                "Select your webcam and choose an action below:"
            )
        else:
            header_text = "WEBCAM SETUP & CALIBRATION"
            title_text = "Eye & Head Tracking"
            desc_text = (
                "No calibration profile was found for your display.\n"
                "Select your webcam and redo calibration to begin tracking:"
            )

        ctk.CTkLabel(card, text=header_text, text_color=ACCENT, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=24, pady=(20, 4))
        ctk.CTkLabel(card, text=title_text, font=("Segoe UI", 24, "bold"), text_color=TEXT).pack(anchor="w", padx=24)
        ctk.CTkLabel(
            card,
            text=desc_text,
            justify="left", text_color=MUTED, font=("Segoe UI", 13),
        ).pack(anchor="w", padx=24, pady=(8, 14))

        # Webcam selection row
        cam_section = ctk.CTkFrame(card, fg_color="transparent")
        cam_section.pack(fill="x", padx=24, pady=(0, 16))

        ctk.CTkLabel(
            cam_section,
            text="Webcam:",
            font=("Segoe UI", 13, "bold"),
            text_color=TEXT,
        ).pack(side="left", padx=(0, 10))

        self.camera_var = ctk.StringVar(value=selected_label)
        self.camera_dropdown = ctk.CTkOptionMenu(
            cam_section,
            variable=self.camera_var,
            values=[c["label"] for c in self.cameras],
            command=self._on_camera_selected,
            fg_color="#212d3d",
            button_color="#2d3d52",
            button_hover_color="#3a4f69",
            dropdown_fg_color="#192230",
            dropdown_hover_color="#273447",
            dropdown_text_color=TEXT,
            text_color=TEXT,
            font=("Segoe UI", 13),
            dropdown_font=("Segoe UI", 13),
            height=34,
        )
        self.camera_dropdown.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.rescan_button = ctk.CTkButton(
            cam_section,
            text="Rescan",
            width=64,
            height=34,
            command=self._rescan_cameras,
            fg_color="transparent",
            border_width=1,
            border_color="#405169",
            text_color=TEXT,
            hover_color="#273447",
            font=("Segoe UI", 12),
        )
        self.rescan_button.pack(side="right")

        self.track_button = ctk.CTkButton(
            card,
            text="Start Tracking",
            command=self._choose_track,
            fg_color=ACCENT if profile else "#212d3d",
            hover_color="#43bea2" if profile else "#2a394c",
            text_color=BACKGROUND if profile else MUTED,
            border_width=0 if profile else 1,
            border_color="#334255",
            font=("Segoe UI", 15, "bold" if profile else "normal"),
            height=42,
        )
        self.track_button.pack(fill="x", padx=24, pady=(0, 10))

        self.recalib_button = ctk.CTkButton(
            card,
            text="Redo Calibration",
            command=self._choose_calibrate,
            fg_color=ACCENT if not profile else "transparent",
            border_width=0 if not profile else 1,
            border_color="#405169",
            text_color=BACKGROUND if not profile else TEXT,
            hover_color="#43bea2" if not profile else "#273447",
            font=("Segoe UI", 14, "bold" if not profile else "normal"),
            height=38,
        )
        self.recalib_button.pack(fill="x", padx=24, pady=(0, 10))
        self.calib_button = self.recalib_button

        self.status_label = ctk.CTkLabel(
            card, text="", wraplength=460, font=("Segoe UI", 12), text_color="#f5b492"
        )
        self.status_label.pack(fill="x", padx=24, pady=(0, 6))

        if profile:
            self.bind("<Return>", lambda event: self._choose_track())
        else:
            self.bind("<Return>", lambda event: self._choose_calibrate())

        ctk.CTkButton(
            card, text="Exit", command=self._choose_exit,
            fg_color="transparent", text_color=MUTED, hover_color="#212d3d", height=30,
        ).pack(fill="x", padx=24)

        self.protocol("WM_DELETE_WINDOW", self._choose_exit)
        self.bind("<Escape>", lambda event: self._choose_exit())

    def _on_camera_selected(self, selected_label):
        self.camera_index = self._label_to_index.get(selected_label, self.camera_index)

    def _rescan_cameras(self):
        new_cameras = list_available_cameras()
        if new_cameras:
            self.cameras = new_cameras
            self._label_to_index = {c["label"]: c["index"] for c in self.cameras}
            self._index_to_label = {c["index"]: c["label"] for c in self.cameras}
            labels = [c["label"] for c in self.cameras]
            self.camera_dropdown.configure(values=labels)
            if self.camera_index in self._index_to_label:
                self.camera_var.set(self._index_to_label[self.camera_index])
            else:
                self.camera_index = self.cameras[0]["index"]
                self.camera_var.set(self.cameras[0]["label"])

    def _choose_track(self):
        if not self.profile:
            self.status_label.configure(
                text="Calibration file is not full yet. Please select Redo Calibration first.",
                text_color="#f5b492",
            )
            return
        self.choice = "track"
        self.destroy()

    def _choose_calibrate(self):
        self.choice = "calibrate"
        self.destroy()

    def _choose_exit(self):
        self.choice = "exit"
        self.destroy()


def prompt_startup_choice(profile=None, default_camera=0, camera_list=None, return_camera=False):
    dialog = StartupDialog(profile=profile, default_camera=default_camera, camera_list=camera_list)
    dialog.mainloop()
    if return_camera:
        return dialog.choice, dialog.camera_index
    return dialog.choice
