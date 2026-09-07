"""
Woeyyy - Desktop GUI
Microphone enhancer, voice effects, and soundboard.
"""

import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog
from typing import Dict, List, Optional
import customtkinter as ctk
from PIL import Image, ImageTk

# Ensure root path is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.audio_engine import AudioDeviceManager, MicBoostEngine
from engine.profiles import DEFAULT_PROFILE_KEY, SOUND_PROFILES
from engine.soundboard import GlobalHotkeyManager, ProceduralSoundGenerator
from engine.voice_changer import VoiceChangerEngine
from engine.security import SingleInstanceLock, secure_file_permissions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, ".config.json")


def load_app_config() -> dict:
    """Load persistent user settings for Woeyyy. Creates default config file if missing."""
    defaults = {
        "input_device_name": "",
        "output_device_name": "",
        "monitor_device_name": "",
        "monitor_enabled": False,
        "engine_enabled": False,
        "gain_db": 0.0,
        "max_gain_db": 60.0,
        "profile": DEFAULT_PROFILE_KEY,
        "limiter_enabled": True,
        "mute": False,
        "vc_enabled": False,
        "vc_preset": "bypass",
        "vc_pitch": 0.0,
        "vc_formant": 0.0,
        "vc_chest_cut": False,
        "vc_breathiness": 0.0,
        "vc_mix": 1.0,
        "sb_volume": 1.0,
        "sb_hotkeys_enabled": False,
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                defaults.update(saved)
        except Exception as e:
            print(f"[App] Notice: could not load config, recreating with defaults: {e}")
            save_app_config(defaults)
    else:
        # Guarantee config file is immediately created so it's never missing or not found
        save_app_config(defaults)

    # Always ensure live streaming, monitoring, and voice changer power start in the OFF state on app launch
    defaults["engine_enabled"] = False
    defaults["monitor_enabled"] = False
    defaults["vc_enabled"] = False
    defaults["mute"] = False

    return defaults


def save_app_config(cfg: dict):
    """Save persistent user settings for Woeyyy and lock permissions."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(CONFIG_FILE)), exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        secure_file_permissions(CONFIG_FILE)
    except Exception as e:
        print(f"[App] Notice: could not save config: {e}")

# Set CustomTkinter theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Design Palette (Refactoring UI: Casual + Reserved Developer/Audio Utility)
BG_MAIN = "#0f172a"          # Slate 900 foundation
BG_CARD = "#1e293b"          # Slate 800 card surface
BG_CARD_SUBTLE = "#0f172a"   # Nested container surface
BORDER_SUBTLE = "#334155"    # Slate 700 1px container border

TEXT_PRIMARY = "#f8fafc"     # High-contrast heading/body
TEXT_SECONDARY = "#94a3b8"   # Balanced metadata/readout
TEXT_MUTED = "#64748b"       # De-emphasized labels

# Functional Colors (Restrained single accents)
COLOR_EMERALD = "#10b981"
COLOR_EMERALD_HOVER = "#059669"
BG_EMERALD_TINT = "#064e3b"

COLOR_BLUE = "#3b82f6"
COLOR_BLUE_HOVER = "#2563eb"

COLOR_ROSE = "#ef4444"
COLOR_ROSE_HOVER = "#dc2626"

COLOR_AMBER = "#f59e0b"
BG_AMBER_TINT = "#451a03"

BTN_SECONDARY = "#334155"
BTN_SECONDARY_HOVER = "#475569"

RADIUS_CARD = 8
RADIUS_BTN = 6


class VUMeterCanvas(tk.Canvas):
    """
    Responsive canvas audio VU meter.
    Displays level in dBFS [-60 dBFS to 0 dBFS] with professional
    green/yellow/red color zones and decaying peak-hold indicator.
    Dynamically resizes to match container width.
    """

    def __init__(self, parent, width: int = 340, height: int = 20, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=BG_CARD_SUBTLE,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self.meter_width = width
        self.meter_height = height

        # Internal state
        self.current_db = -60.0
        self.peak_db = -60.0
        self.peak_hold_frames = 0
        self.min_db = -60.0
        self.max_db = 0.0

        self.bind("<Configure>", self._on_configure)
        self._draw_meter(-60.0, -60.0)

    def _on_configure(self, event):
        if event.width > 20:
            self.meter_width = event.width
            self.meter_height = event.height
            self._draw_meter(self.current_db, self.peak_db)

    def update_level(self, rms_db: float, peak_db: Optional[float] = None):
        """Update meter with new dBFS values and redraw canvas."""
        self.current_db = max(self.min_db, min(self.max_db, rms_db))

        # Peak hold calculation
        in_peak = rms_db if peak_db is None else peak_db
        if in_peak >= self.peak_db:
            self.peak_db = min(self.max_db, in_peak)
            self.peak_hold_frames = 15
        else:
            if self.peak_hold_frames > 0:
                self.peak_hold_frames -= 1
            else:
                self.peak_db = max(self.min_db, self.peak_db - 1.5)

        self._draw_meter(self.current_db, self.peak_db)

    def _db_to_x(self, db: float) -> float:
        """Convert dBFS [-60, 0] to pixel X coordinate."""
        clamped = max(self.min_db, min(self.max_db, db))
        norm = (clamped - self.min_db) / (self.max_db - self.min_db)
        return norm * (self.meter_width - 4) + 2

    def _draw_meter(self, rms_db: float, peak_db: float):
        self.delete("all")

        # Track background
        self.create_rectangle(
            0, 0, self.meter_width, self.meter_height,
            outline="", fill=BG_CARD_SUBTLE
        )

        fill_x = self._db_to_x(rms_db)
        if fill_x > 2:
            x_green = self._db_to_x(-18.0)
            x_amber = self._db_to_x(-6.0)

            # Draw Green zone
            x1 = min(fill_x, x_green)
            if x1 > 2:
                self.create_rectangle(2, 2, x1, self.meter_height - 2, fill=COLOR_EMERALD, outline="")

            # Draw Amber zone
            if fill_x > x_green:
                x2 = min(fill_x, x_amber)
                self.create_rectangle(x_green, 2, x2, self.meter_height - 2, fill=COLOR_AMBER, outline="")

            # Draw Red zone
            if fill_x > x_amber:
                self.create_rectangle(x_amber, 2, fill_x, self.meter_height - 2, fill=COLOR_ROSE, outline="")

        # Peak hold line indicator
        peak_x = self._db_to_x(peak_db)
        if peak_x > 2:
            peak_color = "#f8fafc" if peak_db < -3.0 else COLOR_ROSE
            self.create_line(
                peak_x, 2, peak_x, self.meter_height - 2,
                fill=peak_color, width=2
            )

        # Scale markings (-36, -24, -12, -6)
        for tick_db in [-36, -24, -12, -6]:
            tx = self._db_to_x(tick_db)
            self.create_line(tx, self.meter_height - 5, tx, self.meter_height - 1, fill=BORDER_SUBTLE, width=1)


class WoeyyyApp(ctk.CTk):
    """
    Main application window for Woeyyy.
    Microphone enhancer, voice effects, and soundboard.
    """

    def __init__(self):
        super().__init__()

        # Load Saved State
        self.cfg = load_app_config()

        self.title("Woeyyy")
        self.geometry("980x800")
        self.minsize(920, 740)
        self.configure(fg_color=BG_MAIN)

        # Configure Windows taskbar icon integration
        try:
            import ctypes
            myappid = "woeyyy.audio.desktop.2.0"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

        # Load Window & Taskbar Icon
        icon_ico = os.path.join(BASE_DIR, "assets", "app_icon.ico")
        icon_png = os.path.join(BASE_DIR, "assets", "app_icon.png")

        if os.path.exists(icon_ico):
            try:
                self.iconbitmap(icon_ico)
            except Exception:
                pass
        if os.path.exists(icon_png):
            try:
                self._app_icon_tk = ImageTk.PhotoImage(file=icon_png)
                self.iconphoto(False, self._app_icon_tk)
            except Exception:
                pass

        # Fonts
        self.font_hero = ctk.CTkFont(family="Segoe UI", size=18, weight="bold")
        self.font_section = ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        self.font_label = ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
        self.font_body = ctk.CTkFont(family="Segoe UI", size=11)
        self.font_body_bold = ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
        self.font_btn = ctk.CTkFont(family="Segoe UI", size=11)
        self.font_caption = ctk.CTkFont(family="Segoe UI", size=10)
        self.font_readout = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")

        # Audio Engine Reference
        self.engine: Optional[MicBoostEngine] = None
        self.is_running = False
        self.input_devices_map: Dict[str, int] = {}
        self.output_devices_map: Dict[str, int] = {}
        self.monitor_devices_map: Dict[str, int] = {}

        self.max_gain_db = float(self.cfg.get("max_gain_db", 60.0))
        saved_prof = self.cfg.get("profile", DEFAULT_PROFILE_KEY)
        self.selected_profile_key = saved_prof if saved_prof in SOUND_PROFILES else DEFAULT_PROFILE_KEY
        self.profiles_by_name = {p.name: p.key for p in SOUND_PROFILES.values()}

        # Hotkeys Manager (disabled by default to prevent accidental triggers)
        self.hotkeys = GlobalHotkeyManager(enabled=bool(self.cfg.get("sb_hotkeys_enabled", False)))
        self.hotkeys.start()

        # Soundboard UI components cache
        self.sound_cards: Dict[str, Dict] = {}

        # Voice Changer UI State
        self.vc_preset_buttons: Dict[str, ctk.CTkButton] = {}

        # Setup UI layout
        self._build_ui()

        # Populate devices and preset sounds
        self._refresh_audio_devices()
        self._populate_default_soundboard()

        # Telemetry update loop (~30 FPS)
        self._telemetry_job = self.after(33, self._poll_telemetry)

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _build_ui(self):
        """Construct the clean desktop GUI layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # =========================================================================
        # 1. HEADER BAR
        # =========================================================================
        header_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        header_frame.grid(row=0, column=0, padx=16, pady=(12, 6), sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)

        title_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_box.pack(side="left", padx=16, pady=10)

        # App Icon Logo
        icon_png_path = os.path.join(BASE_DIR, "assets", "app_icon.png")
        if os.path.exists(icon_png_path):
            try:
                self._logo_img = ctk.CTkImage(
                    light_image=Image.open(icon_png_path),
                    dark_image=Image.open(icon_png_path),
                    size=(32, 32),
                )
                lbl_logo = ctk.CTkLabel(title_box, image=self._logo_img, text="")
                lbl_logo.pack(side="left", padx=(0, 10))
            except Exception:
                pass

        # Brand & Subtitle
        brand_box = ctk.CTkFrame(title_box, fg_color="transparent")
        brand_box.pack(side="left", fill="y")

        lbl_app_name = ctk.CTkLabel(
            brand_box,
            text="Woeyyy",
            font=self.font_hero,
            text_color=TEXT_PRIMARY,
        )
        lbl_app_name.pack(anchor="w")

        lbl_sub_name = ctk.CTkLabel(
            brand_box,
            text="Microphone, voice effects, and soundboard",
            font=self.font_caption,
            text_color=TEXT_MUTED,
        )
        lbl_sub_name.pack(anchor="w", pady=(1, 0))

        # Standard Clean On/Off Toggle in Header
        controls_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        controls_box.pack(side="right", padx=16, pady=10)

        self.lbl_engine_status = ctk.CTkLabel(
            controls_box,
            text="Off",
            font=self.font_btn,
            text_color=TEXT_MUTED,
            width=28,
        )
        self.lbl_engine_status.pack(side="right", padx=(4, 0))

        self.switch_engine = ctk.CTkSwitch(
            controls_box,
            text="Engine",
            font=self.font_btn,
            width=44,
            progress_color=COLOR_EMERALD,
            command=self._toggle_engine,
        )
        self.switch_engine.deselect()
        self.switch_engine.pack(side="right")

        # Alias for backward compatibility
        self.btn_toggle_stream = self.switch_engine

        # =========================================================================
        # 2. DEVICE ROUTING CARD
        # =========================================================================
        routing_card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        routing_card.grid(row=1, column=0, padx=16, pady=4, sticky="ew")
        routing_card.grid_columnconfigure((0, 1, 2), weight=1)

        # Input Mic
        in_box = ctk.CTkFrame(routing_card, fg_color="transparent")
        in_box.grid(row=0, column=0, padx=12, pady=10, sticky="ew")
        ctk.CTkLabel(in_box, text="Input", font=self.font_caption, text_color=TEXT_SECONDARY).pack(anchor="w", pady=(0, 2))
        self.opt_input = ctk.CTkOptionMenu(
            in_box,
            values=["Scanning..."],
            command=self._on_input_device_changed,
            font=self.font_body,
            dropdown_font=self.font_body,
            fg_color=BTN_SECONDARY,
            button_color=BORDER_SUBTLE,
            button_hover_color=BTN_SECONDARY_HOVER,
            dropdown_fg_color=BG_CARD,
            height=30,
            corner_radius=RADIUS_BTN,
        )
        self.opt_input.pack(fill="x")

        # Output (Virtual Cable / Game / Discord)
        out_box = ctk.CTkFrame(routing_card, fg_color="transparent")
        out_box.grid(row=0, column=1, padx=12, pady=10, sticky="ew")
        ctk.CTkLabel(out_box, text="Output", font=self.font_caption, text_color=TEXT_SECONDARY).pack(anchor="w", pady=(0, 2))
        self.opt_output = ctk.CTkOptionMenu(
            out_box,
            values=["Scanning..."],
            command=self._on_output_device_changed,
            font=self.font_body,
            dropdown_font=self.font_body,
            fg_color=BTN_SECONDARY,
            button_color=BORDER_SUBTLE,
            button_hover_color=BTN_SECONDARY_HOVER,
            dropdown_fg_color=BG_CARD,
            height=30,
            corner_radius=RADIUS_BTN,
        )
        self.opt_output.pack(fill="x")

        # Headphone Monitor
        mon_box = ctk.CTkFrame(routing_card, fg_color="transparent")
        mon_box.grid(row=0, column=2, padx=12, pady=10, sticky="ew")
        mon_header = ctk.CTkFrame(mon_box, fg_color="transparent")
        mon_header.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(mon_header, text="Monitor", font=self.font_caption, text_color=TEXT_SECONDARY).pack(side="left")
        self.switch_monitor = ctk.CTkSwitch(
            mon_header,
            text="Listen",
            font=self.font_caption,
            width=40,
            progress_color=COLOR_BLUE,
            command=self._on_monitor_toggled,
        )
        self.switch_monitor.deselect()
        self.switch_monitor.pack(side="right")
        self.opt_monitor = ctk.CTkOptionMenu(
            mon_box,
            values=["Scanning..."],
            command=self._on_monitor_device_changed,
            font=self.font_body,
            dropdown_font=self.font_body,
            fg_color=BTN_SECONDARY,
            button_color=BORDER_SUBTLE,
            button_hover_color=BTN_SECONDARY_HOVER,
            dropdown_fg_color=BG_CARD,
            height=30,
            corner_radius=RADIUS_BTN,
        )
        self.opt_monitor.pack(fill="x")

        # =========================================================================
        # 3. TABVIEW (MICROPHONE, VOICE EFFECTS, SOUNDBOARD)
        # =========================================================================
        self.tabview = ctk.CTkTabview(
            self,
            fg_color=BG_CARD,
            segmented_button_fg_color=BG_CARD_SUBTLE,
            segmented_button_selected_color=COLOR_BLUE,
            segmented_button_selected_hover_color=COLOR_BLUE_HOVER,
            segmented_button_unselected_color=BTN_SECONDARY,
            corner_radius=RADIUS_CARD,
            border_width=1,
            border_color=BORDER_SUBTLE,
        )
        self.tabview.grid(row=2, column=0, padx=16, pady=(4, 6), sticky="nsew")

        self.tab_mic = self.tabview.add("Microphone")
        self.tab_vc = self.tabview.add("Voice Effects")
        self.tab_sb = self.tabview.add("Soundboard")

        # Build each tab's UI
        self._build_mic_tab()
        self._build_voice_changer_tab()
        self._build_soundboard_tab()

        # =========================================================================
        # 4. FOOTER STATUS BAR
        # =========================================================================
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=3, column=0, padx=16, pady=(2, 8), sticky="ew")

        self.lbl_footer = ctk.CTkLabel(
            footer,
            text="48 kHz • 128 frames (~2.7 ms)",
            font=self.font_caption,
            text_color=TEXT_MUTED,
        )
        self.lbl_footer.pack(side="left")

        self.lbl_perf = ctk.CTkLabel(
            footer,
            text="Buffer drops: 0",
            font=self.font_caption,
            text_color=TEXT_MUTED,
        )
        self.lbl_perf.pack(side="right")

    # =========================================================================
    # TAB 1: MICROPHONE
    # =========================================================================
    def _build_mic_tab(self):
        tab = self.tab_mic
        tab.grid_columnconfigure(0, weight=1)

        # 1. Gain Card
        gain_card = ctk.CTkFrame(tab, fg_color=BG_CARD_SUBTLE, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        gain_card.pack(fill="x", padx=8, pady=4)

        gain_header = ctk.CTkFrame(gain_card, fg_color="transparent")
        gain_header.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(gain_header, text="Gain", font=self.font_label, text_color=TEXT_PRIMARY).pack(side="left")

        # Configurable Max Mic Boost
        max_box = ctk.CTkFrame(gain_header, fg_color="transparent")
        max_box.pack(side="left", padx=(16, 0))
        ctk.CTkLabel(max_box, text="Max:", font=self.font_caption, text_color=TEXT_MUTED).pack(side="left", padx=(0, 4))
        self.opt_max_gain = ctk.CTkOptionMenu(
            max_box,
            values=["20 dB", "30 dB", "36 dB", "48 dB", "60 dB"],
            width=75,
            height=22,
            font=self.font_caption,
            dropdown_font=self.font_caption,
            fg_color=BTN_SECONDARY,
            button_color=BORDER_SUBTLE,
            button_hover_color=BTN_SECONDARY_HOVER,
            dropdown_fg_color=BG_CARD,
            command=self._on_max_gain_changed,
        )
        self.opt_max_gain.set(f"{int(self.max_gain_db)} dB")
        self.opt_max_gain.pack(side="left")

        cur_gain = float(self.cfg.get("gain_db", 0.0))
        self.lbl_gain_display = ctk.CTkLabel(gain_header, text=f"+{cur_gain:.1f} dB", font=self.font_readout, text_color=COLOR_EMERALD)
        self.lbl_gain_display.pack(side="right")

        self.slider_gain = ctk.CTkSlider(
            gain_card,
            from_=0.0,
            to=self.max_gain_db,
            number_of_steps=int(self.max_gain_db * 10),
            command=self._on_gain_slider_changed,
            progress_color=COLOR_EMERALD,
            button_color=COLOR_EMERALD,
            button_hover_color=COLOR_EMERALD_HOVER,
            height=16,
        )
        self.slider_gain.set(min(cur_gain, self.max_gain_db))
        self.slider_gain.pack(fill="x", padx=12, pady=4)

        presets_box = ctk.CTkFrame(gain_card, fg_color="transparent")
        presets_box.pack(fill="x", padx=12, pady=(2, 8))
        for label, val in [("0 dB", 0.0), ("+12 dB", 12.0), ("+24 dB", 24.0), ("+36 dB", 36.0), ("+48 dB", 48.0)]:
            btn = ctk.CTkButton(
                presets_box,
                text=label,
                width=46,
                height=24,
                fg_color=BTN_SECONDARY,
                hover_color=BTN_SECONDARY_HOVER,
                font=self.font_caption,
                text_color=TEXT_SECONDARY,
                corner_radius=RADIUS_BTN,
                command=lambda v=val: self._set_gain_preset(v),
            )
            btn.pack(side="left", padx=2)

        btn_max = ctk.CTkButton(
            presets_box,
            text="Max",
            width=46,
            height=24,
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            font=self.font_caption,
            text_color=TEXT_SECONDARY,
            corner_radius=RADIUS_BTN,
            command=lambda: self._set_gain_preset(self.max_gain_db),
        )
        btn_max.pack(side="left", padx=2)

        # 2. Sound Profile & Limiter Card
        prof_card = ctk.CTkFrame(tab, fg_color=BG_CARD_SUBTLE, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        prof_card.pack(fill="x", padx=8, pady=4)

        prof_header = ctk.CTkFrame(prof_card, fg_color="transparent")
        prof_header.pack(fill="x", padx=12, pady=(8, 4))
        ctk.CTkLabel(prof_header, text="Preset Profile", font=self.font_label, text_color=TEXT_PRIMARY).pack(side="left")

        prof_row = ctk.CTkFrame(prof_card, fg_color="transparent")
        prof_row.pack(fill="x", padx=12, pady=2)

        profile_names = [p.name for p in SOUND_PROFILES.values()]
        self.opt_profile = ctk.CTkOptionMenu(
            prof_row,
            values=profile_names,
            command=self._on_profile_changed,
            font=self.font_body,
            dropdown_font=self.font_body,
            fg_color=BTN_SECONDARY,
            button_color=BORDER_SUBTLE,
            button_hover_color=BTN_SECONDARY_HOVER,
            dropdown_fg_color=BG_CARD,
            width=240,
            height=28,
            corner_radius=RADIUS_BTN,
        )
        self.opt_profile.set(SOUND_PROFILES[self.selected_profile_key].name)
        self.opt_profile.pack(side="left")

        # Dedicated profile description underneath to avoid clipping
        self.lbl_profile_desc = ctk.CTkLabel(
            prof_card,
            text=SOUND_PROFILES[self.selected_profile_key].description,
            font=self.font_caption,
            text_color=TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=640,
        )
        self.lbl_profile_desc.pack(fill="x", padx=12, pady=(2, 6))

        # Limiter row
        limiter_row = ctk.CTkFrame(prof_card, fg_color="transparent")
        limiter_row.pack(fill="x", padx=12, pady=(4, 8))

        is_limiter = bool(self.cfg.get("limiter_enabled", True))
        self.switch_limiter = ctk.CTkSwitch(
            limiter_row,
            text="Soft Limiter",
            font=self.font_body,
            progress_color=COLOR_EMERALD,
            command=self._on_limiter_toggled,
        )
        if is_limiter:
            self.switch_limiter.select()
        else:
            self.switch_limiter.deselect()
        self.switch_limiter.pack(side="left")

        self.limiter_badge = ctk.CTkLabel(
            limiter_row,
            text="Active" if is_limiter else "Off",
            font=self.font_caption,
            text_color=COLOR_EMERALD if is_limiter else TEXT_MUTED,
            fg_color=BG_EMERALD_TINT if is_limiter else BG_CARD,
            corner_radius=RADIUS_BTN,
            padx=8,
            pady=1,
        )
        self.limiter_badge.pack(side="left", padx=(8, 0))

        # 3. Levels (VU Meters) Card
        meter_card = ctk.CTkFrame(tab, fg_color=BG_CARD_SUBTLE, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        meter_card.pack(fill="x", padx=8, pady=4)
        meter_card.grid_columnconfigure((0, 1), weight=1)

        # Input Meter
        pre_box = ctk.CTkFrame(meter_card, fg_color="transparent")
        pre_box.grid(row=0, column=0, padx=12, pady=8, sticky="ew")
        pre_h = ctk.CTkFrame(pre_box, fg_color="transparent")
        pre_h.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(pre_h, text="Input Level", font=self.font_caption, text_color=TEXT_SECONDARY).pack(side="left")
        self.lbl_pre_db = ctk.CTkLabel(pre_h, text="-60.0 dBFS", font=self.font_caption, text_color=TEXT_SECONDARY)
        self.lbl_pre_db.pack(side="right")
        self.vu_pre = VUMeterCanvas(pre_box, width=300, height=16)
        self.vu_pre.pack(fill="x")

        # Output Meter
        post_box = ctk.CTkFrame(meter_card, fg_color="transparent")
        post_box.grid(row=0, column=1, padx=12, pady=8, sticky="ew")
        post_h = ctk.CTkFrame(post_box, fg_color="transparent")
        post_h.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(post_h, text="Output Level", font=self.font_caption, text_color=COLOR_EMERALD).pack(side="left")
        self.lbl_post_db = ctk.CTkLabel(post_h, text="-60.0 dBFS", font=self.font_caption, text_color=COLOR_EMERALD)
        self.lbl_post_db.pack(side="right")
        self.vu_post = VUMeterCanvas(post_box, width=300, height=16)
        self.vu_post.pack(fill="x")

        # Bottom Bar: Mic Mute
        bottom_row = ctk.CTkFrame(tab, fg_color="transparent")
        bottom_row.pack(fill="x", padx=8, pady=(4, 6))

        self.btn_mute = ctk.CTkButton(
            bottom_row,
            text="Mute",
            font=self.font_btn,
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            text_color=TEXT_PRIMARY,
            height=30,
            width=100,
            corner_radius=RADIUS_BTN,
            command=self._toggle_mute,
        )
        self.btn_mute.pack(side="right")

    # =========================================================================
    # TAB 2: VOICE EFFECTS
    # =========================================================================
    def _build_voice_changer_tab(self):
        """Build voice effects tab with clean controls and presets."""
        tab = self.tab_vc
        tab.grid_columnconfigure(0, weight=1)

        # 1. Master Control Card (Power Toggle & Mix)
        master_card = ctk.CTkFrame(tab, fg_color=BG_CARD_SUBTLE, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        master_card.pack(fill="x", padx=8, pady=4)

        m_header = ctk.CTkFrame(master_card, fg_color="transparent")
        m_header.pack(fill="x", padx=12, pady=(8, 4))

        ctk.CTkLabel(
            m_header,
            text="Voice Effects",
            font=self.font_label,
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        pwr_box = ctk.CTkFrame(m_header, fg_color="transparent")
        pwr_box.pack(side="right")

        vc_on = bool(self.cfg.get("vc_enabled", False))
        self.lbl_vc_status = ctk.CTkLabel(
            pwr_box,
            text="On" if vc_on else "Off",
            font=self.font_caption,
            text_color=COLOR_BLUE if vc_on else TEXT_MUTED,
            width=28,
        )
        self.lbl_vc_status.pack(side="right", padx=(4, 0))

        self.switch_vc = ctk.CTkSwitch(
            pwr_box,
            text="",
            width=40,
            progress_color=COLOR_BLUE,
            command=self._on_vc_power_toggled,
        )
        if vc_on:
            self.switch_vc.select()
        else:
            self.switch_vc.deselect()
        self.switch_vc.pack(side="right")

        # Mix Slider
        mix_row = ctk.CTkFrame(master_card, fg_color="transparent")
        mix_row.pack(fill="x", padx=12, pady=(4, 2))

        ctk.CTkLabel(mix_row, text="Mix", font=self.font_caption, text_color=TEXT_SECONDARY).pack(side="left")
        vc_mix = float(self.cfg.get("vc_mix", 1.0))
        self.lbl_vc_mix = ctk.CTkLabel(mix_row, text=f"{int(vc_mix * 100)}%", font=self.font_caption, text_color=COLOR_BLUE)
        self.lbl_vc_mix.pack(side="right")

        self.slider_vc_mix = ctk.CTkSlider(
            master_card,
            from_=0.0,
            to=1.0,
            number_of_steps=50,
            progress_color=COLOR_BLUE,
            button_color=COLOR_BLUE,
            command=self._on_vc_mix_changed,
            height=16,
        )
        self.slider_vc_mix.set(vc_mix)
        self.slider_vc_mix.pack(fill="x", padx=12, pady=(0, 8))

        # 2. Preset Selection Cards (3x2 Grid)
        preset_card = ctk.CTkFrame(tab, fg_color=BG_CARD_SUBTLE, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        preset_card.pack(fill="x", padx=8, pady=4)

        p_header = ctk.CTkFrame(preset_card, fg_color="transparent")
        p_header.pack(fill="x", padx=12, pady=(8, 4))
        ctk.CTkLabel(
            p_header,
            text="Presets",
            font=self.font_label,
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        grid_frame = ctk.CTkFrame(preset_card, fg_color="transparent")
        grid_frame.pack(fill="x", padx=10, pady=(2, 8))
        grid_frame.grid_columnconfigure((0, 1, 2), weight=1)

        presets = [
            ("Normal", "bypass", "Original mic audio"),
            ("Woman Voice", "woman", "Pitch +4 st & formant EQ"),
            ("Deep Voice", "deep_voice", "Pitch -5 semitones"),
            ("Chipmunk", "chipmunk", "Pitch +8 semitones"),
            ("Robot", "robot", "65 Hz ring modulation"),
            ("Radio", "radio", "Bandpass filter"),
            ("Monster", "monster", "Pitch -12 semitones"),
        ]

        saved_vc_preset = self.cfg.get("vc_preset", "bypass")
        self.vc_preset_buttons = {}
        for idx, (title, key, desc) in enumerate(presets):
            r = idx // 3
            c = idx % 3
            p_box = ctk.CTkFrame(grid_frame, fg_color=BG_CARD, corner_radius=RADIUS_BTN, border_width=1, border_color=BORDER_SUBTLE)
            p_box.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

            btn = ctk.CTkButton(
                p_box,
                text=title,
                font=self.font_btn,
                fg_color=COLOR_BLUE if key == saved_vc_preset else BTN_SECONDARY,
                hover_color=COLOR_BLUE_HOVER,
                height=30,
                corner_radius=RADIUS_BTN,
                command=lambda k=key: self._on_select_vc_preset(k),
            )
            btn.pack(fill="x", padx=6, pady=(6, 2))
            self.vc_preset_buttons[key] = btn

            ctk.CTkLabel(
                p_box,
                text=desc,
                font=self.font_caption,
                text_color=TEXT_MUTED,
            ).pack(padx=6, pady=(0, 6))

        # 3. Fine-Tuning Pitch & Formant Sliders (Roland VT-4 Vocal Tract)
        tuning_card = ctk.CTkFrame(tab, fg_color=BG_CARD_SUBTLE, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        tuning_card.pack(fill="x", padx=8, pady=4)

        # --- Pitch Shift Row ---
        t_header = ctk.CTkFrame(tuning_card, fg_color="transparent")
        t_header.pack(fill="x", padx=12, pady=(6, 2))

        ctk.CTkLabel(t_header, text="Pitch Shift (Tone / F0)", font=self.font_label, text_color=TEXT_PRIMARY).pack(side="left")
        vc_pitch = float(self.cfg.get("vc_pitch", 0.0))
        self.lbl_vc_pitch = ctk.CTkLabel(t_header, text=f"{vc_pitch:+.1f} st", font=self.font_readout, text_color=COLOR_BLUE)
        self.lbl_vc_pitch.pack(side="right")

        pitch_slider_box = ctk.CTkFrame(tuning_card, fg_color="transparent")
        pitch_slider_box.pack(fill="x", padx=12, pady=(1, 4))

        self.slider_vc_pitch = ctk.CTkSlider(
            pitch_slider_box,
            from_=-12.0,
            to=12.0,
            number_of_steps=48,
            progress_color=COLOR_BLUE,
            button_color=COLOR_BLUE,
            command=self._on_vc_pitch_changed,
            height=16,
        )
        self.slider_vc_pitch.set(vc_pitch)
        self.slider_vc_pitch.pack(side="left", fill="x", expand=True, padx=(0, 8))

        btn_reset_pitch = ctk.CTkButton(
            pitch_slider_box,
            text="Reset",
            width=55,
            height=22,
            font=self.font_caption,
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            command=lambda: self._set_custom_pitch(0.0),
        )
        btn_reset_pitch.pack(side="right")

        # --- Formant Shift Row (Vocal Tract Length) ---
        f_header = ctk.CTkFrame(tuning_card, fg_color="transparent")
        f_header.pack(fill="x", padx=12, pady=(4, 2))

        ctk.CTkLabel(f_header, text="Formant Shift (Vocal Tract Character)", font=self.font_label, text_color=TEXT_PRIMARY).pack(side="left")
        vc_formant = float(self.cfg.get("vc_formant", 0.0))
        self.lbl_vc_formant = ctk.CTkLabel(f_header, text=f"{vc_formant:+.1f} st", font=self.font_readout, text_color=COLOR_EMERALD)
        self.lbl_vc_formant.pack(side="right")

        formant_slider_box = ctk.CTkFrame(tuning_card, fg_color="transparent")
        formant_slider_box.pack(fill="x", padx=12, pady=(1, 4))

        self.slider_vc_formant = ctk.CTkSlider(
            formant_slider_box,
            from_=-12.0,
            to=12.0,
            number_of_steps=48,
            progress_color=COLOR_EMERALD,
            button_color=COLOR_EMERALD,
            command=self._on_vc_formant_changed,
            height=16,
        )
        self.slider_vc_formant.set(vc_formant)
        self.slider_vc_formant.pack(side="left", fill="x", expand=True, padx=(0, 8))

        btn_reset_formant = ctk.CTkButton(
            formant_slider_box,
            text="Reset",
            width=55,
            height=22,
            font=self.font_caption,
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            command=lambda: self._set_custom_formant(0.0),
        )
        btn_reset_formant.pack(side="right")

        # --- Tonal Enhancement Checkbox ---
        enhance_box = ctk.CTkFrame(tuning_card, fg_color="transparent")
        enhance_box.pack(fill="x", padx=12, pady=(2, 6))

        self.chk_chest_cut = ctk.CTkCheckBox(
            enhance_box,
            text="Chest Cut (Eliminate deep male chest resonance for natural female voice)",
            font=self.font_caption,
            command=self._on_vc_chest_cut_toggled,
            checkbox_width=16,
            checkbox_height=16,
        )
        if bool(self.cfg.get("vc_chest_cut", False)):
            self.chk_chest_cut.select()
        else:
            self.chk_chest_cut.deselect()
        self.chk_chest_cut.pack(side="left")

    # =========================================================================
    # TAB 3: SOUNDBOARD
    # =========================================================================
    def _build_soundboard_tab(self):
        tab = self.tab_sb
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Toolbar
        tb = ctk.CTkFrame(tab, fg_color=BG_CARD_SUBTLE, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        tb.grid(row=0, column=0, padx=8, pady=4, sticky="ew")

        # Volume
        ctk.CTkLabel(tb, text="Volume", font=self.font_caption, text_color=TEXT_SECONDARY).pack(side="left", padx=(12, 6), pady=6)
        sb_vol = float(self.cfg.get("sb_volume", 1.0))
        self.slider_sb_vol = ctk.CTkSlider(
            tb,
            from_=0.0,
            to=2.0,
            number_of_steps=200,
            width=120,
            height=14,
            command=self._on_soundboard_volume_changed,
            progress_color=COLOR_BLUE,
            button_color=COLOR_BLUE,
            button_hover_color=COLOR_BLUE_HOVER,
        )
        self.slider_sb_vol.set(sb_vol)
        self.slider_sb_vol.pack(side="left", padx=4)
        self.lbl_sb_vol = ctk.CTkLabel(tb, text=f"{int(sb_vol * 100)}%", font=self.font_caption, text_color=TEXT_MUTED, width=38)
        self.lbl_sb_vol.pack(side="left", padx=2)

        # Global Hotkeys Switch
        hk_box = ctk.CTkFrame(tb, fg_color="transparent")
        hk_box.pack(side="left", padx=(14, 4), pady=4)
        ctk.CTkLabel(hk_box, text="Hotkeys (Ctrl+1..8)", font=self.font_caption, text_color=TEXT_SECONDARY).pack(side="left", padx=(0, 6))
        self.switch_sb_hotkeys = ctk.CTkSwitch(
            hk_box,
            text="",
            width=36,
            progress_color=COLOR_BLUE,
            command=self._on_sb_hotkeys_toggled,
        )
        if self.cfg.get("sb_hotkeys_enabled", False):
            self.switch_sb_hotkeys.select()
        else:
            self.switch_sb_hotkeys.deselect()
        self.switch_sb_hotkeys.pack(side="left")

        # Stop All Button
        btn_panic = ctk.CTkButton(
            tb,
            text="Stop all",
            font=self.font_btn,
            fg_color=BTN_SECONDARY,
            hover_color=COLOR_ROSE_HOVER,
            text_color=TEXT_PRIMARY,
            height=26,
            width=80,
            corner_radius=RADIUS_BTN,
            command=self._on_soundboard_stop_all,
        )
        btn_panic.pack(side="right", padx=(4, 10), pady=6)

        # Add Custom Sound Button
        btn_add = ctk.CTkButton(
            tb,
            text="Add sound",
            font=self.font_btn,
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            text_color=TEXT_PRIMARY,
            height=26,
            width=90,
            corner_radius=RADIUS_BTN,
            command=self._on_add_custom_sound,
        )
        btn_add.pack(side="right", padx=4, pady=6)

        # Scrollable Sound Pads Grid
        self.sb_scroll = ctk.CTkScrollableFrame(tab, fg_color=BG_CARD_SUBTLE, corner_radius=RADIUS_CARD)
        self.sb_scroll.grid(row=1, column=0, padx=8, pady=4, sticky="nsew")
        self.sb_scroll.grid_columnconfigure((0, 1, 2, 3), weight=1)

    def _populate_default_soundboard(self):
        """Generate and load built-in preset sounds."""
        sounds_dir = os.path.join(BASE_DIR, "sounds")
        try:
            presets = ProceduralSoundGenerator.generate_all_presets(sounds_dir)
            hotkey_mapping = {
                "airhorn": "ctrl+1",
                "badumtss": "ctrl+2",
                "buzzer": "ctrl+3",
                "coin": "ctrl+4",
                "levelup": "ctrl+5",
                "tada": "ctrl+6",
                "siren": "ctrl+7",
                "laser": "ctrl+8",
            }

            for key, (name, path) in presets.items():
                hk = hotkey_mapping.get(key)
                if self.engine:
                    self.engine.soundboard.add_sound(key, name, path, volume=1.0, hotkey=hk)
                self._add_sound_card_ui(key, name, path, hk)
        except Exception as e:
            print(f"[WARN] Failed to populate default soundboard: {e}")

    def _add_sound_card_ui(self, clip_id: str, name: str, file_path: str, hotkey: Optional[str] = None):
        """Render a clickable soundboard pad card in the UI."""
        idx = len(self.sound_cards)
        row = idx // 4
        col = idx % 4

        card = ctk.CTkFrame(self.sb_scroll, fg_color=BG_CARD, corner_radius=RADIUS_BTN, border_width=1, border_color=BORDER_SUBTLE)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="ew")

        # Top bar of card (Hotkey Badge Sign)
        card_top = ctk.CTkFrame(card, fg_color="transparent")
        card_top.pack(fill="x", padx=8, pady=(4, 0))

        if hotkey:
            hk_display = hotkey.replace("+", " + ").upper()
            lbl_hk = ctk.CTkLabel(
                card_top,
                text=f"[{hk_display}]",
                font=("Segoe UI", 9, "bold"),
                text_color=COLOR_BLUE,
            )
            lbl_hk.pack(side="left")

        # Sound Name
        lbl_title = ctk.CTkLabel(
            card,
            text=name[:18],
            font=self.font_body_bold,
            text_color=TEXT_PRIMARY,
        )
        lbl_title.pack(padx=8, pady=2)

        # Play / Stop button
        btn_play = ctk.CTkButton(
            card,
            text="Play",
            font=self.font_btn,
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            text_color=TEXT_PRIMARY,
            height=26,
            corner_radius=RADIUS_BTN,
            command=lambda cid=clip_id: self._on_play_sound_clicked(cid),
        )
        btn_play.pack(fill="x", padx=8, pady=4)

        # Loop checkbox
        chk_loop = ctk.CTkCheckBox(
            card,
            text="Loop",
            font=self.font_caption,
            checkbox_width=14,
            checkbox_height=14,
        )
        chk_loop.pack(pady=(1, 4))

        # Register hotkey (safely marshaled to main Tkinter thread)
        if hotkey:
            self.hotkeys.register_hotkey(
                hotkey,
                lambda cid=clip_id: self.after(0, lambda: self._on_play_sound_clicked(cid))
            )

        self.sound_cards[clip_id] = {
            "name": name,
            "path": file_path,
            "btn_play": btn_play,
            "chk_loop": chk_loop,
            "hotkey": hotkey,
        }

    def _on_play_sound_clicked(self, clip_id: str):
        """Handle pad play/stop toggle."""
        if not self.engine:
            return
        card = self.sound_cards.get(clip_id)
        loop = card["chk_loop"].get() == 1 if card else False

        if self.engine.soundboard.is_playing(clip_id):
            self.engine.soundboard.stop_sound(clip_id)
        else:
            self.engine.soundboard.play_sound(clip_id, loop=loop)

    def _on_soundboard_stop_all(self):
        """Stop all playing sound clips."""
        if self.engine:
            self.engine.soundboard.stop_all()

    def _on_sb_hotkeys_toggled(self):
        """Enable or disable global hotkey triggers for soundboard."""
        enabled = self.switch_sb_hotkeys.get() == 1
        self.cfg["sb_hotkeys_enabled"] = enabled
        self._persist_config()
        self.hotkeys.set_enabled(enabled)

    def _on_soundboard_volume_changed(self, val: float):
        """Update master soundboard volume."""
        pct = int(val * 100)
        self.lbl_sb_vol.configure(text=f"{pct}%")
        self.cfg["sb_volume"] = float(val)
        self._persist_config()
        if self.engine:
            self.engine.soundboard.set_master_volume(val)

    def _on_add_custom_sound(self):
        """Browse disk to add custom audio file."""
        file_path = filedialog.askopenfilename(
            title="Select Audio Sound File",
            filetypes=[("Audio Files", "*.mp3;*.wav;*.ogg;*.flac;*.m4a;*.aac"), ("All Files", "*.*")],
        )
        if not file_path:
            return

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        clip_id = f"custom_{int(time.time() * 1000)}"

        try:
            if self.engine:
                self.engine.soundboard.add_sound(clip_id, base_name, file_path)
            self._add_sound_card_ui(clip_id, base_name, file_path)
        except Exception as e:
            print(f"[ERROR] Failed to load sound: {e}")

    # =========================================================================
    # VOICE CHANGER HANDLERS
    # =========================================================================
    def _on_vc_power_toggled(self):
        enabled = self.switch_vc.get() == 1
        self.cfg["vc_enabled"] = enabled
        self._persist_config()
        if self.engine:
            self.engine.set_voice_changer_enabled(enabled)
        if enabled:
            self.lbl_vc_status.configure(text="On", text_color=COLOR_BLUE)
        else:
            self.lbl_vc_status.configure(text="Off", text_color=TEXT_MUTED)

    def _on_select_vc_preset(self, preset_key: str):
        self.cfg["vc_preset"] = preset_key
        if self.engine:
            self.engine.set_voice_changer_preset(preset_key)
            pitch = self.engine.voice_changer.pitch_semitones
            formant = self.engine.voice_changer.formant_semitones
            chest_cut = self.engine.voice_changer.chest_cut
        else:
            cfg = VoiceChangerEngine.PRESETS.get(preset_key, {})
            pitch = cfg.get("pitch_semitones", 0.0)
            formant = cfg.get("formant_semitones", 0.0)
            chest_cut = cfg.get("chest_cut", False)

        self.slider_vc_pitch.set(pitch)
        self.lbl_vc_pitch.configure(text=f"{pitch:+.1f} st")
        self.cfg["vc_pitch"] = pitch

        self.slider_vc_formant.set(formant)
        self.lbl_vc_formant.configure(text=f"{formant:+.1f} st")
        self.cfg["vc_formant"] = formant

        if chest_cut:
            self.chk_chest_cut.select()
        else:
            self.chk_chest_cut.deselect()
        self.cfg["vc_chest_cut"] = chest_cut

        self._persist_config()

        # Update button visual styling
        for k, btn in self.vc_preset_buttons.items():
            btn.configure(fg_color=COLOR_BLUE if k == preset_key else BTN_SECONDARY)

    def _on_vc_pitch_changed(self, val: float):
        semitones = round(val * 2.0) / 2.0
        self.lbl_vc_pitch.configure(text=f"{semitones:+.1f} st")
        self.cfg["vc_pitch"] = semitones
        self._persist_config()
        if self.engine:
            self.engine.set_voice_changer_pitch(semitones)

    def _set_custom_pitch(self, semitones: float):
        self.slider_vc_pitch.set(semitones)
        self._on_vc_pitch_changed(semitones)

    def _on_vc_formant_changed(self, val: float):
        semitones = round(val * 2.0) / 2.0
        self.lbl_vc_formant.configure(text=f"{semitones:+.1f} st")
        self.cfg["vc_formant"] = semitones
        self._persist_config()
        if self.engine:
            self.engine.set_voice_changer_formant(semitones)

    def _set_custom_formant(self, semitones: float):
        self.slider_vc_formant.set(semitones)
        self._on_vc_formant_changed(semitones)

    def _on_vc_chest_cut_toggled(self):
        enabled = self.chk_chest_cut.get() == 1
        self.cfg["vc_chest_cut"] = enabled
        self._persist_config()
        if self.engine:
            self.engine.set_voice_changer_chest_cut(enabled)


    def _on_vc_mix_changed(self, val: float):
        pct = int(val * 100)
        self.lbl_vc_mix.configure(text=f"{pct}%")
        self.cfg["vc_mix"] = float(val)
        self._persist_config()
        if self.engine:
            self.engine.set_voice_changer_mix(val)

    # =========================================================================
    # AUDIO ROUTING & LOGIC CALLBACKS
    # =========================================================================
    def _find_matching_device(self, saved_name: str, device_keys: List[str]) -> Optional[str]:
        if not saved_name or not device_keys:
            return None
        if saved_name in device_keys:
            return saved_name
        clean_saved = saved_name.replace(" (Default)", "").strip().lower()
        for k in device_keys:
            if k.replace(" (Default)", "").strip().lower() == clean_saved:
                return k
        return None

    def _persist_config(self):
        """Save persistent user settings."""
        save_app_config(self.cfg)

    def _refresh_audio_devices(self):
        """Rescan system audio devices."""
        input_devs = AudioDeviceManager.get_input_devices()
        output_devs = AudioDeviceManager.get_output_devices()

        self.input_devices_map.clear()
        self.output_devices_map.clear()
        self.monitor_devices_map.clear()

        def_in = AudioDeviceManager.get_default_input_index()
        def_out = AudioDeviceManager.get_default_output_index()

        # Format input device labels
        in_names = []
        default_in_label = None
        for d in input_devs:
            is_def = " (Default)" if d["index"] == def_in else ""
            lbl = f"{d['name']}{is_def}"
            self.input_devices_map[lbl] = d["index"]
            in_names.append(lbl)
            if d["index"] == def_in:
                default_in_label = lbl

        # Format output device labels
        out_names = []
        default_out_label = None
        for d in output_devs:
            is_def = " (Default)" if d["index"] == def_out else ""
            lbl = f"{d['name']}{is_def}"
            self.output_devices_map[lbl] = d["index"]
            self.monitor_devices_map[lbl] = d["index"]
            out_names.append(lbl)
            if d["index"] == def_out:
                default_out_label = lbl

        if in_names:
            self.opt_input.configure(values=in_names)
            saved_in = self.cfg.get("input_device_name", "")
            matched_in = self._find_matching_device(saved_in, in_names)
            self.opt_input.set(matched_in if matched_in else (default_in_label if default_in_label else in_names[0]))

        if out_names:
            self.opt_output.configure(values=out_names)
            saved_out = self.cfg.get("output_device_name", "")
            matched_out = self._find_matching_device(saved_out, out_names)
            if matched_out:
                self.opt_output.set(matched_out)
            else:
                vc = AudioDeviceManager.find_virtual_cable_index()
                vc_label = next((k for k, v in self.output_devices_map.items() if v == vc), None)
                self.opt_output.set(vc_label if vc_label else (default_out_label if default_out_label else out_names[0]))

            self.opt_monitor.configure(values=out_names)
            saved_mon = self.cfg.get("monitor_device_name", "")
            matched_mon = self._find_matching_device(saved_mon, out_names)
            if matched_mon:
                self.opt_monitor.set(matched_mon)
            else:
                non_vc = next((k for k, v in self.output_devices_map.items() if "cable" not in k.lower() and v == def_out), out_names[0])
                self.opt_monitor.set(non_vc)

    def _init_engine(self):
        """Start the audio processing pipeline."""
        in_sel = self.opt_input.get()
        out_sel = self.opt_output.get()
        mon_sel = self.opt_monitor.get()

        in_idx = self.input_devices_map.get(in_sel)
        out_idx = self.output_devices_map.get(out_sel)
        mon_idx = self.monitor_devices_map.get(mon_sel)

        if in_idx is None or out_idx is None:
            return

        if self.engine is not None:
            self.engine.stop()

        self.engine = MicBoostEngine(
            input_device=in_idx,
            output_device=out_idx,
            monitor_device=mon_idx,
            sample_rate=48000,
            block_size=128,
            gain_db=self.slider_gain.get(),
            profile=self.selected_profile_key,
            limiter_enabled=self.switch_limiter.get() == 1,
            mute=bool(self.cfg.get("mute", False)),
        )

        # Set monitor state
        self.engine.set_monitor_enabled(self.switch_monitor.get() == 1, monitor_device=mon_idx)

        # Set voice changer state
        vc_preset = self.cfg.get("vc_preset", "bypass")
        self.engine.set_voice_changer_preset(vc_preset)
        vc_enabled = self.switch_vc.get() == 1
        self.engine.set_voice_changer_enabled(vc_enabled)
        vc_pitch = float(self.slider_vc_pitch.get())
        self.engine.set_voice_changer_pitch(vc_pitch)
        vc_mix = float(self.slider_vc_mix.get())
        self.engine.set_voice_changer_mix(vc_mix)

        # Set soundboard volume
        sb_vol = float(self.slider_sb_vol.get())
        self.engine.soundboard.set_master_volume(sb_vol)

        # Restore soundboard clips into the newly instantiated engine
        sounds_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "sounds"))
        ProceduralSoundGenerator.generate_all_presets(sounds_dir)
        for cid, card in self.sound_cards.items():
            try:
                self.engine.soundboard.add_sound(cid, card["name"], card["path"], hotkey=card.get("hotkey"))
            except Exception:
                pass

        try:
            self.engine.start()
            self.is_running = True
            self.lbl_engine_status.configure(text="On", text_color=COLOR_EMERALD)
            self.switch_engine.select()
        except Exception as e:
            self.is_running = False
            self.lbl_engine_status.configure(text="Error", text_color=COLOR_ROSE)
            self.switch_engine.deselect()
            print(f"Error starting audio engine: {e}")

    def _toggle_engine(self):
        """Start or stop the audio engine."""
        if self.is_running:
            if self.engine:
                self.engine.stop()
            self.is_running = False
            self.lbl_engine_status.configure(text="Off", text_color=TEXT_MUTED)
            self.switch_engine.deselect()
            self.cfg["engine_enabled"] = False
            self._persist_config()
        else:
            self._init_engine()
            self.cfg["engine_enabled"] = self.is_running
            self._persist_config()

    # Alias for any external or legacy references
    def _toggle_stream(self):
        self._toggle_engine()

    def _on_profile_changed(self, choice: str):
        key = self.profiles_by_name.get(choice, DEFAULT_PROFILE_KEY)
        self.selected_profile_key = key
        self.cfg["profile"] = key
        self._persist_config()
        prof = SOUND_PROFILES[key]
        self.lbl_profile_desc.configure(text=prof.description)
        if self.engine:
            self.engine.set_profile(key)

    def _on_input_device_changed(self, choice: str):
        self.cfg["input_device_name"] = choice
        self._persist_config()
        idx = self.input_devices_map.get(choice)
        if idx is not None and self.engine and self.is_running:
            self.engine.restart(input_device=idx)

    def _on_output_device_changed(self, choice: str):
        self.cfg["output_device_name"] = choice
        self._persist_config()
        idx = self.output_devices_map.get(choice)
        if idx is not None and self.engine and self.is_running:
            self.engine.restart(output_device=idx)

    def _on_monitor_device_changed(self, choice: str):
        self.cfg["monitor_device_name"] = choice
        self._persist_config()
        idx = self.monitor_devices_map.get(choice)
        if idx is not None and self.engine:
            self.engine.set_monitor_enabled(self.switch_monitor.get() == 1, monitor_device=idx)

    def _on_monitor_toggled(self):
        enabled = self.switch_monitor.get() == 1
        self.cfg["monitor_enabled"] = enabled
        self._persist_config()
        mon_idx = self.monitor_devices_map.get(self.opt_monitor.get())
        if self.engine:
            self.engine.set_monitor_enabled(enabled, monitor_device=mon_idx)

    def _on_max_gain_changed(self, choice: str):
        val = float(choice.replace(" dB", ""))
        self.max_gain_db = val
        self.cfg["max_gain_db"] = val
        self._persist_config()
        self.slider_gain.configure(to=val, number_of_steps=int(val * 10))
        if self.slider_gain.get() > val:
            self._set_gain_preset(val)

    def _on_gain_slider_changed(self, value: float):
        db_val = round(value, 1)
        self.lbl_gain_display.configure(text=f"+{db_val:.1f} dB")
        self.cfg["gain_db"] = db_val
        self._persist_config()
        if self.engine:
            self.engine.set_gain_db(db_val)

    def _set_gain_preset(self, db_value: float):
        self.slider_gain.set(db_value)
        self._on_gain_slider_changed(db_value)

    def _on_limiter_toggled(self):
        enabled = self.switch_limiter.get() == 1
        self.cfg["limiter_enabled"] = enabled
        self._persist_config()
        if self.engine:
            self.engine.set_limiter_enabled(enabled)
        if enabled:
            self.limiter_badge.configure(text="Active", text_color=COLOR_EMERALD, fg_color=BG_EMERALD_TINT)
        else:
            self.limiter_badge.configure(text="Off", text_color=TEXT_MUTED, fg_color=BG_CARD)

    def _toggle_mute(self):
        new_mute = not self.cfg.get("mute", False)
        self.cfg["mute"] = new_mute
        self._persist_config()
        if self.engine:
            self.engine.set_mute(new_mute)
        if new_mute:
            self.btn_mute.configure(text="Muted", fg_color=COLOR_ROSE, hover_color=COLOR_ROSE_HOVER, text_color="#ffffff")
        else:
            self.btn_mute.configure(text="Mute", fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER, text_color=TEXT_PRIMARY)

    def _poll_telemetry(self):
        """Telemetry update loop (~30 FPS) for level meters and indicators."""
        if self.engine and self.is_running:
            telem = self.engine.get_telemetry()

            pre_p = telem["pre_peak_db"]
            pre_r = telem["pre_rms_db"]
            post_p = telem["post_peak_db"]
            post_r = telem["post_rms_db"]

            self.vu_pre.update_level(pre_r, pre_p)
            self.vu_post.update_level(post_r, post_p)

            self.lbl_pre_db.configure(text=f"{pre_p:.1f} dBFS")
            self.lbl_post_db.configure(text=f"{post_p:.1f} dBFS")

            # Limiter badge
            if telem["is_limiting"]:
                self.limiter_badge.configure(text="Limiting", text_color=COLOR_ROSE, fg_color="#4c0519")
            elif telem["limiter_enabled"]:
                self.limiter_badge.configure(text="Active", text_color=COLOR_EMERALD, fg_color=BG_EMERALD_TINT)
            else:
                self.limiter_badge.configure(text="Off", text_color=TEXT_MUTED, fg_color=BG_CARD)

            # Update Soundboard Pad button colors if playing
            for cid, card in self.sound_cards.items():
                is_p = self.engine.soundboard.is_playing(cid)
                if is_p:
                    card["btn_play"].configure(text="Stop", fg_color=COLOR_EMERALD, text_color="#0a1a12")
                else:
                    card["btn_play"].configure(text="Play", fg_color=BTN_SECONDARY, text_color=TEXT_PRIMARY)

            # Performance stats
            drops = telem["overflows"] + telem["underflows"]
            self.lbl_perf.configure(text=f"Buffer drops: {drops}")
        else:
            self.vu_pre.update_level(-60.0)
            self.vu_post.update_level(-60.0)
            self.lbl_pre_db.configure(text="-60.0 dBFS")
            self.lbl_post_db.configure(text="-60.0 dBFS")

        self._telemetry_job = self.after(33, self._poll_telemetry)

    def _on_closing(self):
        """Clean up audio streams and threads on window close."""
        if hasattr(self, "_telemetry_job") and self._telemetry_job:
            try:
                self.after_cancel(self._telemetry_job)
            except Exception:
                pass
        # Ensure active audio switches are saved as Off for safe next launch
        self.cfg["engine_enabled"] = False
        self.cfg["monitor_enabled"] = False
        self.cfg["vc_enabled"] = False
        self.cfg["mute"] = False
        self._persist_config()
        self.hotkeys.stop()
        if self.engine:
            self.engine.stop()
        self.destroy()


def main():
    lock = SingleInstanceLock()
    if not lock.acquire():
        print("[Security] Another session of Woeyyy is already running. Switching focus to active window...")
        SingleInstanceLock.focus_existing_window("Woeyyy")
        sys.exit(0)

    try:
        app = WoeyyyApp()
        app.mainloop()
    finally:
        lock.release()


if __name__ == "__main__":
    main()
