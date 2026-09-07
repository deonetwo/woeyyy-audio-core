"""
Woeyyy Lite
Compact desktop microphone booster, voice effects, and soundboard.
"""

import json
import os
import sys
import threading
import time
import tkinter as tk
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk
from PIL import Image, ImageTk

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from engine.audio_engine import AudioDeviceManager, MicBoostEngine
from engine.profiles import DEFAULT_PROFILE_KEY
from engine.voice_changer import VoiceChangerEngine
from engine.security import SingleInstanceLock, secure_file_permissions

# CustomTkinter theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Design Palette (Refactoring UI: Casual + Reserved Utility)
BG_MAIN = "#0f172a"          # Slate 900 base
BG_CARD = "#1e293b"          # Slate 800 card surface
BG_CARD_SUBTLE = "#0f172a"   # Deep nested surface
BORDER_SUBTLE = "#334155"    # Slate 700 container border

COLOR_EMERALD = "#10b981"
BG_EMERALD_TINT = "#064e3b"

COLOR_BLUE = "#3b82f6"
COLOR_BLUE_HOVER = "#2563eb"

COLOR_ROSE = "#ef4444"
COLOR_ROSE_HOVER = "#dc2626"

COLOR_AMBER = "#f59e0b"

TEXT_PRIMARY = "#f8fafc"     # High-contrast readable
TEXT_SECONDARY = "#94a3b8"   # Secondary metadata & labels
TEXT_MUTED = "#64748b"       # De-emphasized captions

BTN_SECONDARY = "#334155"
BTN_SECONDARY_HOVER = "#475569"

RADIUS_CARD = 8
RADIUS_BTN = 6

CONFIG_FILE = os.path.join(BASE_DIR, ".lite_config.json")


def load_lite_config() -> dict:
    """Load persistent user settings for Woeyyy Lite. Creates default config file if missing."""
    defaults = {
        "input_device_name": "",
        "output_device_name": "",
        "mic_enabled": False,
        "gain_db": 12.0,
        "max_gain_db": 60.0,
        "limiter_enabled": True,
        "mute": False,
        "vc_enabled": False,
        "vc_preset": "deep_voice",
        "vc_pitch": -5.0,
        "sb_volume": 0.8,
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                defaults.update(saved)
        except Exception as e:
            print(f"[Lite] Notice: could not load config, recreating with defaults: {e}")
            save_lite_config(defaults)
    else:
        # Guarantee config file is immediately created so it's never missing or not found
        save_lite_config(defaults)

    # Always ensure live streaming, monitoring, and voice changer power start in the OFF state on app launch
    defaults["mic_enabled"] = False
    defaults["vc_enabled"] = False
    defaults["mute"] = False

    return defaults


def save_lite_config(cfg: dict):
    """Save persistent user settings for Woeyyy Lite and lock file permissions."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(CONFIG_FILE)), exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        secure_file_permissions(CONFIG_FILE)
    except Exception as e:
        print(f"[Lite] Notice: could not save config: {e}")


try:
    from tkinter_icons import LucideIcon
except ImportError:
    LucideIcon = None


def get_icon(name: str, color: str = TEXT_SECONDARY, size: int = 14) -> Optional[ctk.CTkImage]:
    """Safely fetch and render a vector icon as CTkImage with PIL fallback."""
    if LucideIcon is not None:
        try:
            ic = LucideIcon(name=name, size=size, color=color)
            if hasattr(ic, "to_pil"):
                pil_img = ic.to_pil()
                return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(size, size))
            elif isinstance(ic, Image.Image):
                return ctk.CTkImage(light_image=ic, dark_image=ic, size=(size, size))
            elif isinstance(ic, ctk.CTkImage):
                return ic
        except Exception:
            pass

    # High-contrast PNG asset fallback
    clean_name = name.replace("-", "_").lower()
    asset_file = os.path.join(BASE_DIR, "assets", f"icon_{clean_name}.png")
    if os.path.exists(asset_file):
        try:
            img = Image.open(asset_file).convert("RGBA")
            return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
        except Exception:
            pass

    return None


class CompactVUMeter(tk.Canvas):
    """Responsive canvas VU meter with dynamic resizing."""

    def __init__(self, parent, width: int = 190, height: int = 14, **kwargs):
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
        self.min_db = -60.0
        self.max_db = 0.0

        self.current_db = -60.0
        self.peak_db = -60.0
        self.peak_hold_frames = 0

        self.bind("<Configure>", self._on_configure)
        self._draw(-60.0, -60.0)

    def _on_configure(self, event):
        if event.width > 20:
            self.meter_width = event.width
            self.meter_height = event.height
            self._draw(self.current_db, self.peak_db)

    def update_level(self, rms_db: float, peak_db: Optional[float] = None):
        """Update meter with new dBFS values and redraw canvas."""
        self.current_db = max(self.min_db, min(self.max_db, rms_db))
        if peak_db is not None:
            p = max(self.min_db, min(self.max_db, peak_db))
            if p >= self.peak_db:
                self.peak_db = p
                self.peak_hold_frames = 15
            elif self.peak_hold_frames > 0:
                self.peak_hold_frames -= 1
            else:
                self.peak_db = max(self.current_db, self.peak_db - 1.5)

        self._draw(self.current_db, self.peak_db)

    def reset(self):
        self._draw(-60.0, -60.0)

    def _db_to_x(self, db: float) -> float:
        val = max(self.min_db, min(self.max_db, db))
        return (val - self.min_db) / (self.max_db - self.min_db) * self.meter_width

    def _draw(self, rms_db: float, peak_db: float):
        self.delete("all")
        # Track background
        self.create_rectangle(0, 0, self.meter_width, self.meter_height, fill=BG_CARD_SUBTLE, outline="")

        x_rms = self._db_to_x(rms_db)
        x_green = self._db_to_x(-18.0)
        x_amber = self._db_to_x(-6.0)

        # Draw green zone
        if x_rms > 0:
            self.create_rectangle(0, 1, min(x_rms, x_green), self.meter_height - 1, fill=COLOR_EMERALD, outline="")
        # Amber zone
        if x_rms > x_green:
            self.create_rectangle(x_green, 1, min(x_rms, x_amber), self.meter_height - 1, fill=COLOR_AMBER, outline="")
        # Red zone
        if x_rms > x_amber:
            self.create_rectangle(x_amber, 1, min(x_rms, self.meter_width), self.meter_height - 1, fill=COLOR_ROSE, outline="")

        # Peak indicator line
        if peak_db > -58.0:
            px = max(0, min(self.meter_width - 2, self._db_to_x(peak_db)))
            pcol = COLOR_ROSE if peak_db > -6.0 else (COLOR_AMBER if peak_db > -18.0 else "#38bdf8")
            self.create_line(px, 1, px, self.meter_height - 1, fill=pcol, width=2)


class WoeyyyLiteApp(ctk.CTk):
    """
    Woeyyy Lite Desktop Application.
    Clean interface: Microphone booster, voice effects, and soundboard.
    """

    def __init__(self, lock: Optional[SingleInstanceLock] = None):
        super().__init__()
        self.lock = lock

        # Load Saved State
        self.cfg = load_lite_config()

        self.title("Woeyyy Lite")
        self.geometry("520x780")
        self.minsize(500, 680)
        self.configure(fg_color=BG_MAIN)

        # Set Windows AppUserModelID for taskbar icon
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("woeyyy.audio.lite.2.0")
        except Exception:
            pass

        # Load Icon
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

        # Typography Scale
        self.font_brand = ctk.CTkFont(family="Segoe UI", size=16, weight="bold")
        self.font_section = ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        self.font_label = ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
        self.font_body = ctk.CTkFont(family="Segoe UI", size=11)
        self.font_body_bold = ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
        self.font_btn = ctk.CTkFont(family="Segoe UI", size=11)
        self.font_caption = ctk.CTkFont(family="Segoe UI", size=10)
        self.font_readout = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")

        # Audio Engine State
        self.engine: Optional[MicBoostEngine] = None
        self.is_mic_on = False
        self.raw_input_devices: List[Tuple[int, str]] = []
        self.raw_output_devices: List[Tuple[int, str]] = []
        self.input_devices_map: Dict[str, int] = {}
        self.output_devices_map: Dict[str, int] = {}

        # Build Interface
        self._build_header()
        self._build_mic_card()
        self._build_voice_changer_card()
        self._build_soundboard_card()

        # Initialize Audio Engine
        self._init_mic_engine()

        # Telemetry Polling Loop (~30 FPS)
        self._poll_mic_telemetry()

        # Clean Exit Hook
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _persist_config(self):
        """Save current runtime state to persistent config file."""
        save_lite_config(self.cfg)

    def _build_header(self):
        """Clean header with title and concise purpose."""
        header = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        header.pack(fill="x", padx=14, pady=(8, 4))

        box = ctk.CTkFrame(header, fg_color="transparent")
        box.pack(side="left", padx=12, pady=6)

        # Logo
        icon_png_path = os.path.join(BASE_DIR, "assets", "app_icon.png")
        if os.path.exists(icon_png_path):
            try:
                self._logo_img = ctk.CTkImage(
                    light_image=Image.open(icon_png_path),
                    dark_image=Image.open(icon_png_path),
                    size=(26, 26),
                )
                ctk.CTkLabel(box, image=self._logo_img, text="").pack(side="left", padx=(0, 8))
            except Exception:
                pass

        title_col = ctk.CTkFrame(box, fg_color="transparent")
        title_col.pack(side="left")

        ctk.CTkLabel(title_col, text="Woeyyy Lite", font=self.font_brand, text_color=TEXT_PRIMARY).pack(anchor="w")

        ctk.CTkLabel(
            title_col,
            text="Microphone, voice effects, and soundboard",
            font=self.font_caption,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(1, 0))

    def _build_mic_card(self):
        """Card 1: Microphone Booster & Controls."""
        card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        card.pack(fill="x", padx=14, pady=3)

        # Top Bar: Title & ON/OFF Power Toggle
        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(6, 2))

        ctk.CTkLabel(
            top, text="Microphone",
            font=self.font_label, text_color=TEXT_PRIMARY
        ).pack(side="left")

        pwr_box = ctk.CTkFrame(top, fg_color="transparent")
        pwr_box.pack(side="right")

        self.lbl_mic_status = ctk.CTkLabel(
            pwr_box, text="Off",
            font=self.font_caption, text_color=TEXT_MUTED, width=28
        )
        self.lbl_mic_status.pack(side="right", padx=(4, 0))

        self.switch_mic_pwr = ctk.CTkSwitch(
            pwr_box, text="", width=40, progress_color=COLOR_EMERALD, command=self._toggle_mic_power
        )
        self.switch_mic_pwr.deselect()
        self.switch_mic_pwr.pack(side="right")

        # Routing Selectors (2 Columns)
        route_row = ctk.CTkFrame(card, fg_color="transparent")
        route_row.pack(fill="x", padx=14, pady=2)
        route_row.grid_columnconfigure(0, weight=1)
        route_row.grid_columnconfigure(1, weight=1)

        # Mic Input
        in_col = ctk.CTkFrame(route_row, fg_color="transparent")
        in_col.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkLabel(in_col, text="Input", font=self.font_caption, text_color=TEXT_SECONDARY).pack(anchor="w", pady=(0, 1))

        self.opt_input = ctk.CTkOptionMenu(
            in_col, values=["Scanning..."], font=self.font_caption, dropdown_font=self.font_caption,
            fg_color=BTN_SECONDARY, button_color=BORDER_SUBTLE, height=26, corner_radius=RADIUS_BTN,
            command=self._on_input_changed
        )
        self.opt_input.pack(fill="x")

        # Output
        out_col = ctk.CTkFrame(route_row, fg_color="transparent")
        out_col.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ctk.CTkLabel(out_col, text="Output", font=self.font_caption, text_color=TEXT_SECONDARY).pack(anchor="w", pady=(0, 1))

        self.opt_output = ctk.CTkOptionMenu(
            out_col, values=["Scanning..."], font=self.font_caption, dropdown_font=self.font_caption,
            fg_color=BTN_SECONDARY, button_color=BORDER_SUBTLE, height=26, corner_radius=RADIUS_BTN,
            command=self._on_output_changed
        )
        self.opt_output.pack(fill="x")

        # Gain Boost Control & Presets
        gain_row = ctk.CTkFrame(card, fg_color="transparent")
        gain_row.pack(fill="x", padx=14, pady=(4, 1))

        ctk.CTkLabel(gain_row, text="Gain", font=self.font_caption, text_color=TEXT_SECONDARY).pack(side="left")

        # Configurable Max Mic Boost
        max_box = ctk.CTkFrame(gain_row, fg_color="transparent")
        max_box.pack(side="left", padx=(12, 0))
        ctk.CTkLabel(max_box, text="Max:", font=self.font_caption, text_color=TEXT_MUTED).pack(side="left", padx=(0, 4))

        self.max_gain_db = float(self.cfg.get("max_gain_db", 60.0))
        self.opt_max_gain = ctk.CTkOptionMenu(
            max_box,
            values=["20 dB", "30 dB", "36 dB", "48 dB", "60 dB"],
            width=72,
            height=20,
            font=self.font_caption,
            dropdown_font=self.font_caption,
            fg_color=BTN_SECONDARY,
            button_color=BORDER_SUBTLE,
            button_hover_color=BTN_SECONDARY_HOVER,
            command=self._on_max_gain_changed,
        )
        self.opt_max_gain.set(f"{int(self.max_gain_db)} dB")
        self.opt_max_gain.pack(side="left")

        self.lbl_gain_db = ctk.CTkLabel(
            gain_row, text=f"+{self.cfg.get('gain_db', 12.0):.1f} dB",
            font=self.font_readout, text_color=COLOR_EMERALD
        )
        self.lbl_gain_db.pack(side="right")

        # Slider + Quick Presets Row
        slider_box = ctk.CTkFrame(card, fg_color="transparent")
        slider_box.pack(fill="x", padx=14, pady=1)

        self.slider_gain = ctk.CTkSlider(
            slider_box, from_=0.0, to=self.max_gain_db, number_of_steps=int(self.max_gain_db * 2),
            progress_color=COLOR_EMERALD, button_color=COLOR_EMERALD,
            command=self._on_gain_changed, height=14
        )
        self.slider_gain.set(min(self.cfg.get("gain_db", 12.0), self.max_gain_db))
        self.slider_gain.pack(side="left", fill="x", expand=True, padx=(0, 6))

        # Quick Preset Buttons
        for db in [0.0, 12.0, 24.0, 36.0, 48.0]:
            btn = ctk.CTkButton(
                slider_box, text=f"{int(db)} dB", width=38, height=22,
                font=self.font_caption, fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER,
                command=lambda d=db: self._set_gain(d)
            )
            btn.pack(side="left", padx=1)

        btn_max = ctk.CTkButton(
            slider_box, text="Max", width=38, height=22,
            font=self.font_caption, fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER,
            command=lambda: self._set_gain(self.max_gain_db)
        )
        btn_max.pack(side="left", padx=1)

        # Mute & Limiter Bar
        fx_bar = ctk.CTkFrame(card, fg_color="transparent")
        fx_bar.pack(fill="x", padx=14, pady=(4, 4))

        self.btn_mute = ctk.CTkButton(
            fx_bar, text="Mute",
            width=70, height=24, font=self.font_btn,
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            command=self._toggle_mute
        )
        self.btn_mute.pack(side="left")

        # Soft Limiter Switch
        lim_box = ctk.CTkFrame(fx_bar, fg_color="transparent")
        lim_box.pack(side="right")

        ctk.CTkLabel(lim_box, text="Limiter", font=self.font_caption, text_color=TEXT_MUTED).pack(side="left", padx=(0, 6))

        self.switch_limiter = ctk.CTkSwitch(
            lim_box, text="", width=36, progress_color=COLOR_EMERALD, command=self._on_limiter_toggled
        )
        if self.cfg.get("limiter_enabled", True):
            self.switch_limiter.select()
        else:
            self.switch_limiter.deselect()
        self.switch_limiter.pack(side="left")

        # Live VU Meters
        vu_box = ctk.CTkFrame(card, fg_color=BG_CARD_SUBTLE, corner_radius=RADIUS_BTN)
        vu_box.pack(fill="x", padx=14, pady=(0, 6))

        # Input Meter Row
        r_in = ctk.CTkFrame(vu_box, fg_color="transparent")
        r_in.pack(fill="x", padx=8, pady=(2, 1))
        ctk.CTkLabel(r_in, text="In", font=self.font_caption, text_color=TEXT_MUTED, width=24).pack(side="left")
        self.vu_in = CompactVUMeter(r_in, width=340, height=12)
        self.vu_in.pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_in_db = ctk.CTkLabel(r_in, text="-60.0", font=self.font_caption, text_color=TEXT_SECONDARY, width=42)
        self.lbl_in_db.pack(side="right")

        # Output Meter Row
        r_out = ctk.CTkFrame(vu_box, fg_color="transparent")
        r_out.pack(fill="x", padx=8, pady=(1, 2))
        ctk.CTkLabel(r_out, text="Out", font=self.font_caption, text_color=COLOR_EMERALD, width=24).pack(side="left")
        self.vu_out = CompactVUMeter(r_out, width=340, height=12)
        self.vu_out.pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_out_db = ctk.CTkLabel(r_out, text="-60.0", font=self.font_caption, text_color=COLOR_EMERALD, width=42)
        self.lbl_out_db.pack(side="right")

    def _build_voice_changer_card(self):
        """Card 2: Voice Effects."""
        card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        card.pack(fill="x", padx=14, pady=3)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(6, 2))

        ctk.CTkLabel(
            top, text="Voice Effects",
            font=self.font_label, text_color=TEXT_PRIMARY
        ).pack(side="left")

        pwr_box = ctk.CTkFrame(top, fg_color="transparent")
        pwr_box.pack(side="right")

        self.lbl_vc_status = ctk.CTkLabel(
            pwr_box, text="Off",
            font=self.font_caption, text_color=TEXT_MUTED, width=28
        )
        self.lbl_vc_status.pack(side="right", padx=(4, 0))

        self.switch_vc_pwr = ctk.CTkSwitch(
            pwr_box, text="", width=40, progress_color=COLOR_BLUE, command=self._toggle_vc_power
        )
        self.switch_vc_pwr.deselect()
        self.switch_vc_pwr.pack(side="right")

        # Presets Grid (3 columns x 2 rows, matching Soundboard pads)
        presets_frame = ctk.CTkFrame(card, fg_color="transparent")
        presets_frame.pack(fill="x", padx=14, pady=2)
        presets_frame.grid_columnconfigure((0, 1, 2), weight=1)

        presets = [
            ("Normal", "bypass"),
            ("Deep Voice", "deep_voice"),
            ("Chipmunk", "chipmunk"),
            ("Robot", "robot"),
            ("Radio", "radio"),
            ("Monster", "monster"),
        ]

        self.vc_buttons: Dict[str, ctk.CTkButton] = {}
        active_preset = self.cfg.get("vc_preset", "deep_voice")

        for idx, (label, key) in enumerate(presets):
            r = idx // 3
            c = idx % 3
            btn = ctk.CTkButton(
                presets_frame, text=label, height=26, font=self.font_btn,
                fg_color=COLOR_BLUE if active_preset == key else BTN_SECONDARY,
                hover_color=COLOR_BLUE_HOVER,
                command=lambda k=key: self._on_select_vc_preset(k)
            )
            btn.grid(row=r, column=c, padx=2, pady=2, sticky="ew")
            self.vc_buttons[key] = btn

        # Pitch Semitone Fine Tuning Slider
        pitch_row = ctk.CTkFrame(card, fg_color="transparent")
        pitch_row.pack(fill="x", padx=14, pady=(4, 1))

        ctk.CTkLabel(pitch_row, text="Pitch Shift", font=self.font_caption, text_color=TEXT_SECONDARY).pack(side="left")

        current_pitch = float(self.cfg.get("vc_pitch", -5.0))
        self.lbl_pitch_val = ctk.CTkLabel(
            pitch_row, text=f"{current_pitch:+.1f} st", font=self.font_readout, text_color=COLOR_BLUE
        )
        self.lbl_pitch_val.pack(side="right")

        slider_p_box = ctk.CTkFrame(card, fg_color="transparent")
        slider_p_box.pack(fill="x", padx=14, pady=(0, 6))

        self.slider_pitch = ctk.CTkSlider(
            slider_p_box, from_=-12.0, to=12.0, number_of_steps=48,
            progress_color=COLOR_BLUE, button_color=COLOR_BLUE,
            command=self._on_pitch_slider_changed, height=14
        )
        self.slider_pitch.set(current_pitch)
        self.slider_pitch.pack(side="left", fill="x", expand=True, padx=(0, 8))

        btn_reset_pitch = ctk.CTkButton(
            slider_p_box, text="Reset", width=50, height=20, font=self.font_caption,
            fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER,
            command=lambda: self._set_custom_pitch(0.0)
        )
        btn_reset_pitch.pack(side="right")

    def _build_soundboard_card(self):
        """Card 3: Soundboard."""
        card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=RADIUS_CARD, border_width=1, border_color=BORDER_SUBTLE)
        card.pack(fill="x", padx=14, pady=(3, 8))

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(6, 2))

        ctk.CTkLabel(
            top, text="Soundboard",
            font=self.font_label, text_color=TEXT_PRIMARY
        ).pack(side="left")

        # Soundboard Volume Slider
        vol_box = ctk.CTkFrame(top, fg_color="transparent")
        vol_box.pack(side="right")

        ctk.CTkLabel(vol_box, text="Volume", font=self.font_caption, text_color=TEXT_MUTED).pack(side="left", padx=(0, 4))
        self.slider_sb_vol = ctk.CTkSlider(
            vol_box, from_=0.0, to=1.0, width=90, height=14,
            progress_color=COLOR_BLUE, button_color=COLOR_BLUE,
            command=self._on_sb_vol_changed
        )
        self.slider_sb_vol.set(self.cfg.get("sb_volume", 0.8))
        self.slider_sb_vol.pack(side="left")

        # Sound Pads Grid
        pads_frame = ctk.CTkFrame(card, fg_color="transparent")
        pads_frame.pack(fill="x", padx=14, pady=(2, 6))
        pads_frame.grid_columnconfigure((0, 1, 2), weight=1)

        sounds = [
            ("Airhorn", "airhorn"),
            ("Ba-Dum-Tss", "badumtss"),
            ("Tada", "tada"),
            ("Level Up", "levelup"),
            ("Siren", "siren"),
            ("Laser", "laser"),
        ]

        for idx, (label, sound_name) in enumerate(sounds):
            r = idx // 3
            c = idx % 3
            btn = ctk.CTkButton(
                pads_frame, text=label, height=26, font=self.font_btn,
                fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER,
                border_width=1, border_color=BORDER_SUBTLE,
                command=lambda s=sound_name: self._trigger_sound(s)
            )
            btn.grid(row=r, column=c, padx=2, pady=2, sticky="ew")

    # =========================================================================
    # AUDIO ENGINE MANAGEMENT
    # =========================================================================
    def _init_mic_engine(self):
        """Initialize audio pipeline with fallback device discovery."""
        input_devs = AudioDeviceManager.get_input_devices()
        output_devs = AudioDeviceManager.get_output_devices()

        self.input_devices_map = {d["name"]: d["index"] for d in input_devs}
        self.output_devices_map = {d["name"]: d["index"] for d in output_devs}

        in_names = list(self.input_devices_map.keys()) or ["No Input Devices"]
        out_names = list(self.output_devices_map.keys()) or ["No Output Devices"]

        self.opt_input.configure(values=in_names)
        self.opt_output.configure(values=out_names)

        # Match saved device or defaults
        target_in = self._resolve_device(self.cfg.get("input_device_name"), self.input_devices_map, AudioDeviceManager.get_default_input_index())
        target_out = self._resolve_device(self.cfg.get("output_device_name"), self.output_devices_map, AudioDeviceManager.find_virtual_cable_index() or AudioDeviceManager.get_default_output_index())

        if target_in is not None:
            name = next((k for k, v in self.input_devices_map.items() if v == target_in), in_names[0])
            self.opt_input.set(name)
        if target_out is not None:
            name = next((k for k, v in self.output_devices_map.items() if v == target_out), out_names[0])
            self.opt_output.set(name)

        # Instantiate engine
        try:
            self.engine = MicBoostEngine(
                input_device=target_in,
                output_device=target_out,
                sample_rate=48000,
                block_size=128,
                gain_db=float(self.cfg.get("gain_db", 12.0)),
                limiter_enabled=bool(self.cfg.get("limiter_enabled", True)),
                mute=bool(self.cfg.get("mute", False)),
            )

            # Configure Voice Changer
            vc_enabled = bool(self.cfg.get("vc_enabled", False))
            preset = self.cfg.get("vc_preset", "deep_voice")
            self.engine.set_voice_changer_preset(preset)
            self.engine.set_voice_changer_enabled(vc_enabled)
            pitch = float(self.cfg.get("vc_pitch", -5.0))
            self.engine.set_voice_changer_pitch(pitch)

            # Configure Soundboard volume
            sb_vol = float(self.cfg.get("sb_volume", 0.8))
            self.engine.soundboard.set_master_volume(sb_vol)

            if self.is_mic_on:
                self.engine.start()
        except Exception as e:
            print(f"[Lite] Audio Engine Init Error: {e}")

    def _resolve_device(self, saved_name: str, dev_map: Dict[str, int], fallback_idx: Optional[int]) -> Optional[int]:
        if not saved_name:
            return fallback_idx
        if saved_name in dev_map:
            return dev_map[saved_name]
        clean_saved = saved_name.replace(" (Default)", "").strip().lower()
        for name, idx in dev_map.items():
            if name.replace(" (Default)", "").strip().lower() == clean_saved:
                return idx
        return fallback_idx

    def _toggle_mic_power(self):
        self.is_mic_on = not self.is_mic_on
        self.cfg["mic_enabled"] = self.is_mic_on
        self._persist_config()

        if self.is_mic_on:
            self.lbl_mic_status.configure(text="On", text_color=COLOR_EMERALD)
            if self.engine and not self.engine.is_running:
                try:
                    self.engine.start()
                except Exception as e:
                    print(f"[Lite] Error starting engine: {e}")
        else:
            self.lbl_mic_status.configure(text="Off", text_color=TEXT_MUTED)
            if self.engine and self.engine.is_running:
                self.engine.stop()
            self.vu_in.reset()
            self.vu_out.reset()
            self.lbl_in_db.configure(text="-60.0")
            self.lbl_out_db.configure(text="-60.0")

    def _on_input_changed(self, choice: str):
        idx = self.input_devices_map.get(choice)
        self.cfg["input_device_name"] = choice
        self._persist_config()
        if self.engine and idx is not None:
            self.engine.restart(input_device=idx)

    def _on_output_changed(self, choice: str):
        idx = self.output_devices_map.get(choice)
        self.cfg["output_device_name"] = choice
        self._persist_config()
        if self.engine and idx is not None:
            self.engine.restart(output_device=idx)

    def _on_max_gain_changed(self, choice: str):
        val = float(choice.replace(" dB", ""))
        self.max_gain_db = val
        self.cfg["max_gain_db"] = val
        self._persist_config()
        self.slider_gain.configure(to=val, number_of_steps=int(val * 2))
        if self.slider_gain.get() > val:
            self._set_gain(val)

    def _on_gain_changed(self, val: float):
        db = round(val * 2.0) / 2.0
        self.lbl_gain_db.configure(text=f"+{db:.1f} dB")
        self.cfg["gain_db"] = db
        self._persist_config()
        if self.engine:
            self.engine.set_gain_db(db)

    def _set_gain(self, db_value: float):
        self.slider_gain.set(db_value)
        self._on_gain_changed(db_value)

    def _toggle_mute(self):
        new_mute = not self.cfg.get("mute", False)
        self.cfg["mute"] = new_mute
        self._persist_config()
        if self.engine:
            self.engine.set_mute(new_mute)
        self.btn_mute.configure(
            text="Muted" if new_mute else "Mute",
            fg_color=COLOR_ROSE if new_mute else BTN_SECONDARY,
            hover_color=COLOR_ROSE_HOVER if new_mute else BTN_SECONDARY_HOVER,
        )

    def _on_limiter_toggled(self):
        enabled = bool(self.switch_limiter.get())
        self.cfg["limiter_enabled"] = enabled
        self._persist_config()
        if self.engine:
            self.engine.set_limiter_enabled(enabled)

    # =========================================================================
    # VOICE CHANGER CONTROLS
    # =========================================================================
    def _toggle_vc_power(self):
        enabled = bool(self.switch_vc_pwr.get())
        self.cfg["vc_enabled"] = enabled
        self._persist_config()

        self.lbl_vc_status.configure(
            text="On" if enabled else "Off",
            text_color=COLOR_BLUE if enabled else TEXT_MUTED
        )
        if self.engine:
            self.engine.set_voice_changer_enabled(enabled)

    def _on_select_vc_preset(self, preset_key: str):
        self.cfg["vc_preset"] = preset_key
        if self.engine:
            self.engine.set_voice_changer_preset(preset_key)
            self.slider_pitch.set(self.engine.voice_changer.pitch_semitones)
            self.lbl_pitch_val.configure(text=f"{self.engine.voice_changer.pitch_semitones:+.1f} st")

        # Update button highlights
        for k, btn in self.vc_buttons.items():
            btn.configure(fg_color=COLOR_BLUE if k == preset_key else BTN_SECONDARY)

        self._persist_config()

    def _on_pitch_slider_changed(self, val: float):
        semitones = round(val * 2.0) / 2.0
        self.lbl_pitch_val.configure(text=f"{semitones:+.1f} st")
        self.cfg["vc_pitch"] = semitones
        self._persist_config()
        if self.engine:
            self.engine.set_voice_changer_pitch(semitones)

    def _set_custom_pitch(self, semitones: float):
        self.slider_pitch.set(semitones)
        self._on_pitch_slider_changed(semitones)

    # =========================================================================
    # SOUNDBOARD CONTROLS
    # =========================================================================
    def _on_sb_vol_changed(self, val: float):
        self.cfg["sb_volume"] = float(val)
        self._persist_config()
        if self.engine:
            self.engine.soundboard.set_master_volume(float(val))

    def _trigger_sound(self, sound_name: str):
        """Play sound effect into active stream."""
        if not self.engine:
            return

        sounds_dir = os.path.join(BASE_DIR, "sounds")
        path = os.path.join(sounds_dir, f"{sound_name}.wav")
        if os.path.exists(path):
            try:
                self.engine.soundboard.play(path, volume=1.0)
            except Exception as e:
                print(f"[Lite] Play sound error: {e}")
        else:
            print(f"[Lite] Sound file not found: {path}")

    # =========================================================================
    # TELEMETRY POLLING
    # =========================================================================
    def _poll_mic_telemetry(self):
        """Poll VU meters from engine at ~30 FPS."""
        if self.engine and self.engine.is_running and self.is_mic_on:
            in_peak = self.engine.pre_peak_db
            in_rms = self.engine.pre_rms_db
            out_peak = self.engine.post_peak_db
            out_rms = self.engine.post_rms_db

            self.vu_in.update_level(in_rms, in_peak)
            self.vu_out.update_level(out_rms, out_peak)

            self.lbl_in_db.configure(text=f"{in_rms:4.1f}")
            self.lbl_out_db.configure(text=f"{out_rms:4.1f}")

        self._telemetry_job = self.after(33, self._poll_mic_telemetry)

    def _on_closing(self):
        """Safe shutdown."""
        if hasattr(self, "_telemetry_job") and self._telemetry_job:
            try:
                self.after_cancel(self._telemetry_job)
            except Exception:
                pass

        # Ensure live mic and effects are saved as Off for safe next launch
        self.cfg["mic_enabled"] = False
        self.cfg["vc_enabled"] = False
        self.cfg["mute"] = False
        self._persist_config()

        try:
            if self.engine and self.engine.is_running:
                self.engine.stop()
        except Exception:
            pass

        if self.lock is not None:
            try:
                self.lock.release()
            except Exception:
                pass

        self.destroy()
        sys.exit(0)


def main():
    lock = SingleInstanceLock("Local\\Woeyyy_Audio_Lite_SingleInstance_Mutex")
    if not lock.acquire():
        print("[Lite] Another Woeyyy Lite instance is already running.")
        sys.exit(0)

    app = WoeyyyLiteApp(lock=lock)
    app.mainloop()


if __name__ == "__main__":
    main()
