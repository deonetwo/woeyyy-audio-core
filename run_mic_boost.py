"""
Interactive CLI for mic boost, EQ, and voice changer.
"""

import os
import sys
import time
from typing import Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.audio_engine import AudioDeviceManager, MicBoostEngine
from engine.profiles import SOUND_PROFILES


def render_vu_bar(db_value: float, width: int = 24) -> str:
    """
    Render a horizontal ASCII VU meter with color gradients:
    Green: [-96 dB to -18 dB]
    Yellow: [-18 dB to -3 dB]
    Red: [-3 dB to 0 dB / clipping]
    """
    # Clip dB between -60 dBFS and 0 dBFS for meter display
    min_db = -60.0
    max_db = 0.0
    clamped_db = max(min_db, min(db_value, max_db))
    ratio = (clamped_db - min_db) / (max_db - min_db)
    filled = int(round(ratio * width))

    # ANSI color codes
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31;1m"
    DIM = "\033[90m"
    RESET = "\033[0m"

    # Colorize segments
    bar_chars = []
    for i in range(width):
        frac = i / width
        if i < filled:
            if frac < 0.70:  # < -18 dB
                bar_chars.append(f"{GREEN}█{RESET}")
            elif frac < 0.90:  # -18 dB to -6 dB
                bar_chars.append(f"{YELLOW}█{RESET}")
            else:  # Peak / Limiter zone
                bar_chars.append(f"{RED}█{RESET}")
        else:
            bar_chars.append(f"{DIM}░{RESET}")

    bar_str = "".join(bar_chars)
    return f"[{bar_str}] {db_value:6.1f} dBFS"


def select_device(prompt: str, devices, default_idx: Optional[int] = None) -> int:
    """Prompt user to select an audio device or press Enter for default."""
    print(f"\n--- {prompt} ---")
    valid_indices = []
    for d in devices:
        idx = d["index"]
        name = d["name"]
        is_def = " (DEFAULT)" if idx == default_idx else ""
        print(f"  [{idx:2d}] {name}{is_def}")
        valid_indices.append(idx)

    choice = input(f"\nSelect device ID [Enter for default {default_idx}]: ").strip()
    if not choice:
        return default_idx if default_idx is not None else valid_indices[0]
    try:
        chosen = int(choice)
        if chosen in valid_indices:
            return chosen
    except ValueError:
        pass
    print(f"Invalid selection, defaulting to {default_idx}")
    return default_idx if default_idx is not None else valid_indices[0]


def run_interactive_cli():
    """Run interactive mic boost engine with live terminal dashboard."""
    # Enable ANSI escape sequences on Windows
    os.system("")

    print("=" * 60)
    print("   WOEYYY AUDIO - MIC BOOST & VOICE CHANGER")
    print("=" * 60)

    input_devs = AudioDeviceManager.get_input_devices()
    output_devs = AudioDeviceManager.get_output_devices()

    if not input_devs:
        print("[ERROR] No audio input devices (microphones) detected!")
        return
    if not output_devs:
        print("[ERROR] No audio output devices detected!")
        return

    default_in = AudioDeviceManager.get_default_input_index()
    virtual_cable = AudioDeviceManager.find_virtual_cable_index()
    default_out = virtual_cable if virtual_cable is not None else AudioDeviceManager.get_default_output_index()

    print(f"\nAuto-detected Default Mic: Device #{default_in}")
    if virtual_cable is not None:
        print(f"Auto-detected Virtual Cable: Device #{virtual_cable}")
    else:
        print(f"Auto-detected Monitor Out: Device #{default_out} (Headphones/Speakers)")

    # Device selection options
    quick_start = input("\nUse auto-detected devices? [Y/n]: ").strip().lower()
    if quick_start in ("n", "no"):
        in_idx = select_device("Select Input Microphone", input_devs, default_in)
        out_idx = select_device("Select Output Device (Virtual Cable or Headphones)", output_devs, default_out)
    else:
        in_idx = default_in
        out_idx = default_out

    # Initialize Engine
    print("\nStarting Woeyyy Core Audio Stream...")
    print("Sample Rate: 48,000 Hz | Buffer Size: 128 frames (~2.7ms latency)")

    engine = MicBoostEngine(
        input_device=in_idx,
        output_device=out_idx,
        sample_rate=48000,
        block_size=128,
        in_channels=1,
        out_channels=2,
        gain_db=6.0,  # Default starting boost of +6dB
        limiter_enabled=True,
    )

    try:
        engine.start()
    except Exception as e:
        print(f"\n[ERROR] Failed to start audio stream: {e}")
        print("Tip: If sample rate 48000Hz is unsupported by the selected device, trying 44100Hz...")
        try:
            engine.sample_rate = 44100
            engine.start()
        except Exception as e2:
            print(f"[FATAL] Could not initialize audio stream: {e2}")
            return

    print("\n" + "=" * 65)
    print(" [CONTROLS]")
    print("   [+] / [-] : Adjust Boost Gain (+1 dB / -1 dB)")
    print("   []] / [[] : Big Gain Step   (+5 dB / -5 dB)")
    print("   [P]       : Cycle Sound Profile (Clear Voice / Comms / Warm / Flat)")
    print("   [V]       : Cycle Voice Changer (Bypass / Woman / Deep / Chipmunk / Robot / Radio / Monster)")
    print("   [L]       : Toggle Soft-Limiter ON / OFF")
    print("   [M]       : Toggle Mute ON / OFF")
    print("   [Q]       : Quit")
    print("=" * 65 + "\n")

    # Windows non-blocking keyboard input
    has_msvcrt = False
    try:
        import msvcrt
        has_msvcrt = True
    except ImportError:
        pass

    CYAN = "\033[36;1m"
    BOLD = "\033[1m"
    RED = "\033[31;1m"
    GREEN = "\033[32;1m"
    YELLOW = "\033[33;1m"
    MAGENTA = "\033[35;1m"
    BLUE = "\033[34;1m"
    RESET = "\033[0m"

    profile_keys = list(SOUND_PROFILES.keys())
    vc_presets = ["bypass", "woman", "deep_voice", "chipmunk", "robot", "radio", "monster"]

    try:
        while True:
            # Poll keyboard input
            if has_msvcrt and msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"+", b"="):
                    engine.set_gain_db(min(engine.gain_db + 1.0, 40.0))
                elif ch in (b"-", b"_"):
                    engine.set_gain_db(max(engine.gain_db - 1.0, -20.0))
                elif ch == b"]":
                    engine.set_gain_db(min(engine.gain_db + 5.0, 40.0))
                elif ch == b"[":
                    engine.set_gain_db(max(engine.gain_db - 5.0, -20.0))
                elif ch in (b"p", b"P"):
                    curr_idx = profile_keys.index(engine.current_profile) if engine.current_profile in profile_keys else 0
                    next_key = profile_keys[(curr_idx + 1) % len(profile_keys)]
                    engine.set_profile(next_key)
                elif ch in (b"v", b"V"):
                    curr_vc = engine.voice_changer.current_preset
                    curr_vc_idx = vc_presets.index(curr_vc) if curr_vc in vc_presets else 0
                    next_vc = vc_presets[(curr_vc_idx + 1) % len(vc_presets)]
                    engine.set_voice_changer_preset(next_vc)
                elif ch in (b"l", b"L"):
                    engine.set_limiter_enabled(not engine.limiter_enabled)
                elif ch in (b"m", b"M"):
                    engine.set_mute(not engine.mute)
                elif ch in (b"q", b"Q", b"\x03"):  # \x03 is Ctrl+C
                    break

            telem = engine.get_telemetry()

            # Format status badges
            gain_str = f"{telem['gain_db']:+5.1f} dB"
            if telem["is_muted"]:
                status_badge = f"{RED}[MUTED]{RESET}"
            else:
                status_badge = f"{GREEN}[ACTIVE]{RESET}"

            prof_key = telem.get("profile", "clear_voice")
            prof_short = SOUND_PROFILES.get(prof_key, SOUND_PROFILES["clear_voice"]).name.split(" (")[0]
            profile_badge = f"{MAGENTA}[{prof_short}]{RESET}"

            vc_name = engine.voice_changer.current_preset.upper()
            if engine.voice_changer.enabled:
                vc_badge = f"{BLUE}[VC: {vc_name}]{RESET}"
            else:
                vc_badge = f"{YELLOW}[VC: OFF]{RESET}"

            if telem["limiter_enabled"]:
                if telem["is_limiting"]:
                    limiter_badge = f"{RED}[LIMITING!]{RESET}"
                else:
                    limiter_badge = f"{CYAN}[LIMITER ON]{RESET}"
            else:
                limiter_badge = f"{YELLOW}[LIMITER OFF]{RESET}"

            pre_bar = render_vu_bar(telem["pre_peak_db"], width=18)
            post_bar = render_vu_bar(telem["post_peak_db"], width=18)

            # Live single-line dashboard update
            dashboard = (
                f"\r{status_badge} "
                f"{profile_badge} "
                f"{vc_badge} "
                f"Mic: {pre_bar}  "
                f"Boost: {BOLD}{gain_str}{RESET}  "
                f"{limiter_badge}  "
                f"Out: {post_bar}  "
            )
            sys.stdout.write(dashboard)
            sys.stdout.flush()

            time.sleep(0.04)  # ~25 FPS UI refresh

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nStopping audio stream...")
        engine.stop()
        print("Woeyyy Engine stopped cleanly. Goodbye!\n")


if __name__ == "__main__":
    run_interactive_cli()
