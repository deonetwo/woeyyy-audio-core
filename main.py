"""
Woeyyy launcher.
"""

import argparse
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description="Woeyyy - Mic Boost, Voice Changer & Soundboard")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal CLI")
    parser.add_argument("--lite", action="store_true", help="Run minimalist Lite GUI")
    args = parser.parse_args()

    if args.cli:
        from run_mic_boost import run_interactive_cli
        run_interactive_cli()
    elif args.lite:
        from gui_lite import main as run_gui_lite
        run_gui_lite()
    else:
        try:
            import customtkinter  # noqa: F401
            from gui import main as run_gui
            run_gui()
        except ImportError as e:
            print(f"[INFO] GUI library not found ({e}). Falling back to terminal CLI...")
            from run_mic_boost import run_interactive_cli
            run_interactive_cli()
        except Exception:
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
