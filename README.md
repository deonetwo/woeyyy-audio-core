# Woeyyy Audio

Desktop microphone booster, voice changer, and soundboard for Windows.

## Features

- **Mic Booster**: Ultra-low-latency digital gain control (0 dB to +60 dB), parametric EQ profiles, and soft limiter.
- **Voice Changer**: Low-latency pitch shifter (-12 to +12 semitones), formant shaping, ring modulation (Robot), bandpass filtering (Radio), and vocal presets (Woman Voice, Deep Voice, Chipmunk, Robot, Radio, Monster).
- **Soundboard**: Polyphonic soundboard with built-in procedural clips and custom audio support. Hotkeys (`Ctrl+1` through `Ctrl+8`) with dedicated master toggle switch.
- **Headphone Monitoring**: Zero-echo self-monitoring to hear your soundboard, voice effects, and microphone.
- **Music & Auto-Ducking**: Intelligent auto-ducking that automatically lowers background audio when speaking.

## Requirements

- Windows 10 / 11
- Python 3.10+
- [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (for routing to Discord, games, or VoIP)

## Quick Start (Recommended)

Simply double-click the launcher batch script:
- **Full GUI**: `.\run.bat`
- **Lite GUI**: `.\run_lite.bat`

*The launcher automatically creates a virtual environment (`.venv`) and installs required packages from `requirements.txt` if not already present.*

## Manual Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Running Manually

- **Full GUI**: `python main.py` or `python gui.py`
- **Lite GUI**: `python main.py --lite` or `python gui_lite.py`
- **Terminal CLI**: `python main.py --cli`

## Routing Setup

1. In Woeyyy, select your physical microphone as **Input** and **CABLE Input** as **Output**.
2. In your communication app (Discord, Zoom, game), select **CABLE Output** as your microphone.
3. In Discord settings (Voice & Video), disable noise suppression and echo cancellation so they do not interfere with effects.

## Project Layout

- `engine/`: Audio processing (gain, limiter, EQ, pitch shifter, soundboard, music)
- `gui.py`: Full desktop interface (CustomTkinter)
- `gui_lite.py`: Minimalist compact widget interface
- `run_mic_boost.py`: Interactive terminal CLI interface
- `sounds/`: Built-in soundboard audio presets
- `tests/`: Unit test suite

## Running Tests

```powershell
python -m unittest discover tests
```
