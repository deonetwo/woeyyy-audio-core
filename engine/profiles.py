"""
Equalizer profiles for microphone adjustments.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EQBand:
    """Parametric equalizer band definition."""
    filter_type: str  # 'highpass', 'peaking', 'highshelf', 'lowshelf'
    freq: float       # Frequency in Hz
    gain_db: float    # Gain in dB
    q: float = 0.707  # Quality factor


@dataclass
class SoundProfile:
    """Named collection of EQ bands."""
    key: str
    name: str
    description: str
    bands: List[EQBand] = field(default_factory=list)


SOUND_PROFILES: Dict[str, SoundProfile] = {
    "clear_voice": SoundProfile(
        key="clear_voice",
        name="Clear Voice (Default)",
        description="Highpass filter with midrange cut and treble boost.",
        bands=[
            EQBand(filter_type="highpass", freq=100.0, gain_db=0.0, q=0.707),
            EQBand(filter_type="peaking", freq=320.0, gain_db=-4.0, q=1.2),
            EQBand(filter_type="peaking", freq=3200.0, gain_db=4.5, q=1.0),
            EQBand(filter_type="highshelf", freq=8500.0, gain_db=2.5, q=0.707),
        ],
    ),
    "crisp_comms": SoundProfile(
        key="crisp_comms",
        name="Crisp Comms",
        description="Low-cut with vocal presence boost for gaming voice chat.",
        bands=[
            EQBand(filter_type="highpass", freq=150.0, gain_db=0.0, q=0.707),
            EQBand(filter_type="peaking", freq=400.0, gain_db=-5.0, q=1.4),
            EQBand(filter_type="peaking", freq=2800.0, gain_db=6.0, q=1.1),
            EQBand(filter_type="highshelf", freq=7500.0, gain_db=2.0, q=0.707),
        ],
    ),
    "broadcast_warm": SoundProfile(
        key="broadcast_warm",
        name="Broadcast Warmth",
        description="Low-end shelf with balanced presence.",
        bands=[
            EQBand(filter_type="highpass", freq=70.0, gain_db=0.0, q=0.707),
            EQBand(filter_type="lowshelf", freq=160.0, gain_db=2.0, q=0.707),
            EQBand(filter_type="peaking", freq=3500.0, gain_db=2.5, q=1.0),
            EQBand(filter_type="highshelf", freq=9000.0, gain_db=2.0, q=0.707),
        ],
    ),
    "flat": SoundProfile(
        key="flat",
        name="Flat (Bypass)",
        description="Direct microphone input without equalization.",
        bands=[],
    ),
}

DEFAULT_PROFILE_KEY = "clear_voice"
