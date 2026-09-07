"""
Woeyyy - Real-Time Microphone Enhancer, Soundboard & Voice Changer
Core Engine Package
"""

# Audio DSP & device manager (requires numpy, scipy, sounddevice)
try:
    from .audio_engine import AudioDeviceManager, MicBoostEngine
except ImportError:
    AudioDeviceManager, MicBoostEngine = None, None

try:
    from .voice_changer import LowLatencyPitchShifter, VoiceChangerEngine
except ImportError:
    LowLatencyPitchShifter, VoiceChangerEngine = None, None

try:
    from .soundboard import ProceduralSoundGenerator, SoundboardEngine, load_audio_file
except ImportError:
    ProceduralSoundGenerator, SoundboardEngine, load_audio_file = None, None, None

try:
    from .dsp import (
        BiquadFilter,
        ParametricEQChain,
        SoftLimiter,
        calculate_levels,
        db_to_linear,
        linear_to_db,
    )
except ImportError:
    pass

try:
    from .profiles import DEFAULT_PROFILE_KEY, SOUND_PROFILES, SoundProfile
except ImportError:
    pass

__all__ = [
    "MicBoostEngine",
    "AudioDeviceManager",
    "VoiceChangerEngine",
    "LowLatencyPitchShifter",
    "SoundboardEngine",
    "ProceduralSoundGenerator",
    "load_audio_file",
    "SoftLimiter",
    "BiquadFilter",
    "ParametricEQChain",
    "SoundProfile",
    "SOUND_PROFILES",
    "DEFAULT_PROFILE_KEY",
    "calculate_levels",
    "db_to_linear",
    "linear_to_db",
]
