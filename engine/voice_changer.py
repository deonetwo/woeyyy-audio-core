"""
Voice changer module providing pitch shifting, ring modulation,
and bandpass filtering.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class LowLatencyPitchShifter:
    """Time-domain dual delay-line pitch shifter with crossfade windowing."""

    def __init__(self, sample_rate: int = 48000, window_size: int = 1536):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.half_window = window_size // 2
        self.buf_size = window_size * 4
        self.buffer = np.zeros((self.buf_size, 2), dtype=np.float32)
        self.write_pos = 0
        self.pos = 0.0

    def reset(self):
        self.buffer.fill(0.0)
        self.write_pos = 0
        self.pos = 0.0

    def process(self, chunk: np.ndarray, semitones: float) -> np.ndarray:
        if abs(semitones) < 0.05:
            return chunk

        is_1d = chunk.ndim == 1
        c_data = chunk[:, np.newaxis] if is_1d else chunk

        n_samples, n_channels = c_data.shape
        out = np.zeros_like(c_data, dtype=np.float32)

        rate = float(2.0 ** (semitones / 12.0))
        step = 1.0 - rate

        buf = self.buffer
        buf_size = self.buf_size
        win = self.window_size
        half_win = self.half_window
        write_pos = self.write_pos
        pos = self.pos

        for i in range(n_samples):
            for ch in range(min(n_channels, 2)):
                buf[write_pos, ch] = c_data[i, ch]

            offset1 = pos % win
            offset2 = (pos + half_win) % win

            w1 = 1.0 - abs(2.0 * offset1 / win - 1.0)
            w2 = 1.0 - w1

            read1 = (write_pos - offset1) % buf_size
            read2 = (write_pos - offset2) % buf_size

            idx1_0 = int(read1)
            idx1_1 = (idx1_0 + 1) % buf_size
            frac1 = read1 - idx1_0

            idx2_0 = int(read2)
            idx2_1 = (idx2_0 + 1) % buf_size
            frac2 = read2 - idx2_0

            for ch in range(min(n_channels, 2)):
                s1 = (1.0 - frac1) * buf[idx1_0, ch] + frac1 * buf[idx1_1, ch]
                s2 = (1.0 - frac2) * buf[idx2_0, ch] + frac2 * buf[idx2_1, ch]
                out[i, ch] = w1 * s1 + w2 * s2

            write_pos = (write_pos + 1) % buf_size
            pos += step

        self.write_pos = write_pos
        self.pos = pos % win

        return out[:, 0] if is_1d else out


class VoiceChangerEngine:
    """Real-time vocal effects processor."""

    PRESETS = {
        "bypass": {
            "name": "Normal",
            "description": "No effect.",
            "mode": "bypass",
            "pitch_semitones": 0.0,
            "mix": 1.0,
        },
        "deep_voice": {
            "name": "Deep Voice",
            "description": "Pitch shifted down (-5 st).",
            "mode": "deep_voice",
            "pitch_semitones": -5.0,
            "mix": 1.0,
        },
        "chipmunk": {
            "name": "Chipmunk",
            "description": "Pitch shifted up (+8 st).",
            "mode": "chipmunk",
            "pitch_semitones": 8.0,
            "mix": 1.0,
        },
        "robot": {
            "name": "Robot",
            "description": "Ring modulation at 65 Hz.",
            "mode": "robot",
            "pitch_semitones": 0.0,
            "carrier_freq": 65.0,
            "mix": 0.95,
        },
        "radio": {
            "name": "Radio",
            "description": "Bandpass filter (400-3400 Hz) with soft clipping.",
            "mode": "radio",
            "pitch_semitones": 0.0,
            "mix": 1.0,
        },
        "monster": {
            "name": "Monster",
            "description": "One octave down (-12 st) with saturation.",
            "mode": "monster",
            "pitch_semitones": -12.0,
            "mix": 1.0,
        },
        "custom": {
            "name": "Custom",
            "description": "User pitch.",
            "mode": "pitch",
            "pitch_semitones": 0.0,
            "mix": 1.0,
        },
    }

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.enabled = False
        self.current_preset = "bypass"
        self.mode = "bypass"
        self.pitch_semitones = 0.0
        self.mix = 1.0
        self.carrier_freq = 65.0

        self.pitch_shifter = LowLatencyPitchShifter(sample_rate=sample_rate)
        self._carrier_phase = 0.0
        self._init_filters()

    def _init_filters(self):
        if HAS_SCIPY:
            nyq = self.sample_rate * 0.5
            low = max(100.0, min(400.0, nyq - 100)) / nyq
            high = max(low + 0.05, min(3400.0, nyq - 100)) / nyq
            self.b_radio, self.a_radio = signal.butter(2, [low, high], btype="bandpass")
            self.zi_radio = signal.lfilter_zi(self.b_radio, self.a_radio)
        else:
            self.b_radio, self.a_radio, self.zi_radio = None, None, None

    def reset(self):
        self.pitch_shifter.reset()
        self._carrier_phase = 0.0
        if HAS_SCIPY and self.zi_radio is not None:
            self.zi_radio = signal.lfilter_zi(self.b_radio, self.a_radio)

    def set_preset(self, preset_key: str):
        if preset_key not in self.PRESETS:
            preset_key = "bypass"
        self.current_preset = preset_key
        cfg = self.PRESETS[preset_key]
        self.mode = cfg.get("mode", "bypass")
        self.pitch_semitones = cfg.get("pitch_semitones", 0.0)
        self.mix = cfg.get("mix", 1.0)
        if "carrier_freq" in cfg:
            self.carrier_freq = cfg["carrier_freq"]

        self.enabled = (self.mode != "bypass")

    def set_pitch_semitones(self, semitones: float):
        self.pitch_semitones = max(-12.0, min(12.0, float(semitones)))
        if self.current_preset != "custom" and self.mode in ("pitch", "bypass"):
            self.current_preset = "custom"
            self.mode = "pitch"

    def set_mix(self, mix: float):
        self.mix = max(0.0, min(1.0, float(mix)))

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled:
            self.mode = "bypass"
        elif self.current_preset == "bypass":
            self.set_preset("deep_voice")

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if not self.enabled or self.mode == "bypass" or len(chunk) == 0:
            return chunk

        dry = chunk.copy()
        wet = chunk

        if self.mode in ("pitch", "deep_voice", "chipmunk", "monster"):
            wet = self.pitch_shifter.process(wet, self.pitch_semitones)
            if self.mode == "monster":
                wet = np.tanh(wet * 1.5) * 0.9

        elif self.mode == "robot":
            n = len(chunk)
            t = np.arange(n) / float(self.sample_rate)
            phase_vec = self._carrier_phase + 2.0 * np.pi * self.carrier_freq * t
            carrier = np.sin(phase_vec).astype(np.float32)
            self._carrier_phase = float((phase_vec[-1] + 2.0 * np.pi * self.carrier_freq / self.sample_rate) % (2.0 * np.pi))

            if chunk.ndim == 2:
                carrier = carrier[:, np.newaxis]

            wet = wet * (0.2 + 0.8 * carrier)

        elif self.mode == "radio":
            if HAS_SCIPY and self.b_radio is not None:
                if chunk.ndim == 1:
                    wet, _ = signal.lfilter(self.b_radio, self.a_radio, wet, zi=self.zi_radio * wet[0])
                else:
                    wet_filtered = np.zeros_like(wet)
                    for ch in range(wet.shape[1]):
                        filtered, _ = signal.lfilter(self.b_radio, self.a_radio, wet[:, ch], zi=self.zi_radio * wet[0, ch])
                        wet_filtered[:, ch] = filtered
                    wet = wet_filtered

            wet = np.tanh(wet * 2.2) * 0.85

        if self.mix < 0.99:
            out = (1.0 - self.mix) * dry + self.mix * wet
        else:
            out = wet

        return out.astype(np.float32)
