"""
DSP utilities for gain, metering, equalization, and soft limiting.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def db_to_linear(gain_db: float) -> float:
    """Convert decibels (dB) to linear amplitude multiplier."""
    return float(10.0 ** (gain_db / 20.0))


def linear_to_db(gain_linear: float, floor_db: float = -96.0) -> float:
    """Convert linear amplitude multiplier to decibels (dB)."""
    if gain_linear <= 1e-5:
        return floor_db
    return float(20.0 * np.log10(gain_linear))


def calculate_levels(buffer: np.ndarray, floor_db: float = -96.0) -> Tuple[float, float]:
    """Calculate peak and RMS levels of an audio buffer in dBFS."""
    if buffer.size == 0:
        return floor_db, floor_db

    peak = float(np.max(np.abs(buffer)))
    peak_db = 20.0 * np.log10(peak) if peak > 1e-5 else floor_db

    rms = float(np.sqrt(np.mean(buffer * buffer)))
    rms_db = 20.0 * np.log10(rms) if rms > 1e-5 else floor_db

    return max(peak_db, floor_db), max(rms_db, floor_db)


class SoftLimiter:
    """Soft-knee saturation limiter to prevent digital clipping."""

    def __init__(
        self,
        threshold_db: float = -1.0,
        ceiling_db: float = -0.1,
        sample_rate: int = 48000,
        attack_ms: float = 1.0,
        release_ms: float = 60.0,
        mode: str = "soft_knee",
    ):
        """
        Initialize the limiter.
        
        Args:
            threshold_db: Level in dBFS above which limiting begins (default -1.0 dBFS)
            ceiling_db: Hard ceiling in dBFS that output will never exceed (default -0.1 dBFS)
            sample_rate: Audio sample rate in Hz
            attack_ms: Envelope follower attack time in ms
            release_ms: Envelope follower release time in ms
            mode: 'soft_knee' for zero-latency musical saturation,
                  'dynamic' for envelope gain reduction,
                  'hybrid' for envelope gain reduction + safety soft-knee.
        """
        self.sample_rate = sample_rate
        self.mode = mode
        self.enabled = True

        # Threshold and ceiling in linear amplitude
        self.threshold = db_to_linear(threshold_db)
        self.ceiling = db_to_linear(ceiling_db)
        self.margin = max(self.ceiling - self.threshold, 1e-4)

        # Dynamic envelope coefficients
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self._update_coeffs()

        # State variable for envelope follower
        self.envelope = 0.0

    def _update_coeffs(self):
        """Calculate single-pole filter coefficients for attack and release."""
        # attack_coef = exp(-1 / (fs * t_attack))
        self.attack_coef = float(np.exp(-1.0 / (self.sample_rate * (self.attack_ms / 1000.0))))
        self.release_coef = float(np.exp(-1.0 / (self.sample_rate * (self.release_ms / 1000.0))))

    def set_parameters(
        self,
        threshold_db: Union[float, None] = None,
        ceiling_db: Union[float, None] = None,
        mode: Union[str, None] = None,
    ):
        """Dynamically update limiter parameters."""
        if threshold_db is not None:
            self.threshold = db_to_linear(threshold_db)
        if ceiling_db is not None:
            self.ceiling = db_to_linear(ceiling_db)
        self.margin = max(self.ceiling - self.threshold, 1e-4)
        if mode is not None:
            self.mode = mode

    def reset(self):
        """Reset envelope state."""
        self.envelope = 0.0

    def process(self, buffer: np.ndarray) -> np.ndarray:
        """
        Process an incoming audio buffer through the limiter in-place or returning processed copy.
        
        Args:
            buffer: Float32 numpy array (channels x samples or samples x channels)
            
        Returns:
            Limited float32 numpy array strictly constrained to [-ceiling, +ceiling].
        """
        if not self.enabled or buffer.size == 0:
            return buffer

        # Mode: soft_knee (Zero latency, smooth analog saturation)
        if self.mode == "soft_knee":
            return self._process_soft_knee(buffer)

        # Mode: dynamic envelope limiter
        elif self.mode == "dynamic":
            return self._process_dynamic(buffer)

        # Mode: hybrid (envelope limiter with soft-knee safety)
        elif self.mode == "hybrid":
            dyn = self._process_dynamic(buffer)
            return self._process_soft_knee(dyn)

        return self._process_soft_knee(buffer)

    def _process_soft_knee(self, buffer: np.ndarray) -> np.ndarray:
        """
        C1 continuous soft-knee saturation:
        For |x| <= threshold: y = x
        For |x| > threshold:  y = sign(x) * [threshold + margin * tanh((|x| - threshold) / margin)]
        """
        abs_x = np.abs(buffer)
        over = abs_x > self.threshold

        if not np.any(over):
            return buffer

        # Allocate output buffer (or copy)
        out = buffer.copy()

        # Vectorized soft-knee computation only on samples exceeding threshold
        x_over = abs_x[over]
        delta = x_over - self.threshold
        sat = self.threshold + self.margin * np.tanh(delta / self.margin)

        # Preserve sign and enforce hard clamp to ceiling
        out[over] = np.sign(buffer[over]) * np.minimum(sat, self.ceiling)
        return out

    def _process_dynamic(self, buffer: np.ndarray) -> np.ndarray:
        """
        Fast block envelope follower with sample-rate independent attack/release.
        Smoothly reduces gain if peaks exceed ceiling.
        """
        peak = float(np.max(np.abs(buffer)))

        # Update envelope follower based on block peak
        if peak > self.envelope:
            self.envelope = self.attack_coef * self.envelope + (1.0 - self.attack_coef) * peak
        else:
            self.envelope = self.release_coef * self.envelope + (1.0 - self.release_coef) * peak

        # Compute gain reduction
        if self.envelope > self.ceiling:
            gain_reduction = self.ceiling / self.envelope
        else:
            gain_reduction = 1.0

        # Apply gain reduction and ensure safety clip
        out = buffer * gain_reduction
        return np.clip(out, -self.ceiling, self.ceiling)


class BiquadFilter:
    """
    Second-order IIR biquad filter using Robert Bristow-Johnson (RBJ) Audio EQ formulas.
    Maintains filter state (zi) across consecutive audio buffers for click-free real-time streaming.
    """

    def __init__(
        self,
        filter_type: str,
        freq: float,
        gain_db: float = 0.0,
        q: float = 0.707,
        sample_rate: int = 48000,
        channels: int = 1,
    ):
        self.filter_type = filter_type.lower()
        self.freq = float(freq)
        self.gain_db = float(gain_db)
        self.q = max(float(q), 0.01)
        self.sample_rate = sample_rate
        self.channels = channels

        self.b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.zi: Optional[np.ndarray] = None

        self._compute_coeffs()
        self.reset()

    def _compute_coeffs(self):
        """Calculate normalized [b0, b1, b2] and [1.0, a1, a2] biquad coefficients."""
        fs = float(self.sample_rate)
        f0 = max(10.0, min(self.freq, fs * 0.495))  # Clamp below Nyquist
        w0 = 2.0 * math.pi * (f0 / fs)
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * self.q)
        A = 10.0 ** (self.gain_db / 40.0)

        if self.filter_type == "highpass":
            b0 = (1.0 + cos_w0) / 2.0
            b1 = -(1.0 + cos_w0)
            b2 = (1.0 + cos_w0) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        elif self.filter_type == "peaking":
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_w0
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha / A

        elif self.filter_type == "highshelf":
            # 2 * sqrt(A) * alpha
            sqrt_A_2_alpha = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + sqrt_A_2_alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - sqrt_A_2_alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_w0 + sqrt_A_2_alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
            a2 = (A + 1.0) - (A - 1.0) * cos_w0 - sqrt_A_2_alpha

        elif self.filter_type == "lowpass":
            b0 = (1.0 - cos_w0) / 2.0
            b1 = 1.0 - cos_w0
            b2 = (1.0 - cos_w0) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        elif self.filter_type == "lowshelf":
            sqrt_A_2_alpha = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + sqrt_A_2_alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - sqrt_A_2_alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_w0 + sqrt_A_2_alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
            a2 = (A + 1.0) + (A - 1.0) * cos_w0 - sqrt_A_2_alpha

        else:
            # Bypass
            b0, b1, b2 = 1.0, 0.0, 0.0
            a0, a1, a2 = 1.0, 0.0, 0.0

        # Normalize by a0
        self.b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
        self.a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)

    def reset(self):
        """Reset filter delay state to zero."""
        if HAS_SCIPY:
            # scipy zi shape: (order, channels) or (order,)
            try:
                base_zi = signal.lfilter_zi(self.b, self.a)
                if self.channels > 1:
                    self.zi = np.zeros((len(base_zi), self.channels), dtype=np.float32)
                else:
                    self.zi = np.zeros((len(base_zi), 1), dtype=np.float32)
            except Exception:
                self.zi = np.zeros((2, self.channels), dtype=np.float32)
        else:
            # Pure numpy direct form state (2 delay values per channel)
            self.zi = np.zeros((2, self.channels), dtype=np.float32)

    def process(self, buffer: np.ndarray) -> np.ndarray:
        """
        Process a buffer of audio frames in-place or returning processed copy.
        Buffer shape: (frames, channels) or (frames,)
        """
        if buffer.size == 0:
            return buffer

        is_1d = buffer.ndim == 1
        if is_1d:
            buf = buffer[:, np.newaxis]
        else:
            buf = buffer

        ch = buf.shape[1]
        if self.zi is None or self.zi.shape[1] != ch:
            self.channels = ch
            self.reset()

        if HAS_SCIPY:
            out, self.zi = signal.lfilter(self.b, self.a, buf, axis=0, zi=self.zi)
        else:
            # Fallback Direct Form II transposed implementation
            out = np.zeros_like(buf, dtype=np.float32)
            b0, b1, b2 = self.b
            a1, a2 = self.a[1], self.a[2]
            for c in range(ch):
                d1, d2 = self.zi[0, c], self.zi[1, c]
                for n in range(buf.shape[0]):
                    x = buf[n, c]
                    y = b0 * x + d1
                    d1 = b1 * x - a1 * y + d2
                    d2 = b2 * x - a2 * y
                    out[n, c] = y
                self.zi[0, c] = d1
                self.zi[1, c] = d2

        if is_1d:
            return out[:, 0]
        return out


class ParametricEQChain:
    """
    Multi-band cascaded parametric equalizer chain.
    Sequentially processes audio through a list of Biquad filters.
    """

    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.filters: List[BiquadFilter] = []
        self.enabled = True

    def configure_bands(self, bands: List[object]):
        """
        Reconfigure the EQ chain from a list of EQBand definitions.
        Smoothly resets filter states.
        """
        new_filters = []
        for band in bands:
            # If band has 0dB gain and is peaking/shelf, it's a no-op
            if band.filter_type in ("peaking", "highshelf", "lowshelf") and abs(band.gain_db) < 0.01:
                continue

            filt = BiquadFilter(
                filter_type=band.filter_type,
                freq=band.freq,
                gain_db=band.gain_db,
                q=band.q,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )
            new_filters.append(filt)

        self.filters = new_filters

    def reset(self):
        """Reset internal filter states for all bands."""
        for f in self.filters:
            f.reset()

    def process(self, buffer: np.ndarray) -> np.ndarray:
        """Process buffer sequentially through all active EQ bands."""
        if not self.enabled or not self.filters or buffer.size == 0:
            return buffer

        out = buffer
        for f in self.filters:
            out = f.process(out)
        return out

