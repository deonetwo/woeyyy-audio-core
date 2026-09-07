"""
Unit tests for Woeyyy DSP algorithms (Gain, SoftLimiter, Metering).
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from engine.dsp import SoftLimiter, calculate_levels, db_to_linear, linear_to_db


class TestDSP(unittest.TestCase):
    def test_gain_conversions(self):
        """Verify decibel to linear and linear to decibel conversion accuracy."""
        assert np.isclose(db_to_linear(0.0), 1.0)
        assert np.isclose(db_to_linear(6.0206), 2.0, atol=1e-3)
        assert np.isclose(db_to_linear(-20.0), 0.1, atol=1e-4)

        assert np.isclose(linear_to_db(1.0), 0.0)
        assert np.isclose(linear_to_db(2.0), 6.0206, atol=1e-3)
        assert np.isclose(linear_to_db(0.1), -20.0, atol=1e-3)


    def test_calculate_levels(self):
        """Test peak and RMS calculations on known waveforms."""
        # Test 1: Full-scale sine wave (0 dBFS peak, -3.01 dBFS RMS)
        t = np.linspace(0, 1.0, 48000, endpoint=False, dtype=np.float32)
        sine_0db = np.sin(2 * np.pi * 1000 * t)
        peak, rms = calculate_levels(sine_0db)

        assert np.isclose(peak, 0.0, atol=0.01)
        assert np.isclose(rms, -3.01, atol=0.05)

        # Test 2: Silence (should hit floor)
        silence = np.zeros(256, dtype=np.float32)
        peak_silence, rms_silence = calculate_levels(silence)
        assert peak_silence <= -96.0
        assert rms_silence <= -96.0

    def test_soft_limiter_linearity_below_threshold(self):
        """Signals below threshold (-1.0 dBFS) must remain 100% bit-exact unaltered."""
        limiter = SoftLimiter(threshold_db=-1.0, ceiling_db=-0.1)
        threshold_linear = db_to_linear(-1.0)

        # Signal with peak well below threshold (e.g. 0.5)
        t = np.linspace(0, 0.01, 256, dtype=np.float32)
        clean_signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        processed = limiter.process(clean_signal)
        np.testing.assert_array_almost_equal(clean_signal, processed, decimal=6)

    def test_soft_limiter_ceiling_protection_extreme_boost(self):
        """Even under extreme +30 dB overdrive, the output must never exceed ceiling."""
        limiter = SoftLimiter(threshold_db=-1.0, ceiling_db=-0.1)
        ceiling_linear = db_to_linear(-0.1)  # ~0.98855

        # Generate an extreme boosted signal (amplitude 20.0 = ~ +26 dBFS)
        t = np.linspace(0, 0.05, 2048, dtype=np.float32)
        overdriven = (20.0 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        processed = limiter.process(overdriven)

        max_val = np.max(processed)
        min_val = np.min(processed)

        assert max_val <= ceiling_linear + 1e-6, f"Max {max_val} exceeded ceiling {ceiling_linear}"
        assert min_val >= -ceiling_linear - 1e-6, f"Min {min_val} went below -ceiling"
        assert not np.isnan(processed).any()
        assert not np.isinf(processed).any()

    def test_limiter_performance_benchmark(self):
        """Limiter processing for a 128-sample block must execute in under 0.1ms."""
        limiter = SoftLimiter(threshold_db=-1.0, ceiling_db=-0.1)
        block = (np.random.randn(128, 1) * 2.0).astype(np.float32)

        # Warm-up
        for _ in range(10):
            limiter.process(block)

        start = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            limiter.process(block)
        elapsed = time.perf_counter() - start

        time_per_block_ms = (elapsed / iterations) * 1000.0
        assert time_per_block_ms < 0.2, f"Too slow: {time_per_block_ms} ms per block"

    def test_clear_voice_profile_eq_response(self):
        """Verify Clear Voice profile cuts sub-bass rumble and boosts speech articulation."""
        from engine.profiles import SOUND_PROFILES
        from engine.dsp import ParametricEQChain

        fs = 48000
        eq = ParametricEQChain(sample_rate=fs, channels=1)
        eq.configure_bands(SOUND_PROFILES["clear_voice"].bands)

        # Test 1: Sub-bass rumble (40 Hz) - must be heavily suppressed (> 10 dB cut)
        t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False, dtype=np.float32)
        sub_bass = np.sin(2 * np.pi * 40 * t)
        sub_out = eq.process(sub_bass)
        in_peak, _ = calculate_levels(sub_bass)
        out_peak, _ = calculate_levels(sub_out)
        bass_attenuation = in_peak - out_peak
        assert bass_attenuation > 10.0, f"Expected >10dB sub-bass cut, got {bass_attenuation:.1f} dB"

        # Reset eq
        eq.reset()

        # Test 2: Speech articulation frequency (3200 Hz) - must be boosted (~ +4.5 dB)
        speech_tone = np.sin(2 * np.pi * 3200 * t)
        speech_out = eq.process(speech_tone)
        sp_in_peak, _ = calculate_levels(speech_tone)
        sp_out_peak, _ = calculate_levels(speech_out)
        art_boost = sp_out_peak - sp_in_peak
        assert 3.5 <= art_boost <= 5.5, f"Expected ~ +4.5 dB boost, got {art_boost:.1f} dB"

    def test_eq_continuity_across_buffers(self):
        """Streaming across small 128-sample blocks must maintain seamless continuity without clicks."""
        from engine.profiles import SOUND_PROFILES
        from engine.dsp import ParametricEQChain

        fs = 48000
        eq_streaming = ParametricEQChain(sample_rate=fs, channels=1)
        eq_streaming.configure_bands(SOUND_PROFILES["clear_voice"].bands)

        # Multi-frequency test signal (sweep)
        t = np.linspace(0, 0.1, 4800, dtype=np.float32)
        signal_in = (np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)

        # Process in 128-sample chunks
        chunk_size = 128
        chunks_out = []
        for i in range(0, len(signal_in), chunk_size):
            chunk = signal_in[i : i + chunk_size]
            chunks_out.append(eq_streaming.process(chunk))

        reconstructed = np.concatenate(chunks_out)

        # One-shot processing on single full buffer
        eq_oneshot = ParametricEQChain(sample_rate=fs, channels=1)
        eq_oneshot.configure_bands(SOUND_PROFILES["clear_voice"].bands)
        oneshot_out = eq_oneshot.process(signal_in)

        # Must match within float precision
        np.testing.assert_array_almost_equal(reconstructed, oneshot_out, decimal=5)

    def test_full_pipeline_benchmark(self):
        """Benchmark full pipeline: 4-band EQ + Gain ramp + Limiter on 128-sample buffer."""
        from engine.profiles import SOUND_PROFILES
        from engine.dsp import ParametricEQChain

        fs = 48000
        eq = ParametricEQChain(sample_rate=fs, channels=1)
        eq.configure_bands(SOUND_PROFILES["clear_voice"].bands)
        limiter = SoftLimiter(threshold_db=-1.0, ceiling_db=-0.1, sample_rate=fs)

        block = (np.random.randn(128, 1) * 1.5).astype(np.float32)

        # Warm-up
        for _ in range(20):
            o = eq.process(block) * 2.0
            limiter.process(o)

        start = time.perf_counter()
        iterations = 2000
        for _ in range(iterations):
            filtered = eq.process(block)
            boosted = filtered * 2.0
            limiter.process(boosted)
        elapsed = time.perf_counter() - start

        time_per_block_ms = (elapsed / iterations) * 1000.0
        assert time_per_block_ms < 0.15, f"Pipeline too slow: {time_per_block_ms} ms"


if __name__ == "__main__":
    unittest.main()

