"""
Unit tests for Woeyyy Voice Changer subsystem (engine/voice_changer.py).
Tests low-latency pitch shifting, vocal presets (Deep Voice, Chipmunk, Robot, Radio, Monster),
dry/wet crossfade blending, and execution performance.
"""

import time
import unittest
import numpy as np

from engine.voice_changer import FormantVocalTractModeler, LowLatencyPitchShifter, VoiceChangerEngine


class TestVoiceChangerEngine(unittest.TestCase):
    def setUp(self):
        self.sr = 48000
        self.vc = VoiceChangerEngine(sample_rate=self.sr)

    def test_pitch_shifter_shape_and_dtype(self):
        """Verify pitch shifter preserves array shapes and float32 dtype for 1D and 2D."""
        ps = LowLatencyPitchShifter(sample_rate=self.sr)

        # 1D array
        chunk_1d = np.random.uniform(-0.5, 0.5, 128).astype(np.float32)
        out_1d = ps.process(chunk_1d, semitones=4.0)
        self.assertEqual(out_1d.shape, chunk_1d.shape)
        self.assertEqual(out_1d.dtype, np.float32)
        self.assertFalse(np.isnan(out_1d).any())
        self.assertFalse(np.isinf(out_1d).any())

        # 2D stereo array
        chunk_2d = np.random.uniform(-0.5, 0.5, (256, 2)).astype(np.float32)
        out_2d = ps.process(chunk_2d, semitones=-6.0)
        self.assertEqual(out_2d.shape, chunk_2d.shape)
        self.assertEqual(out_2d.dtype, np.float32)
        self.assertFalse(np.isnan(out_2d).any())
        self.assertFalse(np.isinf(out_2d).any())

    def test_bypass_mode(self):
        """Verify bypass mode passes audio through without modification."""
        self.vc.set_preset("bypass")
        self.assertFalse(self.vc.enabled)

        chunk = np.random.uniform(-0.4, 0.4, (128, 2)).astype(np.float32)
        out = self.vc.process(chunk)
        np.testing.assert_array_almost_equal(out, chunk)

    def test_all_presets_execution(self):
        """Verify all presets process audio successfully without errors or NaNs."""
        presets = ["woman", "deep_voice", "chipmunk", "robot", "radio", "monster", "custom"]
        chunk = np.random.uniform(-0.3, 0.3, (128, 2)).astype(np.float32)

        for p in presets:
            self.vc.set_preset(p)
            self.assertTrue(self.vc.enabled, f"Preset {p} should enable voice changer")
            out = self.vc.process(chunk)
            self.assertEqual(out.shape, chunk.shape)
            self.assertEqual(out.dtype, np.float32)
            self.assertFalse(np.isnan(out).any(), f"NaN found in preset {p}")
            self.assertFalse(np.isinf(out).any(), f"Inf found in preset {p}")

    def test_woman_voice_preset(self):
        """Verify Woman Voice preset applies +4 semitones pitch and formant filtering."""
        self.vc.set_preset("woman")
        self.assertEqual(self.vc.pitch_semitones, 4.0)
        self.assertEqual(self.vc.mode, "woman")

        chunk = np.random.uniform(-0.4, 0.4, (256, 2)).astype(np.float32)
        out = self.vc.process(chunk)
        self.assertEqual(out.shape, chunk.shape)
        self.assertFalse(np.isnan(out).any())

    def test_dry_wet_mix(self):
        """Verify dry/wet blending ratio works as expected."""
        self.vc.set_preset("deep_voice")
        chunk = np.random.uniform(-0.3, 0.3, (128, 2)).astype(np.float32)

        # 100% Dry
        self.vc.set_mix(0.0)
        out_dry = self.vc.process(chunk)
        np.testing.assert_array_almost_equal(out_dry, chunk)

        # 100% Wet
        self.vc.set_mix(1.0)
        out_wet = self.vc.process(chunk)
        # Wet output should differ from dry
        self.assertFalse(np.allclose(out_wet, chunk, atol=1e-3))

    def test_realtime_performance_benchmark(self):
        """Verify that a 128-sample block processes in under 1.0 ms (low latency budget)."""
        self.vc.set_preset("deep_voice")
        chunk = np.random.uniform(-0.5, 0.5, (128, 2)).astype(np.float32)

        # Warm-up
        for _ in range(10):
            self.vc.process(chunk)

        # Timed execution over 50 blocks
        t0 = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            self.vc.process(chunk)
        dt_per_block_ms = ((time.perf_counter() - t0) / iterations) * 1000.0

        # Must execute in under 1ms (128 samples at 48kHz is 2.67ms)
        self.assertLess(dt_per_block_ms, 1.0, f"Block processing time {dt_per_block_ms:.4f} ms exceeded 1.0 ms limit")

    def test_engine_voice_changer_power_decoupling(self):
        """Verify MicBoostEngine decouples preset selection from on/off power state."""
        from engine.audio_engine import MicBoostEngine

        engine = MicBoostEngine(
            input_device=None,
            output_device=None,
            sample_rate=48000,
            block_size=128,
            gain_db=0.0,
            profile="flat",
            limiter_enabled=False,
        )

        # 1. Initially disabled
        self.assertFalse(engine.vc_enabled)
        self.assertFalse(engine.voice_changer.enabled)

        # 2. Selecting a preset while OFF must NOT enable the voice changer
        engine.set_voice_changer_preset("robot")
        self.assertEqual(engine.voice_changer.current_preset, "robot")
        self.assertFalse(engine.vc_enabled, "Preset selection must not turn on master VC power")
        self.assertFalse(engine.voice_changer.enabled, "Engine voice changer must remain disabled")

        # 3. Audio callback must pass audio unaltered when VC is OFF
        chunk_in = np.ones((128, 1), dtype=np.float32) * 0.4
        chunk_out = np.zeros((128, 2), dtype=np.float32)
        engine._audio_callback(chunk_in, chunk_out, 128, None, None)
        # Should be flat 0.4 on both channels (stereo duplicated, no robot modulation)
        np.testing.assert_array_almost_equal(chunk_out[:, 0], chunk_in[:, 0], decimal=4)

        # 4. Turning ON master VC power activates the selected preset
        engine.set_voice_changer_enabled(True)
        self.assertTrue(engine.vc_enabled)
        self.assertTrue(engine.voice_changer.enabled)
        self.assertEqual(engine.voice_changer.current_preset, "robot")

        # 5. Selecting another preset while ON immediately changes effect and stays ON
        engine.set_voice_changer_preset("chipmunk")
        self.assertTrue(engine.vc_enabled)
        self.assertTrue(engine.voice_changer.enabled)
        self.assertEqual(engine.voice_changer.current_preset, "chipmunk")

        # 6. Selecting 'bypass' / 'Normal' while ON does NOT turn off the master switch
        engine.set_voice_changer_preset("bypass")
        self.assertTrue(engine.vc_enabled, "Selecting Normal/bypass must not toggle off master power")

        # 7. Turning OFF master power leaves preset configured
        engine.set_voice_changer_preset("deep_voice")
        engine.set_voice_changer_enabled(False)
        self.assertFalse(engine.vc_enabled)
        self.assertFalse(engine.voice_changer.enabled)
        self.assertEqual(engine.voice_changer.current_preset, "deep_voice")

    def test_formant_vocal_tract_modeler(self):
        """Verify FormantVocalTractModeler processing, parameter updates, and reset."""
        modeler = FormantVocalTractModeler(sample_rate=self.sr)
        modeler.set_parameters(formant_st=2.5, chest_cut=True, breathiness=0.4)
        self.assertEqual(modeler.formant_semitones, 2.5)
        self.assertTrue(modeler.chest_cut_enabled)
        self.assertEqual(modeler.breathiness, 0.4)

        chunk_2d = np.random.uniform(-0.3, 0.3, (128, 2)).astype(np.float32)
        out = modeler.process(chunk_2d)
        self.assertEqual(out.shape, chunk_2d.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertFalse(np.isnan(out).any())

        # Test reset
        modeler.reset()

        # Inactive / flat check (0 semitones, no chest cut, no breathiness)
        modeler.set_parameters(0.0, False, 0.0)
        out_flat = modeler.process(chunk_2d)
        np.testing.assert_array_almost_equal(out_flat, chunk_2d, decimal=3)

    def test_voice_changer_formant_controls(self):
        """Verify VoiceChangerEngine methods for controlling formant, chest cut, and breathiness."""
        self.vc.set_preset("woman")
        self.assertEqual(self.vc.formant_semitones, 2.2)
        self.assertTrue(self.vc.chest_cut)
        self.assertAlmostEqual(self.vc.breathiness, 0.35)

        # Custom adjustment
        self.vc.set_formant_semitones(4.0)
        self.assertEqual(self.vc.formant_semitones, 4.0)
        self.assertEqual(self.vc.current_preset, "custom")

        self.vc.set_chest_cut(False)
        self.assertFalse(self.vc.chest_cut)

        self.vc.set_breathiness(0.5)
        self.assertEqual(self.vc.breathiness, 0.5)


if __name__ == "__main__":
    unittest.main()

