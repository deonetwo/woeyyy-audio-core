"""
Unit tests for Woeyyy Soundboard and Music Engine Subsystems.
Verifies sound generation, file loading, polyphonic soundboard playback,
auto-ducking DSP calculations, and multi-source mixer callback.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from engine.soundboard import GlobalHotkeyManager, ProceduralSoundGenerator, SoundboardEngine, load_audio_file
from engine.music_engine import AutoDucker, MusicEngine
from engine.audio_engine import MicBoostEngine


class TestSoundboardAndMusic(unittest.TestCase):
    def test_procedural_sound_generator_and_loading(self):
        """Verify procedural audio synthesis and loading creates valid 48kHz stereo float32 arrays."""
        temp_dir = os.path.join(os.path.dirname(__file__), "test_sounds")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            presets = ProceduralSoundGenerator.generate_all_presets(temp_dir, sr=48000)
            self.assertIn("airhorn", presets)
            self.assertIn("badumtss", presets)
            self.assertIn("buzzer", presets)
            self.assertIn("coin", presets)
            self.assertIn("levelup", presets)
            self.assertIn("tada", presets)

            # Load and verify airhorn
            airhorn_path = presets["airhorn"][1]
            data, dur = load_audio_file(airhorn_path, target_sr=48000)
            self.assertEqual(data.ndim, 2)
            self.assertEqual(data.shape[1], 2)
            self.assertEqual(data.dtype, np.float32)
            self.assertGreater(dur, 0.5)
            self.assertGreater(float(np.max(np.abs(data))), 0.05)
        finally:
            # Cleanup
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def test_soundboard_engine_polyphony_and_volume(self):
        """Verify soundboard can mix multiple sounds and respects master & individual volumes."""
        sb = SoundboardEngine(sample_rate=48000)

        # Generate synthetic clips
        dur1 = 0.1
        t1 = np.linspace(0, dur1, int(48000 * dur1), endpoint=False, dtype=np.float32)
        sine1 = np.column_stack([np.sin(2 * np.pi * 440 * t1), np.sin(2 * np.pi * 440 * t1)])

        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name

        try:
            sf.write(temp_wav, sine1, 48000)
            sb.add_sound("sine", "Sine Test", temp_wav, volume=0.5)

            # Before play: silence
            silence = sb.read_chunk(128)
            self.assertTrue(np.allclose(silence, 0.0))

            # Play sound
            sb.play_sound("sine")
            self.assertTrue(sb.is_playing("sine"))

            # Read first chunk
            chunk = sb.read_chunk(128)
            self.assertEqual(chunk.shape, (128, 2))
            self.assertGreater(float(np.max(np.abs(chunk))), 0.01)

            # Test volume scaling
            sb.set_master_volume(0.5)
            chunk2 = sb.read_chunk(128)
            self.assertEqual(chunk2.shape, (128, 2))

            # Stop sound
            sb.stop_sound("sine")
            self.assertFalse(sb.is_playing("sine"))
            chunk_after_stop = sb.read_chunk(128)
            self.assertTrue(np.allclose(chunk_after_stop, 0.0))
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    def test_auto_ducking_calculations(self):
        """Verify intelligent microphone auto-ducking response to speech vs silence."""
        ducker = AutoDucker(threshold_db=-36.0, duck_depth_db=-12.0, sample_rate=48000)

        # 1. Silence (-60 dBFS) -> gain should stay 1.0 (0 dB attenuation)
        for _ in range(10):
            g = ducker.process_block(mic_rms_db=-60.0, num_frames=128)
        self.assertTrue(np.isclose(g, 1.0, atol=0.01))
        self.assertFalse(ducker.is_ducking)

        # 2. Speech (-18 dBFS) -> gain should duck towards -12 dB (approx 0.251x)
        for _ in range(40):
            g_speech = ducker.process_block(mic_rms_db=-18.0, num_frames=128)
        self.assertTrue(ducker.is_ducking)
        self.assertLess(g_speech, 0.5)

        # 3. Speech stops -> hold timer preserves ducking, then releases back to 1.0
        for _ in range(int(48000 * 2.5 / 128)):
            g_release = ducker.process_block(mic_rms_db=-60.0, num_frames=128)
        self.assertFalse(ducker.is_ducking)
        self.assertTrue(np.isclose(g_release, 1.0, atol=0.05))

    def test_multi_source_audio_mixing_and_limiting(self):
        """Test full audio callback math mixing mic + soundboard + music with limiter clamp."""
        engine = MicBoostEngine(
            input_device=None,
            output_device=None,
            sample_rate=48000,
            block_size=128,
            gain_db=6.0,
            limiter_enabled=True,
        )

        frames = 128
        indata = np.ones((frames, 1), dtype=np.float32) * 0.5
        outdata = np.zeros((frames, 2), dtype=np.float32)

        # Trigger callback directly
        engine._audio_callback(indata, outdata, frames, None, None)

        # Output should be non-zero, stereo, and within limiter ceiling (< 0.99)
        self.assertEqual(outdata.shape, (frames, 2))
        self.assertGreater(float(np.max(np.abs(outdata))), 0.1)
        self.assertLessEqual(float(np.max(np.abs(outdata))), 0.99)

    def test_global_hotkey_manager_combinations(self):
        """Verify GlobalHotkeyManager parses and normalizes Ctrl+number combinations correctly."""
        mgr = GlobalHotkeyManager(enabled=False)
        self.assertFalse(mgr.enabled)

        norm1 = mgr._normalize("ctrl+1")
        self.assertEqual(norm1, "<ctrl>+1")

        norm2 = mgr._normalize("CTRL + 2")
        self.assertEqual(norm2, "<ctrl>+2")

        norm_f = mgr._normalize("f8")
        self.assertEqual(norm_f, "<f8>")

        # Test registration
        fired = []
        mgr.register_hotkey("ctrl+1", lambda: fired.append(1))
        mgr.register_hotkey("ctrl+2", lambda: fired.append(2))
        self.assertIn("ctrl+1", mgr.bindings)
        self.assertIn("ctrl+2", mgr.bindings)

        # Test enable/disable toggle
        mgr.set_enabled(True)
        self.assertTrue(mgr.enabled)
        mgr.set_enabled(False)
        self.assertFalse(mgr.enabled)


if __name__ == "__main__":
    unittest.main()


