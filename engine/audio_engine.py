"""
Audio engine managing input/output streams, gain, limiter,
voice changer, and soundboard mixing.
"""

import queue
import sys
import threading
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import sounddevice as sd

from .dsp import ParametricEQChain, SoftLimiter, calculate_levels, db_to_linear, linear_to_db
from .profiles import DEFAULT_PROFILE_KEY, SOUND_PROFILES, SoundProfile
from .soundboard import SoundboardEngine
from .music_engine import MusicEngine
from .voice_changer import VoiceChangerEngine


class AudioDeviceManager:
    """Manages audio device querying, filtering, and virtual cable discovery."""

    @staticmethod
    def get_wasapi_hostapi_index() -> Optional[int]:
        """Find host API index for Windows WASAPI (lowest latency, clean names)."""
        try:
            for idx, api in enumerate(sd.query_hostapis()):
                if "wasapi" in api.get("name", "").lower():
                    return idx
        except Exception:
            pass
        return None

    @staticmethod
    def get_input_devices(wasapi_only: bool = True) -> List[Dict]:
        """
        Return list of clean audio input devices.
        On Windows, filters to WASAPI to eliminate duplicates from legacy MME/DirectSound.
        """
        devices = sd.query_devices()
        wasapi_idx = AudioDeviceManager.get_wasapi_hostapi_index() if wasapi_only else None

        input_devs = []
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) <= 0:
                continue
            if wasapi_idx is not None and dev.get("hostapi") != wasapi_idx:
                continue
            name = dev.get("name", "")
            if any(ign in name.lower() for ign in ["sound mapper", "primary sound capture"]):
                continue

            d = dict(dev)
            d["index"] = idx
            input_devs.append(d)

        if not input_devs and wasapi_only:
            return AudioDeviceManager.get_input_devices(wasapi_only=False)
        return input_devs

    @staticmethod
    def get_output_devices(wasapi_only: bool = True) -> List[Dict]:
        """Return list of clean audio output devices."""
        devices = sd.query_devices()
        wasapi_idx = AudioDeviceManager.get_wasapi_hostapi_index() if wasapi_only else None

        output_devs = []
        for idx, dev in enumerate(devices):
            if dev.get("max_output_channels", 0) <= 0:
                continue
            if wasapi_idx is not None and dev.get("hostapi") != wasapi_idx:
                continue
            name = dev.get("name", "")
            if any(ign in name.lower() for ign in ["sound mapper", "primary sound capture"]):
                continue

            d = dict(dev)
            d["index"] = idx
            output_devs.append(d)

        if not output_devs and wasapi_only:
            return AudioDeviceManager.get_output_devices(wasapi_only=False)
        return output_devs

    @staticmethod
    def get_default_input_index() -> Optional[int]:
        """Index of the system's default input device."""
        try:
            dev = sd.default.device[0]
            return dev if dev >= 0 else None
        except Exception:
            return None

    @staticmethod
    def get_default_output_index() -> Optional[int]:
        """Index of the system's default output device."""
        try:
            dev = sd.default.device[1]
            return dev if dev >= 0 else None
        except Exception:
            return None

    @staticmethod
    def find_virtual_cable_index() -> Optional[int]:
        """
        Auto-discover Virtual Audio Cable (e.g. VB-Audio Cable / Virtual Audio Cable).
        Returns device index if found, None otherwise.
        """
        devices = sd.query_devices()
        keywords = ["cable input", "vb-audio", "virtual audio cable", "virtual cable"]
        for idx, dev in enumerate(devices):
            if dev.get("max_output_channels", 0) > 0:
                name = dev.get("name", "").lower()
                if any(kw in name for kw in keywords):
                    return idx
        return None


class MicBoostEngine:
    """Audio stream pipeline for mic input, EQ, voice changer, and soundboard."""

    def __init__(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        monitor_device: Optional[int] = None,
        sample_rate: int = 48000,
        block_size: int = 128,
        gain_db: float = 0.0,
        profile: str = DEFAULT_PROFILE_KEY,
        limiter_enabled: bool = True,
        mute: bool = False,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.monitor_device = monitor_device
        self.sample_rate = sample_rate
        self.block_size = block_size

        # Parameter lock for thread-safe mutations
        self._lock = threading.Lock()

        # Gain parameters
        self._target_gain_db = float(gain_db)
        self._target_gain_linear = db_to_linear(gain_db)
        self._current_gain_linear = self._target_gain_linear
        self.mute = mute

        # Equalization & Filtering chain
        self._profile_key = profile
        prof = SOUND_PROFILES.get(profile, SOUND_PROFILES[DEFAULT_PROFILE_KEY])
        self.eq_chain = ParametricEQChain(sample_rate=sample_rate)
        self.eq_chain.configure_bands(prof.bands)

        # Broadcast Limiter (Dynamic envelope limiter prevents waveshaping distortion on music)
        self.limiter_enabled = limiter_enabled
        self.limiter = SoftLimiter(
            threshold_db=-0.5,
            ceiling_db=-0.1,
            attack_ms=0.5,
            release_ms=40.0,
            sample_rate=sample_rate,
            mode="hybrid",
        )

        # Subsystems: Soundboard, Voice Changer, and Music Engines
        self.soundboard = SoundboardEngine(sample_rate=sample_rate)
        self.voice_changer = VoiceChangerEngine(sample_rate=sample_rate)
        self.vc_enabled = False
        self.voice_changer.enabled = False
        self.music = MusicEngine(sample_rate=sample_rate)

        # Headphone Monitoring (User Self-Listen)
        self.monitor_enabled = False
        self.monitor_mic = False
        self._monitor_stream: Optional[sd.OutputStream] = None
        self._monitor_queue: queue.Queue = queue.Queue(maxsize=32)

        # Query hardware channel counts
        if in_channels is not None:
            self.in_channels = in_channels
        else:
            try:
                in_info = sd.query_devices(self.input_device, "input")
                self.in_channels = min(2, in_info.get("max_input_channels", 1))
            except Exception:
                self.in_channels = 1

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            try:
                out_info = sd.query_devices(self.output_device, "output")
                self.out_channels = min(2, out_info.get("max_output_channels", 2))
            except Exception:
                self.out_channels = 2

        # Telemetry / VU Meter metrics
        self.pre_peak_db = -96.0
        self.pre_rms_db = -96.0
        self.post_peak_db = -96.0
        self.post_rms_db = -96.0
        self.is_limiting = False
        self.buffer_underflow_count = 0
        self.buffer_overflow_count = 0

        # Stream instance & state
        self._stream: Optional[sd.Stream] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the audio stream is currently active."""
        return self._running and self._stream is not None and self._stream.active

    @property
    def gain_db(self) -> float:
        """Current target gain in decibels."""
        return self._target_gain_db

    @property
    def current_profile(self) -> str:
        """Currently active sound profile key."""
        return self._profile_key

    def set_profile(self, profile_key: str):
        """Switch active sound profile (updates EQ filter bands seamlessly)."""
        with self._lock:
            if profile_key not in SOUND_PROFILES:
                profile_key = DEFAULT_PROFILE_KEY
            self._profile_key = profile_key
            prof = SOUND_PROFILES[profile_key]
            self.eq_chain.configure_bands(prof.bands)

    def set_gain_db(self, gain_db: float):
        """Set digital boost gain in decibels (smoothly interpolated across buffer)."""
        with self._lock:
            self._target_gain_db = float(gain_db)
            self._target_gain_linear = db_to_linear(gain_db)

    def set_limiter_enabled(self, enabled: bool):
        """Enable or disable the soft-knee limiter."""
        with self._lock:
            self.limiter_enabled = enabled
            self.limiter.enabled = enabled

    def set_voice_changer_preset(self, preset_key: str):
        """Set voice changer vocal effect preset."""
        with self._lock:
            self.voice_changer.set_preset(preset_key)
            if not self.vc_enabled:
                self.voice_changer.enabled = False

    def set_voice_changer_pitch(self, semitones: float):
        """Set custom voice changer pitch in semitones (-12.0 to +12.0)."""
        with self._lock:
            self.voice_changer.set_pitch_semitones(semitones)

    def set_voice_changer_enabled(self, enabled: bool):
        """Toggle voice changer on or off."""
        with self._lock:
            self.vc_enabled = enabled
            self.voice_changer.set_enabled(enabled)

    def set_voice_changer_mix(self, mix: float):
        """Set voice changer dry/wet mix ratio (0.0 to 1.0)."""
        with self._lock:
            self.voice_changer.set_mix(mix)

    def set_mute(self, mute: bool):
        """Mute or unmute the microphone stream."""
        with self._lock:
            self.mute = mute

    def set_monitor_enabled(self, enabled: bool, monitor_device: Optional[int] = None):
        """Enable or disable local headphone monitoring."""
        with self._lock:
            self.monitor_enabled = enabled
            if monitor_device is not None:
                self.monitor_device = monitor_device

        if enabled and self._running:
            self._start_monitor_stream()
        elif not enabled:
            self._stop_monitor_stream()

    def _start_monitor_stream(self):
        """Start the secondary monitor audio stream to headphones."""
        if self.monitor_device is None or self._monitor_stream is not None:
            return

        def monitor_cb(outdata, frames, time_info, status):
            try:
                data = self._monitor_queue.get_nowait()
                outdata[:] = data
            except queue.Empty:
                outdata.fill(0.0)

        try:
            self._monitor_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.monitor_device,
                channels=2,
                dtype=np.float32,
                callback=monitor_cb,
            )
            self._monitor_stream.start()
        except Exception as e:
            print(f"[WARN] Failed to start monitor stream: {e}")
            self._monitor_stream = None

    def _stop_monitor_stream(self):
        """Stop headphone monitoring stream."""
        if self._monitor_stream is not None:
            try:
                self._monitor_stream.stop()
                self._monitor_stream.close()
            except Exception:
                pass
            self._monitor_stream = None
        while not self._monitor_queue.empty():
            try:
                self._monitor_queue.get_nowait()
            except queue.Empty:
                break

    def get_telemetry(self) -> Dict[str, Union[float, bool, int, str]]:
        """Retrieve latest snapshot of VU levels, limiter state, profile, and stream stats."""
        return {
            "pre_peak_db": self.pre_peak_db,
            "pre_rms_db": self.pre_rms_db,
            "post_peak_db": self.post_peak_db,
            "post_rms_db": self.post_rms_db,
            "gain_db": self._target_gain_db,
            "profile": self._profile_key,
            "is_limiting": self.is_limiting,
            "is_muted": self.mute,
            "limiter_enabled": self.limiter_enabled,
            "overflows": self.buffer_overflow_count,
            "underflows": self.buffer_underflow_count,
            # Music & Soundboard Telemetry
            "music_peak_db": self.music.peak_db,
            "music_rms_db": self.music.rms_db,
            "is_ducking": self.music.ducker.is_ducking,
            "duck_gain": self.music.ducker.current_gain,
            "soundboard_active_voices": len(self.soundboard.active_voices),
            "loopback_active": self.music.loopback.is_active,
            "stream_playing": self.music.stream_player._running,
        }

    def _audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time_info, status):
        """
        High-priority real-time audio processing callback.
        Seamlessly mixes Microphone + Soundboard + YouTube Music into master output.
        Kept lean and 100% vectorized for sub-millisecond execution.
        """
        if status:
            if status.input_overflow:
                self.buffer_overflow_count += 1
            if status.output_underflow:
                self.buffer_underflow_count += 1

        # 1. Pre-gain metering (raw mic input level)
        pre_peak, pre_rms = calculate_levels(indata)
        self.pre_peak_db = pre_peak
        self.pre_rms_db = pre_rms

        # 2. Process Microphone Signal
        if self.mute:
            mic_processed = np.zeros((frames, 2), dtype=np.float32)
        else:
            equalized = self.eq_chain.process(indata)

            # Apply Voice Changer (Pitch Shift, Robot, Megaphone, etc.)
            if self.vc_enabled and self.voice_changer.enabled:
                transformed = self.voice_changer.process(equalized)
            else:
                transformed = equalized

            target_g = self._target_gain_linear
            curr_g = self._current_gain_linear

            if abs(curr_g - target_g) > 1e-4:
                ramp = np.linspace(curr_g, target_g, frames, dtype=np.float32)
                if transformed.ndim > 1:
                    ramp = ramp[:, np.newaxis]
                boosted = transformed * ramp
                self._current_gain_linear = target_g
            else:
                boosted = transformed * target_g

            # Ensure mic signal is stereo (frames, 2)
            if self.in_channels == 1 or boosted.shape[1] == 1:
                mic_processed = np.repeat(boosted[:, :1], 2, axis=1)
            else:
                mic_processed = boosted[:, :2]

        # 3. Pull Soundboard audio chunk
        soundboard_chunk = self.soundboard.read_chunk(frames)

        # 4. Pull Music audio chunk (with microphone-triggered Auto-Ducking)
        active_mic_rms = -96.0 if self.mute else pre_rms
        music_chunk = self.music.read_chunk(frames, active_mic_rms)

        # 5. Master Multi-Source Mix
        mixed = mic_processed + soundboard_chunk + music_chunk

        # 6. Master Limiter / Soft-Clipping Protection (Prevents distortion on combined signal)
        if self.limiter_enabled:
            boosted_peak = float(np.max(np.abs(mixed)))
            self.is_limiting = boosted_peak > self.limiter.threshold
            final_output = self.limiter.process(mixed)
        else:
            self.is_limiting = False
            final_output = np.clip(mixed, -1.0, 1.0)

        # 7. Map to hardware output channels (VB-Audio Virtual Cable)
        if self.out_channels == 1:
            outdata[:, 0] = np.mean(final_output, axis=1)
        else:
            outdata[:, 0] = final_output[:, 0]
            outdata[:, 1] = final_output[:, 1]

        # 8. Post-gain metering
        post_peak, post_rms = calculate_levels(outdata)
        self.post_peak_db = post_peak
        self.post_rms_db = post_rms

        # 9. Send audio to Headphone Monitor (User Self-Listen)
        if self.monitor_enabled and self._monitor_stream is not None:
            # Send Soundboard and Stream Player music to headphones so user can hear it
            # (Note: Loopback music is already heard by user via their browser, so not duplicated)
            mon_music = self.music.stream_player.read_samples(frames) if not self.music.mute else np.zeros((frames, 2), dtype=np.float32)
            monitor_mix = soundboard_chunk + (mon_music * self.music.volume)
            if self.monitor_mic and not self.mute:
                monitor_mix += mic_processed
            try:
                self._monitor_queue.put_nowait(monitor_mix.astype(np.float32))
            except Exception:
                pass

    def start(self):
        """Start the real-time audio stream."""
        with self._lock:
            if self._running:
                return

            self._current_gain_linear = self._target_gain_linear
            self.eq_chain.reset()
            self.voice_changer.reset()
            self.limiter.reset()

            self._stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(self.input_device, self.output_device),
                channels=(self.in_channels, self.out_channels),
                dtype=np.float32,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True

        if self.monitor_enabled and self.monitor_device is not None:
            self._start_monitor_stream()

    def stop(self):
        """Stop and close the real-time audio stream."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

        self._stop_monitor_stream()
        self.music.enable_loopback(False)
        self.music.stream_player.stop()
        self.soundboard.stop_all()

    def restart(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        monitor_device: Optional[int] = None,
    ):
        """Restart the stream with updated devices if specified."""
        was_running = self.is_running
        self.stop()
        if input_device is not None:
            self.input_device = input_device
        if output_device is not None:
            self.output_device = output_device
        if monitor_device is not None:
            self.monitor_device = monitor_device

        # Re-query channel counts for the new devices
        try:
            in_info = sd.query_devices(self.input_device, "input")
            self.in_channels = min(2, in_info.get("max_input_channels", 1))
        except Exception:
            self.in_channels = 1

        try:
            out_info = sd.query_devices(self.output_device, "output")
            self.out_channels = min(2, out_info.get("max_output_channels", 2))
        except Exception:
            self.out_channels = 2

        if was_running:
            self.start()

    def switch_devices(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        monitor_device: Optional[int] = None,
    ):
        """Switch audio devices (alias for restart)."""
        self.restart(
            input_device=input_device,
            output_device=output_device,
            monitor_device=monitor_device,
        )
