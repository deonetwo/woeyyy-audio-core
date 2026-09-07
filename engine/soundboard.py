"""
Soundboard engine for audio playback and global hotkey handling.
"""

import os
import threading
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.io import wavfile

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import av
except ImportError:
    av = None

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 2D numpy audio array using scipy.signal.resample_poly."""
    if orig_sr == target_sr or len(data) == 0:
        return data
    from math import gcd
    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled = resample_poly(data, up, down, axis=0)
    return resampled.astype(np.float32)


def load_audio_file(file_path: str, target_sr: int = 48000) -> Tuple[np.ndarray, float]:
    """
    Load any audio file (WAV, MP3, OGG, FLAC, M4A, etc.) and convert to
    normalized float32 stereo array at target_sr (48,000 Hz).
    Returns (data_array, duration_seconds).
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # 1. Try soundfile first if available
    if sf is not None:
        try:
            data, sr = sf.read(file_path, dtype="float32", always_2d=True)
            if sr != target_sr:
                data = resample_audio(data, sr, target_sr)
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            elif data.shape[1] > 2:
                data = data[:, :2]
            duration = float(len(data)) / float(target_sr)
            return data.astype(np.float32), duration
        except Exception:
            pass

    # 2. Try scipy.io.wavfile (standard in scipy)
    try:
        sr, data = wavfile.read(file_path)
        if data.dtype == np.int16:
            data = (data / 32768.0).astype(np.float32)
        elif data.dtype == np.int32:
            data = (data / 2147483648.0).astype(np.float32)
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        if data.ndim == 1:
            data = np.column_stack([data, data])
        elif data.shape[1] > 2:
            data = data[:, :2]

        if sr != target_sr:
            data = resample_audio(data, sr, target_sr)

        duration = float(len(data)) / float(target_sr)
        return data.astype(np.float32), duration
    except Exception:
        pass

    # 2. Fallback to PyAV (robust FFmpeg decoder for any format including M4A, AAC, MP3)
    if av is not None:
        try:
            container = av.open(file_path)
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_sr)
            chunks = []
            for frame in container.decode(audio=0):
                resampled_frames = resampler.resample(frame)
                for rf in resampled_frames:
                    arr = rf.to_ndarray()  # shape: (2, N) float32
                    chunks.append(arr.T)  # transpose to (N, 2)
            container.close()

            if chunks:
                data = np.concatenate(chunks, axis=0).astype(np.float32)
                duration = float(len(data)) / float(target_sr)
                return data, duration
        except Exception as e:
            raise RuntimeError(f"Failed to decode audio file {file_path}: {e}")

    raise RuntimeError(f"Unable to read audio file format: {file_path}")


class SoundClip:
    """Represents a cached audio soundboard clip."""

    def __init__(
        self,
        clip_id: str,
        name: str,
        file_path: str,
        data: np.ndarray,
        duration: float,
        volume: float = 1.0,
        hotkey: Optional[str] = None,
    ):
        self.id = clip_id
        self.name = name
        self.file_path = file_path
        self.data = data.astype(np.float32)
        self.duration = duration
        self.volume = max(0.0, min(2.0, float(volume)))
        self.hotkey = hotkey


class ActiveVoice:
    """Represents an active playing instance of a SoundClip."""

    def __init__(self, clip: SoundClip, loop: bool = False, volume: float = 1.0):
        self.clip = clip
        self.position = 0
        self.loop = loop
        self.volume = volume

    @property
    def is_finished(self) -> bool:
        return not self.loop and self.position >= len(self.clip.data)


class SoundboardEngine:
    """
    Thread-safe polyphonic soundboard audio engine.
    Mixes multiple simultaneous sound clips into a continuous audio stream buffer.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.clips: Dict[str, SoundClip] = {}
        self.active_voices: List[ActiveVoice] = []
        self._lock = threading.Lock()
        self.master_volume = 1.0
        self.enabled = True

    def add_sound(
        self,
        clip_id: str,
        name: str,
        file_path: str,
        volume: float = 1.0,
        hotkey: Optional[str] = None,
    ) -> SoundClip:
        """Load and cache an audio file into the soundboard library."""
        data, duration = load_audio_file(file_path, target_sr=self.sample_rate)
        clip = SoundClip(
            clip_id=clip_id,
            name=name,
            file_path=file_path,
            data=data,
            duration=duration,
            volume=volume,
            hotkey=hotkey,
        )
        with self._lock:
            self.clips[clip_id] = clip
        return clip

    def remove_sound(self, clip_id: str):
        """Remove a sound clip from the soundboard library."""
        with self._lock:
            self.active_voices = [v for v in self.active_voices if v.clip.id != clip_id]
            if clip_id in self.clips:
                del self.clips[clip_id]

    def play_sound(self, clip_id: str, loop: bool = False, volume: Optional[float] = None):
        """Trigger playback of a sound clip."""
        with self._lock:
            if not self.enabled or clip_id not in self.clips:
                return
            clip = self.clips[clip_id]
            vol = clip.volume if volume is None else volume

            # If clip is already playing and loop is requested, or stop previous instance
            self.active_voices = [v for v in self.active_voices if v.clip.id != clip_id]
            self.active_voices.append(ActiveVoice(clip, loop=loop, volume=vol))

    def stop_sound(self, clip_id: str):
        """Stop playback of a specific sound clip."""
        with self._lock:
            self.active_voices = [v for v in self.active_voices if v.clip.id != clip_id]

    def stop_all(self):
        """Panic stop: instantly stop all active playing sounds."""
        with self._lock:
            self.active_voices.clear()

    def is_playing(self, clip_id: str) -> bool:
        """Check if a specific sound clip is currently playing."""
        with self._lock:
            return any(v.clip.id == clip_id for v in self.active_voices)

    def set_master_volume(self, volume: float):
        """Set soundboard master volume multiplier (0.0 to 2.0)."""
        with self._lock:
            self.master_volume = max(0.0, min(2.0, float(volume)))

    def play(self, file_path: str, volume: Optional[float] = None):
        """Play an audio file directly into the soundboard mix."""
        if not os.path.isfile(file_path):
            return
        clip_id = f"file_{os.path.abspath(file_path)}"
        with self._lock:
            if clip_id not in self.clips:
                data, dur = load_audio_file(file_path, target_sr=self.sample_rate)
                self.clips[clip_id] = SoundClip(
                    clip_id=clip_id,
                    name=os.path.basename(file_path),
                    file_path=file_path,
                    data=data,
                    duration=dur,
                    volume=1.0 if volume is None else volume,
                )
        self.play_sound(clip_id, loop=False, volume=volume)

    def read_chunk(self, num_frames: int) -> np.ndarray:
        """
        Pull a mixed audio block of `num_frames` stereo samples.
        Called directly by the high-priority real-time audio callback.
        Execution time: <0.02 ms.
        """
        out = np.zeros((num_frames, 2), dtype=np.float32)

        with self._lock:
            if not self.enabled or not self.active_voices or self.master_volume <= 0.0:
                return out

            still_active = []
            for voice in self.active_voices:
                clip_data = voice.clip.data
                total_len = len(clip_data)
                remaining_frames = num_frames
                current_out_idx = 0
                voice_vol = voice.volume * voice.clip.volume * self.master_volume

                while remaining_frames > 0:
                    available = total_len - voice.position
                    if available <= 0:
                        if voice.loop:
                            voice.position = 0
                            available = total_len
                        else:
                            break

                    chunk_len = min(remaining_frames, available)
                    out[current_out_idx : current_out_idx + chunk_len] += (
                        clip_data[voice.position : voice.position + chunk_len] * voice_vol
                    )

                    voice.position += chunk_len
                    current_out_idx += chunk_len
                    remaining_frames -= chunk_len

                if not voice.is_finished:
                    still_active.append(voice)

            self.active_voices = still_active

        return out


class ProceduralSoundGenerator:
    """
    Generates high-quality procedural sound effects directly into .wav files.
    Allows testing and using the soundboard immediately out-of-the-box!
    """

    @staticmethod
    def generate_all_presets(target_dir: str, sr: int = 48000) -> Dict[str, str]:
        """Generate full suite of preset sounds and return dict of {name: file_path}."""
        os.makedirs(target_dir, exist_ok=True)
        presets = {
            "airhorn": ("Airhorn MLG", ProceduralSoundGenerator.create_airhorn),
            "badumtss": ("Ba-Dum-Tss (Rimshot)", ProceduralSoundGenerator.create_badumtss),
            "buzzer": ("Buzzer (Wrong)", ProceduralSoundGenerator.create_buzzer),
            "coin": ("8-Bit Coin", ProceduralSoundGenerator.create_coin),
            "levelup": ("Level Up Chime", ProceduralSoundGenerator.create_levelup),
            "tada": ("Tada! Victory Fanfare", ProceduralSoundGenerator.create_tada),
            "siren": ("Emergency Siren", ProceduralSoundGenerator.create_siren),
            "laser": ("Laser Blaster", ProceduralSoundGenerator.create_laser),
        }

        generated_paths = {}
        for key, (name, func) in presets.items():
            path = os.path.join(target_dir, f"{key}.wav")
            if not os.path.exists(path):
                data = func(sr=sr)
                if sf is not None:
                    sf.write(path, data, sr)
                else:
                    wavfile.write(path, sr, data.astype(np.float32))
            generated_paths[key] = (name, path)

        return generated_paths

    @staticmethod
    def create_airhorn(sr: int = 48000) -> np.ndarray:
        """Iconic multi-tone stadium airhorn fanfare."""
        # Rhythm: beep-beep-beep-beeeep
        beeps = [(0.10, 0.04), (0.10, 0.04), (0.10, 0.04), (0.45, 0.0)]
        freqs = [466.16, 523.25, 587.33]  # Bb4, C5, D5 brass stack

        audio_parts = []
        for dur, gap in beeps:
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            wave = np.zeros_like(t)
            for f in freqs:
                # Add harmonics for bright brass sound
                wave += np.sin(2 * np.pi * f * t) * 0.4
                wave += np.sin(2 * np.pi * (f * 2) * t) * 0.25
                wave += np.sin(2 * np.pi * (f * 3) * t) * 0.15
            # Envelope: fast attack, sustained, fast decay
            attack = int(sr * 0.01)
            decay = int(sr * 0.02)
            env = np.ones_like(t)
            env[:attack] = np.linspace(0, 1, attack)
            env[-decay:] = np.linspace(1, 0, decay)
            wave = wave * env

            audio_parts.append(wave)
            if gap > 0:
                audio_parts.append(np.zeros(int(sr * gap)))

        full = np.concatenate(audio_parts)
        stereo = np.column_stack([full, full])
        return (stereo / (np.max(np.abs(stereo)) + 1e-6) * 0.85).astype(np.float32)

    @staticmethod
    def create_badumtss(sr: int = 48000) -> np.ndarray:
        """Stand-up comedy drum rimshot: Ba-Dum-Tss!"""
        # Drum 1 (Ba): low kick/snare
        dur1 = 0.14
        t1 = np.linspace(0, dur1, int(sr * dur1), endpoint=False)
        f_sweep1 = np.linspace(180, 70, len(t1))
        drum1 = np.sin(2 * np.pi * f_sweep1 * t1) * np.exp(-t1 * 25)

        # Gap
        gap1 = np.zeros(int(sr * 0.05))

        # Drum 2 (Dum): lower kick/snare
        dur2 = 0.16
        t2 = np.linspace(0, dur2, int(sr * dur2), endpoint=False)
        f_sweep2 = np.linspace(160, 55, len(t2))
        drum2 = np.sin(2 * np.pi * f_sweep2 * t2) * np.exp(-t2 * 20)

        # Gap
        gap2 = np.zeros(int(sr * 0.06))

        # Tss (Cymbal/Hihat): filtered noise burst
        dur3 = 0.65
        noise = (np.random.rand(int(sr * dur3)) * 2 - 1)
        t3 = np.linspace(0, dur3, len(noise), endpoint=False)
        cymbal = noise * np.exp(-t3 * 7)

        full = np.concatenate([drum1, gap1, drum2, gap2, cymbal])
        stereo = np.column_stack([full, full])
        return (stereo / (np.max(np.abs(stereo)) + 1e-6) * 0.85).astype(np.float32)

    @staticmethod
    def create_buzzer(sr: int = 48000) -> np.ndarray:
        """Game show wrong buzzer (heavy distorted dual 120Hz buzz)."""
        dur = 0.55
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        # Sawtooth / harsh square-like wave
        wave = np.sign(np.sin(2 * np.pi * 125 * t)) * 0.5 + np.sign(np.sin(2 * np.pi * 187.5 * t)) * 0.5
        env = np.ones_like(t)
        decay = int(sr * 0.05)
        env[-decay:] = np.linspace(1, 0, decay)
        full = wave * env
        stereo = np.column_stack([full, full])
        return (stereo / (np.max(np.abs(stereo)) + 1e-6) * 0.85).astype(np.float32)

    @staticmethod
    def create_coin(sr: int = 48000) -> np.ndarray:
        """Classic 8-bit arcade coin chime."""
        dur1 = 0.08
        dur2 = 0.35
        t1 = np.linspace(0, dur1, int(sr * dur1), endpoint=False)
        t2 = np.linspace(0, dur2, int(sr * dur2), endpoint=False)

        # B5 (987.77 Hz) -> E6 (1318.51 Hz)
        w1 = np.sin(2 * np.pi * 987.77 * t1)
        w2 = np.sin(2 * np.pi * 1318.51 * t2) * np.exp(-t2 * 9)

        full = np.concatenate([w1, w2])
        stereo = np.column_stack([full, full])
        return (stereo / (np.max(np.abs(stereo)) + 1e-6) * 0.85).astype(np.float32)

    @staticmethod
    def create_levelup(sr: int = 48000) -> np.ndarray:
        """Celebratory ascending RPG level up arpeggio."""
        notes = [523.25, 659.25, 783.99, 1046.50]  # C5, E5, G5, C6
        parts = []
        for i, freq in enumerate(notes):
            dur = 0.12 if i < len(notes) - 1 else 0.45
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            decay = 5 if i < len(notes) - 1 else 3
            tone = (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)) * np.exp(-t * decay)
            parts.append(tone)

        full = np.concatenate(parts)
        stereo = np.column_stack([full, full])
        return (stereo / (np.max(np.abs(stereo)) + 1e-6) * 0.85).astype(np.float32)

    @staticmethod
    def create_tada(sr: int = 48000) -> np.ndarray:
        """Harmonious tada bell chord."""
        # Short preamble note then big major chord
        t_pre = np.linspace(0, 0.15, int(sr * 0.15), endpoint=False)
        w_pre = np.sin(2 * np.pi * 587.33 * t_pre)  # D5

        t_chord = np.linspace(0, 0.7, int(sr * 0.7), endpoint=False)
        chord_freqs = [523.25, 659.25, 783.99, 1046.50]
        w_chord = np.zeros_like(t_chord)
        for f in chord_freqs:
            w_chord += np.sin(2 * np.pi * f * t_chord) * np.exp(-t_chord * 4)

        full = np.concatenate([w_pre, w_chord])
        stereo = np.column_stack([full, full])
        return (stereo / (np.max(np.abs(stereo)) + 1e-6) * 0.85).astype(np.float32)

    @staticmethod
    def create_siren(sr: int = 48000) -> np.ndarray:
        """Emergency alternating police siren."""
        dur = 1.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        # Modulating frequency between 600 Hz and 950 Hz
        freq = 775.0 + 175.0 * np.sin(2 * np.pi * 3.0 * t)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        wave = np.sin(phase) * 0.85
        stereo = np.column_stack([wave, wave])
        return stereo.astype(np.float32)

    @staticmethod
    def create_laser(sr: int = 48000) -> np.ndarray:
        """Sci-Fi arcade laser beam."""
        dur = 0.28
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        freq = 1800 * np.exp(-t * 18) + 120
        phase = 2 * np.pi * np.cumsum(freq) / sr
        wave = np.sin(phase) * np.exp(-t * 7)
        stereo = np.column_stack([wave, wave])
        return (stereo / (np.max(np.abs(stereo)) + 1e-6) * 0.85).astype(np.float32)


class GlobalHotkeyManager:
    """
    Listens for global keyboard hotkeys in background and invokes bound callbacks.
    Supports key combinations like 'ctrl+1', 'ctrl+2', function keys ('f8'), etc.
    Safe on Windows, works even when user is tabbed out into Discord or a full-screen game.
    """

    def __init__(self, enabled: bool = False):
        self.bindings: Dict[str, Callable[[], None]] = {}
        self._hotkeys: List[Tuple[str, Any, Callable[[], None]]] = []
        self._listener = None
        self._lock = threading.Lock()
        self._running = False
        self.enabled = enabled

    def set_enabled(self, enabled: bool):
        """Enable or disable dispatching of hotkey actions."""
        with self._lock:
            self.enabled = enabled

    @staticmethod
    def _normalize(key_str: str) -> str:
        """Convert friendly hotkey strings (e.g. 'ctrl+1', 'ctrl+shift+a') to pynput parse format."""
        raw = key_str.strip().lower()
        if "<" in raw and ">" in raw:
            return raw
        parts = [p.strip() for p in raw.replace("+", " ").replace("-", " ").split() if p.strip()]
        formatted = []
        for p in parts:
            if p in ("ctrl", "control"):
                formatted.append("<ctrl>")
            elif p in ("alt", "menu"):
                formatted.append("<alt>")
            elif p in ("shift",):
                formatted.append("<shift>")
            elif p in ("cmd", "win", "windows", "super"):
                formatted.append("<cmd>")
            elif p.startswith("f") and p[1:].isdigit():
                formatted.append(f"<{p}>")
            elif p in ("space", "enter", "tab", "esc", "escape", "backspace", "delete"):
                formatted.append(f"<{p}>")
            else:
                formatted.append(p)
        return "+".join(formatted)

    def register_hotkey(self, key_name: str, callback: Callable[[], None]):
        """Register a hotkey (e.g. 'ctrl+1', 'ctrl+2', 'f8', 'num_1', etc.)."""
        with self._lock:
            clean_name = key_name.lower().strip()
            self.bindings[clean_name] = callback
            if keyboard is not None and hasattr(keyboard, "HotKey"):
                norm = self._normalize(key_name)
                try:
                    hk = keyboard.HotKey(keyboard.HotKey.parse(norm), callback)
                    self._hotkeys = [(k, h, c) for k, h, c in self._hotkeys if k != clean_name]
                    self._hotkeys.append((clean_name, hk, callback))
                except Exception as e:
                    print(f"[WARN] Failed to parse hotkey '{key_name}': {e}")

    def unregister_hotkey(self, key_name: str):
        """Unregister a hotkey."""
        clean_name = key_name.lower().strip()
        with self._lock:
            self.bindings.pop(clean_name, None)
            self._hotkeys = [(k, h, c) for k, h, c in self._hotkeys if k != clean_name]

    def start(self):
        """Start listening for keyboard events."""
        if keyboard is None or self._running:
            return

        self._running = True

        def on_press(key):
            try:
                with self._lock:
                    if not self.enabled:
                        return
                    hks = list(self._hotkeys)
                    listener = self._listener

                canon = listener.canonical(key) if listener and hasattr(listener, "canonical") else key
                for _, hk, _ in hks:
                    hk.press(canon)
            except Exception:
                pass

        def on_release(key):
            try:
                with self._lock:
                    if not self.enabled:
                        return
                    hks = list(self._hotkeys)
                    listener = self._listener

                canon = listener.canonical(key) if listener and hasattr(listener, "canonical") else key
                for _, hk, _ in hks:
                    hk.release(canon)
            except Exception:
                pass

        try:
            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.daemon = True
            self._listener.start()
        except Exception as e:
            print(f"[WARN] Could not initialize global keyboard listener: {e}")

    def stop(self):
        """Stop keyboard listener."""
        self._running = False
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
