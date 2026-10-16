"""Audio synthesis, ADSR envelope shaping, and tone generation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import wavfile

from act_2.scales import note_to_freq

DEFAULT_SAMPLE_RATE = 44100

def apply_envelope(
    samples: np.ndarray,
    attack: float = 0.04,
    decay: float = 0.04,
    sustain_level: float = 0.85,
    release: float = 0.05,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """Apply an ADSR amplitude envelope to prevent clicks and achieve natural acoustics."""
    total_samples = len(samples)
    n_attack = int(attack * sample_rate)
    n_decay = int(decay * sample_rate)
    n_release = int(release * sample_rate)

    if n_attack + n_decay + n_release >= total_samples:
        fade = total_samples // 4
        envelope = np.ones(total_samples, dtype=np.float32)
        if fade > 0:
            envelope[:fade] = np.linspace(0.0, 1.0, fade)
            envelope[-fade:] = np.linspace(1.0, 0.0, fade)
        return samples * envelope

    n_sustain = total_samples - (n_attack + n_decay + n_release)

    env_attack = np.linspace(0.0, 1.0, n_attack, endpoint=False)
    env_decay = np.linspace(1.0, sustain_level, n_decay, endpoint=False)
    env_sustain = np.full(n_sustain, sustain_level)
    env_release = np.linspace(sustain_level, 0.0, n_release)

    envelope = np.concatenate([env_attack, env_decay, env_sustain, env_release]).astype(np.float32)
    return samples * envelope[:total_samples]

def generate_tone(
    freq: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5
) -> np.ndarray:
    """Generate a smoothed sine wave tone for a specified frequency and duration."""
    if freq <= 0.0:
        return np.zeros(int(duration * sample_rate), dtype=np.float32)
    
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    # Fundamental sine wave with subtle second and third harmonic for acoustic warmth
    fundamental = np.sin(2.0 * np.pi * freq * t)
    second_harmonic = 0.20 * np.sin(2.0 * np.pi * 2.0 * freq * t)
    third_harmonic = 0.05 * np.sin(2.0 * np.pi * 3.0 * freq * t)
    raw = (fundamental + second_harmonic + third_harmonic) * amplitude
    
    return apply_envelope(raw.astype(np.float32), sample_rate=sample_rate)

def generate_chord_tone(
    frequencies: list[float],
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.35
) -> np.ndarray:
    """Synthesize a polyphonic chord tone by summing component note frequencies."""
    if not frequencies:
        return np.zeros(int(duration * sample_rate), dtype=np.float32)
    
    n_samples = int(duration * sample_rate)
    chord_wave = np.zeros(n_samples, dtype=np.float32)
    note_amp = amplitude / max(1, len(frequencies))
    
    for f in frequencies:
        chord_wave += generate_tone(f, duration, sample_rate, amplitude=note_amp)
    
    return chord_wave

def export_wav(file_path: str | Path, waveform: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    """Write normalized float waveform to 16 bit PCM WAV file."""
    peak = np.max(np.abs(waveform))
    if peak > 1.0:
        waveform = waveform / peak
    
    pcm_data = (waveform * 32767.0).astype(np.int16)
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(file_path), sample_rate, pcm_data)
