"""Stochastic melody generation with audio synthesis, step printing, and rhythmic duration sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from act_2.markov import MarkovChain
from act_2.scales import CHORDS_D_MAJOR_VOICINGS, note_to_freq
from act_2.tones import DEFAULT_SAMPLE_RATE, export_wav, generate_chord_tone, generate_tone

# Standard rhythmic durations at 120 BPM:
# Quarter note: 0.50 s
# Eighth note: 0.25 s
# Half note: 1.00 s
# Dotted quarter: 0.75 s
RHYTHMIC_DURATIONS = [0.25, 0.50, 0.75, 1.00]
RHYTHMIC_PROBABILITIES = [0.25, 0.50, 0.15, 0.10]

@dataclass
class TransitionEvent:
    step: int
    prev_state: str
    curr_state: str
    frequency_hz: float
    duration_s: float
    cumulative_time_s: float

class MelodyGenerator:
    """Simulates Markovian note sequences with acoustic tone generation and transition tracking."""

    def __init__(
        self,
        chain: MarkovChain,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        rhythmic_durations: list[float] | None = None,
        duration_probabilities: list[float] | None = None
    ):
        self.chain = chain
        self.sample_rate = sample_rate
        self.rhythmic_durations = rhythmic_durations or RHYTHMIC_DURATIONS
        self.duration_probabilities = duration_probabilities or RHYTHMIC_PROBABILITIES

    def generate_melody(
        self,
        initial_note: str,
        n_transitions: int,
        fixed_duration: float | None = None,
        seed: int | None = None,
        verbose: bool = True
    ) -> tuple[np.ndarray, list[TransitionEvent], pd.DataFrame]:
        """Execute n transitions, print each step, and synthesize the full acoustic waveform."""
        rng = np.random.default_rng(seed)
        events: list[TransitionEvent] = []
        audio_segments: list[np.ndarray] = []
        
        curr_note = initial_note
        cum_time = 0.0
        
        # Initial starting tone
        initial_duration = fixed_duration if fixed_duration is not None else float(
            rng.choice(self.rhythmic_durations, p=self.duration_probabilities)
        )
        initial_freq = note_to_freq(curr_note)
        events.append(TransitionEvent(
            step=0,
            prev_state="Inicio",
            curr_state=curr_note,
            frequency_hz=round(initial_freq, 2),
            duration_s=round(initial_duration, 2),
            cumulative_time_s=0.0
        ))
        audio_segments.append(generate_tone(initial_freq, initial_duration, self.sample_rate))
        cum_time += initial_duration
        
        if verbose:
            print(f"Paso 0 (Inicio): Nota {curr_note} | Frecuencia: {initial_freq:.2f} Hz | Duración: {initial_duration:.2f} s")
            
        for step in range(1, n_transitions + 1):
            next_note = self.chain.next_state(curr_note, rng=rng)
            freq = note_to_freq(next_note)
            
            if fixed_duration is not None:
                duration = fixed_duration
            else:
                duration = float(rng.choice(self.rhythmic_durations, p=self.duration_probabilities))
                
            events.append(TransitionEvent(
                step=step,
                prev_state=curr_note,
                curr_state=next_note,
                frequency_hz=round(freq, 2),
                duration_s=round(duration, 2),
                cumulative_time_s=round(cum_time, 2)
            ))
            
            if verbose:
                print(f"Paso {step:02d}: Transición {curr_note} -> {next_note} | Frecuencia: {freq:.2f} Hz | Duración: {duration:.2f} s")
                
            audio_segments.append(generate_tone(freq, duration, self.sample_rate))
            cum_time += duration
            curr_note = next_note

        full_audio = np.concatenate(audio_segments) if audio_segments else np.zeros(0, dtype=np.float32)
        
        df_events = pd.DataFrame([
            {
                "Paso": e.step,
                "Estado anterior": e.prev_state,
                "Nota / Acorde": e.curr_state,
                "Frecuencia (Hz)": e.frequency_hz,
                "Duración (s)": e.duration_s,
                "Tiempo acumulado (s)": e.cumulative_time_s,
            }
            for e in events
        ])
        
        return full_audio, events, df_events

    def generate_harmonized_track(
        self,
        melody_chain: MarkovChain,
        chord_chain: MarkovChain,
        initial_note: str,
        initial_chord: str,
        bars: int = 8,
        beats_per_bar: int = 4,
        tempo_bpm: float = 120.0,
        seed: int | None = None
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Generate a complete multitrack composition with melody over Markov chord harmony."""
        rng = np.random.default_rng(seed)
        sec_per_beat = 60.0 / tempo_bpm
        bar_duration = sec_per_beat * beats_per_bar
        
        # 1. Generate chord progression
        chords = chord_chain.simulate(initial_chord, bars - 1, seed=seed)
        chord_audio_list = []
        for ch in chords:
            voicings = CHORDS_D_MAJOR_VOICINGS.get(ch, ["D3", "F#3", "A3"])
            freqs = [note_to_freq(n) for n in voicings]
            chord_wave = generate_chord_tone(freqs, bar_duration, self.sample_rate, amplitude=0.25)
            chord_audio_list.append(chord_wave)
        full_chords = np.concatenate(chord_audio_list)
        total_time = len(full_chords) / self.sample_rate
        
        # 2. Generate melody filling total_time
        melody_segments = []
        curr_note = initial_note
        cum_melody_time = 0.0
        events = []
        step = 0
        
        while cum_melody_time < total_time:
            dur = float(rng.choice(self.rhythmic_durations, p=self.duration_probabilities))
            if cum_melody_time + dur > total_time:
                dur = total_time - cum_melody_time
                if dur <= 0.01:
                    break
            freq = note_to_freq(curr_note)
            melody_segments.append(generate_tone(freq, dur, self.sample_rate, amplitude=0.45))
            events.append({
                "Paso": step,
                "Nota": curr_note,
                "Frecuencia (Hz)": round(freq, 2),
                "Duración (s)": round(dur, 2),
                "Inicio (s)": round(cum_melody_time, 2)
            })
            cum_melody_time += dur
            step += 1
            curr_note = melody_chain.next_state(curr_note, rng=rng)
            
        full_melody = np.concatenate(melody_segments)
        # Pad or trim to match chord length exactly
        if len(full_melody) < len(full_chords):
            full_melody = np.pad(full_melody, (0, len(full_chords) - len(full_melody)))
        else:
            full_melody = full_melody[:len(full_chords)]
            
        mix = (full_chords + full_melody) * 0.8
        return mix, pd.DataFrame(events)
