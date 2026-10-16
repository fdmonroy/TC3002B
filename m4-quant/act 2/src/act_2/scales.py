"""Music theory scales, note mappings, harmonic matrices, and equal temperament frequency calculations."""

from __future__ import annotations

import numpy as np

# Reference pitch: A4 = 440 Hz at MIDI note 69
A4_MIDI = 69
A4_FREQ = 440.0

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def note_to_midi(note: str) -> int:
    """Convert note name like 'C4', 'F#4', 'A3' to MIDI number."""
    if len(note) < 2:
        raise ValueError(f"Invalid note format: {note}")
    if note[1] == "#":
        name = note[:2].upper()
        octave = int(note[2:])
    else:
        name = note[:1].upper()
        octave = int(note[1:])
    semitone = NOTE_NAMES.index(name)
    return (octave + 1) * 12 + semitone

def midi_to_freq(midi_num: int) -> float:
    """Calculate frequency in Hz using 12 tone equal temperament."""
    return float(A4_FREQ * (2.0 ** ((midi_num - A4_MIDI) / 12.0)))

def note_to_freq(note: str) -> float:
    """Return frequency in Hz for a note name string."""
    return midi_to_freq(note_to_midi(note))

# D Major diatonic scale (as in Netlify example: boring-poitras-2980e0.netlify.app)
D_MAJOR_NOTES_OCT4 = ["D4", "E4", "F#4", "G4", "A4", "B4", "C#5"]

# Scale degrees for harmonic functions:
# I: Tonic (D4)
# ii: Supertonic (E4)
# iii: Mediant (F#4)
# IV: Subdominant (G4)
# V: Dominant (A4)
# vi: Submediant (B4)
# vii°: Leading tone (C#5)

HARMONIC_TRANSITIONS_D_MAJOR = np.array([
    # D4    E4    F#4   G4    A4    B4    C#5
    [0.10, 0.20, 0.15, 0.20, 0.20, 0.10, 0.05],  # D4 (I - Tonic)
    [0.30, 0.05, 0.20, 0.10, 0.25, 0.05, 0.05],  # E4 (ii - Supertonic)
    [0.20, 0.20, 0.05, 0.30, 0.15, 0.05, 0.05],  # F#4 (iii - Mediant)
    [0.15, 0.10, 0.25, 0.05, 0.35, 0.05, 0.05],  # G4 (IV - Subdominant)
    [0.50, 0.10, 0.10, 0.05, 0.05, 0.15, 0.05],  # A4 (V - Dominant)
    [0.20, 0.15, 0.15, 0.25, 0.20, 0.05, 0.00],  # B4 (vi - Submediant)
    [0.70, 0.10, 0.05, 0.00, 0.10, 0.05, 0.00],  # C#5 (vii° - Leading tone)
], dtype=np.float64)

# Chord progression states (I, IV, V, vi in D major)
CHORD_STATES = ["D", "G", "A", "Bm"]

CHORDS_D_MAJOR_VOICINGS = {
    "D": ["D3", "F#3", "A3"],       # I (Tonic)
    "G": ["G3", "B3", "D4"],        # IV (Subdominant)
    "A": ["A3", "C#4", "E4"],       # V (Dominant)
    "Bm": ["B3", "D4", "F#4"],      # vi (Submediant)
}

CHORD_TRANSITION_MATRIX = np.array([
    # D     G     A     Bm
    [0.05, 0.35, 0.35, 0.25],  # D (I)
    [0.30, 0.05, 0.50, 0.15],  # G (IV)
    [0.65, 0.05, 0.05, 0.25],  # A (V)
    [0.20, 0.45, 0.30, 0.05],  # Bm (vi)
], dtype=np.float64)
