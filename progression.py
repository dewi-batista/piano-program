from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from config import DEFAULT_VELOCITY


def midi_note_name(note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = note // 12 - 1
    return f"{names[note % 12]}{octave}"


def detect_chord(notes: List[int]) -> Optional[str]:
    """Return a basic chord name (triads/sevenths/sus) from MIDI note numbers."""
    if not notes:
        return None
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pcs = sorted({n % 12 for n in notes})
    patterns = [
        ("maj7", {4, 7, 11}),
        ("7", {4, 7, 10}),
        ("min7", {3, 7, 10}),
        ("m7b5", {3, 6, 10}),
        ("dim7", {3, 6, 9}),
        ("maj", {4, 7}),
        ("min", {3, 7}),
        ("dim", {3, 6}),
        ("aug", {4, 8}),
        ("sus2", {2, 7}),
        ("sus4", {5, 7}),
    ]
    for root in pcs:
        intervals = sorted({(p - root) % 12 for p in pcs if p != root})
        interval_set = set(intervals)
        for name, target in patterns:
            if target.issubset(interval_set):
                return f"{names[root]}{name}"
    return None


def parse_note_value(raw: object) -> int:
    """Parse a MIDI note number or a note-name string like C#4 / Db3."""
    if isinstance(raw, int):
        if 0 <= raw <= 127:
            return raw
        raise ValueError(f"MIDI note out of range: {raw}")
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            raise ValueError("Empty note string")
        if s.lstrip("-+").isdigit():
            val = int(s)
            if 0 <= val <= 127:
                return val
            raise ValueError(f"MIDI note out of range: {s}")
        m = re.match(r"^([A-Ga-g])([#b]?)(-?\d+)$", s)
        if not m:
            raise ValueError(f"Could not parse note: {s}")
        letter, accidental, octave_str = m.groups()
        base = {
            "c": 0,
            "d": 2,
            "e": 4,
            "f": 5,
            "g": 7,
            "a": 9,
            "b": 11,
        }[letter.lower()]
        if accidental == "#":
            base += 1
        elif accidental == "b":
            base -= 1
        octave = int(octave_str)
        val = 12 * (octave + 1) + base
        if not (0 <= val <= 127):
            raise ValueError(f"Note out of MIDI range: {s}")
        return val
    raise ValueError(f"Unsupported note value: {raw}")


def parse_chord_entry(entry: object) -> Tuple[List[int], List[int], Optional[str], bool, bool, Optional[int]]:
    """Return (notes, velocities, label, velocities_given, roll, roll_ms) for a chord entry."""
    notes_raw = None
    label = None
    if isinstance(entry, dict):
        if "notes" in entry:
            notes_raw = entry["notes"]
        elif "pitches" in entry:
            notes_raw = entry["pitches"]
        label = entry.get("label") or entry.get("name")
        velocities_raw = entry.get("velocities")
    else:
        notes_raw = entry
        velocities_raw = None

    if notes_raw is None:
        raise ValueError("Chord entry missing notes")
    if not isinstance(notes_raw, (list, tuple)) or not notes_raw:
        raise ValueError("Chord notes must be a non-empty list")

    notes: List[int] = []
    velocities: List[int] = []
    velocities_given = False
    roll = False
    roll_ms: Optional[int] = None
    if isinstance(entry, dict):
        roll = bool(entry.get("roll", False))
        if "roll_ms" in entry:
            try:
                roll_ms_val = int(entry["roll_ms"])
                if roll_ms_val > 0:
                    roll_ms = roll_ms_val
            except Exception:
                roll_ms = None

    if all(isinstance(n, dict) for n in notes_raw):
        for obj in notes_raw:
            if "note" not in obj:
                raise ValueError("Note object must contain a 'note' field")
            notes.append(parse_note_value(obj["note"]))
            vel_val = obj.get("velocity", DEFAULT_VELOCITY)
            if not (0 <= int(vel_val) <= 127):
                raise ValueError(f"Velocity out of range: {vel_val}")
            velocities.append(int(vel_val))
            if "velocity" in obj:
                velocities_given = True
    else:
        notes = [parse_note_value(n) for n in notes_raw]
        if velocities_raw is None or (isinstance(velocities_raw, (list, tuple)) and len(velocities_raw) == 0):
            velocities = [min(127, 60 + i * 5) for i in range(len(notes))]
        else:
            if not isinstance(velocities_raw, (list, tuple)):
                velocities_raw = []
            velocities = []
            for i, _ in enumerate(notes):
                if i < len(velocities_raw):
                    vel_val = velocities_raw[i]
                    velocities_given = True
                else:
                    vel_val = DEFAULT_VELOCITY
                if not (0 <= int(vel_val) <= 127):
                    raise ValueError(f"Velocity out of range: {vel_val}")
                velocities.append(int(vel_val))

    return notes, velocities, label, velocities_given, roll, roll_ms


@dataclass
class TargetChord:
    notes: List[int]
    velocities: List[int]
    label: str
    pcs: frozenset[int]
    velocities_given: bool = True
    roll: bool = False
    roll_ms: Optional[int] = None


def load_progression_file(path: str) -> List[TargetChord]:
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            if yaml is None:
                raise RuntimeError("pyyaml is required to load YAML progression files. Install with `pip install pyyaml`.")
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    # New format: top-level chords + velocities arrays.
    if isinstance(data, dict) and "chords" in data and "velocities" in data:
        chords_raw = data["chords"]
        vels_raw = data.get("velocities")
        if not isinstance(chords_raw, list):
            raise ValueError("chords must be a list")
        if vels_raw is not None and not isinstance(vels_raw, list):
            vels_raw = None
        targets: List[TargetChord] = []
        for idx, notes_raw in enumerate(chords_raw):
            vel_list = vels_raw[idx] if vels_raw and idx < len(vels_raw) else None
            if not isinstance(notes_raw, list):
                raise ValueError("Each chords entry must be a list of notes")
            notes = [parse_note_value(n) for n in notes_raw]
            velocities = []
            vel_given = isinstance(vel_list, list) and len(vel_list) > 0
            if vel_list is None or not isinstance(vel_list, list):
                vel_list = []
            for i, _ in enumerate(notes):
                if i < len(vel_list):
                    vel = vel_list[i]
                else:
                    vel = min(127, 60 + i * 5)
                if not (0 <= int(vel) <= 127):
                    raise ValueError(f"Velocity out of range: {vel}")
                velocities.append(int(vel))
            pcs = frozenset(n % 12 for n in notes)
            label = detect_chord(notes) or ",".join(midi_note_name(n) for n in sorted(notes))
            targets.append(
                TargetChord(
                    notes=notes,
                    velocities=velocities,
                    label=label,
                    pcs=pcs,
                    velocities_given=vel_given,
                    roll=False,
                    roll_ms=None,
                )
            )
        return targets

    progression_raw = data.get("progression") if isinstance(data, dict) else data
    if not isinstance(progression_raw, list) or not progression_raw:
        raise ValueError("Progression file must contain a non-empty list of chords")

    targets: List[TargetChord] = []
    for idx, entry in enumerate(progression_raw):
        notes, velocities, label, vel_given, roll, roll_ms = parse_chord_entry(entry)
        pcs = frozenset(n % 12 for n in notes)
        auto_label = detect_chord(notes) or ",".join(midi_note_name(n) for n in sorted(notes))
        final_label = label or auto_label
        targets.append(
            TargetChord(
                notes=notes,
                velocities=velocities,
                label=final_label,
                pcs=pcs,
                velocities_given=vel_given,
                roll=roll,
                roll_ms=roll_ms,
            )
        )
    return targets
