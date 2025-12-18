"""
Static chord/velocity practice visual (no scrolling).

Reads a JSON progression with target notes and velocities and shows fixed
columns (one per chord). Target strokes are always visible; your last played
chord is drawn on the active column. Hit detection advances through the list.

Progression file formats:

1) Top-level chords + velocities (your example):
{
  "chords": [
    ["C4", "E4", "G4", "C5"],
    ["E4", "G4", "C5", "E5"],
    ["G4", "C5", "E5", "G5"]
  ],
  "velocities": [
    [60, 70, 80, 90],
    [60, 70, 80, 90],
    [60, 70, 80, 90]
  ]
}

2) Progression list with velocities on each chord:
{
  "progression": [
    {"label": "Cmaj", "notes": ["C4", "E4", "G4"], "velocities": [80, 70, 90]},
    {"notes": [67, 71, 74], "velocities": [85, 80, 78]}
  ]
}

3) Notes as objects (each with its own velocity):
{
  "progression": [
    {"label": "G7", "notes": [
      {"note": "G3", "velocity": 82},
      {"note": "B3", "velocity": 76},
      {"note": "D4", "velocity": 78},
      {"note": "F4", "velocity": 74}
    ]}
  ]
}

Notes may be MIDI numbers or note names like C#4 / Db3 (middle C is C4 = 60).
Velocities must be 0-127; if omitted they default to 64.

Dependencies:
  pip install mido python-rtmidi pyqtgraph PySide6

Usage:
  python main.py --progression-file progression.yaml \
      [--range LO HI] [--require-exact] [port substring | midi_file.mid]

- If a .mid file path is given, plays that file into the visual.
- Otherwise opens the first MIDI input port containing the substring (or the first port).
- Displays expected chord + velocities vs. what you just played.
- Matching ignores octaves/inversions; --require-exact fails on extra pitch classes.
- Velocity matching uses a tolerance window (see VELOCITY_TOLERANCE).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

import mido
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


# ---- Config ----
CHORD_THRESHOLD = 0.1  # seconds
CHORD_HISTORY = 8
CHORD_SPACING = 1.8
LINE_HALF_WIDTH = 0.32
OVERLAP_STEP = 2.0
MIN_GAP = 1.2
DEFAULT_VELOCITY = 80
VISIBLE_CHORDS = 6
VELOCITY_TOLERANCE = 8  # velocity +/- allowed for a hit
BANNER_Y = 131.0
ROLL_WINDOW_MS_DEFAULT = 150

DYNAMIC_LEVELS = {
    "ppp": 15,
    "pp": 25,
    "p": 40,
    "mp": 55,
    "mf": 70,
    "f": 85,
    "ff": 100,
    "fff": 127,
}


# ---- Helpers ----
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


@dataclass
class ProgressStatus:
    next_label: str
    index: int
    total: int
    last_label: Optional[str]
    last_ok: Optional[bool]
    done: bool
    current_target: Optional[TargetChord]
    upcoming_labels: List[str]


class ProgressionChecker:
    def __init__(self, targets: List[TargetChord], require_exact: bool) -> None:
        self.targets = targets
        self.require_exact = require_exact
        self.index = 0
        self._last_processed_start: Optional[float] = None
        self.last_label: Optional[str] = None
        self.last_ok: Optional[bool] = None
        self.roll_notes: Optional[List[Tuple[int, int]]] = None
        self.roll_start: Optional[float] = None
        self.roll_last_start: Optional[float] = None
        self.finalized_chords: List[Tuple[float, List[Tuple[int, int]]]] = []
        self.processed_starts: set[float] = set()

    def _matches(self, actual_notes: List[Tuple[int, int]], target: TargetChord) -> bool:
        """Return True if pitches match the target (velocity is not required for a hit)."""
        used: List[bool] = [False] * len(actual_notes)
        for t_note, t_vel in zip(target.notes, target.velocities):
            found_idx = None
            for i, (p, v) in enumerate(actual_notes):
                if used[i]:
                    continue
                if p % 12 != t_note % 12:
                    continue
                found_idx = i
                break
            if found_idx is None:
                return False
            used[found_idx] = True

        if self.require_exact:
            remaining_pcs = {p % 12 for i, (p, _) in enumerate(actual_notes) if not used[i]}
            if remaining_pcs:
                return False
        return True

    def _reset_roll_state(self) -> None:
        self.roll_notes = None
        self.roll_start = None
        self.roll_last_start = None
        self.processed_starts.clear()
        self._processed_roll_starts = set()

    def _finalize_roll(self, target: TargetChord) -> None:
        if self.roll_notes is None or self.roll_start is None:
            return
        combined_notes = sorted(self.roll_notes, key=lambda x: x[0])
        label = detect_chord([n for n, _ in combined_notes]) or ",".join(
            midi_note_name(n) for n, _ in sorted(combined_notes, key=lambda t: t[0])
        )
        self.last_label = label
        hit = self._matches(combined_notes, target)
        self.last_ok = hit
        if self.roll_last_start is not None:
            self._last_processed_start = self.roll_last_start
            self.processed_starts.add(self.roll_last_start)
        # Advance progression regardless, but remember accuracy.
        self.index += 1
        self.finalized_chords.append((self.roll_start, combined_notes))
        self._reset_roll_state()

    def update(self, chords: List[Tuple[float, List[Tuple[int, int]]]]) -> ProgressStatus:
        self.finalized_chords.clear()
        now = time.monotonic()
        for start, notes in sorted(chords, key=lambda x: x[0]):
            if start in self.processed_starts:
                continue
            if self.index >= len(self.targets):
                continue
            target = self.targets[self.index]

            if not target.roll and self._last_processed_start is not None and start <= self._last_processed_start:
                continue

            # If target is non-rolled, finalize any lingering roll state.
            if not target.roll and self.roll_start is not None:
                prev_target = self.targets[self.index] if self.index < len(self.targets) else None
                if prev_target and prev_target.roll:
                    self._finalize_roll(prev_target)
            if not target.roll:
                label = detect_chord([n for n, _ in notes]) or ",".join(
                    midi_note_name(n) for n, _ in sorted(notes, key=lambda t: t[0])
                )
                self.last_label = label
                hit = self._matches(notes, target)
                self.last_ok = hit
                self._last_processed_start = start
                self.processed_starts.add(start)
                # Advance progression regardless of hit/miss to keep flow moving.
                self.index += 1
                self.finalized_chords.append((start, notes))
                continue

            # Rolled chord handling
            window_sec = (target.roll_ms or ROLL_WINDOW_MS_DEFAULT) / 1000.0
            # If we are starting or extending the current roll
            if self.roll_start is None or (start - self.roll_start) > window_sec:
                # If we had an active roll that timed out before this note, finalize it first.
                if self.roll_start is not None:
                    self._finalize_roll(target)
                    if self.index >= len(self.targets):
                        continue
                    target = self.targets[self.index]
                    if not target.roll:
                        # Handle this chord as a regular chord.
                        label = detect_chord([n for n, _ in notes]) or ",".join(
                            midi_note_name(n) for n, _ in sorted(notes, key=lambda t: t[0])
                        )
                        self.last_label = label
                        hit = self._matches(notes, target)
                        self.last_ok = hit
                        self._last_processed_start = start
                        if hit:
                            self.index += 1
                        self.finalized_chords.append((start, notes))
                        continue
                # Start a new roll window.
                self.roll_start = start
                self.roll_last_start = start
                self.roll_notes = list(notes)
                self.processed_starts.add(start)
                appended = list(notes)
            else:
                # Still within the roll window: accumulate.
                self.roll_notes.extend(notes)
                self.roll_last_start = start
                self.processed_starts.add(start)
                appended = list(notes)

            # Check completion; if not complete, defer processing further chords until next tick.
            if self.index < len(self.targets) and target.roll and self.roll_notes is not None:
                needed = Counter(target.notes)
                have = Counter(n for n, _ in self.roll_notes)
                complete = all(have[n] >= needed[n] for n in needed)
                last_note = target.notes[-1] if target.notes else None
                last_hit = last_note is not None and any(p == last_note for p, _ in appended)
                if complete or last_hit:
                    self._finalize_roll(target)
                else:
                    break

        # Finalize pending roll if window elapsed
        if self.index < len(self.targets):
            target = self.targets[self.index]
            if target.roll and self.roll_start is not None:
                window_sec = (target.roll_ms or ROLL_WINDOW_MS_DEFAULT) / 1000.0
                if now - self.roll_start >= window_sec:
                    self._finalize_roll(target)
        elif self.roll_start is not None:
            # Safety: clear stale roll state if we moved past a rolled chord.
            self._reset_roll_state()

        done = self.index >= len(self.targets)
        next_label = "Done" if done else self.targets[self.index].label
        current_target = None if done else self.targets[self.index]
        upcoming = []
        if not done:
            for nxt in self.targets[self.index + 1 : self.index + 4]:
                upcoming.append(nxt.label)
        return ProgressStatus(
            next_label=next_label,
            index=self.index,
            total=len(self.targets),
            last_label=self.last_label,
            last_ok=self.last_ok,
            done=done,
            current_target=current_target,
            upcoming_labels=upcoming,
        )


# ---- MIDI stream + note tracking ----
class ChordStream:
    def __init__(self) -> None:
        self.window: deque[Tuple[float, List[Tuple[int, int]]]] = deque(maxlen=CHORD_HISTORY)
        self._lock = threading.Lock()
        self._current_notes: List[Tuple[int, int]] = []
        self._current_start: Optional[float] = None

    def _finalize_current(self) -> None:
        if not self._current_notes or self._current_start is None:
            return
        notes_sorted = sorted(self._current_notes, key=lambda x: x[0])
        self.window.append((self._current_start, notes_sorted))
        self._current_notes = []
        self._current_start = None

    def ingest_note_on(self, note: int, velocity: int, t: float) -> None:
        with self._lock:
            if not self._current_notes:
                self._current_start = t
                self._current_notes = [(note, velocity)]
                return
            if t - (self._current_start or t) <= CHORD_THRESHOLD:
                self._current_notes.append((note, velocity))
            else:
                self._finalize_current()
                self._current_start = t
                self._current_notes = [(note, velocity)]

    def flush_stale(self, now: float) -> None:
        with self._lock:
            if self._current_start is not None and now - self._current_start > CHORD_THRESHOLD:
                self._finalize_current()

    def snapshot(self) -> List[Tuple[float, List[Tuple[int, int]]]]:
        with self._lock:
            return list(self.window)

    def reset(self) -> None:
        with self._lock:
            self.window.clear()
            self._current_notes = []
            self._current_start = None


class ActiveNotes:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: set[int] = set()

    def note_on(self, note: int) -> None:
        with self._lock:
            self._pressed.add(note)

    def note_off(self, note: int) -> None:
        with self._lock:
            self._pressed.discard(note)

    def current_pressed(self) -> List[int]:
        with self._lock:
            return sorted(self._pressed)

    def reset(self) -> None:
        with self._lock:
            self._pressed.clear()


# ---- Theme + Visual ----
class Theme:
    def __init__(self, dark: bool = True) -> None:
        self.dark = dark
        self.bg = "#0f1118"
        self.band = "#171b24"
        self.grid = "#3c4350"
        self.axis = "#dfe3ec"
        self.label = "#e8ebf2"
        self.highlight_fill = (42, 110, 209, 90)
        self.highlight_pen = pg.mkPen("#5fa8ff", width=1.0)
        self.palette = [
            "#8ec7ff",
            "#f6b26b",
            "#7bdcb5",
            "#e06666",
            "#b4a7d6",
            "#d5e07b",
            "#6fa8dc",
            "#f9cb9c",
            "#a4c2f4",
            "#f4c2d5",
        ]
        # Target strokes use a mint hue distinct from the live palette.
        self.target_stroke = "#8bf0d4"
        self.target_label = "#d5ffef"

    def color_for_index(self, idx: int) -> str:
        return self.palette[idx % len(self.palette)]


class Visual(QtWidgets.QWidget):
    def __init__(
        self,
        theme: Theme,
        highlight_range: Optional[Tuple[float, float]],
        active_notes: ActiveNotes,
        targets: List[TargetChord],
        highlight_visible: bool = False,
        dynamics_visible: bool = True,
        free_play: bool = False,
    ) -> None:
        super().__init__()
        self.theme = theme
        self.highlight_range = highlight_range
        self.highlight_visible = highlight_visible
        self.dynamics_visible = dynamics_visible
        self.active_notes = active_notes
        self.targets = targets
        self.free_play = free_play
        self.start_time = time.monotonic()
        self.dynamic_pixmaps = self._load_dynamic_pixmaps()
        self._last_seen_start: float = -1.0
        self.scale = self._compute_scale(len(self.targets))
        self.captured_chords: List[Optional[List[Tuple[int, int]]]] = [None for _ in targets]
        self.history: List[Tuple[int, Optional[List[Tuple[int, int]]], float]] = []
        self.free_history: List[List[Tuple[int, int]]] = [] if free_play else []
        self.window_start = 0
        self.window_size = min(VISIBLE_CHORDS, max(1, len(self.targets)))
        self.follow_latest = True
        self.score_items: List[QtWidgets.QGraphicsItem] = []
        self.score_top: List[Optional[QtWidgets.QGraphicsPathItem]] = [None for _ in targets]
        self.score_bottom: List[Optional[QtWidgets.QGraphicsPathItem]] = [None for _ in targets]
        self.target_items: List[QtWidgets.QGraphicsItem] = []
        self.target_label_items: List[Tuple[int, List[Tuple[int, pg.TextItem]]]] = []
        self._suppress_slider = False

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.plot = pg.PlotWidget(background=self.theme.bg)
        self.plot.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.plot.setStyleSheet("border: 0px;")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.showGrid(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.setClipToView(True)
        self.plot.getPlotItem().hideButtons()
        self.plot.getPlotItem().getViewBox().setBorder(None)
        self.plot.getPlotItem().hideAxis("left")
        self.plot.getPlotItem().hideAxis("bottom")
        self.plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
        self.plot.getPlotItem().setDefaultPadding(0)
        layout.addWidget(self.plot)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, max(0, len(self.targets) - self.window_size))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setVisible(len(self.targets) > self.window_size)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.setFixedHeight(26)
        self.slider.setStyleSheet(
            """
            QSlider {
                margin: 8px 12px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #1c2230;
                border: 1px solid #2e3545;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6fa8dc;
                border: 1px solid #9bc7f3;
                width: 18px;
                height: 18px;
                margin: -7px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #8ec7ff;
                border: 1px solid #c2e0ff;
            }
            QSlider::sub-page:horizontal {
                background: #2f3a4d;
                border-radius: 4px;
            }
            QSlider::add-page:horizontal {
                background: #1a1f2b;
                border-radius: 4px;
            }
            """
        )
        layout.addWidget(self.slider)

        self.plot.setYRange(0, 134, padding=0)
        chord_count = max(1, self.window_size)
        x_max = CHORD_SPACING * (chord_count - 1) + CHORD_SPACING * 0.55
        x_left = -CHORD_SPACING * 0.55
        self.plot.setXRange(x_left, x_max, padding=0)

        self.static_items: List[pg.GraphicsObject] = []
        self.note_items: List[pg.GraphicsObject] = []
        self.banner_item: Optional[pg.TextItem] = None
        self.banner_chord_item: Optional[pg.TextItem] = None
        self.score_banner: Optional[pg.TextItem] = None
        self.last_banner_signature: Optional[Tuple[Tuple[int, int], Optional[str]]] = None
        self.dynamic_items: List[QtWidgets.QGraphicsItem] = []
        self.dynamics_items: List[QtWidgets.QGraphicsItem] = []
        self.highlight_item: Optional[QtWidgets.QGraphicsRectItem] = None

        self._init_static()
        self._sync_slider()

    def _populate_free_window(self) -> None:
        """Rebuild the captured window from free-play history."""
        if not self.free_play:
            return
        start_idx = max(0, self.window_start)
        end_idx = start_idx + self.window_size
        window_slice = self.free_history[start_idx:end_idx]
        self.captured_chords = [None for _ in range(self.window_size)]
        for i, chord in enumerate(window_slice):
            if i < self.window_size:
                self.captured_chords[i] = chord

    def _load_dynamic_pixmaps(self) -> dict:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(base_dir, "assets")
        pixmaps = {}
        if os.path.isdir(assets_dir):
            for fname in os.listdir(assets_dir):
                if fname.lower().endswith(".png"):
                    stem = os.path.splitext(fname)[0].lower()
                    path = os.path.join(assets_dir, fname)
                    pix = QtGui.QPixmap(path)
                    if not pix.isNull():
                        pixmaps[stem] = pix
        for lab in DYNAMIC_LEVELS.keys():
            lower = lab.lower()
            path = os.path.join(assets_dir, f"{lab}.png")
            if os.path.isfile(path):
                pix = QtGui.QPixmap(path)
                if not pix.isNull():
                    pixmaps[lower] = pix
        return pixmaps

    def _compute_scale(self, chord_count: int) -> float:
        if chord_count <= 4:
            return 1.25
        if chord_count <= 8:
            return 1.12
        if chord_count <= 12:
            return 0.95
        return 0.85

    def _max_window_start(self) -> int:
        if self.free_play:
            return max(0, len(self.free_history) - self.window_size)
        return max(0, len(self.targets) - self.window_size)

    def _on_slider_changed(self, value: int) -> None:
        if self._suppress_slider:
            return
        self.window_start = max(0, min(value, self._max_window_start()))
        self.follow_latest = False
        self.update_view(None, [])

    def _sync_slider(self) -> None:
        max_start = self._max_window_start()
        self.slider.setMaximum(max_start)
        self.slider.setVisible(max_start > 0)
        self._suppress_slider = True
        self.slider.blockSignals(True)
        self.slider.setValue(max(0, min(self.window_start, max_start)))
        self.slider.blockSignals(False)
        self._suppress_slider = False

    def toggle_highlight(self) -> None:
        if not self.highlight_range:
            return
        # Create on demand in case it was never built (e.g., after a future refresh).
        if self.highlight_item is None:
            chord_count = max(1, len(self.targets))
            x_max = CHORD_SPACING * (chord_count - 1) + 0.6
            lo, hi = self.highlight_range
            rect = QtWidgets.QGraphicsRectItem(-1, lo, x_max + 1.0, hi - lo)
            rect.setBrush(pg.mkBrush(self.theme.highlight_fill))
            rect.setPen(self.theme.highlight_pen)
            rect.setZValue(-1.5)
            rect.setVisible(False)
            self.plot.addItem(rect)
            self.static_items.append(rect)
            self.highlight_item = rect
        self.highlight_visible = not self.highlight_visible
        if self.highlight_item:
            self.highlight_item.setVisible(self.highlight_visible)

    def toggle_dynamics(self) -> None:
        self.dynamics_visible = not self.dynamics_visible
        for item in self.dynamics_items:
            item.setVisible(self.dynamics_visible)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        """Allow trackpad/mouse wheel to scroll the chord window."""
        if not self.free_play and len(self.targets) <= self.window_size:
            return super().wheelEvent(event)
        if self.free_play and len(self.free_history) <= self.window_size:
            return super().wheelEvent(event)
        delta = event.angleDelta()
        steps = delta.x() if abs(delta.x()) > abs(delta.y()) else delta.y()
        if steps == 0:
            return super().wheelEvent(event)
        direction = -1 if steps > 0 else 1  # natural scroll: up/left moves window left
        self.follow_latest = False
        self.window_start = max(0, min(self.window_start + direction, self._max_window_start()))
        self._sync_slider()
        self.update_view(None, [])
        event.accept()

    def _match_components(self, played: Optional[List[Tuple[int, int]]], target: TargetChord) -> Tuple[float, Optional[float]]:
        """Return (pitch_score, velocity_score) where velocity_score may be None if not applicable."""
        if not played:
            return 0.0, None if not target.velocities_given else 0.0
        available = list(enumerate(played))
        matched = 0
        vel_score_sum = 0.0
        denom = max(len(target.notes), len(played), 1)
        for t_note, t_vel in zip(target.notes, target.velocities):
            best_pos = None
            best_cost = None
            for pos, (_idx_played, (p, v)) in enumerate(available):
                if p != t_note:
                    continue
                cost = abs(v - t_vel)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            if best_pos is not None:
                matched += 1
                _, (_, v) = available.pop(best_pos)
                vel_score_sum += max(0.0, 1.0 - min(1.0, abs(v - t_vel) / 50.0))
        extras = max(len(played) - len(target.notes), 0)
        missing = max(len(target.notes) - matched, 0)
        if target.notes:
            pitch_denom = len(target.notes) + extras * 3.0 + missing * 2.0
            pitch_score = matched / pitch_denom
        else:
            pitch_score = 0.0
        vel_score = vel_score_sum / denom if target.velocities_given else None
        return pitch_score, vel_score

    def _color_from_score(self, score: float) -> QtGui.QColor:
        score = max(0.0, min(1.0, score))
        if score < 0.5:
            t = score / 0.5
            r, g, b = 231, int(76 + (182 - 76) * t), 60
        else:
            t = (score - 0.5) / 0.5
            r, g, b = int(231 + (39 - 231) * t), int(182 + (174 - 182) * t), int(60 + (96 - 60) * t)
        return QtGui.QColor(r, g, b)

    def _init_static(self) -> None:
        chord_count = max(1, len(self.targets))
        x_max = CHORD_SPACING * (chord_count - 1) + 0.6
        for i in range(chord_count):
            if i % 2 == 0:
                center_x = i * CHORD_SPACING
                rect = QtWidgets.QGraphicsRectItem(center_x - CHORD_SPACING * 0.5, 0, CHORD_SPACING, 128)
                rect.setBrush(pg.mkBrush(self.theme.band))
                rect.setPen(pg.mkPen(None))
                rect.setZValue(-2)
                self.plot.addItem(rect)
                self.static_items.append(rect)

        if self.highlight_range:
            lo, hi = self.highlight_range
            rect = QtWidgets.QGraphicsRectItem(-1, lo, x_max + 1.0, hi - lo)
            rect.setBrush(pg.mkBrush(self.theme.highlight_fill))
            rect.setPen(self.theme.highlight_pen)
            rect.setZValue(-1.5)
            rect.setVisible(self.highlight_visible)
            self.plot.addItem(rect)
            self.static_items.append(rect)
            self.highlight_item = rect

        dyn_vals = sorted(DYNAMIC_LEVELS.values())
        for val in dyn_vals[:-1]:
            line = pg.InfiniteLine(pos=val, angle=0, pen=pg.mkPen(self.theme.grid, width=0.8, style=QtCore.Qt.DashLine))
            line.setZValue(-1)
            self.plot.addItem(line)
            self.static_items.append(line)

        boundaries = [0] + dyn_vals + [128]
        tick_positions = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(dyn_vals))]
        dyn_labels = [k for k, _ in sorted(DYNAMIC_LEVELS.items(), key=lambda kv: kv[1])]
        label_x = 0.0
        fade_frac = min((time.monotonic() - self.start_time) / 5.0, 1.0)
        dyn_opacity = 0.5 + 0.5 * (1.0 - fade_frac)
        for pos, lab in zip(tick_positions, dyn_labels):
            target_h_px = 20
            pix = self.dynamic_pixmaps.get(lab.lower())
            if pix:
                scaled = pix.scaledToHeight(target_h_px, QtCore.Qt.SmoothTransformation)
                item = QtWidgets.QGraphicsPixmapItem(scaled)
                item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
                item.setOffset(-scaled.width() / 2, -scaled.height() / 2)
                item.setPos(label_x, pos)
                item.setOpacity(dyn_opacity)
                item.setZValue(-2)
                effect = QtWidgets.QGraphicsColorizeEffect()
                effect.setColor(QtGui.QColor(self.theme.axis))
                effect.setStrength(1.0)
                item.setGraphicsEffect(effect)
                item.setVisible(self.dynamics_visible)
                self.plot.addItem(item)
                self.static_items.append(item)
                self.dynamic_items.append(item)
                self.dynamics_items.append(item)
            else:
                font = QtGui.QFont()
                font.setPointSize(11)
                font.setBold(True)
                font.setItalic(True)
                txt_item = QtWidgets.QGraphicsSimpleTextItem(lab)
                txt_item.setFont(font)
                txt_item.setBrush(pg.mkBrush(self.theme.axis))
                txt_item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
                txt_item.setOpacity(dyn_opacity)
                txt_item.setZValue(-2)
                txt_item.setPos(label_x - txt_item.boundingRect().width() / 2, pos - txt_item.boundingRect().height() / 2)
                txt_item.setVisible(self.dynamics_visible)
                self.plot.addItem(txt_item)
                self.static_items.append(txt_item)
                self.dynamic_items.append(txt_item)
                self.dynamics_items.append(txt_item)

        mid_x = CHORD_SPACING * (self.window_size - 1) * 0.5
        right_x = CHORD_SPACING * (self.window_size - 1) + 0.45
        banner = pg.TextItem(
            html=f"<span style='color:{self.theme.axis}; font-size:16pt; font-weight:bold;'>&nbsp;</span>",
            anchor=(0.5, 0.5),
        )
        banner.setPos(mid_x, BANNER_Y)
        banner.setOpacity(0.0)
        banner.setZValue(3)
        self.plot.addItem(banner)
        self.banner_item = banner

        chord_banner = pg.TextItem(
            html=f"<span style='color:{self.theme.axis}; font-size:16pt; font-weight:bold;'>&nbsp;</span>",
            anchor=(1.0, 0.5),
        )
        chord_banner.setPos(right_x, BANNER_Y)
        chord_banner.setOpacity(0.0)
        chord_banner.setZValue(3)
        self.plot.addItem(chord_banner)
        self.banner_chord_item = chord_banner

        score_banner = pg.TextItem(
            html=f"<span style='color:{self.theme.axis}; font-size:16pt; font-weight:bold;'>Score</span>",
            anchor=(0.0, 0.5),
        )
        score_banner.setOpacity(0.9)
        score_banner.setZValue(3)
        self.plot.addItem(score_banner)
        self.score_banner = score_banner

        self._draw_targets()

    def _draw_targets(self) -> None:
        """Render target chords for the visible window."""
        label_color = QtGui.QColor("#ffffff")
        pen_width = max(6.5, min(11.0, 7.8 * self.scale))
        font_size = 20 # max(14, min(22, int(round(15.5 * self.scale))))
        self.target_items.clear()
        self.target_label_items = []
        if not self.targets:
            return

        visible = range(self.window_start, min(len(self.targets), self.window_start + self.window_size))

        for rel_idx, idx in enumerate(visible):
            target = self.targets[idx]
            x = rel_idx * CHORD_SPACING
            placed: List[float] = []
            chord_labels: List[Tuple[float, str, int]] = []

            def place_velocity(target_vel: float) -> float:
                if all(abs(target_vel - other) >= MIN_GAP for other in placed):
                    placed.append(target_vel)
                    return target_vel
                step = 1
                while True:
                    for direction in (-1, 1):
                        candidate = target_vel + direction * step * MIN_GAP
                        if all(abs(candidate - other) >= MIN_GAP for other in placed):
                            placed.append(candidate)
                            return candidate
                    step += 1

            for r, (pitch, vel) in enumerate(sorted(zip(target.notes, target.velocities), key=lambda t: t[0])):
                y = place_velocity(vel)
                pen = pg.mkPen(self.theme.color_for_index(r), width=pen_width, cap=QtCore.Qt.RoundCap)
                curve = pg.PlotCurveItem(
                    x=[x - LINE_HALF_WIDTH * self.scale, x + LINE_HALF_WIDTH * self.scale], y=[y, y], pen=pen
                )
                curve.setZValue(0.4)
                self.plot.addItem(curve)
                self.target_items.append(curve)
                chord_labels.append((y, midi_note_name(pitch), pitch))

            chord_labels.sort(key=lambda t: t[0])
            for idx_label, (y, note_name, pitch) in enumerate(chord_labels):
                dir_sign = -1 if idx_label % 2 == 0 else 1
                x_offset = 0.45 * dir_sign * self.scale
                y_offset = 0.0
                anchor = (1, 0.5) if dir_sign < 0 else (0, 0.5)
                text = pg.TextItem(text=note_name, color=label_color, anchor=anchor)
                font = QtGui.QFont()
                font.setPointSize(font_size)
                text.setFont(font)
                text.setPos(x + x_offset, y + y_offset)
                text.setZValue(0.6)
                self.plot.addItem(text)
                self.target_items.append(text)
                self.target_label_items.append((pitch, text, idx))

    def clear_notes(self) -> None:
        for item in self.note_items:
            self.plot.removeItem(item)
        self.note_items.clear()
        for item in self.target_items:
            self.plot.removeItem(item)
        self.target_items.clear()
        for item in self.score_items:
            self.plot.removeItem(item)
        self.score_items.clear()

    def _update_banner(self) -> None:
        if not self.banner_item or not self.banner_chord_item:
            return
        active_notes = self.active_notes.current_pressed()
        if active_notes:
            chord_name = detect_chord(active_notes)
            signature = (tuple(sorted(active_notes)), chord_name)
            if signature != self.last_banner_signature:
                names = [midi_note_name(p) for p in active_notes]
                note_text = " ".join(names)
                chord_text = chord_name if chord_name else ""
                self.banner_item.setHtml(
                    f"<span style='color:{self.theme.axis}; font-size:16pt; font-weight:bold;'>{note_text}</span>"
                )
                self.banner_chord_item.setHtml(
                    f"<span style='color:{self.theme.axis}; font-size:16pt; font-weight:bold;'>{chord_text}</span>"
                )
                self.last_banner_signature = signature
            self.banner_item.setOpacity(0.85)
            self.banner_chord_item.setOpacity(0.85 if chord_name else 0.0)
        else:
            self.banner_item.setOpacity(0.0)
            self.banner_chord_item.setOpacity(0.0)

    def _update_progress(self, status: Optional[ProgressStatus]) -> None:
        # Intentionally no progress overlay.
        return

    def update_view(
        self,
        progress_status: Optional[ProgressStatus],
        chords: List[Tuple[float, List[Tuple[int, int]]]],
    ) -> None:
        fade_frac = min((time.monotonic() - self.start_time) / 5.0, 1.0)
        dyn_opacity = 0.5 + 0.5 * (1.0 - fade_frac)
        for item in self.dynamic_items:
            item.setOpacity(dyn_opacity)

        if self.free_play:
            self._populate_free_window()

        self.clear_notes()
        self._draw_targets()

        # Capture the first N chords only (N = len(targets)); ignore any beyond.
        if self.free_play:
            for start, played in sorted(chords, key=lambda x: x[0]):
                if start <= self._last_seen_start:
                    continue
                self._last_seen_start = start
                self.free_history.append(played)
                if self.follow_latest:
                    self.window_start = max(0, len(self.free_history) - self.window_size)
            # Build captured window from history
            self._populate_free_window()
        elif self.captured_chords:
            for start, played in sorted(chords, key=lambda x: x[0]):
                if start <= self._last_seen_start:
                    continue
                self._last_seen_start = start
                try:
                    next_slot = self.captured_chords.index(None)
                except ValueError:
                    if self.free_play:
                        # shift left to make room
                        self.captured_chords.pop(0)
                        self.captured_chords.append(None)
                        if self.score_top:
                            item = self.score_top.pop(0)
                            if item:
                                self.plot.removeItem(item)
                            self.score_top.append(None)
                        if self.score_bottom:
                            item = self.score_bottom.pop(0)
                            if item:
                                self.plot.removeItem(item)
                            self.score_bottom.append(None)
                        next_slot = len(self.captured_chords) - 1
                    else:
                        break  # all slots filled
                self.captured_chords[next_slot] = played
                if self.follow_latest:
                    self.window_start = max(
                        0, min(next_slot - self.window_size + 1, max(0, len(self.targets) - self.window_size))
                    )

        self._sync_slider()

        max_x = CHORD_SPACING * max(1, self.window_size - 1) + CHORD_SPACING * 0.55
        x_left = -CHORD_SPACING * 0.55
        self.plot.setXRange(x_left, max_x, padding=0)

        # Reposition banners to the current view width.
        left, right = self.plot.getPlotItem().viewRange()[0]
        mid_x = (left + right) * 0.5
        right_x = right - 0.35
        if self.banner_item:
            self.banner_item.setPos(mid_x, BANNER_Y)
        if self.banner_chord_item:
            self.banner_chord_item.setPos(right_x, BANNER_Y)
        if self.score_banner:
            self.score_banner.setPos(left + 0.2, BANNER_Y)

        # Draw captured chords aligned to their column index for the visible window.
        total_cols = len(self.free_history) if self.free_play else len(self.targets)
        visible = range(self.window_start, min(total_cols, self.window_start + self.window_size))
        for rel_idx, idx_col in enumerate(visible):
            if self.free_play:
                played = self.captured_chords[rel_idx] if rel_idx < len(self.captured_chords) else None
                target = TargetChord([], [], "", frozenset(), False, False, None)
            else:
                played = self.captured_chords[idx_col]
                target = self.targets[idx_col] if idx_col < len(self.targets) else TargetChord([], [], "", frozenset(), False, False, None)
            if played is None:
                continue
            x = rel_idx * CHORD_SPACING
            placed: List[float] = []
            chord_labels: List[Tuple[float, str, int]] = []

            def place_velocity(target_vel: float) -> float:
                if all(abs(target_vel - other) >= MIN_GAP for other in placed):
                    placed.append(target_vel)
                    return target_vel
                step = 1
                while True:
                    for direction in (-1, 1):
                        candidate = target_vel + direction * step * MIN_GAP
                        if all(abs(candidate - other) >= MIN_GAP for other in placed):
                            placed.append(candidate)
                            return candidate
                    step += 1

            for r, (pitch, vel) in enumerate(sorted(played, key=lambda x: x[0])):
                y = place_velocity(vel)
                is_extra = pitch not in target.notes if (target.notes and not self.free_play) else False
                color_hex = "#ff2b2b" if is_extra else self.theme.color_for_index(r)
                pen = pg.mkPen(color_hex, width=max(7.0, min(12.0, 8.2 * self.scale)), cap=QtCore.Qt.RoundCap)
                curve = pg.PlotCurveItem(
                    x=[x - LINE_HALF_WIDTH * self.scale, x + LINE_HALF_WIDTH * self.scale], y=[y, y], pen=pen
                )
                curve.setZValue(1)
                self.plot.addItem(curve)
                self.note_items.append(curve)
                chord_labels.append((y, midi_note_name(pitch), pitch, is_extra))

            chord_labels.sort(key=lambda t: t[0])
            for idx_label, (y, note_name, pitch, is_extra) in enumerate(chord_labels):
                if not is_extra and not self.free_play:
                    continue  # hide labels for intended notes; rely on target label turning green
                # Flip starting side so played labels avoid overlapping target labels when pitches/velocities match.
                dir_sign = 1 if idx_label % 2 == 0 else -1
                x_offset = 0.45 * dir_sign * self.scale
                y_offset = 0.0
                anchor = (1, 0.5) if dir_sign < 0 else (0, 0.5)
                is_extra = pitch not in target.notes
                if self.free_play:
                    text_color = "#ffffff"
                else:
                    text_color = "#ff7171" if is_extra else self.theme.label
                text = pg.TextItem(text=note_name, color=text_color, anchor=anchor)
                font = QtGui.QFont()
                font.setPointSize(max(14, min(22, int(round(16.0 * self.scale)))))
                text.setFont(font)
                text.setPos(x + x_offset, y + y_offset)
                text.setZValue(2)
                self.plot.addItem(text)
                self.note_items.append(text)

        self._update_banner()
        self._update_progress(progress_status)
        self._draw_scores()
        self._update_target_labels()
        self._update_score_banner()

    def _jump_latest(self) -> None:
        """Center window on latest captured chord and resume follow."""
        last_filled = -1
        for i in range(len(self.captured_chords) - 1, -1, -1):
            if self.captured_chords[i] is not None:
                last_filled = i
                break
        if last_filled >= 0:
            self.window_start = max(
                0, min(last_filled - self.window_size + 1, max(0, len(self.targets) - self.window_size))
            )
        else:
            self.window_start = 0
        self.follow_latest = True
        self.update_view(None, [])

    def _update_target_labels(self) -> None:
        """Color preset labels green when their exact pitch has been played; keep others dim."""
        if not self.target_label_items:
            return
        counters: dict[int, Counter[int]] = {}
        for idx in range(len(self.targets)):
            played = self.captured_chords[idx] if idx < len(self.captured_chords) else None
            counters[idx] = Counter(p for p, _ in played) if played else Counter()
        for pitch, item, abs_idx in self.target_label_items:
            ctr = counters.get(abs_idx, Counter())
            matched = ctr[pitch] > 0
            if matched:
                ctr[pitch] -= 1
            color = "#7cff8a" if matched else "#ffffff"
            item.setColor(color)
            item.setOpacity(1.0)

    def _update_score_banner(self) -> None:
        if not self.score_banner:
            return
        if self.free_play:
            self.score_banner.setOpacity(0.0)
            return
        self.score_banner.setOpacity(0.9)
        pitch_total = 0.0
        pitch_count = 0
        vel_total = 0.0
        vel_count = 0
        any_velocities = any(t.velocities_given for t in self.targets)
        for idx, played in enumerate(self.captured_chords):
            if played is None:
                continue
            target = self.targets[idx]
            pitch_score, vel_score = self._match_components(played, target)
            pitch_total += pitch_score
            pitch_count += 1
            if vel_score is not None and target.velocities_given:
                vel_total += vel_score
                vel_count += 1
        pitch_avg = pitch_total / pitch_count if pitch_count else 0.0
        vel_avg = vel_total / vel_count if vel_count else None
        vel_display = None
        if any_velocities:
            vel_display = vel_avg if vel_avg is not None else 0.0
        pitch_pct = int(round(pitch_avg * 100))
        if vel_display is not None:
            text = (
                f"<span style='color:{self.theme.axis}; font-size:16pt; font-weight:bold;'>"
                f"Pitch: {pitch_pct}%&nbsp;&nbsp;&middot;&nbsp;&nbsp;Vel: {int(round(vel_display * 100))}%"
                f"</span>"
            )
        else:
            text = (
                f"<span style='color:{self.theme.axis}; font-size:16pt; font-weight:bold;'>"
                f"Pitch {pitch_pct}%"
                f"</span>"
            )
        self.score_banner.setHtml(text)

    def _draw_scores(self) -> None:
        if getattr(self, "free_play", False):
            return
        # Two bars per column (visible window only): bottom = velocity goodness, top = pitch correctness ratio.
        y_bottom = 1.0
        y_top = 124.0
        bar_h = 3.0
        width = CHORD_SPACING * 0.8
        # Ensure score lists sized to targets
        if len(self.score_top) < len(self.targets):
            self.score_top = [None for _ in self.targets]
        if len(self.score_bottom) < len(self.targets):
            self.score_bottom = [None for _ in self.targets]
        visible = range(self.window_start, min(len(self.targets), self.window_start + self.window_size))
        for rel_idx, idx in enumerate(visible):
            target = self.targets[idx]
            played = self.captured_chords[idx]
            pitch_score, vel_score = self._match_components(played, target) if played else (0.0, None)
            x_center = rel_idx * CHORD_SPACING

            def update_bar(existing: Optional[QtWidgets.QGraphicsPathItem], score: float, ypos: float, store: str):
                color = self._color_from_score(score)
                if existing is None and played is None:
                    return
                if existing is None:
                    rect = QtCore.QRectF(x_center - width * 0.5, ypos, width, bar_h)
                    item = QtWidgets.QGraphicsRectItem(rect)
                    item.setBrush(pg.mkBrush(color))
                    item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                    item.setOpacity(1.0 if played is not None else 0.0)
                    item.setZValue(2.5)
                    self.plot.addItem(item)
                    getattr(self, store)[idx] = item
                else:
                    existing.setBrush(pg.mkBrush(color))
                    existing.setOpacity(1.0 if played is not None else 0.0)

            update_bar(self.score_top[idx], pitch_score, y_top, "score_top")
            if vel_score is not None and target.velocities_given:
                update_bar(self.score_bottom[idx], vel_score, y_bottom, "score_bottom")
            else:
                # Hide/remove existing velocity bar if target has no velocities.
                if self.score_bottom[idx]:
                    self.score_bottom[idx].setOpacity(0.0)


# ---- MIDI handling ----
def pick_port(preferred: Optional[str] = None) -> Optional[str]:
    ports = mido.get_input_names()
    if not ports:
        print("No MIDI input ports found. Connect a keyboard or enable a virtual port (IAC on macOS).")
        return None
    if preferred:
        for name in ports:
            if preferred.lower() in name.lower():
                return name
    return ports[0]


def listener_thread(
    port_name: str,
    stream: ChordStream,
    stop_flag: threading.Event,
    active_notes: ActiveNotes,
    request_reset: Callable[[], None],
    highlight_toggle_flag: threading.Event,
    dynamics_toggle_flag: threading.Event,
) -> None:
    with mido.open_input(port_name) as port:
        for msg in port:
            if stop_flag.is_set():
                break
            if msg.type == "note_on" and msg.velocity > 0:
                if msg.note == 21:  # A0 as reset trigger
                    request_reset()
                    continue
                if msg.note == 22:  # A#0 toggle highlight band
                    highlight_toggle_flag.set()
                    continue
                if msg.note == 23:  # B0 toggle dynamics labels
                    dynamics_toggle_flag.set()
                    continue
                stream.ingest_note_on(msg.note, msg.velocity, time.monotonic())
                active_notes.note_on(msg.note)
            if msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note == 21:
                    continue
                active_notes.note_off(msg.note)


def file_player_thread(
    file_path: str,
    stream: ChordStream,
    stop_flag: threading.Event,
    active_notes: ActiveNotes,
    request_reset: Callable[[], None],
    highlight_toggle_flag: threading.Event,
    dynamics_toggle_flag: threading.Event,
) -> None:
    mid = mido.MidiFile(file_path)
    start = time.monotonic()
    for msg in mid:
        if stop_flag.is_set():
            break
        time.sleep(msg.time)
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.note == 21:  # A0 as reset trigger
                request_reset()
                continue
            if msg.note == 22:
                highlight_toggle_flag.set()
                continue
            if msg.note == 23:
                dynamics_toggle_flag.set()
                continue
            stream.ingest_note_on(msg.note, msg.velocity, time.monotonic())
            active_notes.note_on(msg.note)
        if msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note == 21:
                continue
            active_notes.note_off(msg.note)


# ---- Entry point ----
def main() -> None:
    parser = argparse.ArgumentParser(description="PyQtGraph MIDI chord practice visualizer")
    parser.add_argument("target", nargs="?", help="Port substring or .mid file")
    parser.add_argument(
        "--progression-file",
        nargs="?",
        const="__LATEST__",
        help="YAML/JSON file containing the chord list. If provided without a value, uses the latest file in progressions/",
    )
    parser.add_argument("--range", nargs=2, metavar=("LO", "HI"), type=float, help="Highlight velocity range")
    parser.add_argument("--require-exact", action="store_true", help="Fail if extra pitch classes are present")
    args = parser.parse_args()

    theme = Theme(dark=True)

    highlight_range = (40.0, 70.0)
    highlight_visible = False
    if args.range:
        lo, hi = sorted(args.range)
        lo = max(0.0, lo)
        hi = min(128.0, hi)
        if lo < hi:
            highlight_range = (lo, hi)
            highlight_visible = True

    progression_path: Optional[str] = None
    if args.progression_file is not None:
        if args.progression_file == "__LATEST__":
            base = os.path.join(os.path.dirname(__file__), "progressions")
            candidates = [os.path.join(base, f) for f in os.listdir(base) if f.lower().endswith(".yaml")]
            if not candidates:
                raise FileNotFoundError("No .yaml files found in progressions/ and no progression file provided.")
            progression_path = max(candidates, key=os.path.getmtime)
            print(f"Using latest progression file: {progression_path}")
        else:
            progression_path = args.progression_file

    targets = load_progression_file(progression_path) if progression_path else []
    free_play = False
    if not targets:
        free_play = True
        targets = [
            TargetChord(notes=[], velocities=[], label=f"Free {i+1}", pcs=frozenset(), velocities_given=False, roll=False, roll_ms=None)
            for i in range(VISIBLE_CHORDS)
        ]

    file_mode = bool(args.target and os.path.isfile(args.target) and args.target.lower().endswith(".mid"))
    port_name = None if file_mode else pick_port(args.target)
    if not port_name and not file_mode:
        return

    stream = ChordStream()
    active_notes = ActiveNotes()
    stop_flag = threading.Event()
    checker = ProgressionChecker(targets, require_exact=args.require_exact) if targets else None
    reset_flag = threading.Event()
    highlight_toggle_flag = threading.Event()
    dynamics_toggle_flag = threading.Event()

    app = QtWidgets.QApplication([])
    visual = Visual(
        theme,
        highlight_range,
        active_notes,
        targets,
        highlight_visible=highlight_visible,
        dynamics_visible=True if free_play else False,
        free_play=free_play,
    )
    visual.resize(1400, 750)
    visual.show()

    def reset_all() -> None:
        visual.clear_notes()
        visual._last_seen_start = -1.0
        visual.window_start = 0
        visual.follow_latest = True
        visual.captured_chords = [None for _ in targets]
        visual.history = []
        visual.free_history = [] if free_play else visual.free_history
        for item in visual.score_items:
            visual.plot.removeItem(item)
        visual.score_items.clear()
        for item in visual.score_top:
            if item:
                visual.plot.removeItem(item)
        for item in visual.score_bottom:
            if item:
                visual.plot.removeItem(item)
        visual.score_top = [None for _ in targets]
        visual.score_bottom = [None for _ in targets]
        visual.target_items.clear()
        visual.target_label_items = []
        active_notes.reset()
        stream.reset()
        checker.index = 0
        checker._last_processed_start = None
        checker.last_label = None
        checker.last_ok = None
        checker._reset_roll_state()
        checker.finalized_chords.clear()
        visual._sync_slider()
        visual.update_view(None, [])

    # Reset captured chords on 'R'
    def handle_key(event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_R:
            reset_all()
        elif event.key() == QtCore.Qt.Key_Left:
            visual.follow_latest = False
            visual.window_start = max(0, visual.window_start - 1)
            visual._sync_slider()
            visual.update_view(None, [])
        elif event.key() == QtCore.Qt.Key_Right:
            visual.follow_latest = False
            max_start = max(0, len(targets) - visual.window_size)
            visual.window_start = min(max_start, visual.window_start + 1)
            visual._sync_slider()
            visual.update_view(None, [])
        elif event.key() == QtCore.Qt.Key_End:
            visual._jump_latest()
        elif event.key() == QtCore.Qt.Key_H:
            visual.toggle_highlight()
        elif event.key() == QtCore.Qt.Key_D:
            visual.toggle_dynamics()
    visual.keyPressEvent = handle_key  # type: ignore[assignment]

    def request_reset() -> None:
        reset_flag.set()

    # Start MIDI source thread
    if file_mode and args.target:
        t = threading.Thread(
            target=file_player_thread,
            args=(
                args.target,
                stream,
                stop_flag,
                active_notes,
                request_reset,
                highlight_toggle_flag,
                dynamics_toggle_flag,
            ),
            daemon=True,
        )
    else:
        t = threading.Thread(
            target=listener_thread,
            args=(
                port_name,
                stream,
                stop_flag,
                active_notes,
                request_reset,
                highlight_toggle_flag,
                dynamics_toggle_flag,
            ),
            daemon=True,
        )
    t.start()

    def update() -> None:
        if reset_flag.is_set():
            reset_flag.clear()
            reset_all()
            return
        if highlight_toggle_flag.is_set():
            highlight_toggle_flag.clear()
            visual.toggle_highlight()
        if dynamics_toggle_flag.is_set():
            dynamics_toggle_flag.clear()
            visual.toggle_dynamics()
        stream.flush_stale(time.monotonic())
        chords = stream.snapshot()
        if free_play:
            visual.update_view(None, chords)
        else:
            status = checker.update(chords) if checker else None
            visual.update_view(status, checker.finalized_chords if checker else chords)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    try:
        app.exec()
    finally:
        stop_flag.set()
        if 't' in locals() and t:
            t.join(timeout=1)


if __name__ == "__main__":
    main()
