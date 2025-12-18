"""
Record chords from a MIDI keyboard and write them to a YAML progression file.

Usage:
  python record_to_yaml.py --out progression.yaml [--port SUBSTRING] [--max-chords N] [--threshold SECONDS]

Behaviour:
  - Groups notes into a chord if they arrive within `threshold` seconds (default 0.05s).
  - Each captured chord is stored with its note names and velocities.
  - Press Ctrl+C (or let --max-chords be reached) to stop and write the YAML file.
  - YAML format matches the viewer: progression: - notes: [...] velocities: [...]
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import mido

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


CHORD_THRESHOLD = 0.05  # seconds to group simultaneous notes


def midi_note_name(note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = note // 12 - 1
    return f"{names[note % 12]}{octave}"


class ChordStream:
    """Aggregate incoming note_on events into chord buckets."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.window: Deque[Tuple[float, List[Tuple[int, int]]]] = deque()
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
            if t - (self._current_start or t) <= self.threshold:
                self._current_notes.append((note, velocity))
            else:
                self._finalize_current()
                self._current_start = t
                self._current_notes = [(note, velocity)]

    def flush_stale(self, now: float) -> None:
        with self._lock:
            if self._current_start is not None and now - self._current_start > self.threshold:
                self._finalize_current()

    def snapshot(self) -> List[Tuple[float, List[Tuple[int, int]]]]:
        with self._lock:
            return list(self.window)

    def pop_until(self, cutoff: float) -> List[Tuple[float, List[Tuple[int, int]]]]:
        """Pop and return chords whose start <= cutoff."""
        out: List[Tuple[float, List[Tuple[int, int]]]] = []
        with self._lock:
            while self.window and self.window[0][0] <= cutoff:
                out.append(self.window.popleft())
        return out

    def reset(self) -> None:
        with self._lock:
            self.window.clear()
            self._current_notes = []
            self._current_start = None


def pick_port(preferred: Optional[str] = None) -> Optional[str]:
    ports = mido.get_input_names()
    if not ports:
        print("No MIDI input ports found.")
        return None
    if preferred:
        for name in ports:
            if preferred.lower() in name.lower():
                return name
    return ports[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record chords from a MIDI keyboard into a YAML progression file.")
    parser.add_argument("--out", help="Output YAML file path. Defaults to progressions/recorded_YYYYMMDDHHMM.yaml")
    parser.add_argument("--port", help="Substring to match the MIDI input port.")
    parser.add_argument("--max-chords", type=int, help="Stop after recording this many chords.")
    parser.add_argument("--threshold", type=float, default=CHORD_THRESHOLD, help="Chord grouping window in seconds.")
    args = parser.parse_args()

    if yaml is None:
        raise RuntimeError("pyyaml is required. Install with `pip install pyyaml`.")

    port_name = pick_port(args.port)
    if not port_name:
        return

    stream = ChordStream(threshold=args.threshold)
    stop_flag = threading.Event()
    roll_lock = threading.Lock()
    roll_active = False
    roll_notes: List[Tuple[int, int, float]] = []
    roll_first: Optional[float] = None
    roll_last: Optional[float] = None
    roll_completed: Deque[Tuple[List[Tuple[int, int]], int]] = deque()

    def listener() -> None:
        nonlocal roll_active, roll_notes, roll_first, roll_last
        with mido.open_input(port_name) as port:
            for msg in port:
                if stop_flag.is_set():
                    break
                if msg.type == "note_on" and msg.velocity > 0:
                    now = time.monotonic()
                    if msg.note == 21:  # A0 toggles roll capture on/off
                        with roll_lock:
                            if not roll_active:
                                roll_active = True
                                roll_notes = []
                                roll_first = None
                                roll_last = None
                                print("Roll capture ON")
                            else:
                                if roll_notes and roll_first is not None and roll_last is not None:
                                    roll_ms = max(1, int((roll_last - roll_first) * 1000))
                                    roll_completed.append(([(n, v) for n, v, _ in roll_notes], roll_ms))
                                    print(f"Roll captured ({roll_ms} ms)")
                                roll_active = False
                                roll_notes = []
                                roll_first = None
                                roll_last = None
                                print("Roll capture OFF")
                        continue
                    with roll_lock:
                        if roll_active:
                            roll_notes.append((msg.note, msg.velocity, now))
                            if roll_first is None:
                                roll_first = now
                            roll_last = now
                            continue
                    stream.ingest_note_on(msg.note, msg.velocity, now)
                # note_off not needed for grouping; we rely on timing only.

    t = threading.Thread(target=listener, daemon=True)
    t.start()

    recorded: List[dict] = []
    last_cutoff = -1.0
    print(f"Recording from '{port_name}'. Play chords; press Ctrl+C to finish.")
    try:
        while not stop_flag.is_set():
            stream.flush_stale(time.monotonic())
            chords = stream.pop_until(time.monotonic())
            for start, notes in chords:
                if start <= last_cutoff:
                    continue
                last_cutoff = start
                entry = {
                    "notes": [midi_note_name(n) for n, _ in notes],
                    "velocities": [int(v) for _, v in notes],
                }
                recorded.append(entry)
                names = " ".join(entry["notes"])
                print(f"Captured chord {len(recorded)}: {names} | velocities {entry['velocities']}")
                if args.max_chords and len(recorded) >= args.max_chords:
                    stop_flag.set()
                    break
            # handle completed rolls
            with roll_lock:
                while roll_completed:
                    notes, roll_ms = roll_completed.popleft()
                    entry = {
                        "notes": [midi_note_name(n) for n, _ in notes],
                        "velocities": [int(v) for _, v in notes],
                        "roll": True,
                        "roll_ms": int(roll_ms),
                    }
                    recorded.append(entry)
                    names = " ".join(entry["notes"])
                    print(f"Captured ROLLED chord {len(recorded)}: {names} | vel {entry['velocities']} | roll_ms={roll_ms}")
                    if args.max_chords and len(recorded) >= args.max_chords:
                        stop_flag.set()
                        break
            time.sleep(0.02)
    except KeyboardInterrupt:
        stop_flag.set()

    stop_flag.set()
    t.join(timeout=1.0)

    # Finalize any in-progress roll on exit
    with roll_lock:
        if roll_active and roll_notes and roll_first is not None and roll_last is not None:
            roll_ms = max(1, int((roll_last - roll_first) * 1000))
            entry = {
                "notes": [midi_note_name(n) for n, _, _ in roll_notes],
                "velocities": [int(v) for _, v, _ in roll_notes],
                "roll": True,
                "roll_ms": int(roll_ms),
            }
            recorded.append(entry)
            print(f"Captured ROLLED chord {len(recorded)} on exit: {' '.join(entry['notes'])} | roll_ms={roll_ms}")

    data = {"progression": recorded}

    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        recordings_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "progressions"))
        os.makedirs(recordings_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d%H%M%S")
        out_path = os.path.join(recordings_dir, f"recorded_{ts}.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"Saved {len(recorded)} chords to {out_path}")


if __name__ == "__main__":
    main()
