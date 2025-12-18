from __future__ import annotations

import threading
from collections import deque
from typing import List, Optional, Tuple

from config import CHORD_HISTORY, CHORD_THRESHOLD


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
