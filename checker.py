from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import ROLL_WINDOW_MS_DEFAULT
from progression import TargetChord, detect_chord, midi_note_name


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
        self.processed_starts = set()

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
        # Advance progression regardless of hit/miss to keep flow moving.
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
                        self.processed_starts.add(start)
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
