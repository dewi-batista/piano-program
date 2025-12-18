"""
PyQtGraph MIDI chord practice visualizer.

- Runs in preset mode when a progression file is provided (or the latest is auto-selected).
- Runs in free-play mode when no progression is given (just visualize what you play).
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Callable, Optional

import mido
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from checker import ProgressionChecker
from config import VISIBLE_CHORDS
from progression import TargetChord, load_progression_file
from streaming import ActiveNotes, ChordStream
from visual import Theme, Visual


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
            TargetChord(
                notes=[],
                velocities=[],
                label=f"Free {i+1}",
                pcs=frozenset(),
                velocities_given=False,
                roll=False,
                roll_ms=None,
            )
            for i in range(VISIBLE_CHORDS)
        ]

    file_mode = bool(args.target and os.path.isfile(args.target) and args.target.lower().endswith(".mid"))
    port_name = None if file_mode else pick_port(args.target)
    if not port_name and not file_mode:
        return

    stream = ChordStream()
    active_notes = ActiveNotes()
    stop_flag = threading.Event()
    checker = ProgressionChecker(targets, require_exact=args.require_exact) if not free_play else None
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
        if checker:
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
