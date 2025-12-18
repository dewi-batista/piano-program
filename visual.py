from __future__ import annotations

import os
import time
from collections import Counter
from typing import List, Optional, Tuple

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from checker import ProgressStatus
from config import (
    BANNER_Y,
    CHORD_SPACING,
    DEFAULT_VELOCITY,
    DYNAMIC_LEVELS,
    LINE_HALF_WIDTH,
    MIN_GAP,
    VISIBLE_CHORDS,
)
from progression import TargetChord, detect_chord, midi_note_name


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
        active_notes,
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
        font_size = 20
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
                pen = pg.mkPen(self.theme.target_stroke, width=pen_width, cap=QtCore.Qt.RoundCap)
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
                text = pg.TextItem(text=note_name, color=self.theme.target_label, anchor=anchor)
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
