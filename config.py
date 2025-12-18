# Shared configuration and constants.

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
