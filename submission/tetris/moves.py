from tetris.board import Action

TRANSLATIONS = [
    [],
    [Action.MOVE_LEFT],
    [Action.MOVE_LEFT] * 2,
    [Action.MOVE_LEFT] * 3,
    [Action.MOVE_LEFT] * 4,
    [Action.MOVE_LEFT] * 5,
    [Action.MOVE_RIGHT],
    [Action.MOVE_RIGHT] * 2,
    [Action.MOVE_RIGHT] * 3,
    [Action.MOVE_RIGHT] * 4,
    [Action.MOVE_RIGHT] * 5,
]

ROTATIONS = [
    [],
    [Action.ROTATE_ANTICLOCKWISE],
    [Action.ROTATE_CLOCKWISE],
]

MOVES = [
    translation + rotation + [Action.HARD_DROP]
    for translation in TRANSLATIONS
    for rotation in ROTATIONS
] + [
    rotation + translation + [Action.HARD_DROP]
    for translation in TRANSLATIONS
    for rotation in ROTATIONS
]