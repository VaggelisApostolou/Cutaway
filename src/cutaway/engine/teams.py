"""Fictional clubs and squads. No real clubs, players or league data are used."""

from __future__ import annotations

from .models import Player, Team

# Base positions in the team's own frame (attacking towards x=105), mid-block shape.
FORMATIONS: dict[str, dict[str, tuple[float, float]]] = {
    "4-3-3": {
        "GK": (5, 34), "RB": (24, 10), "RCB": (18, 26), "LCB": (18, 42), "LB": (24, 58),
        "RCM": (40, 22), "DM": (33, 34), "LCM": (40, 46),
        "RW": (60, 12), "ST": (64, 34), "LW": (60, 56),
    },
    "4-2-3-1": {
        "GK": (5, 34), "RB": (24, 10), "RCB": (18, 27), "LCB": (18, 41), "LB": (24, 58),
        "RDM": (34, 27), "LDM": (34, 41),
        "RAM": (52, 14), "CAM": (54, 34), "LAM": (52, 54), "ST": (65, 34),
    },
}

# (number, name, position, pace, passing, finishing, defending)
_HOME = [
    (1, "Tomas Varga", "GK", 50, 60, 20, 82),
    (2, "Elias Brennan", "RB", 80, 70, 40, 74),
    (4, "Kofi Mensah-Hale", "RCB", 70, 64, 35, 84),
    (5, "Nikolai Strand", "LCB", 62, 68, 38, 81),
    (3, "Mateo Ruelas", "LB", 84, 72, 42, 70),
    (8, "Theo Lindqvist", "RCM", 70, 80, 60, 66),
    (6, "Idris Abara", "DM", 66, 78, 45, 80),
    (10, "Luca Moretti-Vance", "LCM", 72, 91, 72, 55),
    (7, "Samir Daoud", "RW", 93, 76, 75, 40),
    (9, "Jonah Okafor-Reid", "ST", 82, 70, 88, 38),
    (11, "Felix Arnaud", "LW", 86, 74, 74, 42),
    (14, "Dario Castell", "ST", 85, 66, 80, 35),
    (17, "Kenji Morita", "LCM", 74, 82, 64, 60),
    (21, "Callum Price", "RB", 76, 64, 34, 76),
]

_AWAY = [
    (1, "Pavel Novak", "GK", 48, 58, 18, 85),
    (2, "Aaron Whitlock", "RB", 81, 68, 38, 73),
    (5, "Bruno Ferreira-Lind", "RCB", 64, 66, 36, 85),
    (6, "Malik Osei", "LCB", 74, 62, 34, 82),
    (3, "Ezra Holm", "LB", 79, 70, 40, 72),
    (4, "Leon Baptiste", "RDM", 68, 80, 48, 81),
    (8, "Andre Kovac", "LDM", 70, 77, 55, 74),
    (7, "Milan Radek", "RAM", 88, 75, 72, 44),
    (10, "Rafael Quintero", "CAM", 76, 92, 78, 46),
    (11, "Yusuf Demir", "LAM", 90, 72, 70, 40),
    (9, "Oscar Bjerke", "ST", 78, 66, 89, 40),
    (19, "Ilya Sorokin", "ST", 84, 64, 81, 36),
    (16, "Joaquin Ibarra", "CAM", 80, 79, 70, 52),
    (23, "Marcus Thorne", "LCB", 68, 60, 32, 80),
]


def _build(team_id: str, name: str, short: str, color: str, formation: str, rows) -> Team:
    players = [
        Player(
            id=f"{team_id}-{num}", name=nm, number=num, position=pos, team_id=team_id,
            pace=pace, passing=pas, finishing=fin, defending=dfn, starter=i < 11,
        )
        for i, (num, nm, pos, pace, pas, fin, dfn) in enumerate(rows)
    ]
    return Team(id=team_id, name=name, short_name=short, color=color, formation=formation, players=players)


def home_team() -> Team:
    return _build("HCY", "Harbour City", "HCY", "#2563eb", "4-3-3", _HOME)


def away_team() -> Team:
    return _build("NGR", "Northgate Rovers", "NGR", "#ea580c", "4-2-3-1", _AWAY)
