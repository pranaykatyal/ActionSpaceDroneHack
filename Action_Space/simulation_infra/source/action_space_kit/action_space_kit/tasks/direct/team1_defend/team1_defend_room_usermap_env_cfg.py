"""The 3v2 room tasks (team1_defend_room_env.py) on the user's own tag map instead of the repo's map.yaml.

The map is the user's TELLO-3 mapping run (``map_tello_3_20261024_095602.yaml``), copied unchanged into
``simulation_infra/assets/room_maps/``. In the room task the map sets the right wall (tags facing into
the room); the asset, attacker box and defender start ring are as in the room task. Scored with the hosts'
``scripts/evaluate.py --task Team1-Defend-3v2-Room-UserMap-v0`` (or ``-Room-UserMap-Realistic-v0``).
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from .team1_defend_room_env import Team1Defend3v2RoomEnvCfg, Team1Defend3v2RoomRealisticEnvCfg

USER_MAP = Path(__file__).resolve().parents[6] / "assets" / "room_maps" / "map_tello_3_20261024_095602.yaml"


@configclass
class Team1Defend3v2RoomUserMapEnvCfg(Team1Defend3v2RoomEnvCfg):
    room_map_path = str(USER_MAP)


@configclass
class Team1Defend3v2RoomUserMapRealisticEnvCfg(Team1Defend3v2RoomRealisticEnvCfg):
    room_map_path = str(USER_MAP)
