"""AS-Defend-v0 where only real contact counts as a capture.

The hosts' capture fires when a defender's centre is within 0.35 m of an attacker's, about 25 cm before the
bodies meet. A Tello's frame is 9.8 x 9.25 cm (the asset's collision box), so frames or propeller guards
touch at about 0.10-0.12 m centre to centre. In real flights contact is the capture, so these tasks set
``capture_radius`` to 0.12 m and change nothing else. Scored with the hosts' ``scripts/evaluate.py --task <id>``.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..tello_defend.tello_defend_env_cfg import TelloDefendEnvCfg
from .team1_defend_3v2_env_cfg import Team1Defend3v2EnvCfg

TOUCH_RADIUS = 0.12  # [m] centre to centre when two Tello frames touch


@configclass
class Team1DefendTouchEnvCfg(TelloDefendEnvCfg):
    capture_radius = TOUCH_RADIUS


@configclass
class Team1Defend3v2TouchEnvCfg(Team1Defend3v2EnvCfg):
    capture_radius = TOUCH_RADIUS
