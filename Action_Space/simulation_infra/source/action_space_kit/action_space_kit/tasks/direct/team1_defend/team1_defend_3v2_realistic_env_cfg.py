"""Team1-Defend-Realistic-v0 with three defenders against the same two attackers.

The realistic attackers (team1_attackers.py: 70% weave, slow heading wobble, 0.2-1.0 m/s) on the
official geometry, with the 3v2 team sizes of ``team1_defend_3v2_env_cfg.py``. The 23-number
observation and the action are unchanged. Scored with the hosts' ``scripts/evaluate.py --task
Team1-Defend-3v2-Realistic-v0``.
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.utils import configclass

from ..tello_defend.tello_defend_env_cfg import _OBS_DIM
from .team1_defend_3v2_env_cfg import NUM_DEFENDERS_3V2
from .team1_defend_shaped_env import Team1DefendRealisticEnvCfg


@configclass
class Team1Defend3v2RealisticEnvCfg(Team1DefendRealisticEnvCfg):
    num_defenders = NUM_DEFENDERS_3V2
    possible_agents = [f"defender_{i}" for i in range(NUM_DEFENDERS_3V2)]
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}
