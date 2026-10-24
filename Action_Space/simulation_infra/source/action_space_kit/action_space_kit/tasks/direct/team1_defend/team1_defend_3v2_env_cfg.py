"""AS-Defend-v0 with three defenders against the same two attackers (the real demo's team sizes).

Everything else is the hosts' task: geometry, attackers, rewards, the 23-number observation per
defender (it depends only on the attacker count) and the [vx, vy, vz, yaw_rate] action. The hosts'
TelloDefendEnv builds ``num_defenders`` drones, so only the config changes. Scored with the hosts'
``scripts/evaluate.py --task Team1-Defend-3v2-v0``.
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.utils import configclass

from ..tello_defend.tello_defend_env_cfg import _OBS_DIM, TelloDefendEnvCfg

NUM_DEFENDERS_3V2 = 3


@configclass
class Team1Defend3v2EnvCfg(TelloDefendEnvCfg):
    num_defenders = NUM_DEFENDERS_3V2
    possible_agents = [f"defender_{i}" for i in range(NUM_DEFENDERS_3V2)]
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}
