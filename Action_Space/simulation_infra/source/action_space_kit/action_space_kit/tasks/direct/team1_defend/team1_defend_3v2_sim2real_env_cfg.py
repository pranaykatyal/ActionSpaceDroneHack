"""3v2 versions of the sim-to-real variants: realistic sensing (IRL) and the Tello command path (Fitted).

Each cfg is its 2v2 parent with three defenders and nothing else changed, so the 3v2 networks can be
scored where the 2v2 ones already were (hosts' ``scripts/evaluate.py --task``):

- ``3v2-IRL``: positions-only sensing (15 Hz fixes, 4 cm noise, 10% dropouts, velocities from the
  deployed Kalman filter), asset position inferred, 3-6 m spawns, 0.2 s command delay, realistic attackers.
- ``3v2-FittedT024``: IRL plus the Tello command path at the hosts' motion priors (single-lag tau 0.5 s
  matched, 0.75 m/s cap, 0.1 s delay, 0.2 s fix age, 13.5 Hz integer rc, deadband, rc loss).
- ``3v2-FittedT024-Spawn6``: the same with attackers on the official 6 m ring.
- ``3v2-FittedT024-Spawn6-PredictCVSep``: the same plus the laptop's delay compensation and teammate
  separation filter (``drone_deployment_infra/core/defend_deploy.py``): what a flight would run.
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.utils import configclass

from ..tello_defend.tello_defend_env_cfg import _OBS_DIM
from .team1_defend_3v2_env_cfg import NUM_DEFENDERS_3V2
from .team1_defend_deploy_env import Team1DefendFittedT024EnvCfg
from .team1_defend_env_cfg import Team1DefendIRLEnvCfg
from .team1_defend_flight_env import Team1DefendFittedT024Spawn6EnvCfg, Team1DefendFittedT024Spawn6PredictCVSepEnvCfg

AGENTS_3V2 = [f"defender_{i}" for i in range(NUM_DEFENDERS_3V2)]


@configclass
class Team1Defend3v2IRLEnvCfg(Team1DefendIRLEnvCfg):
    num_defenders = NUM_DEFENDERS_3V2
    possible_agents = AGENTS_3V2
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in AGENTS_3V2}
    observation_spaces = {agent: _OBS_DIM for agent in AGENTS_3V2}


@configclass
class Team1Defend3v2FittedT024EnvCfg(Team1DefendFittedT024EnvCfg):
    num_defenders = NUM_DEFENDERS_3V2
    possible_agents = AGENTS_3V2
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in AGENTS_3V2}
    observation_spaces = {agent: _OBS_DIM for agent in AGENTS_3V2}


@configclass
class Team1Defend3v2FittedT024Spawn6EnvCfg(Team1DefendFittedT024Spawn6EnvCfg):
    num_defenders = NUM_DEFENDERS_3V2
    possible_agents = AGENTS_3V2
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in AGENTS_3V2}
    observation_spaces = {agent: _OBS_DIM for agent in AGENTS_3V2}


@configclass
class Team1Defend3v2FittedT024Spawn6PredictCVSepEnvCfg(Team1DefendFittedT024Spawn6PredictCVSepEnvCfg):
    num_defenders = NUM_DEFENDERS_3V2
    possible_agents = AGENTS_3V2
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in AGENTS_3V2}
    observation_spaces = {agent: _OBS_DIM for agent in AGENTS_3V2}
