"""AS-Defend-v0 with attackers starting closer than the official 6 m ring, and nothing else changed.

The real room is about 4.5 x 3.4 m, so real attackers may start well inside 6 m. These tasks keep
the official dynamics, attacker speeds, zero command delay and true asset position, and change only
the spawn radius (per attacker, per wave). Scored with the hosts' ``scripts/evaluate.py --task <id>``;
observations and actions are the official layout.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from action_space_kit.control.tello_command import TelloCommandCfg

from .team1_defend_env import Team1DefendRobustEnv
from .team1_defend_env_cfg import Team1DefendRobustEnvCfg


class Team1DefendCloseEnv(Team1DefendRobustEnv):
    """Team1DefendRobustEnv's spawn placement with the true asset position, as on the official task."""

    def believed_site_xy(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 2, device=self.device)

    def site_confidence(self) -> torch.Tensor:
        return torch.ones(self.num_envs, device=self.device)


@configclass
class Team1DefendCloseEnvCfg(Team1DefendRobustEnvCfg):
    """Spawns 3-6 m out; official command path (no delay) and no asset bias."""

    spawn_radius_range = [3.0, 6.0]
    site_bias_xy = 0.0
    command: TelloCommandCfg = TelloCommandCfg()


@configclass
class Team1DefendSpawn25EnvCfg(Team1DefendCloseEnvCfg):
    spawn_radius_range = [2.5, 2.5]


@configclass
class Team1DefendSpawn3EnvCfg(Team1DefendCloseEnvCfg):
    spawn_radius_range = [3.0, 3.0]


@configclass
class Team1DefendSpawn4EnvCfg(Team1DefendCloseEnvCfg):
    spawn_radius_range = [4.0, 4.0]


@configclass
class Team1DefendSpawn5EnvCfg(Team1DefendCloseEnvCfg):
    spawn_radius_range = [5.0, 5.0]


@configclass
class Team1DefendSpawn6EnvCfg(Team1DefendCloseEnvCfg):
    """Sanity check: should score like AS-Defend-v0."""

    spawn_radius_range = [6.0, 6.0]


# Where the defenders start matters as much as where the attackers do: from the hosts' 2.5 m ring a
# defender can reach a 3 m attacker before it crosses 1.5 m only if it starts within ~50-70 deg of its
# bearing. The real room (about 4.5 x 3.4 m) can't hold a 2.5 m ring around a central asset anyway.


@configclass
class Team1DefendSpawn3Def1EnvCfg(Team1DefendSpawn3EnvCfg):
    """3 m spawns, defenders start 1.0 m from the asset."""

    defender_radius = 1.0


@configclass
class Team1DefendSpawn3Def15EnvCfg(Team1DefendSpawn3EnvCfg):
    """3 m spawns, defenders start 1.5 m from the asset."""

    defender_radius = 1.5


@configclass
class Team1DefendHalfEnvCfg(Team1DefendSpawn3EnvCfg):
    """The official geometry at half scale (asset 0.75 m, defenders 1.25 m, attackers 3 m); speeds unchanged."""

    site_radius = 0.75
    defender_radius = 1.25
