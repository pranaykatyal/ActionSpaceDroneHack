"""Single-change ablations of Team1-Defend-Robust-Realistic-v0, to split its gap to the official task.

Robust-Realistic adds three things to the realistic attackers at once: 3-6 m spawns, a 0.2 s command
delay and an inferred asset position. Each task below removes exactly one of them. Scored with the
hosts' ``scripts/evaluate.py --task <id>``; observations and actions are the official layout.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from action_space_kit.control.tello_command import TelloCommandCfg

from .team1_defend_env import Team1DefendRobustRealisticEnv
from .team1_defend_env_cfg import Team1DefendRobustRealisticEnvCfg


@configclass
class Team1DefendRRSpawn6EnvCfg(Team1DefendRobustRealisticEnvCfg):
    """(a) official 6 m spawns."""

    spawn_radius_range = [6.0, 6.0]


@configclass
class Team1DefendRRNoDelayEnvCfg(Team1DefendRobustRealisticEnvCfg):
    """(b) no command delay."""

    command = TelloCommandCfg(command_latency_steps=0)


class Team1DefendRRTrueSiteEnv(Team1DefendRobustRealisticEnv):
    """(c) the true asset position in the observation, as on the official task (the real asset is a map point)."""

    def believed_site_xy(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 2, device=self.device)

    def site_confidence(self) -> torch.Tensor:
        return torch.ones(self.num_envs, device=self.device)
