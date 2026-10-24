"""The organizers' real-flight capture rule in the sim: an attacker is captured only once it is
knocked out of the air, not when a defender merely comes within ``capture_radius``.

Knocked down = tilted past ``knockdown_tilt_deg`` (30 deg, the flip definition used in the crash
autopsies; only a collision tilts an attacker, its controller holds it level) or below
``knockdown_height`` above the floor. A knocked attacker is captured: alive off and counted in
``_episode_captures`` exactly like the official capture, so ``evaluate.py`` and ``es_train.py``
count it unchanged. Its rotors are then cut, so it falls instead of hovering where it was hit.
Proximity alone captures nothing (``capture_radius`` 0). Physics is the hosts': a defender that
tips over in the hit is down.

The hosts' capture reward and ``cleared`` bonus never fire here (the knockdown marks the attacker
dead before their check). That doesn't matter for ES or ``evaluate.py``; it does for PPO.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from action_space_kit.tasks.direct.tello_defend.tello_defend_env import TelloDefendEnv
from action_space_kit.tasks.direct.tello_defend.tello_defend_env_cfg import TelloDefendEnvCfg

from .team1_defend_deploy_env import Team1DefendFittedDeployEnv
from .team1_defend_flight_env import Team1DefendFittedT024Spawn6PredictCVSepEnvCfg


class KnockdownCaptureMixin:
    """Goes before the env class: ``class X(KnockdownCaptureMixin, SomeDefendEnv)``."""

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        upright_min = math.cos(math.radians(self.cfg.knockdown_tilt_deg))
        for i, drone in enumerate(self._attackers):
            upright = -drone.data.projected_gravity_b[:, 2]
            height = drone.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
            knocked = ((upright < upright_min) | (height < self.cfg.knockdown_height)) & self._attacker_alive[:, i]
            self._attacker_alive[:, i] &= ~knocked
            self._episode_captures += knocked.float()
        return super()._get_rewards()

    def _apply_action(self) -> None:
        super()._apply_action()
        # a downed attacker falls out of the air instead of hovering where it was hit
        for i, drone in enumerate(self._attackers):
            ids = (~self._attacker_alive[:, i]).nonzero(as_tuple=False).flatten()
            if len(ids) > 0:  # an empty call would disable the wrench for every env
                zeros = torch.zeros(len(ids), 1, 3, device=self.device)
                drone.set_external_force_and_torque(forces=zeros, torques=zeros, body_ids=[0], env_ids=ids)


class Team1DefendKnockdownEnv(KnockdownCaptureMixin, TelloDefendEnv):
    pass


class Team1DefendFlightKnockdownEnv(KnockdownCaptureMixin, Team1DefendFittedDeployEnv):
    pass


@configclass
class Team1DefendKnockdownEnvCfg(TelloDefendEnvCfg):
    """Official AS-Defend-v0 with the knockdown capture."""

    capture_radius = 0.0
    knockdown_tilt_deg: float = 30.0
    knockdown_height: float = 0.3


@configclass
class Team1DefendFlightKnockdownEnvCfg(Team1DefendFittedT024Spawn6PredictCVSepEnvCfg):
    """What we fly (FittedT024 at 6 m, predictor + separation filter) with the knockdown capture."""

    capture_radius = 0.0
    knockdown_tilt_deg: float = 30.0
    knockdown_height: float = 0.3
