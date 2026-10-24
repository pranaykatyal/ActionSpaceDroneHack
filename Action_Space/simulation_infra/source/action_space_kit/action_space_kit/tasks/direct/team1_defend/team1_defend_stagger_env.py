"""AS-Defend-v0 with staggered first episodes, for PPO training (scored on AS-Defend-v0).

Every env starts its first episode at the same instant, and an attacker needs at least
(spawn_radius - site_radius) / max attacker speed = 5 s to reach the asset. With 16-step rollouts at
60 Hz, PPO's first ~20 updates therefore see the same early slice of one synchronized episode and
not a single breach (skrl logs the first finished episode at timestep ~324 in every run). Here the
first episode of each env begins at a random point of an episode instead: its clock is advanced by
a random elapsed time and each attacker is moved as far along its inbound line as it would have
flown by then (wrapping into a later wave once it would have arrived). Every later episode is the
official one. Scenario, reward, observations and actions are AS-Defend-v0's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from ..tello_defend.tello_defend_env import TelloDefendEnv
from ..tello_defend.tello_defend_env_cfg import TelloDefendEnvCfg


@configclass
class Team1DefendStaggerEnvCfg(TelloDefendEnvCfg):
    """Official scenario and reward; staggered first episodes."""

    stagger_margin = 0.3  # [m] a pre-aged attacker stays at least this far outside the protected cylinder


class Team1DefendStaggerEnv(TelloDefendEnv):
    cfg: Team1DefendStaggerEnvCfg

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)
        if getattr(self, "_staggered", False) or len(env_ids) < self.num_envs:
            return
        self._staggered = True  # only the very first, all-env reset
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        num = len(ids)
        elapsed = torch.randint(0, int(self.max_episode_length), (num,), device=self.device)
        self.episode_length_buf[ids] = elapsed
        seconds = elapsed.float() * self.step_dt
        origins = self.scene.env_origins[ids]
        inner = self.cfg.site_radius + self.cfg.stagger_margin
        span = self.cfg.spawn_radius - inner
        for i, drone in enumerate(self._attackers):
            # straight-line attackers: distance flown = speed x time, wrapping into a later wave
            radius = self.cfg.spawn_radius - torch.remainder(self._attacker_speed[ids, i] * seconds, span)
            angle = sample_uniform(0.0, 2.0 * math.pi, (num,), self.device)
            state = drone.data.default_root_state[ids].clone()
            state[:, 0] = origins[:, 0] + radius * torch.cos(angle)
            state[:, 1] = origins[:, 1] + radius * torch.sin(angle)
            state[:, 2] = origins[:, 2] + sample_uniform(1.5, 3.0, (num,), self.device)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], ids)
            drone.write_root_velocity_to_sim(state[:, 7:], ids)
