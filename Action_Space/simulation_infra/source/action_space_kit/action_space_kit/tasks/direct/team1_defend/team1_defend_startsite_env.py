"""The asset is wherever the defenders take off around: the policy's asset is the centre of their start positions.

The real flights take the asset's position as the average of the defenders' positions right after takeoff,
held for the whole flight. These tasks feed the policy exactly that (no attacker-line inference, no extra
bias); breaches are still judged on the true asset. ``-Even`` places the defenders evenly around the asset
(120 deg apart for three, opposite for two, random rotation), so the centre is exact; without it they start
at the hosts' random angles on the ring and the centre can be well off. Otherwise the official tasks
(6 m spawns, no command delay). Scored with the hosts' ``scripts/evaluate.py --task <id>``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from action_space_kit.control.tello_command import TelloCommandCfg

from ..tello_defend.tello_defend_env_cfg import _OBS_DIM
from .team1_defend_env import Team1DefendRobustEnv
from .team1_defend_env_cfg import Team1DefendRobustEnvCfg


@configclass
class Team1DefendStartSiteEnvCfg(Team1DefendRobustEnvCfg):
    """2 defenders, hosts' random start angles; the policy's asset = centre of the start positions."""

    spawn_radius_range = [6.0, 6.0]
    command: TelloCommandCfg = TelloCommandCfg()
    site_inference_off_prob = 1.0  # always the start-centre prior, never the attacker-line estimate
    site_bias_off_xy = 0.0
    site_bias_xy = 0.0
    even_start = False


@configclass
class Team1DefendStartSiteEvenEnvCfg(Team1DefendStartSiteEnvCfg):
    even_start = True


@configclass
class Team1Defend3v2StartSiteEnvCfg(Team1DefendStartSiteEnvCfg):
    num_defenders = 3
    possible_agents = [f"defender_{i}" for i in range(3)]
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}


@configclass
class Team1Defend3v2StartSiteEvenEnvCfg(Team1Defend3v2StartSiteEnvCfg):
    even_start = True


class Team1DefendStartSiteEnv(Team1DefendRobustEnv):
    cfg: Team1DefendStartSiteEnvCfg

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(self._defenders[0]._ALL_INDICES if env_ids is None else env_ids, device=self.device, dtype=torch.long)
        if self.cfg.even_start:
            # evenly spaced around the asset with a random rotation, same radius and heights as the hosts'
            num = len(ids)
            origins = self.scene.env_origins[ids]
            base = sample_uniform(0.0, 2.0 * math.pi, (num,), self.device)
            count = len(self._defenders)
            for i, drone in enumerate(self._defenders):
                angle = base + 2.0 * math.pi * i / count
                state = drone.data.default_root_state[ids].clone()
                state[:, 0] = origins[:, 0] + self.cfg.defender_radius * torch.cos(angle)
                state[:, 1] = origins[:, 1] + self.cfg.defender_radius * torch.sin(angle)
                state[:, 2] = origins[:, 2] + sample_uniform(1.0, 2.0, (num,), self.device)
                state[:, 7:] = 0.0
                drone.write_root_pose_to_sim(state[:, :7], ids)
                drone.write_root_velocity_to_sim(state[:, 7:], ids)
                self._defender_ctl[i].reset(ids)
        # the asset the policy sees: the centre of where the defenders start, held for the episode
        start = torch.stack([d.data.root_pos_w[ids, :2] for d in self._defenders], dim=1).mean(dim=1)
        if self.cfg.even_start:
            # write_root_pose_to_sim updates the data buffers on the next sim step; use the placed positions
            start = torch.stack(
                [self.scene.env_origins[ids, :2] + self.cfg.defender_radius * torch.stack([torch.cos(base + 2.0 * math.pi * i / count), torch.sin(base + 2.0 * math.pi * i / count)], dim=-1) for i in range(count)],
                dim=1,
            ).mean(dim=1)
        prior = start - self.scene.env_origins[ids, :2]
        self._site_prior[ids] = prior
        self._site_estimator.reset(prior, ids)
        self._site_bias[ids] = 0.0
        self._site_infer[ids] = False
