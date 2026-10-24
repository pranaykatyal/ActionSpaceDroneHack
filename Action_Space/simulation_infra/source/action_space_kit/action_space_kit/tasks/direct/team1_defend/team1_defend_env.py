"""Team 1's training variant of AS-Defend-v0 (see team1_defend_env_cfg.py)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils.math import sample_uniform

from action_space_kit.control.defense import SiteEstimator

from ..tello_defend.tello_defend_env import TelloDefendEnv
from .team1_defend_env_cfg import Team1DefendRobustEnvCfg


class Team1DefendRobustEnv(TelloDefendEnv):
    cfg: Team1DefendRobustEnvCfg

    def __init__(self, cfg: Team1DefendRobustEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ensure_site_state()

    def _ensure_site_state(self) -> None:
        if not hasattr(self, "_site_estimator"):
            self._site_estimator = SiteEstimator(memory_s=self.cfg.site_memory_s)
            self._site_estimator.reset(torch.zeros(self.num_envs, 2, device=self.device))
            self._site_bias = torch.zeros(self.num_envs, 2, device=self.device)
            self._site_prior = torch.zeros(self.num_envs, 2, device=self.device)
            self._site_infer = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

    def believed_site_xy(self) -> torch.Tensor:
        """Believed asset position relative to each env origin (the true site is at 0, 0), (N, 2)."""
        inferred = torch.where(self._site_infer[:, None], self._site_estimator.believed, self._site_prior)
        return inferred + self._site_bias

    def site_confidence(self) -> torch.Tensor:
        """The estimator's confidence, 0 in episodes where inference is off, (N,)."""
        return torch.where(self._site_infer, self._site_estimator.confidence(), torch.zeros(self.num_envs, device=self.device))

    def _site_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Attacker positions (env-relative) and velocities the asset estimator sees: true here."""
        att_pos, att_vel = self._attacker_states()
        return att_pos - self.scene.env_origins[:, None, :], att_vel

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        super()._pre_physics_step(actions)
        att_pos, att_vel = self._site_inputs()
        self._site_estimator.update(att_pos, att_vel, self._attacker_alive, self.step_dt)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = super()._get_observations()
        if hasattr(self, "_site_estimator"):
            believed = self.believed_site_xy()
            for agent in obs:
                # own position relative to the believed site instead of the true one
                obs[agent][:, 0:2] = obs[agent][:, 0:2] - believed
        return obs

    def _place_attackers(self, env_ids: torch.Tensor) -> None:
        """Re-place this wave's attackers at a radius drawn from ``spawn_radius_range``."""
        num = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        low, high = self.cfg.spawn_radius_range
        for drone in self._attackers:
            angle = sample_uniform(0.0, 2.0 * math.pi, (num,), self.device)
            radius = sample_uniform(low, high, (num,), self.device)
            state = drone.data.default_root_state[env_ids].clone()
            state[:, 0] = origins[:, 0] + radius * torch.cos(angle)
            state[:, 1] = origins[:, 1] + radius * torch.sin(angle)
            state[:, 2] = origins[:, 2] + sample_uniform(1.5, 3.0, (num,), self.device)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)

    def _respawn_attackers(self, env_ids: torch.Tensor) -> None:
        super()._respawn_attackers(env_ids)
        if len(env_ids):
            self._place_attackers(env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._place_attackers(ids)
        self._ensure_site_state()
        # prior: we place the defenders around the asset, so start from their centroid
        defenders = torch.stack([d.data.root_pos_w[ids, :2] for d in self._defenders], dim=1)
        prior = defenders.mean(dim=1) - self.scene.env_origins[ids, :2]
        self._site_estimator.reset(prior, ids)
        self._site_prior[ids] = prior
        infer = torch.rand(len(ids), device=self.device) >= self.cfg.site_inference_off_prob
        self._site_infer[ids] = infer
        bias = torch.where(infer, torch.full_like(infer, self.cfg.site_bias_xy, dtype=prior.dtype), torch.full_like(infer, self.cfg.site_bias_off_xy, dtype=prior.dtype))
        self._site_bias[ids] = (torch.rand(len(ids), 2, device=self.device) * 2.0 - 1.0) * bias[:, None]


from .team1_attackers import RealisticAttackersMixin  # noqa: E402


class Team1DefendRobustRealisticEnv(RealisticAttackersMixin, Team1DefendRobustEnv):
    """Robust v1 with realistic attacker paths."""


from .team1_sensing import EstimatedStateMixin  # noqa: E402


class Team1DefendIRLEnv(EstimatedStateMixin, RealisticAttackersMixin, Team1DefendRobustEnv):
    """Robust-Realistic with positions-only sensing and Kalman-estimated velocities."""
