"""What the defenders really know in the room: positions only, velocities estimated.

Real-world assumption (team): every drone's position is known (defenders from their own
AprilTags, attackers from the room's tracking), but no speed is reported and the attackers
carry no tags. So positions arrive as noisy fixes at a limited rate, some are lost, and every
velocity -- the defender's own and each attacker's -- comes from the laptop's Kalman filter
(``drone_deployment_infra/core/positional_kalman_filter.py`` via ``tags/kalman.py``, the same
code the flight loop runs). The policy then trains on the estimate it will get in the room.

Fixes: every ``1 / pos_fix_hz`` s, each drone's true position plus Gaussian ``pos_fix_noise``
per axis, dropped with probability ``pos_fix_dropout``. Tilt (projected gravity) stays true:
the Tello's IMU reports it. Rewards, captures and breaches stay on the true state.
The mixin goes first: ``class X(EstimatedStateMixin, ..., SomeDefendEnv)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from action_space_kit.tags.kalman import KalmanCfg, KalmanCV


class EstimatedStateMixin:
    def _ensure_estimator(self) -> None:
        if not hasattr(self, "_kf"):
            self._kf_drones = self.cfg.num_defenders + self.cfg.num_attackers
            self._kf = KalmanCV(KalmanCfg(), batch=(self.num_envs, self._kf_drones), xp=torch, device=self.device)
            self._kf_needs_fix = torch.ones(self.num_envs, self._kf_drones, dtype=torch.bool, device=self.device)
            self._kf_tick = 0
            self._kf_seen_tick = -1
            self._fix_every = max(1, round(1.0 / (self.cfg.pos_fix_hz * self.step_dt)))

    def _true_positions(self) -> torch.Tensor:
        """Defenders then attackers, relative to each env origin, (N, D + A, 3)."""
        drones = list(self._defenders) + list(self._attackers)
        return torch.stack([d.data.root_pos_w for d in drones], dim=1) - self.scene.env_origins[:, None, :]

    def _update_estimates(self) -> None:
        """Once per env step: predict, then fuse whatever fixes arrived."""
        self._ensure_estimator()
        if self._kf_seen_tick == self._kf_tick:
            return
        self._kf_seen_tick = self._kf_tick
        self._kf.predict(self.step_dt)
        on_schedule = (self.episode_length_buf % self._fix_every == 0)[:, None].expand(-1, self._kf_drones)
        arrived = on_schedule & (torch.rand(on_schedule.shape, device=self.device) >= self.cfg.pos_fix_dropout)
        mask = arrived | self._kf_needs_fix  # a fresh episode always starts from a fix
        if mask.any():
            z = self._true_positions() + torch.randn(self.num_envs, self._kf_drones, 3, device=self.device) * self.cfg.pos_fix_noise
            self._kf.update(z, mask)
        self._kf_needs_fix[:] = False

    def _site_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Attacker positions (env-relative) and velocities as estimated, for the asset estimator."""
        self._ensure_estimator()
        n_def = self.cfg.num_defenders
        return self._kf.p[:, n_def:], self._kf.v[:, n_def:]

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self._ensure_estimator()
        self._kf_tick += 1
        super()._pre_physics_step(actions)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """The official 23-wide layout, built from estimated positions and velocities."""
        self._update_estimates()
        n_def = self.cfg.num_defenders
        pos, vel = self._kf.p, self._kf.v
        site = torch.zeros(self.num_envs, 3, device=self.device)
        site[:, 2] = self.cfg.site_height * 0.5
        if hasattr(self, "believed_site_xy"):  # asset position inferred, not known
            site[:, :2] = self.believed_site_xy()
        att_pos, att_vel = pos[:, n_def:], vel[:, n_def:]
        alive = self._attacker_alive.float().unsqueeze(-1)
        obs = {}
        for i, drone in enumerate(self._defenders):
            rel_pos = (att_pos - pos[:, i : i + 1]).clamp(-self.cfg.obs_clip_pos, self.cfg.obs_clip_pos) * alive
            rel_vel = (att_vel - vel[:, i : i + 1]).clamp(-self.cfg.obs_clip_vel, self.cfg.obs_clip_vel) * alive
            obs[f"defender_{i}"] = torch.cat(
                [
                    pos[:, i] - site,
                    vel[:, i],
                    drone.data.projected_gravity_b,
                    rel_pos.reshape(self.num_envs, -1),
                    rel_vel.reshape(self.num_envs, -1),
                    alive.squeeze(-1),
                ],
                dim=-1,
            )
        return obs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)
        self._ensure_estimator()
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        rows = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        rows[ids] = True
        self._kf.reset(rows[:, None].expand(-1, self._kf_drones))
        self._kf_needs_fix[ids] = True
