"""Attackers that fly like our real Tellos do, for Team 1's training variants.

The official attackers dive in a perfectly straight line (``attacker_evasion = 0``). Real
attackers are Tellos flown by a script or a person: they head for the asset, some weave on
purpose, and none hold a perfect heading. Each attacker, every wave:

  style     weaves with probability ``attacker_weave_prob``, otherwise flies straight
  weave     sideways sine on its line to the site; strength (``attacker_evasion_range``,
            1.0 = up to +-45 degrees), period (``attacker_weave_period``) and phase are random
  wobble    a slow random heading error on every attacker: Ornstein-Uhlenbeck with stationary
            std ``attacker_wobble_std`` [rad] and time constant ``attacker_wobble_tau`` [s]
  floor     attackers never get near the ground: below ``attacker_min_height`` [m] they climb

Speeds, spawns and everything else are inherited, so the official values apply unless a
variant changes them. The mixin goes before the env class:
``class X(RealisticAttackersMixin, SomeDefendEnv)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils.math import sample_uniform


class RealisticAttackersMixin:
    def _ensure_attacker_style(self) -> None:
        if not hasattr(self, "_att_evasion"):
            shape = (self.num_envs, self.cfg.num_attackers)
            self._att_evasion = torch.zeros(shape, device=self.device)
            self._att_period = torch.full(shape, 3.0, device=self.device)
            self._att_phase = torch.zeros(shape, device=self.device)
            self._att_wobble = torch.zeros(shape, device=self.device)

    def _sample_attacker_style(self, env_ids: torch.Tensor) -> None:
        self._ensure_attacker_style()
        shape = (len(env_ids), self.cfg.num_attackers)
        weave = torch.rand(shape, device=self.device) < self.cfg.attacker_weave_prob
        low, high = self.cfg.attacker_evasion_range
        self._att_evasion[env_ids] = torch.where(weave, sample_uniform(low, high, shape, self.device), torch.zeros(shape, device=self.device))
        self._att_period[env_ids] = sample_uniform(*self.cfg.attacker_weave_period, shape, self.device)
        self._att_phase[env_ids] = sample_uniform(0.0, 2.0 * math.pi, shape, self.device)
        self._att_wobble[env_ids] = torch.randn(shape, device=self.device) * self.cfg.attacker_wobble_std

    def _scripted_attacker_actions(self) -> list[torch.Tensor]:
        self._ensure_attacker_style()
        # heading wobble: Ornstein-Uhlenbeck, stepped at the physics rate this is called at
        dt, tau, std = self.physics_dt, self.cfg.attacker_wobble_tau, self.cfg.attacker_wobble_std
        self._att_wobble += -self._att_wobble * (dt / tau) + std * math.sqrt(2.0 * dt / tau) * torch.randn_like(self._att_wobble)
        t = self.episode_length_buf.float().unsqueeze(-1) * self.step_dt
        swing = self._att_evasion * torch.sin(2.0 * math.pi * t / self._att_period + self._att_phase)
        swing = swing + torch.tan(self._att_wobble.clamp(-1.2, 1.2))
        actions = []
        for i, drone in enumerate(self._attackers):
            to_site = self.scene.env_origins - drone.data.root_pos_w
            to_site[:, 2] += self.cfg.site_height * 0.5
            direction = to_site / torch.clamp(torch.linalg.norm(to_site, dim=-1, keepdim=True), min=1e-6)
            lateral = torch.stack([-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])], dim=-1)
            lateral = lateral / torch.clamp(torch.linalg.norm(lateral, dim=-1, keepdim=True), min=1e-6)
            heading = direction + swing[:, i].unsqueeze(-1) * lateral
            heading = heading / torch.clamp(torch.linalg.norm(heading, dim=-1, keepdim=True), min=1e-6)
            speed = (self._attacker_speed[:, i] / self.cfg.command.max_lin_speed).unsqueeze(-1)
            velocity = heading * speed
            # never near the ground: below the floor, climb back regardless of weave or wobble
            height = drone.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
            climb = (self.cfg.attacker_min_height - height).clamp(min=0.0) * 2.0 / self.cfg.command.max_lin_speed
            velocity[:, 2] = torch.where(height < self.cfg.attacker_min_height, torch.maximum(velocity[:, 2], climb), velocity[:, 2])
            action = torch.cat([velocity, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
            action = action * self._attacker_alive[:, i].unsqueeze(-1).float()  # a downed attacker stops
            actions.append(action.clamp(-1.0, 1.0))
        return actions

    def _respawn_attackers(self, env_ids: torch.Tensor) -> None:
        super()._respawn_attackers(env_ids)
        if len(env_ids):
            self._sample_attacker_style(env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)
        self._sample_attacker_style(torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
