"""Team1-Defend-IRL-v0 with the real Tello's command path: the sim-to-real test bed for flight controllers.

The official task assumes a 0.15 s velocity lag, 1 m/s at full stick, fresh positions and a new
command every 60 Hz step. The hosts' priors for the real Tello (until our step-response flight
fits them) are a ~0.5 s lag, ~0.75 m/s at rc 100, integer rc sticks with a deadband, sent at
13.5 Hz with jitter and some lost packets, and a total loop delay of ~0.3 s. That delay is split
the way it arises: ~0.2 s of fix age (video and tag detection) applied to the positions before
the Kalman filter, and ~0.1 s on the command. Every value is a config field; replace them with
fitted ones. One task ID per lag, because scripts/evaluate.py takes no env overrides.

Scored with the hosts' ``scripts/evaluate.py --task Team1-Defend-Fitted-v0``.

Note (2026-10-24, peer review): the hosts' TelloVelocityController applies ``response_time`` twice (a
first-order lag on the target, then ``accel = error / response_time``), so the velocity reaches 63% after
about 2.15 x response_time: ~1.07 s here at 0.5, not the single-lag 0.5 s of the hosts' motion-fit prior.
``Team1-Defend-FittedT024-v0`` (response_time 0.235, in team1_defend_deploy_env.py) matches that prior.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

from action_space_kit.control.tello_command import TelloCommandCfg

from .team1_defend_env import Team1DefendIRLEnv
from .team1_defend_env_cfg import Team1DefendIRLEnvCfg
from .team1_sensing import EstimatedStateMixin


@configclass
class Team1DefendFittedEnvCfg(Team1DefendIRLEnvCfg):
    """IRL task plus the real command path (hosts' priors; replace with fitted values)."""

    # velocity response: tau, top speed at rc 100, command delay (0.1 s in 120 Hz physics steps)
    command = TelloCommandCfg(response_time=0.5, max_lin_speed=0.75, command_latency_steps=12)
    obs_delay_s = 0.2  # [s] age of a position fix when it reaches the Kalman filter
    rc_hz = 13.5  # [Hz] rc packets the laptop sends (one per flight-loop tick)
    rc_jitter_s = 0.016  # [s] extra delay per tick, uniform
    rc_deadband = 5  # |rc| below this does nothing
    rc_loss_range = [0.0, 0.15]  # fraction of rc packets lost, drawn per episode


@configclass
class Team1DefendFittedTau015EnvCfg(Team1DefendFittedEnvCfg):
    command = TelloCommandCfg(response_time=0.15, max_lin_speed=0.75, command_latency_steps=12)


@configclass
class Team1DefendFittedTau03EnvCfg(Team1DefendFittedEnvCfg):
    command = TelloCommandCfg(response_time=0.3, max_lin_speed=0.75, command_latency_steps=12)


class Team1DefendFittedEnv(Team1DefendIRLEnv):
    cfg: Team1DefendFittedEnvCfg

    def _ensure_fitted(self) -> None:
        if hasattr(self, "_rc_held"):
            return
        n = self.num_envs
        self._rc_held = {a: torch.zeros(n, 4, device=self.device) for a in self.cfg.possible_agents}
        self._rc_next = torch.zeros(n, device=self.device)  # [s] episode time of the next rc packet
        low, high = self.cfg.rc_loss_range
        self._rc_loss = low + (high - low) * torch.rand(n, device=self.device)
        self._delay_steps = round(self.cfg.obs_delay_s / self.step_dt)
        drones = self.cfg.num_defenders + self.cfg.num_attackers
        self._pos_hist = torch.zeros(self._delay_steps + 1, n, drones, 3, device=self.device)
        self._hist_head = 0

    # -- command path -------------------------------------------------------------------------
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Sample the controller at the rc rate, as integer sticks; a lost packet holds the last one."""
        self._ensure_fitted()
        now = self.episode_length_buf.float() * self.step_dt
        due = now >= self._rc_next
        take = due & (torch.rand(self.num_envs, device=self.device) >= self._rc_loss)
        for agent, action in actions.items():
            rc = (action.clamp(-1.0, 1.0) * 100.0).round()
            rc = torch.where(rc.abs() < self.cfg.rc_deadband, torch.zeros_like(rc), rc)
            self._rc_held[agent] = torch.where(take[:, None], rc / 100.0, self._rc_held[agent])
        period = 1.0 / self.cfg.rc_hz + self.cfg.rc_jitter_s * torch.rand(self.num_envs, device=self.device)
        self._rc_next = torch.where(due, self._rc_next + period, self._rc_next)
        super()._pre_physics_step({a: self._rc_held[a] for a in actions})

    # -- fix age ------------------------------------------------------------------------------
    def _update_estimates(self) -> None:
        """Record this step's true positions, then let the filter fuse the delayed ones."""
        self._ensure_estimator()
        self._ensure_fitted()
        if self._kf_seen_tick != self._kf_tick:  # once per env step, as the mixin itself does
            self._hist_head = (self._hist_head + 1) % self._pos_hist.shape[0]
            self._pos_hist[self._hist_head] = EstimatedStateMixin._true_positions(self)
        super()._update_estimates()

    def _true_positions(self) -> torch.Tensor:
        """Positions as old as a real fix: obs_delay_s ago, or the episode start if that is sooner."""
        age = torch.clamp(self.episode_length_buf, max=self._delay_steps)
        slot = (self._hist_head - age) % self._pos_hist.shape[0]
        return self._pos_hist[slot, torch.arange(self.num_envs, device=self.device)]

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)
        self._ensure_fitted()
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for agent in self._rc_held:
            self._rc_held[agent][ids] = 0.0
        self._rc_next[ids] = 0.0
        low, high = self.cfg.rc_loss_range
        self._rc_loss[ids] = low + (high - low) * torch.rand(len(ids), device=self.device)


@configclass
class Team1DefendFittedSpawn6EnvCfg(Team1DefendFittedEnvCfg):
    """Ablation: Fitted with the official 6 m spawns (the 3 m ones leave as little as 1.5 m to the asset)."""

    spawn_radius_range = [6.0, 6.0]


@configclass
class Team1DefendFittedSlowEnvCfg(Team1DefendFittedEnvCfg):
    """Ablation: Fitted with attackers at 0.2-0.5 m/s (the hosts' formation attack flies ~0.2-0.52 m/s at k=0.5)."""

    attacker_speed = [0.2, 0.5]
