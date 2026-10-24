"""Team 1's reward-tuning variants of AS-Defend-v0: a single-defender task and a shaped reward.

Follows the hosts' reward-tuning practice (simulation_infra/SINGLE_DRONE_SIMULATION.md):
tune on one drone first, pay progress as discounted potential shaping, and pay each event once.
Scenario, observations and actions are the official ones, so a policy trained here is scored
on AS-Defend-v0 with the hosts' scripts/evaluate.py.

``reward_version = 2`` replaces the official per-defender reward:

  progress  gamma * Phi(d') - Phi(d), Phi(d) = progress_scale * (1 - min(d, D) / D) >= 0, where d and
            d' are the distance to the SAME attacker (the one nearest at the previous step) before
            and after the step. A capture, a new wave or a target switch causes no jump (the
            official ``closing`` term pays -5 whenever the nearest attacker changes); a breach
            ends the episode with Phi' = 0, so it gives back the progress earned
  capture   + rew_scale_capture per attacker this defender downs
  cleared   + rew_scale_cleared when a wave is cleared
  breach    + rew_scale_breach, once (the official reward pays it on two consecutive steps)
  effort, alive as the official reward; no ``proximity`` (it pays for hovering near attackers)
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import torch

from isaaclab.utils import configclass

from ..tello_defend.tello_defend_env import TelloDefendEnv
from ..tello_defend.tello_defend_env_cfg import TelloDefendEnvCfg


@configclass
class Team1DefendShapedEnvCfg(TelloDefendEnvCfg):
    """Official scenario, shaped reward."""

    reward_version = 2
    progress_scale = 10.0  # Phi at contact; a full 6 m approach earns ~10, a third of a capture
    progress_range = 6.0  # [m] D: Phi is 0 at or beyond this distance
    shaping_gamma = 0.99  # must equal the PPO discount_factor (agent.agent.discount_factor)
    rew_scale_crash = 0.0  # once per defender per episode when it drops below crash_height (0 = off)
    crash_height = 0.3  # [m]


@configclass
class Team1DefendShapedCrashEnvCfg(Team1DefendShapedEnvCfg):
    """Shaped reward plus a crash penalty the size of a breach: pays for not knocking yourself out."""

    rew_scale_crash = -20.0


@configclass
class Team1DefendSingleEnvCfg(TelloDefendEnvCfg):
    """One defender against the official two attackers, official reward (the reward-tuning baseline)."""

    num_defenders = 1
    possible_agents = ["defender_0"]
    action_spaces = {"defender_0": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))}
    observation_spaces = {"defender_0": 9 + 7 * 2}  # same 23-wide layout as AS-Defend-v0
    reward_version = 1
    progress_scale = 10.0
    progress_range = 6.0
    shaping_gamma = 0.99


@configclass
class Team1DefendSingleShapedEnvCfg(Team1DefendSingleEnvCfg):
    """One defender against two attackers, shaped reward."""

    reward_version = 2


@configclass
class Team1DefendRealisticEnvCfg(TelloDefendEnvCfg):
    """Official scenario and reward, attackers that fly like real Tellos (team1_attackers.py)."""

    attacker_weave_prob = 0.7
    attacker_evasion_range = [0.2, 1.0]
    attacker_weave_period = [2.0, 6.0]  # [s]
    attacker_wobble_std = 0.15  # [rad] ~9 degrees
    attacker_wobble_tau = 1.0  # [s]
    attacker_min_height = 0.7  # [m] attackers climb if they get lower than this
    attacker_speed = [0.2, 1.0]  # [m/s] per attacker per wave: wider than the official [0.4, 0.9]
    reward_version = 1
    progress_scale = 10.0
    progress_range = 6.0
    shaping_gamma = 0.99


@configclass
class Team1DefendWeave05EnvCfg(TelloDefendEnvCfg):
    """Official scenario and reward, attackers weave (README curriculum knob): robustness check."""

    attacker_evasion = 0.5
    reward_version = 1
    progress_scale = 10.0
    progress_range = 6.0
    shaping_gamma = 0.99


@configclass
class Team1DefendWeave10EnvCfg(Team1DefendWeave05EnvCfg):
    attacker_evasion = 1.0


class Team1DefendShapedEnv(TelloDefendEnv):
    cfg: Team1DefendShapedEnvCfg | Team1DefendSingleEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ensure_reward_state()

    def _ensure_reward_state(self) -> None:
        if not hasattr(self, "_target"):
            shape = (self.num_envs, self.cfg.num_defenders)
            self._target = torch.full(shape, -1, dtype=torch.long, device=self.device)
            self._target_dist = torch.zeros(shape, device=self.device)
            self._breach_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.round_breaches = torch.zeros(self.num_envs, device=self.device)  # timed-round mode
            self._crashed = torch.zeros(shape, dtype=torch.bool, device=self.device)

    def _potential(self, distance: torch.Tensor) -> torch.Tensor:
        d = distance.clamp(min=0.0, max=self.cfg.progress_range)
        return self.cfg.progress_scale * (1.0 - d / self.cfg.progress_range)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        if self.cfg.reward_version < 2:
            return super()._get_rewards()
        self._ensure_reward_state()
        attacker_pos, _ = self._attacker_states()  # (N, A, 3)
        defender_pos = torch.stack([d.data.root_pos_w for d in self._defenders], dim=1)  # (N, D, 3)
        dist = torch.linalg.norm(attacker_pos[:, None] - defender_pos[:, :, None], dim=-1)  # (N, D, A)
        alive_before = self._attacker_alive.clone()

        super()._get_rewards()  # official bookkeeping: captures, breach, cleared, episode counts

        hit = (dist < self.cfg.capture_radius) & alive_before[:, None, :]
        credit = hit.float().sum(dim=-1)  # (N, D)
        captured = alive_before & ~self._attacker_alive
        cleared = captured.any(dim=-1) & ~self._attacker_alive.any(dim=-1)
        breach_now = self._breached & ~self._breach_paid
        self._breach_paid |= self._breached

        # progress toward the attacker targeted at the previous step
        has_target = self._target >= 0
        d_same = dist.gather(-1, self._target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        phi_prev = self._potential(self._target_dist)
        phi_next = torch.where(breach_now[:, None], torch.zeros_like(phi_prev), self._potential(d_same))
        progress = torch.where(has_target, self.cfg.shaping_gamma * phi_next - phi_prev, torch.zeros_like(phi_prev))

        # next step's target: the nearest live attacker now
        live = self._attacker_alive[:, None, :].expand_as(dist)
        masked = torch.where(live, dist, torch.full_like(dist, float("inf")))
        nearest = masked.min(dim=-1)
        any_live = self._attacker_alive.any(dim=-1, keepdim=True).expand_as(nearest.values)
        self._target = torch.where(any_live, nearest.indices, torch.full_like(nearest.indices, -1))
        self._target_dist = torch.where(any_live, nearest.values, torch.zeros_like(nearest.values))

        # crash: a defender low enough to be on the floor (a knocked-over drone can't recover), once
        height = defender_pos[..., 2] - self.scene.env_origins[:, None, 2]
        crash_now = (height < getattr(self.cfg, "crash_height", 0.3)) & ~self._crashed
        self._crashed |= crash_now

        rewards = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            effort = torch.sum(torch.square(self._actions[agent]), dim=-1)
            rewards[agent] = (
                getattr(self.cfg, "rew_scale_crash", 0.0) * crash_now[:, i].float()
                + progress[:, i]
                + self.cfg.rew_scale_capture * credit[:, i]
                + self.cfg.rew_scale_cleared * cleared.float()
                + self.cfg.rew_scale_breach * breach_now.float()
                + (self.cfg.rew_scale_effort * effort + self.cfg.rew_scale_alive) * self.step_dt
            )
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if getattr(self.cfg, "breach_ends_episode", True):
            return super()._get_dones()
        # a timed round (the final is 5 minutes): a breach is counted and its wave replaced by a
        # fresh one; only the clock ends the episode
        self._ensure_reward_state()
        hit = self._breached.clone()
        if hit.any():
            self.round_breaches += hit.float()
            self._attacker_alive[hit] = False  # the official wave logic then sends a new wave
            self._breached[hit] = False
            self._breach_paid[hit] = False
        return super()._get_dones()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)
        self._ensure_reward_state()
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._target[ids] = -1
        self._target_dist[ids] = 0.0
        self._breach_paid[ids] = False
        self.round_breaches[ids] = 0.0
        self._crashed[ids] = False


from .team1_attackers import RealisticAttackersMixin  # noqa: E402


class Team1DefendRealisticEnv(RealisticAttackersMixin, Team1DefendShapedEnv):
    """Official scenario with realistic attacker paths; reward per ``reward_version``."""


@configclass
class Team1DefendRoundOfficialEnvCfg(TelloDefendEnvCfg):
    """A 5-minute round with the official attackers: breaches are counted, not episode-ending."""

    episode_length_s = 300.0
    breach_ends_episode = False
    reward_version = 1
    progress_scale = 10.0
    progress_range = 6.0
    shaping_gamma = 0.99


@configclass
class Team1DefendRoundRealisticEnvCfg(Team1DefendRealisticEnvCfg):
    """A 5-minute round with realistic attackers (weave, wobble, 0.2-1.0 m/s, never near the ground)."""

    episode_length_s = 300.0
    breach_ends_episode = False
