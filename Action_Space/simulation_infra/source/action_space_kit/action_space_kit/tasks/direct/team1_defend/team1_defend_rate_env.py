"""AS-Defend-v0 with the real Tello's command rate and nothing else changed.

The policy still decides every 1/60 s, but only some decisions reach the drone: like the SDK's ``rc``
packets (about 13.5 Hz in flight logs), each sent command is held until the next one. A phase
accumulator keeps the average rate exact (packets every 4 or 5 policy steps at 13.5 Hz). Scored
with the hosts' ``scripts/evaluate.py --task <id>``; observations and actions are the official layout.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

from action_space_kit.control.tello_command import TelloCommandCfg

from ..tello_defend.tello_defend_env import TelloDefendEnv
from ..tello_defend.tello_defend_env_cfg import TelloDefendEnvCfg


@configclass
class Team1DefendRateEnvCfg(TelloDefendEnvCfg):
    rc_hz = 13.5  # [Hz] commands that reach the drone


@configclass
class Team1DefendRate10EnvCfg(Team1DefendRateEnvCfg):
    rc_hz = 10.0


@configclass
class Team1DefendRateLatEnvCfg(Team1DefendRateEnvCfg):
    """13.5 Hz plus 0.1 s SDK command latency (12 physics steps at 1/120 s)."""

    command: TelloCommandCfg = TelloCommandCfg(command_latency_steps=12)


class Team1DefendRateEnv(TelloDefendEnv):
    cfg: Team1DefendRateEnvCfg

    def __init__(self, cfg: Team1DefendRateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ensure_rate_state()

    def _ensure_rate_state(self) -> None:
        if not hasattr(self, "_rc_held"):
            self._rc_held = {a: torch.zeros(self.num_envs, 4, device=self.device) for a in self.cfg.possible_agents}
            self._rc_since = torch.full((self.num_envs,), 1e9, device=self.device)  # time since the last packet

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self._ensure_rate_state()
        period = 1.0 / self.cfg.rc_hz
        self._rc_since += self.step_dt
        send = self._rc_since >= period - 1e-6
        for agent, action in actions.items():
            self._rc_held[agent] = torch.where(send[:, None], action, self._rc_held[agent])
        # keep the phase so the average rate is exact; a fresh episode sends at once
        self._rc_since = torch.where(send, torch.where(self._rc_since > 2.0 * period, torch.zeros_like(self._rc_since), self._rc_since - period), self._rc_since)
        super()._pre_physics_step({agent: self._rc_held[agent].clone() for agent in actions})

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self._ensure_rate_state()
        ids = self._defenders[0]._ALL_INDICES if env_ids is None else env_ids
        self._rc_since[ids] = 1e9
        for agent in self._rc_held:
            self._rc_held[agent][ids] = 0.0
