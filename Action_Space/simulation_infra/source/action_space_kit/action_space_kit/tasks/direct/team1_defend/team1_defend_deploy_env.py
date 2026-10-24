"""The laptop's deploy-side layer around the defend policy, run in the sim on every env.

Loads ``drone_deployment_infra/core/defend_deploy.py`` (the file a defend flight loop will call)
by path, like ``tags/kalman.py`` loads the Kalman filter, so what is scored here is what flies.
The policy, its 23-dim observation and its action space are unchanged:

- latency predictor (``deploy_predict`` "cv" / "model"): the network sees the observation moved
  forward to when a command sent now reaches the motors (``command_latency_steps``).
- teammate separation filter (``deploy_separation``): the defenders' velocity commands are
  corrected before the env's own command path, from their states at that same moment.

Scored with the hosts' ``scripts/evaluate.py --task <id>``. Not an official score: AS-Defend-v0
has neither layer.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from isaaclab.utils import configclass

from action_space_kit.control.tello_command import TelloCommandCfg
from action_space_kit.tasks.direct.tello_defend.tello_defend_env import TelloDefendEnv
from action_space_kit.tasks.direct.tello_defend.tello_defend_env_cfg import TelloDefendEnvCfg

from .team1_defend_env import Team1DefendRobustRealisticEnv
from .team1_defend_env_cfg import Team1DefendRobustRealisticEnvCfg
from .team1_defend_fitted_env import Team1DefendFittedEnv, Team1DefendFittedEnvCfg

_REPO = Path(__file__).resolve().parents[7]
DEPLOY_PATH = Path(os.environ.get("AS_DEFEND_DEPLOY_PATH", _REPO / "drone_deployment_infra" / "core" / "defend_deploy.py"))


def _load():
    name = "as_defend_deploy"
    if name in sys.modules:
        return sys.modules[name]
    if not DEPLOY_PATH.exists():
        raise FileNotFoundError(f"no deploy layer at {DEPLOY_PATH} (set AS_DEFEND_DEPLOY_PATH)")
    spec = importlib.util.spec_from_file_location(name, DEPLOY_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dd = _load()

# cfg fields read by the mixin, with their defaults (a task cfg sets only what it changes)
DEPLOY_DEFAULTS = {
    "deploy_predict": "off",  # "off" | "cv" | "model": what the network sees
    "deploy_tau": None,  # [s] velocity lag of the "model" predictor and the filter; None: 2 x response_time
    "deploy_extra_s": 0.0,  # [s] look-ahead on top of the command delay: the age of the observed state
    "deploy_separation": False,
    "deploy_sep_predict": "cv",  # the filter's own look-ahead, independent of the network's
    "deploy_sep_d_min": 0.3,  # [m] centre distance the filter holds (Tello box ~0.1 m, plus estimate error)
    "deploy_sep_lam": None,  # [1/s] how hard the approach to d_min is damped; None: 0.9 / tau (3.0 at tau 0.3)
    "deploy_sep_active": 1.5,  # [m] the filter ignores pairs farther apart than this
}


class DeployLayerMixin:
    """Leftmost base: ``class X(DeployLayerMixin, SomeDefendEnv)``."""

    def _dcfg(self, key: str):
        return getattr(self.cfg, key, DEPLOY_DEFAULTS[key])

    def _ensure_deploy(self) -> None:
        if hasattr(self, "_dep_hist"):
            return
        cmd = self.cfg.command
        # policy steps from the observed state until a command sent now acts (the latency buffer runs
        # per physics step; deploy_extra_s adds the age of the observation itself)
        self._dep_steps = round((cmd.command_latency_steps * self.physics_dt + self._dcfg("deploy_extra_s")) / self.step_dt)
        tau = self._dcfg("deploy_tau")
        # TelloCommand lags twice by response_time (target, then force loop): same mean delay as one lag of 2x
        self._dep_tau = 2.0 * cmd.response_time if tau is None else tau
        lam = self._dcfg("deploy_sep_lam")
        self._dep_lam = 0.9 / self._dep_tau if lam is None else lam
        self._dep_agents = [f"defender_{i}" for i in range(self.cfg.num_defenders)]
        self._dep_hist = torch.zeros(self.num_envs, len(self._dep_agents), max(self._dep_steps, 1), 3, device=self.device)

    def _dep_predict(self, obs: torch.Tensor, i: int, mode: str) -> torch.Tensor:
        return dd.predict_obs(obs, self._dep_hist[:, i], self._dep_steps, self.step_dt, mode, self._dep_tau,
                              self.cfg.command.max_lin_accel, xp=torch)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = super()._get_observations()
        self._ensure_deploy()
        if self._dcfg("deploy_separation"):
            # own position (rel. the same asset for both, so differences are exact) and velocity
            # when a command sent now acts
            ahead = [self._dep_predict(obs[a], i, self._dcfg("deploy_sep_predict")) for i, a in enumerate(self._dep_agents)]
            self._dep_pos = torch.stack([o[:, 0:3] for o in ahead], dim=1)
            self._dep_vel = torch.stack([o[:, 3:6] for o in ahead], dim=1)
        mode = self._dcfg("deploy_predict")
        if mode == "off":
            return obs
        return {a: (self._dep_predict(obs[a], self._dep_agents.index(a), mode) if a in self._dep_agents else obs[a]) for a in obs}

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self._ensure_deploy()
        speed = self.cfg.command.max_lin_speed
        actions = {a: act.clamp(-1.0, 1.0) for a, act in actions.items()}
        cmd = torch.stack([actions[a][:, :3] * speed for a in self._dep_agents], dim=1)  # world frame [m/s]
        if self._dcfg("deploy_separation") and hasattr(self, "_dep_pos"):
            cmd = dd.separate(self._dep_pos, self._dep_vel, cmd, d_min=self._dcfg("deploy_sep_d_min"),
                              lam=self._dep_lam, tau=self._dep_tau,
                              active_dist=self._dcfg("deploy_sep_active"), max_speed=speed, xp=torch)
            for i, a in enumerate(self._dep_agents):
                actions[a] = torch.cat([cmd[:, i] / speed, actions[a][:, 3:]], dim=-1)
        # newest first: what the predictor replays as the commands still in flight
        self._dep_hist = torch.cat([cmd[:, :, None], self._dep_hist[:, :, :-1]], dim=2)
        super()._pre_physics_step(actions)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self._ensure_deploy()
        if env_ids is None:
            self._dep_hist.zero_()
        else:
            self._dep_hist[torch.as_tensor(env_ids, device=self.device, dtype=torch.long)] = 0.0


class Team1DefendRRDeployEnv(DeployLayerMixin, Team1DefendRobustRealisticEnv):
    """Robust-Realistic (3-6 m spawns, 0.2 s command delay, inferred asset) with the deploy layer."""


class Team1DefendOfficialDeployEnv(DeployLayerMixin, TelloDefendEnv):
    """The hosts' official scenario with the deploy layer (no command delay: the filter looks ahead 0 s)."""


class Team1DefendFittedDeployEnv(DeployLayerMixin, Team1DefendFittedEnv):
    """Fitted (IRL sensing + the Tello command path: tau 0.5, 0.75 m/s, 0.1 s, 13.5 Hz rc) with the deploy layer."""


@configclass
class Team1DefendRRPredictCVEnvCfg(Team1DefendRobustRealisticEnvCfg):
    """Latency predictor, constant velocity: the network sees every drone 0.2 s ahead."""

    deploy_predict: str = "cv"


@configclass
class Team1DefendRRPredictModelEnvCfg(Team1DefendRobustRealisticEnvCfg):
    """Latency predictor that replays the commands in flight through the velocity lag."""

    deploy_predict: str = "model"


@configclass
class Team1DefendRRSepEnvCfg(Team1DefendRobustRealisticEnvCfg):
    """Teammate separation filter only; the network sees the raw observation."""

    deploy_separation: bool = True


@configclass
class Team1DefendRRPredictCVSepEnvCfg(Team1DefendRobustRealisticEnvCfg):
    """Both: constant-velocity predictor and separation filter."""

    deploy_predict: str = "cv"
    deploy_separation: bool = True


@configclass
class Team1DefendOfficialSepEnvCfg(TelloDefendEnvCfg):
    """Official scenario plus the separation filter: what the filter is worth, not an official score."""

    deploy_separation: bool = True


@configclass
class Team1DefendFittedPredictCVEnvCfg(Team1DefendFittedEnvCfg):
    """Fitted with the constant-velocity predictor over fix age (0.2 s) + mean rc hold + command delay (0.1 s)."""

    deploy_predict: str = "cv"
    deploy_extra_s: float = 0.2 + 0.5 / 13.5  # obs_delay_s + half an rc period


@configclass
class Team1DefendFittedPredictCVSepEnvCfg(Team1DefendFittedPredictCVEnvCfg):
    """Fitted with the predictor and the separation filter."""

    deploy_separation: bool = True


# TelloCommand lags twice by response_time, so response_time 0.5 rises to 63% in 1.07 s; the hosts'
# prior (tello_motion_fit.yaml, rc_action.py: kp_vel = 1 / tau) is ONE lag of 0.5 s. 0.235 gives the
# same 63% rise time (0.5 s).
@configclass
class Team1DefendFittedT024EnvCfg(Team1DefendFittedEnvCfg):
    """Fitted with the hosts' single-lag tau 0.5 s matched (response_time 0.235); no deploy layer."""

    command = TelloCommandCfg(response_time=0.235, max_lin_speed=0.75, command_latency_steps=12)


@configclass
class Team1DefendFittedT024PredictCVEnvCfg(Team1DefendFittedT024EnvCfg):
    deploy_predict: str = "cv"
    deploy_extra_s: float = 0.2 + 0.5 / 13.5


@configclass
class Team1DefendFittedT024PredictCVSepEnvCfg(Team1DefendFittedT024PredictCVEnvCfg):
    deploy_separation: bool = True
