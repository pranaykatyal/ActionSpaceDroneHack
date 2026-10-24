"""The official 3v2 task with the submission's real-drone runner (final_submission/policy/defend.py) flying the defenders.

defend.py is Mo's runner with Team 1's intercept guard: per drone an ``Eye`` (pose and velocity smoothing,
ToF height), the fallback chaser, the guard (settings from defend_checkpoint.pt via ``load_guard_settings``),
the defender separation push, the geofence and ``to_rc``. On the real drones it runs inside
swarm_policies.py; here its ``DefendPolicy`` runs unchanged except for the parts that need hardware:

  cameras        each Eye is fed the drone's simulated position once per runner tick (one fresh fix per
                 tick), then smoothed exactly as Eye does (POSE_ALPHA, VEL_ALPHA, ToF height = true Z).
                 No localization noise: the official task has none. A respawned or reset drone gets a
                 fresh Eye (real drones don't teleport, so the jump filter would only add a 1 s blackout).
  attackers      the hosts' scripted attackers and referee (0.35 m capture, 1.5 m site, waves) replace
                 the runner's own 0.3 m/s attackers and 0.45 m referee; the runner sees which attackers
                 are alive, as its referee would tell it. A new wave is a new GO (t0, fallback timer).
  tag-6 line     the official site is a 1.5 m cylinder that attackers approach from any side, so the
                 fallback's "within FALLBACK_BEFORE_M of the line" becomes "within FALLBACK_BEFORE_M of
                 the site's edge" (radial distance), most advanced = closest to the site.
  room box       the geofence box is the open arena around the 6 m spawn ring, floor 0.15 m, top 3.5 m
                 (attackers fly at 1.5-3.0 m; the room's 1.78 m ceiling would keep defenders below them).
  rc             the runner's rc sticks become velocities through its own stick model (GAIN per rc unit
                 past RC_DEADBAND), then the official action (velocity / max_lin_speed). Yaw is held at
                 the runner's heading (psi = 0, yaw rc 0), as yaw is locked in the sim.
  tick           the runner ticks every PERIOD_S (74 ms, 13.5 Hz) and its rc is held between ticks.

Scored with the hosts' unmodified ``scripts/evaluate.py --task Team1-Defend-3v2-MoRunner-v0 --baseline zero``
(the zero actions are replaced by the runner's) and recorded with ``scripts/record_defend.py --pilot zero``.
``TEAM1_MO_RUNNER`` points at another defend.py; ``DEFEND_FALLBACK=0`` turns its chaser off, as on the drones.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils import configclass

from ..tello_defend.tello_defend_env import TelloDefendEnv
from .team1_defend_3v2_env_cfg import Team1Defend3v2EnvCfg

_REPO = Path(__file__).resolve().parents[7]
DEFAULT_RUNNER = str(_REPO / "final_submission" / "policy" / "defend.py")


@configclass
class Team1Defend3v2MoRunnerEnvCfg(Team1Defend3v2EnvCfg):
    runner_file: str = os.environ.get("TEAM1_MO_RUNNER", DEFAULT_RUNNER)
    """The submission's defend.py; its defend_checkpoint.pt is read from next to it (or $DEFEND_CHECKPOINT)."""
    arena_half_width = 7.0  # [m] geofence box around the site: the 6 m spawn ring plus 1 m
    arena_z = [0.15, 3.5]  # [m] geofence floor and top
    runner_log_env = 0  # env whose runner prints its fallback switches (-1: none)


# the same defaults for cfgs without the runner fields (configclass makes them instance fields, not class attributes)
RUNNER_DEFAULTS = {"runner_file": os.environ.get("TEAM1_MO_RUNNER", DEFAULT_RUNNER), "arena_half_width": 7.0,
                   "arena_z": [0.15, 3.5], "runner_log_env": 0}


def load_runner(path: str):
    """Import defend.py as a module. It does ``from .base import Policy``, so it gets a package of its
    own with swarm_ops' Policy interface (setup / step / describe; the hardware-free part of base.py)."""
    pkg_name = "_team1_submission_runner"
    if pkg_name + ".defend" in sys.modules:
        return sys.modules[pkg_name + ".defend"]
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [os.path.dirname(os.path.abspath(path))]
    base = types.ModuleType(pkg_name + ".base")

    class Policy:
        name = "policy"

        def setup(self, drones):
            self.drones = list(drones)

        def step(self, d, t):
            raise NotImplementedError

        def describe(self, d):
            return self.name

    base.Policy = Policy
    sys.modules[pkg_name] = pkg
    sys.modules[pkg_name + ".base"] = base
    spec = importlib.util.spec_from_file_location(pkg_name + ".defend", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name + ".defend"] = module
    spec.loader.exec_module(module)
    return module


def make_sim_classes(defend):
    """The runner's Eye and DefendPolicy with the hardware parts swapped for the sim (see module docstring)."""

    class SimDrone:
        def __init__(self, num, name):
            self.num, self.name, self.z = num, name, None

        def height_cm(self):  # the Tello's ToF: true height above the floor
            return None if self.z is None else 100.0 * self.z

    class SimEye(defend.Eye):
        def __init__(self, d):
            super().__init__(d, None, None, None, None, None, None)

        def start(self):
            pass

        def feed(self, P, t):
            """One tag fix at room position P (X, Y, Z), smoothed as Eye.perception_loop does."""
            self.d.z = float(P[2])
            p = np.array([P[1], defend.FLOOR_M - P[2], -P[0]])  # inverse of to_room (swarm_formation frame)
            psi = 0.0
            if self.fix is not None and t - self.fix[2] < defend.FIX_STALE_S:
                p = defend.POSE_ALPHA * p + (1 - defend.POSE_ALPHA) * self.fix[0]
            self.fix, self.seen_t, self.reason = (p, psi, t), t, "sim"

    class SimDefendPolicy(defend.DefendPolicy):
        def __init__(self, guard_cfg, site_radius, arena, verbose):
            self.drones = [SimDrone(n, f"defender_{n}") for n in range(3)] + [SimDrone(3 + k, f"attacker_{k}") for k in range(2)]
            self.defenders, self.attackers = self.drones[:3], self.drones[3:]
            self.guard_cfg, self.site_radius, self.arena, self.verbose = guard_cfg, site_radius, arena, verbose
            self.site = np.zeros(3)  # the protected zone stands on the floor at the env origin
            self.breach_x = 0.0  # progress() is measured from the site's edge
            self.phase, self.result, self.heights = "GO", None, {}
            self.new_episode(0.0)

        def new_episode(self, now):
            self.eyes = {d.num: SimEye(d) for d in self.drones}
            self.new_wave(now)
            self.cmds = {}

        def new_wave(self, now):
            for a in self.attackers:
                self.eyes[a.num] = SimEye(a)
            self.alive, self.t0 = [True, True], now
            self.guard, self.guard_alive, self.vguard, self.targets, self._chasing = None, None, {}, {}, {}

        def progress(self, P):
            """The fallback's attacker X, for a site approached from any side: 0 on the site's edge."""
            return self.breach_x - (float(np.linalg.norm(P[:2] - self.site[:2])) - self.site_radius)

        def _chasers(self, now):
            """DefendPolicy._chasers with the attacker's X replaced by progress()."""
            eyes = self.eyes
            trig = [k for k, a in enumerate(self.attackers) if self.alive[k]
                    and (self.progress(eyes[a.num].P) > self.breach_x - defend.FALLBACK_BEFORE_M
                         or now - self.t0 > defend.FALLBACK_T)]
            trig.sort(key=lambda k: -self.progress(eyes[self.attackers[k].num].P))  # most advanced first
            out, free = {}, list(self.defenders)
            for k in trig:
                if not free:
                    break
                ak = eyes[self.attackers[k].num].P
                d = min(free, key=lambda o: float(np.linalg.norm(eyes[o.num].P - ak)))
                out[d.num] = k
                free.remove(d)
            if out != getattr(self, "_chasing", {}):
                for n, k in out.items():
                    if self.verbose and getattr(self, "_chasing", {}).get(n) != k:
                        print(f"[runner] FALLBACK: defender_{n} chases attacker_{k} "
                              f"({-self.progress(eyes[self.attackers[k].num].P):.2f} m from the site, "
                              f"{now - self.t0:.1f} s into the wave)", flush=True)
                self._chasing = dict(out)
            return out

        def tick(self, now, alive):
            """The defender half of DefendPolicy._tick (the attacker half is the hosts' task)."""
            self.alive = list(alive)
            eyes = self.eyes
            for e in eyes.values():
                e.update(now)
            cmds = {}
            chasers = self._chasers(now) if defend.FALLBACK else {}
            self._guard(now)
            for i, d in enumerate(self.defenders):
                e = eyes[d.num]
                if self.cmds.get(d.num, 0) is None or now - e.seen_t > defend.LOST_LAND_S:
                    cmds[d.num] = None
                    continue
                if not e.has_fix(now):
                    cmds[d.num] = (0, 0, 0, 0)
                    continue
                k = chasers.get(d.num)
                if k is not None:
                    v = self._chase(e, eyes[self.attackers[k].num])
                else:
                    if d.num not in self.vguard:  # no attacker left: hold
                        cmds[d.num] = (0, 0, 0, 0)
                        continue
                    v = self.vguard[d.num].copy()
                for o in self.defenders:
                    gap = e.P - eyes[o.num].P
                    dist = float(np.linalg.norm(gap))
                    if o is not d and 1e-6 < dist < defend.SEPARATION_M:
                        v += defend.SEPARATION_MPS * gap / dist
                cmds[d.num] = defend.to_rc(self._fence(e.P, v), e.psi, defend.MAX_RC_DEFENDER)
            self.cmds = cmds
            return [cmds[d.num] for d in self.defenders]

    return SimDefendPolicy


def rc_to_room(defend, rc, psi_deg):
    """Inverse of defend.to_rc: rc sticks -> room-frame velocity, with the runner's own stick model."""
    if rc is None:  # land: the sim has no landing, hover instead
        return np.zeros(3)
    right, fwd, up, _ = rc

    def speed(stick, gain):
        return 0.0 if stick == 0 else math.copysign(max(abs(stick) - defend.RC_DEADBAND, 0) * gain, stick)

    r, f, u = speed(right, defend.GAIN["right"]), speed(fwd, defend.GAIN["forward"]), speed(up, defend.GAIN["up"])
    s, c = math.sin(math.radians(psi_deg)), math.cos(math.radians(psi_deg))
    vx_i, vz_i = f * s + r * c, f * c - r * s
    return np.array([-vz_i, vx_i, u])


class MoRunnerMixin:
    """Left of the env class: ``class X(MoRunnerMixin, SomeDefendEnv)``, or right of RoomGeometryMixin so a
    defender that hit a wall still drops. Settings come from the cfg when it has them (Team1Defend3v2MoRunnerEnvCfg),
    else the same defaults, so any 3v2 defend cfg works. In a room the geofence is the room's walls."""

    def _runner_setting(self, key: str):
        return getattr(self.cfg, key, RUNNER_DEFAULTS[key])

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        assert self.cfg.num_defenders == 3 and self.cfg.num_attackers == 2, "defend.py flies 3 defenders vs 2 attackers"
        runner_file = self._runner_setting("runner_file")
        self._defend = load_runner(runner_file)
        d = self._defend
        guard_cfg = dict(d.load_guard_settings(), speed=d.MAX_HSPEED_MPS)
        print(f"[runner] {runner_file}: fallback {'on' if d.FALLBACK else 'off'}, guard {guard_cfg}", flush=True)
        w, z = self._runner_setting("arena_half_width"), tuple(self._runner_setting("arena_z"))
        arena = ((-w, w), (-w, w), z)
        if hasattr(self, "_ensure_room"):  # room task: the walls and ceiling, in env-origin (= asset centre) coordinates
            self._ensure_room()
            lo = (self._room_lo - self._room_site).tolist()
            hi = (self._room_hi - self._room_site).tolist()
            arena = ((lo[0], hi[0]), (lo[1], hi[1]), (z[0], hi[2]))
        print(f"[runner] geofence box {tuple(tuple(round(v, 2) for v in a) for a in arena)}", flush=True)
        sim_policy = make_sim_classes(d)
        log_env = self._runner_setting("runner_log_env")
        self._runners = [
            sim_policy(guard_cfg, self.cfg.site_radius, arena, verbose=(e == log_env))
            for e in range(self.num_envs)
        ]
        self._runner_vel = torch.zeros(self.num_envs, self.cfg.num_defenders, 3, device=self.device)
        self._runner_clock = d.PERIOD_S  # tick on the first step

    def _env_time(self, env_id: int) -> float:
        return float(self.episode_length_buf[env_id]) * self.step_dt

    def _runner_tick(self) -> None:
        origins = self.scene.env_origins
        def_pos = (torch.stack([x.data.root_pos_w for x in self._defenders], dim=1) - origins.unsqueeze(1)).cpu().numpy()
        att_pos = (torch.stack([x.data.root_pos_w for x in self._attackers], dim=1) - origins.unsqueeze(1)).cpu().numpy()
        alive = self._attacker_alive.cpu().numpy()
        times = (self.episode_length_buf.float() * self.step_dt).cpu().numpy()
        vel = np.zeros((self.num_envs, self.cfg.num_defenders, 3))
        for e, runner in enumerate(self._runners):
            now = float(times[e])
            for i, dr in enumerate(runner.defenders):
                runner.eyes[dr.num].feed(def_pos[e, i], now)
            for k, a in enumerate(runner.attackers):
                runner.eyes[a.num].feed(att_pos[e, k], now)
            for i, rc in enumerate(runner.tick(now, alive[e])):
                vel[e, i] = rc_to_room(self._defend, rc, runner.eyes[runner.defenders[i].num].psi)
        self._runner_vel = torch.tensor(vel, device=self.device, dtype=torch.float32)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        super()._pre_physics_step(actions)
        self._runner_clock += self.step_dt
        if self._runner_clock >= self._defend.PERIOD_S - 1e-9:
            self._runner_clock -= self._defend.PERIOD_S
            self._runner_tick()
        # the runner flies the defenders; the policy's actions (zero for evaluate.py --baseline zero) are ignored
        scale = self.cfg.command.max_lin_speed
        for i in range(self.cfg.num_defenders):
            action = torch.zeros(self.num_envs, 4, device=self.device)
            action[:, :3] = self._runner_vel[:, i] / scale
            self._actions[f"defender_{i}"] = action.clamp(-1.0, 1.0)

    def _respawn_attackers(self, env_ids: torch.Tensor) -> None:
        super()._respawn_attackers(env_ids)
        for e in env_ids.tolist():
            self._runners[e].new_wave(self._env_time(e))

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_runners"):
            return
        ids = range(self.num_envs) if env_ids is None else (env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids))
        for e in ids:
            self._runners[e].new_episode(0.0)
        self._runner_vel[list(ids)] = 0.0


class Team1DefendMoRunnerEnv(MoRunnerMixin, TelloDefendEnv):
    """The official 3v2 task (hosts' attackers) flown by the runner."""

    cfg: Team1Defend3v2MoRunnerEnvCfg


def _other_tasks():
    """Runner versions of the other 3v2 tasks (imported lazily: those modules pull in more of the kit)."""
    from .team1_defend_room_env import RoomGeometryMixin
    from .team1_defend_shaped_env import Team1DefendRealisticEnv

    class Team1DefendMoRunnerRealisticEnv(MoRunnerMixin, Team1DefendRealisticEnv):
        """Weaving attackers on the official geometry, flown by the runner."""

    class Team1DefendMoRunnerRoomEnv(RoomGeometryMixin, MoRunnerMixin, TelloDefendEnv):
        """The room (official attackers), flown by the runner; a defender that hits a wall still drops."""

    class Team1DefendMoRunnerRoomRealisticEnv(RoomGeometryMixin, MoRunnerMixin, Team1DefendRealisticEnv):
        """The room with weaving attackers, flown by the runner."""

    return Team1DefendMoRunnerRealisticEnv, Team1DefendMoRunnerRoomEnv, Team1DefendMoRunnerRoomRealisticEnv


Team1DefendMoRunnerRealisticEnv, Team1DefendMoRunnerRoomEnv, Team1DefendMoRunnerRoomRealisticEnv = _other_tasks()
