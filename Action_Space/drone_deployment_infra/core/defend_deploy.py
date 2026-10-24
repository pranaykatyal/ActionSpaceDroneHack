"""Deploy-side layer around the defend policy: latency prediction and teammate separation.

The SAME file runs on the laptop (numpy) and in Isaac Sim (torch, every env at once), through
the ``xp`` module passed in, like positional_kalman_filter.py. The sim loads it by path
(``team1_defend/team1_defend_deploy_env.py``), so the wrappers scored in sim are the ones flown.

Both sit outside the network: the skrl policy, its 23-dim observation and its
[vx, vy, vz, yaw_rate] action are unchanged.

- ``predict_obs``: a command reaches the motors ``delay`` after it is sent, so the policy should
  decide on the state at that moment. Moves the observation forward by the delay (plus the fix
  age, if the position is stale): own motion from the commands already in flight, attackers at
  constant velocity.
- ``separate``: teammate-only separation filter on the velocity commands, a second-order control
  barrier function on defender-defender distance that accounts for the velocity lag. Contact
  with attackers is the capture in real flights (hosts), so attackers are never avoided.

Not a flight runner: there is no defend flight loop on the laptop yet. One would, every tick:
``predict_obs`` on each defender's observation -> policy -> ``separate`` on the world-frame
velocity commands -> rotate into each drone's heading frame -> rc. ``separate`` handles any
number of defenders (the real demo flies 3).

Observation layout (tello_defend_env.py, A attackers): own position rel. asset [0:3], own
velocity [3:6], projected gravity [6:9], attacker rel. positions [9:9+3A], rel. velocities
[9+3A:9+6A], alive flags [9+6A:9+7A]. Downed attackers are zeroed.

Self-test: ``python3 drone_deployment_infra/core/defend_deploy.py`` (pass: ``SELFTEST PASSED``).
Only numpy is imported at module level.
"""

from __future__ import annotations

import math

import numpy as np

CLIP_POS = 12.0  # [m] obs_clip_pos of tello_defend_env_cfg.py
CLIP_VEL = 3.0  # [m/s] obs_clip_vel


def num_attackers(obs_dim: int) -> int:
    return (obs_dim - 9) // 7


def _lag(v, u, dt: float, tau: float, max_accel: float, xp):
    """One step of a first-order velocity lag towards ``u`` (exact discretisation), accel-clamped."""
    step = (u - v) * (1.0 - math.exp(-dt / max(tau, 1e-6)))
    return v + xp.clip(step, -max_accel * dt, max_accel * dt)


def own_motion(vel, cmd_hist, steps: int, dt: float, mode: str = "cv", tau: float = 0.3,
               max_accel: float = 4.0, xp=np):
    """Own displacement and velocity ``steps`` policy steps ahead.

    ``cmd_hist[..., k, :]`` is the world-frame velocity command [m/s] sent k+1 steps ago (index 0 is
    the newest). With the horizon equal to fix age + command delay, the command acting during
    step m of the horizon is ``cmd_hist[..., steps - 1 - m, :]``; commands newer than that are not
    acting yet, and past the history the oldest one is held.
    """
    if mode == "cv":
        return vel * (steps * dt), vel
    if mode != "model":
        raise ValueError(f"unknown predict mode {mode!r}")
    v, disp = vel, 0.0 * vel
    last = cmd_hist.shape[-2] - 1
    for m in range(steps):
        u = cmd_hist[..., min(steps - 1 - m, last), :]
        v_new = _lag(v, u, dt, tau, max_accel, xp)
        disp = disp + 0.5 * (v + v_new) * dt
        v = v_new
    return disp, v


def predict_obs(obs, cmd_hist, steps: int, dt: float, mode: str = "cv", tau: float = 0.3,
                max_accel: float = 4.0, xp=np):
    """The observation ``steps`` policy steps ahead (fix age + command delay), same layout."""
    if steps <= 0:
        return obs
    A = num_attackers(obs.shape[-1])
    lead = tuple(obs.shape[:-1])
    own_pos, own_vel = obs[..., 0:3], obs[..., 3:6]
    rel_pos = obs[..., 9:9 + 3 * A].reshape(lead + (A, 3))
    rel_vel = obs[..., 9 + 3 * A:9 + 6 * A].reshape(lead + (A, 3))
    alive = obs[..., 9 + 6 * A:9 + 7 * A][..., None]
    disp, vel = own_motion(own_vel, cmd_hist, steps, dt, mode, tau, max_accel, xp)
    att_vel = rel_vel + own_vel[..., None, :]  # attackers hold their velocity
    new_rel_pos = rel_pos + att_vel * (steps * dt) - disp[..., None, :]
    new_rel_vel = att_vel - vel[..., None, :]
    out = obs * 1.0
    out[..., 0:3] = own_pos + disp
    out[..., 3:6] = vel
    out[..., 9:9 + 3 * A] = (xp.clip(new_rel_pos, -CLIP_POS, CLIP_POS) * alive).reshape(lead + (3 * A,))
    out[..., 9 + 3 * A:9 + 6 * A] = (xp.clip(new_rel_vel, -CLIP_VEL, CLIP_VEL) * alive).reshape(lead + (3 * A,))
    return out


def separate(pos, vel, cmd, d_min: float = 0.3, lam: float = 3.0, tau: float = 0.3,
             active_dist: float = 1.5, max_speed: float | None = None, weight=None, xp=np):
    """Teammate separation on world-frame velocity commands [m/s], shapes (..., N, 3).

    Per pair, with h = distance - d_min, s = its rate and the commanded rate along the line of
    centres reached through a first-order lag tau, it enforces h'' + 2 lam h' + lam^2 h >= 0
    (critically damped approach to d_min), i.e. commanded rate >= s - tau (2 lam s + lam^2 h).
    The shortfall is split between the pair by ``weight`` (default equal), along the line of
    centres only, so motion across it is untouched. Pass positions and velocities at the time
    the command takes effect (see ``predict_obs``).
    """
    cmd = cmd * 1.0
    N = cmd.shape[-2]
    for i in range(N):
        for j in range(i + 1, N):
            r = pos[..., i, :] - pos[..., j, :]
            dist = (r * r).sum(-1) ** 0.5
            n = r / xp.clip(dist, 1e-6, None)[..., None]
            h = dist - d_min
            s = ((vel[..., i, :] - vel[..., j, :]) * n).sum(-1)
            u_rel = ((cmd[..., i, :] - cmd[..., j, :]) * n).sum(-1)
            need = s - tau * (2.0 * lam * s + lam * lam * h)
            short = xp.where(dist < active_dist, xp.clip(need - u_rel, 0.0, None), 0.0 * dist)
            if weight is None:
                wi = 0.5
            else:
                wi = weight[..., i] / xp.clip(weight[..., i] + weight[..., j], 1e-6, None)
            cmd[..., i, :] = cmd[..., i, :] + (short * wi)[..., None] * n
            cmd[..., j, :] = cmd[..., j, :] - (short * (1.0 - wi))[..., None] * n
    if max_speed is not None:
        cmd = xp.clip(cmd, -max_speed, max_speed)
    return cmd


# -- self-test ---------------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    dt, tau, acc, A = 1 / 60, 0.3, 4.0, 2

    def make_obs(p, v, pa, va, alive):
        rel_p, rel_v = (pa - p) * alive[:, None], (va - v) * alive[:, None]
        return np.concatenate([p, v, [0, 0, -1], rel_p.ravel(), rel_v.ravel(), alive])

    # 1) predictor: a defender through the same lag, commands acting D steps after they are sent,
    #    fixes A_age steps old. The prediction at t must equal the truth at t + D.
    for D, age in [(12, 0), (6, 12), (1, 0)]:
        T = 200
        cmds = rng.uniform(-0.75, 0.75, (T, 3))
        p, v = np.zeros((T + 1, 3)), np.zeros((T + 1, 3))
        for k in range(T):  # the command acting during step k was sent at step k - D
            u = cmds[k - D] if k - D >= 0 else np.zeros(3)
            v[k + 1] = _lag(v[k], u, dt, tau, acc, np)
            p[k + 1] = p[k] + 0.5 * (v[k] + v[k + 1]) * dt
        pa0, va = rng.uniform(-3, 3, (A, 3)), rng.uniform(-1, 1, (A, 3))
        alive = np.array([1.0, 0.0])
        t = 150  # now: commands up to step t-1 sent; the fix shows step t - age
        hist = np.stack([cmds[t - 1 - k] for k in range(D + age)])  # index 0 = newest
        seen = t - age
        obs = make_obs(p[seen], v[seen], pa0 + va * seen * dt, va, alive)
        want = make_obs(p[t + D], v[t + D], pa0 + va * (t + D) * dt, va, alive)
        got = predict_obs(obs, hist, D + age, dt, "model", tau, acc)
        err = np.abs(got - want).max()
        assert err < 1e-9, f"predictor (D={D}, age={age}) off by {err}"
        cv_err = np.abs(predict_obs(obs, hist, D + age, dt, "cv") - want).max()
        assert cv_err > 1e-4, "cv should differ from the exact model under changing commands"
        assert np.all(got[9 + 3:9 + 6] == 0) and np.all(got[9 + 9:9 + 12] == 0), "downed slot must stay zero"
    print("[ok] predictor matches the lagged, delayed defender exactly (D=12; D=6 + 12 steps fix age; D=1)")

    # 2) separation: two defenders commanded straight at each other at 0.75 m/s each from 2 m apart
    def fly(filtered: bool) -> float:
        pos = np.array([[0.0, 0, 1.5], [2.0, 0, 1.5]])
        vel = np.zeros((2, 3))
        want = np.array([[0.75, 0, 0], [-0.75, 0, 0]])
        closest = 9.0
        for _ in range(600):
            cmd = separate(pos, vel, want, tau=tau, max_speed=0.75) if filtered else want
            vel = _lag(vel, cmd, dt, tau, acc, np)
            pos = pos + vel * dt
            closest = min(closest, float(np.linalg.norm(pos[0] - pos[1])))
        return closest

    raw, safe = fly(False), fly(True)
    assert raw < 0.05 and safe > 0.25, f"head-on: closest {raw:.3f} m unfiltered, {safe:.3f} m filtered"
    print(f"[ok] head-on at 1.5 m/s closing: closest {raw:.2f} m unfiltered -> {safe:.2f} m filtered (d_min 0.30)")

    # 3) no change when far apart, or when moving side by side
    pos = np.array([[0.0, 0, 1.5], [0.6, 0, 1.5]])
    vel = np.array([[0.0, 0.5, 0], [0.0, 0.5, 0]])
    side = np.array([[0.0, 0.5, 0], [0.0, 0.5, 0]])
    assert np.allclose(separate(pos, vel, side), side), "side by side must be untouched"
    far = np.array([[0.0, 0, 1.5], [3.0, 0, 1.5]])
    head = np.array([[0.75, 0, 0], [-0.75, 0, 0]])
    assert np.allclose(separate(far, np.zeros((2, 3)), head), head), "beyond active_dist must be untouched"
    three = separate(np.array([[0.0, 0, 1.5], [0.4, 0, 1.5], [0.2, 0.35, 1.5]]), np.zeros((3, 3)), np.zeros((3, 3)))
    assert np.isfinite(three).all(), "three defenders"
    print("[ok] side-by-side and far-apart commands unchanged; three defenders handled")

    # 4) torch gives the same numbers, if it is installed (the sim runs this file on torch)
    try:
        import torch
    except ImportError:
        print("[skip] torch not installed: numpy only")
    else:
        o = rng.normal(size=(5, 23)); o[:, 21:] = rng.integers(0, 2, (5, 2))
        h = rng.normal(size=(5, 12, 3))
        for mode in ("cv", "model"):
            a = predict_obs(o, h, 12, dt, mode, tau, acc)
            b = predict_obs(torch.tensor(o), torch.tensor(h), 12, dt, mode, tau, acc, xp=torch).numpy()
            assert np.allclose(a, b), f"torch predict_obs ({mode}) differs"
        P, V, C = rng.normal(size=(3, 5, 2, 3)) * 0.4
        a = separate(P, V, C, max_speed=0.75)
        b = separate(*(torch.tensor(x) for x in (P, V, C)), max_speed=0.75, xp=torch).numpy()
        assert np.allclose(a, b), "torch separate differs"
        print("[ok] torch matches numpy")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    _selftest()
