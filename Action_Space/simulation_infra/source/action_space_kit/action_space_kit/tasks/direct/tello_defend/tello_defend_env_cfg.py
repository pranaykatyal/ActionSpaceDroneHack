"""Configuration for the multi-agent defend task: the hackathon scenario."""

from __future__ import annotations

import gymnasium as gym

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from action_space_kit.control import TelloCommandCfg

NUM_DEFENDERS = 2
NUM_ATTACKERS = 2

# obs per defender: own pos rel. site (3) + own vel (3) + gravity (3)
#                 + per attacker (rel pos 3, rel vel 3, visible flag 1)
_OBS_DIM = 9 + 7 * NUM_ATTACKERS


@configclass
class TelloDefendEnvCfg(DirectMARLEnvCfg):
    """Defenders intercept attackers before they reach the protected site.

    Attackers are scripted by default (``trainable_attackers = False``), which is the
    fastest path to a working defender policy. Set it True to train both sides.
    """

    # env
    decimation = 2
    episode_length_s = 20.0

    # multi-agent spaces -- every agent uses the four SDK rc channels
    possible_agents = [f"defender_{i}" for i in range(NUM_DEFENDERS)]
    # Bounded on purpose: the four SDK rc channels live in [-1, 1], and an unbounded
    # space leaves skrl nothing to clip sampled actions against.
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}
    state_space = -1  # concatenate agent observations

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene: 1024 envs x 4 drones fits comfortably on a 24 GB GPU shared by several teams
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=12.0, replicate_physics=True)

    # Tello SDK velocity interface (shared by both sides)
    command: TelloCommandCfg = TelloCommandCfg()

    # - scenario geometry
    num_defenders = NUM_DEFENDERS
    num_attackers = NUM_ATTACKERS
    site_radius = 1.5  # [m] protected perimeter around the env origin
    site_height = 2.5  # [m] height of the protected cylinder
    spawn_radius = 6.0  # [m] where attackers enter from
    defender_radius = 2.5  # [m] where defenders start, around the site
    capture_radius = 0.35  # [m] distance counting as an interception
    obs_clip_pos = 12.0  # [m] relative positions are clipped to this before normalisation
    obs_clip_vel = 3.0  # [m/s] relative velocities are clipped to this
    trainable_attackers = False

    # - scripted attacker behaviour (curriculum stage 0 = straight dive)
    attacker_speed = [0.4, 0.9]  # [m/s]
    attacker_evasion = 0.0  # 0 = straight line, 1 = strong lateral weave

    # - reward scales (per defender)
    # Balance matters here: a breach happens in nearly every early episode, so a huge
    # penalty drowns the shaping terms and training drifts below a do-nothing baseline
    # (measured: -45.8 return for zero actions). Keep the penalty comparable to what a
    # defender can earn by doing its job.
    rew_scale_closing = 5.0  # getting nearer its target attacker
    rew_scale_proximity = 2.0  # continuous pull towards the nearest attacker
    rew_scale_capture = 30.0  # intercepting one
    rew_scale_cleared = 30.0  # bonus for downing the last attacker of a wave
    respawn_waves = True
    """Launch a fresh wave when the field is cleared, instead of ending the episode.

    Ending on success truncates the value bootstrap, which makes a short successful
    episode look worse than a long passive one -- measured as a policy that started at
    +137 return and unlearned its way to -260 while episodes grew longer.
    """
    rew_scale_breach = -20.0  # an attacker reaching the site (shared)
    rew_scale_effort = -0.01
    rew_scale_alive = 0.1

    # - retrying attackers (opt-in; off keeps the wave behaviour above)
    attacker_retry = False
    """After an interception each attacker independently drops out for the rest of the wave,
    flies back to the spawn ring and waits, or holds where it was caught and waits, then
    attacks again. The choice is random per capture because the real rules are unknown.
    Downed attackers stay in the observation at their real position instead of being zeroed,
    so defenders can keep track of a drone that may come back. A new wave starts only once
    every attacker is out."""
    retry_mode_probs = [0.25, 0.35, 0.4]  # out, return to the ring, hold in place
    retry_wait_return = [1.0, 6.0]  # [s] wait at the ring before attacking again
    retry_wait_hold = [0.5, 6.0]  # [s] hold where caught before attacking again (quick retry .. linger)
    retry_out_descent = 0.5  # [m/s] sink rate of an attacker that is out
    observe_status = True
    """Per-attacker "attacking" flags in the observation. False zeroes them, for when the
    real game does not say which drones were intercepted (positions are still observed)."""

    # - reward fixes (opt-in)
    fix_reward_glitches = False
    """Pay the breach penalty once (rewards run after dones, so it was paid on two steps) and
    measure closing on the same attacker across steps, so target switches and respawns no
    longer add +-5 jumps."""

    # - scripted guard base and residual policy (opt-in, see guard.py)
    attacker_command: TelloCommandCfg | None = None
    """Attacker dynamics, if different from the defenders' ``command`` (None: the same)."""
    defender_base = ""
    """"guard": the action is the scripted interceptor's command plus residual_scale x policy."""
    residual_scale = 0.5
    observe_guard = False
    """Append teammate rel pos/vel and the guard's aim point (rel) to each observation."""
    guard_uses_status = True
    """The guard reads the attacking flags; False infers them from motion, as on real drones."""
    guard_kwargs: dict = {}
    world_frame_force = False
    """Rotate the controller's world-frame force into the body frame before applying it (Isaac
    Lab applies external forces in the body frame). False keeps the stock, buggy behaviour."""


@configclass
class TelloDefendGuardEnvCfg(TelloDefendEnvCfg):
    """Retrying, teleporting, faster attackers against a guard base plus a learned residual.

    15 Hz control matches the measured real rc rate (13.4 Hz) and gives gamma 0.99 a 6.7 s
    horizon instead of 1.7 s at 60 Hz. Defenders get the slower, laggier response measured on
    TELLO-1; attackers keep the snappy default, as the organisers' may be tuned aggressively.
    """

    decimation = 8
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=8)
    command: TelloCommandCfg = TelloCommandCfg(response_time=0.3, command_latency_steps=6)
    attacker_command: TelloCommandCfg = TelloCommandCfg(response_time=0.15)
    attacker_speed = [0.5, 1.0]
    attacker_evasion = 0.3
    attacker_retry = True
    retry_mode_probs = [0.2, 0.25, 0.3, 0.25]  # out, return, hold, teleport
    fix_reward_glitches = True
    defender_base = "guard"
    observe_guard = True
    world_frame_force = True
    guard_kwargs: dict = {"park_radius": 4.0}  # measured best of 3 / 4 / 5 / 8 m
    observation_spaces = {f"defender_{i}": _OBS_DIM + 6 * (NUM_DEFENDERS - 1) + 3 for i in range(NUM_DEFENDERS)}
