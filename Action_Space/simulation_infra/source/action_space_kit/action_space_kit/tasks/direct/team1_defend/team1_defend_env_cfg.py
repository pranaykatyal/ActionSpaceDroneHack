"""Team 1's training variant of AS-Defend-v0: same task, harder on what transfers to real drones."""

from __future__ import annotations

from isaaclab.utils import configclass

from action_space_kit.control import TelloCommandCfg

from ..tello_defend.tello_defend_env_cfg import TelloDefendEnvCfg


@configclass
class Team1DefendRobustEnvCfg(TelloDefendEnvCfg):
    """AS-Defend-v0 with the uncertainties of the real room.

    Observation and action layouts are unchanged, so a policy trained here is scored on the
    official task with ``scripts/evaluate.py --task AS-Defend-v0``.
    """

    # We assume the asset's position is unknown. The policy sees its own position relative to
    # a believed site: where the attackers' flight lines meet (control/defense.py
    # SiteEstimator), starting each episode from the centroid of the defenders' start
    # positions, plus a per-episode bias for estimation error on real drones. Breaches are
    # still judged on the true site.
    site_memory_s = 2.0  # [s] the estimator's averaging window
    site_bias_xy = 0.2  # [m] uniform +- per horizontal axis, per episode
    # Inference can fail (attackers not moving, parallel or feinting runs, lost tracking). In
    # this fraction of episodes it is off: the believed site stays at the starting guess, with
    # a larger error, so the policy learns not to lean on the site input when it says nothing.
    site_inference_off_prob = 0.0
    site_bias_off_xy = 1.0  # [m] uniform +- per horizontal axis, in inference-off episodes

    # The real room is about 4.5 x 3.4 m, so attackers start closer than the 6 m ring.
    # Each attacker's start radius is drawn per wave from this range.
    spawn_radius_range = [3.0, 6.0]  # [m]

    # SDK latency. The controller runs every physics step (1/120 s), so 24 steps = 0.2 s,
    # the middle of the 0.1-0.3 s expected on real Tellos.
    command: TelloCommandCfg = TelloCommandCfg(command_latency_steps=24)


@configclass
class Team1DefendRobustV1EnvCfg(Team1DefendRobustEnvCfg):
    """v0 plus episodes where the asset position cannot be inferred at all."""

    site_inference_off_prob = 0.3


@configclass
class Team1DefendRobustRealisticEnvCfg(Team1DefendRobustV1EnvCfg):
    """Robust v1 (room-scale spawns, latency, asset unknown) with attackers that fly like real Tellos."""

    attacker_weave_prob = 0.7
    attacker_evasion_range = [0.2, 1.0]
    attacker_weave_period = [2.0, 6.0]  # [s]
    attacker_wobble_std = 0.15  # [rad]
    attacker_wobble_tau = 1.0  # [s]
    attacker_min_height = 0.7  # [m] attackers climb if they get lower than this
    attacker_speed = [0.2, 1.0]  # [m/s] per attacker per wave: wider than the official [0.4, 0.9]


@configclass
class Team1DefendIRLEnvCfg(Team1DefendRobustRealisticEnvCfg):
    """As close to the real room as we can model: Robust-Realistic plus positions-only sensing.

    Every velocity the policy sees (its own and the attackers') is estimated from noisy position
    fixes by the deployed Kalman filter (team1_sensing.py); attackers carry no tags.
    """

    pos_fix_hz = 15.0  # [Hz] position fixes per drone
    pos_fix_noise = 0.04  # [m] per axis, per fix
    pos_fix_dropout = 0.1  # fraction of fixes lost
