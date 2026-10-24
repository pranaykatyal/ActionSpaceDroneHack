"""Defend task reshaped to the real flying space: a corridor from tag 10 to the base past tag 6.

Simulation frame (env origin = the floor directly below tag 10's centre):
    X = map z (out of the tag-10 wall, towards the base)
    Y = map x (tag 10's +x, "right")
    Z = height above the floor = map y + 1.207 (floor at the lowest mapped tag, tag 4)

Room bounds from the mapped tags: map x -0.40..2.00 (tag span 0..2.0 plus 20% on the
left), height 0..1.776 m (highest tag plus 20% of the vertical span), breach line at
tag 6 (map z 6.483); the base lies beyond it.
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ..tello_defend.tello_defend_env_cfg import TelloDefendEnvCfg

NUM_DEFENDERS = 3
NUM_ATTACKERS = 2
# TelloDefendEnv's layout, then each teammate's relative position and velocity (3 + 3)
_OBS_DIM = 9 + 7 * NUM_ATTACKERS + 6 * (NUM_DEFENDERS - 1)


@configclass
class TelloDefendRoomEnvCfg(TelloDefendEnvCfg):
    """3 defenders start in front of tag 6; 2 attackers fly from the tag-10 wall past tag 6.

    Defaults are the evaluation conditions. Train harder with overrides, e.g.
    ``env.attacker_speed=[0.3,1.2] env.attacker_evasion_max=0.6``.
    """

    # one policy step = 9 x 1/120 s = 75 ms: the real Tellos take rc commands at ~13.5 Hz
    decimation = 9
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    possible_agents = [f"defender_{i}" for i in range(NUM_DEFENDERS)]
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=10.0, replicate_physics=True)

    num_defenders = NUM_DEFENDERS
    num_attackers = NUM_ATTACKERS

    # - room (simulation frame, metres, relative to the env origin)
    arena_x = (0.0, 7.0)  # tag-10 wall .. a little past the breach line
    arena_y = (-0.40, 2.00)
    arena_height = 1.776
    breach_x = 6.483  # tag 6: an attacker crossing this line has reached the base
    wall_margin = 0.10  # [m] a defender centre closer than this to a wall/ceiling has crashed
    min_height = 0.15  # [m] below this a defender has hit the floor
    collision_distance = 0.25  # [m] two defenders closer than this have collided

    # - starts
    attacker_start_x = (0.3, 1.0)
    attacker_height = (0.5, 1.3)
    defender_start_x = (5.483, 6.183)  # 0.3-1.0 m in front of tag 6
    defender_height = (0.7, 1.1)  # measured: the ceiling caused most random-flight crashes

    # - defender sticks. Deploy the same way: rc up = 100 * vz_scale * action, rc yaw = 0.
    vz_scale = 0.3  # vertical speed limit 0.3 m/s; the room is only 1.78 m tall
    lock_yaw = True  # never turn: the real rc is in the heading frame, the sim controller is not
    lane_margin = 0.30  # [m] starts and attacker targets keep this far from the side walls
    park_position = (-2.0, -2.0, 0.5)  # caught attackers wait here, behind the tag-10 wall

    # - scripted attackers
    attacker_speed = [0.4, 0.9]  # [m/s], per attacker per episode
    attacker_evasion_max = 0.3  # weave strength ~ U(0, max) per attacker per episode; 0 = straight

    # - extra penalties (per defender)
    rew_scale_crash = -30.0  # wall, floor, ceiling
    rew_scale_collision = -30.0  # another defender within collision_distance
    # warnings before those (per second): exp(-gap / warning_distance), gap to the nearest
    # wall/floor/ceiling or past collision_distance to the nearest teammate
    warning_distance = 0.2  # [m]
    rew_scale_wall = -2.0
    rew_scale_separation = -2.0
