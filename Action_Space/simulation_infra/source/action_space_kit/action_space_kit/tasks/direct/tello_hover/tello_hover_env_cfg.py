"""Configuration for the Tello hover task."""

from __future__ import annotations

import gymnasium as gym

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from action_space_kit.assets import TELLO_CFG
from action_space_kit.control import TelloCommandCfg


@configclass
class TelloHoverEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces: action = [vx, vy, vz, yaw_rate] (the four SDK rc channels).
    # Bounded so sampled actions can be clipped to the interface the real drone accepts.
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    # - obs = goal error (3) + lin vel (3) + ang vel (3) + projected gravity (3)
    observation_space = 12
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: RigidObjectCfg = TELLO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene: 1024 envs keeps VRAM modest so several teams can share one GPU
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)

    # Tello SDK velocity interface
    command: TelloCommandCfg = TelloCommandCfg()

    # - goal sampling (relative to the env origin) [m]
    goal_range_xy = 1.0
    goal_range_z = [0.8, 1.8]
    # - reward scales
    rew_scale_distance = 8.0
    rew_scale_lin_vel = -0.05
    rew_scale_ang_vel = -0.01
    rew_scale_action_rate = -0.02
    rew_scale_upright = 0.5
    # - termination
    max_distance = 3.0  # [m] from goal
    min_height = 0.05  # [m] above env origin

    # - realism options (off by default; see TelloHoverRealEnvCfg)
    world_frame_force = False  # rotate the controller's world-frame force into the body frame
    lock_yaw = False  # zero the yaw channel
    obs_noise_pos = 0.0  # [m] std on the goal error, which is also rounded to 1 cm
    obs_noise_vel = 0.0  # [m/s] std on the velocity


@configclass
class TelloHoverRealEnvCfg(TelloHoverEnvCfg):
    """Hover as a real Tello flies it (measured 2026-10-24 on TELLO-7, which bobbed +-9 cm with the
    stock policy): 15 Hz commands, ~0.3 s response plus ~0.1 s link latency, a range sensor in
    1 cm steps and a noisy differenced velocity, full stick = 0.3 m/s (the cap we fly with), no
    yaw. The stock policy learned near bang-bang control from exact 60 Hz state."""

    decimation = 8
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=8)
    command: TelloCommandCfg = TelloCommandCfg(max_lin_speed=0.3, response_time=0.3, command_latency_steps=12)
    world_frame_force = True
    lock_yaw = True
    obs_noise_pos = 0.005
    obs_noise_vel = 0.05
    goal_range_z = [0.5, 1.5]
    rew_scale_action_rate = -0.5
