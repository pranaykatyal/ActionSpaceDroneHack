"""Defend task in the real flying space: attackers cross the room from tag 10 towards the base.

Same agents, observation layout and reward terms as ``TelloDefendEnv``; differences:
the breach is crossing the tag-6 line, defenders crash on walls/floor/ceiling or each
other (episode ends), and starts/targets are placed in the room corridor.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import sample_uniform

from ..tello_defend.tello_defend_env import TelloDefendEnv
from .tello_defend_room_env_cfg import TelloDefendRoomEnvCfg


class TelloDefendRoomEnv(TelloDefendEnv):
    cfg: TelloDefendRoomEnvCfg

    def __init__(self, cfg: TelloDefendRoomEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        shape = (self.num_envs, self.cfg.num_attackers)
        self._attacker_target = torch.zeros(*shape, 3, device=self.device)  # env-relative
        self._attacker_evasion = torch.zeros(shape, device=self.device)
        self._attacker_phase = torch.zeros(shape, device=self.device)
        self._crashed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.outcome_crashed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    # -- helpers ----------------------------------------------------------------

    def _lanes(self, count: int, num: int) -> torch.Tensor:
        """Y positions for ``count`` drones: one lane each across the corridor, jittered within it."""
        lo, hi = self.cfg.arena_y[0] + self.cfg.lane_margin, self.cfg.arena_y[1] - self.cfg.lane_margin
        width = (hi - lo) / count
        centres = lo + width * (torch.arange(count, device=self.device) + 0.5)
        jitter = sample_uniform(-0.25 * width, 0.25 * width, (num, count), self.device)
        return centres.unsqueeze(0) + jitter

    def _place(self, drone, ctl, env_ids, x, y, z) -> None:
        state = drone.data.default_root_state[env_ids].clone()
        origins = self.scene.env_origins[env_ids]
        state[:, 0] = origins[:, 0] + x
        state[:, 1] = origins[:, 1] + y
        state[:, 2] = origins[:, 2] + z
        state[:, 7:] = 0.0
        drone.write_root_pose_to_sim(state[:, :7], env_ids)
        drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
        ctl.reset(env_ids)

    def _park_attackers(self, captured: torch.Tensor) -> None:
        """Move caught attackers out of the room: a downed drone hovering in the corridor is an obstacle."""
        for i, drone in enumerate(self._attackers):
            env_ids = captured[:, i].nonzero(as_tuple=False).flatten()
            if len(env_ids) == 0:
                continue
            state = drone.data.default_root_state[env_ids].clone()
            state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor(self.cfg.park_position, device=self.device)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)

    def _launch_attackers(self, env_ids: torch.Tensor) -> None:
        """Attackers start at the tag-10 end, one per lane, and fly to a random point past the breach line."""
        num = len(env_ids)
        n = self.cfg.num_attackers
        lanes = self._lanes(n, num)[:, torch.randperm(n, device=self.device)]
        for i, drone in enumerate(self._attackers):
            x = sample_uniform(*self.cfg.attacker_start_x, (num,), self.device)
            z = sample_uniform(*self.cfg.attacker_height, (num,), self.device)
            self._place(drone, self._attacker_ctl[i], env_ids, x, lanes[:, i], z)
        lo, hi = self.cfg.arena_y[0] + self.cfg.lane_margin, self.cfg.arena_y[1] - self.cfg.lane_margin
        target = self._attacker_target[env_ids]
        target[..., 0] = self.cfg.breach_x + 1.0
        target[..., 1] = sample_uniform(lo, hi, (num, n), self.device)
        target[..., 2] = sample_uniform(*self.cfg.attacker_height, (num, n), self.device)
        self._attacker_target[env_ids] = target
        self._attacker_evasion[env_ids] = sample_uniform(0.0, self.cfg.attacker_evasion_max, (num, n), self.device)
        self._attacker_phase[env_ids] = sample_uniform(0.0, 6.2832, (num, n), self.device)
        self._attacker_alive[env_ids] = True
        self._attacker_speed[env_ids] = sample_uniform(
            self.cfg.attacker_speed[0], self.cfg.attacker_speed[1], (num, n), self.device
        )
        self._prev_distance[env_ids] = self.cfg.spawn_radius

    # -- stepping ---------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        super()._pre_physics_step(actions)
        for agent in self.cfg.possible_agents:
            self._actions[agent][:, 2] *= self.cfg.vz_scale
            if self.cfg.lock_yaw:
                self._actions[agent][:, 3] = 0.0

    def _scripted_attacker_actions(self) -> list[torch.Tensor]:
        """Fly each attacker at its target past the breach line, weaving sideways if it has evasion."""
        actions = []
        t = self.episode_length_buf.float() * self.step_dt * 2.0
        for i, drone in enumerate(self._attackers):
            to_target = self.scene.env_origins + self._attacker_target[:, i] - drone.data.root_pos_w
            direction = to_target / torch.clamp(torch.linalg.norm(to_target, dim=-1, keepdim=True), min=1e-6)
            lateral = torch.stack([-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])], dim=-1)
            weave = self._attacker_evasion[:, i] * torch.sin(t + self._attacker_phase[:, i])
            direction = direction + weave.unsqueeze(-1) * lateral
            direction = direction / torch.clamp(torch.linalg.norm(direction, dim=-1, keepdim=True), min=1e-6)
            speed = (self._attacker_speed[:, i] / self.cfg.command.max_lin_speed).unsqueeze(-1)
            action = torch.cat([direction * speed, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
            action = action * self._attacker_alive[:, i].unsqueeze(-1).float()
            actions.append(action.clamp(-1.0, 1.0))
        return actions

    # -- MDP --------------------------------------------------------------------

    def _defender_positions(self) -> torch.Tensor:
        """Env-relative defender positions, shaped (num_envs, num_defenders, 3)."""
        pos = torch.stack([d.data.root_pos_w for d in self._defenders], dim=1)
        return pos - self.scene.env_origins.unsqueeze(1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        attacker_pos, attacker_vel = self._attacker_states()
        own_pos = self._defender_positions()
        own_vel = torch.stack([d.data.root_lin_vel_w for d in self._defenders], dim=1)
        alive = self._attacker_alive.float().unsqueeze(-1)
        obs = {}
        for i, drone in enumerate(self._defenders):
            rel_pos = attacker_pos - drone.data.root_pos_w.unsqueeze(1)
            rel_vel = attacker_vel - drone.data.root_lin_vel_w.unsqueeze(1)
            rel_pos = rel_pos.clamp(-self.cfg.obs_clip_pos, self.cfg.obs_clip_pos) * alive
            rel_vel = rel_vel.clamp(-self.cfg.obs_clip_vel, self.cfg.obs_clip_vel) * alive
            # teammates in index order, after the attacker block (evaluate.py's chase reads that at 9:)
            mates = [j for j in range(self.cfg.num_defenders) if j != i]
            mate_pos = (own_pos[:, mates] - own_pos[:, i : i + 1]).clamp(-self.cfg.obs_clip_pos, self.cfg.obs_clip_pos)
            mate_vel = (own_vel[:, mates] - own_vel[:, i : i + 1]).clamp(-self.cfg.obs_clip_vel, self.cfg.obs_clip_vel)
            obs[f"defender_{i}"] = torch.cat(
                [
                    own_pos[:, i],  # room position (walls are fixed in this frame)
                    drone.data.root_lin_vel_w,
                    drone.data.projected_gravity_b,
                    rel_pos.reshape(self.num_envs, -1),
                    rel_vel.reshape(self.num_envs, -1),
                    alive.squeeze(-1),
                    mate_pos.reshape(self.num_envs, -1),
                    mate_vel.reshape(self.num_envs, -1),
                ],
                dim=-1,
            )
        return obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        attacker_pos, _ = self._attacker_states()

        captured = torch.zeros_like(self._attacker_alive)
        defender_credit = torch.zeros(self.num_envs, self.cfg.num_defenders, device=self.device)
        for i, drone in enumerate(self._defenders):
            dist = torch.linalg.norm(attacker_pos - drone.data.root_pos_w.unsqueeze(1), dim=-1)
            hit = (dist < self.cfg.capture_radius) & self._attacker_alive
            captured |= hit
            defender_credit[:, i] = hit.float().sum(dim=-1)
        self._attacker_alive &= ~captured
        self._park_attackers(captured)

        # breach: a live attacker past the tag-6 line
        attacker_x = attacker_pos[..., 0] - self.scene.env_origins[:, 0].unsqueeze(1)
        self._breached = ((attacker_x >= self.cfg.breach_x) & self._attacker_alive).any(dim=-1)

        # crashes: walls, floor, ceiling, or another defender
        pos = self._defender_positions()
        m = self.cfg.wall_margin
        out = (
            (pos[..., 0] < self.cfg.arena_x[0] + m)
            | (pos[..., 0] > self.cfg.arena_x[1] - m)
            | (pos[..., 1] < self.cfg.arena_y[0] + m)
            | (pos[..., 1] > self.cfg.arena_y[1] - m)
            | (pos[..., 2] < self.cfg.min_height)
            | (pos[..., 2] > self.cfg.arena_height - m)
        )
        pair = torch.cdist(pos, pos)
        pair = pair + torch.eye(self.cfg.num_defenders, device=self.device) * 1e6
        collided = (pair < self.cfg.collision_distance).any(dim=-1)
        self._crashed = (out | collided).any(dim=-1)

        wall_gap = torch.stack(
            [
                pos[..., 0] - self.cfg.arena_x[0] - m,
                self.cfg.arena_x[1] - m - pos[..., 0],
                pos[..., 1] - self.cfg.arena_y[0] - m,
                self.cfg.arena_y[1] - m - pos[..., 1],
                pos[..., 2] - self.cfg.min_height,
                self.cfg.arena_height - m - pos[..., 2],
            ],
            dim=-1,
        ).amin(dim=-1)
        mate_gap = pair.amin(dim=-1) - self.cfg.collision_distance
        wall_warning = torch.exp(-wall_gap.clamp(min=0.0) / self.cfg.warning_distance)
        mate_warning = torch.exp(-mate_gap.clamp(min=0.0) / self.cfg.warning_distance)

        cleared = captured.any(dim=-1) & ~self._attacker_alive.any(dim=-1)
        self._episode_captures += captured.float().sum(dim=-1)

        rewards = {}
        any_alive = self._attacker_alive.any(dim=-1)
        for i, drone in enumerate(self._defenders):
            dist = torch.linalg.norm(attacker_pos - drone.data.root_pos_w.unsqueeze(1), dim=-1)
            dist = torch.where(self._attacker_alive, dist, torch.full_like(dist, float("inf")))
            nearest = torch.where(any_alive, dist.min(dim=-1).values, self._prev_distance[:, i])
            closing = self._prev_distance[:, i] - nearest
            self._prev_distance[:, i] = nearest
            effort = torch.sum(torch.square(self._actions[f"defender_{i}"]), dim=-1)
            proximity = torch.exp(-nearest.clamp(min=0.0, max=self.cfg.spawn_radius * 2.0) / self.cfg.spawn_radius)
            # same scaling as TelloDefendEnv: rates by step_dt, per-step deltas and events not
            shaping = (
                self.cfg.rew_scale_proximity * proximity
                + self.cfg.rew_scale_effort * effort
                + self.cfg.rew_scale_alive
                + self.cfg.rew_scale_wall * wall_warning[:, i]
                + self.cfg.rew_scale_separation * mate_warning[:, i]
            ) * self.step_dt + self.cfg.rew_scale_closing * closing.clamp(-1.0, 1.0)
            events = (
                self.cfg.rew_scale_capture * defender_credit[:, i]
                + self.cfg.rew_scale_breach * self._breached.float()
                + self.cfg.rew_scale_cleared * cleared.float()
                + self.cfg.rew_scale_crash * out[:, i].float()
                + self.cfg.rew_scale_collision * collided[:, i].float()
            )
            rewards[f"defender_{i}"] = shaping + events
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        all_down = ~self._attacker_alive.any(dim=-1)
        if self.cfg.respawn_waves:
            if all_down.any():
                self._respawn_attackers(all_down.nonzero(as_tuple=False).flatten())
            done = self._breached | self._crashed
        else:
            done = self._breached | self._crashed | all_down
        finished = done | time_out
        self.outcome_valid = finished
        self.outcome_breached = torch.where(finished, self._breached, self.outcome_breached)
        self.outcome_crashed = torch.where(finished, self._crashed, self.outcome_crashed)
        self.outcome_captures = torch.where(finished, self._episode_captures, self.outcome_captures)
        terminated = {agent: done for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _respawn_attackers(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._launch_attackers(env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)  # buffers and controllers; the poses are replaced below
        env_ids = torch.as_tensor(env_ids, device=self.device)
        num = len(env_ids)

        lanes = self._lanes(self.cfg.num_defenders, num)
        for i, drone in enumerate(self._defenders):
            x = sample_uniform(*self.cfg.defender_start_x, (num,), self.device)
            z = sample_uniform(*self.cfg.defender_height, (num,), self.device)
            self._place(drone, self._defender_ctl[i], env_ids, x, lanes[:, i], z)
        self._launch_attackers(env_ids)
        self._crashed[env_ids] = False

    def _debug_vis_callback(self, event):
        site = self.scene.env_origins.clone()
        site[:, 0] += self.cfg.breach_x + 0.5
        site[:, 1] += 0.5 * (self.cfg.arena_y[0] + self.cfg.arena_y[1])
        self._site_markers.visualize(site)
