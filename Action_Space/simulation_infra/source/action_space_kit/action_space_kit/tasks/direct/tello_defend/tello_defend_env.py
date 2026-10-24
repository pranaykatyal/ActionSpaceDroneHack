"""Multi-agent defend task: Tello defenders intercept incoming attackers.

Each defender is an agent with the same four-channel velocity action as the single-drone
tasks, so trained policies remain flyable on real hardware. Attackers are scripted by
default and dive towards the protected site.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from action_space_kit.assets import TELLO_CFG, TELLO_MASS
from action_space_kit.control import TelloVelocityController
from action_space_kit.rl.terminal_observation import TerminalObservationMixin

from .tello_defend_env_cfg import TelloDefendEnvCfg

# attacker modes for cfg.attacker_retry; _attacker_alive is True exactly when ATTACKING
ATTACKING, OUT, RETURNING, HOLDING = 0, 1, 2, 3


class TelloDefendEnv(TerminalObservationMixin, DirectMARLEnv):
    cfg: TelloDefendEnvCfg

    def __init__(self, cfg: TelloDefendEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        inertia_zz = float(self._defenders[0].root_physx_view.get_inertias()[0].reshape(3, 3)[2, 2])
        make_controller = lambda: TelloVelocityController(  # noqa: E731
            cfg=self.cfg.command,
            num_envs=self.num_envs,
            mass=TELLO_MASS,
            inertia_zz=inertia_zz,
            device=self.device,
        )
        self._defender_ctl = [make_controller() for _ in range(self.cfg.num_defenders)]
        self._attacker_ctl = [make_controller() for _ in range(self.cfg.num_attackers)]

        self._actions = {
            agent: torch.zeros(self.num_envs, 4, device=self.device) for agent in self.cfg.possible_agents
        }
        # attacker bookkeeping
        self._attacker_alive = torch.ones(self.num_envs, self.cfg.num_attackers, device=self.device, dtype=torch.bool)
        self._attacker_speed = torch.zeros(self.num_envs, self.cfg.num_attackers, device=self.device)
        self._breached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._prev_distance = torch.zeros(self.num_envs, self.cfg.num_defenders, device=self.device)
        # cfg.attacker_retry: per-attacker mode, wait timer and return point
        shape = (self.num_envs, self.cfg.num_attackers)
        self._attacker_mode = torch.full(shape, ATTACKING, device=self.device, dtype=torch.long)
        self._attacker_timer = torch.zeros(shape, device=self.device)
        self._attacker_goal = torch.zeros(*shape, 3, device=self.device)
        # cfg.fix_reward_glitches: last defender-attacker distances, and whether they are still
        # comparable (False after a teleport)
        pairs = (self.num_envs, self.cfg.num_defenders, self.cfg.num_attackers)
        self._pair_distance = torch.zeros(pairs, device=self.device)
        self._pair_valid = torch.zeros(pairs, device=self.device, dtype=torch.bool)
        self._breach_paid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Episode outcome, latched in _get_dones before Isaac Lab resets the environment.
        # Reading _breached after env.step() is too late: the reset has already cleared it.
        self._episode_captures = torch.zeros(self.num_envs, device=self.device)
        self.outcome_breached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.outcome_captures = torch.zeros(self.num_envs, device=self.device)
        self.outcome_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    # -- scene ------------------------------------------------------------------

    def _setup_scene(self):
        self._defenders = []
        self._attackers = []
        for i in range(self.cfg.num_defenders):
            cfg = TELLO_CFG.replace(prim_path=f"/World/envs/env_.*/Defender_{i}")
            obj = RigidObject(cfg)
            self._defenders.append(obj)
        for i in range(self.cfg.num_attackers):
            cfg = TELLO_CFG.replace(prim_path=f"/World/envs/env_.*/Attacker_{i}")
            obj = RigidObject(cfg)
            self._attackers.append(obj)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Sensors must exist before cloning: the clone creates their per-env prims, and
        # attaching render annotators after the fact fails inside Replicator with
        # "Unable to write from unknown dtype".
        self._setup_extra_sensors()
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for i, obj in enumerate(self._defenders):
            self.scene.rigid_objects[f"defender_{i}"] = obj
        for i, obj in enumerate(self._attackers):
            self.scene.rigid_objects[f"attacker_{i}"] = obj

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.markers["cuboid"].size = (self.cfg.site_radius, self.cfg.site_radius, 0.05)
        marker_cfg.prim_path = "/Visuals/site"
        self._site_markers = VisualizationMarkers(marker_cfg)

    def _setup_extra_sensors(self) -> None:
        """Hook for subclasses to add sensors. Called before the scene is cloned."""
        return

    # -- stepping ---------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for agent, action in actions.items():
            self._actions[agent] = action.clone().clamp(-1.0, 1.0)
        if self.cfg.attacker_retry:
            self._advance_attackers()

    def _advance_attackers(self) -> None:
        """Retry mode: returning attackers that reached the ring start waiting; waiting ones whose
        timer ran out attack again at a freshly sampled speed."""
        pos, _ = self._attacker_states()
        arrived = (self._attacker_mode == RETURNING) & (torch.linalg.norm(pos - self._attacker_goal, dim=-1) < 0.3)
        self._attacker_mode[arrived] = HOLDING
        holding = self._attacker_mode == HOLDING
        self._attacker_timer = torch.where(holding, self._attacker_timer - self.step_dt, self._attacker_timer)
        go = holding & (self._attacker_timer <= 0.0)
        if go.any():
            self._attacker_mode[go] = ATTACKING
            self._attacker_alive |= go
            speed = sample_uniform(
                self.cfg.attacker_speed[0], self.cfg.attacker_speed[1], self._attacker_speed.shape, self.device
            )
            self._attacker_speed = torch.where(go, speed, self._attacker_speed)

    def _assign_retry_modes(self, captured: torch.Tensor) -> None:
        """Retry mode: pick what each just-captured attacker does next (out / return / hold)."""
        probs = torch.tensor(self.cfg.retry_mode_probs, device=self.device, dtype=torch.float)
        choice = torch.multinomial(probs.expand(captured.numel(), -1), 1).view_as(captured) + OUT
        wait_return = sample_uniform(*self.cfg.retry_wait_return, captured.shape, self.device)
        wait_hold = sample_uniform(*self.cfg.retry_wait_hold, captured.shape, self.device)
        wait = torch.where(choice == RETURNING, wait_return, wait_hold)
        angle = sample_uniform(0.0, 6.2832, captured.shape, self.device)
        goal = self.scene.env_origins.unsqueeze(1).repeat(1, self.cfg.num_attackers, 1)
        goal[..., 0] += self.cfg.spawn_radius * torch.cos(angle)
        goal[..., 1] += self.cfg.spawn_radius * torch.sin(angle)
        goal[..., 2] += sample_uniform(1.5, 3.0, captured.shape, self.device)
        self._attacker_mode = torch.where(captured, choice, self._attacker_mode)
        self._attacker_timer = torch.where(captured, wait, self._attacker_timer)
        self._attacker_goal = torch.where((captured & (choice == RETURNING)).unsqueeze(-1), goal, self._attacker_goal)

    def _apply_action(self) -> None:
        for i, drone in enumerate(self._defenders):
            force, torque = self._defender_ctl[i].compute(
                actions=self._actions[f"defender_{i}"],
                lin_vel_w=drone.data.root_lin_vel_w,
                ang_vel_b=drone.data.root_ang_vel_b,
                projected_gravity_b=drone.data.projected_gravity_b,
                dt=self.physics_dt,
            )
            drone.set_external_force_and_torque(forces=force, torques=torque, body_ids=[0])

        attacker_actions = self._scripted_attacker_actions()
        for i, drone in enumerate(self._attackers):
            force, torque = self._attacker_ctl[i].compute(
                actions=attacker_actions[i],
                lin_vel_w=drone.data.root_lin_vel_w,
                ang_vel_b=drone.data.root_ang_vel_b,
                projected_gravity_b=drone.data.projected_gravity_b,
                dt=self.physics_dt,
            )
            drone.set_external_force_and_torque(forces=force, torques=torque, body_ids=[0])

    def _scripted_attacker_actions(self) -> list[torch.Tensor]:
        """Fly each attacker at the protected site, with optional lateral weaving."""
        actions = []
        for i, drone in enumerate(self._attackers):
            to_site = self.scene.env_origins - drone.data.root_pos_w
            to_site[:, 2] += self.cfg.site_height * 0.5
            direction = to_site / torch.clamp(torch.linalg.norm(to_site, dim=-1, keepdim=True), min=1e-6)
            if self.cfg.attacker_evasion > 0.0:
                phase = self.episode_length_buf.float() * self.step_dt * 2.0 + i
                lateral = torch.stack([-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])], dim=-1)
                direction = direction + self.cfg.attacker_evasion * torch.sin(phase).unsqueeze(-1) * lateral
                direction = direction / torch.clamp(torch.linalg.norm(direction, dim=-1, keepdim=True), min=1e-6)
            speed = (self._attacker_speed[:, i] / self.cfg.command.max_lin_speed).unsqueeze(-1)
            action = torch.cat([direction * speed, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
            # a downed attacker stops moving
            action = action * self._attacker_alive[:, i].unsqueeze(-1).float()
            if self.cfg.attacker_retry:
                mode = self._attacker_mode[:, i].unsqueeze(-1)
                to_goal = self._attacker_goal[:, i] - drone.data.root_pos_w
                back = to_goal / torch.clamp(torch.linalg.norm(to_goal, dim=-1, keepdim=True), min=1e-6)
                back = torch.cat([back * speed, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
                sink = torch.zeros_like(action)
                sink[:, 2] = -self.cfg.retry_out_descent / self.cfg.command.max_lin_speed
                action = torch.where(mode == RETURNING, back, action)
                action = torch.where(mode == OUT, sink, action)
            actions.append(action.clamp(-1.0, 1.0))
        return actions

    # -- MDP --------------------------------------------------------------------

    def _attacker_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attacker positions and velocities, shaped (num_envs, num_attackers, 3)."""
        pos = torch.stack([d.data.root_pos_w for d in self._attackers], dim=1)
        vel = torch.stack([d.data.root_lin_vel_w for d in self._attackers], dim=1)
        return pos, vel

    def _get_observations(self) -> dict[str, torch.Tensor]:
        attacker_pos, attacker_vel = self._attacker_states()
        site = self.scene.env_origins.clone()
        site[:, 2] += self.cfg.site_height * 0.5

        obs = {}
        for i, drone in enumerate(self._defenders):
            own_rel_site = drone.data.root_pos_w - site
            rel_pos = attacker_pos - drone.data.root_pos_w.unsqueeze(1)
            rel_vel = attacker_vel - drone.data.root_lin_vel_w.unsqueeze(1)
            # A downed attacker is zeroed out and flagged, not pushed to a huge sentinel
            # value: a 100 m spike dominates the running normalisation and destabilises
            # the policy. The flag tells the network the slot is empty.
            alive = self._attacker_alive.float().unsqueeze(-1)
            # Retry mode keeps every attacker's real position: a downed one may come back.
            seen = torch.ones_like(alive) if self.cfg.attacker_retry else alive
            rel_pos = rel_pos.clamp(-self.cfg.obs_clip_pos, self.cfg.obs_clip_pos) * seen
            rel_vel = rel_vel.clamp(-self.cfg.obs_clip_vel, self.cfg.obs_clip_vel) * seen
            if self.cfg.attacker_retry and not self.cfg.observe_status:
                alive = torch.zeros_like(alive)
            obs[f"defender_{i}"] = torch.cat(
                [
                    own_rel_site,
                    drone.data.root_lin_vel_w,
                    drone.data.projected_gravity_b,
                    rel_pos.reshape(self.num_envs, -1),
                    rel_vel.reshape(self.num_envs, -1),
                    alive.squeeze(-1),
                ],
                dim=-1,
            )
        return obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        attacker_pos, _ = self._attacker_states()
        site = self.scene.env_origins.clone()
        site[:, 2] += self.cfg.site_height * 0.5

        # interceptions: any live attacker within capture radius of any defender
        captured = torch.zeros_like(self._attacker_alive)
        defender_credit = torch.zeros(self.num_envs, self.cfg.num_defenders, device=self.device)
        for i, drone in enumerate(self._defenders):
            dist = torch.linalg.norm(attacker_pos - drone.data.root_pos_w.unsqueeze(1), dim=-1)
            hit = (dist < self.cfg.capture_radius) & self._attacker_alive
            captured |= hit
            defender_credit[:, i] = hit.float().sum(dim=-1)
        self._attacker_alive &= ~captured
        if self.cfg.attacker_retry and captured.any():
            self._assign_retry_modes(captured)

        # breach: a live attacker inside the protected cylinder
        radial = torch.linalg.norm(attacker_pos[:, :, :2] - site[:, :2].unsqueeze(1), dim=-1)
        inside = (radial < self.cfg.site_radius) & self._attacker_alive
        self._breached = inside.any(dim=-1)
        # Rewards run after dones, so a breach is still set on the step that ends the episode.
        breach_event = self._breached & ~self._breach_paid if self.cfg.fix_reward_glitches else self._breached
        self._breach_paid |= self._breached

        # every attacker down this step: the outcome the task actually wants
        cleared = captured.any(dim=-1) & ~self._attacker_alive.any(dim=-1)
        self._episode_captures += captured.float().sum(dim=-1)

        rewards = {}
        for i, drone in enumerate(self._defenders):
            dist = torch.linalg.norm(attacker_pos - drone.data.root_pos_w.unsqueeze(1), dim=-1)
            any_alive = self._attacker_alive.any(dim=-1)
            dist = torch.where(self._attacker_alive, dist, torch.full_like(dist, float("inf")))
            nearest = dist.min(dim=-1).values
            # With no live attacker left there is no distance to close. Carrying the
            # previous value keeps `closing` at zero instead of swinging to the clamp,
            # which otherwise punishes the defender for the capture it just made.
            nearest = torch.where(any_alive, nearest, self._prev_distance[:, i])
            closing = self._prev_distance[:, i] - nearest
            self._prev_distance[:, i] = nearest
            if self.cfg.fix_reward_glitches:
                # progress on the attacker that is nearest now, against its own previous distance
                pair = torch.linalg.norm(attacker_pos - drone.data.root_pos_w.unsqueeze(1), dim=-1)
                target = dist.argmin(dim=-1, keepdim=True)
                before = self._pair_distance[:, i].gather(1, target).squeeze(1)
                comparable = self._pair_valid[:, i].gather(1, target).squeeze(1) & any_alive
                closing = torch.where(comparable, before - pair.gather(1, target).squeeze(1), torch.zeros_like(before))
                self._pair_distance[:, i] = pair
                self._pair_valid[:, i] = True
            effort = torch.sum(torch.square(self._actions[f"defender_{i}"]), dim=-1)
            proximity = torch.exp(-nearest.clamp(min=0.0, max=self.cfg.spawn_radius * 2.0) / self.cfg.spawn_radius)
            # Rates are scaled by step_dt; per-step deltas are not.
            #
            # proximity/alive/effort are rates (value per second), so without step_dt
            # their income over a 400-step episode dwarfs the capture bonus, and
            # clearing the field -- which ends the episode -- costs more future reward
            # than it pays, making stalling optimal.
            #
            # closing is already a per-step distance delta in metres. Scaling it by
            # step_dt as well shrinks the dense signal ~60x, leaving PPO with almost
            # nothing to follow between sparse captures. It telescopes over an episode
            # to (start distance - end distance), so at scale 5 a full 5 m approach is
            # worth ~25, comparable to one capture.
            shaping = (
                self.cfg.rew_scale_proximity * proximity
                + self.cfg.rew_scale_effort * effort
                + self.cfg.rew_scale_alive
            ) * self.step_dt + self.cfg.rew_scale_closing * closing.clamp(-1.0, 1.0)
            events = (
                self.cfg.rew_scale_capture * defender_credit[:, i]
                + self.cfg.rew_scale_breach * breach_event.float()
                + self.cfg.rew_scale_cleared * cleared.float()
            )
            rewards[f"defender_{i}"] = shaping + events
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        all_down = ~self._attacker_alive.any(dim=-1)
        if self.cfg.attacker_retry:
            # attackers come back on their own; a new wave only once every one of them is out
            all_out = (self._attacker_mode == OUT).all(dim=-1)
            if all_out.any():
                self._respawn_attackers(all_out.nonzero(as_tuple=False).flatten())
            done = self._breached
        elif self.cfg.respawn_waves:
            # a cleared field starts the next wave; only a breach ends the episode
            if all_down.any():
                self._respawn_attackers(all_down.nonzero(as_tuple=False).flatten())
            done = self._breached
        else:
            done = self._breached | all_down
        finished = done | time_out
        self.outcome_valid = finished
        self.outcome_breached = torch.where(finished, self._breached, self.outcome_breached)
        self.outcome_captures = torch.where(finished, self._episode_captures, self.outcome_captures)
        terminated = {agent: done for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _respawn_attackers(self, env_ids: torch.Tensor) -> None:
        """Send a fresh wave into the listed environments, leaving defenders as they are."""
        if len(env_ids) == 0:
            return
        num = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        for i, drone in enumerate(self._attackers):
            angle = sample_uniform(0.0, 6.2832, (num,), self.device)
            state = drone.data.default_root_state[env_ids].clone()
            state[:, 0] = origins[:, 0] + self.cfg.spawn_radius * torch.cos(angle)
            state[:, 1] = origins[:, 1] + self.cfg.spawn_radius * torch.sin(angle)
            state[:, 2] = origins[:, 2] + sample_uniform(1.5, 3.0, (num,), self.device)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
            self._attacker_ctl[i].reset(env_ids)
        self._attacker_alive[env_ids] = True
        self._attacker_speed[env_ids] = sample_uniform(
            self.cfg.attacker_speed[0], self.cfg.attacker_speed[1], (num, self.cfg.num_attackers), self.device
        )
        self._prev_distance[env_ids] = self.cfg.spawn_radius
        self._reset_retry_state(env_ids)

    def _reset_retry_state(self, env_ids: torch.Tensor) -> None:
        """Attackers just teleported: all attacking again, and old distances are not comparable."""
        self._attacker_mode[env_ids] = ATTACKING
        self._attacker_timer[env_ids] = 0.0
        self._pair_valid[env_ids] = False

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)
        num = len(env_ids)
        origins = self.scene.env_origins[env_ids]

        # defenders start on a ring around the site
        for i, drone in enumerate(self._defenders):
            angle = sample_uniform(0.0, 6.2832, (num,), self.device)
            state = drone.data.default_root_state[env_ids].clone()
            state[:, 0] = origins[:, 0] + self.cfg.defender_radius * torch.cos(angle)
            state[:, 1] = origins[:, 1] + self.cfg.defender_radius * torch.sin(angle)
            state[:, 2] = origins[:, 2] + sample_uniform(1.0, 2.0, (num,), self.device)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
            self._defender_ctl[i].reset(env_ids)

        # attackers enter from a wider ring
        for i, drone in enumerate(self._attackers):
            angle = sample_uniform(0.0, 6.2832, (num,), self.device)
            state = drone.data.default_root_state[env_ids].clone()
            state[:, 0] = origins[:, 0] + self.cfg.spawn_radius * torch.cos(angle)
            state[:, 1] = origins[:, 1] + self.cfg.spawn_radius * torch.sin(angle)
            state[:, 2] = origins[:, 2] + sample_uniform(1.5, 3.0, (num,), self.device)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
            self._attacker_ctl[i].reset(env_ids)

        self._attacker_alive[env_ids] = True
        self._attacker_speed[env_ids] = sample_uniform(
            self.cfg.attacker_speed[0], self.cfg.attacker_speed[1], (num, self.cfg.num_attackers), self.device
        )
        self._breached[env_ids] = False
        self._breach_paid[env_ids] = False
        self._reset_retry_state(env_ids)
        self._episode_captures[env_ids] = 0.0
        for agent in self.cfg.possible_agents:
            self._actions[agent][env_ids] = 0.0
        self._prev_distance[env_ids] = self.cfg.spawn_radius

    def _debug_vis_callback(self, event):
        site = self.scene.env_origins.clone()
        self._site_markers.visualize(site)
