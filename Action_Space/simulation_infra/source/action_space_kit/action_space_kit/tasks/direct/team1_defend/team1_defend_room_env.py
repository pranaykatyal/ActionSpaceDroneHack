"""The defend task placed in the real room, built from the organizers' tag map.

Geometry comes from ``drone_deployment_infra/config/map.yaml`` (+ ``tags.yaml``), read with the
hosts' ``tags/room_map.py`` in its sim frame (X out of tag 10's wall, Y = map x, Z up, floor at 0),
plus the room numbers the hosts' room task already uses (``tello_room_env_cfg.py``):

- asset: in front of tag 10, where the organizers' attackers form up (``swarm_formation.py``
  ``--target 10``, slots on a 1.3 m arc). The protected cylinder keeps the official 1.5 m radius
  and touches tag 10's wall: centre at X = wall + site_radius, Y = 0.
- attackers: the hosts' room start box (centre (6.5, 0), 2 m wide, 6-7 m from tag 10), below the
  ceiling; every wave respawns there and flies at the asset as in the official task.
- defenders: the official 2.5 m ring around the asset, on the half facing the attackers, inside
  the room.
- walls: tag-10 wall, right wall from the map (tags facing -Y, i.e. tag 8), far wall from the map
  when it is mapped (tags facing -X; else the hosts' 8.5 m), the hosts' measured left wall and
  ceiling. Attackers never start behind a wall (``attacker_wall_margin``). A defender closer than ``wall_clearance`` to any of them has
  crashed: from then on it is commanded down and is out for the episode. Attackers are left alone.

Everything else (dynamics, 0.35 m capture, breach, 20 s episodes, observation, action) is the
hosts' task. Not an official score: scored with ``scripts/evaluate.py --task Team1-Defend-3v2-Room-v0``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from action_space_kit.tags.room_map import load_room_tags
from action_space_kit.tasks.direct.tello_defend.tello_defend_env import TelloDefendEnv

from .team1_defend_3v2_env_cfg import Team1Defend3v2EnvCfg
from .team1_defend_3v2_realistic_env_cfg import Team1Defend3v2RealisticEnvCfg
from .team1_defend_shaped_env import Team1DefendRealisticEnv

# cfg fields read by the mixin, with their defaults (a task cfg sets only what it changes)
ROOM_DEFAULTS = {
    "room_map_path": None,  # None: the hosts' drone_deployment_infra/config/map.yaml
    "room_tags_path": None,
    "tag10_height": 1.22,  # [m] hosts' room task: centre of tag 10 above the floor
    "wall_x": -0.10,  # [m] hosts' room task: tag-10 wall surface
    "side_wall_y": None,  # [m] right wall; None: from the map (tags facing -Y)
    "room_y_min": -2.0,  # [m] hosts' room task: left wall, measured
    # "room_x_max": 8.5,  # [m] hosts' room task: far wall (not mapped)
    "room_x_max": None,  # [m] far wall; None: from the map (tags facing -X), else the hosts' 8.5 (not mapped)
    "attacker_wall_margin": 0.3,  # [m] attackers start at least this far inside the walls
    "ceiling": 2.74,  # [m] hosts' room task: 9 ft
    "wall_clearance": 0.2,  # [m] hosts' room task: closer counts as a crash
    "site_y": 0.0,  # [m] asset centre on tag 10's normal
    "attacker_box_center": (6.5, 0.0),  # [m] hosts' room task start box
    "attacker_box_size": 2.0,  # [m]
    "attacker_range": (6.0, 7.0),  # [m] horizontal distance to tag 10
    "attacker_height": (1.0, 2.2),  # [m] below the ceiling
    "defender_arc_deg": 75.0,  # defenders start within this angle of the direction to the attackers
    "defender_height": (1.0, 2.0),  # [m] official
    "room_margin": 0.4,  # [m] start positions keep this far from the walls
}


class RoomGeometryMixin:
    """Leftmost base: ``class X(RoomGeometryMixin, SomeDefendEnv)``. Env origin = asset centre."""

    def _rcfg(self, key: str):
        return getattr(self.cfg, key, ROOM_DEFAULTS[key])

    def _ensure_room(self) -> None:
        if hasattr(self, "_room_crashed"):
            return
        map_path, tags_path = self._rcfg("room_map_path"), self._rcfg("room_tags_path")
        tags = load_room_tags(self._rcfg("tag10_height"), Path(map_path) if map_path else None,
                              Path(tags_path) if tags_path else None)
        side = self._rcfg("side_wall_y")
        if side is None:
            right = [t.pos[1] for t in tags.values() if t.normal[1] < -0.9]
            side = float(min(right)) if right else 1.70
        self.room_side_wall_y = side
        far = self._rcfg("room_x_max")
        far_src = "cfg"
        if far is None:
            back = [t.pos[0] for t in tags.values() if t.normal[0] < -0.9]
            far, far_src = (float(min(back)), "map") if back else (8.5, "hosts' 8.5, not mapped")
        self.room_far_wall_x = far
        self.room_tag_ids = sorted(tags)
        # asset centre in room coordinates; room point = (world - env origin) + this
        self._room_site = torch.tensor([self._rcfg("wall_x") + self.cfg.site_radius, self._rcfg("site_y"), 0.0],
                                       device=self.device)
        self._room_lo = torch.tensor([self._rcfg("wall_x"), self._rcfg("room_y_min"), -1.0], device=self.device)
        self._room_hi = torch.tensor([far, side, self._rcfg("ceiling")], device=self.device)
        self._room_crashed = torch.zeros(self.num_envs, self.cfg.num_defenders, dtype=torch.bool, device=self.device)
        self.room_wall_crashes = 0  # defenders lost to a wall or the ceiling, over the run
        self.room_episodes = 0
        print(f"[room] tags {self.room_tag_ids}; walls X {self._room_lo[0]:.2f}..{self._room_hi[0]:.2f} (far wall: {far_src}), "
              f"Y {self._room_lo[1]:.2f}..{side:.2f} (right wall from map), ceiling {self._room_hi[2]:.2f}; "
              f"asset centre X {float(self._room_site[0]):.2f} Y {float(self._room_site[1]):.2f}, "
              f"radius {self.cfg.site_radius}", flush=True)

    def _to_world(self, env_ids: torch.Tensor, room_xyz: torch.Tensor) -> torch.Tensor:
        return self.scene.env_origins[env_ids] + room_xyz - self._room_site

    def _place(self, drone, ctl, env_ids: torch.Tensor, room_xyz: torch.Tensor) -> None:
        state = drone.data.default_root_state[env_ids].clone()
        state[:, :3] = self._to_world(env_ids, room_xyz)
        state[:, 7:] = 0.0
        drone.write_root_pose_to_sim(state[:, :7], env_ids)
        drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
        ctl.reset(env_ids)

    def _place_attackers(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        cx, cy = self._rcfg("attacker_box_center")
        half = 0.5 * self._rcfg("attacker_box_size")
        r_lo, r_hi = self._rcfg("attacker_range")
        h_lo, h_hi = self._rcfg("attacker_height")
        for i, drone in enumerate(self._attackers):
            # inside the box and 6-7 m from tag 10 (horizontal): pick Y in the box, then X on the range
            y = cy + sample_uniform(-half, half, (n,), self.device)
            r = sample_uniform(r_lo, r_hi, (n,), self.device)
            x = torch.sqrt(torch.clamp(r * r - y * y, min=0.0)).clamp(cx - half, cx + half)
            # never behind a wall: a mapped far wall can be nearer than the hosts' start box
            m = self._rcfg("attacker_wall_margin")
            x = x.clamp(max=float(self._room_hi[0]) - m)
            y = y.clamp(float(self._room_lo[1]) + m, float(self._room_hi[1]) - m)
            z = sample_uniform(h_lo, h_hi, (n,), self.device)
            self._place(drone, self._attacker_ctl[i], env_ids, torch.stack([x, y, z], dim=-1))

    def _place_defenders(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        cx, cy = self._rcfg("attacker_box_center")
        site = self._room_site
        facing = float(np.arctan2(cy - float(site[1]), cx - float(site[0])))
        arc = np.radians(self._rcfg("defender_arc_deg"))
        m = self._rcfg("room_margin")
        h_lo, h_hi = self._rcfg("defender_height")
        for i, drone in enumerate(self._defenders):
            angle = facing + sample_uniform(-arc, arc, (n,), self.device)
            x = site[0] + self.cfg.defender_radius * torch.cos(angle)
            y = site[1] + self.cfg.defender_radius * torch.sin(angle)
            x = x.clamp(float(self._room_lo[0]) + m, float(self._room_hi[0]) - m)
            y = y.clamp(float(self._room_lo[1]) + m, float(self._room_hi[1]) - m)
            z = sample_uniform(h_lo, h_hi, (n,), self.device)
            self._place(drone, self._defender_ctl[i], env_ids, torch.stack([x, y, z], dim=-1))

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self._ensure_room()
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._place_defenders(env_ids)
        self._place_attackers(env_ids)
        self._room_crashed[env_ids] = False
        self.room_episodes += len(env_ids)

    def _respawn_attackers(self, env_ids: torch.Tensor) -> None:
        super()._respawn_attackers(env_ids)
        if len(env_ids):
            self._ensure_room()
            self._place_attackers(torch.as_tensor(env_ids, device=self.device, dtype=torch.long))

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        super()._pre_physics_step(actions)
        self._ensure_room()
        clear = self._rcfg("wall_clearance")
        for i, drone in enumerate(self._defenders):
            room = drone.data.root_pos_w - self.scene.env_origins + self._room_site
            hit = ((room[:, :2] < self._room_lo[:2] + clear) | (room[:, :2] > self._room_hi[:2] - clear)).any(dim=-1)
            hit = hit | (room[:, 2] > self._room_hi[2] - clear)
            new = hit & ~self._room_crashed[:, i]
            self.room_wall_crashes += int(new.sum())
            self._room_crashed[:, i] |= hit
            # a crashed defender drops out: level, straight down, for the rest of the episode
            down = torch.tensor([0.0, 0.0, -1.0, 0.0], device=self.device)
            agent = f"defender_{i}"
            self._actions[agent] = torch.where(self._room_crashed[:, i, None], down, self._actions[agent])

    def close(self):
        if hasattr(self, "_room_crashed"):
            print(f"[room] defenders lost to walls/ceiling: {self.room_wall_crashes} over {self.room_episodes} "
                  f"episode starts ({self.room_wall_crashes / max(self.room_episodes, 1):.2f} per episode)", flush=True)
        super().close()


class Team1DefendRoomEnv(RoomGeometryMixin, TelloDefendEnv):
    """The hosts' defend task in the room (official attackers)."""


class Team1DefendRoomRealisticEnv(RoomGeometryMixin, Team1DefendRealisticEnv):
    """The room with realistic attackers (weave, wobble, 0.2-1.0 m/s)."""


@configclass
class Team1Defend3v2RoomEnvCfg(Team1Defend3v2EnvCfg):
    """3 defenders vs 2 official attackers in the room from the organizers' map."""


@configclass
class Team1Defend3v2RoomRealisticEnvCfg(Team1Defend3v2RealisticEnvCfg):
    """3 defenders vs 2 realistic attackers in the room from the organizers' map."""
