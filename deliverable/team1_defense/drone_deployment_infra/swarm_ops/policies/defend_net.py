"""Team 1 defend networks in numpy: observation builder and forward pass. Needs only numpy.

  nets = DefendNets("weights/normal_contact.npz")
  action = nets(i, obs)          # defender i's [vx, vy, vz, yaw_rate] in [-1, 1]

Observation, 23 numbers per defender (the hosts' AS-Defend-v0 layout, 2 attacker slots), all in the
sim's world frame (x, y horizontal, z up; see map_to_sim):
  [0:3]   own position - site centre          site = protected point, 1.25 m above the floor
  [3:6]   own velocity                         m/s
  [6:9]   projected gravity in the body frame  (0, 0, -1) when level
  [9:15]  attacker slot 0, 1 position - own    clipped to +-12 m, zeros if the slot is empty
  [15:21] attacker slot 0, 1 velocity - own    clipped to +-3 m/s, zeros if the slot is empty
  [21:23] attacker slot 0, 1 alive flag        1 or 0
Action: world-frame velocity setpoint, x 1.0 m/s per unit (the official task's max_lin_speed),
yaw_rate x 1.745 rad/s. The observation has no heading, so yaw doesn't matter to the network.
"""
import numpy as np

CLIP_POS, CLIP_VEL = 12.0, 3.0      # obs_clip_pos / obs_clip_vel of tello_defend_env_cfg.py
SITE_HEIGHT = 2.5                    # the site centre is SITE_HEIGHT / 2 above the floor
MAX_LIN_SPEED = 1.0                  # m/s per action unit in training
EPS, CLIP = 1e-8, 5.0                # skrl RunningStandardScaler


def map_to_sim(v):
    """Organizers' map frame (x right, y up, z out of tag 10) -> sim frame (x, y horizontal, z up)."""
    v = np.asarray(v, float)
    return np.array([v[0], -v[2], v[1]])


def sim_to_map(v):
    v = np.asarray(v, float)
    return np.array([v[0], v[2], -v[1]])


def build_obs(own_p, own_v, site, attackers):
    """own_p, own_v, site: sim frame. attackers: two (pos, vel) in the sim frame, or None for an empty slot."""
    own_p, own_v = np.asarray(own_p, float), np.asarray(own_v, float)
    rel_p, rel_v, alive = np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2)
    for k, att in enumerate(attackers[:2]):
        if att is None:
            continue
        rel_p[k] = np.clip(np.asarray(att[0], float) - own_p, -CLIP_POS, CLIP_POS)
        rel_v[k] = np.clip(np.asarray(att[1], float) - own_v, -CLIP_VEL, CLIP_VEL)
        alive[k] = 1.0
    grav = np.array([0.0, 0.0, -1.0])
    return np.concatenate([own_p - np.asarray(site, float), own_v, grav, rel_p.ravel(), rel_v.ravel(), alive]).astype(np.float32)


class DefendNets:
    """One MLP per defender slot (defender_0, defender_1, ...), loaded from export_npz.py's .npz."""

    def __init__(self, path):
        z = np.load(path)
        self.count = len({k.split("/")[0] for k in z.files})
        self.nets = []
        for i in range(self.count):
            a = f"defender_{i}"
            self.nets.append((z[f"{a}/mean"], np.sqrt(z[f"{a}/var"]) + EPS,
                              [(z[f"{a}/W{k}"], z[f"{a}/b{k}"]) for k in range(4)]))
        self.obs_dim = self.nets[0][0].shape[0]

    def __call__(self, i, obs):
        mean, std, layers = self.nets[i]
        h = np.clip((np.asarray(obs, np.float32) - mean) / std, -CLIP, CLIP)
        for k, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if k < 3:
                h = np.where(h > 0, h, np.expm1(np.minimum(h, 0.0)))  # ELU
        return np.clip(h, -1.0, 1.0)
