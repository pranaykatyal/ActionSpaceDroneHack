"""Export a Team 1 defend checkpoint (skrl IPPO agent_N.pt) to a numpy .npz the laptop can run.

  python export_npz.py <agent_N.pt> <out.npz>      # needs torch (CPU is fine); the laptop then needs only numpy

Per defender i the .npz holds ``defender_i/mean``, ``defender_i/var`` (the observation scaler) and
``defender_i/W0..W3``, ``defender_i/b0..b3`` (the policy MLP 23 -> 256 -> 128 -> 64 -> 4, ELU).
Value network and log_std are dropped: at deploy the policy's mean action is flown.
The export is checked against the same forward pass in torch (pass: ``PARITY OK``).
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

LAYERS = ["net_container.0", "net_container.2", "net_container.4", "policy_layer"]
EPS, CLIP = 1e-8, 5.0  # skrl RunningStandardScaler defaults


def torch_forward(ck, agent, x):
    sc = ck[agent]["observation_preprocessor"]
    h = torch.clamp((x - sc["running_mean"].float()) / (torch.sqrt(sc["running_variance"].float()) + EPS), -CLIP, CLIP)
    pol = ck[agent]["policy"]
    for k, name in enumerate(LAYERS):
        h = F.linear(h, pol[f"{name}.weight"], pol[f"{name}.bias"])
        if k < 3:
            h = F.elu(h)
    return h.clamp(-1.0, 1.0)


def main(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    agents = sorted(k for k in ck if k.startswith("defender_"))
    out = {}
    for a in agents:
        sc, pol = ck[a]["observation_preprocessor"], ck[a]["policy"]
        out[f"{a}/mean"] = sc["running_mean"].float().numpy()
        out[f"{a}/var"] = sc["running_variance"].float().numpy()
        for k, name in enumerate(LAYERS):
            out[f"{a}/W{k}"] = pol[f"{name}.weight"].float().numpy()
            out[f"{a}/b{k}"] = pol[f"{name}.bias"].float().numpy()
    np.savez(dst, **out)

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from defend_net import DefendNets
    nets = DefendNets(dst)
    rng = np.random.default_rng(0)
    obs_dim = out[f"{agents[0]}/mean"].shape[0]
    x = rng.normal(size=(256, obs_dim)).astype(np.float32) * 2.0
    for i, a in enumerate(agents):
        want = torch_forward(ck, a, torch.from_numpy(x)).numpy()
        got = np.stack([nets(i, row) for row in x])
        err = float(np.abs(got - want).max())
        assert err < 1e-5, f"{a}: numpy differs from torch by {err}"
        print(f"[ok] {a}: numpy matches torch (max diff {err:.1e}), obs {obs_dim}")
    print(f"wrote {dst}: {len(agents)} defender networks\nPARITY OK")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
