"""Flight-network variants: the realistic tasks at the official 6 m spawns, for ES fine-tuning.

ES (``scripts/skrl/es_train.py --task``) trains a separate network for the real Tellos on these;
the official submission is unchanged. Capture and contact rules are the official ones (0.35 m
capture radius, same physics): only the attackers' spawn ring moves back to the official 6 m.

- ``FittedT024-Spawn6``: FittedT024 (positions-only sensing, inferred asset, realistic attackers,
  0.75 m/s cap, 0.1 s command delay, 0.2 s fix age, 13.5 Hz integer rc, response matched to the
  hosts' single-lag prior) at 6 m, with no deploy layer: the baseline.
- ``FittedT024-Spawn6-PredictCVSep``: the same with the laptop's latency predictor and separation
  filter (``drone_deployment_infra/core/defend_deploy.py``): what we fly.
- ``RR-Spawn6-PredictCVSep``: Robust-Realistic (0.2 s delay, inferred asset, realistic attackers;
  official command path) at 6 m with the same layer. Fallback if ES is flat on the Fitted one.

Attackers are pessimistic for the real room: 70% weave, 0.2-1.0 m/s (Fitted: capped at the
defenders' 0.75 m/s); the hosts' own attack script flies ~0.2-0.52 m/s on smooth paths.
"""

from isaaclab.utils import configclass

from .team1_defend_deploy_env import (
    Team1DefendFittedT024EnvCfg,
    Team1DefendFittedT024PredictCVSepEnvCfg,
    Team1DefendRRPredictCVSepEnvCfg,
)


@configclass
class Team1DefendFittedT024Spawn6EnvCfg(Team1DefendFittedT024EnvCfg):
    spawn_radius_range = [6.0, 6.0]


@configclass
class Team1DefendFittedT024Spawn6PredictCVSepEnvCfg(Team1DefendFittedT024PredictCVSepEnvCfg):
    spawn_radius_range = [6.0, 6.0]


@configclass
class Team1DefendRRSpawn6PredictCVSepEnvCfg(Team1DefendRRPredictCVSepEnvCfg):
    spawn_radius_range = [6.0, 6.0]
