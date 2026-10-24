"""AS-Defend-v0 at other attacker speeds: the Tello's whole range, for robustness checks and training.

The Tello SDK takes speeds of 10-100 cm/s (``speed x``), so attackers can fly anywhere in
0.1-1.0 m/s; the official task draws 0.4-0.9. Scenario, reward, observations and actions are
otherwise the official ones (the hosts' TelloDefendEnv, unchanged), so each task is scored with
``scripts/evaluate.py --task <id>`` and a policy trained on any of them is scored on AS-Defend-v0.
"""

from isaaclab.utils import configclass

from ..tello_defend.tello_defend_env_cfg import TelloDefendEnvCfg


@configclass
class Team1DefendWideSpeedEnvCfg(TelloDefendEnvCfg):
    """Attackers anywhere in the Tello's 0.1-1.0 m/s, drawn per attacker per wave."""

    attacker_speed = [0.1, 1.0]


@configclass
class Team1DefendSpeed010EnvCfg(TelloDefendEnvCfg):
    attacker_speed = [0.1, 0.1]


@configclass
class Team1DefendSpeed025EnvCfg(TelloDefendEnvCfg):
    attacker_speed = [0.25, 0.25]


@configclass
class Team1DefendSpeed050EnvCfg(TelloDefendEnvCfg):
    attacker_speed = [0.5, 0.5]


@configclass
class Team1DefendSpeed075EnvCfg(TelloDefendEnvCfg):
    attacker_speed = [0.75, 0.75]


@configclass
class Team1DefendSpeed100EnvCfg(TelloDefendEnvCfg):
    attacker_speed = [1.0, 1.0]
