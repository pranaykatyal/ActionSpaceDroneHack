# Paste into simulation_infra/source/action_space_kit/action_space_kit/tasks/direct/team1_defend/__init__.py
# (branch `shibani` already has it), with team1_defend_3v2_env_cfg.py next to it.
# Team1-Defend-3v2-Realistic-v0 also needs team1_defend_shaped_env.py and team1_attackers.py (branch `shibani`).
import gymnasium as gym

from action_space_kit.tasks.direct.tello_defend import agents as defend_agents

# three defenders vs the same two attackers, hosts' env class and geometry (team1_defend_3v2_env_cfg.py)
gym.register(
    id="Team1-Defend-3v2-v0",
    entry_point="action_space_kit.tasks.direct.tello_defend.tello_defend_env:TelloDefendEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.team1_defend_3v2_env_cfg:Team1Defend3v2EnvCfg",
        "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

# three defenders vs two realistic attackers on the official geometry (team1_defend_3v2_realistic_env_cfg.py)
gym.register(
    id="Team1-Defend-3v2-Realistic-v0",
    entry_point=f"{__name__}.team1_defend_shaped_env:Team1DefendRealisticEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.team1_defend_3v2_realistic_env_cfg:Team1Defend3v2RealisticEnvCfg",
        "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
