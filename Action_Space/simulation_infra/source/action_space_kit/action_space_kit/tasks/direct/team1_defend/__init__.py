"""Team 1's training variants of the defend task. Scored on the official AS-Defend-v0."""

import gymnasium as gym

from ..tello_defend import agents as defend_agents

gym.register(
    id="Team1-Defend-Robust-v0",
    entry_point=f"{__name__}.team1_defend_env:Team1DefendRobustEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.team1_defend_env_cfg:Team1DefendRobustEnvCfg",
        "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

for _id, _cfg in [
    ("Team1-Defend-Shaped-v0", "Team1DefendShapedEnvCfg"),
    ("Team1-Defend-Shaped-Crash-v0", "Team1DefendShapedCrashEnvCfg"),
    ("Team1-Defend-Single-v0", "Team1DefendSingleEnvCfg"),
    ("Team1-Defend-Single-Shaped-v0", "Team1DefendSingleShapedEnvCfg"),
    ("Team1-Defend-Weave05-v0", "Team1DefendWeave05EnvCfg"),
    ("Team1-Defend-Weave10-v0", "Team1DefendWeave10EnvCfg"),
]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.team1_defend_shaped_env:Team1DefendShapedEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.team1_defend_shaped_env:{_cfg}",
            "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )

gym.register(
    id="Team1-Defend-Robust-v1",
    entry_point=f"{__name__}.team1_defend_env:Team1DefendRobustEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.team1_defend_env_cfg:Team1DefendRobustV1EnvCfg",
        "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

for _id, _entry, _cfg in [
    ("Team1-Defend-Realistic-v0", "team1_defend_shaped_env:Team1DefendRealisticEnv", "team1_defend_shaped_env:Team1DefendRealisticEnvCfg"),
    ("Team1-Defend-Robust-Realistic-v0", "team1_defend_env:Team1DefendRobustRealisticEnv", "team1_defend_env_cfg:Team1DefendRobustRealisticEnvCfg"),
    ("Team1-Defend-Round-Official-v0", "team1_defend_shaped_env:Team1DefendShapedEnv", "team1_defend_shaped_env:Team1DefendRoundOfficialEnvCfg"),
    ("Team1-Defend-Round-Realistic-v0", "team1_defend_shaped_env:Team1DefendRealisticEnv", "team1_defend_shaped_env:Team1DefendRoundRealisticEnvCfg"),
    ("Team1-Defend-IRL-v0", "team1_defend_env:Team1DefendIRLEnv", "team1_defend_env_cfg:Team1DefendIRLEnvCfg"),
    ("Team1-Defend-Stagger-v0", "team1_defend_stagger_env:Team1DefendStaggerEnv", "team1_defend_stagger_env:Team1DefendStaggerEnvCfg"),
    ("Team1-Defend-Fitted-v0", "team1_defend_fitted_env:Team1DefendFittedEnv", "team1_defend_fitted_env:Team1DefendFittedEnvCfg"),
    ("Team1-Defend-Fitted-Tau015-v0", "team1_defend_fitted_env:Team1DefendFittedEnv", "team1_defend_fitted_env:Team1DefendFittedTau015EnvCfg"),
    ("Team1-Defend-Fitted-Tau03-v0", "team1_defend_fitted_env:Team1DefendFittedEnv", "team1_defend_fitted_env:Team1DefendFittedTau03EnvCfg"),
    ("Team1-Defend-Fitted-Spawn6-v0", "team1_defend_fitted_env:Team1DefendFittedEnv", "team1_defend_fitted_env:Team1DefendFittedSpawn6EnvCfg"),
    ("Team1-Defend-Fitted-Slow-v0", "team1_defend_fitted_env:Team1DefendFittedEnv", "team1_defend_fitted_env:Team1DefendFittedSlowEnvCfg"),
]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.{_entry}",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_cfg}",
            "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )

# attacker speeds across the Tello's range, on the hosts' own env class (official reward)
for _id, _cfg in [
    ("Team1-Defend-WideSpeed-v0", "Team1DefendWideSpeedEnvCfg"),
    ("Team1-Defend-Speed010-v0", "Team1DefendSpeed010EnvCfg"),
    ("Team1-Defend-Speed025-v0", "Team1DefendSpeed025EnvCfg"),
    ("Team1-Defend-Speed050-v0", "Team1DefendSpeed050EnvCfg"),
    ("Team1-Defend-Speed075-v0", "Team1DefendSpeed075EnvCfg"),
    ("Team1-Defend-Speed100-v0", "Team1DefendSpeed100EnvCfg"),
]:
    gym.register(
        id=_id,
        entry_point="action_space_kit.tasks.direct.tello_defend.tello_defend_env:TelloDefendEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.team1_defend_speed_env_cfg:{_cfg}",
            "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )

# single-change ablations of Robust-Realistic (team1_defend_ablation_env.py)
for _id, _entry, _cfg in [
    ("Team1-Defend-RR-Spawn6-v0", "team1_defend_env:Team1DefendRobustRealisticEnv", "team1_defend_ablation_env:Team1DefendRRSpawn6EnvCfg"),
    ("Team1-Defend-RR-NoDelay-v0", "team1_defend_env:Team1DefendRobustRealisticEnv", "team1_defend_ablation_env:Team1DefendRRNoDelayEnvCfg"),
    ("Team1-Defend-RR-TrueSite-v0", "team1_defend_ablation_env:Team1DefendRRTrueSiteEnv", "team1_defend_env_cfg:Team1DefendRobustRealisticEnvCfg"),
]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.{_entry}",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_cfg}",
            "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )

# deploy-side layer from the laptop (drone_deployment_infra/core/defend_deploy.py): latency predictor
# and teammate separation filter around the unchanged policy (team1_defend_deploy_env.py)
for _id, _entry, _cfg in [
    ("Team1-Defend-RR-PredictCV-v0", "team1_defend_deploy_env:Team1DefendRRDeployEnv", "team1_defend_deploy_env:Team1DefendRRPredictCVEnvCfg"),
    ("Team1-Defend-RR-PredictModel-v0", "team1_defend_deploy_env:Team1DefendRRDeployEnv", "team1_defend_deploy_env:Team1DefendRRPredictModelEnvCfg"),
    ("Team1-Defend-RR-Sep-v0", "team1_defend_deploy_env:Team1DefendRRDeployEnv", "team1_defend_deploy_env:Team1DefendRRSepEnvCfg"),
    ("Team1-Defend-RR-PredictCVSep-v0", "team1_defend_deploy_env:Team1DefendRRDeployEnv", "team1_defend_deploy_env:Team1DefendRRPredictCVSepEnvCfg"),
    ("Team1-Defend-Official-Sep-v0", "team1_defend_deploy_env:Team1DefendOfficialDeployEnv", "team1_defend_deploy_env:Team1DefendOfficialSepEnvCfg"),
    ("Team1-Defend-Fitted-PredictCV-v0", "team1_defend_deploy_env:Team1DefendFittedDeployEnv", "team1_defend_deploy_env:Team1DefendFittedPredictCVEnvCfg"),
    ("Team1-Defend-Fitted-PredictCVSep-v0", "team1_defend_deploy_env:Team1DefendFittedDeployEnv", "team1_defend_deploy_env:Team1DefendFittedPredictCVSepEnvCfg"),
    ("Team1-Defend-FittedT024-v0", "team1_defend_deploy_env:Team1DefendFittedDeployEnv", "team1_defend_deploy_env:Team1DefendFittedT024EnvCfg"),
    ("Team1-Defend-FittedT024-PredictCV-v0", "team1_defend_deploy_env:Team1DefendFittedDeployEnv", "team1_defend_deploy_env:Team1DefendFittedT024PredictCVEnvCfg"),
    ("Team1-Defend-FittedT024-PredictCVSep-v0", "team1_defend_deploy_env:Team1DefendFittedDeployEnv", "team1_defend_deploy_env:Team1DefendFittedT024PredictCVSepEnvCfg"),
]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.{_entry}",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_cfg}",
            "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )

# attackers start closer than the official 6 m ring; everything else official (team1_defend_spawn_env.py)
for _id, _cfg in [
    ("Team1-Defend-Close-v0", "Team1DefendCloseEnvCfg"),
    ("Team1-Defend-Spawn25-v0", "Team1DefendSpawn25EnvCfg"),
    ("Team1-Defend-Spawn3-v0", "Team1DefendSpawn3EnvCfg"),
    ("Team1-Defend-Spawn4-v0", "Team1DefendSpawn4EnvCfg"),
    ("Team1-Defend-Spawn5-v0", "Team1DefendSpawn5EnvCfg"),
    ("Team1-Defend-Spawn6-v0", "Team1DefendSpawn6EnvCfg"),
    ("Team1-Defend-Spawn3-Def1-v0", "Team1DefendSpawn3Def1EnvCfg"),
    ("Team1-Defend-Spawn3-Def15-v0", "Team1DefendSpawn3Def15EnvCfg"),
    ("Team1-Defend-Half-v0", "Team1DefendHalfEnvCfg"),
]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.team1_defend_spawn_env:Team1DefendCloseEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.team1_defend_spawn_env:{_cfg}",
            "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )

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

# flight-network variants: realistic tasks at the official 6 m spawns (team1_defend_flight_env.py)
for _id, _entry, _cfg in [
    ("Team1-Defend-FittedT024-Spawn6-v0", "team1_defend_deploy_env:Team1DefendFittedDeployEnv", "Team1DefendFittedT024Spawn6EnvCfg"),
    ("Team1-Defend-FittedT024-Spawn6-PredictCVSep-v0", "team1_defend_deploy_env:Team1DefendFittedDeployEnv", "Team1DefendFittedT024Spawn6PredictCVSepEnvCfg"),
    ("Team1-Defend-RR-Spawn6-PredictCVSep-v0", "team1_defend_deploy_env:Team1DefendRRDeployEnv", "Team1DefendRRSpawn6PredictCVSepEnvCfg"),
]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.{_entry}",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.team1_defend_flight_env:{_cfg}",
            "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )

# the real Tello's rc rate (sample-and-hold), everything else official (team1_defend_rate_env.py)
for _id, _cfg in [
    ("Team1-Defend-Rate135-v0", "Team1DefendRateEnvCfg"),
    ("Team1-Defend-Rate10-v0", "Team1DefendRate10EnvCfg"),
    ("Team1-Defend-Rate135-Lat-v0", "Team1DefendRateLatEnvCfg"),
]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.team1_defend_rate_env:Team1DefendRateEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.team1_defend_rate_env:{_cfg}",
            "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )

# organizers' real-flight capture: the attacker must be knocked out of the air (team1_defend_knockdown_env.py)
for _id, _entry, _cfg in [
    ("Team1-Defend-Knockdown-v0", "Team1DefendKnockdownEnv", "Team1DefendKnockdownEnvCfg"),
    ("Team1-Defend-Flight-Knockdown-v0", "Team1DefendFlightKnockdownEnv", "Team1DefendFlightKnockdownEnvCfg"),
]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.team1_defend_knockdown_env:{_entry}",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.team1_defend_knockdown_env:{_cfg}",
            "skrl_cfg_entry_point": f"{defend_agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{defend_agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{defend_agents.__name__}:skrl_mappo_cfg.yaml",
        },
    )
