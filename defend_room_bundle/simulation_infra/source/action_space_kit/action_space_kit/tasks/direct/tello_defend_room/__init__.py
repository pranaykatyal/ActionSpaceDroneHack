"""Defend task in the real flying space (tag 10 -> tag 6 corridor), 3 defenders vs 2 attackers."""

import gymnasium as gym

from . import agents

gym.register(
    id="AS-Defend-Room-v0",
    entry_point=f"{__name__}.tello_defend_room_env:TelloDefendRoomEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_defend_room_env_cfg:TelloDefendRoomEnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
