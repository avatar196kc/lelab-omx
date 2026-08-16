"""Isaac Lab 환경 등록 모듈."""

from __future__ import annotations

import gymnasium as gym

gym.register(
    id="Isaac-Lift-Bimanual-OMX-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bimanual_lift_env_cfg:OmxBimanualLiftEnvCfg",
        "rsl_rl_cfg_entry_point": "lelab_sim.agents.rsl_rl_ppo_cfg:OmxBimanualLiftPPORunnerCfg",
    },
)
