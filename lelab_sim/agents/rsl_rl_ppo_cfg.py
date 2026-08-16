"""Robotis OMX 양팔 물체 들어올리기(Bimanual Lift) PPO 설정 (Isaac Lab / RSL-RL 전용)."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class OmxBimanualLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Robotis OMX 양팔 물체 들어올리기(Bimanual Lift)를 위한 RSL-RL PPO 러너 설정."""

    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "omx_bimanual_lift"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
