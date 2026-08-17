"""OMX 양팔 태스크 전용 MDP 항 (Isaac Lab 전용).

Isaac Lab v2.3.2의 `isaaclab.envs.mdp`에는 양팔 조작에 쓸 수 있는 보상 항이 없고,
`isaaclab_tasks...manipulation.lift.mdp`의 항들은 **단일 엔드이펙터** 전제로
작성되어 있습니다(`object_ee_distance`는 `FrameTransformer` 하나의
`target_pos_w[..., 0, :]`만 읽습니다). 따라서 양팔 의미를 명시한 항을 직접 정의합니다.
"""

from __future__ import annotations

from lelab_sim.envs.mdp.rewards import (
    bimanual_object_distance,
    bimanual_object_grasped,
    both_grippers_touching,
    object_lift_height,
)

__all__ = [
    "bimanual_object_distance",
    "bimanual_object_grasped",
    "both_grippers_touching",
    "object_lift_height",
]
