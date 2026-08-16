"""ROBOTIS OMX 12개 관절 한계각 및 단위 변환 유틸리티.

ROBOTIS OMX 하드웨어 공식 사양 (https://ai.robotis.com/omx/hardware_omx.html)
및 lerobot.robots.omx_follower (RANGE_M100_100, RANGE_0_100) 기준.

주의: 이 모듈은 Isaac Sim 및 PyTorch를 일체 import하지 않으며 numpy만 사용합니다.
"""

from __future__ import annotations

import numpy as np

JOINT_ORDER: list[str] = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
]

# (q_min, q_max) in radians
JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "left_shoulder_pan": (-4.712, 6.283),
    "left_shoulder_lift": (-2.094, 1.571),
    "left_elbow_flex": (-2.094, 1.571),
    "left_wrist_flex": (-1.745, 1.745),
    "left_wrist_roll": (-4.712, 4.712),
    "left_gripper": (0.0, 1.745),
    "right_shoulder_pan": (-4.712, 6.283),
    "right_shoulder_lift": (-2.094, 1.571),
    "right_elbow_flex": (-2.094, 1.571),
    "right_wrist_flex": (-1.745, 1.745),
    "right_wrist_roll": (-4.712, 4.712),
    "right_gripper": (0.0, 1.745),
}

_Q_MIN = np.array([JOINT_LIMITS_RAD[j][0] for j in JOINT_ORDER], dtype=np.float32)
_Q_MAX = np.array([JOINT_LIMITS_RAD[j][1] for j in JOINT_ORDER], dtype=np.float32)
_GRIPPER_INDICES = (5, 11)
_ARM_INDICES = tuple(i for i in range(12) if i not in _GRIPPER_INDICES)


def rad_to_omx_pct(q_rad: np.ndarray) -> np.ndarray:
    """12개 관절의 라디안 각도를 OMX 정규화 백분율로 선형 변환합니다.

    - 팔 5관절 (좌/우 각 5개): [-100.0, 100.0]
    - 그리퍼 관절 (인덱스 5, 11): [0.0, 100.0]

    Args:
        q_rad: Shape [..., 12] 라디안 각도 배열.

    Returns:
        Shape [..., 12] 정규화 백분율 배열 (np.float32).
    """
    arr = np.asarray(q_rad, dtype=np.float32)
    if arr.shape[-1] != 12:
        raise ValueError(f"Expected last dimension to be 12, got shape {arr.shape}")

    # 선형 재조정: (q - q_min) / (q_max - q_min)
    norm = (arr - _Q_MIN) / (_Q_MAX - _Q_MIN)

    # 팔 관절: 200 * norm - 100
    out = np.empty_like(norm)
    out[..., _ARM_INDICES] = 200.0 * norm[..., _ARM_INDICES] - 100.0
    out[..., _ARM_INDICES] = np.clip(out[..., _ARM_INDICES], -100.0, 100.0)

    # 그리퍼 관절: 100 * norm
    out[..., _GRIPPER_INDICES] = 100.0 * norm[..., _GRIPPER_INDICES]
    out[..., _GRIPPER_INDICES] = np.clip(out[..., _GRIPPER_INDICES], 0.0, 100.0)

    return out
