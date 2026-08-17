"""ROBOTIS OMX 12개 관절 한계각 및 단위 변환 유틸리티.

ROBOTIS OMX 하드웨어 공식 사양 (https://ai.robotis.com/omx/hardware_omx.html)
및 lerobot.robots.omx_follower (RANGE_M100_100, RANGE_0_100) 기준.

lerobot 정규화 표준:
- 팔 5관절 (RANGE_M100_100): 엔코더 틱 0~4095가 [-100, +100]%에 매핑되므로, ±100% = ±180° = ±π rad.
- 그리퍼 (RANGE_0_100): [0, +100]%.

주의: 이 모듈은 Isaac Sim 및 PyTorch를 일체 import하지 않으며 numpy만 사용합니다.
"""

from __future__ import annotations

import math

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

# 시뮬레이션 환경의 기계적 가동범위 (q_min, q_max) in radians.
# 도달 불가능한 자세 차단 및 롤아웃 성공 판정 한계 체크에 사용됩니다.
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

_GRIPPER_INDICES = (5, 11)
_ARM_INDICES = tuple(i for i in range(12) if i not in _GRIPPER_INDICES)
_GRIPPER_MAX_RAD = 100.0 * math.pi / 180.0  # 100 degrees in radians (1.745 rad)


def rad_to_omx_pct(q_rad: np.ndarray) -> np.ndarray:
    """12개 관절의 라디안 각도를 lerobot OMX 정규화 백분율로 선형 변환합니다.

    - 팔 5관절 (좌/우 각 5개): ±π rad (±180°) -> [-100.0, 100.0]
    - 그리퍼 관절 (인덱스 5, 11): 0 ~ 1.745 rad (0 ~ 100°) -> [0.0, 100.0]

    Args:
        q_rad: Shape [..., 12] 라디안 각도 배열.

    Returns:
        Shape [..., 12] 정규화 백분율 배열 (np.float32).
    """
    arr = np.asarray(q_rad, dtype=np.float32)
    if arr.shape[-1] != 12:
        raise ValueError(f"Expected last dimension to be 12, got shape {arr.shape}")

    out = np.empty_like(arr)

    # 팔 관절: ±π rad = ±180° = ±100% (q_rad * 100 / π)
    out[..., _ARM_INDICES] = arr[..., _ARM_INDICES] * (100.0 / math.pi)
    out[..., _ARM_INDICES] = np.clip(out[..., _ARM_INDICES], -100.0, 100.0)

    # 그리퍼 관절: 0 ~ 100° (0 ~ 1.745 rad) -> [0, 100]%
    # TODO: 실기 OMX 하드웨어 또는 lerobot 엔코더 틱 원점(완전 닫힘 틱)과 대조 확인 필요
    out[..., _GRIPPER_INDICES] = arr[..., _GRIPPER_INDICES] * (100.0 / _GRIPPER_MAX_RAD)
    out[..., _GRIPPER_INDICES] = np.clip(out[..., _GRIPPER_INDICES], 0.0, 100.0)

    return out
