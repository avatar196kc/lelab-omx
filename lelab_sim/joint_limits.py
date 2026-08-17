"""ROBOTIS OMX 12개 관절 한계각 및 단위 변환 유틸리티.

ROBOTIS OMX 하드웨어 공식 사양 (https://ai.robotis.com/omx/hardware_omx.html)
및 lerobot.robots.omx_follower (RANGE_M100_100, RANGE_0_100) 기준.

lerobot 정규화 표준 — `OmxFollower.calibrate()`가 **전 모터**(그리퍼 포함)에
`range_min=0, range_max=4095`를 쓰므로, 두 모드 모두 "엔코더 1회전" 기준입니다:

- 팔 5관절 (`RANGE_M100_100`): 틱 0~4095 -> [-100, +100]. 1회전에 200단위.
  -> ±100% = ±180° = ±π rad, 즉 `pct = q_rad * 100/π`
- 그리퍼 (`RANGE_0_100`): 틱 0~4095 -> [0, +100]. 1회전에 100단위.
  -> `pct = q_rad * 50/π` (팔 계수의 정확히 절반)

**기계적 가동범위(JOINT_LIMITS_RAD)는 정규화 기준이 아닙니다.** 그것으로 정규화하면
관절마다 배율이 달라져 실기와 어긋납니다. 그리퍼 기계 범위 0~100°는 위 수식에서
0~27.8%로 나오는 것이 정상이며, 이를 0~100%로 "고치면" 실기와 3.6배 어긋납니다.

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

# 1회전(2π rad)당 정규화 단위: 팔은 200(RANGE_M100_100), 그리퍼는 100(RANGE_0_100).
_ARM_PCT_PER_RAD = 100.0 / math.pi  # ≈ 31.831
_GRIPPER_PCT_PER_RAD = 50.0 / math.pi  # ≈ 15.915


def rad_to_omx_pct(q_rad: np.ndarray) -> np.ndarray:
    """12개 관절의 라디안 각도를 lerobot OMX 정규화 백분율로 선형 변환합니다.

    두 계수 모두 엔코더 1회전(틱 0~4095)을 기준으로 하며, 기계적 가동범위와는
    무관합니다 (모듈 docstring 참조).

    - 팔 5관절 (좌/우 각 5개): `q * 100/π`, ±π rad(±180°) -> [-100.0, 100.0]
    - 그리퍼 관절 (인덱스 5, 11): `q * 50/π`, 기계 범위 0~100° -> [0.0, 27.8]

    Args:
        q_rad: Shape [..., 12] 라디안 각도 배열.

    Returns:
        Shape [..., 12] 정규화 백분율 배열 (np.float32).
    """
    arr = np.asarray(q_rad, dtype=np.float32)
    if arr.shape[-1] != 12:
        raise ValueError(f"Expected last dimension to be 12, got shape {arr.shape}")

    out = np.empty_like(arr)

    out[..., _ARM_INDICES] = np.clip(arr[..., _ARM_INDICES] * _ARM_PCT_PER_RAD, -100.0, 100.0)

    # TODO(hardware): 스케일은 lerobot 소스에서 확정됨. 남은 미확인은 **영점**뿐 —
    # URDF의 그리퍼 0 rad(완전 닫힘)이 엔코더 틱 0과 일치하는지 실기 관측으로 1회 확인.
    # 불일치하면 여기에 오프셋 상수를 더하면 되고, 위 계수는 그대로 유효하다.
    out[..., _GRIPPER_INDICES] = np.clip(
        arr[..., _GRIPPER_INDICES] * _GRIPPER_PCT_PER_RAD, 0.0, 100.0
    )

    return out
