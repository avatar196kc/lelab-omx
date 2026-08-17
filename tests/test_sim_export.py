"""B단계(원시 롤아웃 -> LeRobotDataset) 변환기 테스트.

Isaac Sim 없이 Python 3.12 환경에서 전부 실행된다. 이 파일이 import에
성공한다는 사실 자체가 "B단계는 Isaac에 의존하지 않는다"는 Global
Constraint의 회귀 테스트다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lelab_sim.joint_limits import JOINT_ORDER, rad_to_omx_pct

CAMERAS = ("left_top", "left_wrist", "right_wrist")
GRIPPER_IDX = (5, 11)
ARM_IDX = tuple(i for i in range(12) if i not in GRIPPER_IDX)

# lerobot 틱 0~4095 정규화로 직접 계산한 기대값 (구현과 독립).
# 그리퍼: RANGE_0_100 은 1회전(360°)에 100단위 -> deg / 3.6
_GRIPPER_PCT_AT_50DEG = 50.0 / 3.6  # 13.888...
_GRIPPER_PCT_AT_100DEG = 100.0 / 3.6  # 27.777...

EXPECTED_FEATURES = {
    "observation.state",
    "action",
    "observation.images.left_top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    # LeRobotDataset이 자동으로 채우는 키
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


def _boundary_states(num_frames: int) -> np.ndarray:
    """프레임 0/1/2 = 물리 각도 기준점 (-180°/90°/180° 팔, 0°/50°/100° 그리퍼).

    `JOINT_LIMITS_RAD`에 의존하지 않는 독립적인 실기 기준 하드코딩 물리 각도입니다.
    기대 백분율은 lerobot의 틱 0~4095 정규화(팔 200단위/회전, 그리퍼 100단위/회전)로
    직접 계산한 값이며, 구현이 바뀌어도 이 표는 변하지 않습니다.

    | frame | 팔 각도 | 팔 % | 그리퍼 각도 | 그리퍼 % |
    |---|---|---|---|---|
    | 0 | -180° | -100 | 0°  | 0     |
    | 1 | +90°  | +50  | 50° | 13.889 |
    | 2 | +180° | +100 | 100°| 27.778 |
    """
    states = np.zeros((num_frames, 12), dtype=np.float32)
    # Frame 0
    states[0, ARM_IDX] = -np.pi
    states[0, GRIPPER_IDX] = 0.0
    # Frame 1
    states[1, ARM_IDX] = np.pi / 2.0
    states[1, GRIPPER_IDX] = 50.0 * np.pi / 180.0
    # Frame 2
    states[2, ARM_IDX] = np.pi
    states[2, GRIPPER_IDX] = 100.0 * np.pi / 180.0

    return states


def _create_fake_raw_rollouts(root: Path, num_episodes: int = 2, num_frames: int = 10) -> None:
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "fps": 30,
        "task": "Lift the box with both arms",
        "joint_order": list(JOINT_ORDER),
        "num_episodes": num_episodes,
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    for ep_idx in range(num_episodes):
        ep_dir = root / f"episode_{ep_idx:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        states = _boundary_states(num_frames)
        np.savez_compressed(ep_dir / "states.npz", states=states, actions=states.copy())

        for cam in CAMERAS:
            cam_dir = ep_dir / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            for f_idx in range(num_frames):
                # H.264는 짝수 해상도를 요구한다.
                img = np.full((48, 64, 3), (ep_idx * 50, f_idx * 10, 100), dtype=np.uint8)
                cv2.imwrite(str(cam_dir / f"{f_idx:06d}.png"), img)


def test_rad_to_omx_pct_boundaries():
    """변환 함수 단독 검증 — 실기 하드코딩 각도 기준 경계값이 정확히 매핑되는가."""
    states = _boundary_states(3)
    pct = rad_to_omx_pct(states)

    assert np.allclose(pct[0, ARM_IDX], -100.0), "-180°(-π rad)는 팔에서 -100%"
    assert np.allclose(pct[1, ARM_IDX], 50.0), "+90°(π/2 rad)는 팔에서 +50%"
    assert np.allclose(pct[2, ARM_IDX], 100.0), "+180°(π rad)는 팔에서 +100%"

    assert np.allclose(pct[0, GRIPPER_IDX], 0.0), "0 rad(0°)는 그리퍼에서 0%"
    assert np.allclose(pct[1, GRIPPER_IDX], _GRIPPER_PCT_AT_50DEG), "50°는 그리퍼에서 13.889%"
    assert np.allclose(pct[2, GRIPPER_IDX], _GRIPPER_PCT_AT_100DEG), "100°는 그리퍼에서 27.778%"


def test_export_raw_to_lerobot(tmp_path: Path):
    from lelab_sim.export_lerobot import export_raw_to_lerobot
    from lerobot.datasets import LeRobotDataset

    raw_path = tmp_path / "raw"
    _create_fake_raw_rollouts(raw_path, num_episodes=2, num_frames=10)

    repo_id = "test_user/omx_bimanual_sim_test"
    root = tmp_path / "hf" / repo_id

    out_dir = export_raw_to_lerobot(
        raw_dir=raw_path,
        repo_id=repo_id,
        task="Lift the box with both arms",
        fps=30,
        root=root,
    )

    assert out_dir.exists()
    info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))

    # 키 집합이 "정확히" 일치해야 한다 (잉여 키 유입 차단).
    assert set(info["features"]) == EXPECTED_FEATURES
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 20
    assert info["fps"] == 30

    # 저장된 값이 OMX 정규화 단위인지 — 파이프라인 전 구간 검증.
    ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    state0 = np.asarray(ds[0]["observation.state"])
    assert np.allclose(state0[list(ARM_IDX)], -100.0)
    assert np.allclose(state0[list(GRIPPER_IDX)], 0.0)

    state1 = np.asarray(ds[1]["observation.state"])
    assert np.allclose(state1[list(ARM_IDX)], 50.0)
    assert np.allclose(state1[list(GRIPPER_IDX)], _GRIPPER_PCT_AT_50DEG)

    state2 = np.asarray(ds[2]["observation.state"])
    assert np.allclose(state2[list(ARM_IDX)], 100.0)
    assert np.allclose(state2[list(GRIPPER_IDX)], _GRIPPER_PCT_AT_100DEG)


def test_invalid_repo_id_raises(tmp_path: Path):
    """repo_id 검증은 raw_dir을 건드리기 전에 일어나야 한다.

    raw_dir이 존재하지 않으므로, 검증 순서가 뒤바뀌면 ValueError 대신
    FileNotFoundError가 난다.
    """
    from lelab_sim.export_lerobot import export_raw_to_lerobot

    with pytest.raises(ValueError, match="namespace/name"):
        export_raw_to_lerobot(raw_dir=tmp_path / "does_not_exist", repo_id="no_namespace", task="t")


def test_get_joint_order_indices_mapping():
    """collect_rollouts.py의 관절 순서 매핑 및 fail-fast 예외 검증."""
    from lelab_sim.collect_rollouts import get_joint_order_indices

    # 1. URDF 기본 이름 매핑 검증
    urdf_names = [
        "left_joint1",
        "left_joint2",
        "left_joint3",
        "left_joint4",
        "left_joint5",
        "left_gripper_joint_1",
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "right_gripper_joint_1",
    ]
    indices = get_joint_order_indices(urdf_names, list(JOINT_ORDER))
    assert indices == list(range(12))

    # 2. 순서가 뒤섞인 경우 올바른 인덱스 반환 검증
    shuffled_names = list(reversed(urdf_names))
    shuffled_indices = get_joint_order_indices(shuffled_names, list(JOINT_ORDER))
    assert shuffled_indices == list(range(11, -1, -1))

    # 3. 관절명 누락 시 fail-fast ValueError 발생 검증
    incomplete_names = ["left_joint1", "left_joint2"]
    with pytest.raises(ValueError, match="not found in robot joint names"):
        get_joint_order_indices(incomplete_names, list(JOINT_ORDER))

