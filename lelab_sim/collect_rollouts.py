"""A단계 롤아웃 수집 스크립트 (Isaac Sim / Isaac Lab 전용).

PPO로 학습된 정책 모델을 실행하여 Robotis OMX 양팔 물체 들어올리기(Bimanual Lift)
성공 에피소드 궤적(관절 각도, 액션, 카메라 이미지)을 수집하고, B단계 LeRobotDataset 변환기가
요구하는 원시 디렉토리 레이아웃으로 저장합니다.

출력 디렉토리 구조 (A/B단계 계약):
<out>/
  meta.json            # {"fps": 30, "task": "...", "joint_order": [...], "num_episodes": N}
  episode_000/
    states.npz         # states float32[T,12] (라디안), actions float32[T,12] (라디안 절대 목표)
    left_top/000000.png ...
    left_wrist/000000.png ...
    right_wrist/000000.png ...

성공 판정 4조건 (모두 만족해야 저장):
1. 물체 최저점 높이 상승 >= 0.10 m (초기 위치 대비)
2. 연속 15 스텝(0.5초 @ 30Hz) 이상 상승 유지
3. 유지 구간 내내 양 그리퍼가 물체와 접촉/파지 유지
4. 전체 에피소드 동안 어떤 관절도 기계적 한계에 도달하지 않음
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# =============================================================================
# 1. CLI 파서 정의
# =============================================================================
parser = argparse.ArgumentParser(
    description="Collect successful rollouts for LeRobot dataset from Isaac Lab."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Bimanual-OMX-v0",
    help="Name of the registered Gymnasium environment (default: Isaac-Lift-Bimanual-OMX-v0).",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the trained RSL-RL PPO model checkpoint (.pt file).",
)
parser.add_argument(
    "--out",
    type=str,
    required=True,
    help="Output directory path for storing raw rollouts (meta.json and episode_*).",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=50,
    help="Target number of successful episodes to collect (default: 50).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=16,
    help="Number of parallel environments to simulate (default: 16).",
)
parser.add_argument(
    "--lift_height_threshold",
    type=float,
    default=0.10,
    help="Minimum object lift height in meters relative to initial position (default: 0.10).",
)
parser.add_argument(
    "--hold_steps",
    type=int,
    default=15,
    help="Number of consecutive steps required to hold the lift (default: 15).",
)
parser.add_argument(
    "--max_steps_per_episode",
    type=int,
    default=300,
    help="Maximum steps per episode before timeout (default: 300).",
)


# =============================================================================
# 2. 헬퍼 클래스 및 함수 (Isaac Sim 초기화 전/후 공용)
# =============================================================================
CAMERAS: tuple[str, ...] = ("left_top", "left_wrist", "right_wrist")


def validate_out_dir(out_path_str: str) -> Path:
    """출력 디렉토리 안전성을 검증합니다 (기존 수집 데이터 보호).

    Args:
        out_path_str: 출력 디렉토리 경로 문자열.

    Returns:
        검증된 Path 객체.

    Raises:
        FileExistsError: 디렉토리가 이미 존재하고 비어있지 않은 경우.
    """
    out_path = Path(out_path_str).resolve()
    if out_path.exists() and any(out_path.iterdir()):
        raise FileExistsError(
            f"Output directory '{out_path}' already exists and is not empty. "
            "Aborting to protect existing rollout data. Please specify a new or empty directory."
        )
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def get_joint_order_indices(robot_joint_names: list[str], target_order: list[str]) -> list[int]:
    """로봇 관절 이름 리스트에서 target_order에 해당하는 인덱스 매핑을 생성합니다.

    Args:
        robot_joint_names: Isaac Lab 로봇 객체(Articulation)의 관절명 리스트.
        target_order: 표준 정렬 순서 리스트 (예: JOINT_ORDER).

    Returns:
        target_order의 각 관절에 매핑되는 robot_joint_names 내의 인덱스 리스트.

    Raises:
        ValueError: target_order에 포함된 관절이 robot_joint_names(또는 alias)에 존재하지 않는 경우.
    """
    name_alias = {
        "left_joint1": "left_shoulder_pan",
        "left_joint2": "left_shoulder_lift",
        "left_joint3": "left_elbow_flex",
        "left_joint4": "left_wrist_flex",
        "left_joint5": "left_wrist_roll",
        "left_gripper_joint_1": "left_gripper",
        "right_joint1": "right_shoulder_pan",
        "right_joint2": "right_shoulder_lift",
        "right_joint3": "right_elbow_flex",
        "right_joint4": "right_wrist_flex",
        "right_joint5": "right_wrist_roll",
        "right_gripper_joint_1": "right_gripper",
    }
    mapped_names = [name_alias.get(name, name) for name in robot_joint_names]
    indices: list[int] = []
    for target in target_order:
        if target in mapped_names:
            indices.append(mapped_names.index(target))
        elif target in robot_joint_names:
            indices.append(robot_joint_names.index(target))
        else:
            raise ValueError(
                f"Target joint '{target}' not found in robot joint names: {robot_joint_names} "
                f"(resolved alias names: {mapped_names}). "
                "Please ensure the URDF / USD joint naming matches the expected OMX convention."
            )
    return indices


class EpisodeBuffer:
    """단일 시뮬레이션 환경의 에피소드 궤적 버퍼."""

    def __init__(self, initial_obj_z: float) -> None:
        self.initial_obj_z: float = initial_obj_z
        self.states: list[Any] = []
        self.actions: list[Any] = []
        self.images: dict[str, list[Any]] = {cam: [] for cam in CAMERAS}
        self.object_pos: list[Any] = []
        self.contacts: list[bool] = []
        self.joint_limits_valid: list[bool] = []

    def reset(self, initial_obj_z: float) -> None:
        """버퍼를 리셋합니다."""
        self.initial_obj_z = initial_obj_z
        self.states.clear()
        self.actions.clear()
        for cam in CAMERAS:
            self.images[cam].clear()
        self.object_pos.clear()
        self.contacts.clear()
        self.joint_limits_valid.clear()

    def append_step(
        self,
        state: Any,
        action: Any,
        imgs: dict[str, Any],
        obj_pos: Any,
        contact: bool,
        limits_ok: bool,
    ) -> None:
        """스텝 데이터를 버퍼에 추가합니다."""
        self.states.append(state)
        self.actions.append(action)
        for cam, img in imgs.items():
            self.images[cam].append(img)
        self.object_pos.append(obj_pos)
        self.contacts.append(contact)
        self.joint_limits_valid.append(limits_ok)

    def evaluate_success(
        self,
        lift_height_threshold: float = 0.10,
        hold_steps: int = 15,
    ) -> tuple[bool, str]:
        """4중 성공 기준을 평가합니다.

        1. 관절 기계적 한계 초과 여부 (전체 에피소드)
        2. 물체 최저점 높이 상승 >= lift_height_threshold
        3. 연속 hold_steps 동안 상승 유지
        4. 유지 구간 내내 양 그리퍼 접촉
        """
        import numpy as np

        if len(self.states) < hold_steps:
            return False, f"Episode too short ({len(self.states)} < {hold_steps})"

        # 1. 관절 한계 초과 검증 (전체 에피소드)
        if not all(self.joint_limits_valid):
            return False, "Joint mechanical limits exceeded"

        # 2. 물체 상승 높이 검증 (초기 높이 대비)
        obj_arr = np.array(self.object_pos)  # [T, 3]
        lift_deltas = obj_arr[:, 2] - self.initial_obj_z
        is_lifted = lift_deltas >= lift_height_threshold

        # 3. 양 그리퍼 접촉 여부
        contacts_arr = np.array(self.contacts, dtype=bool)

        # 4. 연속 hold_steps 동안 상승 유지 AND 양손 접촉
        valid_hold = is_lifted & contacts_arr
        max_consecutive = 0
        current_consecutive = 0
        for valid in valid_hold:
            if valid:
                current_consecutive += 1
                if current_consecutive > max_consecutive:
                    max_consecutive = current_consecutive
            else:
                current_consecutive = 0

        if max_consecutive < hold_steps:
            return (
                False,
                f"Lift & grasp held for {max_consecutive} steps (< required {hold_steps})",
            )

        return True, f"Success: Lift & grasp held for {max_consecutive} steps"


def save_episode(
    out_dir: Path,
    episode_idx: int,
    states: Any,
    actions: Any,
    images: dict[str, list[Any]],
) -> Path:
    """단일 성공 에피소드를 디스크에 저장합니다 (A/B단계 계약 레이아웃).

    Args:
        out_dir: 롤아웃 루트 디렉토리.
        episode_idx: 에피소드 번호 (0, 1, 2, ...).
        states: [T, 12] float32 라디안 관절 각도 배열.
        actions: [T, 12] float32 라디안 절대 목표 각도 배열.
        images: 카메라별 RGB 이미지 리스트.

    Returns:
        저장된 에피소드 디렉토리 Path.
    """
    import cv2
    import numpy as np

    ep_dir = out_dir / f"episode_{episode_idx:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    # 1. states.npz 저장
    np.savez_compressed(
        ep_dir / "states.npz",
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
    )

    # 2. 카메라별 PNG 이미지 저장
    for cam_name, frame_list in images.items():
        cam_dir = ep_dir / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        for f_idx, img in enumerate(frame_list):
            img_np = np.asarray(img)
            # float [0, 1] 처리
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255.0).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
            # RGBA인 경우 RGB로 변환
            if img_np.ndim == 3 and img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]
            # cv2.imwrite는 BGR을 요구하므로 RGB -> BGR 변환
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                bgr_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                bgr_img = img_np

            img_path = cam_dir / f"{f_idx:06d}.png"
            cv2.imwrite(str(img_path), bgr_img)

    return ep_dir


# =============================================================================
# 3. 메인 실행 함수
# =============================================================================
def main() -> None:
    # 1. AppLauncher 인자 등록 및 파싱
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # 카메라 렌더링 활성화 강제 (롤아웃 수집에 필수)
    if hasattr(args_cli, "enable_cameras") and not args_cli.enable_cameras:
        args_cli.enable_cameras = True

    # 2. 출력 디렉토리 안전 검사
    out_path = validate_out_dir(args_cli.out)

    # 3. SimulationApp 실행
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 4. Isaac Sim 초기화 후 모듈 로드
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner, RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

    import lelab_sim.envs  # noqa: F401 - registers Isaac-Lift-Bimanual-OMX-v0
    from lelab_sim.agents.rsl_rl_ppo_cfg import OmxBimanualLiftPPORunnerCfg
    from lelab_sim.joint_limits import JOINT_LIMITS_RAD, JOINT_ORDER

    q_min_arr = np.array([JOINT_LIMITS_RAD[j][0] for j in JOINT_ORDER], dtype=np.float32)
    q_max_arr = np.array([JOINT_LIMITS_RAD[j][1] for j in JOINT_ORDER], dtype=np.float32)

    # 5. 환경 생성 및 래핑
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    if hasattr(env_cfg, "enable_cameras"):
        env_cfg.enable_cameras = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env)

    # 6. RSL-RL PPO 에이전트 설정 로드 및 체크포인트 복원
    try:
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    except Exception:
        agent_cfg = OmxBimanualLiftPPORunnerCfg()

    runner = RslRlOnPolicyRunner(
        wrapped_env,
        agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else agent_cfg,
        log_dir=None,
        device=wrapped_env.device,
    )
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=wrapped_env.device)

    # 7. 로봇 및 물체 에셋 핸들 조회
    unwrapped_scene = env.unwrapped.scene
    robot = unwrapped_scene["robot"]
    object_asset = unwrapped_scene["object"]

    # 관절 순서 매핑 인덱스 생성
    joint_indices = get_joint_order_indices(robot.data.joint_names, JOINT_ORDER)

    # 정책 액션 -> JOINT_ORDER 매핑 인덱스 생성
    # (ActionsCfg: arm_action [left 1-5, right 1-5] + gripper_action [left_gripper, right_gripper])
    action_joint_names = [
        "left_joint1",
        "left_joint2",
        "left_joint3",
        "left_joint4",
        "left_joint5",
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "left_gripper_joint_1",
        "right_gripper_joint_1",
    ]
    action_indices = get_joint_order_indices(action_joint_names, list(JOINT_ORDER))

    # 양손 엔드이펙터 링크 인덱스 탐색
    body_names = list(robot.data.body_names)
    left_ee_idx = body_names.index("left_link5") if "left_link5" in body_names else 0
    right_ee_idx = body_names.index("right_link5") if "right_link5" in body_names else 0

    # 환경별 에피소드 버퍼 초기화
    initial_obj_z = object_asset.data.root_pos_w[:, 2].detach().cpu().numpy()
    env_buffers = [EpisodeBuffer(float(initial_obj_z[i])) for i in range(args_cli.num_envs)]

    num_saved = 0
    total_attempts = 0

    print("=" * 60)
    print(f"Starting Rollout Collection: Target {args_cli.num_episodes} successful episodes")
    print(f"Task: {args_cli.task}")
    print(f"Environments: {args_cli.num_envs}")
    print(f"Output Directory: {out_path}")
    print(f"Lift Threshold: {args_cli.lift_height_threshold}m, Hold Steps: {args_cli.hold_steps}")
    print("=" * 60)

    # 8. 롤아웃 수집 루프
    obs, _ = wrapped_env.get_observations()

    with torch.inference_mode():
        while simulation_app.is_running() and num_saved < args_cli.num_episodes:
            # 1) 현재(pre-step) 상태, 물체 위치, 엔드이펙터 위치 및 카메라 프레임 캡처
            joint_pos_all = robot.data.joint_pos[:, joint_indices].detach().cpu().numpy()
            obj_pos_all = object_asset.data.root_pos_w[:, :3].detach().cpu().numpy()
            ee_pos_all = robot.data.body_pos_w.detach().cpu().numpy()

            # 카메라 프레임 수집 (pre-step 관측 시점)
            cam_frames: dict[str, np.ndarray] = {}
            for cam in CAMERAS:
                cam_sensor = unwrapped_scene[f"{cam}_cam"]
                cam_frames[cam] = cam_sensor.data.output["rgb"].detach().cpu().numpy()

            # 2) 정책 추론 및 상대 델타 기반 절대 목표 관절 각도 계산 (Spec §4.2: q_target = clip(q + 0.05 * action, q_min, q_max))
            actions = policy(obs)
            action_deltas = actions.detach().cpu().numpy()
            action_deltas_ordered = action_deltas[:, action_indices]
            target_joint_pos_all = np.clip(
                joint_pos_all + 0.05 * action_deltas_ordered,
                q_min_arr,
                q_max_arr,
            )

            # 3) 환경 스텝 진행 (내부적으로 done 시 자동 reset 처리됨)
            obs, _, dones, _ = wrapped_env.step(actions)

            # 4) 각 환경별 스텝 기록 및 에피소드 종료/성공 평가
            for env_idx in range(args_cli.num_envs):
                if num_saved >= args_cli.num_episodes:
                    break

                cur_state = joint_pos_all[env_idx]
                cur_action = target_joint_pos_all[env_idx]
                cur_obj_pos = obj_pos_all[env_idx]

                # 접촉/파지 판정: 양손 엔드이펙터와 물체 중심 간 거리
                left_ee_pos = ee_pos_all[env_idx, left_ee_idx, :3]
                right_ee_pos = ee_pos_all[env_idx, right_ee_idx, :3]
                d_left = float(np.linalg.norm(left_ee_pos - cur_obj_pos))
                d_right = float(np.linalg.norm(right_ee_pos - cur_obj_pos))
                is_contact = (d_left <= 0.12) and (d_right <= 0.12)

                # 기계적 관절 한계 판정
                limits_ok = bool(
                    np.all((cur_state >= q_min_arr - 1e-3) & (cur_state <= q_max_arr + 1e-3))
                )

                # 카메라 프레임 dict
                cur_imgs = {cam: cam_frames[cam][env_idx] for cam in CAMERAS}

                # 버퍼에 추가
                env_buffers[env_idx].append_step(
                    state=cur_state,
                    action=cur_action,
                    imgs=cur_imgs,
                    obj_pos=cur_obj_pos,
                    contact=is_contact,
                    limits_ok=limits_ok,
                )

                # 에피소드 종료 시 성공 판정 및 저장
                if dones[env_idx]:
                    total_attempts += 1
                    is_success, reason = env_buffers[env_idx].evaluate_success(
                        lift_height_threshold=args_cli.lift_height_threshold,
                        hold_steps=args_cli.hold_steps,
                    )

                    if is_success:
                        save_episode(
                            out_dir=out_path,
                            episode_idx=num_saved,
                            states=np.stack(env_buffers[env_idx].states),
                            actions=np.stack(env_buffers[env_idx].actions),
                            images=env_buffers[env_idx].images,
                        )
                        num_saved += 1
                        success_rate = 100.0 * num_saved / total_attempts
                        print(
                            f"[SUCCESS] Episode {num_saved:03d}/{args_cli.num_episodes} saved. "
                            f"(Attempts: {total_attempts}, Success Rate: {success_rate:.1f}%)"
                        )
                    else:
                        success_rate = 100.0 * num_saved / total_attempts
                        print(
                            f"[DISCARD] Episode discarded ({reason}). "
                            f"(Progress: {num_saved}/{args_cli.num_episodes}, Success Rate: {success_rate:.1f}%)"
                        )

                    # 환경 버퍼 리셋 (done 이후 다음 에피소드를 위한 초기 물체 높이)
                    new_obj_z = float(object_asset.data.root_pos_w[env_idx, 2].detach().cpu().item())
                    env_buffers[env_idx].reset(new_obj_z)

    # 9. meta.json 저장 (A/B단계 계약 메타데이터)
    meta_info = {
        "fps": 30,
        "task": args_cli.task,
        "joint_order": list(JOINT_ORDER),
        "num_episodes": num_saved,
    }
    meta_file = out_path / "meta.json"
    meta_file.write_text(json.dumps(meta_info, indent=2), encoding="utf-8")

    # 10. 완료 리포트 및 디스크 용량 출력
    total_bytes = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)

    print("\n" + "=" * 60)
    print("Rollout Collection Finished!")
    print(f"Total Successful Episodes Saved: {num_saved}/{args_cli.num_episodes}")
    print(f"Total Episode Attempts: {total_attempts}")
    print(f"Final Success Rate: {100.0 * num_saved / max(1, total_attempts):.1f}%")
    print(f"Output Directory: {out_path}")
    print(f"Disk Usage: {total_mb:.2f} MB")
    print("=" * 60)
    print("[NEXT STEP] Convert to LeRobot dataset format (B-stage):")
    print(f"  python -m lelab_sim.export_lerobot --raw {out_path} --repo_id <user/name> --task '{args_cli.task}'")
    print("=" * 60 + "\n")

    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
