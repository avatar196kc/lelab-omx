"""B단계 변환기: 시뮬레이션 원시 롤아웃을 LeRobotDataset으로 인코딩.

Isaac Sim 의존성 없이 독립적으로 실행되며, 실기 녹화 데이터셋과 동일한
스키마(OMX 정규화 백분율, 카메라 키)를 갖는 LeRobotDataset을 생성합니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lelab_sim.joint_limits import JOINT_ORDER, rad_to_omx_pct
from lerobot.datasets import LeRobotDataset

CAMERAS = ("left_top", "left_wrist", "right_wrist")


def export_raw_to_lerobot(
    raw_dir: str | Path,
    repo_id: str,
    task: str,
    fps: int = 30,
    root: Path | str | None = None,
) -> Path:
    """원시 롤아웃 디렉토리에서 데이터를 읽어 LeRobotDataset으로 변환합니다.

    Args:
        raw_dir: 원시 롤아웃 데이터가 저장된 디렉토리 (meta.json 및 episode_* 포함).
        repo_id: 허브/로컬 데이터셋 식별자 ('namespace/name' 형식).
        task: 에피소드 태스크 설명 문자열.
        fps: 데이터셋 프레임 레이트 (기본값: 30).
        root: 데이터셋 저장 루트 경로 (기본값: HF_LEROBOT_HOME).

    Returns:
        생성된 데이터셋 루트 디렉토리 Path.

    Raises:
        ValueError: repo_id 형식이 잘못되었거나 메타데이터가 불일치할 때.
        FileNotFoundError: raw_dir 또는 필수 파일이 존재하지 않을 때.
    """
    # 1. repo_id 검증 (파일 I/O 전에 반드시 수행)
    if not repo_id or "/" not in repo_id or len(repo_id.split("/")) != 2:
        raise ValueError(f"repo_id must be in 'namespace/name' format, got '{repo_id}'")
    namespace, name = repo_id.split("/")
    if not namespace or not name:
        raise ValueError(f"repo_id must be in 'namespace/name' format, got '{repo_id}'")

    # 2. raw_dir 경로 확인
    raw_path = Path(raw_dir)
    if not raw_path.exists() or not raw_path.is_dir():
        raise FileNotFoundError(f"Raw rollouts directory not found: {raw_path}")

    # 3. meta.json 로드 및 검증
    meta_file = raw_path / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"meta.json not found in {raw_path}")

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta_joint_order = meta.get("joint_order")
    if meta_joint_order != list(JOINT_ORDER):
        raise ValueError(
            f"joint_order mismatch in meta.json. Expected {list(JOINT_ORDER)}, got {meta_joint_order}"
        )

    # 4. 에피소드 디렉토리 탐색
    episode_dirs = sorted([p for p in raw_path.iterdir() if p.is_dir() and (p / "states.npz").exists()])
    if not episode_dirs:
        raise ValueError(f"No episode directories found in {raw_path}")

    # 5. 카메라 해상도 확인 및 features 명세 구성
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_ORDER),),
            "names": list(JOINT_ORDER),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_ORDER),),
            "names": list(JOINT_ORDER),
        },
    }

    for cam in CAMERAS:
        cam_dir = episode_dirs[0] / cam
        img_files = sorted(cam_dir.glob("*.png"))
        if not img_files:
            raise FileNotFoundError(f"No frames found for camera '{cam}' in {cam_dir}")
        sample_img = cv2.imread(str(img_files[0]))
        if sample_img is None:
            raise ValueError(f"Failed to read image: {img_files[0]}")
        h, w, c = sample_img.shape
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        }

    dataset_root = Path(root) if root is not None else None

    # 6. LeRobotDataset 생성
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=dataset_root,
        robot_type="bimanual_omx_follower",
        features=features,
        use_videos=True,
    )

    # 7. 에피소드별 데이터 변환 및 추가
    for ep_dir in episode_dirs:
        with np.load(ep_dir / "states.npz") as npz_data:
            raw_states = npz_data["states"]
            raw_actions = npz_data["actions"]

        if len(raw_states) != len(raw_actions):
            raise ValueError(
                f"states and actions length mismatch in {ep_dir}: {len(raw_states)} vs {len(raw_actions)}"
            )

        num_frames = len(raw_states)
        states_pct = rad_to_omx_pct(raw_states)
        actions_pct = rad_to_omx_pct(raw_actions)

        # 프레임 순차 추가 (메모리 최적화: 프레임별 스트리밍 로드)
        for f_idx in range(num_frames):
            frame_dict = {
                "observation.state": states_pct[f_idx],
                "action": actions_pct[f_idx],
                "task": task,
            }
            for cam in CAMERAS:
                img_path = ep_dir / cam / f"{f_idx:06d}.png"
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing camera frame: {img_path}")
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    raise ValueError(f"Failed to read image: {img_path}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                frame_dict[f"observation.images.{cam}"] = img_rgb

            dataset.add_frame(frame_dict)

        dataset.save_episode()

    dataset.finalize()
    return Path(dataset.root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw simulation rollouts to LeRobotDataset format.")
    parser.add_argument(
        "--raw",
        type=str,
        required=True,
        help="Path to raw rollouts directory",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Dataset identifier (namespace/name)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task description string",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Dataset recording FPS (default: 30)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Custom dataset output root directory",
    )

    args = parser.parse_args()
    root_path = Path(args.root) if args.root else None

    out_dir = export_raw_to_lerobot(
        raw_dir=args.raw,
        repo_id=args.repo_id,
        task=args.task,
        fps=args.fps,
        root=root_path,
    )
    print(f"Dataset successfully exported to: {out_dir}")


if __name__ == "__main__":
    main()
