"""양팔 Lift 태스크 보상 항 (Isaac Lab 전용).

Isaac Lab v2.3.2 소스 대조 결과:
- `mdp.object_is_grasped` / `mdp.object_lift_height` 는 저장소 어디에도 없습니다.
- `mdp.object_ee_distance` 는 `isaaclab.envs.mdp`가 아니라 lift 태스크 로컬 모듈에만
  있고, 시그니처가 `(env, std, object_cfg, ee_frame_cfg)`이며 `FrameTransformer`
  하나의 `target_pos_w[..., 0, :]`만 읽습니다 — 바디 2개를 넘겨도 양팔 의미를
  계산해 주지 않습니다.

그래서 세 항을 직접 정의합니다. 양팔 협조를 학습시키려면 "두 팔 중 가까운 쪽"이
아니라 **두 팔이 모두** 접근·파지해야 보상이 오르는 형태여야 합니다.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from lelab_sim.envs.omx_cfg import CONTACT_FORCE_THRESHOLD


def both_grippers_touching(
    left_sensor: ContactSensor,
    right_sensor: ContactSensor,
    threshold: float = CONTACT_FORCE_THRESHOLD,
) -> torch.Tensor:
    """양 그리퍼가 모두 필터 대상(타겟 물체)에 임계 이상의 힘으로 닿았는지.

    `ContactSensorCfg.filter_prim_paths_expr`가 설정되어 있으면 `force_matrix_w`
    (필터 대상과의 접촉만)를 쓰고, 없거나 `None`이면 `net_forces_w`(모든 접촉)로
    폴백합니다 — Isaac Lab의 `ContactSensorData.force_matrix_w`는 `Tensor | None`
    타입이라 필터 미설정 시 실제로 `None`입니다.

    Returns:
        Shape `[num_envs]` bool 텐서.
    """

    def _touching(sensor: ContactSensor) -> torch.Tensor:
        forces = sensor.data.force_matrix_w
        if forces is None:
            forces = sensor.data.net_forces_w
        # [N, bodies, (filters,) 3] -> 환경별 최대 접촉력 크기
        magnitude = torch.linalg.norm(forces, dim=-1)
        while magnitude.dim() > 1:
            magnitude = magnitude.amax(dim=-1)
        return magnitude > threshold

    return _touching(left_sensor) & _touching(right_sensor)


def bimanual_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """양팔이 **모두** 물체에 접근할 때 커지는 도달 보상.

    `ee_cfg.body_ids`가 가리키는 각 바디에서 물체 중심까지의 거리를 tanh 커널로
    바꾼 뒤 **곱**합니다. 곱이므로 한 팔만 붙어 있으면 값이 오르지 않아, 양팔 협조가
    아닌 단일 팔 해법으로 수렴하는 것을 억제합니다.

    Args:
        env: 환경 인스턴스.
        std: tanh 커널 폭 [m]. 작을수록 근접 구간에서 급격히 상승.
        object_cfg: 타겟 물체.
        ee_cfg: 엔드이펙터 바디들 (`body_names`로 좌/우 지정).

    Returns:
        Shape `[num_envs]`, 범위 `[0, 1]`.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[ee_cfg.name]

    # [N, num_bodies, 3]
    ee_pos = robot.data.body_pos_w[:, ee_cfg.body_ids, :]
    # [N, 1, 3] 로 브로드캐스트
    obj_pos = obj.data.root_pos_w.unsqueeze(1)

    distance = torch.norm(ee_pos - obj_pos, dim=-1)  # [N, num_bodies]
    per_ee = 1.0 - torch.tanh(distance / std)
    return per_ee.prod(dim=-1)


def bimanual_object_grasped(
    env: ManagerBasedRLEnv,
    left_sensor_name: str = "left_gripper_contact",
    right_sensor_name: str = "right_gripper_contact",
    threshold: float = CONTACT_FORCE_THRESHOLD,
) -> torch.Tensor:
    """양 그리퍼가 물체에 **동시에** 접촉했을 때 1, 아니면 0.

    접촉 센서를 직접 읽으므로 거리 근사와 달리 "옆을 지나가는 것"과 "실제로 잡은 것"을
    구분합니다. `collect_rollouts.py`의 성공 판정과 같은 함수를 공유하므로 학습 신호와
    데이터 채택 기준이 어긋나지 않습니다.
    """
    left: ContactSensor = env.scene[left_sensor_name]
    right: ContactSensor = env.scene[right_sensor_name]
    return both_grippers_touching(left, right, threshold).float()


def object_lift_height(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """물체 높이를 `minimal_height`~`target_height` 구간에서 선형으로 보상.

    Isaac Lab의 `object_is_lifted`는 임계 통과 여부만 보는 0/1 이진 보상이라 들어올리는
    **과정**에 기울기가 없습니다. 여기서는 구간 정규화로 연속 신호를 줍니다.

    Returns:
        Shape `[num_envs]`, 범위 `[0, 1]`.
    """
    if target_height <= minimal_height:
        raise ValueError(
            f"target_height({target_height}) must be greater than minimal_height({minimal_height})"
        )

    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]
    progress = (height - minimal_height) / (target_height - minimal_height)
    return progress.clamp(0.0, 1.0)
