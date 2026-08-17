"""Robotis OMX 양팔 물체 들어올리기(Bimanual Lift) 환경 설정 (Isaac Lab 전용).

ManagerBasedRLEnvCfg 기반으로 태스크의 관측, 액션, 보상, 도메인 랜덤화 및 종료 조건을 정의합니다.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from lelab_sim.envs import mdp as omx_mdp
from lelab_sim.envs.omx_cfg import (
    LEFT_GRIPPER_CONTACT_CFG,
    LEFT_TOP_CAMERA_CFG,
    LEFT_WRIST_CAMERA_CFG,
    OMX_BIMANUAL_CFG,
    RIGHT_GRIPPER_CONTACT_CFG,
    RIGHT_WRIST_CAMERA_CFG,
)

# 물체를 들어올릴 목표 위치 (환경 로컬 좌표). 관측의 목표 3차원과 리프트 보상의
# target_height가 이 상수 하나에서 파생되도록 묶어 둔다.
LIFT_TARGET_POS: tuple[float, float, float] = (0.35, 0.0, 0.60)


def lift_target_pos(env) -> torch.Tensor:
    """고정 목표 위치를 반환하는 관측 항 (3차원).

    `UniformPoseCommandCfg`를 쓰면 (a) pos+quat 7차원이 되어 설계의 60차원이 깨지고,
    (b) 명령이 `body_name` 프레임 기준이라 "고정 목표"가 팔을 따라 움직인다.
    태스크의 목표는 상수이므로 커맨드 매니저를 쓰지 않고 상수를 그대로 관측에 넣는다.
    """
    return torch.tensor(LIFT_TARGET_POS, device=env.device, dtype=torch.float32).repeat(
        env.num_envs, 1
    )


# -----------------------------------------------------------------------------
# Scene Configuration
# -----------------------------------------------------------------------------
@configclass
class OmxBimanualLiftSceneCfg(InteractiveSceneCfg):
    """시뮬레이션 씬 구성: 바닥, 로봇, 테이블, 타겟 물체, 조명, 카메라."""

    # 바닥면
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 돔 조명
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 양팔 OMX 로봇
    robot = OMX_BIMANUAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 작업대 (Table)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, 0.0, 0.2)),
    )

    # 들어올릴 타겟 물체 (Box Object)
    # 양팔 협조 파지가 필요한 크기 (폭 0.22m)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.22, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.1,
                angular_damping=0.1,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.0, 0.45),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 가상 카메라 3대 (학습 시에는 비활성화, 롤아웃 수집 시 활성화)
    left_top_cam: TiledCameraCfg | None = None
    left_wrist_cam: TiledCameraCfg | None = None
    right_wrist_cam: TiledCameraCfg | None = None

    # 그리퍼 접촉 센서 (파지 판정 — 거리 근사 대체). 학습·수집 모두 항상 활성.
    left_gripper_contact = LEFT_GRIPPER_CONTACT_CFG
    right_gripper_contact = RIGHT_GRIPPER_CONTACT_CFG


# -----------------------------------------------------------------------------
# MDP: Actions Configuration
# -----------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """액션 정의: 12개 관절의 상대적 위치 델타 제어 (Relative Joint Position Action).

    Spec §4.2:
        q_target[t] = clip(q_target[t-1] + action * action_scale, q_min, q_max)
    """

    # 1. 좌/우 5자유도 팔 관절 델타 제어 (10개 관절)
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_joint[1-5]", "right_joint[1-5]"],
        scale=0.05,
    )

    # 2. 좌/우 그리퍼 관절 델타 제어 (2개 관절)
    gripper_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_gripper_joint_1", "right_gripper_joint_1"],
        scale=0.05,
    )


# -----------------------------------------------------------------------------
# MDP: Observations Configuration (60차원 Privileged State)
# -----------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    """관측 공간 (Privileged Observation for PPO Policy, 60차원).

    - 양팔 관절 위치 (12)
    - 양팔 관절 속도 (12)
    - 양손 EE 포즈 (14: Left EE 7 + Right EE 7)
    - 타겟 물체 포즈 (7: pos 3 + quat 4)
    - 타겟 목표 위치 (3)
    - 이전 액션 (12)
    합계: 12 + 12 + 14 + 7 + 3 + 12 = 60차원
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # 1. 관절 위치 (12)
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=GaussianNoiseCfg(std=0.01),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # 2. 관절 속도 (12)
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=GaussianNoiseCfg(std=0.01),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # 3. 양손 엔드이펙터 포즈 (14: Left EE 7 + Right EE 7)
        ee_pose = ObservationTermCfg(
            func=mdp.body_pose_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_link5", "right_link5"],
                )
            },
        )
        # 4. 타겟 물체 포즈 (7 = pos 3 + quat 4).
        # Isaac Lab v2.3.2 에는 `mdp.root_pose_w`가 없어 두 항으로 나눈다
        # (`root_pos_w`는 env 원점 기준, `root_quat_w`는 (w,x,y,z)).
        object_pos = ObservationTermCfg(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        object_quat = ObservationTermCfg(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        # 5. 타겟 목표 위치 (3) — 고정 상수 (커맨드 매니저 미사용, 위 함수 주석 참조)
        target_pos = ObservationTermCfg(func=lift_target_pos)
        # 6. 이전 액션 (12)
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# MDP: Event / Domain Randomization Configuration
# -----------------------------------------------------------------------------
@configclass
class EventCfg:
    """도메인 랜덤화 및 에피소드 초기화 이벤트."""

    # 1. 물체 질량 무작위화: ±10% (0.9 ~ 1.1 scale)
    randomize_object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # 2. 물체 및 바닥 마찰계수 무작위화: 0.5 ~ 1.2
    randomize_physics_materials = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    # 3. 로봇 링크 질량 무작위화: ±10%
    randomize_robot_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # 3-b. 에피소드 리셋 시 로봇 관절 초기화 (+노이즈).
    # ManagerBased 환경은 reset 모드 이벤트 없이 로봇을 복원하지 않으므로, 이 항이
    # 없으면 다음 에피소드가 이전 에피소드의 종료 자세에서 시작한다.
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 4. 에피소드 리셋 시 물체 초기 위치 무작위화 (x: 0.30~0.40m, y: -0.05~0.05m)
    reset_object_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


# -----------------------------------------------------------------------------
# MDP: Reward Terms Configuration
# -----------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """보상 함수 명세.

    1. reaching_object: 양 그리퍼와 물체 간 거리 최소화 (가중치 1.0)
    2. object_grasp: 양 그리퍼와 물체 간 동시 접촉 및 파지 (가중치 2.0)
    3. lifting_object: 물체를 목표 높이까지 들어올림 (가중치 5.0)
    4. action_rate: 급격한 액션 변화 패널티 (가중치 -0.01)
    """

    # 1. 도달 보상: 양손이 **모두** 물체에 접근해야 상승 (tanh 커널의 곱)
    reaching_object = RewardTermCfg(
        func=omx_mdp.bimanual_object_distance,
        weight=1.0,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("object"),
            "ee_cfg": SceneEntityCfg("robot", body_names=["left_link5", "right_link5"]),
        },
    )

    # 2. 파지 보상: 양 그리퍼 접촉 센서를 직접 읽음
    # (collect_rollouts.py의 성공 판정과 같은 함수를 공유 -> 학습 신호와 데이터 채택 기준 일치)
    object_grasp = RewardTermCfg(
        func=omx_mdp.bimanual_object_grasped,
        weight=2.0,
        params={
            "left_sensor_name": "left_gripper_contact",
            "right_sensor_name": "right_gripper_contact",
        },
    )

    # 3. 리프트 보상: 물체 높이 상승 (물체 초기 중심 0.45m 대비 0.55m 이상부터 보상)
    lifting_object = RewardTermCfg(
        func=omx_mdp.object_lift_height,
        weight=5.0,
        params={
            # 물체 초기 중심 0.45m + 성공 판정 임계 0.10m = 0.55m 부터 보상 시작
            "minimal_height": 0.55,
            "target_height": LIFT_TARGET_POS[2],
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    # 4. 액션 변화율 패널티: 부드러운 궤적 유도
    action_rate = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )


# -----------------------------------------------------------------------------
# MDP: Termination Terms Configuration
# -----------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    """에피소드 종료 조건.

    1. time_out: 최대 에피소드 길이 도달 (10초 / 300스텝)
    2. object_falling: 물체가 작업대 아래로 추락 (z < 0.15m)
    """

    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )

    object_falling = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.15,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


# -----------------------------------------------------------------------------
# MDP: Commands Configuration
# -----------------------------------------------------------------------------
@configclass
class CommandsCfg:
    """커맨드 매니저 미사용 (항이 없는 빈 설정).

    목표 위치는 상수이므로 `lift_target_pos` 관측 항이 직접 제공합니다. 이전에
    쓰던 `UniformPoseCommandCfg`는 pos+quat 7차원을 반환해 관측이 64차원이 되고,
    명령이 `body_name` 프레임 기준이라 목표가 왼팔을 따라 움직이는 문제가 있었습니다.
    """


# -----------------------------------------------------------------------------
# Main Environment Configuration
# -----------------------------------------------------------------------------
@configclass
class OmxBimanualLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Robotis OMX 양팔 물체 들어올리기(Bimanual Lift) RL 환경 설정."""

    # 씬 구성 (기본 1024개 병렬 환경)
    scene: OmxBimanualLiftSceneCfg = OmxBimanualLiftSceneCfg(num_envs=1024, env_spacing=2.5)

    # MDP 구성
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # 제어 decimation (물리 120 Hz / decimation 4 = 30 Hz 제어)
    decimation: int = 4

    # 시뮬레이션 설정
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=4,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**21,
            gpu_max_rigid_patch_count=2**20,
        ),
    )

    # 에피소드 길이: 10초 (300 스텝 @ 30 Hz)
    episode_length_s: float = 10.0

    # 카메라 활성화 여부 (기본값 False: 훈련 시 고속 처리를 위해 비활성화)
    enable_cameras: bool = False

    def __post_init__(self):
        """후처리: 카메라 활성화 플래그에 따라 씬 카메라 설정 적용 및 제어 스텝 검증."""
        super().__post_init__()

        # 제어 주파수 및 에피소드 스텝 정합성 고정 (10s / (1/120 * 4) == 300)
        expected_steps = int(self.episode_length_s / (self.sim.dt * self.decimation))
        assert expected_steps == 300, f"Expected 300 steps per episode, got {expected_steps}"

        if self.enable_cameras:
            self.scene.left_top_cam = LEFT_TOP_CAMERA_CFG
            self.scene.left_wrist_cam = LEFT_WRIST_CAMERA_CFG
            self.scene.right_wrist_cam = RIGHT_WRIST_CAMERA_CFG
        else:
            self.scene.left_top_cam = None
            self.scene.left_wrist_cam = None
            self.scene.right_wrist_cam = None
