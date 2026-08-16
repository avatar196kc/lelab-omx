"""Robotis OMX 양팔 물체 들어올리기(Bimanual Lift) 환경 설정 (Isaac Lab 전용).

ManagerBasedRLEnvCfg 기반으로 태스크의 관측, 액션, 보상, 도메인 랜덤화 및 종료 조건을 정의합니다.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
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

from lelab_sim.envs.omx_cfg import (
    LEFT_TOP_CAMERA_CFG,
    LEFT_WRIST_CAMERA_CFG,
    OMX_BIMANUAL_CFG,
    RIGHT_WRIST_CAMERA_CFG,
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
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.12, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.1,
                angular_damping=0.1,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
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


# -----------------------------------------------------------------------------
# MDP: Actions Configuration
# -----------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """12차원 액션: 12개 관절의 목표 위치 델타 (action_scale = 0.05 rad ≈ 2.9도/스텝)."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_joint[1-5]",
            "left_gripper_joint_1",
            "right_joint[1-5]",
            "right_gripper_joint_1",
        ],
        scale=0.05,
        use_default_offset=True,
    )


# -----------------------------------------------------------------------------
# MDP: Observations Configuration (60차원 Privileged State)
# -----------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    """관측 공간 명세: PPO 학습 전용 60차원 관측 벡터.

    - 관절 위치 (12) + 노이즈 sigma=0.01
    - 관절 속도 (12) + 노이즈 sigma=0.01
    - 양손 엔드이펙터 포즈 (14: pos 3 + quat 4 각각)
    - 물체 포즈 (7: pos 3 + quat 4)
    - 목표 위치 (3: x, y, z)
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
        # 4. 타겟 물체 포즈 (7: pos 3 + quat 4)
        object_pose = ObservationTermCfg(
            func=mdp.root_pose_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        # 5. 타겟 목표 위치 (3)
        target_pos = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "lift_target"},
        )
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

    # 2. 물체 마찰계수 무작위화: 0.5 ~ 1.2
    randomize_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 3. 로봇 관절 초기 위치 리셋 (약간의 노이즈 추가)
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 4. 물체 초기 위치 리셋 (작업대 위 x, y ±0.05m 오프셋)
    reset_object_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi / 4, math.pi / 4),
            },
            "velocity_range": {},
        },
    )


# -----------------------------------------------------------------------------
# MDP: Reward Terms Configuration
# -----------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """보상 함수 명세.

    - reach (+1.0): 양 그리퍼와 물체 간 거리 최소화
    - grasp (+2.0): 물체 파지 / 양손 근접 접촉
    - lift (+5.0): 물체 들어올리기 높이 상승
    - action_rate (-0.01): 급격한 액션 변화 억제 패널티
    """

    # 1. 도달 보상: 그리퍼들과 물체 중심 사이 거리 최소화
    reaching_object = RewardTermCfg(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("object"),
            "ee_cfg": SceneEntityCfg("robot", body_names=["left_link5", "right_link5"]),
        },
    )

    # 2. 파지 보상: 양손과 물체 간 동시 접촉/근접
    object_grasp = RewardTermCfg(
        func=mdp.object_is_grasped,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ee_cfg": SceneEntityCfg("robot", body_names=["left_link5", "right_link5"]),
        },
    )

    # 3. 리프트 보상: 물체 높이 상승
    lifting_object = RewardTermCfg(
        func=mdp.object_lift_height,
        weight=5.0,
        params={
            "minimal_height": 0.45,
            "target_height": 0.60,
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
    """명령 생성기: 목표 물체 리프트 위치 (고정 높이 0.60m)."""

    lift_target = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="left_link5",
        resampling_time_range=(1e9, 1e9),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.35),
            pos_y=(0.0, 0.0),
            pos_z=(0.60, 0.60),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# -----------------------------------------------------------------------------
# Main Environment Configuration
# -----------------------------------------------------------------------------
@configclass
class OmxBimanualLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Robotis OMX 양팔 물체 들어올리기(Bimanual Lift) RL 환경 설정."""

    # 씬 구성
    scene: OmxBimanualLiftSceneCfg = OmxBimanualLiftSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP 구성
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # 시뮬레이션 설정 (120 Hz 물리, decimation=4 -> 30 Hz 제어)
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        decimation=4,
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
        """후처리: 카메라 활성화 플래그에 따라 씬 카메라 설정 적용."""
        super().__post_init__()

        if self.enable_cameras:
            self.scene.left_top_cam = LEFT_TOP_CAMERA_CFG
            self.scene.left_wrist_cam = LEFT_WRIST_CAMERA_CFG
            self.scene.right_wrist_cam = RIGHT_WRIST_CAMERA_CFG
        else:
            self.scene.left_top_cam = None
            self.scene.left_wrist_cam = None
            self.scene.right_wrist_cam = None
