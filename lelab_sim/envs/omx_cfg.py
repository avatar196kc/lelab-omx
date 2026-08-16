"""Robotis OMX 양팔(Bimanual) 로봇 및 센서 구성 (Isaac Lab 전용).

ROBOTIS OMX 하드웨어 사양 및 lelab_sim.joint_limits.JOINT_LIMITS_RAD 단일 출처를 따릅니다.
URDF에 명시되지 않은 물리 물성(마찰, 관성, 솔버 반복)은 파일 상단 상수로 집중 관리합니다.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg

from lelab_sim.joint_limits import JOINT_LIMITS_RAD

# -----------------------------------------------------------------------------
# Tunable Constants (튜닝 대상 물리 및 솔버 파라미터)
# -----------------------------------------------------------------------------
# URDF가 제공하지 않아 손으로 설정해야 하는 값들입니다.
# 시뮬레이션 중 물체 미끄러짐, 진동, 그리퍼 관통 현상 발생 시 아래 상수를 조정합니다.

# USD 자산 경로 (환경 변수 OMX_USD_PATH 로 재정의 가능)
OMX_USD_PATH: str = os.environ.get("OMX_USD_PATH", "lelab_sim/assets/omx.usd")

# 그리퍼 패드 접촉 마찰계수 (물체 파지 안정성 향상)
GRIPPER_PAD_FRICTION: float = 1.0

# PhysX 접촉 Solver 반복 횟수 (최소 8회 권장)
PHYSX_SOLVER_POSITION_ITERATION_COUNT: int = 8
PHYSX_SOLVER_VELOCITY_ITERATION_COUNT: int = 1

# 5-DOF 팔 액추에이터 PD 게인 및 파라미터 (XL430 / XL330)
ARM_STIFFNESS: float = 80.0
ARM_DAMPING: float = 4.0
ARM_ARMATURE: float = 0.01
ARM_EFFORT_LIMIT: float = 10.0
ARM_VELOCITY_LIMIT: float = 4.8

# 1-DOF 그리퍼 액추에이터 PD 게인 및 파라미터 (XL330)
GRIPPER_STIFFNESS: float = 40.0
GRIPPER_DAMPING: float = 2.0
GRIPPER_ARMATURE: float = 0.005
GRIPPER_EFFORT_LIMIT: float = 10.0
GRIPPER_VELOCITY_LIMIT: float = 4.8

# -----------------------------------------------------------------------------
# 관절 한계 (Single Source of Truth: lelab_sim.joint_limits.JOINT_LIMITS_RAD)
# -----------------------------------------------------------------------------
# 좌/우 12개 관절의 가동 범위는 반드시 joint_limits.py의 값을 참조합니다.
_LEFT_SHOULDER_PAN_LIMITS = JOINT_LIMITS_RAD["left_shoulder_pan"]
_LEFT_SHOULDER_LIFT_LIMITS = JOINT_LIMITS_RAD["left_shoulder_lift"]
_LEFT_ELBOW_FLEX_LIMITS = JOINT_LIMITS_RAD["left_elbow_flex"]
_LEFT_WRIST_FLEX_LIMITS = JOINT_LIMITS_RAD["left_wrist_flex"]
_LEFT_WRIST_ROLL_LIMITS = JOINT_LIMITS_RAD["left_wrist_roll"]
_LEFT_GRIPPER_LIMITS = JOINT_LIMITS_RAD["left_gripper"]

# -----------------------------------------------------------------------------
# Robot Articulation Configuration
# -----------------------------------------------------------------------------
OMX_BIMANUAL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=OMX_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=PHYSX_SOLVER_POSITION_ITERATION_COUNT,
            solver_velocity_iteration_count=PHYSX_SOLVER_VELOCITY_ITERATION_COUNT,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Left Arm (joint1 ~ joint5, gripper_joint_1)
            "left_joint1": 0.0,
            "left_joint2": -0.5,
            "left_joint3": 0.5,
            "left_joint4": 0.0,
            "left_joint5": 0.0,
            "left_gripper_joint_1": 0.0,
            # Right Arm (joint1 ~ joint5, gripper_joint_1)
            "right_joint1": 0.0,
            "right_joint2": -0.5,
            "right_joint3": 0.5,
            "right_joint4": 0.0,
            "right_joint5": 0.0,
            "right_gripper_joint_1": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm_left": ImplicitActuatorCfg(
            joint_names_expr=["left_joint[1-5]"],
            effort_limit=ARM_EFFORT_LIMIT,
            velocity_limit=ARM_VELOCITY_LIMIT,
            stiffness=ARM_STIFFNESS,
            damping=ARM_DAMPING,
            armature=ARM_ARMATURE,
        ),
        "arm_right": ImplicitActuatorCfg(
            joint_names_expr=["right_joint[1-5]"],
            effort_limit=ARM_EFFORT_LIMIT,
            velocity_limit=ARM_VELOCITY_LIMIT,
            stiffness=ARM_STIFFNESS,
            damping=ARM_DAMPING,
            armature=ARM_ARMATURE,
        ),
        "gripper_left": ImplicitActuatorCfg(
            joint_names_expr=["left_gripper_joint_1"],
            effort_limit=GRIPPER_EFFORT_LIMIT,
            velocity_limit=GRIPPER_VELOCITY_LIMIT,
            stiffness=GRIPPER_STIFFNESS,
            damping=GRIPPER_DAMPING,
            armature=GRIPPER_ARMATURE,
        ),
        "gripper_right": ImplicitActuatorCfg(
            joint_names_expr=["right_gripper_joint_1"],
            effort_limit=GRIPPER_EFFORT_LIMIT,
            velocity_limit=GRIPPER_VELOCITY_LIMIT,
            stiffness=GRIPPER_STIFFNESS,
            damping=GRIPPER_DAMPING,
            armature=GRIPPER_ARMATURE,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# -----------------------------------------------------------------------------
# Virtual Cameras Configuration (TiledCamera 3대: 640x480 RGB)
# -----------------------------------------------------------------------------
# 1. 작업대 전체 뷰 (left_top)
LEFT_TOP_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera_LeftTop",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.4, 0.4, 0.6),
        rot=(0.3826834, -0.1624598, 0.3535534, 0.8365163),
        convention="world",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 20.0),
    ),
    width=640,
    height=480,
)

# 2. 왼팔 손목 카메라 (left_wrist)
LEFT_WRIST_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_link5/Camera_LeftWrist",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.05),
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.05, 10.0),
    ),
    width=640,
    height=480,
)

# 3. 오른팔 손목 카메라 (right_wrist)
RIGHT_WRIST_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_link5/Camera_RightWrist",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.05),
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.05, 10.0),
    ),
    width=640,
    height=480,
)
