# Design Spec: Isaac Lab 강화학습 및 합성 데이터 생성 파이프라인 (Isaac Lab RL & Synthetic Data Pipeline)

- **Date**: 2026-08-17
- **Target Project**: `lelab-omx` (Robotis OMX Support for LeLab / LeRobot)
- **Status**: Revised (v2) — 초안의 실제 `lelab` 코드 대조 검증 반영

---

## 1. 개요 및 목적 (Overview & Goals)

본 문서는 `lelab-omx` 프로젝트에 **NVIDIA Isaac Lab 기반 강화학습(RL) 및 합성 데이터셋 생성(Synthetic Demonstration Generator)** 파이프라인을 추가하기 위한 상세 아키텍처 및 구현 명세를 정의합니다.

### 주요 목표
1. **합성 데모 데이터 자동 생성**: Isaac Lab 시뮬레이션 환경에서 학습된 강화학습(PPO) 정책을 바탕으로 성공적인 양팔 조작 궤적을 대량 생성하여, **`lelab`의 실기 녹화와 키·단위가 동일한** `LeRobotDataset`으로 내보냅니다.
2. **기존 LeLab 코어 무결성 유지**: 기존 `lelab/` 패키지 코드를 한 줄도 수정하지 않고 `lelab_sim/` 독립 모듈로 구성합니다.
3. **Robotis OMX 양팔(Bimanual) 스키마 1:1 정합**: 실기 녹화가 실제로 생성하는 키(`left_*.pos`, `observation.images.left_top` 등)와 단위(정규화 %)를 그대로 따릅니다. → §3.4
4. **유연한 실험 환경 제공**: 보상 가중치·도메인 랜덤화는 env cfg, PPO 하이퍼파라미터는 agent cfg에 분리하여 Isaac Lab CLI 오버라이드로 튜닝 가능.
5. **Isaac 없이도 검증 가능한 변환 단계**: 파이프라인을 **2단계로 분리**하여, Isaac Sim 없이도 데이터셋 변환 로직 전체를 표준 `pytest`로 검증합니다. → §4.1

### 목표가 아닌 것 (Non-Goals)
- **PPO 정책의 실기 배포.** 관측에 물체 포즈·목표 위치 등 실기에서 측정 불가능한 privileged state가 포함되므로(§3.2), PPO 정책은 **합성 데이터 생성 전용**입니다. 실기 배포는 이 데이터로 학습한 ACT/Diffusion Policy가 담당합니다.

---

## 2. 실행 환경 분리 (Runtime Split) — 가장 중요한 제약

Isaac Sim은 자체 임베디드 Python(4.5 → 3.10, 5.0 → 3.11)과 자체 빌드 PyTorch를 사용하고, `lelab`은 [`pyproject.toml`](../../../pyproject.toml) 기준 `requires-python = ">=3.12"` 입니다. **두 환경은 한 인터프리터에 공존할 수 없습니다.** 따라서 파이프라인을 다음과 같이 두 단계로 자른다.

| 단계 | 인터프리터 | 역할 | 산출물 |
|---|---|---|---|
| **A. 롤아웃 수집** | Isaac Sim 내장 Python (`isaaclab.sh -p`) | 학습된 PPO 정책으로 시뮬레이션 실행, 성공 에피소드만 원시 형태로 덤프 | scratch 디렉토리 (`.npz` + PNG 프레임) |
| **B. 데이터셋 변환** | lelab의 Python ≥3.12 (`lerobot` 설치됨) | 원시 에피소드를 `LeRobotDataset`으로 인코딩 | `$HF_LEROBOT_HOME/{repo_id}` |

이 분리로 얻는 것:
- Isaac Sim 환경에 `lerobot`을 설치하지 않아도 됨 → **PyTorch 재설치로 Isaac Sim이 깨지는 사고를 원천 차단**.
- B단계가 순수 Python + lerobot이므로 **Isaac 없이 CI에서 그대로 테스트 가능**.

> **`pyproject.toml` 변경은 없습니다.** Isaac 측 의존성은 lelab의 env에 설치할 수 없으므로 `[project.optional-dependencies]`에 넣지 않습니다. B단계는 이미 있는 `lerobot` 의존성만 사용합니다. 패키지 탐색 글롭 `include = ["lelab*", ...]`가 `lelab_sim`을 이미 포함하므로 packaging 설정도 그대로 둡니다.

---

## 3. 시스템 아키텍처 및 디렉토리 구조 (System Architecture)

```
lelab-omx/
├── lelab/                             # [기존 LeLab 코어] - 수정 없음
│   ├── record.py  train.py  rollout.py
│   └── utils/{bimanual.py, devices.py}
│
├── lelab_sim/                         # [신규 독립 모듈]
│   ├── README.md                      # Isaac Lab 설치, USD 변환, 4단계 실행 가이드
│   ├── __init__.py                    # ★ 반드시 빈 파일 (아래 주의 참조)
│   ├── joint_limits.py                # ★ 12관절 (q_min,q_max) + rad→% 변환 (Isaac import 금지)
│   ├── envs/
│   │   ├── __init__.py                # gym.register("Isaac-Lift-Bimanual-OMX-v0")
│   │   ├── omx_cfg.py                 # OMX ArticulationCfg(USD), 카메라 (한계는 joint_limits에서 읽음)
│   │   └── bimanual_lift_env_cfg.py   # 관측/액션/보상/종료/도메인 랜덤화
│   ├── agents/
│   │   ├── __init__.py
│   │   └── rsl_rl_ppo_cfg.py          # PPO 하이퍼파라미터
│   ├── collect_rollouts.py            # [A단계] 체크포인트 -> 원시 에피소드 덤프
│   └── export_lerobot.py              # [B단계] 원시 에피소드 -> LeRobotDataset
│
└── tests/
    └── test_sim_export.py             # B단계 전체를 Isaac 없이 검증
```

> **주의 — `lelab_sim/__init__.py`는 비어 있어야 합니다.** 여기서 `envs`나 `isaaclab`을 import하면 B단계(Python 3.12, Isaac 없음)에서 `export_lerobot`을 불러올 수 없게 되고 `tests/test_sim_export.py`가 collection 단계에서 실패합니다.

> **`joint_limits.py`는 sim과 export의 유일한 공유 지점입니다.** 관절 한계는 A단계(`omx_cfg.py`의 ArticulationCfg)와 B단계(단위 변환)가 **같은 값**을 써야 데이터셋 단위가 맞습니다. 그런데 `omx_cfg.py`는 Isaac 전용이라 B단계가 import할 수 없으므로, 한계 테이블만 `isaaclab`을 import하지 않는 별도 모듈로 분리합니다. 여기에 Isaac import가 한 줄이라도 들어가면 B단계 전체가 무너집니다.

> **훈련 스크립트는 새로 만들지 않습니다.** Isaac Lab이 제공하는 표준 `scripts/reinforcement_learning/rsl_rl/train.py`를 그대로 쓰고, 우리는 `envs/__init__.py`에서 태스크를 `gym.register` 하기만 합니다(Isaac Lab의 관용 패턴). 마찬가지로 URDF→USD 변환도 Isaac Lab 내장 `scripts/tools/convert_urdf.py`를 사용합니다.

**RL 라이브러리는 `rsl_rl`로 고정합니다.** 체크포인트 파일명(`model_{iteration}.pt`), config 스키마(`RslRlOnPolicyRunnerCfg`), CLI 인자(`--max_iterations`)가 모두 여기서 결정됩니다.

---

## 4. 세부 컴포넌트 명세 (Detailed Components)

### 4.1 로봇 자산 및 관절 정의 (`lelab_sim/envs/omx_cfg.py`)

* **로봇 모델**: **Robotis OMX** — OMX-F 팔로워 (5-DOF Arm + 1-DOF Gripper) × 2 (Left / Right)

  > ⚠️ **OpenMANIPULATOR-X와 혼동 금지.** 이름이 비슷하지만 별개 제품입니다. OpenMANIPULATOR-X는 4-DOF + **프리즈매틱** 그리퍼이고, OMX는 5-DOF + **리볼루트** 그리퍼입니다(`omx_f.urdf`의 `gripper_joint_2`가 `multiplier="-1"` mimic). 관절 축 구성(z/y/y/y/x)도 SO-101과 동일합니다. **OpenMANIPULATOR-X의 e-Manual 사양표를 이 로봇에 적용하면 안 됩니다.**
* **자산 출처**: 저장소에 이미 있는 [`frontend/public/omx-urdf/urdf/omx_f.urdf`](../../../frontend/public/omx-urdf/urdf/omx_f.urdf)를 단일 진실 공급원으로 삼아 USD로 1회 변환합니다.
  ```bash
  isaaclab.sh -p scripts/tools/convert_urdf.py <repo>/frontend/public/omx-urdf/urdf/omx_f.urdf <out>/omx.usd --merge-joints --joint-stiffness 0 --joint-damping 0
  ```
  URDF의 관절명은 `joint1..joint5` + `gripper_joint_1`(`gripper_joint_2`는 `<mimic>`으로 종속)이며, 이는 [`lelab/teleoperate.py:87`](../../../lelab/teleoperate.py:87)의 `_OMX_URDF_MAPPING`과 동일한 매핑입니다.

  | 모터 이름 (lerobot) | URDF 관절 |
  |---|---|
  | `shoulder_pan` | `joint1` |
  | `shoulder_lift` | `joint2` |
  | `elbow_flex` | `joint3` |
  | `wrist_flex` | `joint4` |
  | `wrist_roll` | `joint5` |
  | `gripper` | `gripper_joint_1` |

* **관절 가동범위 (`lelab_sim/joint_limits.py`의 단일 출처)** — [ROBOTIS OMX 하드웨어 사양](https://ai.robotis.com/omx/hardware_omx.html) 기준:

  | 모터 이름 | URDF | 범위 (deg) | 범위 (rad) | 액추에이터 |
  |---|---|---|---|---|
  | `shoulder_pan` | `joint1` | −270 ~ +360 | −4.712 ~ +6.283 | XL430-W250-T |
  | `shoulder_lift` | `joint2` | −120 ~ +90 | −2.094 ~ +1.571 | XL430-W250-T |
  | `elbow_flex` | `joint3` | −120 ~ +90 | −2.094 ~ +1.571 | XL430-W250-T |
  | `wrist_flex` | `joint4` | −100 ~ +100 | −1.745 ~ +1.745 | XL330-M288-T |
  | `wrist_roll` | `joint5` | ±270 | ±4.712 | XL330-M288-T |
  | `gripper` | `gripper_joint_1` | 0 ~ +100 | 0 ~ +1.745 | XL330-M288-T |

  > **`omx_f.urdf`의 `<limit>` 값은 쓰지 마세요.** 7개 관절 전부 `±6.283`(±2π) 플레이스홀더입니다 — 이 URDF는 프론트엔드 3D 뷰어 표시용이라 한계값이 의미를 가질 이유가 없었습니다. 그대로 쓰면 실제 가동범위(±1.5 rad 수준)가 정규화 후 ±25% 안에만 들어가 실기 데이터와 스케일이 4배 어긋납니다.

  > **검증 완료 (Task 0):** `lerobot/robots/omx_follower/omx_follower.py` 소스 확인 결과, 팔 5개 모터는 `RANGE_M100_100` (`[-100, 100]`), 그리퍼 모터는 `RANGE_0_100` (`[0, 100]`)을 기본 모드로 사용하며 공장 출하 기본 캘리브레이션(`range_min=0, range_max=4095`)을 따릅니다. 이는 기계적 가동범위가 아니라 엔코더 1회전(0–4095 틱) 기준 정규화이므로, **±100% = ±180° = ±π rad**입니다. §4.1의 관절 한계는 정규화가 아니라 시뮬레이션의 물리 가동범위 및 도달 불가 자세 차단에 사용됩니다.

* **⚠️ 튜닝이 필요한 항목 (URDF가 제공하지 않음)**: 링크 관성 텐서, 그리퍼 패드 마찰계수, 관절 armature/damping, 접촉 solver 반복 횟수. USD 변환 직후 물체가 미끄러지거나 그리퍼가 관통하면 **먼저 여기를 의심**합니다. 초기값은 `omx_cfg.py` 상단 상수로 모아 두고 실기 거동에 맞춰 조정합니다.

* **가상 센서 구성** (`TiledCamera`, 640×480 RGB):
  | 센서 | 데이터셋 키 | 비고 |
  |---|---|---|
  | 작업대 전체 뷰 | `observation.images.left_top` | 이름 규칙은 §4.4 참조 |
  | 왼팔 손목 | `observation.images.left_wrist` | |
  | 오른팔 손목 | `observation.images.right_wrist` | |

  > **카메라는 학습 중에는 끕니다.** 렌더링을 켠 채 `--num_envs 1024`로 PPO를 돌리면 처리량이 수십 배 떨어지고 VRAM이 넘칩니다. 카메라는 A단계(롤아웃 수집, `--num_envs` 소수 + `--enable_cameras`)에서만 활성화합니다.

### 4.2 태스크 정의 (`lelab_sim/envs/bimanual_lift_env_cfg.py`)

**1단계 태스크: Bimanual Lift** — 단일 팔로는 잡을 수 없는 크기의 물체를 양팔이 마주 잡고 들어올립니다. (Handover는 §6의 2단계로 이관)

* **제어 주파수 (하나로 고정)**: `sim.dt = 1/120`, `decimation = 4` → **제어·카메라·데이터셋 모두 30 Hz**. 이 값은 [`lelab/record.py`](../../../lelab/record.py)의 기본 녹화 fps(30)와 일치하며, 셋 중 하나라도 어긋나면 데이터셋의 `timestamp`와 영상이 밀립니다. 변경 시 세 곳을 함께 바꿔야 합니다.
* **에피소드 길이**: 300 스텝 (10초 @ 30 Hz)

* **관측 공간 (Observation Space, 60차원)** — PPO 학습 전용 (privileged):
  | 항목 | 차원 |
  |---|---|
  | 양팔 관절 위치 | 12 |
  | 양팔 관절 속도 | 12 |
  | 양팔 엔드이펙터 포즈 (위치 3 + 쿼터니언 4) × 2 | 14 |
  | 타겟 물체 상태 (위치 3 + 쿼터니언 4) | 7 |
  | 타겟 목표 위치 | 3 |
  | 이전 액션 | 12 |
  | **합계** | **60** |

  > 초안의 "그리퍼 상태(2)"는 관절 위치 12에 이미 포함되어 있어 제거했습니다.
  > 물체 포즈·목표 위치·EE 포즈는 시뮬레이터만 알 수 있는 값입니다. ACT/DP는 이 관측을 쓰지 않고, 데이터셋에 저장되는 `observation.state`는 관절 12개뿐입니다(§4.4). 이 비대칭은 의도된 것이며 — 특권 정보를 가진 교사(PPO)가 카메라만 보는 학생(ACT)의 학습 데이터를 만드는 구조입니다.

* **액션 공간 (Action Space, 12차원)**: 12개 관절의 목표 위치 **델타**, `[-1, 1]` 정규화.
  `q_target[t] = clip(q_target[t-1] + action * action_scale, q_min, q_max)`
  `action_scale`(기본 0.05 rad ≈ 스텝당 2.9°)은 env cfg 상수. 값이 크면 학습은 빠르지만 실기에서 재현 불가능한 급격한 궤적이 나옵니다.

* **보상 가중치 (Reward Weights)**:
  | 항목 | 가중치 | 설명 |
  |---|---|---|
  | `reward_reach` | `+1.0` | 양 그리퍼와 물체 간 거리 최소화 |
  | `reward_grasp` | `+2.0` | 물체 양측 동시 접촉 |
  | `reward_lift` | `+5.0` | 물체 높이 상승 |
  | `reward_action_rate` | `-0.01` | 급격한 관절 변화 패널티 |

* **도메인 랜덤화 (Sim-to-Real)**:
  | 항목 | 범위 |
  |---|---|
  | `randomize_mass` | ±10% |
  | `randomize_friction` | 0.5 ~ 1.2 |
  | `observation_noise` | 관절 센서 가우시안 노이즈 σ=0.01 |

### 4.3 롤아웃 수집 (`lelab_sim/collect_rollouts.py`) — A단계

* **역할**: `rsl_rl` 체크포인트를 로드해 시뮬레이션을 실행하고, **성공한 에피소드만** scratch 디렉토리에 원시 형태로 덤프.
* **성공 판정 (모두 만족해야 채택)**:
  1. 물체 최저점이 작업대 기준 `lift_height_threshold = 0.10 m` 이상 상승
  2. 위 상태를 연속 `hold_steps = 15` 프레임(0.5초) 이상 유지
  3. 유지 구간 내내 양 그리퍼가 물체와 접촉 (중간 낙하 배제)
  4. 어떤 관절도 한계에 도달하지 않음
* **출력 레이아웃** (인터프리터 간 경계이므로 포맷을 고정):
  ```
  <scratch>/
    episode_000/
      states.npz        # joints_rad float32[T,12], actions_rad float32[T,12] (절대 목표 위치)
      left_top/000000.png  000001.png ...
      left_wrist/...
      right_wrist/...
    episode_001/ ...
    meta.json           # {"fps":30, "task":"...", "joint_order":[...], "num_episodes":N}
  ```
  PNG 저장은 Isaac Sim 번들에 포함된 `cv2.imwrite` 사용(없으면 `imageio`로 대체). 원시 프레임은 압축 PNG 기준 에피소드당 약 90 MB이므로 50 에피소드에 4~5 GB의 scratch 공간이 필요합니다. **자동 삭제하지 않습니다** — B단계 성공 확인 후 직접 지우세요.
* `joint_order`는 `[left_shoulder_pan, left_shoulder_lift, left_elbow_flex, left_wrist_flex, left_wrist_roll, left_gripper, right_*(동일 순서)]`로 고정.

### 4.4 데이터셋 변환 (`lelab_sim/export_lerobot.py`) — B단계

**이 파일의 유일한 책임은 `lelab`의 실기 녹화와 바이트 수준에서 호환되는 스키마를 만드는 것입니다.**

* **저장 데이터 포맷** ([`lelab/record.py:726`](../../../lelab/record.py:726)의 `hw_to_dataset_features` 결과와 동일):
  | 키 | 타입 | 비고 |
  |---|---|---|
  | `observation.state` | `float32[12]` | 관절 위치, **OMX 정규화 단위**(§ 아래) |
  | `action` | `float32[12]` | 관절 목표 위치, 동일 단위 |
  | `observation.images.left_top` | `video/mp4` | |
  | `observation.images.left_wrist` | `video/mp4` | |
  | `observation.images.right_wrist` | `video/mp4` | |
  | `task` | `str` | **프레임마다 필수** — `add_frame(frame, task=...)` |

  나머지 `timestamp` / `frame_index` / `episode_index` / `index` / `task_index`는 `LeRobotDataset`이 자동으로 채웁니다.

  > 초안에 있던 `next.done` / `next.reward`는 **제거했습니다.** 현재 `lelab`의 실기 녹화가 생성하지 않는 키이고 ACT/Diffusion Policy도 사용하지 않습니다. 목표 3(스키마 1:1 정합)을 위해 실기가 만드는 키만 남깁니다.

* **⚠️ 카메라 키 이름 규칙** (초안의 `observation.images.top`은 틀렸습니다):
  [`lelab/record.py:158`](../../../lelab/record.py:158)의 `_split_cameras_by_side()`가 접두사 없는 카메라 이름을 **왼팔에 배정**하고, [`lelab/utils/bimanual.py:226`](../../../lelab/utils/bimanual.py:226)의 `cameras` 프로퍼티가 다시 `left_`를 붙입니다. 따라서 웹 UI에서 카메라를 `top`으로 부르든 `left_top`으로 부르든 **최종 데이터셋 키는 항상 `observation.images.left_top`** 입니다. 합성 데이터도 이 이름을 써야 실기 데이터와 같은 ACT로 학습됩니다.

* **⚠️ 관절 단위 변환** (초안에 없던 항목 — 없으면 에러 없이 무의미한 데이터가 생성됨):
  Isaac Lab은 라디안 절대각을 사용하고, lerobot의 `omx_follower`는 **엔코더 틱 0~4095 대비 정규화 백분율**을 보고합니다 (팔 = `RANGE_M100_100`, 그리퍼 = `RANGE_0_100`; [`lelab/teleoperate.py:108`](../../../lelab/teleoperate.py:108) 참조). 틱 `[0, 4095]`가 `[-100, +100]%`에 대응하므로 **$\pm 100\% = 1\text{ rev} = \pm 180^\circ = \pm \pi\text{ rad}$** 입니다:

  ```
  팔 5관절:   pct = q_rad * 100 / π                                 # [-100, 100]
  그리퍼:     pct = q_rad * 100 / (100° in rad)                     # [0, 100]
  ```

  `JOINT_LIMITS_RAD`는 정규화 기준이 아니라 **sim의 기계적 관절 한계**(도달 불가능한 자세 차단 및 롤아웃 성공 판정)로 사용됩니다.

  > **OMX는 사용자 캘리브레이션이 없습니다.** [LeRobot OMX 문서](https://huggingface.co/docs/lerobot/en/omx)에 따르면 모터 ID·통신 파라미터·관절 오프셋이 공장에서 설정돼 출하되며 캘리브레이션 단계 자체가 존재하지 않습니다. lerobot의 `omx_follower`는 0~4095 전체 틱을 기준으로 `[-100, 100]`에 매핑합니다.

  > SO-101은 degree 단위를 쓰므로 위 변환은 OMX 전용입니다. 향후 SO-101 sim을 추가하면 분기가 필요합니다.

* **`repo_id`는 반드시 `namespace/name` 형식**: `lerobot`의 `sanity_check_dataset_name()`이 무조건 `repo_id.split("/")`를 수행하므로 이름만 주면 예외가 발생합니다. 이 제약은 [`frontend/src/pages/Landing.tsx:152`](../../../frontend/src/pages/Landing.tsx:152)에 이미 문서화되어 있습니다. Hub에 올릴 계획이 없어도 `local/omx_bimanual_sim_demo`처럼 네임스페이스를 붙이세요.

* **저장 경로**: 환경변수 `HF_LEROBOT_HOME`(미설정 시 `~/.cache/huggingface/lerobot`)을 따릅니다 — [`lelab/datasets.py:29`](../../../lelab/datasets.py:29)와 동일한 규칙. 경로를 하드코딩하지 않습니다.

---

## 5. 실행 및 테스트 워크플로우 (Execution & Testing)

### 5.1 Isaac 없이 검증 (로컬 / CI)

```bash
pytest tests/test_sim_export.py
```

`tests/test_sim_export.py`는 §4.3의 원시 레이아웃을 임시 디렉토리에 가짜로 만들고 `export_lerobot`을 태워 다음을 단언합니다:

- 생성된 `meta/info.json`의 feature 키 집합이 §4.4 표와 **정확히 일치**(`==` 비교 — 잉여 키 유입 차단)
- 단위 변환이 정확한가: 프레임을 각 관절의 `q_min` / 중앙 / `q_max`로 채우고 저장된 `observation.state`가 팔 `-100 / 0 / +100`, 그리퍼 `0 / 50 / 100`인지 확인
- 프레임 수 = 원시 프레임 수, `fps == 30`
- 네임스페이스 없는 `repo_id`는 거부

> 관절값을 0으로 채우면 변환 함수가 무엇을 반환하든 테스트가 통과합니다. **경계값을 넣는 것이 이 테스트의 핵심**입니다 — 단위 변환은 실패해도 예외가 나지 않고 조용히 잘못된 데이터를 만드는 유일한 지점이기 때문입니다.

> **검증 범위**: 이 테스트가 커버하는 것은 **B단계 전체**입니다. Isaac Lab 환경 코드(보상 함수, 도메인 랜덤화, 물리 파라미터)는 시뮬레이터 없이 검증할 수 없으며, 이는 §5.2의 1~2단계를 실제로 돌려서 확인해야 합니다.

### 5.2 Isaac Sim 실제 연결 4단계

1. **환경 시각화 (GUI)** — `--headless`는 값을 받지 않는 플래그이므로 GUI는 **생략**합니다:
   ```bash
   isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Bimanual-OMX-v0 --num_envs 1
   ```

2. **PPO 고속 학습 (headless, 카메라 off)**:
   ```bash
   isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Bimanual-OMX-v0 --headless --num_envs 1024 --max_iterations 500
   ```
   체크포인트는 `logs/rsl_rl/omx_bimanual_lift/<run>/model_{iteration}.pt`로 저장됩니다.

3. **롤아웃 수집 (A단계, 카메라 on)**:
   ```bash
   isaaclab.sh -p lelab_sim/collect_rollouts.py --task Isaac-Lift-Bimanual-OMX-v0 --checkpoint logs/rsl_rl/omx_bimanual_lift/<run>/model_500.pt --enable_cameras --num_envs 16 --num_episodes 50 --out /tmp/omx_rollouts
   ```

4. **데이터셋 변환 (B단계, lelab의 Python)**:
   ```bash
   python -m lelab_sim.export_lerobot --raw /tmp/omx_rollouts --repo_id <hf_user>/omx_bimanual_sim_demo --task "Lift the box with both arms"
   ```

### 5.3 LeLab 웹 UI에서 확인 및 모방학습

```bash
uv run lelab
```

기본 모드는 백엔드가 빌드된 프론트엔드를 함께 서빙하므로 **`http://localhost:8000` 한 곳**만 열면 됩니다([`lelab/scripts/lelab.py:52`](../../../lelab/scripts/lelab.py:52)). (`--dev`는 Vite :8080 + 백엔드 :8000 이중 구성이며 실제 접속 URL에 `?api=http://localhost:8000`가 붙습니다 — 프론트엔드를 고칠 때만 사용.)

* **데이터셋 목록**: 로컬 데이터셋은 `GET /datasets`가 캐시를 스캔해 바로 보여줍니다([`lelab/datasets.py:48`](../../../lelab/datasets.py:48)).
* **⚠️ 영상 재생은 Hub 업로드가 선행돼야 합니다.** 재생 뷰어는 백엔드 라우트가 아니라 HuggingFace Space `lerobot/visualize_dataset` 임베드입니다([`lelab/server.py:787`](../../../lelab/server.py:787), [`frontend/src/pages/Landing.tsx:92`](../../../frontend/src/pages/Landing.tsx:92)). 로컬 전용 데이터셋은 목록에는 뜨지만 미리보기가 되지 않으므로, 확인이 필요하면 웹 UI의 Upload 기능으로 먼저 Hub에 올리세요.
* **학습은 로컬 경로로 가능합니다.** Training 탭은 `--dataset.root`를 전달하므로([`lelab/train.py`](../../../lelab/train.py)) 업로드 없이 ACT / Diffusion Policy 학습을 시작할 수 있습니다.

---

## 6. 결론 및 향후 로드맵 (Roadmap)

* **1단계**: OMX 자산 USD 변환, Bimanual **Lift** 환경, PPO 학습, 롤아웃 수집 + `LeRobotDataset` 변환, B단계 테스트
* **2단계**: 물체 다양성(CAD 모델) 확대 및 복합 조작 태스크 — **Handover**, Peg insertion
* **3단계**: LeLab 웹 UI의 Job Runner 메뉴와 연동하여 버튼 클릭 기반 RL 학습/생성 노코드화

### 착수 전 확인이 필요한 미해결 항목
1. 대상 Isaac Sim 버전과 그 내장 Python 버전 (4.5 = 3.10 / 5.0 = 3.11) — §2의 분리 전제는 동일하나 설치 가이드가 달라집니다.
2. `omx_f.urdf`의 USD 변환 후 관성·마찰 파라미터 초기값 (§4.1) — 실기 팔로 검증 가능한지 여부.
3. ~~실기 캘리브레이션 범위 확보~~ **해결 및 검증 완료** — OMX는 공장 출하 시 설정되어 사용자 캘리브레이션이 없습니다. lerobot `omx_follower` 소스 대조 결과 팔 `RANGE_M100_100`, 그리퍼 `RANGE_0_100` 사용이 확정되었으며, §4.1의 ROBOTIS 공식 사양표 관절 한계가 `lelab_sim/joint_limits.py`의 단일 출처로 사용됩니다.

### 문서 동기화 이월 항목
대회 결과보고서(`2026_오픈소스_개발자대회_결과보고서_접수번호(로봇팔랩).docx`)가 이 설계와 어긋나 있습니다. 제출 전 수정 필요:

1. **「주요기능」/「데이터 흐름」의 "PPO 정책을 실제 OMX 로봇에 연결하여 배포"** → 본 설계의 Non-Goals와 정면충돌. PPO 관측에 privileged state가 포함되어 실기 배포가 불가능하므로, "PPO로 합성 데모 생성 → ACT/DP 모방학습 → 실기 배포"로 서술 교정 필요.
2. **「주요기능」과 「한계점」의 내부 모순** — 전자는 Sim-to-Real 파이프라인을 완성 기능으로, 후자는 "시뮬레이션 기반 데모 자동 생성기는 후속 로드맵"으로 서술.
3. **SBOM 누락** — Isaac Lab, rsl_rl, numpy, opencv가 자재명세서에 없음.
4. **AI 유형 3 필수 항목 공란** — "가중치 공개 저장소 URL", "가중치 파일 정보 및 배포방식"이 `[기재 필요]` 상태. PPO 체크포인트 공개가 전제되므로 §5.2 2단계 산출물을 어디에 올릴지 결정 필요.
5. **시연영상 URL** `[기재 예정]`.
