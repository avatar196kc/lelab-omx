# LeLab-OMX Isaac Lab 강화학습 및 시뮬레이션 파이프라인

이 디렉토리는 **NVIDIA Isaac Lab (Isaac Sim)** 기반의 **Robotis OpenManipulator-X (OMX) 양팔(Bimanual) 강화학습(PPO)** 및 **LeRobot 데이터셋 합성 파이프라인**을 제공합니다.

---

## 1. 아키텍처 및 런타임 분리 (Runtime Split)

Isaac Lab 환경과 LeLab 웹/학습 환경은 의존성 격리를 위해 서로 다른 Python 인터프리터를 사용합니다.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ [A단계: Isaac Sim Runtime] (Python 3.10 / Omniverse Python)                     │
│  - URDF -> USD 변환                                                             │
│  - Gymnasium 환경 시뮬레이션 (Isaac-Lift-Bimanual-OMX-v0)                       │
│  - RSL-RL PPO 병렬 강화학습 (1024 envs)                                         │
│  - 정책 롤아웃 및 원시 데이터 수집 (meta.json, states.npz, PNG)                 │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ Raw Rollouts (<out>/)
┌──────────────────────────────────────▼──────────────────────────────────────────┐
│ [B단계: LeLab Runtime] (Python 3.12 / uv)                                       │
│  - LeRobotDataset 변환 및 H.264 mp4 비디오 인코딩                               │
│  - 라디안(Rad) -> OMX UI 퍼센트(-100% ~ +100%) 단위 정규화                     │
│  - LeLab Web UI (:8000) 연동 (데이터셋 확인, ACT / Diffusion Policy 학습)       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 런타임별 역할 및 실행 환경 요약

| 구분 | 실행 인터프리터 / 바이너리 | 대상 모듈 및 스크립트 | 주 목적 |
| :--- | :--- | :--- | :--- |
| **Isaac Sim 런타임** | `isaaclab.sh -p` (또는 `python.sh`) | `scripts/tools/convert_urdf.py`<br>`scripts/reinforcement_learning/rsl_rl/train.py`<br>`lelab_sim/collect_rollouts.py`<br>`lelab_sim/envs/*`, `lelab_sim/agents/*` | GPU 가속 물리 시뮬레이션, PPO 강화학습, 원시 롤아웃 데이터 수집 |
| **LeLab 런타임** | `uv run python` / `uv run lelab` | `lelab_sim/export_lerobot.py`<br>`lelab_sim/joint_limits.py`<br>`lelab/*` (Web App, Trajectory, Policy Server) | LeRobot v2.0 데이터셋 패키징, 단위 변환, 웹 UI 서빙 및 모방학습 |

---

## 2. 사전 준비 및 USD 자산 생성

### 2.1 양팔 URDF 생성 (Bimanual Assembly)

단일 팔 URDF(`frontend/public/omx-urdf/urdf/omx_f.urdf`)로부터 작업대 간격 0.40m를 두고 좌/우 접두사가 붙은 결합 URDF 및 기계 가동범위를 생성합니다.

```bash
uv run python scripts/build_bimanual_urdf.py
```
- 생성 파일: `lelab_sim/assets/omx_bimanual.urdf`

### 2.2 URDF to USD 변환

생성된 양팔 URDF를 Isaac Sim용 USD(Universal Scene Description) 포맷으로 변환합니다.

```bash
# Isaac Lab 환경에서 실행
isaaclab.sh -p scripts/tools/convert_urdf.py \
  lelab_sim/assets/omx_bimanual.urdf \
  lelab_sim/assets/omx_bimanual.usd \
  --merge-joints
```

> **옵션 설명**:
> - `--merge-joints`: 고정 조인트(fixed joint) 및 불필요한 링크를 병합하여 물리 엔진(PhysX) 연산 효율을 극대화합니다.
> - 변환된 `omx_bimanual.usd`는 `lelab_sim/envs/omx_cfg.py`의 Articulation 에셋으로 자동 로드됩니다.

---

## 3. 4단계 엔드투엔드 파이프라인 가이드

### 1단계: 시뮬레이션 환경 GUI 시각화 확인

물리 충돌, 로봇 기본 관절 제어, 물체 스폰 위치를 GUI 화면으로 직접 확인합니다.

```bash
# GUI 모드 (단일 로봇 환경)
isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Bimanual-OMX-v0 \
  --num_envs 1
```

> 💡 **참고**: GUI 모드를 실행할 때는 `--headless` 옵션을 생략합니다.

---

### 2단계: Headless 대규모 PPO 학습

1024개의 병렬 시뮬레이션 환경에서 양팔 들어올리기 PPO 정책을 학습합니다.

```bash
isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Bimanual-OMX-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 500
```

- **체크포인트 저장 위치**: `logs/rsl_rl/omx_bimanual_lift/<날짜_시간>/model_500.pt`
- **학습 모니터링**: `tensorboard --logdir logs/rsl_rl/omx_bimanual_lift`

---

### 3단계: 성공 궤적 롤아웃 수집 (A단계)

학습된 PPO 체크포인트를 불러와 3개 카메라 뷰(Left-Top, Left-Wrist, Right-Wrist)와 함께 4중 성공 기준(10cm 인양, 15스텝 유지, 양손 파지, 관절 한계 미도달)을 만족하는 성공 에피소드를 수집합니다.

```bash
isaaclab.sh -p lelab_sim/collect_rollouts.py \
  --task Isaac-Lift-Bimanual-OMX-v0 \
  --checkpoint logs/rsl_rl/omx_bimanual_lift/<run_dir>/model_500.pt \
  --enable_cameras \
  --num_envs 16 \
  --num_episodes 50 \
  --out data/raw_rollouts
```

- **출력 디렉토리 구조 (`data/raw_rollouts/`)**:
  ```
  data/raw_rollouts/
  ├── meta.json             # fps, task, joint_order, num_episodes 메타정보
  ├── episode_000/
  │   ├── states.npz        # 관절 각도(states) 및 절대 목표(actions) (float32 [T, 12] Radian)
  │   ├── left_top/         # 000000.png ~ 000xxx.png
  │   ├── left_wrist/       # 000000.png ~ 000xxx.png
  │   └── right_wrist/      # 000000.png ~ 000xxx.png
  └── episode_001/ ...
  ```

---

### 4단계: LeRobotDataset 변환 (B단계)

수집된 원시 롤아웃을 LeRobot 2.0 표준 데이터셋(H.264 mp4 비디오 인코딩 및 Parquet 메타데이터, OMX 표준 퍼센트 단위 변환)으로 내보냅니다.

```bash
# LeLab 가상환경(uv)에서 실행
uv run python -m lelab_sim.export_lerobot \
  --raw data/raw_rollouts \
  --repo_id "lelab-omx/bimanual_lift_sim" \
  --task "Lift the box with both arms"
```

- **생성 결과 (`~/.cache/huggingface/lerobot/lelab-omx/bimanual_lift_sim/` 또는 지정 경로)**:
  - `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`
  - `videos/observation.images.left_top/chunk-000/file-000.mp4`
  - `videos/observation.images.left_wrist/chunk-000/file-000.mp4`
  - `videos/observation.images.right_wrist/chunk-000/file-000.mp4`
  - `data/chunk-000/file-000.parquet`

---

## 4. LeLab 웹 UI 연동 및 모방학습

LeLab은 단일 포트(`:8000`) 기반의 고성능 통합 웹 애플리케이션을 제공합니다.

### 4.1 서버 실행

```bash
uv run lelab
```
브라우저에서 `http://localhost:8000`으로 접속합니다.

### 4.2 데이터셋 확인 및 모방학습 시작

1. **데이터셋 목록**: 상단 **Datasets** 탭에서 변환된 `lelab-omx/bimanual_lift_sim` 데이터셋이 즉시 로드됩니다.
2. **비디오 재생 및 시각화**:
   - ⚠️ **안내**: 로컬 전용 데이터셋은 목록 및 메타데이터 확인이 가능하며, 내장 뷰어(HuggingFace Space `lerobot/visualize_dataset`)를 통한 영상 재생은 Hugging Face Hub에 푸시된 후 활성화됩니다.
   - **Hub 업로드 방법**:
     ```bash
     uv run huggingface-cli upload lelab-omx/bimanual_lift_sim ~/.cache/huggingface/lerobot/lelab-omx/bimanual_lift_sim --repo-type dataset
     ```
3. **모방학습(Imitation Learning) 실행**:
   - Hub 업로드 여부와 무관하게 **로컬 경로**를 통해 ACT (Action Chunking with Transformers) 또는 Diffusion Policy 학습을 즉시 시작할 수 있습니다:
     ```bash
     uv run lerobot-train \
       --dataset.repo_id="lelab-omx/bimanual_lift_sim" \
       --policy.type=act \
       --output_dir=outputs/train/act_omx_sim
     ```

---

## 5. 트러블슈팅 (Troubleshooting)

### 5.1 `repo_id` 네임스페이스 형식 오류
- **현상**: `ValueError: repo_id must be in the format 'namespace/name' (e.g. 'user/dataset_name')`
- **해결**: Hugging Face 및 LeRobot 규격에 맞게 `owner/dataset_name` (예: `my_team/omx_sim_lift`) 형태로 네임스페이스를 포함하여 지정하세요.

### 5.2 USD 변환 후 그리퍼에서 물체 미끄러짐
- **원인**: URDF 기본 마찰 계수가 너무 낮거나 관성(Inertia) 텐서가 부정확한 경우.
- **해결**:
  1. `lelab_sim/envs/omx_bimanual_env_cfg.py`의 `RigidBodyMaterialCfg`에서 `static_friction=1.5`, `dynamic_friction=1.2`로 마찰력을 상향 조정합니다.
  2. 물체 스폰 시 `MassPropertiesCfg`의 질량이 너무 무겁지 않은지(0.1 ~ 0.3 kg 권장) 확인합니다.

### 5.3 카메라 렌더링 시 VRAM 부족 (OOM)
- **원인**: 수백~수천 개의 병렬 환경에서 동시 카메라 렌더링 활성화 시 GPU 메모리 초과.
- **해결**:
  - PPO 대규모 학습(2단계) 시에는 카메라 렌더링을 끄고 `--headless`로 실행하세요.
  - 롤아웃 수집(3단계) 시에는 병렬 환경 수를 `--num_envs 16` 또는 `--num_envs 8`로 낮춰서 실행하세요.

### 5.4 단위 변환 정합성 (-100% ~ +100%)
- **원인**: 시뮬레이션의 라디안(Radian) 값과 OMX 실제 로봇 모터의 퍼센트 단위 불일치.
- **해결**:
  - `lelab_sim.joint_limits.rad_to_omx_pct`가 모든 상태(state) 및 액션(action)에 대해 관절별 `[q_min, q_max]` 기반 정밀 선형 변환을 자동 적용합니다. (팔: `-100% ~ +100%`, 그리퍼: `0% ~ 100%`).
