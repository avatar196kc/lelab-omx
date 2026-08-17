#!/usr/bin/env python3
"""Robotis OMX 단일 팔 URDF를 양팔(Bimanual) 결합 URDF로 변환하는 스크립트.

단일 팔 URDF(frontend/public/omx-urdf/urdf/omx_f.urdf)를 파싱하여:
1. 베이스 world 링크에 좌/우 로봇 팔을 마운트 (기본 y 간격: ±0.20m, 총 0.40m)
2. 모든 링크와 관절에 left_ / right_ 접두사 부여
3. mimic 관절 참조 이름 갱신 + mimic 종속 관절의 <limit> 유도
4. joint_limits.py의 실제 기계 가동범위를 <limit> 태그에 주입
5. `package://` 메시 URI를 출력 파일 기준 상대 경로로 재작성 (Isaac Lab URDF Importer용)
6. lelab_sim/assets/omx_bimanual.urdf 파일로 출력

사용법:
    python scripts/build_bimanual_urdf.py [--spacing 0.40] [--out lelab_sim/assets/omx_bimanual.urdf]
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# lelab_sim.joint_limits 모듈 참조
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lelab_sim.joint_limits import JOINT_LIMITS_RAD  # noqa: E402

_JOINT_NAME_MAP = {
    "joint1": "shoulder_pan",
    "joint2": "shoulder_lift",
    "joint3": "elbow_flex",
    "joint4": "wrist_flex",
    "joint5": "wrist_roll",
    "gripper_joint_1": "gripper",
}

# 원본 URDF의 메시 URI 접두사. 프론트엔드 3D 뷰어는 urdf-loader의 package 매핑으로
# 해결하지만 Isaac Lab의 URDF Importer는 `package://`를 해석하지 못하므로,
# 출력 파일 위치 기준 상대 경로로 재작성한다.
_MESH_PACKAGE_PREFIX = "package://open_manipulator_description/"
_DEFAULT_MESH_ROOT = Path("frontend") / "public" / "omx-urdf"


def _rewrite_mesh_paths(elem: ET.Element, rel_mesh_root: str) -> None:
    """`<mesh filename="package://...">`를 출력 URDF 기준 상대 경로로 치환한다."""
    for mesh in elem.iter("mesh"):
        filename = mesh.get("filename") or ""
        if filename.startswith(_MESH_PACKAGE_PREFIX):
            suffix = filename[len(_MESH_PACKAGE_PREFIX) :]
            mesh.set("filename", f"{rel_mesh_root}/{suffix}")


def build_bimanual_urdf(
    single_urdf_path: Path,
    out_path: Path,
    y_spacing: float = 0.40,
    mesh_root: Path | None = None,
) -> Path:
    tree = ET.parse(single_urdf_path)
    root = tree.getroot()

    # 메시 루트를 출력 URDF 디렉토리 기준 상대 경로로 계산 (커밋 가능·이식 가능).
    resolved_mesh_root = (mesh_root or (REPO_ROOT / _DEFAULT_MESH_ROOT)).resolve()
    if not resolved_mesh_root.is_dir():
        raise FileNotFoundError(
            f"Mesh root not found: {resolved_mesh_root}. Pass --mesh-root to override."
        )
    rel_mesh_root = Path(
        os.path.relpath(resolved_mesh_root, out_path.parent.resolve())
    ).as_posix()


    bimanual_root = ET.Element("robot", name="omx_bimanual")
    ET.SubElement(bimanual_root, "link", name="world")

    half_spacing = y_spacing / 2.0
    mount_configs = [
        ("left", f"0.0 {half_spacing:.4f} 0.0", "0 0 0"),
        ("right", f"0.0 {-half_spacing:.4f} 0.0", "0 0 0"),
    ]

    for prefix, xyz, rpy in mount_configs:
        # Base mount joint
        mount_joint = ET.SubElement(
            bimanual_root,
            "joint",
            name=f"{prefix}_base_fixed",
            type="fixed",
        )
        ET.SubElement(mount_joint, "parent", link="world")
        ET.SubElement(mount_joint, "child", link=f"{prefix}_link0")
        ET.SubElement(mount_joint, "origin", xyz=xyz, rpy=rpy)

        # Clone and rename links and joints
        for elem in root:
            if elem.tag == "link":
                orig_name = elem.get("name")
                if orig_name == "world":
                    continue
                new_link = ET.fromstring(ET.tostring(elem))
                new_link.set("name", f"{prefix}_{orig_name}")
                _rewrite_mesh_paths(new_link, rel_mesh_root)
                bimanual_root.append(new_link)

            elif elem.tag == "joint":
                orig_name = elem.get("name")
                if orig_name == "world_fixed":
                    continue

                new_joint = ET.fromstring(ET.tostring(elem))
                new_joint.set("name", f"{prefix}_{orig_name}")

                # Update parent and child link references
                parent = new_joint.find("parent")
                if parent is not None:
                    parent.set("link", f"{prefix}_{parent.get('link')}")

                child = new_joint.find("child")
                if child is not None:
                    child.set("link", f"{prefix}_{child.get('link')}")

                # Update mimic joint target if present
                mimic = new_joint.find("mimic")
                target_joint = mimic.get("joint") if mimic is not None else None
                if mimic is not None:
                    mimic.set("joint", f"{prefix}_{target_joint}")

                limit = new_joint.find("limit")
                if limit is not None and orig_name in _JOINT_NAME_MAP:
                    # 사양표의 기계 가동범위 주입
                    spec_key = f"{prefix}_{_JOINT_NAME_MAP[orig_name]}"
                    if spec_key in JOINT_LIMITS_RAD:
                        q_min, q_max = JOINT_LIMITS_RAD[spec_key]
                        limit.set("lower", f"{q_min:.4f}")
                        limit.set("upper", f"{q_max:.4f}")
                elif limit is not None and target_joint is not None:
                    # mimic 종속 관절(예: gripper_joint_2)은 원본 ±2π 플레이스홀더를
                    # 그대로 두면 안 되므로, 대상 관절 한계 × multiplier 로 유도한다.
                    spec_key = f"{prefix}_{_JOINT_NAME_MAP.get(target_joint, target_joint)}"
                    if spec_key in JOINT_LIMITS_RAD:
                        multiplier = float(mimic.get("multiplier", "1"))
                        bounds = sorted(v * multiplier for v in JOINT_LIMITS_RAD[spec_key])
                        limit.set("lower", f"{bounds[0]:.4f}")
                        limit.set("upper", f"{bounds[1]:.4f}")

                bimanual_root.append(new_joint)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(bimanual_root, space="  ", level=0)
    out_tree = ET.ElementTree(bimanual_root)
    out_tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Generated bimanual URDF at {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Bimanual OMX URDF from single arm URDF.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "frontend" / "public" / "omx-urdf" / "urdf" / "omx_f.urdf",
        help="Input single arm URDF path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "lelab_sim" / "assets" / "omx_bimanual.urdf",
        help="Output bimanual URDF path",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.40,
        help="Y-axis spacing between arms in meters (default: 0.40m)",
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=None,
        help=(
            "Directory that 'package://open_manipulator_description/' resolves to "
            f"(default: {_DEFAULT_MESH_ROOT.as_posix()})"
        ),
    )
    args = parser.parse_args()

    build_bimanual_urdf(args.input, args.out, args.spacing, args.mesh_root)


if __name__ == "__main__":
    main()
