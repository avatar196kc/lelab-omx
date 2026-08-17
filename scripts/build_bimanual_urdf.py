#!/usr/bin/env python3
"""Robotis OMX 단일 팔 URDF를 양팔(Bimanual) 결합 URDF로 변환하는 스크립트.

단일 팔 URDF(frontend/public/omx-urdf/urdf/omx_f.urdf)를 파싱하여:
1. 베이스 world 링크에 좌/우 로봇 팔을 마운트 (기본 y 간격: ±0.20m, 총 0.40m)
2. 모든 링크와 관절에 left_ / right_ 접두사 부여
3. mimic 관절 참조 이름 갱신
4. joint_limits.py의 실제 기계 가동범위를 <limit> 태그에 주입
5. lelab_sim/assets/omx_bimanual.urdf 파일로 출력

사용법:
    python scripts/build_bimanual_urdf.py [--spacing 0.40] [--out lelab_sim/assets/omx_bimanual.urdf]
"""

from __future__ import annotations

import argparse
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


def build_bimanual_urdf(
    single_urdf_path: Path,
    out_path: Path,
    y_spacing: float = 0.40,
) -> Path:
    tree = ET.parse(single_urdf_path)
    root = tree.getroot()

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
                if mimic is not None:
                    target_joint = mimic.get("joint")
                    mimic.set("joint", f"{prefix}_{target_joint}")

                # Inject calibrated joint limits from joint_limits.py
                limit = new_joint.find("limit")
                if limit is not None and orig_name in _JOINT_NAME_MAP:
                    spec_key = f"{prefix}_{_JOINT_NAME_MAP[orig_name]}"
                    if spec_key in JOINT_LIMITS_RAD:
                        q_min, q_max = JOINT_LIMITS_RAD[spec_key]
                        limit.set("lower", f"{q_min:.4f}")
                        limit.set("upper", f"{q_max:.4f}")

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
    args = parser.parse_args()

    build_bimanual_urdf(args.input, args.out, args.spacing)


if __name__ == "__main__":
    main()
