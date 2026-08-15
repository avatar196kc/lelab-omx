# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for lelab.utils.bimanual — the composite left/right device wrappers."""

from __future__ import annotations

from typing import Any


class FakeArm:
    """Minimal single-arm device double exposing the full surface
    BimanualRobot/BimanualTeleoperator delegate to."""

    def __init__(self, name: str, motors: dict[str, float]) -> None:
        self.name = name
        self.calls: list[str] = []
        self._connected = False
        self.calibration: dict[str, Any] = {m: f"cal-{m}" for m in motors}
        self.action_features = {f"{m}.pos": float for m in motors}
        self.observation_features = {f"{m}.pos": float for m in motors}
        self.cameras = {"wrist": object()}
        self._obs = {f"{m}.pos": v for m, v in motors.items()}
        self.bus = _FakeBus(self)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self.calls.append(f"connect(calibrate={calibrate})")
        self._connected = True

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        self._connected = False

    def configure(self) -> None:
        self.calls.append("configure")

    def calibrate(self) -> None:
        self.calls.append("calibrate")

    def get_observation(self) -> dict[str, float]:
        return dict(self._obs)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.calls.append(f"send_action({action})")
        return action

    def get_action(self) -> dict[str, float]:
        return dict(self._obs)


class _FakeBus:
    def __init__(self, arm: FakeArm) -> None:
        self._arm = arm
        self.written: dict[str, Any] | None = None

    def write_calibration(self, calibration: dict[str, Any]) -> None:
        self.written = calibration


def _make_pair():
    left = FakeArm("left", {"shoulder_pan": 1.0, "gripper": 2.0})
    right = FakeArm("right", {"shoulder_pan": 3.0, "gripper": 4.0})
    return left, right


def test_bimanual_robot_prefixes_action_and_observation_features() -> None:
    from lelab.utils.bimanual import BimanualRobot

    left, right = _make_pair()
    robot = BimanualRobot(left, right)

    assert robot.action_features == {
        "left_shoulder_pan.pos": float,
        "left_gripper.pos": float,
        "right_shoulder_pan.pos": float,
        "right_gripper.pos": float,
    }
    assert set(robot.observation_features) == {
        "left_shoulder_pan.pos",
        "left_gripper.pos",
        "right_shoulder_pan.pos",
        "right_gripper.pos",
    }


def test_bimanual_robot_prefixes_cameras() -> None:
    from lelab.utils.bimanual import BimanualRobot

    left, right = _make_pair()
    robot = BimanualRobot(left, right)

    assert set(robot.cameras) == {"left_wrist", "right_wrist"}


def test_bimanual_robot_get_observation_merges_with_prefix() -> None:
    from lelab.utils.bimanual import BimanualRobot

    left, right = _make_pair()
    robot = BimanualRobot(left, right)

    obs = robot.get_observation()
    assert obs == {
        "left_shoulder_pan.pos": 1.0,
        "left_gripper.pos": 2.0,
        "right_shoulder_pan.pos": 3.0,
        "right_gripper.pos": 4.0,
    }


def test_bimanual_robot_send_action_splits_by_prefix() -> None:
    from lelab.utils.bimanual import BimanualRobot

    left, right = _make_pair()
    robot = BimanualRobot(left, right)

    action = {
        "left_shoulder_pan.pos": 10.0,
        "left_gripper.pos": 20.0,
        "right_shoulder_pan.pos": 30.0,
        "right_gripper.pos": 40.0,
    }
    result = robot.send_action(action)

    assert left.calls == ["send_action({'shoulder_pan.pos': 10.0, 'gripper.pos': 20.0})"]
    assert right.calls == ["send_action({'shoulder_pan.pos': 30.0, 'gripper.pos': 40.0})"]
    assert result == action


def test_bimanual_teleoperator_get_action_merges_with_prefix() -> None:
    from lelab.utils.bimanual import BimanualTeleoperator

    left, right = _make_pair()
    teleop = BimanualTeleoperator(left, right)

    action = teleop.get_action()
    assert action == {
        "left_shoulder_pan.pos": 1.0,
        "left_gripper.pos": 2.0,
        "right_shoulder_pan.pos": 3.0,
        "right_gripper.pos": 4.0,
    }


def test_bimanual_device_calibration_property_round_trips() -> None:
    from lelab.utils.bimanual import BimanualRobot

    left, right = _make_pair()
    robot = BimanualRobot(left, right)

    merged = robot.calibration
    assert merged == {
        "left_shoulder_pan": "cal-shoulder_pan",
        "left_gripper": "cal-gripper",
        "right_shoulder_pan": "cal-shoulder_pan",
        "right_gripper": "cal-gripper",
    }

    robot.calibration = merged
    assert left.calibration == {"shoulder_pan": "cal-shoulder_pan", "gripper": "cal-gripper"}
    assert right.calibration == {"shoulder_pan": "cal-shoulder_pan", "gripper": "cal-gripper"}


def test_bimanual_bus_shim_write_calibration_splits_by_prefix() -> None:
    from lelab.utils.bimanual import BimanualRobot

    left, right = _make_pair()
    robot = BimanualRobot(left, right)

    robot.bus.write_calibration(
        {"left_shoulder_pan": "L", "right_shoulder_pan": "R"},
    )
    assert left.bus.written == {"shoulder_pan": "L"}
    assert right.bus.written == {"shoulder_pan": "R"}


def test_bimanual_device_connect_disconnect_configure_calibrate_delegate_to_both() -> None:
    from lelab.utils.bimanual import BimanualRobot

    left, right = _make_pair()
    robot = BimanualRobot(left, right)

    robot.connect(calibrate=False)
    robot.configure()
    robot.calibrate()
    assert robot.is_connected
    robot.disconnect()

    for arm in (left, right):
        assert "connect(calibrate=False)" in arm.calls
        assert "configure" in arm.calls
        assert "calibrate" in arm.calls
        assert "disconnect" in arm.calls
    assert not robot.is_connected


def test_bimanual_robot_name_combines_both_sides() -> None:
    from lelab.utils.bimanual import BimanualRobot

    left, right = _make_pair()
    robot = BimanualRobot(left, right)
    assert robot.name == "bimanual_left_right"


# --- Transient comm-failure retry ------------------------------------------
#
# Hardware testing (see work_log/2026-08-08_bimanual_inference_and_fixes.md)
# showed recording failures follow whichever side is queried in the control
# loop, not a specific cable/board - consistent with an occasional single
# dropped status packet rather than a hard fault. lerobot's own single-arm
# robots call `bus.sync_read(...)` with zero retries, so any one-off miss
# would otherwise kill the whole session; `_with_retry` absorbs that per side.


def test_with_retry_returns_immediately_on_success() -> None:
    from lelab.utils.bimanual import _with_retry

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    assert _with_retry("left", fn) == "ok"
    assert calls["n"] == 1


def test_with_retry_recovers_from_transient_connection_error(monkeypatch) -> None:
    from lelab.utils import bimanual

    monkeypatch.setattr(bimanual.time, "sleep", lambda s: None)

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("no status packet")
        return "recovered"

    assert bimanual._with_retry("right", fn) == "recovered"
    assert calls["n"] == 3


def test_with_retry_raises_after_exhausting_attempts(monkeypatch) -> None:
    import pytest

    from lelab.utils import bimanual

    monkeypatch.setattr(bimanual.time, "sleep", lambda s: None)

    def fn():
        raise ConnectionError("no status packet")

    with pytest.raises(ConnectionError, match="no status packet"):
        bimanual._with_retry("right", fn)


def test_bimanual_robot_get_observation_retries_only_the_failing_side(monkeypatch) -> None:
    """A transient failure on one arm must not affect the other, and must not
    surface once the failing side recovers within the retry budget."""
    from lelab.utils import bimanual
    from lelab.utils.bimanual import BimanualRobot

    monkeypatch.setattr(bimanual.time, "sleep", lambda s: None)

    left, right = _make_pair()
    calls = {"n": 0}
    real_right_get_observation = right.get_observation

    def flaky_get_observation():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("no status packet")
        return real_right_get_observation()

    right.get_observation = flaky_get_observation
    robot = BimanualRobot(left, right)

    obs = robot.get_observation()

    assert calls["n"] == 2  # failed once, succeeded on retry
    assert "left_shoulder_pan.pos" in obs
    assert "right_shoulder_pan.pos" in obs


def test_bimanual_teleoperator_get_action_propagates_persistent_failure(monkeypatch) -> None:
    """A side that never recovers within the retry budget must still raise -
    the retry absorbs one-off misses, not a genuine disconnect."""
    import pytest

    from lelab.utils import bimanual
    from lelab.utils.bimanual import BimanualTeleoperator

    monkeypatch.setattr(bimanual.time, "sleep", lambda s: None)

    left, right = _make_pair()

    def always_fails():
        raise ConnectionError("no status packet")

    right.get_action = always_fails
    teleop = BimanualTeleoperator(left, right)

    with pytest.raises(ConnectionError, match="no status packet"):
        teleop.get_action()
