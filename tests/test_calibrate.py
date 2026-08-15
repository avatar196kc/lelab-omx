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
"""Tests for lelab.calibrate — manager initial state and request schema."""

from __future__ import annotations


def test_calibration_status_defaults_to_idle() -> None:
    from lelab.calibrate import CalibrationStatus

    status = CalibrationStatus()
    assert status.calibration_active is False
    assert status.status == "idle"
    assert status.device_type is None
    assert status.error is None
    assert status.step == 0


def test_calibration_request_dataclass_round_trip() -> None:
    from lelab.calibrate import CalibrationRequest

    req = CalibrationRequest(
        device_type="teleop",
        port="/dev/ttyUSB0",
        config_file="my_calib",
    )
    assert req.device_type == "teleop"
    assert req.port == "/dev/ttyUSB0"
    assert req.config_file == "my_calib"
    assert req.robot_name is None


def test_calibration_manager_starts_idle() -> None:
    from lelab.calibrate import CalibrationManager

    mgr = CalibrationManager()
    assert mgr.status.calibration_active is False
    assert mgr.status.status == "idle"
    assert mgr.device is None
    assert mgr.calibration_thread is None


def test_calibration_manager_get_status_when_idle_returns_status_object() -> None:
    from lelab.calibrate import CalibrationManager, CalibrationStatus

    mgr = CalibrationManager()
    s = mgr.get_status()
    assert isinstance(s, CalibrationStatus)
    assert s.status == "idle"


def test_calibration_manager_rejects_double_start_via_message() -> None:
    """When calibration_active is True, start_calibration returns success=False."""
    from lelab.calibrate import CalibrationManager, CalibrationRequest

    mgr = CalibrationManager()
    mgr.status.calibration_active = True  # simulate already running

    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )
    assert result.get("success") is False
    assert "already" in result.get("message", "").lower()


def test_cleanup_device_force_releases_and_clears_when_disconnect_fails() -> None:
    """A failed device.disconnect() must still force-close the port and clear the
    device handle — otherwise the COM port stays busy and blocks the next run."""
    from lelab.calibrate import CalibrationManager

    class PortHandler:
        def __init__(self) -> None:
            self.closed = False

        def closePort(self) -> None:  # noqa: N802 - mirrors LeRobot port handler API
            self.closed = True

    class Device:
        def __init__(self) -> None:
            self.bus = type("Bus", (), {"port_handler": PortHandler()})()

        def disconnect(self) -> None:
            raise RuntimeError("Failed to write 'Torque_Enable' on id_=6")

    mgr = CalibrationManager()
    device = Device()
    mgr.device = device

    mgr._cleanup_device()

    assert device.bus.port_handler.closed is True  # force-released despite failure
    assert mgr.device is None  # handle cleared so a new calibration can start


def test_calibration_request_side_defaults_to_left() -> None:
    from lelab.calibrate import CalibrationRequest

    req = CalibrationRequest(device_type="teleop", port="/dev/ttyUSB0", config_file="my_calib")
    assert req.side == "left"


def _make_fake_calibration_device():
    class FakeBus:
        motors: dict = {}

        def write_calibration(self, calibration: dict) -> None:
            pass

    class FakeDevice:
        bus = FakeBus()
        calibration_fpath = "/tmp/fake.json"

        def __init__(self) -> None:
            self.calibration = None

        def _save_calibration(self) -> None:
            pass

    return FakeDevice()


def test_complete_calibration_write_back_patches_left_fields_by_default(monkeypatch) -> None:
    """side='left' (the default) must patch the plain leader/follower fields,
    not the right_* ones."""
    from lelab.calibrate import CalibrationManager, CalibrationRequest

    calls = []
    monkeypatch.setattr(
        "lelab.utils.config.save_robot_record", lambda name, patch, **kw: calls.append((name, patch))
    )

    mgr = CalibrationManager()
    mgr.device = _make_fake_calibration_device()
    mgr._homing_offsets = {}
    mgr._mins = {}
    mgr._maxes = {}
    mgr._current_request = CalibrationRequest(
        device_type="robot", port="/dev/ttyUSB1", config_file="rig", robot_name="rig", side="left"
    )

    mgr._complete_calibration()

    assert calls == [("rig", {"follower_port": "/dev/ttyUSB1", "follower_config": "rig.json"})]


def test_complete_calibration_write_back_patches_right_fields_for_right_side(monkeypatch) -> None:
    from lelab.calibrate import CalibrationManager, CalibrationRequest

    calls = []
    monkeypatch.setattr(
        "lelab.utils.config.save_robot_record", lambda name, patch, **kw: calls.append((name, patch))
    )

    mgr = CalibrationManager()
    mgr.device = _make_fake_calibration_device()
    mgr._homing_offsets = {}
    mgr._mins = {}
    mgr._maxes = {}
    mgr._current_request = CalibrationRequest(
        device_type="teleop", port="/dev/ttyUSB2", config_file="rig_right", robot_name="rig", side="right"
    )

    mgr._complete_calibration()

    assert calls == [("rig", {"right_leader_port": "/dev/ttyUSB2", "right_leader_config": "rig_right.json"})]


def test_position_operating_mode_matches_bus_protocol() -> None:
    # Feetech and Dynamixel use different POSITION register values (0 vs 3);
    # the helper must pick the enum matching the bus's protocol, or OMX
    # (Dynamixel) motors would be put in CURRENT mode during calibration.
    from lelab.calibrate import _position_operating_mode

    class FakeDynamixelBus:
        pass

    FakeDynamixelBus.__module__ = "lerobot.motors.dynamixel.dynamixel"

    class FakeFeetechBus:
        pass

    FakeFeetechBus.__module__ = "lerobot.motors.feetech.feetech"

    assert _position_operating_mode(FakeDynamixelBus()) == 3
    assert _position_operating_mode(FakeFeetechBus()) == 0
