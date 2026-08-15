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

"""Composite bimanual Robot/Teleoperator wrappers.

Wraps two independent single-arm LeRobot devices (left, right) behind the
same duck-typed interface record.py/teleoperate.py already call against a
single-arm device (`action_features`, `observation_features`, `cameras`,
`get_observation`, `send_action`, `get_action`, `connect`, `disconnect`,
`configure`, `calibration`, `bus.write_calibration`). Feature/observation/
action keys are namespaced with a "left_"/"right_" prefix so a bimanual
dataset's action/observation space is simply the union of both arms' spaces
under lerobot's normal naming convention (prefixing the raw motor/camera key,
e.g. "shoulder_pan.pos" -> "left_shoulder_pan.pos", is safe because
`hw_to_dataset_features` tells joints from cameras apart via the ".pos"
suffix / dtype, not by position).

`lerobot.robots.make_robot_from_config` / `lerobot.teleoperators.
make_teleoperator_from_config` don't know about `BimanualRobotConfig` /
`BimanualTeleoperatorConfig` (they're not registered in lerobot's draccus
choice registry), so callers that go through those factories must special-
case `isinstance(cfg, BimanualRobotConfig)` and use `BimanualRobot.from_config`
instead - see record.py's `record_with_web_events`.

`BimanualRobot`/`BimanualTeleoperator` subclass lerobot's `Robot`/`Teleoperator`
ABCs (rather than just duck-typing them) because `lerobot.scripts.lerobot_record
.record_loop` branches on `isinstance(teleop, Teleoperator)` to decide whether a
teleoperator was even provided; a plain duck-typed class fails that check
silently, so `record_loop` thinks there's no teleop and skips get_action/
send_action/add_frame every iteration - no motion, no error, and an empty
dataset. Subclassing costs implementing a couple more abstract members
(`is_calibrated`, `feedback_features`, `send_feedback`) - see `_BimanualDevice`
and `BimanualTeleoperator` below - but its `__init__` deliberately does not
call `Robot.__init__`/`Teleoperator.__init__` (those assume a single on-disk
calibration file keyed by one `id`, which doesn't apply to a left+right pair).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator

logger = logging.getLogger(__name__)

_LEFT = "left_"
_RIGHT = "right_"

# lerobot's OmxFollower/OmxLeader (and most single-arm robots) call
# `bus.sync_read(...)` with the library default `num_retry=0`, i.e. exactly
# one attempt - any single dropped/corrupted status packet raises
# `ConnectionError` immediately. That default lives in lerobot's own call
# sites, not something lelab can pass through. In bimanual mode each control
# loop tick does 4 of these round-trips (left+right x follower+leader) instead
# of 2, doubling the exposure to a one-off miss, and hardware testing showed
# failures aren't tied to a specific cable/board - they follow whichever side
# is queried, consistent with an occasional transient miss rather than a
# hard fault. Retrying just the side that failed - a few times, with a short
# backoff - absorbs that without masking a genuine persistent disconnect
# (which keeps failing every retry and still raises).
_TRANSIENT_RETRIES = 3
_TRANSIENT_RETRY_DELAY_S = 0.01


def _with_retry(label: str, fn):
    last_exc: ConnectionError | None = None
    for attempt in range(_TRANSIENT_RETRIES):
        try:
            return fn()
        except ConnectionError as exc:
            last_exc = exc
            if attempt < _TRANSIENT_RETRIES - 1:
                logger.warning(
                    "Bimanual %s comm attempt %d/%d failed, retrying: %s",
                    label,
                    attempt + 1,
                    _TRANSIENT_RETRIES,
                    exc,
                )
                time.sleep(_TRANSIENT_RETRY_DELAY_S)
    assert last_exc is not None
    raise last_exc


def _prefix(d: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}


def _split(d: dict) -> tuple[dict, dict]:
    """Split a left_/right_-prefixed dict back into (left, right) dicts with
    the prefix stripped."""
    left = {k[len(_LEFT) :]: v for k, v in d.items() if k.startswith(_LEFT)}
    right = {k[len(_RIGHT) :]: v for k, v in d.items() if k.startswith(_RIGHT)}
    return left, right


class _BimanualBusShim:
    """Stands in for `.bus` so `device.bus.write_calibration(device.calibration)`
    (and the `disable_torque()` call that must precede it - some motor
    protocols reject Homing_Offset writes while torque is enabled) keep
    working unchanged at existing single-arm call sites."""

    def __init__(self, left: Any, right: Any):
        self._left = left
        self._right = right

    def disable_torque(self) -> None:
        self._left.bus.disable_torque()
        self._right.bus.disable_torque()

    def write_calibration(self, calibration: dict) -> None:
        left_cal, right_cal = _split(calibration)
        if left_cal:
            self._left.bus.write_calibration(left_cal)
        if right_cal:
            self._right.bus.write_calibration(right_cal)


@dataclass
class BimanualRobotConfig:
    left: Any
    right: Any
    # lerobot's rollout code (build_rollout_context, base strategy) logs
    # `cfg.robot.type`; provide it so an in-process bimanual rollout that
    # feeds this config through that machinery doesn't AttributeError. Not a
    # draccus-registered choice - purely a label.
    type: str = "bimanual_follower"


@dataclass
class BimanualTeleoperatorConfig:
    left: Any
    right: Any
    type: str = "bimanual_leader"


class _BimanualDevice:
    """Shared plumbing for BimanualRobot/BimanualTeleoperator.

    Deliberately skips `Robot.__init__`/`Teleoperator.__init__` (see module
    docstring) - `left`/`right` are already-connected-or-connectable device
    instances, not a single config to load one calibration file from.
    """

    def __init__(self, left: Any, right: Any):
        self.left = left
        self.right = right
        self.bus = _BimanualBusShim(left, right)

    @property
    def is_calibrated(self) -> bool:
        return self.left.is_calibrated and self.right.is_calibrated

    @property
    def name(self) -> str:
        return f"bimanual_{getattr(self.left, 'name', 'left')}_{getattr(self.right, 'name', 'right')}"

    @property
    def robot_type(self) -> str:
        # lerobot's Robot.__init__ sets `self.robot_type = self.name`, but we
        # deliberately skip that __init__ (see module docstring), so mirror the
        # convention here. The rollout path wraps the robot in ThreadSafeRobot,
        # whose `robot_type` delegates to `self._robot.robot_type` - without
        # this the in-process bimanual rollout dies with AttributeError.
        return self.name

    @property
    def is_connected(self) -> bool:
        return self.left.is_connected and self.right.is_connected

    @property
    def calibration(self) -> dict:
        return {
            **_prefix(self.left.calibration or {}, _LEFT),
            **_prefix(self.right.calibration or {}, _RIGHT),
        }

    @calibration.setter
    def calibration(self, value: dict) -> None:
        left_cal, right_cal = _split(value)
        self.left.calibration = left_cal
        self.right.calibration = right_cal

    def connect(self, calibrate: bool = True) -> None:
        self.left.connect(calibrate=calibrate)
        self.right.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        self.left.disconnect()
        self.right.disconnect()

    def configure(self) -> None:
        self.left.configure()
        self.right.configure()

    def calibrate(self) -> None:
        self.left.calibrate()
        self.right.calibrate()


class BimanualRobot(_BimanualDevice, Robot):
    """Composite follower pair (left + right)."""

    @classmethod
    def from_config(cls, config: BimanualRobotConfig) -> BimanualRobot:
        from lerobot.robots import make_robot_from_config

        return cls(make_robot_from_config(config.left), make_robot_from_config(config.right))

    @property
    def cameras(self) -> dict:
        return {
            **_prefix(getattr(self.left, "cameras", {}) or {}, _LEFT),
            **_prefix(getattr(self.right, "cameras", {}) or {}, _RIGHT),
        }

    @property
    def action_features(self) -> dict:
        return {
            **_prefix(self.left.action_features, _LEFT),
            **_prefix(self.right.action_features, _RIGHT),
        }

    @property
    def observation_features(self) -> dict:
        return {
            **_prefix(self.left.observation_features, _LEFT),
            **_prefix(self.right.observation_features, _RIGHT),
        }

    def get_observation(self) -> dict:
        return {
            **_prefix(_with_retry("left get_observation", self.left.get_observation), _LEFT),
            **_prefix(_with_retry("right get_observation", self.right.get_observation), _RIGHT),
        }

    def send_action(self, action: dict) -> dict:
        left_action, right_action = _split(action)
        sent_left = _with_retry("left send_action", lambda: self.left.send_action(left_action))
        sent_right = _with_retry("right send_action", lambda: self.right.send_action(right_action))
        return {**_prefix(sent_left, _LEFT), **_prefix(sent_right, _RIGHT)}


class BimanualTeleoperator(_BimanualDevice, Teleoperator):
    """Composite leader pair (left + right)."""

    @classmethod
    def from_config(cls, config: BimanualTeleoperatorConfig) -> BimanualTeleoperator:
        from lerobot.teleoperators import make_teleoperator_from_config

        return cls(make_teleoperator_from_config(config.left), make_teleoperator_from_config(config.right))

    @property
    def action_features(self) -> dict:
        return {
            **_prefix(self.left.action_features, _LEFT),
            **_prefix(self.right.action_features, _RIGHT),
        }

    def get_action(self) -> dict:
        return {
            **_prefix(_with_retry("left get_action", self.left.get_action), _LEFT),
            **_prefix(_with_retry("right get_action", self.right.get_action), _RIGHT),
        }

    @property
    def feedback_features(self) -> dict:
        return {
            **_prefix(self.left.feedback_features, _LEFT),
            **_prefix(self.right.feedback_features, _RIGHT),
        }

    def send_feedback(self, feedback: dict) -> None:
        left_feedback, right_feedback = _split(feedback)
        if left_feedback:
            self.left.send_feedback(left_feedback)
        if right_feedback:
            self.right.send_feedback(right_feedback)
