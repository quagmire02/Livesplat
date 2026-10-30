"""Robot description for simulation mode, loaded from sim/robots/<name>.yaml (or a lab's own yaml).

All robot-specific values live here: model file, joints, actuators, TCP, approach/close axes,
gripper type and range, IK mode. sim.robot / sim.grasp / sim.view read only this.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROBOTS = Path(__file__).with_name("robots")


@dataclass
class RobotSpec:
    name: str
    model: Path
    root_body: str
    prefix: str
    arm_joints: list
    arm_actuators: list
    home: np.ndarray
    tcp_body: str
    tcp_offset: np.ndarray
    approach_axis: np.ndarray
    close_axis: np.ndarray
    ik: str
    gripper_type: str
    gripper_actuator: str
    grip_open: float
    grip_closed: float
    max_opening: float
    palm_clearance: float
    base_gap: float
    side_min_height: float = 0.045
    approach_tol: float = 5.0
    grip_kp: float = 0.0                # >0: override the gripper position-actuator stiffness (weak model defaults)

    @classmethod
    def load(cls, name_or_path):
        p = Path(name_or_path)
        if not p.suffix:
            p = ROBOTS / f"{name_or_path}.yaml"
        y = yaml.safe_load(p.read_text())
        g = y["gripper"]
        unit = lambda v: np.array(v, float) / np.linalg.norm(v)
        return cls(name=y["name"], model=Path(y["model"]).expanduser(), root_body=y["root_body"], prefix=y["prefix"],
                   arm_joints=y["arm_joints"], arm_actuators=y["arm_actuators"], home=np.array(y["home"], float),
                   tcp_body=y["tcp"]["body"], tcp_offset=np.array(y["tcp"]["offset"], float),
                   approach_axis=unit(y["approach_axis"]), close_axis=unit(y["close_axis"]), ik=y.get("ik", "full"),
                   gripper_type=g["type"], gripper_actuator=g["actuator"], grip_open=float(g["open"]),
                   grip_closed=float(g["closed"]), max_opening=float(g["max_opening_m"]),
                   palm_clearance=float(g.get("palm_clearance_m", 0.03)), base_gap=float(y.get("base_gap_m", 0.1)),
                   side_min_height=float(y.get("side_min_height_m", 0.045)),
                   approach_tol=float(y.get("approach_tolerance_deg", 5.0)), grip_kp=float(g.get("kp", 0.0)))

    # derived limits
    @property
    def top_depth(self):
        """Top-down grasp: pads this far below the object top, so the palm stays clear."""
        return max(0.01, self.palm_clearance - 0.013)

    @property
    def side_max_in(self):
        return max(0.01, self.palm_clearance - 0.003)

    @property
    def fit_width(self):
        """Largest object width the gripper closes on reliably."""
        return self.max_opening - 0.005
