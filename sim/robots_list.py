"""Robot arms available in Simulation mode: every sim/robots/*.yaml (labs add their own yaml there)."""
from pathlib import Path

import yaml

ROBOTS = Path(__file__).with_name("robots")


def robot_choices():
    """{display label: yaml path}."""
    out = {}
    for p in sorted(ROBOTS.glob("*.yaml")):
        y = yaml.safe_load(p.read_text())
        dof = len(y.get("arm_joints", []))
        out[f"{y['name']} ({dof}-DoF, {y['gripper']['type']}, {y['gripper']['max_opening_m'] * 100:.0f} cm)"] = str(p)
    return out
