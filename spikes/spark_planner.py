"""Check the Muse Spark planner on free-form commands (paid, cached, budget-guarded).

Run (WSL): python -m spikes.spark_planner <run_dir>
"""
import sys

from agent.controller import Controller
from agent.planner import handle
from backends.meta import budget
from physics.world import World
from viewer.app import Scene

COMMANDS = [
    "could you give the cetaphil a little shove to my left",
    "pile the loo roll onto the deodorant",
    "turn the aerosol crimson and make it twice the size",
    "what happens if I flick the pump bottle really hard?",
    "make it melt",
]


def main(run):
    sc = Scene(run)
    w = World(sc.objects, sc.polygon, floor_z=sc.floor_z)
    ctrl = Controller(sc, w, run)
    for text in COMMANDS:
        res, note = handle(text, ctrl, backend="meta")
        print(f"> {text}")
        for c, r in res:
            print(f"  {c['name']}({c['args']}) -> {r}")
        if note:
            print(f"  note: {note}")
        w.step(1.0)
    print(f"spent total ${budget.spent():.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
