#!/usr/bin/env python3
"""A training loop skeleton against `tinybrains env`.

    python3 example_env.py                 # 200 decisions, a slow forward drive
    TINYBRAINS_CLI=/usr/local/bin/tinybrains python3 example_env.py

The environment is JSON Lines: one request on stdin, one response on stdout. `hello` names the game
and its maps, an `observe` answers the current turn, and a `step` with one action answers the next.
This script collects a short episode and prints what a training loop would consume: the 360 lidar
ranges, the 256 x 256 map window and the battery. Replace `policy()` with a model, and the same
action object is what you submit.

No rules live here. Every observation and every transition comes from the cartridge.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLI = os.environ.get("TINYBRAINS_CLI", "tinybrains")


def send(proc, message):
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def read(proc):
    line = proc.stdout.readline()
    if not line:
        raise SystemExit("the environment closed without an answer")
    return json.loads(line)


def policy(obs):
    """A slow forward drive with an empty map update.

    A real model would write `map_update.cells` from its map head and choose the wheels from its
    policy. Keeping the window where it is (`read_origin`) is the cheapest way to watch the map the
    engine holds; moving it is a normal action, not an extra call.
    """
    return {
        "wheel_speed_mm_s": [120, 120],
        "map_update": {
            "read_origin": obs["map_window"]["origin"],
            "reset": False,
            "cells": [],
        },
    }


def main():
    if not Path(CLI).is_file() and "/" in CLI:
        raise SystemExit(f"no tinybrains binary at {CLI}; set TINYBRAINS_CLI")
    proc = subprocess.Popen(
        [CLI, "env", "--maps", "simple-room", "--waves", "1", "--matches-per-wave", "1",
         "--max-turns", "200"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        hello = read(proc)
        if not hello.get("hello"):
            raise SystemExit(f"expected hello, got {hello!r}")
        print(f"game {hello['hello']['game']}, maps {hello['hello'].get('maps')}")

        send(proc, {"op": "observe"})
        turn = read(proc)
        for step in range(200):
            seat = turn["seats"][0]
            obs = seat["obs"]
            if step % 20 == 0:
                ranges = [r for r in obs["sensors"]["lidar_mm"] if r > 0]
                battery = obs["sensors"]["battery"]["level_ppm"] / 10000
                print(
                    f"turn {obs['step']:3d}  lidar hits {len(ranges):3d}  "
                    f"battery {battery:5.1f}%  window {obs['map_window']['origin']}  "
                    f"tiles {obs['map_budget']['used_tiles']}"
                )
            send(proc, {"op": "step", "actions": [policy(obs)]})
            turn = read(proc)
            if turn.get("done") or not turn.get("seats"):
                print("the run ended")
                break
            # The environment starts the next episode after a run ends. This demo is one episode;
            # a trainer would keep looping and treat the new turn-zero observation as a reset.
            if step and turn["seats"][0]["obs"]["step"] == 0:
                print("the next episode began")
                break
        send(proc, {"op": "close"})
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()
