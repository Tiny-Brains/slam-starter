# SLAM starter

A working SLAM entry: a small ONNX adapter, its generated manifest, a basic scenario, a fresh-and-
return suite, and a training loop skeleton. It exists to prove the whole path — sensors, map
window, both game heads, private memory and the saved-map carry — with a model you can read.

It is **not a baseline and not a good vacuum**. The graph steers toward the more open lidar side and
writes the same four map cells every decision. Improving it is the exercise.

## Five commands

From this checkout (the sibling `slam` cartridge must be built first):

```sh
../slam/build.sh                               # the cartridge, scenarios and viewer
tinybrains check model.onnx manifest.json      # what admission will say
tinybrains matches/simple-room.json --out replays
tinybrains view replays/first-run.json         # the cartridge's viewer, on loopback
tinybrains suite suite.json --out replays --saved-maps saved-maps
tinybrains report suite.json --out replays --json
```

`games.toml` points at `../slam/dist` in the monorepo. A published starter uses the release
coordinate instead; the format is in that file.

## What is in the box

| File | Purpose |
|---|---|
| `model.onnx` | The generated graph: 25 KB, 6,157 parameters, opset 17. |
| `manifest.json` | Orion adapters: six inputs, two game heads and the memory output. |
| `build.py` | Regenerates the graph with `onnx` and `numpy`, so a review reads Python. |
| `matches/simple-room.json` | One robot, one basic scenario, 1,000 decisions. |
| `suite.json` | A fresh visit and a return visit at the same location, `resume` from the saved map. |
| `example_env.py` | A `tinybrains env` loop: the observation, the action and where a trainer goes. |
| `games.toml` | The local cartridge registry. |

## What the model does

The graph is deliberately plain. It reads two lidar side ranges and turns toward the more open
side; it writes a small free patch every decision; it carries a four-float memory unchanged. What it
proves is what the platform needs to be true:

- the 360-ray lidar and the 256 × 256 map window arrive as tensors;
- `i16[1,2]` `wheel_speed` and `i32[1,2049,3]` `map_update` survive the head decoder and `step`;
- a private `memory` output comes back on the next decision and through `check`;
- a replay carries both outputs, so the submitted map is reconstructed rather than inferred;
- a suite carries the accepted map from the fresh visit into the return.

## Make it yours

Replace `model.onnx` and `manifest.json` together, keep the head contract, and check before you
submit:

```sh
python3 build.py                                # regenerate this graph (needs onnx, numpy)
tinybrains check model.onnx manifest.json
tinybrains matches/simple-room.json --out replays
tinybrains suite suite.json --out replays --saved-maps saved-maps
```

The observation and action contracts are [The world](https://tinybrains.dev/docs/games/slam/world.html) and
[A turn](https://tinybrains.dev/docs/games/slam/turn.html); the scoring rules are
[Ending and scoring](https://tinybrains.dev/docs/games/slam/scoring.html); the saved-map rules are
[Saved maps and suites](https://tinybrains.dev/docs/games/slam/saved-maps.html). The same pages are published
in the site's book.

Rules worth keeping:

- one seat per match; SLAM has no self-play;
- the map frame is the model's own — the evaluator aligns it, so never assume a global origin;
- memory is flat, optional, and at most 262,144 bytes; it resets between visits;
- an invalid action is a strike and a no-op, never a partial map;
- only an accepted finish commits a saved map, and a retry reads the snapshot it was queued with.

## Train

`example_env.py` is the supported training interface: one JSON line per decision, no runner and no
database. Point `TINYBRAINS_CLI` at an installed binary and run it from this directory. A real
trainer replaces `policy()` with inference and keeps the same action object.

## Submit

`check` is the admission check. A submission names the game `slam` and an open season; the platform
runs the season's suite and publishes the aggregate. The model page, the match page and the
evaluation standings all read the same replay and components you see locally.
