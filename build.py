#!/usr/bin/env python3
"""Build the checked-in `slam-nano` ONNX graph.

The graph is deliberately tiny and readable: it is the adapter fixture Phase 3 needs to prove that
both game outputs survive the platform, not a policy with any ambition. It:

  * steers from two lidar side ranges -- turn toward the more open side, clipped to the wheel limit
    -- so the robot curves instead of walking into the first wall;
  * submits a small, fixed map update, so a replay carries a submitted map as well as a wheel
    command;
  * returns its 4-float memory input unchanged, so admission's memory round trip is exercised.

Every input is a tensor the observation already carries as an array. Assembling a vector from
scalars is not part of the adapter vocabulary, and a fixture that needed it would be teaching a
workaround rather than the contract.

Run from this directory:

    python3 build.py

Needs `onnx` and `numpy` (`pip install onnx numpy`). The graph is generated rather than hand-edited,
so a review reads this file and not a protobuf diff.
"""
import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

OUT = "model.onnx"

# A fixed map update: read the window at the origin, submit two floor tiles, a blocked tile and an
# unknown tile, plus one floor tile at a negative coordinate. The remaining rows are the contract's
# unused marker: coordinates zero, label -1.
MAP_ROWS = 2049
map_update = np.zeros((1, MAP_ROWS, 3), dtype=np.int64)
map_update[:, :, 2] = -1
map_update[0, 0] = [-128, -128, 0]
map_update[0, 1] = [0, 0, 1]
map_update[0, 2] = [1, 0, 1]
map_update[0, 3] = [2, 0, 2]
map_update[0, 4] = [-1, -1, 1]

inits = [
    # Ray 90 is 90 degrees counter-clockwise from forward (the robot's left); ray 270 its right.
    numpy_helper.from_array(np.array([90], dtype=np.int64), "left_index"),
    numpy_helper.from_array(np.array([270], dtype=np.int64), "right_index"),
    numpy_helper.from_array(np.array(0.05, dtype=np.float32), "steer_gain"),
    numpy_helper.from_array(np.array(-80.0, dtype=np.float32), "steer_min"),
    numpy_helper.from_array(np.array(80.0, dtype=np.float32), "steer_max"),
    numpy_helper.from_array(np.array(120.0, dtype=np.float32), "base_speed"),
    numpy_helper.from_array(np.array(-300.0, dtype=np.float32), "wheel_min"),
    numpy_helper.from_array(np.array(300.0, dtype=np.float32), "wheel_max"),
    numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "wheel_shape"),
    numpy_helper.from_array(map_update.astype(np.int32), "map_update_value"),
]

nodes = [
    helper.make_node("Gather", ["lidar", "left_index"], ["left_range"], axis=0),
    helper.make_node("Gather", ["lidar", "right_index"], ["right_range"], axis=0),
    # Turn toward the side with more room.
    helper.make_node("Sub", ["right_range", "left_range"], ["opening"]),
    helper.make_node("Mul", ["opening", "steer_gain"], ["steer_raw"]),
    helper.make_node("Clip", ["steer_raw", "steer_min", "steer_max"], ["steer"]),
    helper.make_node("Add", ["base_speed", "steer"], ["left_f"]),
    helper.make_node("Sub", ["base_speed", "steer"], ["right_f"]),
    helper.make_node("Concat", ["left_f", "right_f"], ["wheels_f"], axis=0),
    helper.make_node("Reshape", ["wheels_f", "wheel_shape"], ["wheels_2d"]),
    helper.make_node("Clip", ["wheels_2d", "wheel_min", "wheel_max"], ["wheels_clipped"]),
    helper.make_node("Cast", ["wheels_clipped"], ["wheel_speed"], to=TensorProto.INT16),
    helper.make_node("Identity", ["map_update_value"], ["map_update"]),
    helper.make_node("Identity", ["memory_in"], ["memory"]),
]

graph = helper.make_graph(
    nodes,
    "slam-nano",
    [
        helper.make_tensor_value_info("lidar", TensorProto.FLOAT, [360]),
        helper.make_tensor_value_info("encoders", TensorProto.INT32, [2]),
        helper.make_tensor_value_info("gyro", TensorProto.FLOAT, [3]),
        helper.make_tensor_value_info("accel", TensorProto.FLOAT, [3]),
        helper.make_tensor_value_info("map", TensorProto.UINT8, [1, 1, 256, 256]),
        helper.make_tensor_value_info("memory_in", TensorProto.FLOAT, [1, 4]),
    ],
    [
        helper.make_tensor_value_info("wheel_speed", TensorProto.INT16, [1, 2]),
        helper.make_tensor_value_info("map_update", TensorProto.INT32, [1, MAP_ROWS, 3]),
        helper.make_tensor_value_info("memory", TensorProto.FLOAT, [1, 4]),
    ],
    initializer=inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, OUT)
print(f"wrote {OUT}")
