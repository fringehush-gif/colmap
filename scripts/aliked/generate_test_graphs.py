# SPDX-License-Identifier: BSD-3-Clause
"""Generate tiny model-free custom-op fixtures into a fresh directory."""

import argparse
import json
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def tensor(name, dims):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)


def write_model(path, inputs, outputs, node):
    graph = helper.make_graph([node], path.stem, inputs, outputs)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.colmap", 1),
        ],
    )
    onnx.checker.check_model(model)
    with path.open("xb") as f:
        f.write(model.SerializeToString())


def main():
    print(
        json.dumps(
            {
                "python": sys.executable,
                "onnx": onnx.__file__,
                "onnxVersion": onnx.__version__,
            }
        )
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=False, exist_ok=False)
    for suffix, channels in [("valid", 32), ("bad_channels", 31)]:
        inputs = [
            tensor("a", [1, 32, 32, 32]),
            tensor("b", [1, channels, 16, 16]),
            tensor("c", [1, 32, 4, 4]),
            tensor("d", [1, 32, 1, 1]),
        ]
        node = helper.make_node(
            "FusedMultiScaleConcat",
            ["a", "b", "c", "d"],
            ["out"],
            domain="com.colmap",
        )
        write_model(
            args.output / ("concat_" + suffix + ".onnx"),
            inputs,
            [tensor("out", [1, 128, 32, 32])],
            node,
        )
    for suffix, channels in [("valid", 1), ("bad_norm", 2)]:
        inputs = [
            tensor("raw", [1, 128, 4, 4]),
            tensor("norm", [1, channels, 4, 4]),
            tensor("grid", [1, "samples", 1, 2]),
        ]
        node = helper.make_node(
            "NormalizedGridSample",
            ["raw", "norm", "grid"],
            ["out"],
            domain="com.colmap",
            count_symbol="samples",
        )
        write_model(
            args.output / ("sample_" + suffix + ".onnx"),
            inputs,
            [tensor("out", [1, 128, "samples", 1])],
            node,
        )


if __name__ == "__main__":
    main()
