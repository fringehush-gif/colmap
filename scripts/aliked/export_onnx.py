# SPDX-License-Identifier: BSD-3-Clause AND MIT
"""Export the exact pinned ALIKED N16ROT graph with optional CUDA operators.

The Resize/Concat stage and CLI are BSD-3-Clause; the sampled-normalization
stage retains MIT. See src/colmap/feature/aliked_native/LICENSE-MIT.
No weights are included. Never overwrite the original model.
"""

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx import helper, numpy_helper

PINNED_SHA256 = (
    "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547"
)
RESIZE_INDICES = (1311, 1315, 1319)
DROP_INDICES = set(range(1308, 1320))
CONCAT_INDEX = 1320
INPUTS = (
    "/gate/Selu_output_0",
    "/gate_1/Selu_output_0",
    "/gate_2/Selu_output_0",
    "/gate_3/Selu_output_0",
)
RESIZE_OUTPUTS = (
    "/upsample2/Resize_output_0",
    "/upsample8/Resize_output_0",
    "/upsample32/Resize_output_0",
)
FUSED_SHA256 = (
    "382f52a2b4e3cb8c5debc9ca22e38206fba6e1ff34aeb68ba9b0288fdd6db131"
)
OUTPUT_SHA256 = (
    "66c1b167d88c5e5e6b3b55683f9dc2159362dc1c228d29a790ef027f057aa300"
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(model):
    nodes = model.graph.node
    require(
        len(nodes) == 2519 and model.ir_version == 8,
        "source node count or IR changed",
    )
    require(
        [(o.domain, o.version) for o in model.opset_import] == [("", 17)],
        "source opset changed",
    )
    producers = {out: i for i, n in enumerate(nodes) for out in n.output if out}
    users = {}
    for i, n in enumerate(nodes):
        for name in n.input:
            users.setdefault(name, []).append(i)
    concat = nodes[CONCAT_INDEX]
    require(
        concat.op_type == "Concat" and concat.name == "/Concat",
        "concat identity changed",
    )
    require(
        list(concat.input) == [INPUTS[0], *RESIZE_OUTPUTS],
        "concat input contract changed",
    )
    require(
        list(concat.output) == ["/Concat_output_0"],
        "concat output contract changed",
    )
    require(
        [(a.name, a.i) for a in concat.attribute] == [("axis", 1)],
        "concat axis changed",
    )
    require(
        users["/Concat_output_0"] == [1321, 1330, 1334, 1336],
        "concat consumers changed",
    )
    next_conv = nodes[1321]
    require(
        next_conv.op_type == "Conv"
        and next_conv.name == "/score_head/score_head.0/Conv",
        "score convolution identity changed",
    )
    require(
        list(next_conv.input) == ["/Concat_output_0", "score_head.0.weight"],
        "score convolution inputs changed",
    )
    weight = next(
        x for x in model.graph.initializer if x.name == "score_head.0.weight"
    )
    require(list(weight.dims) == [8, 128, 1, 1], "score weight shape changed")
    for branch, (idx, scale, source, out) in enumerate(
        zip(
            RESIZE_INDICES, (2, 8, 32), INPUTS[1:], RESIZE_OUTPUTS, strict=False
        )
    ):
        node = nodes[idx]
        require(
            node.op_type == "Resize"
            and node.name == f"/upsample{scale}/Resize",
            "resize identity changed",
        )
        require(
            list(node.input[:2]) == [source, ""] and list(node.output) == [out],
            "resize input/output contract changed",
        )
        attrs = {
            a.name: onnx.helper.get_attribute_value(a) for a in node.attribute
        }
        require(
            attrs["mode"] == b"linear"
            and attrs["coordinate_transformation_mode"] == b"align_corners",
            "resize interpolation changed",
        )
        require(users[out] == [CONCAT_INDEX], "resize consumers changed")
        scales_concat = nodes[producers[node.input[2]]]
        require(
            scales_concat.op_type == "Concat"
            and producers[node.input[2]] in DROP_INDICES,
            "resize scale producer changed",
        )
        values = [
            numpy_helper.to_array(
                onnx.helper.get_attribute_value(
                    nodes[producers[name]].attribute[0]
                )
            ).tolist()
            for name in scales_concat.input
        ]
        require(
            values == [[1.0, 1.0], [float(scale), float(scale)]],
            f"resize scale values changed: branch {branch}: {values}",
        )
    for source in INPUTS:
        require(
            nodes[producers[source]].op_type == "Selu",
            "feature source operator changed",
        )
    for idx in DROP_INDICES:
        for out in nodes[idx].output:
            require(
                all(
                    user in DROP_INDICES or user == CONCAT_INDEX
                    for user in users.get(out, [])
                ),
                f"removed output has another consumer: {out}",
            )
    return {
        "originalNodeCount": len(nodes),
        "removedNodeCount": len(DROP_INDICES),
        "replacementIndex": CONCAT_INDEX,
        "consumers": [nodes[i].name for i in users["/Concat_output_0"]],
        "weightsShape": list(weight.dims),
        "scales": [2, 8, 32],
    }


def fuse_multiscale_concat(model):
    report = validate(model)
    before = [n.SerializeToString() for n in model.graph.node]
    before_inputs = [v.SerializeToString() for v in model.graph.input]
    before_outputs = [v.SerializeToString() for v in model.graph.output]
    before_weights = [v.SerializeToString() for v in model.graph.initializer]
    replacement = onnx.helper.make_node(
        "FusedMultiScaleConcat",
        INPUTS,
        ["/Concat_output_0"],
        name="/FusedMultiScaleConcat",
        domain="com.colmap",
    )
    kept = [
        replacement if i == CONCAT_INDEX else n
        for i, n in enumerate(model.graph.node)
        if i not in DROP_INDICES
    ]
    require(
        all(
            n.SerializeToString() == before[i]
            for i, n in enumerate(model.graph.node)
            if i not in DROP_INDICES and i != CONCAT_INDEX
        ),
        "source nodes changed before rewrite",
    )
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.opset_import.add(domain="com.colmap", version=1)
    require(
        [
            n.SerializeToString()
            for n in model.graph.node
            if n.name != "/FusedMultiScaleConcat"
        ]
        == [
            before[i]
            for i in range(len(before))
            if i not in DROP_INDICES and i != CONCAT_INDEX
        ],
        "undeclared retained node edit",
    )
    require(
        [v.SerializeToString() for v in model.graph.input] == before_inputs,
        "graph inputs changed",
    )
    require(
        [v.SerializeToString() for v in model.graph.output] == before_outputs,
        "graph outputs changed",
    )
    require(
        [v.SerializeToString() for v in model.graph.initializer]
        == before_weights,
        "graph weights changed",
    )
    onnx.checker.check_model(model)
    return model, report


# MIT: sampled-normalization transformation.
RAW, NORM, DIV = "/Concat_output_0", "/Clip_output_0", "/Div_output_0"
REMOVED = {
    "/Shape",
    "/Expand",
    "/Div",
    "/Gather_8",
    "/Unsqueeze_111",
    "/Constant_213",
}
SHAPES = {"/Shape_40", "/Shape_41", "/Shape_42"}
SAMPLERS = {
    "/GridSample_1": (
        "/Reshape_14_output_0",
        "colmap_sparse_patch_sample_count",
    ),
    "/GridSample_2": (
        "/Reshape_19_output_0",
        "colmap_sparse_learned_sample_count",
    ),
}


def attributes(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def validate_sparse_model(model):
    nodes = list(model.graph.node)
    named = {n.name: n for n in nodes}
    require(
        len(nodes) == 2507 and len(named) == len(nodes),
        "source node identity changed",
    )
    require(
        model.ir_version == 8
        and [(x.domain, x.version) for x in model.opset_import]
        == [("", 17), ("com.colmap", 1)],
        "source IR/opset identity changed",
    )
    users = {}
    for node in nodes:
        for value in node.input:
            users.setdefault(value, []).append(node.name)
    expected_users = {
        RAW: ["/score_head/score_head.0/Conv", "/ReduceL2", "/Shape", "/Div"],
        NORM: ["/Expand"],
        DIV: ["/Shape_40", "/Shape_41", "/Shape_42", "/Gather_8"],
        "/Shape_output_0": ["/Expand"],
        "/Expand_output_0": ["/Div"],
        "/Gather_8_output_0": ["/Unsqueeze_111"],
        "/Constant_213_output_0": ["/Unsqueeze_111"],
        "/Unsqueeze_111_output_0": ["/GridSample_1", "/GridSample_2"],
        "/GridSample_1_output_0": ["/Reshape_15"],
        "/GridSample_2_output_0": ["/Reshape_20"],
    }
    for value, expected in expected_users.items():
        require(
            users.get(value) == expected, f"consumer contract changed: {value}"
        )
    exact = {
        "/Shape": ("Shape", [RAW], ["/Shape_output_0"], {}),
        "/Expand": (
            "Expand",
            [NORM, "/Shape_output_0"],
            ["/Expand_output_0"],
            {},
        ),
        "/Div": ("Div", [RAW, "/Expand_output_0"], [DIV], {}),
        "/Gather_8": (
            "Gather",
            [DIV, "/block3/conv1/Constant_3_output_0"],
            ["/Gather_8_output_0"],
            {"axis": 0},
        ),
        "/Unsqueeze_111": (
            "Unsqueeze",
            ["/Gather_8_output_0", "/Constant_213_output_0"],
            ["/Unsqueeze_111_output_0"],
            {},
        ),
        "/ReduceL2": (
            "ReduceL2",
            [RAW],
            ["/ReduceL2_output_0"],
            {"axes": [1], "keepdims": 1},
        ),
        "/Clip": (
            "Clip",
            ["/ReduceL2_output_0", "/Cast_output_0", ""],
            [NORM],
            {},
        ),
    }
    for name in SHAPES:
        exact[name] = ("Shape", [DIV], [name + "_output_0"], {})
    for name, (grid, _) in SAMPLERS.items():
        exact[name] = (
            "GridSample",
            ["/Unsqueeze_111_output_0", grid],
            [name + "_output_0"],
            {"align_corners": 1, "mode": b"bilinear", "padding_mode": b"zeros"},
        )
    for name, (op, inputs, outputs, attrs) in exact.items():
        node = named[name]
        require(
            node.domain == ""
            and node.op_type == op
            and list(node.input) == inputs
            and list(node.output) == outputs
            and attributes(node) == attrs,
            f"node contract changed: {name}",
        )
    for name, expected in [
        ("/Constant_213", [0]),
        ("/block3/conv1/Constant_3", 0),
    ]:
        node = named[name]
        require(
            node.op_type == "Constant"
            and len(node.attribute) == 1
            and node.attribute[0].name == "value",
            f"constant contract changed: {name}",
        )
        value = helper.get_attribute_value(node.attribute[0])
        require(
            value.data_type == onnx.TensorProto.INT64
            and numpy_helper.to_array(value).tolist() == expected,
            f"batch selection constant changed: {name}",
        )
    weights = {t.name: t for t in model.graph.initializer}
    for name, dims in {
        "score_head.0.weight": [8, 128, 1, 1],
        "desc_head.offset_conv.0.weight": [32, 128, 3, 3],
        "desc_head.offset_conv.2.weight": [32, 32, 1, 1],
        "desc_head.agg_weights": [16, 128, 128],
    }.items():
        require(
            list(weights[name].dims) == dims
            and weights[name].data_type == onnx.TensorProto.FLOAT,
            f"weight shape/type changed: {name}",
        )
    removed_values = {out for name in REMOVED for out in named[name].output}
    require(
        not any(
            v.name in removed_values
            for v in [*model.graph.output, *model.graph.value_info]
        ),
        "removed output is exported or has retained metadata",
    )
    return named


def attach_native_shape_metadata(model):
    """Add absent custom-output metadata and validate existing type/shapes."""
    values = [*model.graph.input, *model.graph.output, *model.graph.value_info]
    by_name = {}
    for value in values:
        require(
            value.name not in by_name,
            f"duplicate graph value metadata: {value.name}",
        )
        by_name[value.name] = value
    required = [
        (
            RAW,
            [1, 128, "colmap_fused_concat_height", "colmap_fused_concat_width"],
        ),
        (
            "/GridSample_1_output_0",
            [1, 128, "colmap_sparse_patch_sample_count", 1],
        ),
        (
            "/GridSample_2_output_0",
            [1, 128, "colmap_sparse_learned_sample_count", 1],
        ),
    ]
    for name, shape in required:
        if name not in by_name:
            model.graph.value_info.append(
                helper.make_tensor_value_info(
                    name, onnx.TensorProto.FLOAT, shape
                )
            )
            continue
        tensor = by_name[name].type.tensor_type
        require(
            tensor.elem_type == onnx.TensorProto.FLOAT
            and len(tensor.shape.dim) == 4,
            f"incompatible retained output type or rank: {name}",
        )
        for actual, expected in zip(tensor.shape.dim, shape, strict=False):
            if isinstance(expected, int):
                require(
                    actual.HasField("dim_value")
                    and actual.dim_value == expected,
                    f"incompatible retained output dimension: {name}",
                )
            else:
                require(
                    not actual.HasField("dim_value"),
                    f"unexpected static output dimension: {name}",
                )


def normalize_sampled_corners(model):
    named = validate_sparse_model(model)
    result = onnx.ModelProto()
    result.CopyFrom(model)
    expected = []
    for node in model.graph.node:
        if node.name in REMOVED:
            continue
        changed = onnx.NodeProto()
        changed.CopyFrom(node)
        if node.name in SHAPES:
            changed.input[0] = RAW
        if node.name in SAMPLERS:
            grid, symbol = SAMPLERS[node.name]
            changed = helper.make_node(
                "NormalizedGridSample",
                [RAW, NORM, grid],
                list(node.output),
                name=node.name.replace("GridSample", "NormalizedGridSample"),
                domain="com.colmap",
                count_symbol=symbol,
            )
        expected.append(changed)
    del result.graph.node[:]
    result.graph.node.extend(expected)
    require(len(result.graph.node) == 2501, "result node count mismatch")
    # Exact declared edits; every other node must keep its serialized bytes.
    retained = {n.name: n for n in result.graph.node}
    for name, node in named.items():
        if name not in REMOVED | SHAPES | SAMPLERS.keys():
            require(
                retained[name].SerializeToString() == node.SerializeToString(),
                f"undeclared retained node edit: {name}",
            )
        elif name in SHAPES:
            original = onnx.NodeProto()
            original.CopyFrom(retained[name])
            original.input[0] = DIV
            require(
                original.SerializeToString() == node.SerializeToString(),
                f"extra shape edit: {name}",
            )
    before_shell, after_shell = onnx.ModelProto(), onnx.ModelProto()
    before_shell.CopyFrom(model)
    after_shell.CopyFrom(result)
    del before_shell.graph.node[:]
    del after_shell.graph.node[:]
    require(
        before_shell.SerializeToString() == after_shell.SerializeToString(),
        "weights, graph metadata or model metadata changed",
    )
    produced = {out for node in result.graph.node for out in node.output}
    require(
        not any(
            out in produced for name in REMOVED for out in named[name].output
        ),
        "removed volume path survived",
    )
    attach_native_shape_metadata(result)
    onnx.checker.check_model(result)
    return result, {
        "sourceNodeCount": 2507,
        "resultNodeCount": 2501,
        "removedNodes": sorted(REMOVED),
        "retargetedShapeNodes": sorted(SHAPES),
        "replacedSamplers": sorted(SAMPLERS),
        "undeclaredNodesAndWeightsBytePreserved": True,
        "customOutputValueInfoMayBeAdded": True,
    }


def export_model(raw):
    """Return the final graph bytes only after both pinned stages validate."""
    require(
        hashlib.sha256(raw).hexdigest() == PINNED_SHA256,
        "input is not the pinned ALIKED N16ROT model",
    )
    fused, concat_report = fuse_multiscale_concat(onnx.load_from_string(raw))
    intermediate = fused.SerializeToString()
    require(
        hashlib.sha256(intermediate).hexdigest() == FUSED_SHA256,
        "intermediate graph checksum changed",
    )
    # Preserve the original two-file pipeline's serialization/reparse boundary.
    result, sparse_report = normalize_sampled_corners(
        onnx.load_from_string(intermediate)
    )
    payload = result.SerializeToString()
    require(
        hashlib.sha256(payload).hexdigest() == OUTPUT_SHA256,
        "final graph checksum changed",
    )
    return payload, {
        "sourceSha256": PINNED_SHA256,
        "fusedSha256": FUSED_SHA256,
        "fusedBytes": len(intermediate),
        "outputSha256": OUTPUT_SHA256,
        "outputBytes": len(payload),
        "concat": concat_report,
        "sparse": sparse_report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    payload, report = export_model(args.input.read_bytes())
    if args.check_only:
        print(
            json.dumps({"status": "validated-only", **report}, sort_keys=True)
        )
        return
    require(
        args.output is not None
        and args.output.resolve() != args.input.resolve()
        and not args.output.exists(),
        "provide a fresh output path distinct from the pinned input",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as target:
        target.write(payload)
    print(
        json.dumps(
            {"status": "written-opt-in", "output": str(args.output), **report},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
