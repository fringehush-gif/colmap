# SPDX-License-Identifier: BSD-3-Clause
"""Hermetic graph checks, discoverable by pytest; also run under python -O."""

import contextlib
import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import onnx
except ModuleNotFoundError as error:
    if error.name != "onnx":
        raise
    onnx = None

if onnx is not None:
    helper = onnx.helper
    path = Path(__file__).resolve().parents[2] / "scripts/aliked/export_onnx.py"
    spec = importlib.util.spec_from_file_location("aliked_export_onnx", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    attach_native_shape_metadata = adapter.attach_native_shape_metadata


def seam_model():
    """Metadata-only seam fixture; no pretrained parameters or inference."""
    nodes = [
        helper.make_node("Identity", [], [], name=f"unused_{i}")
        for i in range(2519)
    ]
    for i, output in enumerate(adapter.INPUTS):
        nodes[i] = helper.make_node("Selu", [], [output])
    for start, scale, source, output in zip(
        (1308, 1312, 1316),
        (2, 8, 32),
        adapter.INPUTS[1:],
        adapter.RESIZE_OUTPUTS,
        strict=False,
    ):
        names = [f"scale_{scale}_{part}" for part in range(3)]
        for offset, values in enumerate(([1.0, 1.0], [float(scale)] * 2)):
            tensor = helper.make_tensor(
                "scale", onnx.TensorProto.FLOAT, [2], values
            )
            nodes[start + offset] = helper.make_node(
                "Constant", [], [names[offset]], value=tensor
            )
        nodes[start + 2] = helper.make_node(
            "Concat", names[:2], [names[2]], axis=0
        )
        nodes[start + 3] = helper.make_node(
            "Resize",
            [source, "", names[2]],
            [output],
            name=f"/upsample{scale}/Resize",
            mode="linear",
            coordinate_transformation_mode="align_corners",
        )
    nodes[1320] = helper.make_node(
        "Concat",
        [adapter.INPUTS[0], *adapter.RESIZE_OUTPUTS],
        ["/Concat_output_0"],
        name="/Concat",
        axis=1,
    )
    nodes[1321] = helper.make_node(
        "Conv",
        ["/Concat_output_0", "score_head.0.weight"],
        [],
        name="/score_head/score_head.0/Conv",
    )
    for i in (1330, 1334, 1336):
        nodes[i] = helper.make_node(
            "Identity", ["/Concat_output_0"], [], name=f"consumer_{i}"
        )
    # Shape metadata is sufficient: validation does not inspect payloads.
    weight = onnx.TensorProto(
        name="score_head.0.weight",
        data_type=onnx.TensorProto.FLOAT,
        dims=[8, 128, 1, 1],
    )
    return helper.make_model(
        helper.make_graph(nodes, "metadata_seam", [], [], [weight]),
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 17)],
    )


@unittest.skipIf(onnx is None, "optional ONNX package is not installed")
class ReproHardeningTest(unittest.TestCase):
    def test_valid_metadata_seam(self):
        self.assertEqual(adapter.validate(seam_model())["removedNodeCount"], 12)

    def test_mutated_contracts_rejected(self):
        mutations = {
            "IR": lambda m: setattr(m, "ir_version", 9),
            "opset": lambda m: setattr(m.opset_import[0], "version", 18),
            "axis": lambda m: setattr(m.graph.node[1320].attribute[0], "i", 0),
            "interpolation": lambda m: setattr(
                next(
                    a for a in m.graph.node[1311].attribute if a.name == "mode"
                ),
                "s",
                b"nearest",
            ),
            "consumer": lambda m: m.graph.node[1400].input.append(
                adapter.RESIZE_OUTPUTS[0]
            ),
            "weight_shape": lambda m: m.graph.initializer[0].dims.__setitem__(
                1, 64
            ),
            "source_operator": lambda m: setattr(
                m.graph.node[1], "op_type", "Relu"
            ),
            "scale": lambda m: (
                m.graph.node[1309].attribute[0].t.float_data.__setitem__(0, 3.0)
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(contract=name):
                model = seam_model()
                mutate(model)
                with self.assertRaises(ValueError):
                    adapter.validate(model)

    def test_wrong_sha_precedes_parser_and_check_only_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.onnx"
            path.write_bytes(b"not the pinned input")
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    ["export_onnx", "--input", str(path), "--check-only"],
                ),
                mock.patch.object(adapter.onnx, "load_from_string") as parse,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                with self.assertRaisesRegex(ValueError, "pinned ALIKED"):
                    adapter.main()
                parse.assert_not_called()
                self.assertEqual(output.getvalue(), "")

    def test_check_only_validates_before_success(self):
        # Mock the digest/parser to exercise CLI sequencing without weights.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.onnx"
            path.write_bytes(b"metadata fixture")
            model = seam_model()
            model.ir_version = 9
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    ["export_onnx", "--input", str(path), "--check-only"],
                ),
                mock.patch.object(adapter.hashlib, "sha256") as digest,
                mock.patch.object(
                    adapter.onnx, "load_from_string", return_value=model
                ),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                digest.return_value.hexdigest.return_value = (
                    adapter.PINNED_SHA256
                )
                with self.assertRaisesRegex(ValueError, "IR"):
                    adapter.main()
                self.assertEqual(output.getvalue(), "")

    def test_shape_metadata_rejects_actual_bad_rank(self):
        model = helper.make_model(
            helper.make_graph(
                [],
                "bad_rank",
                [],
                [],
                value_info=[
                    helper.make_tensor_value_info(
                        "/Concat_output_0",
                        onnx.TensorProto.FLOAT,
                        [1, 128, "h"],
                    )
                ],
            )
        )
        with self.assertRaises(ValueError):
            attach_native_shape_metadata(model)

    def test_intermediate_checksum_rejects_before_sparse_stage(self):
        raw = b"original fixture"
        fused = mock.Mock()
        fused.SerializeToString.return_value = b"wrong intermediate"
        with (
            mock.patch.object(
                adapter, "PINNED_SHA256", hashlib.sha256(raw).hexdigest()
            ),
            mock.patch.object(adapter.onnx, "load_from_string"),
            mock.patch.object(
                adapter, "fuse_multiscale_concat", return_value=(fused, {})
            ),
            mock.patch.object(adapter, "normalize_sampled_corners") as sparse,
        ):
            with self.assertRaisesRegex(
                ValueError, "intermediate graph checksum"
            ):
                adapter.export_model(raw)
            sparse.assert_not_called()

    def test_final_checksum_rejects(self):
        raw, intermediate = b"original fixture", b"intermediate fixture"
        fused, final = mock.Mock(), mock.Mock()
        fused.SerializeToString.return_value = intermediate
        final.SerializeToString.return_value = b"wrong final"
        with (
            mock.patch.object(
                adapter, "PINNED_SHA256", hashlib.sha256(raw).hexdigest()
            ),
            mock.patch.object(
                adapter,
                "FUSED_SHA256",
                hashlib.sha256(intermediate).hexdigest(),
            ),
            mock.patch.object(adapter.onnx, "load_from_string"),
            mock.patch.object(
                adapter, "fuse_multiscale_concat", return_value=(fused, {})
            ),
            mock.patch.object(
                adapter, "normalize_sampled_corners", return_value=(final, {})
            ),
        ):
            with self.assertRaisesRegex(ValueError, "final graph checksum"):
                adapter.export_model(raw)


@unittest.skipIf(onnx is None, "optional ONNX package is not installed")
class NativeMetadataTest(unittest.TestCase):
    def test_adds_required_shapes_and_is_idempotent(self):
        model = helper.make_model(helper.make_graph([], "test", [], []))
        attach_native_shape_metadata(model)
        before = model.SerializeToString()
        attach_native_shape_metadata(model)
        self.assertEqual(model.SerializeToString(), before)
        values = {v.name: v for v in model.graph.value_info}
        self.assertEqual(
            set(values),
            {
                "/Concat_output_0",
                "/GridSample_1_output_0",
                "/GridSample_2_output_0",
            },
        )
        self.assertEqual(
            values["/Concat_output_0"].type.tensor_type.elem_type,
            onnx.TensorProto.FLOAT,
        )

    def test_rejects_bad_existing_rank_or_type(self):
        model = helper.make_model(
            helper.make_graph(
                [],
                "test",
                [],
                [],
                value_info=[
                    helper.make_tensor_value_info(
                        "/Concat_output_0",
                        onnx.TensorProto.INT64,
                        [1, 128, "h", "w"],
                    )
                ],
            )
        )
        with self.assertRaises(ValueError):
            attach_native_shape_metadata(model)


if __name__ == "__main__":
    unittest.main()
