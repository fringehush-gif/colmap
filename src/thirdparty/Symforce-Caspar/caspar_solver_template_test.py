# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E501 -- Preserve exact upstream template anchors and fragments.

import ast
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location(
    "caspar_solver_template",
    Path(__file__).with_name("caspar_solver_template.py"),
)
template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template)

# Only COLMAP's remaining overlay regions are needed here. The real generator
# separately validates complete canonical inputs before calling these patches.
SOURCE = """namespace {

}
void solve() {
  {{solver.linear_t}} diag = params_.diag_init;
  result.initial_score = score_best;
        {{solver.linear_t}} score_new_pcg = DoRetractScore();
        score_best_pcg = score_new_pcg;
        // Retain the prediction before the next PCG update overwrites the step.
        pred_decrease_best_pcg = GetPredDecrease();
}
{{solver.linear_t}} {{ solver.struct_name }}::DoRetractScore() {
    return 0.5 * ReadCuMem({{solver_key("res_tot")}});
}

void {{ solver.struct_name }}::DoBeta() {
    BetaFromNumDenom({{solver_key(\"beta_numerator\")}}, {{solver_key(\"alpha_numerator\")}}, {{solver_key(\"beta\")}});
}
{{solver.linear_t}} {{ solver.struct_name }}::GetPredDecrease() {
    return 0.5 * ReadCuMem({{solver_key("pred_decrease_tot")}});
}
void GetHost() {
  if (offset > {{num_key(nodetype)}} || num > {{num_key(nodetype)}} - offset) {
    throw std::runtime_error("{{nodetype.__name__}} readback range exceeds node count");
  }
  if (num == 0) {
    return;
  }
  constexpr size_t kDimension = {{Ops.storage_dim(nodetype)}};
  // The current mapper rounds up unsigned counts in blocks of 32.
  const size_t limit = std::numeric_limits<unsigned int>::max();
  if ({{num_max_key(nodetype)}} > limit || offset > limit || num > limit - 31 ||
      kDimension == 0 ||
      num > std::numeric_limits<size_t>::max() / sizeof({{solver.storage_t}}) / kDimension) {
    throw std::runtime_error("{{nodetype.__name__}} readback exceeds mapping argument limits");
  }
  CheckCuda(cudaSetDevice(device_id_), "cudaSetDevice");
  CheckCuda(cudaMemcpy(data, marker__scratch_inout_,
                               kDimension * num * sizeof({{solver.storage_t}}), cudaMemcpyDeviceToHost),
                    "cudaMemcpy device-to-host");
}
void GetDevice() {
  if (offset > {{num_key(nodetype)}} || num > {{num_key(nodetype)}} - offset) {
    throw std::runtime_error("{{nodetype.__name__}} readback range exceeds node count");
  }
  if (num == 0) {
    return;
  }
  constexpr size_t kDimension = {{Ops.storage_dim(nodetype)}};
  // The current mapper rounds up unsigned counts in blocks of 32.
  const size_t limit = std::numeric_limits<unsigned int>::max();
  if ({{num_max_key(nodetype)}} > limit || offset > limit || num > limit - 31 ||
      kDimension == 0 ||
      num > std::numeric_limits<size_t>::max() / sizeof({{solver.storage_t}}) / kDimension) {
    throw std::runtime_error("{{nodetype.__name__}} readback exceeds mapping argument limits");
  }
  CheckCuda(cudaSetDevice(device_id_), "cudaSetDevice");
}
"""
HEADER = """enum class ExitReason {
  MAX_ITERATIONS,
  CONVERGED_SCORE_THRESHOLD,
  CONVERGED_DIAG_EXIT
};
  {{solver.linear_t}} DoRetractScore();
  {{solver.linear_t}} GetPredDecrease();
"""
BUILDFILE = (
    "add_library({{caslib.name}}_core STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})\n"
)
TEMPLATES = {
    "solver.cc.jinja": SOURCE,
    "solver.h.jinja": HEADER,
    "buildfiles/CMakeLists.txt.jinja": BUILDFILE,
}


def generation_tail():
    path = Path(__file__).with_name("caspar_generate.py")
    module = ast.parse(path.read_text())
    start = next(
        i
        for i, node in enumerate(module.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "out_dir"
            for target in node.targets
        )
    )
    return compile(
        ast.Module(body=module.body[start:], type_ignores=[]), str(path), "exec"
    )


def generation_namespace(installed, output, precision):
    loader = SimpleNamespace(
        get_source=lambda env, name: (TEMPLATES[name], None, None)
    )
    return {
        "Path": Path,
        "sys": SimpleNamespace(argv=["caspar_generate.py", str(output)]),
        "precision": precision,
        "print": Mock(),
        "caspar": SimpleNamespace(__file__=str(installed / "__init__.py")),
        "caslib": SimpleNamespace(generate=Mock()),
        "template_env": SimpleNamespace(loader=loader),
        "ChoiceLoader": Mock(),
        "DictLoader": Mock(),
        "validate_generation_inputs": template.validate_generation_inputs,
        "patch_solver_templates": template.patch_solver_templates,
        "buildfile_overrides": template.buildfile_overrides,
    }


class CasparSolverTemplateTest(unittest.TestCase):
    def test_initial_damping_uses_configured_minimum(self) -> None:
        source, _ = template.patch_solver_templates(SOURCE, HEADER)
        self.assertIn(
            "diag = std::max(params_.diag_init, params_.diag_min);", source
        )

    def test_paired_score_prediction_and_mapping_bounds(self) -> None:
        source, header = template.patch_solver_templates(SOURCE, HEADER)
        for expected in (
            "pred_decrease_best_pcg = predicted_decrease_pcg;",
            "cudaMemcpyDeviceToDevice",
            "cudaMemcpy PCG residual product",
            "if (!std::isfinite(score_best))",
            "ExitReason::FAILURE_NONFINITE_INITIAL_SCORE",
            "DoRetractScore(&predicted_decrease_pcg)",
            "if (predicted_decrease != nullptr)",
            "Caspar score/prediction scratch layout changed",
            "Caspar score/prediction readback failed",
            "*predicted_decrease = 0.5 * values[0];",
            "return 0.5 * values[1];",
            "std::numeric_limits<unsigned int>::max() - 32",
            "capacity > kIndexLimit / StorageDimension",
            'solver.fields[node_key(nodetype, "storage_current")].dim_real',
        ):
            self.assertIn(expected, source)
        self.assertEqual(source.count("  CheckReadbackRange<"), 2)
        self.assertNotIn("kDimension", source)
        self.assertEqual(
            source.count("CheckCuda("),
            SOURCE.count("CheckCuda(") + 1,
        )
        self.assertIn("FAILURE_NONFINITE_INITIAL_SCORE", header)
        self.assertIn("* predicted_decrease = nullptr", header)
        self.assertEqual(header.count("void LaunchPredDecrease();"), 1)

    def test_missing_or_duplicate_overlay_regions_are_refused(self) -> None:
        for anchor in (
            "        {{solver.linear_t}} score_new_pcg = DoRetractScore();",
            template._RETAINED_PROPOSAL,
            "{{solver.linear_t}} {{ solver.struct_name }}::DoRetractScore() {",
            "{{solver.linear_t}} {{ solver.struct_name }}::GetPredDecrease() {",
            '    return 0.5 * ReadCuMem({{solver_key("pred_decrease_tot")}});',
            "namespace {\n\n",
            template._READBACK_GUARD,
        ):
            for changed in (SOURCE.replace(anchor, "", 1), SOURCE + anchor):
                with self.subTest(anchor=anchor, changed=changed):
                    with self.assertRaises(RuntimeError):
                        template.patch_solver_templates(changed, HEADER)
        patched = template.patch_solver_templates(SOURCE, HEADER)
        with self.assertRaises(RuntimeError):
            template.patch_solver_templates(*patched)

    def test_f64_profile_is_public_and_f32_is_unchanged(self) -> None:
        self.assertEqual(template.buildfile_overrides(BUILDFILE, "f32"), {})
        result = template.buildfile_overrides(BUILDFILE, "f64")
        self.assertEqual(
            result,
            {
                "buildfiles/CMakeLists.txt.jinja": BUILDFILE + "\n"
                "target_compile_definitions({{caslib.name}}_core PUBLIC CASPAR_LEGACY_DOUBLE_SHARED_SUM=1)\n"
            },
        )
        with self.assertRaises(ValueError):
            template.buildfile_overrides(BUILDFILE, "f16")

    def test_unsupported_runtime_refuses_before_outputs_change(self) -> None:
        canonical = (
            Path(__file__).parent / "generated/f64/memops.cuh"
        ).read_text()
        for precision in ("f32", "f64"):
            for runtime in (
                "// CASPAR_LEGACY_DOUBLE_SHARED_SUM has no implementation\n",
                canonical.replace("tot += other_tot;", "tot -= other_tot;"),
            ):
                with (
                    self.subTest(precision=precision, runtime=runtime[:40]),
                    TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    installed = root / "installed"
                    (installed / "source/runtime").mkdir(parents=True)
                    (installed / "source/runtime/memops.cuh").write_text(
                        runtime
                    )
                    output = root / "output"
                    output.mkdir()
                    retained = output / "memops.cuh"
                    retained.write_text("retained numerical profile")
                    environment = generation_namespace(
                        installed, output, precision
                    )
                    loader = environment["template_env"].loader
                    with self.assertRaisesRegex(
                        RuntimeError, "source/runtime/memops.cuh"
                    ):
                        exec(generation_tail(), environment)
                    environment["caslib"].generate.assert_not_called()
                    self.assertIs(environment["template_env"].loader, loader)
                    self.assertEqual(list(output.iterdir()), [retained])
                    self.assertEqual(
                        retained.read_text(), "retained numerical profile"
                    )

    def test_generation_restores_original_template_loader(self) -> None:
        for fail in (False, True):
            with self.subTest(fail=fail), TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "output"
                environment = generation_namespace(
                    root / "installed", output, "f64"
                )
                environment["validate_generation_inputs"] = Mock()
                loader = environment["template_env"].loader
                generate = environment["caslib"].generate
                if fail:
                    generate.side_effect = RuntimeError("generation failure")
                    with self.assertRaisesRegex(
                        RuntimeError, "generation failure"
                    ):
                        exec(generation_tail(), environment)
                else:
                    exec(generation_tail(), environment)
                self.assertIs(environment["template_env"].loader, loader)
                generate.assert_called_once_with(
                    output, use_symlinks=False, python_bindings=False
                )


if __name__ == "__main__":
    unittest.main()
