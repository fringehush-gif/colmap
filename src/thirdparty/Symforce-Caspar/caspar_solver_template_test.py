# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E501 -- Preserve exact upstream template anchors and fragments.

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "caspar_solver_template",
    Path(__file__).with_name("caspar_solver_template.py"),
)
template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template)


# Only the repaired regions are needed; no SymForce, CUDA or generated kernels.
SOURCE = """#include <chrono>
void {{ solver.struct_name }}::DoUpdateRFirst() {
    Zero({{solver_key("r_0_norm2_tot")}}, {{solver_key("r_0_norm2_tot")}}+1);

    {% for nodetype in solver.node_types %}
}
void solve() {
    bool have_pcg_proposal = false;
        {{solver.linear_t}} score_new_pcg = DoRetractScore();
        if (!(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min)) {
        score_best_pcg = score_new_pcg;
      quality = (score_best - score_best_pcg) / GetPredDecrease();
}
{{solver.linear_t}} {{ solver.struct_name }}::DoRetractScore() {
    return 0.5 * ReadCuMem({{solver_key("res_tot")}});
}

void {{ solver.struct_name }}::DoBeta() {
}
{{solver.linear_t}} {{ solver.struct_name }}::GetPredDecrease() {
    return 0.5 * ReadCuMem({{solver_key("pred_decrease_tot")}});
}
"""

HEADER = """class {{ solver.struct_name }} {
  {{solver.linear_t}} DoRetractScore();
  {{solver.linear_t}} GetPredDecrease();
private:
  SolverParams<{{solver.linear_t}}> params_;
};
"""


REPAIR_REGIONS = (
    (
        "#include <chrono>",
        """#include <chrono>
#include <cmath>""",
    ),
    (
        """void {{ solver.struct_name }}::DoUpdateRFirst() {
    Zero({{solver_key("r_0_norm2_tot")}}, {{solver_key("r_0_norm2_tot")}}+1);

    {% for nodetype in solver.node_types %}""",
        """void {{ solver.struct_name }}::DoUpdateRFirst() {
    Zero({{solver_key("r_0_norm2_tot")}}, {{solver_key("r_0_norm2_tot")}}+1);
    Zero({{solver_key("r_kp1_norm2_tot")}}, {{solver_key("r_kp1_norm2_tot")}}+1);

    {% for nodetype in solver.node_types %}""",
    ),
    (
        "    bool have_pcg_proposal = false;",
        """    bool have_pcg_proposal = false;
    {{solver.linear_t}} pred_decrease_best_pcg = 0;""",
    ),
    (
        "        if (!(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min)) {",
        """        if (!std::isfinite(score_new_pcg) ||
            (params_.pcg_rel_decrease_min != -1.0f &&
             !(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min))) {""",
    ),
    (
        "      quality = (score_best - score_best_pcg) / GetPredDecrease();",
        """      const {{solver.linear_t}} pred_decrease =
          params_.pcg_rel_decrease_min == -1.0f &&
                  params_.pcg_rel_score_exit == -1.0f
              ? GetPredDecrease()
              : pred_decrease_best_pcg;
      quality = (score_best - score_best_pcg) / pred_decrease;""",
    ),
    (
        "        score_best_pcg = score_new_pcg;",
        """        score_best_pcg = score_new_pcg;
        // Retain the prediction before the next PCG update overwrites the step.
        pred_decrease_best_pcg = GetPredDecrease();""",
    ),
)


# Keep the owner fixture and partial-repair cases on the same independent data.
OWNER_SOURCE = SOURCE
for original, repaired in REPAIR_REGIONS:
    OWNER_SOURCE = OWNER_SOURCE.replace(original, repaired)
OWNER_HEADER = HEADER.replace(
    "private:\n", "private:\n  friend struct GraphSolverTestAccess;\n"
)


class CasparSolverTemplateTest(unittest.TestCase):
    def test_buildfile_precision_profile(self) -> None:
        from jinja2 import Environment

        source = "add_library({{caslib.name}}_core STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})\n"
        self.assertEqual(template.buildfile_overrides(source, "f32"), {})
        override = template.buildfile_overrides(source, "f64")
        rendered = (
            Environment()
            .from_string(override["buildfiles/CMakeLists.txt.jinja"])
            .render(caslib=type("Caslib", (), {"name": "caspar_lib"})())
        )
        self.assertIn(
            "target_compile_definitions(caspar_lib_core PUBLIC CASPAR_LEGACY_DOUBLE_SHARED_SUM=1)",
            rendered,
        )
        for invalid in (
            "",
            source + source,
            source + "CASPAR_LEGACY_DOUBLE_SHARED_SUM",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(RuntimeError):
                    template.buildfile_overrides(invalid, "f64")
        with self.assertRaises(ValueError):
            template.buildfile_overrides(source, "other")

    def test_repairs_scored_proposals_and_both_norm_totals(self) -> None:
        result, _ = template.patch_solver_templates(SOURCE, HEADER)
        for expected in (
            "#include <cmath>",
            'Zero({{solver_key("r_kp1_norm2_tot")}}',
            "!std::isfinite(score_new_pcg)",
            "params_.pcg_rel_decrease_min != -1.0f &&",
            "pred_decrease_best_pcg = predicted_decrease_pcg;",
            "DoRetractScore(&predicted_decrease_pcg)",
            "quality = (score_best - score_best_pcg) / pred_decrease;",
            "if (predicted_decrease != nullptr)",
            "cudaMemcpyDeviceToHost) != cudaSuccess",
            "Caspar score/prediction scratch layout changed",
            "Caspar score/prediction readback failed",
            "*predicted_decrease = 0.5 * values[0];",
            "return 0.5 * values[1];",
        ):
            self.assertIn(expected, result)
        self.assertEqual(
            result.count("void {{ solver.struct_name }}::LaunchPredDecrease()"),
            1,
        )
        self.assertEqual(
            result.count(
                "{{solver.linear_t}} {{ solver.struct_name }}::GetPredDecrease()"
            ),
            1,
        )

    def test_header_matches_optional_readback_and_test_access(self) -> None:
        _, result = template.patch_solver_templates(SOURCE, HEADER)
        self.assertIn(
            "{{solver.linear_t}}* predicted_decrease = nullptr", result
        )
        self.assertIn("void LaunchPredDecrease();", result)
        self.assertEqual(
            result.count("friend struct GraphSolverTestAccess;"), 1
        )

    def test_missing_or_duplicate_anchors_are_refused(self) -> None:
        common_anchors = (
            "        {{solver.linear_t}} score_new_pcg = DoRetractScore();",
            "{{solver.linear_t}} {{ solver.struct_name }}::DoRetractScore() {",
            "{{solver.linear_t}} {{ solver.struct_name }}::GetPredDecrease() {",
            '    return 0.5 * ReadCuMem({{solver_key("pred_decrease_tot")}});',
            """    return 0.5 * ReadCuMem({{solver_key("res_tot")}});
}

void {{ solver.struct_name }}::DoBeta() {""",
        )
        for corrected, source, header in (
            (False, SOURCE, HEADER),
            (True, OWNER_SOURCE, OWNER_HEADER),
        ):
            anchors = (
                *common_anchors,
                *(region[corrected] for region in REPAIR_REGIONS),
            )
            for anchor in anchors:
                for duplicate in (False, True):
                    with self.subTest(
                        corrected=corrected, anchor=anchor, duplicate=duplicate
                    ):
                        changed = (
                            source + "\n" + anchor
                            if duplicate
                            else source.replace(anchor, "", 1)
                        )
                        with self.assertRaises(RuntimeError):
                            template.patch_solver_templates(changed, header)
            for anchor in (
                "  {{solver.linear_t}} DoRetractScore();",
                "  {{solver.linear_t}} GetPredDecrease();",
                "friend struct GraphSolverTestAccess;"
                if corrected
                else "private:\n  SolverParams<{{solver.linear_t}}> params_;",
            ):
                for duplicate in (False, True):
                    with self.subTest(
                        corrected=corrected,
                        header_anchor=anchor,
                        duplicate=duplicate,
                    ):
                        changed = (
                            header + "\n" + anchor
                            if duplicate
                            else header.replace(anchor, "", 1)
                        )
                        with self.assertRaises(RuntimeError):
                            template.patch_solver_templates(source, changed)

    def test_already_repaired_templates_are_refused(self) -> None:
        source, header = template.patch_solver_templates(SOURCE, HEADER)
        for candidate in ((source, header), (source, HEADER), (SOURCE, header)):
            with self.subTest(candidate=candidate):
                with self.assertRaises(RuntimeError):
                    template.patch_solver_templates(*candidate)

    def test_already_reset_second_norm_is_refused(self) -> None:
        original = '    Zero({{solver_key("r_0_norm2_tot")}}, {{solver_key("r_0_norm2_tot")}}+1);'
        repaired = (
            original
            + '\n    Zero({{solver_key("r_kp1_norm2_tot")}}, {{solver_key("r_kp1_norm2_tot")}}+1);'
        )
        with self.assertRaises(RuntimeError):
            template.patch_solver_templates(
                SOURCE.replace(original, repaired), HEADER
            )

    def test_corrected_owner_converges_to_exact_legacy_output(self) -> None:
        self.assertEqual(
            template.patch_solver_templates(SOURCE, HEADER),
            template.patch_solver_templates(OWNER_SOURCE, OWNER_HEADER),
        )

    def test_partial_owner_repairs_are_refused(self) -> None:
        for old, new in REPAIR_REGIONS:
            for source, header in (
                (SOURCE.replace(old, new), HEADER),
                (OWNER_SOURCE.replace(new, old), OWNER_HEADER),
            ):
                with self.subTest(region=new, source=source):
                    with self.assertRaises(RuntimeError):
                        template.patch_solver_templates(source, header)

    def test_mixed_source_and_header_shapes_are_refused(self) -> None:
        for source, header in ((SOURCE, OWNER_HEADER), (OWNER_SOURCE, HEADER)):
            with self.subTest(source=source):
                with self.assertRaises(RuntimeError):
                    template.patch_solver_templates(source, header)


if __name__ == "__main__":
    unittest.main()
