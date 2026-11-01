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


class CasparSolverTemplateTest(unittest.TestCase):
    def test_repairs_scored_proposals_and_both_norm_totals(self) -> None:
        result = template.patch_solver_template(SOURCE)
        self.assertIn("#include <cmath>", result)
        self.assertIn('Zero({{solver_key("r_kp1_norm2_tot")}}', result)
        self.assertIn("!std::isfinite(score_new_pcg)", result)
        self.assertIn("params_.pcg_rel_decrease_min != -1.0f &&", result)
        self.assertIn(
            "pred_decrease_best_pcg = predicted_decrease_pcg;", result
        )
        self.assertIn("DoRetractScore(&predicted_decrease_pcg)", result)
        self.assertIn(
            "quality = (score_best - score_best_pcg) / pred_decrease;", result
        )
        self.assertIn("if (predicted_decrease != nullptr)", result)
        self.assertIn("cudaMemcpyDeviceToHost) != cudaSuccess", result)
        self.assertIn("Caspar score/prediction scratch layout changed", result)
        self.assertIn("Caspar score/prediction readback failed", result)
        self.assertIn("*predicted_decrease = 0.5 * values[0];", result)
        self.assertIn("return 0.5 * values[1];", result)
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
        result = template.patch_solver_header_template(HEADER)
        self.assertIn(
            "{{solver.linear_t}}* predicted_decrease = nullptr", result
        )
        self.assertIn("void LaunchPredDecrease();", result)
        self.assertIn("friend struct GraphSolverTestAccess;", result)

    def test_missing_or_duplicate_anchors_are_refused(self) -> None:
        cases = (
            (
                template.patch_solver_template,
                SOURCE,
                (
                    "#include <chrono>",
                    "    bool have_pcg_proposal = false;",
                    "        score_best_pcg = score_new_pcg;",
                    "        {{solver.linear_t}} score_new_pcg = DoRetractScore();",
                    "      quality = (score_best - score_best_pcg) / GetPredDecrease();",
                    "{{solver.linear_t}} {{ solver.struct_name }}::DoRetractScore() {",
                    "{{solver.linear_t}} {{ solver.struct_name }}::GetPredDecrease() {",
                    '    return 0.5 * ReadCuMem({{solver_key("pred_decrease_tot")}});',
                    "        if (!(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min)) {",
                    """void {{ solver.struct_name }}::DoUpdateRFirst() {
    Zero({{solver_key("r_0_norm2_tot")}}, {{solver_key("r_0_norm2_tot")}}+1);

    {% for nodetype in solver.node_types %}""",
                    """    return 0.5 * ReadCuMem({{solver_key("res_tot")}});
}

void {{ solver.struct_name }}::DoBeta() {""",
                ),
            ),
            (
                template.patch_solver_header_template,
                HEADER,
                (
                    "  {{solver.linear_t}} DoRetractScore();",
                    "  {{solver.linear_t}} GetPredDecrease();",
                    "private:\n  SolverParams<{{solver.linear_t}}> params_;",
                ),
            ),
        )
        for transform, source, anchors in cases:
            for anchor in anchors:
                for mutation in ("missing", "duplicate"):
                    with self.subTest(anchor=anchor, mutation=mutation):
                        changed = (
                            source.replace(anchor, "", 1)
                            if mutation == "missing"
                            else source + "\n" + anchor
                        )
                        with self.assertRaises(RuntimeError):
                            transform(changed)

    def test_already_repaired_templates_are_refused(self) -> None:
        for transform, source in (
            (template.patch_solver_template, SOURCE),
            (template.patch_solver_header_template, HEADER),
        ):
            with self.subTest(transform=transform.__name__):
                with self.assertRaises(RuntimeError):
                    transform(transform(source))

    def test_already_reset_second_norm_is_refused(self) -> None:
        original = '    Zero({{solver_key("r_0_norm2_tot")}}, {{solver_key("r_0_norm2_tot")}}+1);'
        repaired = (
            original
            + '\n    Zero({{solver_key("r_kp1_norm2_tot")}}, {{solver_key("r_kp1_norm2_tot")}}+1);'
        )
        with self.assertRaises(RuntimeError):
            template.patch_solver_template(SOURCE.replace(original, repaired))


if __name__ == "__main__":
    unittest.main()
