# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E501 -- Preserve exact upstream template anchors and fragments.

"""Process-local repairs for compatible upstream Caspar solver templates."""


def patch_solver_template(source: str) -> str:
    """Keep COLMAP's scored-proposal repairs in the upstream template owner.

    Refuse template drift before generation instead of silently dropping a fix.
    The overlay is process-local and never changes the installed Symforce
    package.
    """
    replacements = (
        ("#include <chrono>", "#include <chrono>\n#include <cmath>"),
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
            "    bool have_pcg_proposal = false;\n"
            "    {{solver.linear_t}} pred_decrease_best_pcg = 0;",
        ),
        (
            "        if (!(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min)) {",
            """        if (!std::isfinite(score_new_pcg) ||
            (params_.pcg_rel_decrease_min != -1.0f &&
             !(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min))) {""",
        ),
        (
            "        score_best_pcg = score_new_pcg;",
            "        score_best_pcg = score_new_pcg;\n"
            "        pred_decrease_best_pcg = predicted_decrease_pcg;",
        ),
        (
            "        {{solver.linear_t}} score_new_pcg = DoRetractScore();",
            "        {{solver.linear_t}} predicted_decrease_pcg;\n"
            "        {{solver.linear_t}} score_new_pcg = "
            "DoRetractScore(&predicted_decrease_pcg);",
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
            "{{solver.linear_t}} {{ solver.struct_name }}::DoRetractScore() {",
            "{{solver.linear_t}} {{ solver.struct_name }}::DoRetractScore("
            "{{solver.linear_t}}* predicted_decrease) {",
        ),
        (
            """    return 0.5 * ReadCuMem({{solver_key("res_tot")}});
}

void {{ solver.struct_name }}::DoBeta() {""",
            """    if (predicted_decrease != nullptr) {
      // These two scratch scalars are allocated consecutively by the generator.
      if ({{solver_key("res_tot")}} != {{solver_key("pred_decrease_tot")}} + 1) {
        throw std::runtime_error("Caspar score/prediction scratch layout changed");
      }
      LaunchPredDecrease();
      {{solver.linear_t}} values[2];
      if (cudaMemcpy(values, {{solver_key("pred_decrease_tot")}}, sizeof(values),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("Caspar score/prediction readback failed");
      }
      *predicted_decrease = 0.5 * values[0];
      return 0.5 * values[1];
    }
    return 0.5 * ReadCuMem({{solver_key("res_tot")}});
}

void {{ solver.struct_name }}::DoBeta() {""",
        ),
        (
            "{{solver.linear_t}} {{ solver.struct_name }}::GetPredDecrease() {",
            "void {{ solver.struct_name }}::LaunchPredDecrease() {",
        ),
        (
            '    return 0.5 * ReadCuMem({{solver_key("pred_decrease_tot")}});',
            """}

{{solver.linear_t}} {{ solver.struct_name }}::GetPredDecrease() {
    LaunchPredDecrease();
    return 0.5 * ReadCuMem({{solver_key("pred_decrease_tot")}});""",
        ),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(
                "Caspar solver template changed; review scored-proposal repairs"
            )
        source = source.replace(old, new)
    return source


def patch_solver_header_template(source: str) -> str:
    for old, new in (
        (
            "private:\n  SolverParams<{{solver.linear_t}}> params_;",
            "private:\n  friend struct GraphSolverTestAccess;\n"
            "  SolverParams<{{solver.linear_t}}> params_;",
        ),
        (
            "  {{solver.linear_t}} DoRetractScore();",
            "  {{solver.linear_t}} DoRetractScore("
            "{{solver.linear_t}}* predicted_decrease = nullptr);",
        ),
        (
            "  {{solver.linear_t}} GetPredDecrease();",
            "  void LaunchPredDecrease();\n"
            "  {{solver.linear_t}} GetPredDecrease();",
        ),
    ):
        if source.count(old) != 1:
            raise RuntimeError("Caspar solver header template changed")
        source = source.replace(old, new)
    return source
