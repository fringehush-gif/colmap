# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E501 -- Preserve exact upstream template anchors and fragments.

"""Process-local compatibility for two complete Caspar solver template shapes."""

_CORRECTNESS_REPAIRS = (
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
)

_RETAINED_PROPOSAL = (
    "        score_best_pcg = score_new_pcg;",
    """        score_best_pcg = score_new_pcg;
        // Retain the prediction before the next PCG update overwrites the step.
        pred_decrease_best_pcg = GetPredDecrease();""",
)
_TEST_ACCESS = (
    "private:\n  SolverParams<{{solver.linear_t}}> params_;",
    "private:\n  friend struct GraphSolverTestAccess;\n"
    "  SolverParams<{{solver.linear_t}}> params_;",
)


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            "Caspar solver template changed; review scored-proposal repairs"
        )
    return source.replace(old, new)


def patch_solver_templates(source: str, header: str) -> tuple[str, str]:
    """Preserve paired readback for complete legacy or owner-corrected inputs.

    Validate both templates before transforming either. Unknown, partial and
    already-overlaid inputs are refused; installed templates are never changed.
    """
    regions = (*_CORRECTNESS_REPAIRS, _RETAINED_PROPOSAL)
    legacy = all(
        source.count(old) == 1 and source.count(new) == 0
        for old, new in regions
    )
    corrected = all(
        source.count(new) == 1 and source.count(old) == new.count(old)
        for old, new in regions
    )
    old_friend, new_friend = _TEST_ACCESS
    friend_count = header.count("friend struct GraphSolverTestAccess;")
    legacy_header = (
        header.count(old_friend) == 1
        and header.count(new_friend) == 0
        and friend_count == 0
    )
    corrected_header = (
        header.count(new_friend) == 1
        and header.count(old_friend) == 0
        and friend_count == 1
    )
    if legacy and legacy_header:
        for old, new in _CORRECTNESS_REPAIRS:
            source = _replace_once(source, old, new)
        header = _replace_once(header, old_friend, new_friend)
        retention = _RETAINED_PROPOSAL[0]
    elif corrected and corrected_header:
        retention = _RETAINED_PROPOSAL[1]
    else:
        raise RuntimeError(
            "Caspar source/header repairs are incomplete or incompatible"
        )

    # Replace the owner's retained direct call with the same paired result.
    source = _replace_once(
        source,
        retention,
        """        score_best_pcg = score_new_pcg;
        pred_decrease_best_pcg = predicted_decrease_pcg;""",
    )
    replacements = (
        (
            "        {{solver.linear_t}} score_new_pcg = DoRetractScore();",
            """        {{solver.linear_t}} predicted_decrease_pcg;
        {{solver.linear_t}} score_new_pcg = DoRetractScore(&predicted_decrease_pcg);""",
        ),
        (
            "{{solver.linear_t}} {{ solver.struct_name }}::DoRetractScore() {",
            "{{solver.linear_t}} {{ solver.struct_name }}::DoRetractScore({{solver.linear_t}}* predicted_decrease) {",
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
        source = _replace_once(source, old, new)
    for old, new in (
        (
            "  {{solver.linear_t}} DoRetractScore();",
            "  {{solver.linear_t}} DoRetractScore({{solver.linear_t}}* predicted_decrease = nullptr);",
        ),
        (
            "  {{solver.linear_t}} GetPredDecrease();",
            """  void LaunchPredDecrease();
  {{solver.linear_t}} GetPredDecrease();""",
        ),
    ):
        header = _replace_once(header, old, new)
    return source, header
