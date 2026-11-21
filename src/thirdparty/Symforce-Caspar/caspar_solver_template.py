# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E501 -- Preserve exact upstream template anchors and fragments.

"""Process-local compatibility for two complete Caspar solver template shapes."""

from pathlib import Path

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

_PARAMETER_GUARDS = """  if (params.pcg_rel_error_exit <= 0.0f) {
    throw std::runtime_error("params.pcg_rel_error_exit must be positive");
  }
  if (params.diag_init < 0.0f) {
    throw std::runtime_error("params.diag_init must be positive");
  }
  if (params.pcg_iter_max < 1) {
    throw std::runtime_error("params.pcg_iter_max must be at least 1");
  }
"""
_PARAMETER_REGIONS = (
    (
        "  indices_valid_ = false;\n"
        + _PARAMETER_GUARDS.replace("exit <=", "exit<=").replace(") {", "){")
        + "  allocation_size_ = get_nbytes();",
        "  indices_valid_ = false;\n  set_params(params);\n"
        "  allocation_size_ = get_nbytes();",
    ),
    (
        "void {{ solver.struct_name }}::set_params(const SolverParams<double> &params){\n"
        "  this->params_ = params;\n}",
        "void {{ solver.struct_name }}::set_params(const SolverParams<double> &params){\n"
        + _PARAMETER_GUARDS
        + "  this->params_ = params;\n}",
    ),
)


def _patch_parameter_validation(source: str) -> str:
    """Retain the owner's shared constructor/setter precondition checks."""
    if all(
        source.count(new) == 1 and source.count(old) == 0
        for old, new in _PARAMETER_REGIONS
    ):
        return source
    if not all(
        source.count(old) == 1 and source.count(new) == 0
        for old, new in _PARAMETER_REGIONS
    ):
        raise RuntimeError(
            "Caspar parameter validation template is incomplete or incompatible"
        )
    for old, new in _PARAMETER_REGIONS:
        source = source.replace(old, new)
    return source


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
    source = _patch_parameter_validation(source)
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


_READBACK_HELPERS = """void CheckReadbackCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             cudaGetErrorString(status));
  }
}

template <typename T, size_t Dimension, size_t StorageDimension>
void CheckReadbackRange(size_t offset, size_t num, size_t count,
                        size_t capacity, const char* node) {
  static_assert(Dimension > 0 && StorageDimension >= Dimension);
  if (offset > count || num > count - offset) {
    throw std::runtime_error(std::string(node) +
                             " readback range exceeds node count");
  }
  if (num == 0) {
    return;
  }
  // Mappers use unsigned component indices and increment output lanes by 32.
  constexpr size_t kIndexLimit = std::numeric_limits<unsigned int>::max() - 32;
  if (capacity > kIndexLimit / StorageDimension ||
      num > std::numeric_limits<size_t>::max() / sizeof(T) / Dimension) {
    throw std::runtime_error(std::string(node) +
                             " readback exceeds mapping argument limits");
  }
}

"""

_READBACK_GETTERS = (
    """void {{solver.struct_name}}::Get{{nodetype.__name__}}NodesToStackedHost(
    {{solver.storage_t}}* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > {{num_key(nodetype)}}){
    throw std::runtime_error(std::to_string(offset + num) + " > {{num_key(nodetype)}}");
  }
  {{nodetype.__name__}}CasparToStacked({{node_key(nodetype, "storage_current")}},
                                          marker__scratch_inout_, {{num_max_key(nodetype)}},
                                          offset, num);
  cudaMemcpy(data, marker__scratch_inout_, {{Ops.storage_dim(nodetype)}} * num * sizeof({{solver.storage_t}}),
             cudaMemcpyDeviceToHost);
}

void {{solver.struct_name}}::Get{{nodetype.__name__}}NodesToStackedDevice(
    {{solver.storage_t}}* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > {{num_key(nodetype)}}){
    throw std::runtime_error(std::to_string(offset + num) + " > {{num_key(nodetype)}}");
  }
  {{nodetype.__name__}}CasparToStacked({{node_key(nodetype, "storage_current")}}, data,
                                          {{num_max_key(nodetype)}}, offset, num);
}
""",
    """void {{solver.struct_name}}::Get{{nodetype.__name__}}NodesToStackedHost(
    {{solver.storage_t}}* const data, const size_t offset, const size_t num) {
  CheckReadbackRange<{{solver.storage_t}}, {{Ops.storage_dim(nodetype)}}, {{solver.fields[node_key(nodetype, "storage_current")].dim_real}}>(
      offset, num, {{num_key(nodetype)}}, {{num_max_key(nodetype)}}, "{{nodetype.__name__}}");
  if (num == 0) {
    return;
  }
  CheckReadbackCuda(cudaSetDevice(device_id_), "cudaSetDevice");
  CheckReadbackCuda(
      {{nodetype.__name__}}CasparToStacked({{node_key(nodetype, "storage_current")}},
                            marker__scratch_inout_, {{num_max_key(nodetype)}}, offset, num),
      "{{nodetype.__name__}}CasparToStacked");
  CheckReadbackCuda(cudaMemcpy(data, marker__scratch_inout_,
                               {{Ops.storage_dim(nodetype)}} * num * sizeof({{solver.storage_t}}), cudaMemcpyDeviceToHost),
                    "cudaMemcpy device-to-host");
}

void {{solver.struct_name}}::Get{{nodetype.__name__}}NodesToStackedDevice(
    {{solver.storage_t}}* const data, const size_t offset, const size_t num) {
  CheckReadbackRange<{{solver.storage_t}}, {{Ops.storage_dim(nodetype)}}, {{solver.fields[node_key(nodetype, "storage_current")].dim_real}}>(
      offset, num, {{num_key(nodetype)}}, {{num_max_key(nodetype)}}, "{{nodetype.__name__}}");
  if (num == 0) {
    return;
  }
  CheckReadbackCuda(cudaSetDevice(device_id_), "cudaSetDevice");
  CheckReadbackCuda(
      {{nodetype.__name__}}CasparToStacked({{node_key(nodetype, "storage_current")}},
                            data, {{num_max_key(nodetype)}}, offset, num),
      "{{nodetype.__name__}}CasparToStacked");
}
""",
)

_READBACK_SCRATCH = (
    """    at_least = std::max(at_least, offset + {{solver.at_least[name]}}*sizeof({{field.dtype}}));""",
    """    {% if name == "marker__scratch_inout_" %}
    {# Include factor constants and host indices in the transfer scratch. #}
    at_least = std::max(at_least, offset + std::max({
      {% for nodetype in solver.node_types %}
      {{Ops.storage_dim(nodetype)}} * {{num_max_key(nodetype)}},
      {% endfor %}
      {% for fac in solver.factors %}
      {% set scratch = namespace(dim=0) %}
      {% for arg, typ in fac.const_arg_types.items() %}
        {% if fac.isconstuniq[arg] %}
      size_t({{Ops.storage_dim(typ)}}),
        {% else %}
          {% set scratch.dim = [scratch.dim, Ops.storage_dim(typ)] | max %}
        {% endif %}
      {% endfor %}
      {% for shared in fac.isnodeshared.values() | list + fac.isconstshared.values() | list %}
        {% if shared %}{% set scratch.dim = [scratch.dim, 1] | max %}{% endif %}
      {% endfor %}
      {% if scratch.dim %}
      {{scratch.dim}} * {{num_max_key(fac)}},
      {% endif %}
      {% endfor %}
      size_t(0)}) * sizeof({{field.dtype}}));
    {% else %}
    at_least = std::max(at_least, offset + {{solver.at_least[name]}}*sizeof({{field.dtype}}));
    {% endif %}""",
)

_SCALAR_READBACK = (
    """  T result;
  cudaMemcpy(&result, data, sizeof(T), cudaMemcpyDeviceToHost);
  return result;""",
    """  T result{};
  const cudaError_t status =
      cudaMemcpy(&result, data, sizeof(T), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("ReadCuMem cudaMemcpy failed: ") +
                             cudaGetErrorString(status));
  }
  return result;""",
)


def patch_readback_template(source: str) -> str:
    """Repair the known getter/scratch owner; refuse template drift."""
    source = _replace_once(
        source,
        "#include <chrono>",
        "#include <chrono>\n#include <limits>\n#include <string>",
    )
    source = _replace_once(
        source, "namespace {\n\n", "namespace {\n\n" + _READBACK_HELPERS
    )
    for old, new in (_READBACK_GETTERS, _READBACK_SCRATCH):
        source = _replace_once(source, old, new)
    return source


def patch_solver_tools(source: str) -> str:
    """Check the scalar read copied from the supported SymForce runtime."""
    source = _replace_once(source, *_SCALAR_READBACK)
    return _replace_once(
        source,
        '#include "solver_tools.h"',
        '#include <stdexcept>\n#include <string>\n\n#include "solver_tools.h"',
    )


def write_solver_tools(path: Path, source: str) -> None:
    """Replace copied runtime output without following a prior generator symlink."""
    path.unlink(missing_ok=True)
    path.write_text(source)
