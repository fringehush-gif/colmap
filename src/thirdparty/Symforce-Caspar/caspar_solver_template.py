# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E501 -- Preserve exact upstream template anchors and fragments.

"""Small COLMAP overlays on the reviewed canonical Caspar generator."""

from hashlib import sha256
from pathlib import Path

# SymForce cfff699b with the reviewed SharedIndices owner follow-up. Keeping
# one source contract removes legacy repair paths and prevents regeneration
# from silently restoring an older numerical or readback implementation.
GENERATOR_OWNER = "fa394ffe696b7b0d5de10a900543f67ab8e0b104"
_GENERATION_INPUTS = {
    "source/runtime/memops.cuh": "2e5cbedb7f9c5a66f6ffd123c050e00546a2f951546627f42a4197b4ad22ce11",
    "source/templates/__init__.py": "fb4b26de0f6bf7d9eaeef928b4256e51159d2fb88e42685a239fabb2df4d736e",
    "code_generation/library.py": "251bd162ab85975c812a7e46991c83d91d9e52ff08953e8e7e067711b64154b4",
    "source/templates/solver.cc.jinja": "b3e316c409d9e2eff0c88c403ec980b0c3002548150f3a6c0ddbe01a57da4d53",
    "source/templates/solver.h.jinja": "a38bfe252e41a3d8c05f179a9cfee749931927545e8b444c0cc0e617f2794963",
    "source/templates/buildfiles/CMakeLists.txt.jinja": "724402194513c78f776609344c8d77368253346a0124f67aa92d18dd60609831",
    "source/runtime/solver_tools.cu": "6e55cfcecabe8a1f8e33493e215e16c2962451ce416912006032dad6a4399b2b",
    "source/runtime/shared_indices.cu": "9d0c1fa422728384f71bf5fcb1f5be54eb19d96f5213109134fc74af0d3f7e95",
    "code_generation/solver.py": "1c528c6130b58c0fee30d5a64fa24337e0d88cd39c747796a334b70d59ae317d",
    "code_generation/solver_utils.py": "63b5ddca9a0886f7247df4ad8df8248356b3c671acb999584925e7b351926673",
}


def validate_generation_inputs(
    caspar_dir: Path, templates: dict[str, str]
) -> None:
    """Check actual loaded templates and copied runtime before writing outputs."""
    for name, digest in _GENERATION_INPUTS.items():
        try:
            text = (
                templates[name.removeprefix("source/templates/")]
                if name.endswith(".jinja")
                else (caspar_dir / name).read_text(encoding="utf-8")
            )
        except (OSError, KeyError) as error:
            raise RuntimeError(
                f"Missing Caspar generator input: {name}"
            ) from error
        # Normalize a Windows checkout without accepting other source changes.
        if sha256(text.replace("\r\n", "\n").encode()).hexdigest() != digest:
            raise RuntimeError(
                f"Unsupported Caspar generator input: {name}. "
                f"Install SymForce {GENERATOR_OWNER}, or review this input "
                "before updating its digest. No output has been changed."
            )


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            "Caspar solver template changed; review COLMAP overlays"
        )
    return source.replace(old, new)


_RETAINED_PROPOSAL = """        score_best_pcg = score_new_pcg;
        // Retain the prediction before the next PCG update overwrites the step.
        pred_decrease_best_pcg = GetPredDecrease();"""

_READBACK_RANGE = """template <typename T, size_t Dimension, size_t StorageDimension>
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

_READBACK_GUARD = """  if (offset > {{num_key(nodetype)}} || num > {{num_key(nodetype)}} - offset) {
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
"""

_CHECK_READBACK_RANGE = """  CheckReadbackRange<{{solver.storage_t}}, {{Ops.storage_dim(nodetype)}}, {{solver.fields[node_key(nodetype, "storage_current")].dim_real}}>(
      offset, num, {{num_key(nodetype)}}, {{num_max_key(nodetype)}}, "{{nodetype.__name__}}");
  if (num == 0) {
    return;
  }
"""


def patch_solver_templates(source: str, header: str) -> tuple[str, str]:
    """Apply COLMAP overlays after validating the canonical generator inputs."""
    # Replace the owner's retained direct call with the same paired result.
    source = _replace_once(
        source,
        _RETAINED_PROPOSAL,
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
    source = _replace_once(
        source, "namespace {\n\n", "namespace {\n\n" + _READBACK_RANGE
    )
    if source.count(_READBACK_GUARD) != 2:
        raise RuntimeError(
            "Caspar readback template changed; review mapping bounds"
        )
    source = source.replace(_READBACK_GUARD, _CHECK_READBACK_RANGE)
    source = _replace_once(
        source,
        "kDimension * num * sizeof({{solver.storage_t}})",
        "{{Ops.storage_dim(nodetype)}} * num * sizeof({{solver.storage_t}})",
    )
    return source, header


def buildfile_overrides(source: str, precision: str) -> dict[str, str]:
    """Keep the shared double reduction order in this generated COLMAP profile."""
    if precision == "f32":
        return {}
    if precision != "f64":
        raise ValueError(f"Unexpected Caspar precision: {precision}")
    return {
        "buildfiles/CMakeLists.txt.jinja": source.rstrip()
        + "\n\ntarget_compile_definitions({{caslib.name}}_core PUBLIC "
        "CASPAR_LEGACY_DOUBLE_SHARED_SUM=1)\n"
    }
