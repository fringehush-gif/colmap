# SPDX-License-Identifier: BSD-3-Clause

import importlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pycolmap
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(pycolmap, "PoissonMeshingOptions"),
    reason="PoissonMeshingOptions not available",
)


def test_poisson_meshing_options_init() -> None:
    options = pycolmap.PoissonMeshingOptions()
    assert options is not None


def test_poisson_meshing_options_depth() -> None:
    options = pycolmap.PoissonMeshingOptions()
    options.depth = 10
    assert options.depth == 10


def test_poisson_meshing_options_num_threads() -> None:
    options = pycolmap.PoissonMeshingOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_poisson_meshing_options_check() -> None:
    options = pycolmap.PoissonMeshingOptions()
    assert options.check()


def _write_oriented_cube(path: Path, *, with_normals: bool) -> None:
    # Match the small 96-sample, outward-facing cube used by the native test.
    lines = [
        "ply",
        "format ascii 1.0",
        "element vertex 96",
        "property float x",
        "property float y",
        "property float z",
    ]
    if with_normals:
        lines += ["property float nx", "property float ny", "property float nz"]
    lines += [
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for axis in range(3):
        for sign in (-1, 1):
            for u in range(4):
                for v in range(4):
                    position = [0.0, 0.0, 0.0]
                    normal = [0.0, 0.0, 0.0]
                    position[axis] = float(sign)
                    position[(axis + 1) % 3] = (2 * u - 3) * 0.25
                    position[(axis + 2) % 3] = (2 * v - 3) * 0.25
                    normal[axis] = float(sign)
                    fields = position + (normal if with_normals else [])
                    fields += [17, 109, 233]
                    lines.append(" ".join(map(str, fields)))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _small_poisson_options() -> pycolmap.PoissonMeshingOptions:
    options = pycolmap.PoissonMeshingOptions()
    options.point_weight = 1.0
    options.depth = 4
    options.trim = 0.0
    options.color = True
    options.num_threads = 1
    return options


def test_poisson_meshing_returns_true(tmp_path: Path) -> None:
    input_path = tmp_path / "cube.ply"
    output_path = tmp_path / "mesh.ply"
    _write_oriented_cube(input_path, with_normals=True)

    options = _small_poisson_options()
    assert pycolmap.poisson_meshing(input_path, output_path, options) is True
    assert output_path.stat().st_size > 4
    with output_path.open("rb") as stream:
        assert stream.read(4) == b"ply\n"


def test_poisson_meshing_keeps_precondition_exceptions(tmp_path: Path) -> None:
    output_path = tmp_path / "mesh.ply"
    options = _small_poisson_options()
    with pytest.raises(ValueError, match="does not exist"):
        pycolmap.poisson_meshing(tmp_path / "missing.ply", output_path, options)
    with pytest.raises(ValueError, match="does not match file extension"):
        pycolmap.poisson_meshing(tmp_path / "points.txt", output_path, options)
    assert not output_path.exists()


def test_poisson_meshing_returns_false_for_missing_normals_in_child(
    tmp_path: Path,
) -> None:
    # Missing normals fail during PoissonRecon header preflight, before the
    # stream constructor. Keep this early-error cleanup path in one child.
    input_path = tmp_path / "missing_normals.ply"
    output_path = tmp_path / "mesh.ply"
    _write_oriented_cube(input_path, with_normals=False)
    script = textwrap.dedent(
        """\
        import importlib
        import json
        import sys
        from pathlib import Path

        sys.path[:] = json.loads(sys.argv[1])
        import pycolmap

        core = importlib.import_module("pycolmap._core")
        assert Path(core.__file__).resolve() == Path(sys.argv[2]).resolve()
        options = pycolmap.PoissonMeshingOptions()
        options.point_weight = 1.0
        options.depth = 4
        options.trim = 0.0
        options.color = True
        options.num_threads = 1
        result = pycolmap.poisson_meshing(
            Path(sys.argv[3]), Path(sys.argv[4]), options
        )
        assert result is False, result
        print("POISSON_HANDLED_FALSE")
        """
    )
    core_path = importlib.import_module("pycolmap._core").__file__
    assert core_path is not None
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            json.dumps(sys.path),
            str(Path(core_path).resolve()),
            str(input_path),
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "POISSON_HANDLED_FALSE" in result.stdout


def test_mesh_simplification_options_init() -> None:
    options = pycolmap.MeshSimplificationOptions()
    assert options is not None


def test_mesh_simplification_options_target_face_ratio() -> None:
    options = pycolmap.MeshSimplificationOptions()
    options.target_face_ratio = 0.5
    assert options.target_face_ratio == pytest.approx(0.5)


def test_mesh_simplification_options_check() -> None:
    options = pycolmap.MeshSimplificationOptions()
    assert options.check()


@pytest.mark.skipif(
    not hasattr(pycolmap, "DelaunayMeshingOptions"),
    reason="DelaunayMeshingOptions not available",
)
def test_delaunay_meshing_options_init() -> None:
    options = pycolmap.DelaunayMeshingOptions()
    assert options is not None
