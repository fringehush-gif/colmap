#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Build complete SIFT owners against an already built, clean native baseline."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import native_common as q
from run_campaign import runtime_providers, LOADER_VARIABLES


OWNERS = ["src/colmap/feature/sift.cc", "src/thirdparty/VLFeat/sift.c",
          "src/thirdparty/VLFeat/covdet.c", "src/thirdparty/VLFeat/scalespace.c"]


def source_identity(path, expected):
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if head != expected:
        raise RuntimeError(f"Source HEAD mismatch: {path}: {head} != {expected}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError(f"Source checkout must have no tracked changes: {path}")
    return {"ref": head, "path": str(path), "tree": subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD^{tree}"], text=True).strip()}


def eigen_allocator_abi():
    entry, command = q.context("src/colmap/feature/sift.cc", q.OFFICIAL)
    command = list(command)
    for flag in ["-o", "-c"]:
        index = command.index(flag)
        del command[index:index + 2]
    command += ["-dM", "-E", "-x", "c++", "-"]
    result = subprocess.run(command, input="#include <Eigen/Core>\n", text=True,
                            capture_output=True, check=True, cwd=entry["directory"], env=q.ENV)
    match = re.search(r"^#define EIGEN_MALLOC_ALREADY_ALIGNED ([01])$", result.stdout, re.M)
    if not match:
        raise RuntimeError("Cannot determine the native Eigen allocator ABI")
    q.SAN[:] = [flag for flag in q.SAN if not flag.startswith("-DEIGEN_MALLOC_ALREADY_ALIGNED=")]
    q.SAN.append("-DEIGEN_MALLOC_ALREADY_ALIGNED=" + match.group(1))
    q.save(q.OUT / "eigen-allocator-abi.json", {"command": command, "cwd": entry["directory"], "value": int(match.group(1))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["native", "san"], default="native")
    args = parser.parse_args()
    for name in ["base_source", "candidate_source", "native_build", "out"]:
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {"A": source_identity(args.base_source, args.base_ref),
               "B": source_identity(args.candidate_source, args.candidate_ref)}
    q.configure(args.base_source, args.candidate_source, args.native_build, args.out)
    q.save(args.out / "sources.json", sources)
    eigen_allocator_abi()
    q.prepare()
    here = Path(__file__).resolve().parent
    manifest = {"mode": args.mode, "arms": {}, "frozen_files": {},
                "loader_environment": {key: q.ENV.get(key) for key in LOADER_VARIABLES}}
    stage = q.compile("src/colmap/controllers/feature_extraction.cc", "stage", q.OFFICIAL,
                      args.mode, here / "stage.cc")
    oracle = q.compile("src/colmap/feature/sift_test.cc", "raw-oracle", q.OFFICIAL,
                       args.mode, here / "raw_oracle.cc")
    witnesses = {}
    for arm, tree in [("A", q.OFFICIAL), ("B", q.WORK)]:
        owners = [q.compile(relative, arm + "-" + Path(relative).name, tree, args.mode)
                  for relative in OWNERS]
        test = q.compile("src/colmap/feature/sift_test.cc", arm + "-tests", tree, args.mode)
        binary = q.link([*owners, test], "src/colmap/feature/sift_test", arm + "-sift-test", args.mode)
        q.test(binary, arm + "-sift-test", selection="*:-*GPU*")
        stage_binary = q.link([*owners, stage], "src/colmap/controllers/feature_extraction_test",
                              arm + "-stage", args.mode)
        manifest["arms"][arm] = {**sources[arm], "binary": str(stage_binary),
                                  "binary_sha256": q.sha(stage_binary),
                                  "runtime_providers": runtime_providers(stage_binary, q.ENV)}
        kinds = [("ordinary", oracle)]
        if "vl_sift_calc_raw_descriptor_prepared" in (tree / "src/thirdparty/VLFeat/sift.h").read_text():
            source = q.OUT / (arm + "-prepared.cc")
            source.write_text('#define PREPARED_ORACLE 1\n#include "' + str(here / "raw_oracle.cc") + '"\n')
            prepared = q.compile("src/colmap/feature/sift_test.cc", arm + "-prepared", tree, args.mode, source)
            kinds.append(("prepared", prepared))
        for kind, obj in kinds:
            tag = arm + "-raw-" + kind
            probe = q.link([owners[1], obj], "src/colmap/feature/sift_test", tag, args.mode)
            witness = q.OUT / (tag + ".bin")
            q.run(tag + "-run", [probe, witness])
            if witness.stat().st_size != 2592 * 128 * 4:
                raise RuntimeError("Raw oracle has the wrong size")
            witnesses[tag] = {"sha256": q.sha(witness), "bytes": witness.stat().st_size}
    if len({row["sha256"] for row in witnesses.values()}) != 1:
        raise RuntimeError("Raw float oracle outputs disagree")
    q.save(q.OUT / "raw-oracles.json", witnesses)
    for artifact in q.OUT.glob("*-identities.json"):
        data = json.loads(artifact.read_text())
        manifest["frozen_files"].update(data.get("dependencies", {}))
    for artifact in q.OUT.glob("*-providers.json"):
        manifest["frozen_files"].update(json.loads(artifact.read_text()))
    for artifact in q.OUT.iterdir():
        if artifact.is_file() and artifact.suffix in [".json", ".log", ".xml", ".map", ".bin", ".o", ".d"]:
            manifest["frozen_files"][str(artifact)] = q.sha(artifact)
    for path in [*here.glob("*.py"), *here.glob("*.cc"), here / "photos.json",
                 here / "targets.json", here / "README.md",
                 q.BUILD / "CMakeCache.txt", q.BUILD / "compile_commands.json", q.BUILD / "build.ninja"]:
        manifest["frozen_files"][str(path)] = q.sha(path)
    for arm, source in sources.items():
        source_identity(Path(source["path"]), source["ref"])
    q.save(q.OUT / "campaign-manifest.json", manifest)
    print(json.dumps({"status": "qualified", "mode": args.mode,
                      "manifest": str(q.OUT / "campaign-manifest.json")}))


if __name__ == "__main__":
    main()
