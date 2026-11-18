# SPDX-License-Identifier: BSD-3-Clause
"""Qualify complete SIFT and VLFeat owners against unchanged pinned native support."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

OUT = OFFICIAL = BUILD = WORK = None
COMMANDS = NINJA = None
CXX = None
ARCHIVES = {}
SAN = ["-DEIGEN_MALLOC_ALREADY_ALIGNED=1", "-O1", "-g1", "-UNDEBUG", "-fsanitize=address,undefined,float-cast-overflow",
       "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", "-fno-pie"]
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           ASAN_OPTIONS="detect_leaks=0:halt_on_error=1",
           UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
def configure(base_source, candidate_source, native_build, output):
    global OUT, OFFICIAL, WORK, BUILD, COMMANDS, NINJA, CXX, ARCHIVES
    OUT, OFFICIAL, WORK, BUILD = map(Path, (output, base_source, candidate_source, native_build))
    COMMANDS = json.loads((BUILD / "compile_commands.json").read_text())
    NINJA = (BUILD / "build.ninja").read_text()
    cache = (BUILD / "CMakeCache.txt").read_text()
    CXX = re.search(r"^CMAKE_CXX_COMPILER:[^=]+=(.+)$", cache, re.M).group(1)
    ARCHIVES = {
        "src/colmap/feature/libcolmap_feature.a": OUT / "libcolmap_feature_support.a",
        "src/thirdparty/VLFeat/libcolmap_vlfeat.a": OUT / "libcolmap_vlfeat_support.a",
    }



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(name, args, expected=(0,), extra_env=None, timeout=900, cwd=None):
    args = list(map(str, args))
    cwd = cwd or OUT
    row = {"args": args, "cwd": str(cwd),
           "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "environment": {key: ENV[key] for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ASAN_OPTIONS", "UBSAN_OPTIONS"]}}
    for key in ["TSAN_OPTIONS", "OPENIMAGEIO_THREADS"]:
        if key in ENV:
            row["environment"][key] = ENV[key]
    record = OUT / (name + ".command.json")
    record.parent.mkdir(parents=True, exist_ok=True)
    save(record, row)
    with (OUT / (name + ".log")).open("w") as log:
        result = subprocess.run(args, cwd=cwd, env=dict(ENV, **(extra_env or {})),
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    row.update(returncode=result.returncode,
               end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(record, row)
    print(name, result.returncode, flush=True)
    if expected is not None:
        assert result.returncode in expected, name
    return result.returncode


def context(relative, tree):
    entry = next(item for item in COMMANDS if item["file"] == str(OFFICIAL / relative))
    args = shlex.split(entry["command"])
    if any(arg.startswith("@") for arg in args):
        raise RuntimeError("Response-file compilation requires a full matched build, not this selected-owner helper")
    for index, arg in enumerate(args):
        prefix = next((flag for flag in ["-isystem", "-iquote", "-idirafter", "-I"] if arg.startswith(flag)), None)
        if prefix is None:
            continue
        destination = index + 1 if arg == prefix else index
        value = args[destination] if arg == prefix else arg[len(prefix):]
        if value and not value.startswith(("/", "=")):
            value = str((Path(entry["directory"]) / value).resolve())
        args[destination] = value if arg == prefix else prefix + value
    args = [arg.replace(str(OFFICIAL / "src"), str(tree / "src")) for arg in args]
    return entry, args


def compile(relative, tag, tree, mode="native", source=None):
    entry, args = context(relative, tree)
    source = source or tree / relative
    obj, dep = OUT / (tag + ".o"), OUT / (tag + ".d")
    args[args.index("-c") + 1] = str(source)
    args[args.index("-o") + 1] = str(obj)
    args += ["-MD", "-MF", str(dep)]
    if mode == "san":
        args += SAN
    elif mode == "tsan":
        args += ["-O1", "-g1", "-UNDEBUG", "-fsanitize=thread",
                 "-fno-omit-frame-pointer", "-fno-pie"]
    run(tag + "-compile", args, cwd=entry["directory"])
    dependencies = {}
    for item in shlex.split(dep.read_text().replace("\\\n", " ").split(":", 1)[1]):
        path = Path(item)
        if not path.is_absolute():
            path = Path(entry["directory"]) / path
        dependencies[str(path.resolve())] = sha(path)
    save(OUT / (tag + "-identities.json"), {
        "source": str(source), "source_sha256": sha(source), "tree": str(tree),
        "object": str(obj), "object_sha256": sha(obj),
        "configured_command": entry, "dependencies": dependencies})
    return obj


def prepare():
    for relative, archive in ARCHIVES.items():
        original = BUILD / relative
        removed = ["sift.cc.o"] if "/feature/" in relative else ["sift.c.o", "covdet.c.o", "scalespace.c.o"]
        shutil.copyfile(original, archive)
        tag = "prune-" + archive.stem
        run(tag, ["ar", "d", archive, *removed])
        members = subprocess.check_output(["ar", "t", archive], text=True).splitlines()
        assert not set(members) & set(removed)
        hashes = {}
        for name in members:
            old = subprocess.check_output(["ar", "p", original, name])
            new = subprocess.check_output(["ar", "p", archive, name])
            assert old == new
            hashes[name] = hashlib.sha256(new).hexdigest()
        save(OUT / (tag + "-identities.json"), {
            "original": str(original), "original_sha256": sha(original),
            "pruned": str(archive), "pruned_sha256": sha(archive),
            "removed": removed, "retained_members": hashes})


def link(objects, target, tag, mode="native", extra_flags=()):
    block = re.search(r"^build " + re.escape(target) + r":.*?(?=\n\n)", NINJA, re.M | re.S).group(0)
    values = dict(re.findall(r"^  (\w+) = (.*)$", block, re.M))
    libraries, identities = [], {}
    for arg in shlex.split(values["LINK_LIBRARIES"]):
        if arg in ARCHIVES:
            arg = str(ARCHIVES[arg])
        elif not arg.startswith(("-", "/")):
            arg = str(BUILD / arg)
        libraries.append(arg)
        if not arg.startswith("-"):
            identities[str(Path(arg).resolve())] = sha(arg)
    binary = OUT / tag
    flags = [arg for arg in shlex.split(values.get("LINK_FLAGS", ""))
             if not arg.startswith("-Wl,--dependency-file=")]
    args = [str(CXX), *shlex.split(values["FLAGS"]), *flags,
            *map(str, objects), "-o", str(binary),
            "-Wl,-Map=" + str(OUT / (tag + ".map")), *extra_flags, *libraries]
    if mode == "san":
        args += ["-fsanitize=address,undefined,float-cast-overflow", "-no-pie"]
    elif mode == "tsan":
        args += ["-fsanitize=thread", "-no-pie"]
    run(tag + "-link", args)
    run(tag + "-ldd", ["ldd", binary])
    for line in (OUT / (tag + "-ldd.log")).read_text().splitlines():
        match = re.search(r"(?:=>\s+)?(/\S+)\s+\(", line)
        if match:
            path = Path(match.group(1)).resolve()
            identities[str(path)] = sha(path)
    save(OUT / (tag + "-providers.json"), identities)
    return binary


def test(binary, tag, expected=(0,), selection="*"):
    fixture = OUT / "fixtures" / tag
    fixture.mkdir(parents=True, exist_ok=True)
    xml = OUT / (tag + ".xml")
    rc = run(tag + "-tests", [binary, "--gtest_filter=" + selection,
                              "--gtest_output=xml:" + str(xml)],
             expected=expected, extra_env={"TMPDIR": str(fixture)}, timeout=180)
    if rc == 0 and not xml.exists():
        raise RuntimeError("Successful test process produced no XML receipt: " + tag)
    if xml.exists():
        result = ET.parse(xml).getroot().attrib
        if rc == 0:
            assert int(result["tests"]) > 0 and int(result["failures"]) == 0
            assert int(result.get("errors", 0)) == 0
            assert int(result.get("disabled", 0)) == 0 and int(result.get("skipped", 0)) == 0
        save(OUT / (tag + "-result.json"), {"returncode": rc, "xml": result,
              "binary_sha256": sha(binary), "selection": selection})
