#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Run a predeclared, exclusive, fresh-process CPU comparison with full witnesses."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import time


LOADER_VARIABLES = ["LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "LD_BIND_NOW",
                    "LD_ASSUME_KERNEL", "LD_HWCAP_MASK", "GLIBC_TUNABLES"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def runtime_providers(binary, environment):
    result = subprocess.run(["ldd", str(binary)], env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    if "not found" in result.stdout:
        raise RuntimeError("An ELF runtime provider could not be resolved")
    providers = {}
    for line in result.stdout.splitlines():
        match = re.search(r"(?:=>\s+)?(/\S+)\s+\(", line)
        if match:
            path = Path(match.group(1)).resolve()
            providers[str(path)] = digest(path)
    if not providers:
        raise RuntimeError("No ELF runtime providers were recorded")
    return providers


def system_state():
    state = {}
    for name in ["/proc/stat", "/proc/loadavg", "/proc/self/status",
                 "/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu.max"]:
        path = Path(name)
        if path.exists():
            state[name] = path.read_text()
    return state


def inspect_witness(path, inputs, metrics):
    """Independently parse the complete little-endian feature-output format."""
    data = Path(path).read_bytes()
    offset = 0

    def take(size):
        nonlocal offset
        if size < 0 or offset + size > len(data):
            raise ValueError("Truncated or invalid witness")
        value = data[offset:offset + size]
        offset += size
        return value

    def integer(size):
        return int.from_bytes(take(size), "little")

    if take(8) != b"SIFTOUT1" or integer(8) != len(inputs):
        raise ValueError("Wrong witness format or image count")
    products = []
    for expected in inputs:
        name = take(integer(8)).decode("utf-8")
        width, height, features = integer(4), integer(4), integer(8)
        if name != Path(expected["member"]).name or (width, height) != (3072, 2304):
            raise ValueError("Wrong image identity or resolution in witness")
        take(features * 6 * 4)
        descriptor_type, rows, columns = integer(4), integer(8), integer(8)
        if descriptor_type != 0 or rows != features or columns != 128:
            raise ValueError("Wrong SIFT descriptor layout in witness")
        take(rows * columns)
        products.append({"name": name, "width": width, "height": height,
                         "features": features, "descriptor_type": descriptor_type})
    if offset != len(data):
        raise ValueError("Trailing bytes in witness")
    if sum(p["features"] for p in products) != metrics["features"]:
        raise ValueError("Witness and reported total feature counts disagree")
    if [p["features"] for p in products] != metrics["per_photo_features"]:
        raise ValueError("Witness and reported per-photo counts disagree")
    return products


def plan(primary):
    other = "covariant" if primary == "dsp10" else "dsp10"
    rows = []
    for cohort, setting, order in [
        ("same_arm_before", primary, "AA"),
        ("primary", primary, "ABBABAAB"),
        ("related", other, "ABBA"),
        ("ordinary_control", "ordinary", "BAAB"),
        ("same_arm_after", primary, "BB"),
    ]:
        for position, arm in enumerate(order):
            rows.append({"cohort": cohort, "setting": setting,
                         "position": position, "arm": arm})
    return rows


def summarize(rows):
    result = {"completed_runs": len(rows), "cohorts": {}}
    for cohort in dict.fromkeys(r["cohort"] for r in rows):
        selected = [r for r in rows if r["cohort"] == cohort]
        group = {"runs": [r["id"] for r in selected], "arms": {}}
        for arm in sorted(set(r["arm"] for r in selected)):
            values = [r["metrics"]["wall_seconds"] for r in selected if r["arm"] == arm]
            group["arms"][arm] = {"wall_seconds": values,
                "mean": statistics.mean(values), "median": statistics.median(values),
                "range": max(values) - min(values)}
        pairs = []
        for offset in range(0, len(selected) - 1, 2):
            pair = selected[offset:offset + 2]
            if {r["arm"] for r in pair} != {"A", "B"}:
                continue
            a = next(r for r in pair if r["arm"] == "A")["metrics"]
            b = next(r for r in pair if r["arm"] == "B")["metrics"]
            pairs.append({"runs": [r["id"] for r in pair],
                "saved_seconds": {key: a[key] - b[key] for key in
                    ["wall_seconds", "cpu_seconds", "user_seconds", "system_seconds"]},
                "wall_reduction_percent": 100 * (a["wall_seconds"] - b["wall_seconds"]) / a["wall_seconds"]})
        group["pairs"] = pairs
        if pairs:
            group["median_paired_wall_saving"] = statistics.median(p["saved_seconds"]["wall_seconds"] for p in pairs)
            group["median_paired_wall_reduction_percent"] = statistics.median(p["wall_reduction_percent"] for p in pairs)
        result["cohorts"][cohort] = group
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--photos", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--primary", choices=["covariant", "dsp10"], required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--count", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--run-timeout", type=int, default=900)
    parser.add_argument("--deadline-seconds", type=int, default=3600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        execute(args)
    except BaseException as error:
        if not (args.out / "completed.json").exists():
            partial = args.out / "partial.json"
            status = json.loads(partial.read_text()) if partial.exists() else {}
            status.update(status="failed_or_partial", error_type=type(error).__name__, error=str(error))
            results = args.out / "results.json"
            status["completed_runs"] = len(json.loads(results.read_text())) if results.exists() else 0
            status["planned_runs"] = len(plan(args.primary))
            save(partial, status)
        raise


def execute(args):
    campaign_deadline = time.monotonic() + args.deadline_seconds
    if args.deadline_seconds <= 0 or args.run_timeout <= 0:
        raise ValueError("Timeouts must be positive")
    lock = Path("/tmp/colmap-cpu-qualification.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("mode") != "native":
        raise ValueError("Timing requires the native manifest, not a sanitized build")
    if set(manifest["arms"]) != {"A", "B"}:
        raise ValueError("Manifest must identify exactly arms A and B")
    allowed = sorted(os.sched_getaffinity(0))
    cpu = args.cpu if args.cpu is not None else allowed[0]
    if cpu not in allowed:
        raise ValueError("Requested CPU is outside the permitted affinity set")
    inputs = json.loads((Path(__file__).parent / "photos.json").read_text())
    for row in inputs["members"][:args.count]:
        path = args.photos / Path(row["member"]).name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"Photo identity mismatch: {path}")
    binaries = {}
    frozen = {str(args.manifest.resolve()): digest(args.manifest),
              str(Path(__file__).resolve()): digest(Path(__file__)),
              str((Path(__file__).parent / "photos.json").resolve()): digest(Path(__file__).parent / "photos.json")}
    for row in inputs["members"][:args.count]:
        frozen[str((args.photos / Path(row["member"]).name).resolve())] = row["sha256"]
    for arm, row in manifest["arms"].items():
        path = Path(row["binary"]).resolve()
        if digest(path) != row["binary_sha256"]:
            raise ValueError(f"Binary identity mismatch: {arm}")
        binaries[arm] = str(path)
        frozen[str(path)] = row["binary_sha256"]
    for path, expected in manifest["frozen_files"].items():
        path = str(Path(path).resolve())
        if digest(path) != expected:
            raise ValueError(f"Source/provider identity mismatch: {path}")
        if path in frozen and frozen[path] != expected:
            raise ValueError(f"Conflicting frozen identity: {path}")
        frozen[path] = expected
    save(args.out / "freeze.json", frozen)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               OPENIMAGEIO_THREADS="1")
    if {key: env.get(key) for key in LOADER_VARIABLES} != manifest["loader_environment"]:
        raise ValueError("ELF loader environment differs from qualification")
    for arm, row in manifest["arms"].items():
        if runtime_providers(binaries[arm], env) != row["runtime_providers"]:
            raise ValueError(f"Runtime providers differ under the timing environment: {arm}")
    schedule = plan(args.primary)
    save(args.out / "plan.json", {"manifest": manifest, "manifest_sha256": digest(args.manifest),
        "driver_sha256": digest(Path(__file__)), "photos": inputs,
        "schedule": schedule, "count": args.count, "cpu": cpu, "allowed_cpus": allowed,
        "deadline_seconds": args.deadline_seconds, "run_timeout_seconds": args.run_timeout,
        "platform": platform.platform(), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "thread_environment": {key: env[key] for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENIMAGEIO_THREADS"]},
        "limitation": "One benchmark process in this container; dedicated physical hardware is not established."})
    rows, canonical = [], {}
    for index, item in enumerate(schedule):
        remaining = campaign_deadline - time.monotonic()
        if remaining <= 0:
            save(args.out / "partial.json", {"reason": "campaign deadline", "completed_runs": len(rows), "planned_runs": len(schedule)})
            raise RuntimeError("Campaign deadline reached; fixed schedule was not shortened")
        run_id = f"{index:02d}-{item['cohort']}-{item['arm']}"
        directory = args.out / run_id
        directory.mkdir()
        witness, database = directory / "output.witness", directory / "database.db"
        command = ["taskset", "-c", str(cpu), binaries[item["arm"]], "controller",
                   str(args.photos.resolve()), str(args.count), item["setting"],
                   str(witness.resolve()), str(database.resolve())]
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        record = {**item, "id": run_id, "command": command,
                  "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "system_before": system_state()}
        save(directory / "run.json", record)
        start = time.monotonic()
        with (directory / "process.log").open("w") as log:
            try:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=min(args.run_timeout, remaining))
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        record.update(returncode=code, process_wall_seconds=time.monotonic() - start,
            process_user_seconds=after.ru_utime - before.ru_utime,
            process_system_seconds=after.ru_stime - before.ru_stime,
            system_after=system_state())
        save(directory / "run.json", record)
        if code != 0:
            save(args.out / "partial.json", {"reason": code, "failed_run": run_id, "completed_runs": len(rows), "planned_runs": len(schedule)})
            raise RuntimeError(f"Run failed; artifacts preserved: {run_id}: {code}")
        lines = (directory / "process.log").read_text().splitlines()
        metrics = [json.loads(line.removeprefix("METRICS ")) for line in lines if line.startswith("METRICS ")]
        if len(metrics) != 1:
            raise RuntimeError(f"Expected exactly one metrics record: {run_id}")
        record["metrics"] = metrics[0]
        for name in ["wall_seconds", "cpu_seconds", "user_seconds", "system_seconds"]:
            if metrics[0][name] < 0:
                raise RuntimeError(f"Invalid timing metric: {run_id}: {name}")
        if metrics[0]["photos"] != args.count or metrics[0]["setting"] != item["setting"]:
            raise RuntimeError(f"Driver settings mismatch: {run_id}")
        record["witness_bytes"] = witness.stat().st_size
        record["witness_sha256"] = digest(witness)
        record["witness_products"] = inspect_witness(witness, inputs["members"][:args.count], metrics[0])
        if item["setting"] not in canonical:
            if item["arm"] != "A":
                # The ordinary control starts with B; retain it until A anchors it.
                canonical[item["setting"]] = {"path": witness, "arm": "B"}
            else:
                canonical[item["setting"]] = {"path": witness, "arm": "A"}
        else:
            reference = canonical[item["setting"]]
            # This byte comparison is deliberately outside all stage timers.
            if reference["path"].read_bytes() != witness.read_bytes():
                save(directory / "run.json", record)
                raise RuntimeError(f"Complete output mismatch: {run_id}")
            record["full_byte_comparison"] = str(reference["path"])
            if item["arm"] == "A" and reference["arm"] == "B":
                canonical[item["setting"]] = {"path": witness, "arm": "A"}
        save(directory / "run.json", record)
        rows.append(record)
        save(args.out / "results.json", rows)
        save(args.out / "summary.json", summarize(rows))
        print(run_id, metrics[0]["wall_seconds"], record["witness_sha256"], flush=True)
    for path, expected in frozen.items():
        if digest(path) != expected:
            raise RuntimeError(f"Frozen input changed during campaign: {path}")
    for arm, row in manifest["arms"].items():
        if runtime_providers(binaries[arm], env) != row["runtime_providers"]:
            raise RuntimeError(f"Runtime provider resolution changed during campaign: {arm}")
    save(args.out / "freeze-verification.json", {"files_checked": len(frozen), "all_match": True})
    save(args.out / "completed.json", {"completed_runs": len(rows), "planned_runs": len(schedule),
        "canonical": {key: {"arm": value["arm"], "sha256": digest(value["path"])} for key, value in canonical.items()},
        "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
