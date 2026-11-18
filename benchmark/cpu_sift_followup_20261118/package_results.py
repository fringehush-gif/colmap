#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Package qualification evidence without overwriting or deleting its inputs."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import zipfile
import zlib

SMALL_BINARY_BYTES = 64 * 1024 * 1024
METADATA = {".json", ".log", ".xml", ".map", ".d", ".txt", ".md",
            ".stdout", ".stderr", ".command", ".py", ".cc", ".c", ".patch"}
TIMES = ("wall_seconds", "cpu_seconds", "user_seconds", "system_seconds")


def identity(path):
    sha, crc, size = hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    return {"bytes": size, "sha256": sha.hexdigest(), "crc32": f"{crc:08x}"}


def read_json(path, warnings):
    if not path.is_file():
        return None
    try:
        def reject(value):
            raise ValueError("Nonfinite JSON value: " + value)
        return json.loads(path.read_text(), parse_constant=reject)
    except (OSError, UnicodeError, ValueError) as error:
        warnings.append(f"Cannot interpret {path}: {error}; raw file retained")
        return None


def files(root):
    for directory, subdirs, names in os.walk(root, followlinks=False):
        subdirs[:] = sorted(d for d in subdirs
                           if d != "__pycache__" and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if not path.is_symlink() and path.is_file():
                yield path


def database(path):
    if re.search(r"\.(?:db|sqlite|sqlite3)(?:-(?:wal|shm|journal))?$", path.name, re.I):
        return True
    with path.open("rb") as stream:
        return stream.read(16) == b"SQLite format 3\x00"


def expected_schedule(primary):
    related = "covariant" if primary == "dsp10" else "dsp10"
    rows = []
    for cohort, setting, order in [("same_arm_before", primary, "AA"),
                                  ("primary", primary, "ABBABAAB"),
                                  ("related", related, "ABBA"),
                                  ("ordinary_control", "ordinary", "BAAB"),
                                  ("same_arm_after", primary, "BB")]:
        for position, arm in enumerate(order):
            rows.append({"cohort": cohort, "setting": setting,
                         "position": position, "arm": arm})
    return rows


def campaign_evidence(root, add, omit, inspect):
    warnings = []
    if not root.is_dir():
        warnings.append("Campaign directory is absent; no campaign execution is established")
    plan = read_json(root / "plan.json", warnings)
    completed = read_json(root / "completed.json", warnings)
    results = read_json(root / "results.json", warnings)
    freeze = read_json(root / "freeze-verification.json", warnings)
    schedule = plan.get("schedule", []) if isinstance(plan, dict) else []
    if not isinstance(schedule, list):
        schedule = []
    build_manifest = plan.get("manifest", {}) if isinstance(plan, dict) else {}
    primary = schedule[0].get("setting") if schedule and isinstance(schedule[0], dict) else None
    protocol = (primary in {"covariant", "dsp10"}
                and schedule == expected_schedule(primary)
                and plan.get("count") == 8
                and isinstance(build_manifest, dict) and build_manifest.get("mode") == "native")
    if not protocol:
        warnings.append("The recorded plan does not establish the fixed native 20-run/eight-image protocol")
    records, payloads = {}, []
    for path in files(root):
        relative = path.relative_to(root).as_posix()
        if database(path):
            omit(path, "SQLite database or sidecar; retained in session")
        elif path.suffix == ".witness":
            payloads.append(path)
        elif path.suffix in METADATA:
            add(path, "campaign/" + relative)
            if path.name == "run.json":
                record = read_json(path, warnings)
                if isinstance(record, dict):
                    records[path.parent] = record
        else:
            omit(path, "Not campaign metadata, a log, or a selected witness")

    candidates, canonical = {}, {}
    for path in payloads:
        record = records.get(path.parent, {})
        actual = inspect(path)
        verified = (record.get("returncode") == 0
                    and record.get("witness_bytes") == actual["bytes"]
                    and record.get("witness_sha256") == actual["sha256"])
        if not verified:
            warnings.append(f"Witness lacks a matching successful receipt: {path}")
        setting = record.get("setting", "unknown")
        if setting not in ("ordinary", "covariant", "dsp10"):
            setting = "unknown"
            warnings.append(f"Witness setting is unknown: {path}")
        candidates[path] = (record, actual, verified, setting)
        if verified and record.get("arm") == "A" and setting not in canonical:
            canonical[setting] = {"arm": "A", "source": str(path),
                                  "member": "campaign/" + path.relative_to(root).as_posix(),
                                  **actual}
    seen = {(key, value["sha256"]): value["member"] for key, value in canonical.items()}
    for path, (record, actual, verified, setting) in candidates.items():
        member = "campaign/" + path.relative_to(root).as_posix()
        anchor = canonical.get(setting)
        if anchor and str(path) == anchor["source"]:
            add(path, member)
        elif verified and (setting, actual["sha256"]) in seen:
            omit(path, "Duplicate witness", actual, duplicate_of=seen[(setting, actual["sha256"])])
        else:
            # Preserve failures and unanchored partial output instead of hiding a mismatch.
            add(path, member)
            seen[(setting, actual["sha256"])] = member
            if anchor and actual["sha256"] != anchor["sha256"]:
                warnings.append(f"Divergent witness retained: {path}")
    settings = sorted({r.get("setting") for r in records.values()
                       if isinstance(r.get("setting"), str)})
    for setting in settings:
        if setting not in canonical:
            warnings.append(f"No verified A canonical for setting {setting}")
    for directory, record in records.items():
        if record.get("returncode") == 0 and not (directory / "output.witness").is_file():
            warnings.append(f"Successful receipt has no witness: {directory}")
        if not (directory / "process.log").is_file():
            warnings.append(f"Run log missing: {directory}")

    timing_rows, matching = [], True
    by_id = {directory.name: record for directory, record in records.items()}
    for index, item in enumerate(schedule):
        if not isinstance(item, dict):
            matching = False
            continue
        run_id = f"{index:02d}-{item.get('cohort')}-{item.get('arm')}"
        record = by_id.get(run_id, {})
        metrics = record.get("metrics", {})
        times = {key: metrics.get(key) for key in TIMES} if isinstance(metrics, dict) else {}
        valid_times = (len(times) == len(TIMES)
                       and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and math.isfinite(v) and v >= 0 for v in times.values()))
        matching &= (record.get("id") == run_id and record.get("returncode") == 0
                     and all(record.get(k) == v for k, v in item.items()) and valid_times)
        timing_rows.append({"id": run_id, **item, "receipt_present": bool(record),
                            "returncode": record.get("returncode"),
                            "times": times if valid_times else None})
    declared = (isinstance(completed, dict) and completed.get("completed_runs") == 20
                and completed.get("planned_runs") == 20)
    declared_canonical = completed.get("canonical", {}) if isinstance(completed, dict) else {}
    anchored = (isinstance(declared_canonical, dict) and all(
        isinstance(declared_canonical.get(setting), dict)
        and declared_canonical[setting].get("arm") == "A"
        and declared_canonical[setting].get("sha256") == row["sha256"]
        for setting, row in canonical.items()) and len(canonical) == 3)
    ordered_records = [by_id.get(row["id"]) for row in timing_rows]
    passed = (protocol and declared and matching and anchored and len(records) == 20
              and isinstance(results, list) and len(results) == 20 and results == ordered_records
              and isinstance(freeze, dict) and freeze.get("all_match") is True and not warnings)
    status = "complete" if passed else ("inconsistent" if completed is not None else "partial")
    return {"source": str(root), "source_exists": root.is_dir(),
            "status": status, "protocol_verified": protocol,
            "completed_json_present": (root / "completed.json").is_file(),
            "recorded_runs": len(records), "planned_runs": len(schedule),
            "timing_rows": timing_rows, "canonical_witnesses": canonical,
            "warnings": warnings}


def build_evidence(root, prefix, add, omit, inspect):
    warnings, raw, canonical = [], [], None
    raw_metadata = read_json(root / "raw-oracles.json", warnings)
    for path in files(root):
        member = prefix + "/" + path.relative_to(root).as_posix()
        if database(path):
            omit(path, "SQLite database or sidecar; retained in session")
        elif path.suffix in METADATA:
            add(path, member)
        elif path.suffix == ".bin" and "-raw-" in path.name:
            raw.append(path)
        elif path.suffix in {".o", ".a", ".so", ".pyc", ".witness"}:
            omit(path, "Object/support archive or unselected witness; identities retained")
        elif path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            if path.stat().st_size <= SMALL_BINARY_BYTES:
                add(path, member)
            else:
                omit(path, "Executable exceeds 64 MiB transport cap", inspect(path))
        else:
            omit(path, "Not build/test/oracle metadata, map, or small executable")
    ordered = sorted(raw, key=lambda p: (p.name != "A-raw-ordinary.bin",
                                        not p.name.startswith("A-"), p.name))
    seen = {}
    for path in ordered:
        actual = inspect(path)
        member = prefix + "/" + path.relative_to(root).as_posix()
        expected = raw_metadata.get(path.stem) if isinstance(raw_metadata, dict) else None
        verified = (isinstance(expected, dict) and actual["sha256"] == expected.get("sha256")
                    and actual["bytes"] == expected.get("bytes"))
        if not verified:
            warnings.append(f"Raw oracle lacks matching metadata: {path}")
        if actual["sha256"] in seen:
            omit(path, "Duplicate raw-float oracle", actual, duplicate_of=seen[actual["sha256"]])
        else:
            add(path, member)
            seen[actual["sha256"]] = member
            if path.name.startswith("A-") and verified and canonical is None:
                canonical = {"arm": "A", "source": str(path), "member": member, **actual}
    if len(seen) > 1:
        warnings.append("Raw-float oracle outputs disagree; all distinct payloads retained")
    if raw and canonical is None:
        warnings.append("No verified A raw-float canonical is available")
    return {"source": str(root), "raw_A_canonical": canonical, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--build", type=Path, action="append", default=[])
    parser.add_argument("--extra", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign, builds = args.campaign.resolve(), [p.resolve() for p in args.build]
    extras, output = [p.resolve() for p in args.extra], args.output.absolute()
    receipt = output.with_name(output.name + ".receipt.json")
    if campaign.exists() and not campaign.is_dir():
        raise ValueError(f"Campaign input is not a directory: {campaign}")
    for root in builds:
        if not root.is_dir():
            raise ValueError(f"Input directory missing: {root}")
    for root in [campaign, *builds]:
        if output.resolve().is_relative_to(root) or receipt.resolve().is_relative_to(root):
            raise ValueError("Output must be outside input evidence directories")
    if len(set(builds)) != len(builds):
        raise ValueError("Duplicate --build directory")
    if output.exists() or output.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise FileExistsError("Archive or receipt already exists; choose a new output")
    selected, omitted, identities = {}, [], {}

    def inspect(path):
        if path not in identities:
            identities[path] = identity(path)
        return identities[path]

    def add(path, member):
        if member in selected:
            raise ValueError(f"Duplicate archive member: {member}")
        selected[member] = {"source": str(path), **inspect(path)}

    def omit(path, reason, info=None, **extra):
        omitted.append({"source": str(path), "reason": reason, **(info or {}), **extra})

    campaign_summary = campaign_evidence(campaign, add, omit, inspect)
    build_summaries = [build_evidence(root, f"builds/{index:02d}-{root.name}", add, omit, inspect)
                       for index, root in enumerate(builds)]
    for index, path in enumerate(extras):
        if not path.is_file():
            raise ValueError(f"Extra must be an existing file: {path}")
        if database(path):
            raise ValueError("SQLite databases are excluded from transport")
        add(path, f"extras/{index:02d}-{path.name}")
    manifest = {"format": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "campaign": campaign_summary, "builds": build_summaries,
                "payloads": selected, "omitted": omitted,
                "policy": "Inputs retained unchanged. 64 MiB executable cap; one verified A canonical per setting/build, duplicate witnesses omitted, divergent/unanchored outputs retained honestly."}
    manifest_data = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    expected = dict(selected)
    expected["package-manifest.json"] = {"bytes": len(manifest_data),
        "sha256": hashlib.sha256(manifest_data).hexdigest(),
        "crc32": f"{zlib.crc32(manifest_data):08x}"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, strict_timestamps=False) as archive:
        for member, row in selected.items():
            archive.write(row["source"], member)
        archive.writestr("package-manifest.json", manifest_data)
    with zipfile.ZipFile(output) as archive:
        if len(archive.infolist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError("Archive member list differs from manifest")
        for member, row in expected.items():
            sha, crc, size = hashlib.sha256(), 0, 0
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    sha.update(block)
                    crc = zlib.crc32(block, crc)
                    size += len(block)
            if (sha.hexdigest() != row["sha256"] or size != row["bytes"]
                    or f"{crc:08x}" != row["crc32"] or archive.getinfo(member).CRC != crc):
                raise ValueError(f"Archive payload changed or failed verification: {member}")
    archive_identity = identity(output)
    result = {"archive": str(output.resolve()), "receipt": str(receipt.resolve()),
              "archive_bytes": archive_identity["bytes"], "archive_sha256": archive_identity["sha256"],
              "payload_count": len(expected), "all_payload_sha256_and_crc_verified": True,
              "campaign_status": campaign_summary["status"], "payloads": expected}
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "payloads"}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Packaging failed: {error}. Evidence is unchanged; use a fresh output path for another attempt.", file=sys.stderr)
        sys.exit(1)
