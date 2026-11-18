#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fetch only the eight pinned original JPEGs from the official release ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zlib


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(__file__).parent / "photos.json").read_text())
    receipts = []
    for item in manifest["members"]:
        target = args.out / Path(item["member"]).name
        if target.exists():
            content = target.read_bytes()
            if len(content) != item["bytes"] or sha(content) != item["sha256"]:
                raise ValueError(f"Existing photo has a different identity: {target}")
            receipts.append({"photo": target.name, "source": "verified existing file", "sha256": sha(content)})
            continue
        segment = item["deflate_range"]
        start, size = segment["offset"], segment["size"]
        request = urllib.request.Request(manifest["source"], headers={
            "Range": f"bytes={start}-{start + size - 1}",
            "Accept-Encoding": "identity", "User-Agent": "COLMAP-CPU-qualification"})
        with urllib.request.urlopen(request, timeout=120) as response:
            expected_range = f"bytes {start}-{start + size - 1}/{manifest['archive_bytes']}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise RuntimeError("Server did not honor the pinned range; use the full official ZIP and verify photos.json instead")
            compressed = response.read(size + 1)
        if len(compressed) != size or sha(compressed) != segment["sha256"]:
            raise ValueError("Compressed source-range identity mismatch")
        content = zlib.decompress(compressed, -15)
        if len(content) != item["bytes"] or sha(content) != item["sha256"]:
            raise ValueError("Decompressed photo identity mismatch")
        if f"{zlib.crc32(content):08x}" != item["zip_crc32_verified"]:
            raise ValueError("ZIP member CRC mismatch")
        with target.open("xb") as stream:
            stream.write(content)
        receipts.append({"photo": target.name, "source": manifest["source"],
                         "range": segment, "sha256": sha(content)})
        print(target.name, item["sha256"], flush=True)
    receipt = args.out / "photo-verification.json"
    with receipt.open("x") as stream:
        json.dump({"photos": receipts, "all_match": True}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
