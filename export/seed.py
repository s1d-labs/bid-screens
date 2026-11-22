#!/usr/bin/env python3
"""Seed a project's queue from push.py output (only verified shards), the way box-seeded families do (doc-ocr, gem-pdf-parse):
one item per shard naming its bucket keys and the pinned code hash. Existing keys are kept (ON CONFLICT DO NOTHING).
  seed.py PROJECT PUSHED.jsonl [PUSHED.jsonl ...]        (runs psql as postgres on scrape_hub)
"""
import io
import json
import subprocess
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_shard import code_sha  # noqa: E402


def main():
    project, files = sys.argv[1], sys.argv[2:]
    import os
    sha, rows, directors = os.environ.get("SEED_CODE") or code_sha(), [], None   # SEED_CODE: a hash both old and new checkouts accept (A1)
    for f in files:
        for l in open(f):
            x = json.loads(l)
            if "directors" in x:
                directors = x["directors"]
    for f in files:
        for l in open(f):
            x = json.loads(l)
            if "shard" not in x or not x["verified"]:
                continue
            key = x["shard"].replace("/", "_")
            payload = {"shard": x["shard"], "source": x["source"], "files": x["keys"], "directors": directors if x["source"] == "gem" else None,
                       "code": sha, "n_tenders": x.get("n_tenders"), "n_bids": x.get("n_bids"), "expected": 1}
            rows.append(f"{project}\t{key}\t{zlib.crc32(key.encode()) % 1_000_000}\t{json.dumps(payload).replace(chr(92), chr(92) * 2)}\n")
    sql = ("\\set ON_ERROR_STOP on\nbegin;\ncreate temp table q (project text, key text, partition_id int, payload jsonb);\n"
           "copy q from stdin;\n" + "".join(rows) + "\\.\n"
           "insert into queue (project, key, partition_id, payload) select project, key, partition_id, payload from q "
           "on conflict (project, key) do nothing;\ncommit;\n")
    out = subprocess.run(["sudo", "-n", "-u", "postgres", "psql", "-q", "-X", "-d", "scrape_hub"], input=sql.encode(),
                         capture_output=True, check=True)
    print(f"{project}: {len(rows)} items offered, code {sha[:12]}", out.stdout.decode().strip(), out.stderr.decode().strip())


if __name__ == "__main__":
    main()
