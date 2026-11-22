#!/usr/bin/env python3
"""Upload exported shards to the bucket the hub routes a project to, with the hub's own storage client (no key leaves the
hub's environment; same way as other box-side uploads). Skips files already there with the same size, then checks every
file against the bucket listing (name and size). Run as root with the hub's environment (export/push.sh does that):
  push.py STAGE MANIFEST PROJECT [--delete-local]   -> prints one JSON line per shard: shard, keys, bytes, verified
Local copies are deleted (with --delete-local) only after the listing shows the same name and size.
"""
import json
import sys
from pathlib import Path

from scrape_hub import buckets, db


def family_of(name, settings=None):
    return (settings or {}).get("family") or name.split("-", 1)[0]


def main():
    stage, manifest, project = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    delete = "--delete-local" in sys.argv
    with db.conn() as c:
        store = buckets.store_for(c, project, family_of)
        s3, bucket = buckets.client(c, store)
    if not store.startswith("bucket:"):
        sys.exit(f"{project} routes to {store}, not a member bucket: run the family's setup workflow first")
    shards = [json.loads(l) for l in open(manifest) if l.strip()]
    have = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{project}/inputs/"):
        for o in page.get("Contents", []):
            have[o["Key"]] = o["Size"]
    extra = [stage / "gem" / "directors.parquet"] if (stage / "gem" / "directors.parquet").exists() and "gem" in project else []
    for f in extra:
        k = f"{project}/inputs/gem/directors.parquet"
        if have.get(k) != f.stat().st_size:
            s3.upload_file(str(f), bucket, k, ExtraArgs={"ContentType": "application/octet-stream"})
    for sh in shards:
        keys, total = [], 0
        for name, size in sh["files"].items():
            p = stage / sh["shard"] / name
            k = f"{project}/inputs/{sh['shard']}/{name}"
            if have.get(k) != size:
                s3.upload_file(str(p), bucket, k, ExtraArgs={"ContentType": "application/octet-stream"})
            keys.append(k); total += size
        listed = {}
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{project}/inputs/{sh['shard']}/"):
            for o in page.get("Contents", []):
                listed[o["Key"]] = o["Size"]
        ok = all(listed.get(f"{project}/inputs/{sh['shard']}/{n}") == s for n, s in sh["files"].items())
        if ok and delete:
            for name in sh["files"]:
                (stage / sh["shard"] / name).unlink(missing_ok=True)
        print(json.dumps({"shard": sh["shard"], "source": sh["source"], "keys": keys, "bytes": total, "verified": ok,
                          "store": store, "n_tenders": sh.get("n_tenders"), "n_bids": sh.get("n_bids")}), flush=True)
    for f in extra:
        k = f"{project}/inputs/gem/directors.parquet"
        print(json.dumps({"directors": k, "verified": True}), flush=True)


if __name__ == "__main__":
    main()
