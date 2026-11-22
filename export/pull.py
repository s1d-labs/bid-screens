#!/usr/bin/env python3
"""Download a prefix of a project's bucket files with the hub's storage client (run like push.py, via export/pull.sh):
  pull.py PROJECT PREFIX DEST   e.g. pull.py bid-screens-rank national/run1/ /path/out"""
import sys
from pathlib import Path

from scrape_hub import buckets, db


def main():
    project, prefix, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    with db.conn() as c:
        s3, bucket = buckets.client(c, buckets.store_for(c, project, lambda n, s=None: (s or {}).get("family") or n.split("-", 1)[0]))
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{project}/{prefix}"):
        for o in page.get("Contents", []):
            out = dest / o["Key"][len(project) + 1 + len(prefix):]
            out.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, o["Key"], str(out)); n += 1
    print(f"{n} files to {dest}")


if __name__ == "__main__":
    main()
