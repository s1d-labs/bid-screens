"""Put this family's projects in the pool. Run by the Setup workflow. Safe to run again. The queues come from the box that
writes the input shards (their items name bucket keys and the pinned code hash), as in doc-ocr."""
from scrape_hub_client import register_project, set_pool

from family import FAMILY, PROJECTS

for p in PROJECTS:
    register_project(p["name"], title=p["title"], unit_label=p["unit_label"], record_label=p["record_label"])
    set_pool(p["name"], FAMILY, p["task"], target=f"{p['name']}.compute.local", geo="", max_workers=p["max_workers"], threads=1, delay_ms=0)
    print(p["name"], "ready")
