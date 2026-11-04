
import csv
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "outputs"
EXPORT_DIR = PROJECT_ROOT / "evaluations" / "qwen3-persona-v1"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

records = []

for event_file in LOG_DIR.rglob("events.out.tfevents*"):
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()

    tags = accumulator.Tags()

    for tag in tags.get("scalars", []):
        for event in accumulator.Scalars(tag):
            records.append({
                "run": str(event_file.parent.relative_to(LOG_DIR)),
                "metric": tag,
                "step": event.step,
                "value": event.value,
                "wall_time": event.wall_time,
            })

    # Some newer logging formats store scalar metrics as tensors.
    for tag in tags.get("tensors", []):
        from tensorboard.util import tensor_util

        for event in accumulator.Tensors(tag):
            value = tensor_util.make_ndarray(event.tensor_proto)

            if value.size != 1 or value.dtype.kind not in "fiub":
                continue

            records.append({
                "run": str(event_file.parent.relative_to(LOG_DIR)),
                "metric": tag,
                "step": event.step,
                "value": float(value.item()),
                "wall_time": event.wall_time,
            })

records.sort(key=lambda r: (r["run"], r["metric"], r["step"]))

json_path = EXPORT_DIR / "tensorboard_metrics.json"
csv_path = EXPORT_DIR / "tensorboard_metrics.csv"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["run", "metric", "step", "value", "wall_time"],
    )
    writer.writeheader()
    writer.writerows(records)

print(f"Exported {len(records)} metric points")
print(f"Metrics: {sorted(set(r['metric'] for r in records))}")
print(f"Saved to {EXPORT_DIR}")
