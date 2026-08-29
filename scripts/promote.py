import argparse
import os

import mlflow
from mlflow import MlflowClient

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--version", type=int, required=True)
p.add_argument("--stage", default="Staging",
               choices=["None", "Staging", "Production", "Archived"])
p.add_argument("--archive", action="store_true")
a = p.parse_args()

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
c = MlflowClient()
c.transition_model_version_stage(
    name=a.name, version=a.version, stage=a.stage,
    archive_existing_versions=a.archive)

print(f"{a.name} v{a.version} -> {a.stage}\n")
for v in c.search_model_versions(f"name='{a.name}'"):
    print(f"  v{v.version}  stage={v.current_stage}  run_id={v.run_id[:8]}")