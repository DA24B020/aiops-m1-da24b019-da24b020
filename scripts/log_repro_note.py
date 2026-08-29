import argparse
import os

import mlflow
from mlflow import MlflowClient

p = argparse.ArgumentParser()
p.add_argument("--run_a", required=True)
p.add_argument("--run_b", required=True)
p.add_argument("--metric", default="final_val_accuracy")
p.add_argument("--tolerance", type=float, default=0.005)
a = p.parse_args()

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
c = MlflowClient()

ra, rb = c.get_run(a.run_a), c.get_run(a.run_b)
va, vb = ra.data.metrics[a.metric], rb.data.metrics[a.metric]
delta = abs(va - vb)
matched = delta <= a.tolerance

note = (f"Reproduction of run {a.run_a[:8]} (git_commit "
        f"{ra.data.tags.get('git_commit', 'unknown')}). "
        f"Partner A {a.metric}={va:.4f}; Partner B {a.metric}={vb:.4f}; "
        f"|delta|={delta:.4f} vs pre-declared tolerance +/-{a.tolerance}. "
        f"Verdict: {'MATCH' if matched else 'MISMATCH'}.")

c.set_tag(a.run_b, "repro_note", note)
c.set_tag(a.run_b, "repro_source_run", a.run_a)
c.set_tag(a.run_b, "repro_matched", str(matched).lower())
c.log_metric(a.run_b, "repro_delta", delta)

print(note)
if not matched:
    print(f"\nA git_dirty={ra.data.tags.get('git_dirty')}  "
          f"seeds A/B={ra.data.params.get('seed')}/{rb.data.params.get('seed')}  "
          f"n_train A/B={ra.data.params.get('n_train')}/{rb.data.params.get('n_train')}")