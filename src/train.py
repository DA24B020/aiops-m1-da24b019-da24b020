import argparse
import os
import random
import subprocess
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SKOPS_TRUSTED = [
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
    "sklearn.neural_network._stochastic_optimizers.SGDOptimizer",
    "numpy.dtype",
]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def git_dirty():
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL).decode().strip()
        return "true" if out else "false"
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/iris.csv")
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment", default="m1-capstone")
    p.add_argument("--run_name", default=None)
    p.add_argument("--register_as", default=None)
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    os.environ["PYTHONHASHSEED"] = str(a.seed)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(a.experiment)

    df = pd.read_csv(a.data)
    y = df["label"]
    X = df.drop(columns=["label"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=a.seed, stratify=y)

    with mlflow.start_run(run_name=a.run_name or f"run-seed{a.seed}") as run:
        mlflow.log_params({
            "model": "MLPClassifier",
            "hidden_size": a.hidden_size,
            "lr": a.lr,
            "epochs": a.epochs,
            "seed": a.seed,
            "data_file": a.data,
            "n_train": len(X_tr),
            "n_test": len(X_te),
        })
        mlflow.set_tags({
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "dataset": "iris.csv (DVC-tracked)",
        })

        clf = MLPClassifier(
            hidden_layer_sizes=(a.hidden_size,),
            learning_rate_init=a.lr,
            max_iter=a.epochs,
            solver="adam",
            activation="relu",
            random_state=a.seed,
        )
        clf.fit(X_tr, y_tr)

        train_acc = float(clf.score(X_tr, y_tr))
        val_acc = float(clf.score(X_te, y_te))
        mlflow.log_metrics({
            "final_train_accuracy": train_acc,
            "final_val_accuracy": val_acc,
            "generalization_gap": train_acc - val_acc,
            "final_train_loss": float(clf.loss_),
        })

        sample = X_tr.iloc[:5]
        sig = infer_signature(sample, clf.predict(sample))

        kw = dict(
            sk_model=clf,
            name="model",
            signature=sig,
            input_example=sample,
            skops_trusted_types=SKOPS_TRUSTED,
        )
        if a.register_as:
            kw["registered_model_name"] = a.register_as

        try:
            mlflow.sklearn.log_model(**kw)
        except Exception:
            kw.pop("skops_trusted_types", None)
            kw["serialization_format"] = "cloudpickle"
            mlflow.sklearn.log_model(**kw)

        print(f"run_id             = {run.info.run_id}")
        print(f"git_commit         = {git_commit()}  dirty={git_dirty()}")
        print(f"final_val_accuracy = {val_acc:.4f}")


if __name__ == "__main__":
    main()