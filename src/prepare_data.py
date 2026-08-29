import argparse
from sklearn.datasets import load_iris

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/iris.csv")
a = p.parse_args()

X, y = load_iris(return_X_y=True, as_frame=True)
df = X.copy()
df["label"] = y
df.to_csv(a.out, index=False)
print(f"wrote {a.out}: {df.shape[0]} rows x {df.shape[1]} cols")