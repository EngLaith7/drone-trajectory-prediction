
import joblib
import json
import numpy as np
import gzip
import os
from copy import deepcopy

MODEL_PKL = "src/models/drone_model_1.pkl"
SCALER_X_PKL = "src/models/scaler_X_1.pkl"
SCALER_Y_PKL = "src/models/scaler_y_1.pkl"
OUT_JSON = "rf_model.json"          
OUT_JSON_GZ = "rf_model.json.gz"    
# 

model = joblib.load(MODEL_PKL)
scaler_X = joblib.load(SCALER_X_PKL)
scaler_y = joblib.load(SCALER_Y_PKL)

n_features = int(scaler_X.mean_.shape[0])
n_outputs = int(model.n_outputs_)

print(f"Model n_estimators: {len(model.estimators_)}, n_features: {n_features}, n_outputs: {n_outputs}")

def tree_to_dict(tree):
    
    # children arrays as lists of ints
    children_left = tree.children_left.tolist()
    children_right = tree.children_right.tolist()
    feature = tree.feature.tolist()
    # threshold as float32
    threshold = [float(np.float32(x)) for x in tree.threshold.tolist()]
    # value: shape (n_nodes, 1, n_outputs) -> (n_nodes, n_outputs)
    raw_value = tree.value  # numpy array
    value_reduced = []
    for node_idx in range(tree.node_count):
        # flatten the [1,n_outputs] -> list
        vals = [float(np.float32(v)) for v in raw_value[node_idx][0].tolist()]
        value_reduced.append(vals)
    return {
        "n_nodes": int(tree.node_count),
        "children_left": children_left,
        "children_right": children_right,
        "feature": feature,
        "threshold": threshold,
        "value": value_reduced
    }

forest = []
for i, est in enumerate(model.estimators_):
    t = est.tree_
    forest.append(tree_to_dict(t))
    if (i+1) % 50 == 0:
        print(f" processed {i+1} trees...")

payload = {
    "n_estimators": len(model.estimators_),
    "n_outputs": n_outputs,
    "forest": forest,
    "scaler_X": {
        "mean": [float(np.float32(x)) for x in scaler_X.mean_.tolist()],
        "scale": [float(np.float32(x)) for x in scaler_X.scale_.tolist()]
    },
    "scaler_y": {
        "mean": [float(np.float32(x)) for x in scaler_y.mean_.tolist()],
        "scale": [float(np.float32(x)) for x in scaler_y.scale_.tolist()]
    }
}

with open(OUT_JSON, "w") as f:
    json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)

with open(OUT_JSON, "rb") as f_in, gzip.open(OUT_JSON_GZ, "wb") as f_out:
    f_out.writelines(f_in)

print("Exported JSON to", OUT_JSON, "and compressed gz to", OUT_JSON_GZ)
print("JSON file size (bytes):", os.path.getsize(OUT_JSON), "  gz size:", os.path.getsize(OUT_JSON_GZ))

def predict_from_json_single(payload, x_flat):
    # x_flat: 1D numpy array length = n_features
    # standardize with scaler_X
    mean = np.array(payload["scaler_X"]["mean"], dtype=np.float32)
    scale = np.array(payload["scaler_X"]["scale"], dtype=np.float32)
    x_std = (x_flat - mean) / scale

    n_out = payload["n_outputs"]
    out_sum = np.zeros(n_out, dtype=np.float64)

    for tree in payload["forest"]:
        node = 0
        while True:
            left = tree["children_left"][node]
            right = tree["children_right"][node]
            if left == -1 and right == -1:
                vals = np.array(tree["value"][node], dtype=np.float64)  # length n_out
                out_sum += vals
                break
            else:
                feat = tree["feature"][node]
                thr = tree["threshold"][node]
                if x_std[feat] <= thr:
                    node = left
                else:
                    node = right

    avg = out_sum / payload["n_estimators"]
    # inverse transform
    mean_y = np.array(payload["scaler_y"]["mean"], dtype=np.float64)
    scale_y = np.array(payload["scaler_y"]["scale"], dtype=np.float64)
    y = avg * scale_y + mean_y
    return y

with open(OUT_JSON, "r") as f:
    loaded = json.load(f)

rng = np.random.RandomState(0)
x_test = rng.randn(n_features).astype(np.float32)

y_model = model.predict(x_test.reshape(1, -1))[0]
y_json = predict_from_json_single(loaded, x_test)

print("compare predictions (original_model vs json-impl):")
print("orig:", y_model)
print("json:", y_json)

diff = np.abs(y_model - y_json)
print("abs diff per output:", diff)
tol = 1e-5
if np.all(diff < 1e-3):
    print("Validation OK: differences are small (numerical rounding expected).")
else:
    print("Warning: differences > tolerance; check dtype/ordering. Max diff:", diff.max())
