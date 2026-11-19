"""Measure errors and draw figures from the two saved training results."""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
data = np.load(RESULTS / "data.npz")
forward = np.load(RESULTS / "forward.npz")
inverse = np.load(RESULTS / "inverse.npz")
truth = data["u_test"]

# Solution errors and PDE residuals

metrics = {}
for name, result in [("forward", forward), ("inverse", inverse)]:
    if not np.array_equal(result["X"], data["X_test"]):
        raise ValueError("Data changed: rerun both training scripts.")
    error = result["u"] - truth
    metrics[name] = {
        "u_relative_l2": float(np.linalg.norm(error) / np.linalg.norm(truth)),
        "u_max_abs": float(np.max(np.abs(error))),
        "pde_rms": float(np.sqrt(np.mean(result["residual"]**2))),
    }

# Coefficient errors using the exact benchmark values

for key, true_value in [("q", -6.0), ("r", 1.0)]:
    found = float(inverse[key])
    metrics["inverse"][key] = found
    metrics["inverse"][key + "_error_percent"] = 100 * abs(found - true_value) / abs(true_value)
text = json.dumps(metrics, indent=2, allow_nan=False)
(RESULTS / "metrics.json").write_text(text + "\n", encoding="utf-8")
print(text)

# Exact solution and absolute errors.
x = np.unique(data["X_test"][:, 0])
t = np.unique(data["X_test"][:, 1])
order = np.lexsort((data["X_test"][:, 0], data["X_test"][:, 1]))

fields = [truth, abs(forward["u"] - truth), abs(inverse["u"] - truth)]
titles = ["Exact solution", "Forward absolute error", "Inverse absolute error"]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
for ax, field, title in zip(axes, fields, titles):
    values = field.ravel()[order].reshape(len(t), len(x))
    plot = ax.pcolormesh(x, t, values, shading="nearest")
    fig.colorbar(plot, ax=ax)
    ax.set(xlabel="x", ylabel="t", title=title)

fig.savefig(RESULTS / "solution.png", dpi=160)
plt.close(fig)

# Weighted training loss for each problem.
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for name, result in [("forward", forward), ("inverse", inverse)]:
    ax.semilogy(result["steps"], result["loss"].sum(axis=1), label=name)
ax.set(xlabel="Training step", ylabel="Weighted training loss")
ax.legend()
fig.savefig(RESULTS / "loss.png", dpi=160)
plt.close(fig)
