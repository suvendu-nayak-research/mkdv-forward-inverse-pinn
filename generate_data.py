"""Generate initial, boundary, observation, and test data for mKdV."""

from pathlib import Path
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

# Exact solution for q = -6 and r = 1.
def exact_u(X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    return np.tanh(x + 2 * t)


# Initial and boundary data.
x = np.linspace(0, 1, 64)
t = np.linspace(0, 0.5, 64)
X_ic = np.column_stack((x, np.zeros_like(x)))
X_left = np.column_stack((np.zeros_like(t), t))
X_right = np.column_stack((np.ones_like(t), t))
X_bc = np.vstack((X_left, X_right))
ux_right = 1 / np.cosh(1 + 2 * t[:, None])**2

# Interior observations: 16 sensors, each measured at 4 times.
sensors = np.linspace(0.05, 0.95, 16)
times = np.array([0.05, 0.15, 0.30, 0.45])
xx, tt = np.meshgrid(sensors, times)
X_obs = np.column_stack((xx.ravel(), tt.ravel()))

#  Evaluation grid
axis = (np.arange(51) + 0.5) / 51
xx, tt = np.meshgrid(axis, 0.5 * axis)
X_test = np.column_stack((xx.ravel(), tt.ravel()))

np.savez(
    RESULTS / "data.npz",
    X_ic=X_ic, u_ic=exact_u(X_ic),
    X_bc=X_bc, u_bc=exact_u(X_bc),
    X_right=X_right, ux_right=ux_right,
    X_obs=X_obs, u_obs=exact_u(X_obs),
    X_test=X_test, u_test=exact_u(X_test),
)
print(f"Saved fixed IC/BC data and {len(X_obs)} noise-free observations.")
