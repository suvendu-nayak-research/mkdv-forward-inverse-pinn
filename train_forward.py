"""Forward PINN for the modified KdV equation."""

import os
os.environ["DDE_BACKEND"] = "pytorch"

from pathlib import Path
import numpy as np
import torch
import torch._dynamo
import deepxde as dde

# Training settings
SEED = 1234
ADAM_STEPS = 3000
RESULTS = Path(__file__).resolve().parent / "results"

torch.set_default_device("cpu")
torch.set_num_threads(1)
dde.config.set_default_float("float64")
dde.config.set_random_seed(SEED)
saved = np.load(RESULTS / "data.npz")
q, r = -6.0, 1.0

# PDE residual. Input columns are [x, t].

def pde(X, u):
    u_x = dde.grad.jacobian(u, X, i=0, j=0)
    u_t = dde.grad.jacobian(u, X, i=0, j=1)
    u_xx = dde.grad.hessian(u, X, component=0, i=0, j=0)
    u_xxx = dde.grad.jacobian(u_xx, X, i=0, j=0)
    return u_t + q * u**2 * u_x + r * u_xxx


def right_derivative(X, u, _):
    return dde.grad.jacobian(u, X, i=0, j=0)


# Domain and prescribed conditions

domain = dde.geometry.GeometryXTime(
    dde.geometry.Interval(0, 1), dde.geometry.TimeDomain(0, 0.5)
)
ic = dde.icbc.PointSetBC(saved["X_ic"], saved["u_ic"])
bc = dde.icbc.PointSetBC(saved["X_bc"], saved["u_bc"])
bc_x = dde.icbc.PointSetOperatorBC(
    saved["X_right"], saved["ux_right"], right_derivative
)
data = dde.data.TimePDE(
    domain, pde, [ic, bc, bc_x], num_domain=512,
    num_boundary=0, num_initial=0, train_distribution="pseudo",
)

# Network and DeepXDE training.
net = dde.nn.FNN([2, 32, 32, 1], "tanh", "Glorot normal")
model = dde.Model(data, net)
weights = [1, 10, 10, 10]  # PDE, initial values, boundary values, boundary slope.
model.compile("adam", lr=0.001, loss_weights=weights)
model.train(iterations=ADAM_STEPS, display_every=1000)
dde.optimizers.config.set_LBFGS_options(maxiter=1000, maxcor=20)
model.compile("L-BFGS", loss_weights=weights)
history, _ = model.train()

# Save predictions and training history
X_test = saved["X_test"]
u_pred = model.predict(X_test)
residual = model.predict(X_test, operator=pde)
if not np.isfinite(u_pred).all() or not np.isfinite(residual).all():
    raise RuntimeError("Training produced non-finite values; inspect the loss.")
np.savez(
    RESULTS / "forward.npz", X=X_test, u=u_pred, residual=residual,
    steps=history.steps, loss=history.loss_train, seed=SEED,
)
print("Saved results/forward.npz")
