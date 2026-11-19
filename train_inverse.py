"""Inverse PINN for the modified KdV equation."""

import os
os.environ["DDE_BACKEND"] = "pytorch"

from pathlib import Path
import numpy as np
import torch
import torch._dynamo
import deepxde as dde

# Settings and initial coefficient guesses.

SEED = 1235
ADAM_STEPS = 5000
RESULTS = Path(__file__).resolve().parent / "results"
torch.set_default_device("cpu")
torch.set_num_threads(1)
dde.config.set_default_float("float64")
dde.config.set_random_seed(SEED)
saved = np.load(RESULTS / "data.npz")
q = dde.Variable(-2.0)
r = dde.Variable(0.5)

# PDE residual with trainable coefficients

def pde(X, u):
    u_x = dde.grad.jacobian(u, X, i=0, j=0)
    u_t = dde.grad.jacobian(u, X, i=0, j=1)
    u_xx = dde.grad.hessian(u, X, component=0, i=0, j=0)
    u_xxx = dde.grad.jacobian(u_xx, X, i=0, j=0)
    return u_t + q * u**2 * u_x + r * u_xxx


def right_derivative(X, u, _):
    return dde.grad.jacobian(u, X, i=0, j=0)


# IC/BC targets and 64 observed interior values of u.

domain = dde.geometry.GeometryXTime(
    dde.geometry.Interval(0, 1), dde.geometry.TimeDomain(0, 0.5)
)
ic = dde.icbc.PointSetBC(saved["X_ic"], saved["u_ic"])
bc = dde.icbc.PointSetBC(saved["X_bc"], saved["u_bc"])
bc_x = dde.icbc.PointSetOperatorBC(
    saved["X_right"], saved["ux_right"], right_derivative
)
observations = dde.icbc.PointSetBC(saved["X_obs"], saved["u_obs"])
data = dde.data.TimePDE(
    domain, pde, [ic, bc, bc_x, observations], num_domain=512,
    num_boundary=0, num_initial=0, train_distribution="pseudo",
    anchors=saved["X_obs"],
)

# Network and joint optimization of u, q, and r

net = dde.nn.FNN([2, 32, 32, 1], "tanh", "Glorot normal")
model = dde.Model(data, net)
weights = [1, 10, 10, 10, 100]  # PDE, IC, BC, right slope, observations.
model.compile("adam", lr=0.001, loss_weights=weights,
              external_trainable_variables=[q, r])
model.train(iterations=ADAM_STEPS, display_every=1000)
dde.optimizers.config.set_LBFGS_options(maxiter=1000, maxcor=20)
model.compile("L-BFGS", loss_weights=weights,
              external_trainable_variables=[q, r])
history, _ = model.train()

# Save the learned coefficients, solution and residuals.
X_test = saved["X_test"]
u_pred = model.predict(X_test)
residual = model.predict(X_test, operator=pde)
q_found, r_found = q.item(), r.item()

if not (
    np.isfinite(u_pred).all()
    and np.isfinite(residual).all()
    and np.isfinite([q_found, r_found]).all()
):
    raise RuntimeError("Training produced non-finite values; inspect the loss.")

np.savez(
    RESULTS / "inverse.npz", X=X_test, u=u_pred, residual=residual,
    q=q_found, r=r_found, steps=history.steps, loss=history.loss_train, seed=SEED,
)
print(f"Recovered q = {q_found:.8f}, r = {r_found:.8f}")
