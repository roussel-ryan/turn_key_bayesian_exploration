import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import binary_constraint
import combine_acquisition
import proximal


# -------------------------------------------------
# test out proximal ucb + constraint on simple test function
# -------------------------------------------------


def main():
    sigma_mult = [0.3 ** 2]

    n_init = 1
    # x_init = torch.rand(n_init, 2)
    x_init = torch.tensor((0.5, 0.4)).reshape(1, -1)

    fig, ax = plt.subplots(len(sigma_mult), 1)

    if isinstance(ax, np.ndarray):
        for a in ax.flatten():
            a.set_ylim(0, 1)
            a.set_xlim(0, 1)
            a.set_aspect('equal')

    else:
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_aspect('equal')
        ax = [ax]

    for i in range(len(sigma_mult)):
        X, Y, mlls, cmlls = optimize(x_init, 35, torch.eye(2) * sigma_mult[i])
        ax[i].plot(X[:, 0][:n_init], X[:, 1][:n_init], '+C0')
        ax[i].plot(X[:, 0][n_init:], X[:, 1][n_init:], '+C1')
        ax[i].plot(X[:, 0][-1], X[:, 1][-1], 'oC1')


def f(X):
    Y = torch.sin(2.0 * np.pi * X[:, 0]) * torch.sin(np.pi * X[:, 1])

    # second index serves as the constraint
    # C = x[:,0].numpy() < 0.75
    C = np.all(np.vstack(((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2 < 0.35 ** 2,
                          X[:, 1] < X[:, 0])), axis=0)

    C = torch.from_numpy(C).float()

    result = torch.cat((Y.reshape(-1, 1), C.reshape(-1, 1)), axis=1)
    return result


def optimize(x_initial, n_steps, sigma_matrix):
    train_X = x_initial
    train_Y = f(train_X)

    mlls = []
    cmlls = []
    for i in range(n_steps):
        print(i)

        # define gp model for objective(s)
        gp = SingleTaskGP(train_X, train_Y[:, 0].reshape(-1, 1))

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        # define constraint GP
        cgp = SingleTaskGP(train_X, train_Y[:, 1].reshape(-1, 1))

        cmll = ExactMarginalLogLikelihood(cgp.likelihood, cgp)
        fit_gpytorch_model(cmll)

        # get candidate for observation and add to training data
        if i % 10 == 0:
            plot = False
        else:
            plot = False

        candidate = max_acqf(gp, cgp, sigma_matrix, plot=plot)
        train_X = torch.cat((train_X, candidate))
        train_Y = f(train_X)

    plot_func(train_X)

    candidate = max_acqf(gp, cgp, sigma_matrix, plot=True)
    print(gp.covar_module.base_kernel.lengthscale)
    print(cgp.covar_module.base_kernel.lengthscale)

    return train_X, train_Y, np.array(mlls), np.array(cmlls)


def max_acqf(gp, cgp, sigma_matrix, plot=False):
    # finds new canidate point based on CPBE acquisition function

    constr = binary_constraint.BinaryConstraint(cgp)
    prox = proximal.ProximalAcqusitionFunction(gp, sigma_matrix)
    UCB = UpperConfidenceBound(gp, beta=1e9)
    comb = combine_acquisition.MultiplyAcquisitionFunction(gp, [constr, prox, UCB])

    bounds = torch.stack([torch.zeros(2), torch.ones(2)])

    if plot:
        plot_acq(comb, bounds.numpy().T, gp.train_inputs[0])
        plot_acq(constr, bounds.numpy().T, gp.train_inputs[0])
        plot_acq(UCB, bounds.numpy().T, gp.train_inputs[0])

    candidate, acq_value = optimize_acqf(
        comb, bounds=bounds, q=1, num_restarts=10, raw_samples=20)

    return candidate


def fit_gp(gp):
    # fits GP model
    mll = ExactMarginalLogLikelihood(gp.likelihood,
                                     gp)

    train_x = gp.train_inputs[0]
    train_y = gp.train_targets.reshape(-1, 1)

    mll = fit_gpytorch_model(mll)

    gp.train()
    mll_val = mll(gp(train_x), train_y.flatten())
    return mll_val


def plot_model(model, lk, obs):
    fig, ax = plt.subplots()

    n = 25
    x = [np.linspace(0, 1, n) for e in [0, 1]]
    xx, yy = np.meshgrid(*x)
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    pts = torch.from_numpy(pts).float()

    with torch.no_grad():
        pred = lk(model(pts))
        f = pred.mean

    c = ax.pcolor(xx, yy, f.detach().reshape(n, n), vmin=0.0, vmax=1.0)
    ax.plot(*obs.detach().numpy().T, '+')

    fig.colorbar(c)


def plot_func(obs):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    axes = [ax1, ax2]
    figs = [fig1, fig2]

    n = 50
    x = [np.linspace(0, 1, n) for e in [0, 1]]
    xx, yy = np.meshgrid(*x)
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    pts = torch.from_numpy(pts).float()

    for i in [0, 1]:
        f_val = f(pts)[:, i]

        ax = axes[i]
        fig = figs[i]
        c = ax.pcolor(xx, yy, f_val.detach().reshape(n, n))
        ax.plot(*obs.detach().numpy().T, '-C1', marker='o')
        ax.set_ylabel('x2')
        ax.set_xlabel('x1')
        ax.set_aspect('equal')
        fig.colorbar(c)


def plot_acq(func, bounds, obs):
    fig, ax = plt.subplots()

    n = 50
    x = [np.linspace(*bnds, n) for bnds in bounds]
    xx, yy = np.meshgrid(*x)
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    pts = torch.from_numpy(pts).float()

    f = torch.zeros(n ** 2)
    for i in range(pts.shape[0]):
        f[i] = func(pts[i].reshape(1, -1))
    c = ax.pcolor(xx, yy, f.detach().reshape(n, n))
    ax.plot(*obs.detach().numpy().T, 'oC1')

    ax.set_title(type(func))
    fig.colorbar(c)
    fig.savefig(f'results/{type(func).__name__}.png')


main()
plt.show()
