import os

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

import transformer

mpl.rc('image', cmap='Greens')

data_folder = os.path.join(os.getcwd(), '../data/')


def get_data(fname):
    data = pd.read_pickle(fname)
    if 'scan' not in fname:
        data = data  # .iloc[:250]

    # select good data

    good_data = data.loc[data['EMIT'].notnull()]
    bad_data = data.loc[data['EMIT'].isnull()]

    if 'DQ4' in good_data.keys():
        train_x = torch.from_numpy(good_data[['FocusingSolenoid', 'MatchingSolenoid', 'DQ4', 'DQ5']].to_numpy())
        constraint_x = torch.from_numpy(data[['FocusingSolenoid', 'MatchingSolenoid', 'DQ4', 'DQ5']].to_numpy())

    else:
        train_x = torch.from_numpy(good_data[['FocusingSolenoid', 'MatchingSolenoid']].to_numpy())
        constraint_x = torch.from_numpy(data[['FocusingSolenoid', 'MatchingSolenoid']].to_numpy())

    train_y = 2.0 * torch.from_numpy(good_data[['EMIT']].to_numpy())
    constraint_y = torch.from_numpy(data[['IMGF']].to_numpy())

    # normalize
    tx = transformer.Transformer(train_x, 'normalize')
    ty = transformer.Transformer(train_y, 'standardize')

    train_x = tx.forward(train_x)
    constraint_x = tx.forward(constraint_x)
    train_y = ty.forward(train_y)

    return train_x, train_y, constraint_x, constraint_y, tx, ty, bad_data, good_data


def plot_valid_region(fname, ax):
    c_fname = data_folder + '2d_scan.pkl'

    _, _, constr_x, constr_y, tx, ty, bad_data, good_data = get_data(c_fname)
    _, _, _, _, tx, ty, bad_data, good_data = get_data(fname)

    cgp = SingleTaskGP(constr_x, constr_y)
    cmll = ExactMarginalLogLikelihood(cgp.likelihood, cgp)
    fit_gpytorch_model(cmll)

    n = 50
    xlim = (6.05, 9.07)
    ylim = (1.0, 2.5)
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    xx = np.meshgrid(x, y)
    pts = np.vstack([ele.ravel() for ele in xx]).T

    pts = torch.from_numpy(pts)
    pts = tx.forward(pts)

    with torch.no_grad():
        post = cgp.posterior(pts)
        cmean = post.mean
        cstd = torch.sqrt(post.variance)

        prob = 1 - 0.5 * (1 + torch.erf((0.5 - cmean) / (cstd * torch.sqrt(torch.tensor(2.0)))))

    pts = tx.backward(pts)

    c = ax.pcolor(pts[:, 0].reshape(n, n),
                  pts[:, 1].reshape(n, n),
                  prob.numpy().reshape(n, n),
                  vmin=0, vmax=1)

    ax.plot(good_data['FocusingSolenoid'],
            good_data['MatchingSolenoid'],
            'C1o-', ms=3, label='Valid')
    ax.plot(bad_data['FocusingSolenoid'],
            bad_data['MatchingSolenoid'],
            'C4+',
            label='Invalid')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    return c


def add_valid_contour(ax, c_fname=data_folder + '2d_scan.pkl'):
    _, _, constr_x, constr_y, tx, ty, bad_data, good_data = get_data(c_fname)

    cgp = SingleTaskGP(constr_x, constr_y)
    cmll = ExactMarginalLogLikelihood(cgp.likelihood, cgp)
    fit_gpytorch_model(cmll)

    n = 50
    xlim = (6.05, 9.07)
    ylim = (1.0, 2.5)
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    xx = np.meshgrid(x, y)
    pts = np.vstack([ele.ravel() for ele in xx]).T

    pts = torch.from_numpy(pts)
    pts = tx.forward(pts)

    with torch.no_grad():
        post = cgp.posterior(pts)
        cmean = post.mean
        cstd = torch.sqrt(post.variance)

        prob = 1 - 0.5 * (1 + torch.erf((0.5 - cmean) / (cstd * torch.sqrt(torch.tensor(2.0)))))

    pts = tx.backward(pts)

    for a in ax:
        c = a.contourf(pts[:, 0].reshape(n, n),
                       pts[:, 1].reshape(n, n),
                       prob.numpy().reshape(n, n),
                       levels=[0.0, 0.5],
                       cmap='cividis',
                       alpha=0.5)

        a.contour(pts[:, 0].reshape(n, n),
                  pts[:, 1].reshape(n, n),
                  prob.numpy().reshape(n, n),
                  levels=[0.5],
                  linestyles='dashed',
                  cmap='cividis')

        a.set_xlim(*xlim)
        a.set_ylim(*ylim)


def plot_prediction(fname, xlim, ylim, ax, ax2, ax3, n=10):
    train_x, train_y, _, _, tx, ty, bad_data, good_data = get_data(fname)

    gp = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    print(tx.backward(gp.covar_module.base_kernel.lengthscale, scale_only=True))

    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    xx = np.meshgrid(x, y)
    pts = np.vstack([ele.ravel() for ele in xx]).T

    if 'scan' not in fname:
        pts = np.hstack([pts, np.zeros((len(pts), 2))])
        pts[:, -1] = 0.0

    pts = torch.from_numpy(pts)
    pts = tx.forward(pts)

    with torch.no_grad():
        post = gp.posterior(pts)
        mean = ty.backward(post.mean) * 42.0 * 1e6 / 0.511

        std = torch.sqrt(post.variance)
        std = std * ty.stds * 42.0 * 1e6 / 0.511

    pts = tx.backward(pts)

    c = ax.pcolor(pts[:, 0].reshape(n, n),
                  pts[:, 1].reshape(n, n),
                  mean.numpy().reshape(n, n),
                  vmin=9 * 2, vmax=26 * 2)
    sample_x = tx.backward(train_x)

    ax.plot(sample_x.T[0], sample_x.T[1], 'C1o', ms=3, label='Valid')
    ax.plot(bad_data['FocusingSolenoid'],
            bad_data['MatchingSolenoid'], 'C4+',
            label='Invalid')

    N = float(len(good_data))
    print(N)

    K0_data = [bad_data['FocusingSolenoid'], good_data['FocusingSolenoid']]
    K1_data = [bad_data['MatchingSolenoid'], good_data['MatchingSolenoid']]

    nbins = 10
    ax2.hist(K0_data, bins=nbins, rwidth=1.0,
             color=['C4', 'C1'], range=xlim, stacked=True, density=True,
             histtype='stepfilled')
    ax3.hist(K1_data, bins=nbins, rwidth=1.0,
             color=['C4', 'C1'], range=ylim, stacked=True, density=True,
             histtype='stepfilled',
             orientation='horizontal')

    # ax2.hist(good_data['FocusingSolenoid'], density=False, bins=19, rwidth=1.0,
    #         color='C1', range=xlim, histtype='stepfilled')
    # ax3.hist(good_data['MatchingSolenoid'], density=False, bins=19, rwidth=1.0,
    #         range=ylim, histtype='stepfilled', color='C1',
    #         orientation='horizontal')

    # ax2.hist(bad_data['FocusingSolenoid'], density=False, bins=19, rwidth=1.0,
    # color='C4', range=xlim,
    # histtype='stepfilled', alpha = 0.5)
    # ax3.hist(bad_data['MatchingSolenoid'], density=False, bins=19, rwidth=1.0,
    #         range=ylim, histtype='stepfilled', color='C4',
    #         orientation='horizontal', alpha=0.5)

    return c


def plot_figure():

    fnames = ['2d_scan.pkl',
              'bayes_exp.pkl']

    fig = plt.figure()
    fig.set_size_inches(8, 4)
    gs0 = gridspec.GridSpec(1, 2,
                            top=0.96,
                            bottom=0.12,
                            left=0.08,
                            right=0.93,
                            hspace=0.2,
                            wspace=0.4)

    xlim = (6.05, 9.07)
    ylim = (1.0, 2.5)

    for name, i in zip(fnames, [0, 1]):
        gs00 = gs0[i].subgridspec(2, 3,
                                  width_ratios=[0.3, 1.0, 0.1],
                                  height_ratios=[0.3, 1.0][::-1],
                                  wspace=0.05,
                                  hspace=0.05)
        ax1 = fig.add_subplot(gs00[0, 1])
        # ax1.set_aspect((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))
        ax1.set_xlim(*xlim)
        ax1.set_ylim(*ylim)

        ax1.set_xlabel('K0 (arb. u.)')
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])

        # horiz axis histogram
        ax2 = fig.add_subplot(gs00[1, 1])
        ax2.set_xlim(*xlim)
        ax2.set_xlabel('K0 (arb. u.)')

        # vertical axis histogram
        ax3 = fig.add_subplot(gs00[0, 0])
        ax3.set_ylim(*ylim)
        ax3.set_ylabel('K1 (arb. u.)')
        ax3.set_xticks([1])

        c = plot_prediction(data_folder + name, xlim, ylim, ax1, ax2, ax3, 300)
        ax3.set_xlim(ax3.get_xlim()[::-1])
        # ax2.set_ylim(ax2.get_ylim()[::-1])

        # add colorbar
        axc = fig.add_subplot(gs00[0, 2])
        fig.colorbar(c, cax=axc, label=r'$<\varepsilon_y>$ (mm mrad)')

        add_valid_contour([ax1])
        ax1.legend()

    ax1.get_legend().remove()

    fig.savefig('nat_comm_exp.png', dpi=600)


plot_figure()
plt.show()
