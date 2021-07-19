import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import matplotlib.gridspec as gridspec

import transformer


def track_data():
    folder = 'D:\\AWA\\mobo_04_15_21\\data\\'
    fname = 'binary_constraint_search.pkl'
    fnames = ['bayes_exp.pkl', '2d_adapt_search.pkl']

    gs_kw = {'width_ratios': [1.0, 0.2]}
    fig, ax = plt.subplots(4, 2, gridspec_kw=gs_kw)
    fig.set_size_inches(8, 4)

    for name, j in zip(fnames, [0]):
        data = pd.read_pickle(folder + name)
        if 'scan' not in name:
            data = data  # .iloc[:250]

        # select good data

        good_data = data.loc[data['EMIT'].notnull()]
        bad_data = data.loc[data['EMIT'].isnull()]

        # calculate good fraction
        emit_data = data['EMIT'].to_numpy()
        max_state = 50
        n_samples = 5
        good_frac = []
        for i in range(1, max_state + 1):
            good_frac += [np.count_nonzero(~np.isnan(emit_data[:i * n_samples])) / (i * n_samples)]
        print(good_frac[-1])

        if 'DQ4' in good_data.keys():
            names = ['FocusingSolenoid', 'MatchingSolenoid', 'DQ4', 'DQ5']
        else:
            names = ['FocusingSolenoid', 'MatchingSolenoid']

        x_full = torch.from_numpy(data[names].to_numpy())
        good_state = good_data['state_idx']
        bad_state = bad_data['state_idx']
        good_x = torch.from_numpy(good_data[names].to_numpy())
        bad_x = torch.from_numpy(bad_data[names].to_numpy())

        tx = transformer.Transformer(x_full, 'normalize')

        good_x = tx.forward(good_x)
        bad_x = tx.forward(bad_x)

        # get lengthscale
        ls = np.load(name.split('.')[0] + '_lengthscale_trace_norm.npy')

        n_params = good_x.shape[-1]
        for i in range(n_params):
            ax[i][j].plot(good_state, good_x[:, i], 'o', label='Valid', c='C1', ms=3)
            ax[i][j].plot(bad_state, bad_x[:, i], '+', label='Invalid', c='C4')
            # ax[i][j].plot(ls[:, -1], ls[:, i], label='Lengthscale')
            data = [bad_x.numpy().T[i], good_x.numpy().T[i]]

            ax[i][j + 1].hist(data, bins=10, rwidth=1.0,
                              color=['C4', 'C1'], range=[0, 1], stacked=True, density = True,
                              histtype='stepfilled', orientation='horizontal')
            ax[i][j + 1].set_xlim(0,3.75)
            ax[i, j].set_yticks([])
            ax[i, j + 1].set_yticks([])

        ax[-1][j].set_xlabel('Sample index')

    fig.subplots_adjust(top=0.97,
                        bottom=0.13,
                        left=0.075,
                        right=0.99,
                        hspace=0.065,
                        wspace=0.015)
    for i in range(3):
        ax[i, 1].set_xticks([])
        ax[i, 0].set_xticks([])

    ax[-1][0].legend(loc = 3)

    labels = ['K0', 'K1', 'DQ4', 'DQ5']
    for i in range(4):
        ax[i, 0].set_ylabel(labels[i])
        ax[i, 0].set_yticks([0, 0.5])

    ax[-1, 1].set_xlabel('Sample Density')

    # ax[0, 0].set_title(r'Bayes. Exp, $\Sigma = 0.01\mathbf{I}$')
    # ax[0, 2].set_title(r'Bayes. Exp, $\Sigma = 100\mathbf{I}$')

    if 0:
        lbl = ['a', 'b']
        for a, label in zip(ax[0], lbl):
            a.text(-0.02, 1.10, f'({label})', ha='right',
                   va='top', transform=a.transAxes,
                   fontdict={'size': 12})

    #fig.savefig('trace.svg')
    #fig.savefig('trace.png', dpi=600)


track_data()
plt.show()
