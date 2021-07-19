import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

import sys, os
import sys, os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from advanced_botorch import proximal
from advanced_botorch import combine_acquisition

import matplotlib.pyplot as plt

#-------------------------------------------------
#test out proximal ucb on simple test function
#-------------------------------------------------


def main():
    #demonstrate difference due to proximal addition
    fig,ax = plt.subplots(2,1)
    for a in ax:
        a.set_ylim(0,1)
        a.set_xlim(0,1)
        
    sigma_matrix_on = torch.eye(2) * 0.005
    sigma_matrix_off = torch.eye(2) * 10.0
    matricies = [sigma_matrix_on, sigma_matrix_off]

    n_init = 3
    x_init = torch.rand(n_init, 2)

    for i in [0,1]:
        X, Y = optimize(x_init, 20, matricies[i])
        
        ax[i].plot(X[:,0][:n_init], X[:,1][:n_init],'C0')
        ax[i].plot(X[:,0][n_init:], X[:,1][n_init:],'C1')

    

def f(X):
    Y = 1 - (X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
    Y += 0.1 * torch.rand_like(Y)
    Y = (Y - Y.mean()) / Y.std()
    return Y


def optimize(x_initial, n_steps, sigma_matrix):
    train_X = x_initial
    train_Y = f(train_X)
    
    for i in range(n_steps):
        print(i)
        gp = SingleTaskGP(train_X, train_Y)
        fit_gp(gp)

        #get candidate for observation and add to training data
        candidate = max_acqf(gp, sigma_matrix)
        train_X = torch.cat((train_X, candidate))
        train_Y = f(train_X)
        
    return train_X, train_Y
        

def max_acqf(gp, sigma_matrix):
    #finds new canidate point based on EHVI acquisition function

    prox = proximal.ProximalAcqusitionFunction(gp, sigma_matrix)
    UCB = UpperConfidenceBound(gp, beta=0.1)
    comb = combine_acquisition.MultiplyAcquisitionFunction(gp, [prox, UCB])
    
    bounds = torch.stack([torch.zeros(2), torch.ones(2)])
    candidate, acq_value = optimize_acqf(
        comb, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )

    return candidate

        
def fit_gp(gp):
    #fits GP model
    mll = ExactMarginalLogLikelihood(gp.likelihood,
                                     gp)

    fit_gpytorch_model(mll)



main()
plt.show()

