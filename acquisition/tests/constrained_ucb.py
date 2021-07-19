import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ScalarizedObjective
import sys, os
import sys, os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from advanced_botorch import constrained
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
        
    constraints = [{1:[-1000,1000]},{1:[0.0,0.4]}]

    #need to create a scalarized objective object for BoTorch
    obj = ScalarizedObjective(torch.tensor((1.0,0.0)))

    n_init = 3
    x_init = torch.rand(n_init, 2)

    for i in [0,1]:
        X, Y = optimize(x_init, 20, constraints[i], obj)
        ax[i].plot(X[:,0][:n_init], X[:,1][:n_init],'+C0')
        ax[i].plot(X[:,0][n_init:], X[:,1][n_init:],'+C1')

    

def f(X):
    Y = 1 - (X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
    Y += 0.1 * torch.rand_like(Y)
    Y = (Y - Y.mean()) / Y.std()

    #second index serves as the constraint
    C = X[:,0]
    result = torch.cat((Y.reshape(-1,1),C.reshape(-1,1)), axis = 1)    
    return result


def optimize(x_initial, n_steps, constraint, objective):
    train_X = x_initial
    train_Y = f(train_X)
    
    for i in range(n_steps):
        print(i)
        gp = SingleTaskGP(train_X, train_Y)
        fit_gp(gp)

        #get candidate for observation and add to training data
        candidate = max_acqf(gp, constraint, objective)
        train_X = torch.cat((train_X, candidate))
        train_Y = f(train_X)
        
    return train_X, train_Y
        

def max_acqf(gp, constraint, objective):
    #finds new canidate point based on EHVI acquisition function

    constr = constrained.ConstrainedAcquisitionFunction(gp, constraint, 0)
    UCB = UpperConfidenceBound(gp, beta = 2.0, objective = objective)
    comb = combine_acquisition.MultiplyAcquisitionFunction(gp, [constr, UCB])
    
    bounds = torch.stack([torch.zeros(2), torch.ones(2)])
    candidate, acq_value = optimize_acqf(
        comb, bounds=bounds, q=1, num_restarts=5, raw_samples=20)

    return candidate

        
def fit_gp(gp):
    #fits GP model
    mll = ExactMarginalLogLikelihood(gp.likelihood,
                                     gp)

    fit_gpytorch_model(mll)



main()
plt.show()

