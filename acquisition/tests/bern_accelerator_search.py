import matplotlib.pyplot as plt

import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from torch.distributions.multivariate_normal import MultivariateNormal

from botorch.acquisition import UpperConfidenceBound, PosteriorMean
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ScalarizedObjective
import sys, os
import sys, os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from advanced_botorch import constrained2
from advanced_botorch import proximal
from advanced_botorch import combine_acquisition

import models
import training


#-------------------------------------------------
#test out proximal ucb + constraint on simple test function
#-------------------------------------------------


def main():
    #demonstrate difference due to proximal addition
        
    #constraints = [{1:[0.0,0.4]},{1:[0.0,0.4]}]
    constraint = {0:[-0.5, 0.5]}
    
    #sigma_matrix_on = torch.eye(2) * 0.5
    #sigma_matrix_off = torch.eye(2) * 1.0
    #matricies = [sigma_matrix_on, sigma_matrix_off]
    sigma_mult = [0.1**2]
    
    #need to create a scalarized objective object for BoTorch
    
    n_init = 5
    x_init = torch.rand(n_init, 2)*0.5 + 0.25


    fig,ax = plt.subplots(len(sigma_mult),1)
    fig2, ax2 = plt.subplots()
    #for a in ax:
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    
    for i in range(len(sigma_mult)):
        X, Y, mlls = optimize(x_init, 35, constraint, torch.eye(2) * sigma_mult[i])
        ax.plot(X[:,0][:n_init], X[:,1][:n_init],'+C0')
        ax.plot(X[:,0][n_init:], X[:,1][n_init:],'-C1')
        ax.plot(X[:,0][-1], X[:,1][-1],'oC1')
        
        #ax2.plot(mlls / np.arange(1,len(mlls)+1) )
    

def f(X):
    #Y = 1 - (x - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
    #Y += 0.1 * torch.rand_like(Y)
    #Y = (Y - Y.mean()) / Y.std()
    mean = torch.tensor((0.4,0.4))
    sigma = torch.eye(2)
    sigma[0, 0] = 1.0**2
    sigma[1, 1] = 0.25**2
    d = MultivariateNormal(mean, sigma)
    Y = torch.exp(d.log_prob(X)) / torch.exp(d.log_prob(mean))

    #second index serves as the constraint
    #C = x[:,0].numpy() < 0.75
    C = np.all(np.vstack(((X[:,0] - 0.5)**2 + (X[:,1] - 0.5)**2 < 0.35**2,
                          X[:,1] < X[:,0])), axis = 0)
    
    C = torch.from_numpy(C).float()
    
    result = torch.cat((Y.reshape(-1,1),C.reshape(-1,1)), axis = 1)    
    return result


def optimize(x_initial, n_steps, constraint, sigma_matrix):
    train_X = x_initial
    train_Y = f(train_X)

    mlls = []
    for i in range(n_steps):
        print(i)

        #define gp model for objective(s)
        gp = SingleTaskGP(train_X, train_Y[:,0].reshape(-1,1))
        mll_val = fit_gp(gp)
        mlls += [mll_val]

        #lk = gpytorch.likelihoods.GaussianLikelihood()
        #cgp = models.NNManifoldGPModel(train_X, train_Y[:,1], lk)
        #cgp = models.ExactGPModel(train_X, train_Y[:,1].reshape(-1,1), lk)
        cgp = SingleTaskGP(train_X, train_Y[:,1].reshape(-1,1))
        
        fit_gp(cgp)
        #define gp model for constraints(s) - using Bernoulli likelihood
        #cgp = models.GPClassificationModel(train_X)

        #enforce a prior mean of 1
        #cgp.mean_module.constant.data = torch.ones(1)
        #cgp.mean_module.constant.requires_grad = False
        #lk = gpytorch.likelihoods.BernoulliLikelihood()
        #training.train_exact_model(cgp, lk, train_X,
        #train_Y[:,1],
        #verbose = False)

        #print(cgp.state_dict())
        
        #get candidate for observation and add to training data
        if i % 5 == 0:
            plot = True
        else:
            plot = False
            
        candidate = max_acqf(gp, cgp, constraint, sigma_matrix, plot = plot)
        train_X = torch.cat((train_X, candidate))
        train_Y = f(train_X)

    #plot_model(cgp, lk, train_X)
        
    candidate = max_acqf(gp, cgp, constraint, sigma_matrix, plot = True)
    print(gp.covar_module.base_kernel.lengthscale)
    print(cgp.covar_module.base_kernel.lengthscale)

    return train_X, train_Y, np.array(mlls)

    

def max_acqf(gp, cgp, constraint, sigma_matrix, plot = False):
    #finds new canidate point based on EHVI acquisition function

    constr = constrained2.ConstrainedAcquisitionFunction(cgp)
    #constr = PosteriorMean(cgp)
    prox = proximal.ProximalAcqusitionFunction(gp, sigma_matrix)
    UCB = UpperConfidenceBound(gp, beta = 1e6)
    comb = combine_acquisition.MultiplyAcquisitionFunction(gp, [constr, prox, UCB])
    
    bounds = torch.stack([torch.zeros(2), torch.ones(2)])

    if plot:
        #plot_acq(comb, bounds.numpy().T, gp.train_inputs[0])
        plot_acq(constr, bounds.numpy().T, gp.train_inputs[0])
        #plot_acq(UCB, bounds.numpy().T, gp.train_inputs[0])
        
    candidate, acq_value = optimize_acqf(
        comb, bounds = bounds, q = 1, num_restarts = 10, raw_samples = 20)

    return candidate

        
def fit_gp(gp):
    #fits GP model
    mll = ExactMarginalLogLikelihood(gp.likelihood,
                                     gp)

    train_x = gp.train_inputs[0]
    train_y = gp.train_targets.reshape(-1,1)

    mll = fit_gpytorch_model(mll)
    #mll_val = mll(gp(train_x), train_y)
    mll_val = None
    
    return mll_val
    

def plot_model(model, lk, obs):
    fig, ax = plt.subplots()

    n = 25
    x = [np.linspace(0, 1, n) for e in [0,1]]
    xx,yy = np.meshgrid(*x)
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    pts = torch.from_numpy(pts).float()
    
    with torch.no_grad():
        pred = lk(model(pts))
        f = pred.mean

    c = ax.pcolor(xx,yy,f.detach().reshape(n,n),vmin = 0.0, vmax = 1.0)
    ax.plot(*obs.detach().numpy().T,'+')

    fig.colorbar(c)

def plot_acq(func, bounds, obs):
    fig, ax = plt.subplots()

    n = 25
    x = [np.linspace(*bnds, n) for bnds in bounds]
    xx,yy = np.meshgrid(*x)
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    pts = torch.from_numpy(pts).float()

    #print(pts)
    
    f = torch.zeros(n**2)
    for i in range(pts.shape[0]):
        f[i] = func(pts[i].reshape(1,-1))
    c = ax.pcolor(xx,yy,f.detach().reshape(n,n))
    ax.plot(*obs.detach().numpy().T,'+')

    ax.set_title(type(func))
    fig.colorbar(c)

main()
plt.show()

