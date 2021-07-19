import torch
import gpytorch
import math
import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
default_seed = 10000
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy
from gpytorch.variational import VariationalStrategy

# Create dataset 
X, y = make_moons(noise=0.3, random_state=0)
Xt = torch.from_numpy(X).float()
yt = torch.from_numpy(y).float()

# Create evalutaion grid
h = 0.05
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
X_eval = np.vstack((xx.reshape(-1), 
                    yy.reshape(-1))).T
X_eval = torch.from_numpy(X_eval).float()


class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations = True
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


# Initialize model and likelihood
model = GPClassificationModel(Xt)
likelihood = gpytorch.likelihoods.BernoulliLikelihood()
training_iterations = 300
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
# num_data refers to the number of training datapoints
mll = gpytorch.mlls.VariationalELBO(likelihood, 
                                    model, 
                                    yt.numel(), 
                                    combine_terms=False)

for i in range(training_iterations):
    # Zero backpropped gradients from previous iteration
    optimizer.zero_grad()
    # Get predictive output
    output = model(Xt)
    # Calc loss and backprop gradients
    log_lik, kl_div, log_prior = mll(output, yt)
    loss = -(log_lik - kl_div + log_prior)
    #loss = -mll(output, yt)
    loss.backward()
    
    print('Iter %d/%d - Loss: %.3f lengthscale: %.3f outputscale: %.3f' % (
        i + 1, training_iterations, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.covar_module.outputscale.item() # There is no noise in the Bernoulli likelihood
    ))
    
    optimizer.step()


# Go into eval mode
model.eval()
likelihood.eval()

with torch.no_grad():    
    test_x = torch.linspace(0, 1, 101)
    # Get classification predictions
    observed_pred = likelihood(model(X_eval))

    p = observed_pred.mean.numpy()
    Z_gpy = p.reshape(xx.shape)

optimizer = 'fmin_l_bfgs_b'
gp_skl = GaussianProcessClassifier(kernel=1.0**2*RBF(length_scale=1.),
                                  optimizer=optimizer).fit(X, y)

# plot the decision function for each datapoint on the grid
Z_skl = gp_skl.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
Z_skl = Z_skl.reshape(xx.shape)

print("Kernel: {}".format(gp_skl.kernel_))
print("Log Marginal Likelihood: {}, {}".format(gp_skl.log_marginal_likelihood(gp_skl.kernel_.theta),
                                               gp_skl.log_marginal_likelihood_value_))

# Initialize fig and axes for plot
f, ax = plt.subplots(1, 2, figsize=(10, 3))
lvls = np.linspace(0,1,16)
c = ax[0].contourf(xx,yy,Z_gpy, levels=lvls)
ax[1].contourf(xx,yy,Z_skl, levels=lvls)
ax[0].scatter(X[y == 0,0], X[y == 0,1])
ax[0].scatter(X[y == 1,0], X[y == 1,1])
ax[1].scatter(X[y == 0,0], X[y == 0,1])
ax[1].scatter(X[y == 1,0], X[y == 1,1])
ax[0].set_title('GPyTorch')
ax[1].set_title('Sklearn')
f.colorbar(c)
plt.show()
