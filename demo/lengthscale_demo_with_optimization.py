import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('image', cmap='Greens')


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


fig, axes = plt.subplots(2, 1)
fig.set_size_inches(4, 6)

orig_test_x = torch.tensor(((0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))).float()
orig_test_y = torch.tensor((1, 1, 1, 1, 1)).float()

ls = [[1, 1], [0.25, 1.0]]
for ax, l in zip(axes, ls):
    lk = gpytorch.likelihoods.GaussianLikelihood()
    lk.noise = torch.tensor(0.0001)

    test_x = orig_test_x
    test_y = orig_test_y

    model = ExactGPModel(test_x, test_y, lk)

    model.covar_module.base_kernel.lengthscale = torch.tensor(l).float()
    model.covar_module.base_kernel.raw_lengthscale.requires_grad = False

    n_iterations = 8
    for i in range(n_iterations):
        print(i)
        n = 1000
        x = np.linspace(-1.0, 1.0, n)
        xx = np.meshgrid(x, x)
        pts = np.vstack([ele.ravel() for ele in xx]).T
        pts = torch.from_numpy(pts).float()

        model.eval()
        lk.eval()

        with torch.no_grad():
            post = lk(model(pts))
            std = torch.sqrt(post.variance)

        best_pt = pts[torch.argmax(std)]

        test_x = torch.vstack((test_x, best_pt))
        test_y = torch.hstack((test_y, torch.tensor(1.0).float()))

        model = ExactGPModel(test_x, test_y, lk)

        model.covar_module.base_kernel.lengthscale = torch.tensor(l).float()
        model.covar_module.base_kernel.raw_lengthscale.requires_grad = False

    n = 300
    x = np.linspace(-1.1, 1.1, n)
    xx = np.meshgrid(x, x)
    pts = np.vstack([ele.ravel() for ele in xx]).T
    pts = torch.from_numpy(pts).float()

    model.eval()
    lk.eval()

    with torch.no_grad():
        post = lk(model(pts))
        std = torch.sqrt(post.variance)

    c = ax.pcolor(*xx, std.reshape(n, n) / torch.max(std), vmin=0, vmax=1)
    ax.plot(*test_x[:5].numpy().T, 'C0o')
    ax.plot(*test_x[5:].numpy().T, 'C1o')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    fig.colorbar(c, ax=ax, label='$\sigma\ (\mathbf{x}) / \sigma_{max}$')
    ax.set_aspect('equal')

fig.tight_layout()
fig.savefig('results/lengthscale.png', dpi=600)

plt.show()
