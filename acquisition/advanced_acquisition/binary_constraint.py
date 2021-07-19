import torch
from torch.distributions import Normal
from botorch.acquisition import analytic
from botorch.acquisition import acquisition

from botorch.utils.transforms import t_batch_mode_transform


class Constraint(acquisition.AcquisitionFunction):
    def __init__(self, model, idx=0, lower_bound=-10**10, upper_bound=10**10):
        """
        Acquisition function that biases away from points that were observed
        to not satisfy a given constraint.

        Arguments
        ---------
        model : model
            A fitted model, usually separate from objective models

        Shamelessly adapted/stolen from BoTorch ConstrainedEI
        https://botorch.org/v/0.3.0/api/_modules/botorch/acquisition/analytic.html


        """
        assert upper_bound > lower_bound

        self.upper_bound = torch.tensor(upper_bound)
        self.lower_bound = torch.tensor(lower_bound)
        self.idx = idx
        super().__init__(model)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, x):
        posterior = self.model.posterior(x)
        means = posterior.mean.squeeze(dim=-2)[:, self.idx]
        sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)[:, self.idx]

        dists = self._construct_dist(means, sigmas, torch.arange(len(means)))
        prob_feas = dists.cdf(self.upper_bound) - dists.cdf(self.lower_bound)

        return prob_feas.squeeze(dim=-1)

    @staticmethod
    def _construct_dist(means, sigmas, inds):
        mean = means.flatten()  # .index_select(dim = -1, index = inds)
        sigma = sigmas.flatten()  # .index_select(dim = -1, index = inds)
        return Normal(loc=mean, scale=sigma)


class BinaryConstraint(Constraint):
    def __init__(self, model, idx=0):
        """
        Acquisition function that biases away from points that were observed
        to not satisfy a given constraint.

        Arguments
        ---------
        model : model
            A fitted model, usually separate from objective models

        Shamelessly adapted/stolen from BoTorch ConstrainedEI
        https://botorch.org/v/0.3.0/api/_modules/botorch/acquisition/analytic.html


        """

        super().__init__(model, idx, lower_bound=0.5)
