import numpy as np
import torch

'''lightweight class for normalization/standardization using numpy'''


class Transformer:
    def __init__(self, x, transform_type='unitary'):
        '''
        Transformer class that allows normalization and standardization of parameters.
        - Use forward method to normalize input vector
        - Use backward method to unnormalize input vector
        Does not support backpropagation!
        
        Arguments
        ---------
        x : ndarray, shape (N x M), optional, default None
             Input data to determine normalization parameters where N is the number of points and M is the dimensionality
        
        bounds : ndarray, shape (M x 2), optional, default None
             Alternate specification of normalization bounds instead of data, bounds[M][0] is the M'th lower bound,
                                                                              bounds[M][1] is the M'th upper bound
        
        transform_type : ['unitary', 'normalize', standardize']
            Transformation method.
                - 'unitary' : No modification of input data
                - 'normalize' : Scales and shifts data s.t. data is between 0 and 1
                - 'standardize' : Scales and shifts data s.t. data has a mean of 0.0 and a rms size of 1.0
        
        
        '''

        possible_transformations = ['unitary', 'normalize', 'standardize']
        assert transform_type in possible_transformations

        self.ttype = transform_type
        assert len(x.shape) == 2
        self.x = x

        self._get_stats()

    def _get_stats(self):
        min_func = torch.min if isinstance(self.x, torch.Tensor) else np.min
        max_func = torch.max if isinstance(self.x, torch.Tensor) else np.max
        mean_func = torch.mean if isinstance(self.x, torch.Tensor) else np.mean
        std_func = torch.std if isinstance(self.x, torch.Tensor) else np.std

        if self.ttype == 'normalize':
            if isinstance(self.x, torch.Tensor):
                self.mins = torch.min(self.x, 0)[0]
                self.maxs = torch.max(self.x, 0)[0]

            else:
                self.mins = np.min(self.x, 0)
                self.maxs = np.max(self.x, 0)

        elif self.ttype == 'standardize':
            self.means = mean_func(self.x, 0)
            self.stds = std_func(self.x, 0)

    def recalculate(self, x):
        # change transformer data and recalculate stats
        self.x = x
        self._get_stats()

    def forward(self, x_old, scale_only=False):
        assert isinstance(x_old, type(self.x))
        assert len(x_old.shape) == 2

        x_new = x_old.clone() if isinstance(x_old, torch.Tensor) else x_old.copy()

        shift_factor = 0.0 if scale_only else 1.0

        if self.ttype == 'normalize':
            for i in range(x_new.shape[1]):
                if self.maxs[i] - self.mins[i] == 0.0:
                    x_new[:, i] = x_new[:, i] - self.mins[i] * shift_factor
                else:
                    x_new[:, i] = (x_new[:, i] - self.mins[i] * shift_factor) / (self.maxs[i] - self.mins[i])

        elif self.ttype == 'standardize':
            for i in range(x_new.shape[1]):
                if self.stds[i] == 0:
                    x_new[:, i] = x_new[:, i] - self.means[i] * shift_factor
                else:
                    x_new[:, i] = (x_new[:, i] - self.means[i] * shift_factor) / self.stds[i]

        return x_new

    def backward(self, x_old, scale_only=False):

        assert isinstance(x_old, type(self.x))
        assert len(x_old.shape) == 2

        x_new = x_old.clone() if isinstance(x_old, torch.Tensor) else x_old.copy()

        shift_factor = 0.0 if scale_only else 1.0

        if self.ttype == 'normalize':
            for i in range(x_new.shape[1]):
                x_new[:, i] = x_new[:, i] * (self.maxs[i] - self.mins[i]) + self.mins[i] * shift_factor

        elif self.ttype == 'standardize':
            for i in range(x_new.shape[1]):
                x_new[:, i] = x_new[:, i] * self.stds[i] + self.means[i] * shift_factor

        return x_new


if __name__ == '__main__':
    # testing suite
    x = np.random.uniform(size=(10, 2)) * 10.0
    print(x)
    t = Transformer(x, 'standardize')
    x_test = t.forward(x)
    print(x_test)
    print(t.backward(t.forward(x)))
