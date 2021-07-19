import botorch
import torch

#define classes that combine acquisition functions

class MultiplyAcquisitionFunction(botorch.acquisition.acquisition.AcquisitionFunction):
    def __init__(self, model, acquisition_functions):
        '''
        Acquisition function class that combines several seperate acquisition functions
        together by multiplying them

        Arguments
        ---------
        acquisition_functions : list
            List of acquisition functions to multiply together

        '''

        super().__init__(model)

        self.acqisition_functions = acquisition_functions

    def forward(self, X):
        value = torch.ones(X.shape[0])

        for function in self.acqisition_functions:
            multiplier = function.forward(X)
            value = value * multiplier

        return value
