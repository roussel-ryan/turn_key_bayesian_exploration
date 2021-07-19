import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from botorch.acquisition import acquisition 

from botorch.utils.transforms import t_batch_mode_transform

class ProximalAcqusitionFunction(acquisition.AcquisitionFunction):
    def __init__(self, model, sigma_matrix, scale_to_gp = False):
        '''
        Acquisition function that biases other acquistion functions towards a
        nearby region in input space

        Arguments
        ---------
        model : Model
            A fitted model
        
        precision_matrix : torch.tensor, shape (D x D)
            Precision matrix used in the biasing multivariate distribution, D is 
            the dimensionality of input space

        '''
        
        super().__init__(model)

        self.register_buffer('sigma_matrix', sigma_matrix)
        self.scale_to_gp = scale_to_gp


    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        #get the last point in the training set (assumed to be the most
        #recently evaluated point)
        #print(self.model.train_inputs[0][0][-1])
        last_pt = self.model.train_inputs[0][0][-1].double()
        #print(self.model.train_inputs)
        #print(last_pt)
        #print(x)
        #L = self.model.covar_module.outputscale
        #dist = torch.linalg.norm((x - last_pt) / L , dim = 0)
        
        #define multivariate normal
        #if self.scale_to_gp:
        #    sm = torch.matmul(,self.sigma_matrix)
        #else:
        sm = self.sigma_matrix
            
        d = MultivariateNormal(last_pt, sm.double())

        #use pdf to calculate the weighting - normalized to 1 at the last point
        norm = torch.exp(d.log_prob(last_pt).flatten())
        weight = torch.exp(d.log_prob(X).flatten()) / norm

        #weight = 1 - dist
        
        return weight
