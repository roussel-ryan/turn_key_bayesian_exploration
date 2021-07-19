import numpy as np
import torch
import copy
import gpytorch


def train_exact_model(model, likelihood, train_x, train_y, n_steps = 500, lr = 0.1, fname = 'exact_', verbose = False):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr = lr)

    #for name, param in model.named_parameters():
    #print(f'{name}:{param.requires_grad}')
    
    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = 10000
    losses = []
    old_avg = best_loss
    for i in range(n_steps):
        optimizer.zero_grad()
        output = model(train_x)

        try:
            loss = -mll(output, train_y)
        except gpytorch.utils.errors.NotPSDError:
            print(f'ran into issues fitting - best_loss {best_loss}')
            break
        
        loss.backward(retain_graph=True)
        if loss.item() < best_loss:
            best_mparam = copy.deepcopy(model.state_dict())
            best_lparam = copy.deepcopy(likelihood.state_dict())
            best_loss = loss.item()

        losses.append([loss.detach().numpy()])

        if i % 10 == 0:
            if verbose:
                print('Iter %d - Loss: %.3f - Best loss %.3f' % (i + 1, loss.item(), best_loss))
            
            try:
                new_avg = np.mean(np.array(losses[-50:])) 
                if np.abs((new_avg - old_avg)) < 0.001:
                    #break
                    pass
                else:
                    old_avg = new_avg
            except IndexError:
                pass
            
        optimizer.step()

    #set model + likelihood params to the best
    model.load_state_dict(best_mparam)
    torch.save(model.state_dict(), f'{fname}model.pth')


def train_variational_model(model, likelihood, train_x, train_y,
                            n_steps = 500, lr = 0.1, fname = 'variational_',
                            verbose = True):
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr = lr)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())

    best_loss = 10000
    losses = []
    old_avg = best_loss
    for i in range(n_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y.flatten())
        loss.backward(retain_graph=True)
        if loss.item() < best_loss:
            best_mparam = copy.deepcopy(model.state_dict())
            best_lparam = copy.deepcopy(likelihood.state_dict())
            best_loss = loss.item()

            
        losses.append([loss.detach().numpy()])

        if i % 10 == 0:
            if verbose:
                print('Iter %d - Loss: %.3f - Best loss %.3f' % (i + 1, loss.item(), best_loss))
            
            try:
                new_avg = np.mean(np.array(losses[-50:])) 
                
                if np.abs(new_avg - old_avg) < 0.001:
                    pass
                    
                else:
                    old_avg = new_avg
            except IndexError:
                pass
            
        optimizer.step()

    #set model + likelihood params to the best
    model.load_state_dict(best_mparam)
    likelihood.load_state_dict(best_lparam)
    torch.save(model.state_dict(), f'{fname}model.pth')
    torch.save(likelihood.state_dict(), f'{fname}lk.pth')

