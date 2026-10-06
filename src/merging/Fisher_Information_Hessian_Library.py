import torch
from torch.utils.data import DataLoader
from model.Create_Neural_Network import * 
from utils.DataLoader import *
from laplace import Laplace


"""
This source code is a modified extract from the following package
https://aleximmer.com/Laplace/

Daxberger et al. (2021)

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_hessian_matrix_library(model, data, fisher_n):
    """
    Calculate the full Hessian Matrix which is equivalent to the Fisher Matrix

    
    Args: 
        model:      The model for which the Fisher Information will be calculated
        data:       The raw data with which to calculate the Hessian Matrix
        fisher_n:   The amount training sampes used for calculation the Hessian


    Returns:
        
    """

    subset = Subset(data, range(min(fisher_n, len(data))))
    dataloader = DataLoader(
        subset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=lambda batch: (
            torch.stack([x for x, y in batch]),
            torch.tensor([y for x, y in batch], dtype=torch.long)
        )
    )
    
    model.to(device)
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure='full')
    la.fit(dataloader)

    F_matrix = la.H

    return F_matrix





   



