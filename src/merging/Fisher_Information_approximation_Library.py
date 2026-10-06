import numpy as np
from torch.nn import functional as F

from model.Create_Neural_Network import *
from utils.DataLoader import *

"""
This source code is a modified extract from the following repository
https://github.com/GMvandeVen/continual-learning/blob/master/models/cl/continual_learner.py 
From Van de Ven et al. (2022)
"""

def estimate_fisher_approx(model, dataset, fisher_labels, fisher_n):
    """
    Estimates the diagonal Fisher Matrix for a given model with its training data set.
    The diagonal approximation computes the expected squared gradient per parameter that 
    measures how sensitive the loss is to changes in each parameter.
    
    Args:
        model:              Neural network for which the Fisher Information is being calculated

        dataset:            Raw dataset that is being used for the estimation

        fisher_labels:      Determines the label strategy that is being used for calculating the Fisher Information:
                            possible values:
                            "all":    Weighted combination of all class labels
                            "true":   uses the ground truth label (empirical Fisher)
                            "pred":   uses the predicted label
                            "sample": samples one label from the predicted distribution

        fisher_n:           Determines how many samples are being used for the estimation

    Returns:
        dict mapping parameter names to their diagonal Fisher Information values
        same structure as model.named_parameters(), normalized by the number of samples used
    
    """

    #Determine model device
    device = next(model.parameters()).device

    # Prepare <dict> to store estimated Fisher Information matrix, initially zero
    fisher_dic = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Save mode
    mode = model.training
    # Switch Evalutation mode, no Droppout
    model.eval()

    fisher_n_batch = 1;

    # Create DataLoader with custom batch_size, order does not matter
    dataloader = DataLoader(
        dataset,
        batch_size=fisher_n_batch,
        shuffle=False,
        num_workers=0
    )

    #Used for averaging
    num_samples = 0

    # Estimate the FI-matrix for fisher_n with batches of size n
    for index, (x, y) in enumerate(dataloader):
        # break from for-loop if max number of samples has been reached
        if fisher_n is not None and index > fisher_n:
                break
        x = x.to(device)
        y = y.to(device)
        num_samples += x.size(0)

        if num_samples % 1000 == 0:
            print(f"Processed {num_samples} fisher samples")

        # run forward pass of model
        output = model(x)
        # calculate FI-matrix (according to one of the four options)
        if fisher_labels == 'all' and fisher_n_batch==1:
            # -use a weighted combination of all labels
            with torch.no_grad():
                label_weights = F.softmax(output, dim=1)  # --> get weights, which shouldn't have gradient tracked
            for label_index in range(output.shape[1]):
                #Size one
                label = torch.LongTensor([label_index]).to(device)
                negloglikelihood = F.cross_entropy(output, label)  # --> get neg log-likelihoods for this class
                # Calculate gradient of negative loglikelihood, resets them, computes gradients for each possible class label one at a time
                model.zero_grad()
                negloglikelihood.backward(retain_graph=True if (label_index + 1) < output.shape[1] else False)
                # Square gradients and keep running sum (using the weights)
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        #Basically Fisher Formula in Pytorch
                        if p.grad is not None:
                            fisher_dic[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)
        else:
            # -only use one particular label for each datapoint
            if fisher_labels == 'true':
                # --> use provided true label to calculate loglikelihood --> "empirical Fisher":
                label = y  # -> shape: [model.fisher_batch]
                # if allowed_classes is not None:
                #     label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                #     label = torch.LongTensor(label)
                label = label.to(device)
            elif fisher_labels == 'pred':
                # --> use predicted label to calculate loglikelihood:
                label = output.max(1)[1]
            elif fisher_labels == 'sample':
                # --> sample one label from predicted probabilities
                with torch.no_grad():
                    label_weights = F.softmax(output, dim=1)  # --> get predicted probabilities
                weights_array = np.array(label_weights[0].cpu())  # --> change to np-array, avoiding rounding errors
                label = np.random.choice(len(weights_array), 1, p=weights_array / weights_array.sum())
                label = torch.LongTensor(label).to(device)  # --> change label to tensor on correct device
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
            # calculate gradient of negative loglikelihood
            model.zero_grad()
            negloglikelihood.backward()
            # square gradients and keep running sum
            for n, p in model.named_parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        fisher_dic[n] += p.grad.detach() ** 2

    # Normalize by sample size used for estimation
    est_fisher_info = {n: p / num_samples for n, p in fisher_dic.items()}

    # Set model back to its initial mode
    model.train(mode=mode)
    return est_fisher_info