import copy
import torch


def merge_weights_simple(models):
    """
    Takes a list of neural networks, average all their parameters with each other
    and creates a new model. Assumes that all models have the same architecture as
    otherwise merging is not possible.

    Args:
        models: list of models, that will be merged
    
    Returns:
        new model with the same architecture as the input models
    """

    # Create new model as copy of first one (both models need the same architecture)
    merged_model = copy.deepcopy(models[0])

    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            # Average all models weights
            merged = sum(m.state_dict()[name] for m in models) / len(models)
            param.data.copy_(merged)

    print("Average merge successfull")
    return merged_model