from model.Create_Neural_Network import *


def merge_fisher_approx(models, fisher_dicts, epsilon=1e-8):

    """
    Fisher Merge two models according to the approximated diagonal Fisher Matrixes. Weights each 
    parameter according to the Fisher value that was calculated. High values are weighted more 
    and vice versa. The logic behind this is from Matena et. al (2022)

    Args:
        models:         List of models that will be Fisher merged
        Fisher_dicts:   List of approximated Fisher dics, fishers_dics[i] assumes models[i] 
        epsinlon:       A

    
    Returns:

    
    """

    merged = clone_a_model(models[0])

    with torch.no_grad():
        for name, param in merged.named_parameters():
            if not param.requires_grad:
                continue

            # Numerator: sum of (fisher_i * weight_i) across all models
            numerator = torch.zeros_like(param)
            # Denominator: sum of fisher_i across all models
            denominator = torch.zeros_like(param)

            for model, fisher in zip(models, fisher_dicts):
                weight = model.state_dict()[name]
                fisher_value = fisher[name]

                numerator += fisher_value * weight
                denominator += fisher_value

            # Avoid division by zero where no model had any importance
            merged_weight = numerator / (denominator + epsilon)
            param.data.copy_(merged_weight)

    return merged


def clone_a_model(model):
    return copy.deepcopy(model)