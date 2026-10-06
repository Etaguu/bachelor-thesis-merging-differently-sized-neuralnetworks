import random

from merging.Fisher_Information_approximation_Library import *


def select_neurons(model_big,data_set_raw,
                   fisher_n,
                   select_neuron_strategy,
                   num_neurons,
                   neuron_importance_calculation_arg="sum",
                   ):

    """
    
    Args:
        model_big:                          
        data_set_raw:                       
        fisher_n:                           
        select_neuron_strategy:             
        num_neurons:                        
        neuron_importance_calculation_arg:  

    Returns:
        
    """
    #Approximated
    fisher_dictionary = estimate_fisher_approx(model_big,data_set_raw,"all",fisher_n)

    neuron_scores_layer_0, neuron_scores_layer_2 = get_neuron_importance(fisher_dictionary, neuron_importance_calculation_arg)

    selected_indices_layer0, selected_indices_layer2 = choose_neuron_strategy(model_big,select_neuron_strategy,
                                              neuron_scores_layer_0,
                                              neuron_scores_layer_2,
                                              num_neurons)
                
        
    return selected_indices_layer0, selected_indices_layer2,fisher_dictionary

def choose_neuron_strategy(model_big, strategy, scores_layer0, scores_layer2, neuron_count):
    """
    

    Args:
        model_big:      
        strategy:       
        scores_layer0:  
        scores_layer2:  
        neuron_count:   

    Returns:
        
    """

    hidden_size = model_big.linear_relu_stack[0].out_features

    if strategy == "fisher_top":
        idx0 = torch.topk(scores_layer0, neuron_count).indices.sort().values.tolist()
        idx2 = torch.topk(scores_layer2, neuron_count).indices.sort().values.tolist()

    elif strategy == "fisher_bad":
        idx0 = torch.topk(scores_layer0, neuron_count, largest=False).indices.sort().values.tolist()
        idx2 = torch.topk(scores_layer2, neuron_count, largest=False).indices.sort().values.tolist()

    else:
        # Random, same indices for both layers
        idx0 = sorted(random.sample(range(hidden_size), neuron_count))
        idx2 = sorted(random.sample(range(hidden_size), neuron_count))

    return idx0, idx2

def initialize_a_model_with_big_model_weights(
    model_small, model_big, selected_indices_layer0, selected_indices_layer2
):
    """
    

    Args:
        model_small:                
        model_big:                  
        selected_indices_layer0:    
        selected_indices_layer2:    

    Returns:
        
    """

    hidden_size = model_small.linear_relu_stack[0].out_features

    if len(selected_indices_layer0) != hidden_size:
        raise ValueError(
            f"Small model expects {hidden_size} neurons, "
            f"but {len(selected_indices_layer0)} layer0 indices were selected."
        )
    if len(selected_indices_layer2) != hidden_size:
        raise ValueError(
            f"Small model expects {hidden_size} neurons, "
            f"but {len(selected_indices_layer2)} layer2 indices were selected."
        )

    with torch.no_grad():
        # Layer 0 — select rows by layer0 indices
        model_small.linear_relu_stack[0].weight.copy_(
            model_big.linear_relu_stack[0].weight[selected_indices_layer0, :]
        )
        model_small.linear_relu_stack[0].bias.copy_(
            model_big.linear_relu_stack[0].bias[selected_indices_layer0]
        )

        # Layer 2 — rows by layer2 indices, columns by layer0 indices
        model_small.linear_relu_stack[2].weight.copy_(
            model_big.linear_relu_stack[2].weight[selected_indices_layer2, :][:, selected_indices_layer0]
        )
        model_small.linear_relu_stack[2].bias.copy_(
            model_big.linear_relu_stack[2].bias[selected_indices_layer2]
        )

        # Layer 4 — columns by layer2 indices
        model_small.linear_relu_stack[4].weight.copy_(
            model_big.linear_relu_stack[4].weight[:, selected_indices_layer2]
        )
        model_small.linear_relu_stack[4].bias.copy_(
            model_big.linear_relu_stack[4].bias.clone()
        )

    return model_small

def merge_neurons_back_into_big_model(
    model_big,
    model_small,
    selected_indices_layer0,
    selected_indices_layer2,
    merge_strategy,
    train_raw,
    fisher_n,
    fisher_matrix_big,
):
    """
    Merges the weights of a small model back into the big model at the selected
    neuron positions using the specified merge strategy. Supports merging at
    selected positions or at non-selected positions for the switched setup.

    Args:
        model_big:                  big neural network that receives the merged weights
        model_small:                small neural network whose weights are merged in
        selected_indices_layer0:    indices of selected neurons in hidden layer 0
        selected_indices_layer2:    indices of selected neurons in hidden layer 2
        merge_strategy:             merging method to use
                                    possible values:
                                      "replace"                      — direct overwrite at selected positions
                                      "average"                      — element-wise average at selected positions
                                      "fisher"                       — Fisher-weighted blend at selected positions
                                      "replace_to_not_selected_neurons" — replace at non-selected positions
                                      "fisher_to_not_selected_neurons"  — Fisher merge at non-selected positions
        train_raw:                  list of raw datasets for Fisher estimation
                                    train_raw[1] is used for the small model
        fisher_n:                   number of samples for Fisher estimation
        fisher_matrix_big:          precomputed Fisher matrix of the big model

    Returns:
        tuple (merged_model, not_selected_layer0, not_selected_layer2)
        merged_model:        big model with merged weights
        not_selected_layer0: indices of non-selected neurons in layer 0
        not_selected_layer2: indices of non-selected neurons in layer 2
    """
    hidden_size = model_big.linear_relu_stack[0].out_features
    all_indices = set(range(hidden_size))

    not_selected_layer0 = sorted(all_indices - set(selected_indices_layer0))
    not_selected_layer2 = sorted(all_indices - set(selected_indices_layer2))

    if merge_strategy == "average":
        merged = average_merge_neurons(
            model_big, model_small, selected_indices_layer0, selected_indices_layer2
        )

    elif merge_strategy == "fisher":
        fisher_small = estimate_fisher_approx(model_small, train_raw[1], "all", fisher_n)
        merged = fisher_merge_neurons(
            model_big, model_small,
            fisher_matrix_big, fisher_small,
            selected_indices_layer0, selected_indices_layer2
        )

    elif merge_strategy == "replace":
        merged = replace_neurons(
            model_big, model_small, selected_indices_layer0, selected_indices_layer2
        )

    elif merge_strategy == "fisher_to_not_selected_neurons":
        fisher_small = estimate_fisher_approx(model_small, train_raw[1], "all", fisher_n)
        merged = fisher_merge_neurons(
            model_big, model_small,
            fisher_matrix_big, fisher_small,
            not_selected_layer0, not_selected_layer2
        )

    elif merge_strategy == "replace_to_not_selected_neurons":
        merged = replace_neurons(
            model_big, model_small, not_selected_layer0, not_selected_layer2
        )

    else:
        raise ValueError(f"Unknown merge strategy: {merge_strategy}")

    return merged, not_selected_layer0, not_selected_layer2

    
def replace_neurons(model_big, model_small,selected_indices_layer0, selected_indices_layer2):

    """
    
    Args:
        model_big:                  
        model_small:                
        selected_indices_layer0:    
        selected_indices_layer2:    

    Returns:
        
    """
    
    
    big = copy.deepcopy(model_big)
    device = next(big.parameters()).device

    idx0 = torch.tensor(selected_indices_layer0, dtype=torch.long, device=device)
    idx2 = torch.tensor(selected_indices_layer2, dtype=torch.long, device=device)

    with torch.no_grad():
        # Layer 0 — rows by layer0 indices
        big.linear_relu_stack[0].weight[idx0] = model_small.linear_relu_stack[0].weight
        big.linear_relu_stack[0].bias[idx0]   = model_small.linear_relu_stack[0].bias

        # Layer 2 — rows by layer2 indices, columns by layer0 indices
        big.linear_relu_stack[2].weight[idx2[:, None], idx0] = model_small.linear_relu_stack[2].weight
        big.linear_relu_stack[2].bias[idx2]                  = model_small.linear_relu_stack[2].bias

        # Layer 4 — columns by layer2 indices
        big.linear_relu_stack[4].weight[:, idx2] = model_small.linear_relu_stack[4].weight

    return big
  


def fisher_merge_neurons(
    model_big, model_small,
    fisher_big_dic, fisher_small_dic,
    selected_indices_layer0, selected_indices_layer2
):

    """
    

    Args:
        model_big:                  
        model_small:                
        fisher_big_dic:             
        fisher_small_dic:           
        selected_indices_layer0:    
        selected_indices_layer2:    

    Returns:
        
    """
    
    merged = clone_model(model_big)

    device = next(merged.parameters()).device

    idx0   = torch.tensor(selected_indices_layer0, dtype=torch.long, device=device)
    idx2   = torch.tensor(selected_indices_layer2, dtype=torch.long, device=device)

    def fisher_weighted(f_big, f_small, w_big, w_small):
        f_sum = f_big + f_small
        return torch.where(
            f_sum > 1e-8,
            (f_big * w_big + f_small * w_small) / f_sum,
            (w_big + w_small) / 2
        )

    with torch.no_grad():
        big = merged.linear_relu_stack
        sml = model_small.linear_relu_stack
        fb  = fisher_big_dic
        fs  = fisher_small_dic

        # Layer 0 — rows by layer0 indices
        big[0].weight[idx0] = fisher_weighted(
            fb["linear_relu_stack.0.weight"][idx0],
            fs["linear_relu_stack.0.weight"],
            big[0].weight[idx0],
            sml[0].weight
        )
        big[0].bias[idx0] = fisher_weighted(
            fb["linear_relu_stack.0.bias"][idx0],
            fs["linear_relu_stack.0.bias"],
            big[0].bias[idx0],
            sml[0].bias
        )

        # Layer 2 — rows by layer2 indices, columns by layer0 indices
        big[2].weight[idx2[:, None], idx0] = fisher_weighted(
            fb["linear_relu_stack.2.weight"][idx2[:, None], idx0],
            fs["linear_relu_stack.2.weight"],
            big[2].weight[idx2[:, None], idx0],
            sml[2].weight
        )
        big[2].bias[idx2] = fisher_weighted(
            fb["linear_relu_stack.2.bias"][idx2],
            fs["linear_relu_stack.2.bias"],
            big[2].bias[idx2],
            sml[2].bias
        )

        # Layer 4 — columns by layer2 indices
        big[4].weight[:, idx2] = fisher_weighted(
            fb["linear_relu_stack.4.weight"][:, idx2],
            fs["linear_relu_stack.4.weight"],
            big[4].weight[:, idx2],
            sml[4].weight
        )
        big[4].bias[:] = fisher_weighted(
            fb["linear_relu_stack.4.bias"],
            fs["linear_relu_stack.4.bias"],
            big[4].bias[:],
            sml[4].bias
        )

    return merged


def average_merge_neurons(model_big, model_small, selected_indices_layer0, selected_indices_layer2):

    """
    

    Args:
        model_big:                  
        model_small:                
        selected_indices_layer0:    
        selected_indices_layer2:    

    Returns:
        
    """



    big    = copy.deepcopy(model_big)

    device = next(big.parameters()).device

    idx0   = torch.tensor(selected_indices_layer0, dtype=torch.long, device=device)

    idx2   = torch.tensor(selected_indices_layer2, dtype=torch.long, device=device)

    def avg(a, b):
        return (a + b) / 2

    with torch.no_grad():
        # Layer 0 — rows by layer0 indices
        big.linear_relu_stack[0].weight[idx0] = avg(big.linear_relu_stack[0].weight[idx0], model_small.linear_relu_stack[0].weight)
        big.linear_relu_stack[0].bias[idx0]   = avg(big.linear_relu_stack[0].bias[idx0],   model_small.linear_relu_stack[0].bias)

        # Layer 2 — rows by layer2 indices, columns by layer0 indices
        big.linear_relu_stack[2].weight[idx2[:, None], idx0] = avg(
            big.linear_relu_stack[2].weight[idx2[:, None], idx0],
            model_small.linear_relu_stack[2].weight
        )
        big.linear_relu_stack[2].bias[idx2] = avg(big.linear_relu_stack[2].bias[idx2], model_small.linear_relu_stack[2].bias)

        # Layer 4 — columns by layer2 indices
        big.linear_relu_stack[4].weight[:, idx2] = avg(big.linear_relu_stack[4].weight[:, idx2], model_small.linear_relu_stack[4].weight)

    return big




def get_neuron_importance(fisher_dict, calculation_argument):
    """
    

    Args:
        fisher_dict:            
        calculation_argument:   

    Returns:
        
    """

    w0 = fisher_dict["linear_relu_stack.0.weight"]
    b0 = fisher_dict["linear_relu_stack.0.bias"]
    w2 = fisher_dict["linear_relu_stack.2.weight"]
    b2 = fisher_dict["linear_relu_stack.2.bias"]
    w4 = fisher_dict["linear_relu_stack.4.weight"]


    if calculation_argument == "sum":
        layer0_scores = w0.sum(dim=1) + b0 + w2.sum(dim=0)
        layer2_scores = w2.sum(dim=1) + b2 + w4.sum(dim=0)


    elif calculation_argument == "mean":
        layer0_scores = w0.mean(dim=1) + b0 + w2.mean(dim=0)
        layer2_scores = w2.mean(dim=1) + b2 + w4.mean(dim=0)

    else:
        raise ValueError(
            f"Unknown calculation argument: '{calculation_argument}'. "
            f"Choose from: 'sum', 'mean'"
        )

    return layer0_scores, layer2_scores










