from utils.Seeding import set_seed
from merging.Fisher_Neuron_Select_Strategy import *
from utils.Results_File import save_results
from utils.Seeding import set_seed


# FIXME Still needs some work
def run_experiment_fisher(
    model_classes,
    train_raw,
    train_loaders,
    test_loaders,
    seeds,
    specialist_results,
    results,
    epochs=10,
    learn_rate=1e-3,
    fisher_n=1000,
    neuron_select_strategy="random",
    merge_method="replace",
    save_path=None,
    neuron_importance_calculation_arg="sum"
):
    
    """
    Runs a merging experiment, where two neural networks with different sizes are being trained, evaluated and
    merged with different configurations. Instead of trimming the big model down, half the neurons are being chosen
    according to a strategy from the two hidden layers, used for initalizing the small model, which has half as many neurons
    in its hidden layers and then merged back into the big model at the same index. This merging can be either a
    complete replacement, average or Fisher Merging for each neuron that was selected. 

    Args:    
        model_classes:          List of models, that are being merged
                                the order matters, the big model must be first

        train_raw:              List of raw datasets (not Dataloaders), used for calculating Fisher Information
                                train_raw[i] must match train_loaders[i]

        train_loaders:          List of training Dataloaders used for the models

        test_loaders:           List of DataLoaders with test data

        seeds:                  List of seeds so every experiment is run on the same set of seeds for reproducibility 
                                experiment runs once per seed

        specialist_results:     Dict, where the accuracies of the neural network pre-merging will be saved (specialists)
                                modified in place

        results:                Dict, where the accuracies of the merge methods are saved for the merged model
                                modified in place

        epochs:                 Epochs used for training the neural networks (default 10)

        learn_rate:             Learning rate used for the optimizer (default 1e-3)

        fisher_n:               Determines, how many training samples are used for calculating the 
                                Fisher Information Matrix (default 1000)

        neuron_select_strategy: Determines the strategy with that certain neurons are chosen from the two hidden layers of the model
                                (default: "random")
                                possible values: "random", "fisher_worst", "fisher_top"   

        merge_method:           Merge methods that is used to merge the weights of the neurons back into the big model
                                at the same position (default: replace)
                                possible values: "replace", "fisher", "average", "fisher_to_other_psotition", "replace_to_other_position"

        save_path:              Path to save results as pkl file (default None)
        
        neuron_importance_calculation_arg: Determines how to calculate the Fisher Infromation importance of the selected neurons
                                           the selected indices are used in the approximated Fisher Matrix to either sum up all 
                                           or to take the mean of all the Fisher Information belonging to a neuron (default: sum)
                                           possible values: "sum", "mean"                          
                                
    Returns:
        None: results dict is modified in place                                            
    """

    specialist_keys = list(specialist_results.keys())
    merge_keys = list(results.keys())

    for seed in seeds:

        set_seed(seed)

        model_big = model_classes[0]().to(DEVICE)
        model_small = model_classes[1]().to(DEVICE)

        # Train big model on dataset

        train(
            model_big,
            train_loaders[0],
            epochs,
            specialist_keys[0],
            learn_rate
        )

        # Evaluate big specialist on Fashion

        acc = evaluate(
            model_big,
            test_loaders[0],
            specialist_keys[0],
        )
        specialist_results[specialist_keys[0]]["fashion"].append(acc)

        # Evaluate big specialist on MNIST

        acc = evaluate(model_big, test_loaders[1], specialist_keys[0])

        specialist_results[specialist_keys[0]]["mnist"].append(acc)

        # Select neurons according to size of small model
        num_neurons = model_small.linear_relu_stack[0].out_features

        # Selection is based on the strategy
        selected_indices_1 , selected_indices_2, fisher_matrix_big = select_neurons(
            model_big=model_big,
            data_set_raw=train_raw[0],
            fisher_n=fisher_n,
            select_neuron_strategy=neuron_select_strategy,
            num_neurons=num_neurons,
            neuron_importance_calculation_arg=neuron_importance_calculation_arg
        )

        model_small_random = model_classes[1]().to(DEVICE)

        # Evaluate Fashion before initializing 

        acc = evaluate(
            model_small_random,
            test_loaders[0],
            specialist_keys[2],
        )

        specialist_results[specialist_keys[2]]["Fashion (Pretraining)"].append(acc)

        # Evaluate MNIST after training with random init
        train(model_small_random, train_loaders[1], epochs, "Small Model Random Init", learn_rate)

        acc = evaluate(
            model_small_random,
            test_loaders[1],
            specialist_keys[0],
        )

        specialist_results[specialist_keys[2]]["MNIST: (random init)"].append(acc)


        # Initialize small model with weights of big model
        # The indices are as big as the hidden layer of the small model

        model_small = initialize_a_model_with_big_model_weights(
            model_small,
            model_big,
            selected_indices_1,
            selected_indices_2
        )

        # Train small model on MNIST

        train(
            model_small,
            train_loaders[1],
            epochs,
            specialist_keys[1],
            learn_rate
        )

        # After training small model, save its output head
        mnist_head = copy.deepcopy(model_small.linear_relu_stack[4])

        # Evaluate small specialist on MNIST

        acc = evaluate(
            model_small,
            test_loaders[1],
            specialist_keys[1],
        )

        specialist_results[specialist_keys[1]]["mnist"].append(acc)


        # Evaluate small specialist on Fashion

        acc = evaluate(
            model_small,
            test_loaders[0],
            specialist_keys[1],
        )

        specialist_results[specialist_keys[1]]["fashion"].append(acc)

        # Merge selected neurons in big model according to merge

        merged, not_selected_layer0, not_selected_layer2 = merge_neurons_back_into_big_model(
            model_big, model_small,
            selected_indices_1, selected_indices_2,
            merge_method, train_raw, fisher_n, fisher_matrix_big,
        )


        # When switchting the position, we dont want other source layer
        # For switched position strategies, pass not_selected indices as destination
        if merge_method in ("fisher_to_not_selected_neurons", "replace_to_not_selected_neurons"):
            destination_layer0 = not_selected_layer0
            destination_layer2 = not_selected_layer2
        else:
            destination_layer0 = None
            destination_layer2 = None


        # Evaluate merged model (With mask and without)

        # With Mask:
        fashion_acc = evaluate_with_task_head(
            merged, test_loaders[0],
            task_head=None,
            selected_indices_layer0=selected_indices_1,
            selected_indices_layer2=selected_indices_2,
            task_name="fashion"
        )

        results[merge_keys[1]]["fashion"].append(fashion_acc)

        mnist_acc = evaluate_with_task_head(
            merged, test_loaders[1],
            task_head=mnist_head,
            selected_indices_layer0=selected_indices_1,
            selected_indices_layer2=selected_indices_2,
            task_name="mnist",
            destination_indices_layer0=destination_layer0,
            destination_indices_layer2=destination_layer2,
        )

        results[merge_keys[1]]["mnist"].append(mnist_acc)

        acc = evaluate(
            merged,
            test_loaders[1],
            merge_keys[0],
        )

        results[merge_keys[0]]["mnist"].append(acc)

        acc = evaluate(
            merged,
            test_loaders[0],
            merge_keys[0],
        )

        results[merge_keys[0]]["fashion"].append(acc)

        print(f"Seed {seed} done")

    # Save experiment
    if save_path:
        save_results(
            save_path,
            results,
            specialist_results,
        )
    print(f"Results saved to {save_path}")        


def evaluate_with_task_head(
    model,
    dataloader,
    task_head,
    selected_indices_layer0,
    selected_indices_layer2,
    task_name,
    destination_indices_layer0=None,  
    destination_indices_layer2=None,  
):
    """
    Evaluates a merged model using task-specific output heads.

    For the standard setup destination indices equal selection indices.
    For the switched position experiment, MNIST weights are merged at
    different positions than where they were selected from — pass the
    destination positions explicitly so the mask isolates the correct neurons.

    Args:
        model:                          merged neural network to evaluate
        dataloader:                     DataLoader with test data
        task_head:                      MNIST output head from small model
                                        pass None for Fashion evaluation
        selected_indices_layer0:        indices used to select and initialize
                                        the small model from the big model
        selected_indices_layer2:        indices used to select and initialize
                                        the small model from the big model
        task_name:                      "fashion" or "mnist"
        destination_indices_layer0:     indices where MNIST weights were merged back
                                        defaults to selected_indices_layer0 if None
        destination_indices_layer2:     indices where MNIST weights were merged back
                                        defaults to selected_indices_layer2 if None

    Returns:
        accuracy as a float between 0 and 1
    """
    model.eval()
    if task_head is not None:
        task_head.eval()
    correct = 0
    total   = 0

    device = next(model.parameters()).device

    # Use destination indices for mask — fall back to selection indices if not provided
    mask_idx0 = destination_indices_layer0 if destination_indices_layer0 is not None else selected_indices_layer0
    mask_idx2 = destination_indices_layer2 if destination_indices_layer2 is not None else selected_indices_layer2

    idx0 = torch.tensor(mask_idx0, dtype=torch.long, device=device)
    idx2 = torch.tensor(mask_idx2, dtype=torch.long, device=device)

    # Binary masks — 1 at destination positions, 0 elsewhere
    mask_layer0 = torch.zeros(model.linear_relu_stack[0].out_features, device=device)
    mask_layer0[idx0] = 1.0

    mask_layer2 = torch.zeros(model.linear_relu_stack[2].out_features, device=device)
    mask_layer2[idx2] = 1.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            x = model.flatten(X)

            x = model.linear_relu_stack[0](x)
            x = model.linear_relu_stack[1](x)

            if task_name == "mnist":
                x = x * mask_layer0

            x = model.linear_relu_stack[2](x)
            x = model.linear_relu_stack[3](x)

            if task_name == "fashion":
                logits = model.linear_relu_stack[4](x)

            elif task_name == "mnist":
                x = x * mask_layer2
                x_selected = x[:, idx2]
                logits      = task_head(x_selected)

            pred     = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)

    return correct / total
