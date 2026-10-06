from model.Create_Neural_Network import *
from utils.Seeding import set_seed
from utils.Results_File import save_results
from merging.Fisher_Information_approximation_Library import estimate_fisher_approx
from merging.Average_Merging import merge_weights_simple
from merging.Fisher_Neuron_Select_Strategy import select_neurons, initialize_a_model_with_big_model_weights
from merging.Fisher_Information_Hessian_Library import get_hessian_matrix_library
from merging.Fisher_Merging import merge_fisher_approx


def train_models(
    models,
    train_loaders,
    keys,
    epochs,
    learn_rate,
    skip_first=False,
):
    """
    Trains multipile neural networks with the same hyperparameters. 

    Args:
        models:        List of neural networks to be trained
        train_loaders: List of Dataloaders with training data
        keys:          List of custom name visible during training process
        epochs:        Number of training epochs
        learn_rate:    Learning rate for the optimizer
        skip_first:    If True, skips training the first model
                       useful when the first model was already trained
                       in a factory function before this call

    Returns:
        list of copies of the trained models 
    """
    
    trained_models = []

    for i, (model, loader, key) in enumerate(
        zip(models, train_loaders, keys)
    ):
        should_skip = skip_first and i == 0
        print(f"i={i}, key={key}, skip={should_skip}")
        # Skip first model if it was already trained externally
        if not (skip_first and i == 0):
            train(model, loader, epochs, key, learn_rate)

        trained_models.append(clone_model(model))

    return trained_models


def evaluate_models(
    models,
    test_loaders,
    keys,
    results,
):
    """
    Evaluates mulipile trained neural networks and stores accuracies.
    
    Args:
        models:       List of neuralnetworks to be evaluated
        test_loaders: Either a list of DataLoaders (one per class/digit)
                      for class-wise evaluation, or a single DataLoader
                      for overall accuracy evaluation of the test data 
        keys:         List of names matching the model, is used as keys in results
        results:      Dict to store accuracies
                      For class-wise: results[key][digit].append(acc)
                      For overall:    results[key].append(acc)

    Returns:
        None: results dict is modified in place    
    """

    for model, key in zip(models, keys):

        # Class-wise evaluation     
        if isinstance(test_loaders, list):
            for digit, loader in enumerate(test_loaders):
                acc = evaluate(model, loader, key)
                results[key][digit].append(acc)

        # Overall evaluation
        else:
            acc = evaluate(model, test_loaders, key)
            results[key].append(acc)


def merge_and_evaluate(
    trained_models,
    train_datasets,
    test_loaders,
    merge_keys,
    results,
    fisher_method,
    fisher_n,
):
    """
    Merges trained neural networks and evaluates the merged models.

    Args:
        trained_models: List of neural networks, that are being merged
        train_datasets: List of raw training datasets, used for calculating Fisher Information,
                        train_dataset[i] must match trained_models[i]
        test_loaders:   Single Dataloader or a list of Dataloaders for evaluation
        merge_keys:     List of keys that are used in the result dictionary e.g. ["Average Merging", "Fisher Merging"]
        results:        Dict, stores the accuracies of the merged model
        fisher_method:  Determines, which Fisher Calculation is used e.g. "hessian_matrix" or "diagonal_fisher_matrix"
        fisher_n:       Determines, how many training samples are used for calculating the Fisher Information Matrix

    Returns:
        None: results dict is modified in place
    """
    
    # Simple average merge
    merged_average = merge_weights_simple(trained_models)

    # Fisher weighted merge
    merged_fisher = fisher_merging(
        method_arg=fisher_method,
        models=trained_models,
        datasets=train_datasets,
        fisher_n=fisher_n,
    )

    # Evaluate
    evaluate_models(
        [merged_average, merged_fisher],
        test_loaders,
        merge_keys,
        results,
    )


def run_experiment(
    model_factory,
    model_classes,
    train_raw,
    train_loaders,
    test_loaders,
    seeds,
    specialist_results,
    merge_results,
    epochs=10,
    learn_rate=1e-3,
    diff_size=False,
    fisher_n=1000,
    fisher_information_calculation_method="diagonal_fisher_matrix",
    save_path=None,
    **factory_kwargs,
):
    """
    Runs a merging experiment, where two neural networks are being trained, evaluated and merged with different configurations.
    The accuracies of the specialist neural networks and corresponding merged model are being saved to a pkl-file. 

    Args;    
        model_factory:         Function that creates and returns the models, handles the initalization and trimming
                               possible values: 
                               -create_same_size_same_init,
                               -create_same_size_random_init, 
                               -create_trimm_train_different_size_same_initialization,
                               -create_trimm_train_different_size_different_initialization

        model_classes:         List of models, that are being merged
                               the order matters, the big model must be first

        train_raw:             List of raw datasets (not Dataloaders), used for calculating Fisher Information
                               train_raw[i] must match train_loaders[i]

        train_loaders:         List of training Dataloaders used for the models
        test_loaders:          Single DataLoader or list of DataLoaders with test data
                               if list, evaluation is class-wise
        seeds:                 list of seeds so every experiment is run on the same set of seeds for reproducibility 
                               experiment runs once per seed

        specialist_results:    Dict, where the accuracies of the neural network pre-merging will be saved (specialists)
                               modified in place

        merge_results:         Dict, where the accuracies of the merge methods are saved for the merged model
                               modified in place
        epochs:                Epochs used for training the neural networks (default 10)
        learn_rate:            Learning rate used for the optimizer (default 1e-3)
        diff_size:             If True the models have different sizes
                               and the bigger neural network has to be trimmed down,
                               if False standard same-size merging (default False)
        fisher_n:              Determines, how many training samples are used for calculating the 
                               Fisher Information Matrix (default 1000)

        fisher_information_calculation_method: Determines, which Fisher Information calculation method is
                               used e.g. "hessian_matrix" or "diagonal_fisher_matrix"
        save_path:             Path to save results as pkl file (default None)
        **factory_kwargs:      Additional keyword arguments passed to the model factory function

    Returns:
        None: results dict is modified in place                                            
    """

    specialist_keys = list(specialist_results.keys())
    merge_keys = list(merge_results.keys())

    for seed in seeds:

        set_seed(seed)

        # Create models via factory
        # diff_size factories handle training the big model internally 
        if diff_size:
            models, model_big_full = model_factory(
                model_classes=model_classes,
                train_raw=train_raw,
                train_loaders=train_loaders,
                fisher_n=fisher_n,
                epochs=epochs,
                learn_rate=learn_rate,
                **factory_kwargs,
            )
        else:
            models = model_factory(
                model_classes=model_classes,
                train_raw=train_raw,
                train_loaders=train_loaders,
                fisher_n=fisher_n,
                **factory_kwargs,
            )

        # Train specialists
        trained_models = train_models(
            models,
            train_loaders,
            ["Model Big (Trimmed)", "Model Small"],  
            epochs,
            learn_rate,
            skip_first=diff_size,
        )

        # Evaluate specialists
        evaluate_models(
            trained_models,
            test_loaders,
            ["Model Big (Trimmed)", "Model Small"],
            specialist_results,
        )

        if "Model Big (Full)" in specialist_results:
            if isinstance(test_loaders, list):
                for digit, loader in enumerate(test_loaders):
                    specialist_results["Model Big (Full)"][digit].append(
                        evaluate(model_big_full, loader, "Big Model Full")
                    )
            else:
                specialist_results["Model Big (Full)"].append(
                    evaluate(model_big_full, test_loaders, "Big Model Full")
                )

        # Merge specialists and evaluate merged models
        merge_and_evaluate(
            trained_models,
            train_raw,
            test_loaders,
            merge_keys,
            merge_results,
            fisher_information_calculation_method,
            fisher_n,
        )

    # Save results to pkl file
    if save_path:
        save_results(
            save_path,
            merge_results,
            specialist_results,
        )
    print(f"Results saved to {save_path}")    


def fisher_merging(
    method_arg,
    models,
    datasets,
    fisher_n,
):
    """
    Executes a Fisher Merge of the passed models and determines which Fisher Information calculation method to use. 
    BOTH methods use only the diagonal of the Fisher Information Matrix for merging. They differ in how that diagonal
    is estimated.

    Args: 
        method_arg:     Determines which diagonal estimation to use for calculating the Fisher Information
                        possible values:
                        "diagonal_fisher_matrix": Diagonal approximation, fast, scales to large networks
                        "hessian_matrix":         Diagonal extracted from the full Fisher Information Matrix via laplace-torch,
                                                  very slow  

        models:         List of models that are being merged 
        datasets:       List of raw datasets that are being used to caclulate the Fisher Information
                        datasets[i] must match models[i]
        fisher_n:       Determines, how many training samples are used for calculating the 
                        Fisher Information Matrix
    
    Returns:
        merged model: the result of diagonal Fisher Merging 
    """

    if method_arg == "diagonal_fisher_matrix":

        # Call a library function to calculate an diagonal approximation of the Fisher Matrix
        # They are various estimations to choose from but "all" is the most accurate 
        fishers = [
            estimate_fisher_approx(
                model,
                dataset,
                "all",
                fisher_n,
            )
            for model, dataset in zip(models, datasets)
        ]

        return merge_fisher_approx(models,fishers)

    elif method_arg == "hessian_matrix":

        fishers = [
            get_hessian_matrix_library(
                model,
                dataset,
                fisher_n,
            )
            for model, dataset in zip(models, datasets)
        ]

        return merge_fisher_approx(models, fishers)

    else:
        raise ValueError(
            f"Unknown Fisher method: {method_arg}"
        )


def create_same_size_same_init(model_classes, **kwargs):
    """
    Factory function to create two models with the same initalization

    Args: 
        model_classes: List with the models that need to be created
    
    Returns:
        List of the created models
    """

    model_a = model_classes[0]().to(DEVICE)
    model_b = model_classes[1]().to(DEVICE)

    model_b.load_state_dict(copy.deepcopy(model_a.state_dict()))

    return [model_a, model_b]

def create_same_size_random_init(model_classes, **kwargs):
    """
    Factory function to create two models with different random initalizations. 
    Assign independent random weights to each model for experimentation.

    Args: 
        model_classes: List with the models that need to be created
    
    Returns:
        List of the created models, different initialization
    """

    model_a = model_classes[0]().to(DEVICE)
    model_b = model_classes[1]().to(DEVICE)

    return [model_a, model_b]


def create_trimm_train_different_size_same_initialization(
    model_classes,
    train_raw,
    train_loaders,
    fisher_n,
    strategy,
    num_neurons,
    epochs=10,
    learn_rate=1e-3,
    **kwargs
):
    """
    Factory function, creates and trains a large model, select certain neurons from it according to
    a strategy (Fisher-guided or random) and initialize a small model with the selected neurons as starting
    weights. The big model is trained on task 1 and is trimmed down to match the size of the small model.
    The small model will be trained on a different task, but is initialized with weights from the trimmed big model.

    Args: 
        model_classes:      list of model classes [BigModel, SmallModel]
                            big model must be first
        train_raw:          list of raw datasets for Fisher estimation
                            train_raw[0] is used for the big model    
        train_loaders:      list of DataLoaders for training
                            train_loaders[0] is used to train the big model
        fisher_n:           Determines, how many training samples are used for calculating the 
                            Fisher Information Matrix
        strategy:           neuron selection strategy
                            possible values: "random", "fisher_top", "fisher_bad"
        num_neurons:        number of neurons to select from the big model
        epochs:             training epochs for the big model (default 10)
        learn_rate:         learning rate for training (default 1e-3)
    
    Returns:
        list [big_model, trimmed_big, model_small]
        big_model:    mbig model, trained on task 1
        trimmed_big:  big models selected neurons in small model architecture,
                      already trained on task 1
        model_small:  small model initialized from same selected neurons
                      not trained 

    """

    # Create and train big model on task 1 (Index 0)
    model_big = model_classes[0]().to(DEVICE)
    train(model_big, train_loaders[0], epochs, "Big Model",learn_rate)

    model_big_full = clone_model(model_big)

    # Select neurons from trained big model
    # ignore return value of fisher dict
    selected_indices_layer_0, selected_indices_layer_2, _ = select_neurons(
        model_big=model_big,
        data_set_raw=train_raw[0],
        fisher_n=fisher_n,
        select_neuron_strategy=strategy,
        num_neurons=num_neurons,
    )

    # Create a trimmed model with the sane size as model_small
    trimmed_big = model_classes[1]().to(DEVICE)

    # Copy selected neurons from the big model according to the chosen indices
    # Already trained
    trimmed_big = initialize_a_model_with_big_model_weights(
        trimmed_big,
        model_big,
        selected_indices_layer_0,
        selected_indices_layer_2
    )

    # Create small model
    model_small = model_classes[1]().to(DEVICE)

    # Initialize small model with selected big-model neurons
    # Not trained yet
    model_small = initialize_a_model_with_big_model_weights(
        model_small,
        model_big,
        selected_indices_layer_0,
        selected_indices_layer_2
    )

    return [trimmed_big, model_small], model_big_full

def create_trimm_train_different_size_different_initialization(
    model_classes,
    train_raw,
    train_loaders,
    fisher_n,
    strategy,
    num_neurons,
    epochs=10,
    learn_rate=1e-3,
    **kwargs
):
    """
    Factory function, creates and trains a large model, select certain neurons from it according to
    a strategy (Fisher-guided or random). Then randomly initalize a small model.
    The big model is trained on task 1 and is trimmed down to match the size of the small model.
    The small model will be trained on a different task with a random initlaization.


    Args: 
        model_classes:      list of model classes [BigModel, SmallModel]
                            big model must be first
        train_raw:          list of raw datasets for Fisher estimation
                            train_raw[0] is used for the big model    
        train_loaders:      list of DataLoaders for training
                            train_loaders[0] is used to train the big model
        fisher_n:           Determines, how many training samples are used for calculating the 
                            Fisher Information Matrix
        strategy:           neuron selection strategy
                            possible values: "random", "fisher_top", "fisher_bad"
        num_neurons:        number of neurons to select from the big model
        epochs:             training epochs for the big model (default 10)
        learn_rate:         learning rate for training (default 1e-3)
    
    Returns:
        list [trimmed_big, model_small]
        model_big:    big model trained on task 1  
        trimmed_big:  big models selected neurons in small model architecture,
                      already trained on task 1
        model_small:  small model, randomly initialized
                      not trained 

    """

    # Train original big model
    model_big = model_classes[0]().to(DEVICE)

    train(model_big, train_loaders[0], epochs, "Big Model", learn_rate)

    model_big_full = clone_model(model_big)  

    # Select neurons according to a strategy
    selected_indices_layer_0,  selected_indices_layer_2, _ = select_neurons(
        model_big=model_big,
        data_set_raw=train_raw[0],
        fisher_n=fisher_n,
        select_neuron_strategy=strategy,
        num_neurons=num_neurons,
    )

    # Create trimmed model with the same architecture as model_small
    trimmed_big = model_classes[1]().to(DEVICE)

    # Copy selected neurons from big model
    trimmed_big = initialize_a_model_with_big_model_weights(
        trimmed_big,
        model_big,
        selected_indices_layer_0,
        selected_indices_layer_2
    )

    # Independently initialize a second small model
    model_small = model_classes[1]().to(DEVICE)

    return [trimmed_big, model_small], model_big_full