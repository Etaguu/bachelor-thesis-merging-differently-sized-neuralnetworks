from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# path to bachelor-code/data
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_dataset(dataset_name="MNIST"):
    """
    Loads a dataset and applies normalization.

    Args:
        dataset_name:   name of the dataset to load
                        possible values: "MNIST", "FASHION_MNIST"

    Returns:
        tuple (train_dataset, test_dataset), raw datasets
        suitable for passing to Fisher estimation
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == "FASHION_MNIST":
        dataset_cls = datasets.FashionMNIST
    else:
        dataset_cls = datasets.MNIST

    train_dataset = dataset_cls(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = dataset_cls(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset

def get_task_loaders(batch_size=64, dataset_name="MNIST",do_shuffle=True):
    """
    Creates DataLoaders for the full train and test dataset.

    Args:
        batch_size:     number of samples per batch (default 64)
        dataset_name:   name of the dataset to load
                        possible values: "MNIST", "FASHION_MNIST"
        do_shuffle:     whether to shuffle the training data (default True)

        
    Returns:
        tuple (train_complete_loader, test_complete_loader)
    """


    train_data, test_data = load_dataset(dataset_name)

    train_complete_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=do_shuffle,
        )
        
    test_complete_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False)

    return train_complete_loader, test_complete_loader


def get_task_loaders_cl(batch_size=64, dataset_name="MNIST"):
    """
    Creates classwise DataLoaders for potential continual learning experiments.
    Each class gets its own train and test DataLoader. Was never used for CL

    Args:
        batch_size:     number of samples per batch (default 64)
        dataset_name:   name of the dataset to load
                        possible values: "MNIST", "FASHION_MNIST"

    Returns:
        tuple (task_train_sets, task_test_sets, task_train_loaders, task_test_loaders)
        each is a list of length 10 — one entry per digit class (0-9)
    """

    train_dataset, test_dataset = load_dataset(dataset_name)

    task_train_sets = []
    task_test_sets = []
    task_train_loaders = []
    task_test_loaders = []

    for n in range(10):
        train_subset = split_by_class(train_dataset, [n])
        test_subset = split_by_class(test_dataset, [n])

        task_train_sets.append(train_subset)
        task_test_sets.append(test_subset)

        task_train_loaders.append(
            DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        )

        task_test_loaders.append(
            DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        )

    return task_train_sets, task_test_sets, task_train_loaders, task_test_loaders

#  Returns a subset of the datasets with certain classes
def split_by_class(dataset, classes):
    """
    Creates a subset of a dataset containing only samples from the specified classes.

    Args:
        dataset:    Dataset to filter
        classes:    list of class labels that are included

    Returns:
        dataset containing only the specified classes
    """
    selected_indices = []
    for index, (image, label) in enumerate(dataset):
        if label in classes:
            selected_indices.append(index)
    return Subset(dataset, selected_indices)

