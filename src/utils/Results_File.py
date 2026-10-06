import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def save_results(filename, results, specialist_results):
    """
    Saves experiment results to a pkl file.

    Args:
        filename:           path to the output file
        results:            dict which contains the merged model accuracies
        specialist_results: dict which contains the specialist model accuracies

    Returns:
        None
    """

    path = PROJECT_ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "results": results,
        "specialist_results": specialist_results,
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_results(filename):
    """
    Loads experiment results from a pkl file.

    Args:
        filename:  path to the output file 

    Returns:
        dict with the data
    """


    with open(filename, "rb") as f:
        data = pickle.load(f)

    print(f"Results loaded from {filename}")
    return data