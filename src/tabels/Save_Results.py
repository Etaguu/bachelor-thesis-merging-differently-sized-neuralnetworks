import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"



def load_specific_results(results_dir, file_names):
    """
    Loads experimental results from pkl-files
    
    Args:
        results_dir: directory of the pkl-files    
        files_names: list[str] of pkl-files names that will be loaded
    
    
    Returns:
        list of dicts with keys:
            "filename": str, filename
            "data": contents of the pkl file   
    """
    specific_data = []

    for file_name in file_names:
        file = results_dir / file_name

        if not file.exists():
            print(f"Warning: File not found: {file_name}")
            continue

        print(f"Loading: {file.name}")

        with open(file, "rb") as f:
            data = pickle.load(f)

        specific_data.append({
            "filename": file.name,
            "data": data
        })

    return specific_data




   