import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabels.Save_Results import *
import matplotlib.patches as mpatches
import textwrap

RESULTS_DIR = PROJECT_ROOT / "results"

#FIXME Clean UP This


def format_class_result(class_results):
    # 10 classes × 5 seeds
    values = np.array(list(class_results.values()))

    # Average the 10 classes for each seed
    per_seed = values.mean(axis=0)

    # Mean and std across the 5 seeds
    mean = per_seed.mean()
    std = per_seed.std(ddof=1)

    return f"{mean * 100:.2f} ± {std * 100:.2f}%"


def format_result(values):
    # Handle class-wise dict {0: [seeds], 1: [seeds], ...}
    if isinstance(values, dict):
        all_values = np.concatenate([values[d] for d in range(10)])
    else:
        all_values = np.array(values)
    
    mean = np.mean(all_values)
    std  = np.std(all_values, ddof=1)
    return f"{mean * 100:.2f} ± {std * 100:.2f}%"

def plot_bar_comparison(files, experiment_names, column_names=None, column_order=None):
    
    all_data = []
    
    for filename, experiment_name in zip(files, experiment_names):
        path = RESULTS_DIR / filename
        with open(path, "rb") as f:
            data = pickle.load(f)

        combined = {
            **data["specialist_results"],
            **data["results"]
        }

        if column_order is not None:
            ordered = [name for name in column_order if name in combined]
        else:
            ordered = list(combined.keys())

        for method_name in ordered:
            values = combined[method_name]
            if isinstance(values, dict):
                flat_values = np.concatenate([values[k] for k in values.keys()])
            else:
                flat_values = np.array(values)

            all_data.append({
                "Experiment": experiment_name,
                "Method":     method_name,
                "Mean":       np.mean(flat_values) * 100,
                "Std":        np.std(flat_values, ddof=1) * 100,
            })

    df = pd.DataFrame(all_data)

    experiments = df["Experiment"].unique()

    # Preserve order from column_order
    if column_order is not None:
        methods = [m for m in column_order if m in df["Method"].values]
    else:
        methods = list(dict.fromkeys(df["Method"].tolist()))

    n_experiments = len(experiments)
    n_methods     = len(methods)

    # Display names
    if column_names is not None:
        display_names = column_names[1:n_methods + 1]
    else:
        display_names = methods

    fig, axes = plt.subplots(1, n_experiments, figsize=(5 * n_experiments, 5), sharey=True)
    if n_experiments == 1:
        axes = [axes]

    colors = ["#5b9bd5", "#70ad47", "#ed7d31", "#9b59b6", "#e74c3c"]
    x      = np.arange(n_methods)
    width  = 0.6

    for ax, experiment in zip(axes, experiments):
        exp_data = df[df["Experiment"] == experiment]

        means = [exp_data[exp_data["Method"] == m]["Mean"].values[0] for m in methods]
        stds  = [exp_data[exp_data["Method"] == m]["Std"].values[0]  for m in methods]

        bars = ax.bar(
            x, means, width,
            yerr=stds,
            capsize=5,
            color=colors[:n_methods],
            edgecolor="white",
            error_kw=dict(linewidth=1.5)
        )

        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.5,
                f"{mean:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        ax.set_title(experiment, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        wrapped = [textwrap.fill(name, width=12) for name in display_names]
        ax.set_xticklabels(wrapped, rotation=0, ha="center", fontsize=9)
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "")
        ax.set_ylim(0, 110)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig

def create_comparison_table_from_files(
    files,
    experiment_names,
    column_names=None,
    column_order=None,
):
    if len(files) != len(experiment_names):
        raise ValueError("files and experiment_names must have the same length")

    # Get all available keys from first file
    first_path = RESULTS_DIR / files[0]
    with open(first_path, "rb") as f:
        first_data = pickle.load(f)

    first_combined = {
        **first_data["specialist_results"],
        **first_data["results"]
    }

    # Determine column order from data
    if column_order is not None:
        ordered_names = [n for n in column_order if n in first_combined]
    else:
        ordered_names = list(first_combined.keys())

    # Determine column labels
    if column_names is not None:
        # Use only as many labels as we have data columns
        _col_labels = column_names[:len(ordered_names) + 1]
    else:
        _col_labels = ["Experiment"] + ordered_names

    # Build rows
    rows = []
    for filename, experiment_name in zip(files, experiment_names):
        path = RESULTS_DIR / filename
        with open(path, "rb") as f:
            data = pickle.load(f)

        combined = {
            **data["specialist_results"],
            **data["results"]
        }

        row = [experiment_name]
        for name in ordered_names:
            row.append(format_result(combined[name]))
        rows.append(row)

    num_columns = len(_col_labels)
    fig_height  = max(1.0, 0.5 * (len(rows) + 1))

    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=_col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    table.auto_set_column_width(col=list(range(num_columns)))

    for col in range(num_columns):
        table[(0, col)].set_facecolor("#4C72B0")
        table[(0, col)].set_text_props(color="white", weight="bold")

    for row_idx in range(1, len(rows) + 1):
        color = "#FFFFFF" if row_idx % 2 == 0 else "#FFFFFF"
        for col in range(num_columns):
            table[(row_idx, col)].set_facecolor(color)

    return fig


def create_single_experiment_table_cls(files, experiment_names, col_labels=None):
    
    if isinstance(files, str):
        files = [files]
        experiment_names = [experiment_names]
    
    for filename, experiment_name in zip(files, experiment_names):
        path = RESULTS_DIR / filename
        with open(path, "rb") as f:
            data = pickle.load(f)

        big_model    = data["specialist_results"]["big_model"]
        small_model  = data["specialist_results"]["small_model"]
        small_random = data["specialist_results"].get("small_random", None)
        merged       = data["results"]["Merged Model"]

        def fmt(values):
            arr = np.array(values)
            return f"{arr.mean()*100:.2f} ± {arr.std(ddof=1)*100:.2f}%"

        fashion_row = ["Fashion", fmt(big_model["fashion"])]
        mnist_row   = ["MNIST",   fmt(big_model["mnist"])]

        if small_random is not None:
            fashion_row.append(fmt(small_random.get("Fashion (Pretraining)", small_random.get("fashion", []))))
            mnist_row.append(fmt(small_random.get("MNIST: (random init)", small_random.get("mnist", []))))

        fashion_row.append(fmt(small_model["fashion"]))
        mnist_row.append(fmt(small_model["mnist"]))

        fashion_row.append(fmt(merged["fashion"]))
        mnist_row.append(fmt(merged["mnist"]))
        rows = [fashion_row, mnist_row]

        # Default column labels
        if col_labels is None:
            _col_labels = ["Dataset", "Big Model", "Small Model", "Merged Model"]
            if small_random is not None:
                _col_labels = ["Dataset", "Big Model", "Small Model", "Small Random", "Merged Model"]
        else:
            _col_labels = col_labels

        num_columns = len(_col_labels)

        fig, ax = plt.subplots(figsize=(10, 1.3))
        ax.axis("off")
        ax.text(0.5, 1.02, experiment_name,
                transform=ax.transAxes,
                fontweight="bold", fontsize=11,
                ha="center", va="bottom")

        table = ax.table(
            cellText=rows,
            colLabels=_col_labels,
            loc="center",
            cellLoc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.7, 1.8)
        table.auto_set_column_width(col=list(range(num_columns)))

        for col in range(num_columns):
            table[(0, col)].set_facecolor("#4C72B0")
            table[(0, col)].set_text_props(color="white", weight="bold")

        for row_index in range(1, len(rows) + 1):
            color = "#EAF7EA" if row_index == 1 else "#EAF2F8"
            for col in range(num_columns):
                table[(row_index, col)].set_facecolor(color)

        plt.tight_layout(pad=0.1)
        plt.show()
        plt.close()


def create_cls_comparison_table(files, experiment_names, column_order=None, col_labels=None):

    def fmt(values):
        arr = np.array(values)
        return f"{arr.mean()*100:.1f} ± {arr.std(ddof=1)*100:.1f}%"

    all_rows = []
    has_small_random = False

    # Load all data first
    loaded = []
    for filename, experiment_name in zip(files, experiment_names):
        path = RESULTS_DIR / filename
        with open(path, "rb") as f:
            data = pickle.load(f)
        loaded.append((experiment_name, data))
        if data["specialist_results"].get("small_random") is not None:
            has_small_random = True

    # Build lookup for all possible columns
    for experiment_name, data in loaded:
        big          = data["specialist_results"]["big_model"]
        small        = data["specialist_results"]["small_model"]
        small_random = data["specialist_results"].get("small_random", None)
        no_mask      = data["results"]["Merged Model (No Mask)"]
        mask         = data["results"]["Merged Model (Mask)"]

        for dataset in ["fashion", "mnist"]:
            # Build full row dict for reordering
            row_dict = {
                "Experiment":       experiment_name if dataset == "fashion" else "",
                "Dataset":          dataset.upper(),
                "Big Model":        fmt(big[dataset]),
                "Small Model":      fmt(small[dataset]),
                "Merged (No Mask)": fmt(no_mask[dataset]),
                "Merged (Mask)":    fmt(mask[dataset]),
            }
            if small_random is not None:
                key = "Fashion (Pretraining)" if dataset == "fashion" else "MNIST: (random init)"
                row_dict["Small Random"] = fmt(small_random[key])

            # Apply column order
            if column_order is not None:
                row = [row_dict[c] for c in column_order if c in row_dict]
            else:
                default_order = ["Experiment", "Dataset", "Big Model", "Small Model"]
                if has_small_random:
                    default_order.append("Small Random")
                default_order += ["Merged (No Mask)", "Merged (Mask)"]
                row = [row_dict[c] for c in default_order]

            all_rows.append(row)

    # Column labels
    if col_labels is not None:
        _col_labels = col_labels
    elif column_order is not None:
        _col_labels = column_order
    else:
        _col_labels = ["Experiment", "Dataset", "Big Model", "Small Model"]
        if has_small_random:
            _col_labels.append("Small Random")
        _col_labels += ["Merged (No Mask)", "Merged (Mask)"]

    num_columns = len(_col_labels)
    fig_height  = max(0.5, 0.35 * (len(all_rows) + 1))

    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=all_rows,
        colLabels=_col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    table.auto_set_column_width(col=list(range(num_columns)))

    # Header
    for col in range(num_columns):
        table[(0, col)].set_facecolor("#2c3e50")
        table[(0, col)].set_text_props(color="white", weight="bold")

    # Color per experiment block
    colors = ["#EAF2F8", "#EAF7EA", "#FFF4E5"]
    for exp_idx in range(len(files)):
        color     = colors[exp_idx % len(colors)]
        row_start = exp_idx * 2 + 1
        for row_index in range(row_start, row_start + 2):
            for col in range(num_columns):
                table[(row_index, col)].set_facecolor(color)


    plt.tight_layout(pad=0)
    return fig

def plot_cls_comparison(files, experiment_names):

    with open(RESULTS_DIR / files[0], "rb") as f:
        first_data = pickle.load(f)

    big_fashion   = np.mean(first_data["specialist_results"]["big_model"]["fashion"])   * 100
    big_mnist     = np.mean(first_data["specialist_results"]["big_model"]["mnist"])     * 100
    small_fashion = np.mean(first_data["specialist_results"]["small_model"]["fashion"]) * 100
    small_mnist   = np.mean(first_data["specialist_results"]["small_model"]["mnist"])   * 100

    fashion_means, fashion_stds = [], []
    mnist_means,   mnist_stds   = [], []
    mask_means,    mask_stds    = [], []

    for filename in files:
        with open(RESULTS_DIR / filename, "rb") as f:
            data = pickle.load(f)

        no_mask = data["results"]["Merged Model (No Mask)"]
        mask    = data["results"]["Merged Model (Mask)"]

        fashion_means.append(np.mean(no_mask["fashion"]) * 100)
        fashion_stds.append(np.std(no_mask["fashion"], ddof=1) * 100)

        mnist_means.append(np.mean(no_mask["mnist"]) * 100)
        mnist_stds.append(np.std(no_mask["mnist"], ddof=1) * 100)

        mask_means.append(np.mean(mask["mnist"]) * 100)
        mask_stds.append(np.std(mask["mnist"], ddof=1) * 100)

    x     = np.arange(len(experiment_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_fashion = ax.bar(
        x - width, fashion_means, width,
        yerr=fashion_stds, capsize=4,
        color="#5b9bd5", label="Fashion", edgecolor="white"
    )
    bars_mnist = ax.bar(
        x, mnist_means, width,
        yerr=mnist_stds, capsize=4,
        color="#ed7d31", label="MNIST (No Mask)", edgecolor="white"
    )
    bars_mask = ax.bar(
        x + width, mask_means, width,
        yerr=mask_stds, capsize=4,
        color="#70ad47", label="MNIST (Mask)", edgecolor="white"
    )

    # Baseline lines
    ax.axhline(big_fashion, linestyle="--", color="#5b9bd5", linewidth=1.5)
    ax.axhline(small_mnist, linestyle="--", color="#ed7d31", linewidth=1.5)

    # Invisible line handles for legend
    line_big_fashion, = ax.plot([], [], linestyle="--", color="#5b9bd5", linewidth=1.5)
    line_small_mnist, = ax.plot([], [], linestyle="--", color="#ed7d31", linewidth=1.5)

    # Value labels
    for bars, means, stds in [
        (bars_fashion, fashion_means, fashion_stds),
        (bars_mnist,   mnist_means,   mnist_stds),
        (bars_mask,    mask_means,    mask_stds),
    ]:
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.5,
                    f"{mean:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Legend as horizontal line at top
    fig.legend(
        handles=[bars_fashion, bars_mnist, bars_mask, line_big_fashion, line_small_mnist],
        labels=[
            "Fashion",
            "MNIST (No Mask)",
            "MNIST (Mask)",
            f"Big Model Fashion ({big_fashion:.1f}%)",
            f"Small Model MNIST ({small_mnist:.1f}%)",
        ],
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        fontsize=8
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def create_accuracy_table(
    specialist_results,
    results,
):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")

    table_data = []

    # Specialist models
    for model, datasets in specialist_results.items():

        if isinstance(datasets, dict):
            # Structure:
            # {model: {dataset: [accuracies]}}
            for dataset, accuracies in datasets.items():
                table_data.append([
                    f"{model} ({dataset})",
                    mean_std(accuracies)
                ])

        else:
            # {model: [accuracies]}
            table_data.append([
                model,
                mean_std(datasets)
            ])

    # Remember how many specialist rows we created
    n_specialist = len(table_data)

    # Merging methods
    for method, datasets in results.items():

        if isinstance(datasets, dict):
            # Structure:
            # {method: {dataset: [accuracies]}}
            for dataset, accuracies in datasets.items():
                table_data.append([
                    f"{method} ({dataset})",
                    mean_std(accuracies)
                ])

        else:
            # Structure:
            # {method: [accuracies]}
            table_data.append([
                method,
                mean_std(datasets)
            ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Model / Method", "Accuracy (mean ± std)"],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.2)

    n_total = len(table_data)

    # Header
    for j in range(2):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(
            color="white",
            fontweight="bold"
        )

    # Specialist rows
    for i in range(1, n_specialist + 1):
        for j in range(2):
            table[i, j].set_facecolor("#d6eaf8")

    # Merge rows
    for i in range(n_specialist + 1, n_total + 1):
        for j in range(2):
            table[i, j].set_facecolor("#d5e8d4")
            table[i, j].set_text_props(fontweight="bold")

    plt.tight_layout()



def mean_std(values):
    values = np.asarray(values)
    return f"{values.mean():.1%} ± {values.std():.1%}"    


def plot_seed_distribution(results):
    data = []

    for method, datasets in results.items():
        for acc in datasets["mnist"]:
            data.append((method, acc))

    df = pd.DataFrame(data, columns=["Method", "Accuracy"])

    df.boxplot(
        column="Accuracy",
        by="Method",
        grid=False
    )

    plt.ylabel("Accuracy")
    plt.xlabel("")
    plt.title("Accuracy Distribution Across Seeds")
    plt.suptitle("")
    plt.show()

    
def display_results_table(filename, figsize=(16, 5)):
    
    path = RESULTS_DIR / filename 
    
    with open(path, "rb") as f:
        data = pickle.load(f)

    results     = data["results"]
    specialists = data["specialist_results"]
    all_methods = {**specialists, **results}

    row_labels = list(all_methods.keys())
    col_labels = [str(d) for d in range(10)] + ["Avg"]

    table_data = []
    for method, digit_dict in all_methods.items():
        row = []
        for digit in range(10):
            values = np.array(digit_dict[digit])
            row.append(f"{values.mean()*100:.1f}±{values.std()*100:.1f}")
        all_values = np.concatenate([digit_dict[d] for d in range(10)])
        row.append(f"{all_values.mean()*100:.1f}±{all_values.std()*100:.1f}")
        table_data.append(row)

    n_specialists = len(specialists)
    n_methods     = len(all_methods)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 2.2)

    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", weight="bold")

    for i in range(1, n_specialists + 1):
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor("#d6eaf8")
        tbl[(i, -1)].set_text_props(weight="bold")

    for i in range(n_specialists + 1, n_methods + 1):
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor("#d5e8d4")
        tbl[(i, -1)].set_text_props(weight="bold")

    for i in range(1, n_methods + 1):
        tbl[(i, len(col_labels) - 1)].set_facecolor("#f9e79f")

    plt.tight_layout()
    return fig


def show_results_table_basic(results, seeds):
    models = list(results.keys())

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")

    # Table data
    data = []

    for i, seed in enumerate(seeds):
        row = [str(seed)]
        row += [f"{results[model][i] * 100:.2f}%" for model in models]
        data.append(row)

    # Add mean and std
    mean_row = ["Mean"]
    std_row = ["Std"]

    for model in models:
        values = np.array(results[model])
        mean_row.append(f"{values.mean() * 100:.2f}%")
        std_row.append(f"{values.std() * 100:.2f}%")

    data.append(mean_row)
    data.append(std_row)

    # Create table
    table = ax.table(
        cellText=data,
        colLabels=["Seed"] + models,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    plt.title("Neural Network Accuracy Across Seeds", pad=20)
    plt.tight_layout()
    plt.show()


def show_results(filename):

    path = Path(RESULTS_DIR) / filename

    # Load results
    with open(path, "rb") as f:
        data = pickle.load(f)

    specialist_results = data["specialist results"]
    pre_results = data["pre results"]

    models = list(specialist_results.keys())
    labels = ["8 Neurons", "64 Neurons", "256 Neurons"]
    seeds = np.arange(5)

    # ============================================================
    # TABLE
    # ============================================================

    table_data = []

    for i, seed in enumerate(seeds):
        row = [f"Seed {seed}"]

        for model in models:
            row.append(f"{pre_results[model][i] * 100:.2f}%")

        for model in models:
            row.append(f"{specialist_results[model][i] * 100:.2f}%")

        table_data.append(row)

    # Mean ± std
    row = ["Mean ± Std"]

    for model in models:
        values = np.array(pre_results[model]) * 100
        row.append(f"{values.mean():.2f} ± {values.std():.2f}%")

    for model in models:
        values = np.array(specialist_results[model]) * 100
        row.append(f"{values.mean():.2f} ± {values.std():.2f}%")

    table_data.append(row)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=[
            "Seed",
            "Pre 8", "Pre 64", "Pre 256",
            "Specialist 8", "Specialist 64", "Specialist 256"
        ],
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    plt.title("MNIST Accuracy: Pre vs Specialist Models")
    plt.tight_layout()
    plt.show()


    # ============================================================
    # PLOT
    # ============================================================

    x = np.arange(len(models))
    width = 0.35

    pre_means = np.array([
        np.mean(pre_results[model]) * 100
        for model in models
    ])

    pre_stds = np.array([
        np.std(pre_results[model]) * 100
        for model in models
    ])

    specialist_means = np.array([
        np.mean(specialist_results[model]) * 100
        for model in models
    ])

    specialist_stds = np.array([
        np.std(specialist_results[model]) * 100
        for model in models
    ])

    fig, ax = plt.subplots(figsize=(9, 5))

    # Mean ± standard deviation
    ax.bar(
        x - width / 2,
        pre_means,
        width,
        yerr=pre_stds,
        capsize=5,
        label="Pre"
    )

    ax.bar(
        x + width / 2,
        specialist_means,
        width,
        yerr=specialist_stds,
        capsize=5,
        label="Specialist"
    )

    ax.set_xlabel("Model Size")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MNIST Accuracy: Pre vs Specialist")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def create_dataset_accuracy_plot(
    files,
    experiment_names,
    dataset,
):
    means = []
    stds = []

    for filename in files:

        path = Path(RESULTS_DIR) / filename

        with open(path, "rb") as f:
            data = pickle.load(f)

        # Get the merged model results
        values = np.array(
            data["results"]["Merged Model"][dataset]
        )

        means.append(values.mean() * 100)
        stds.append(values.std(ddof=1) * 100)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(experiment_names))

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Experiment")

    ax.set_title(
        f"{dataset.capitalize()} Accuracy"
    )

    ax.set_ylim(0, 100)

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    return fig    

def plot_specialist_vs_merged(filename):

    # Load file
    path = Path(RESULTS_DIR) / filename

    with open(path, "rb") as f:
        data = pickle.load(f)

    datasets = ["mnist", "fashion"]

    # Order: small specialist, merged, big specialist
    models = [
        ("Small Specialist", "specialist_results", "small_model"),
        ("Merged Model", "results", "Merged Model"),
        ("Big Specialist", "specialist_results", "big_model"),
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    for ax, dataset in zip(axes, datasets):

        means = []
        stds = []

        for _, result_type, model_name in models:

            values = np.array(
                data[result_type][model_name][dataset]
            )

            means.append(values.mean() * 100)
            stds.append(values.std(ddof=1) * 100)

        x = np.arange(3)

        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([
            "Small\nSpecialist",
            "Merged\nModel",
            "Big\nSpecialist"
        ])

        ax.set_ylabel("Accuracy (%)")
        ax.set_title(dataset.capitalize())

        ax.set_ylim(0, 100)
        ax.grid(
            axis="y",
            alpha=0.3
        )

    plt.tight_layout()

    return fig

def plot_results_heatmap(filenames, titles=None):
    if isinstance(filenames, str):
        filenames = [filenames]
    if titles is None:
        titles = [Path(f).stem for f in filenames]

    figures = []

    for filename, title in zip(filenames, titles):
        path = RESULTS_DIR / filename
        with open(path, "rb") as f:
            data = pickle.load(f)

        all_methods  = {**data["specialist_results"], **data["results"]}
        method_names = list(all_methods.keys())

        # Build mean matrix and std matrix
        mean_matrix = []
        std_matrix  = []

        for method in method_names:
            mean_row = [np.mean(all_methods[method][d]) * 100 for d in range(10)]
            std_row  = [np.std(all_methods[method][d], ddof=1) * 100 for d in range(10)]

            # Overall average and std across all digits and seeds
            all_values = np.concatenate([all_methods[method][d] for d in range(10)])
            mean_row.append(np.mean(all_values) * 100)
            std_row.append(np.std(all_values, ddof=1) * 100)

            mean_matrix.append(mean_row)
            std_matrix.append(std_row)

        mean_matrix = np.array(mean_matrix)
        std_matrix  = np.array(std_matrix)

        fig, ax = plt.subplots(figsize=(14, 4))
        im = ax.imshow(mean_matrix, cmap="RdYlGn", vmin=0, vmax=100)

        x_labels = [f"Class {d}" for d in range(10)] + ["Avg"]
        ax.set_xticks(range(11))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(range(len(method_names)))
        ax.set_yticklabels(method_names)
        ax.set_title(title, fontweight="bold", pad=10)

        ax.axvline(9.5, color="white", linewidth=2)

        # Write values inside cells
        for i in range(len(method_names)):
            for j in range(11):
                mean = mean_matrix[i, j]
                if j == 10:  # average column — show mean ± std
                    std = std_matrix[i, j]
                    ax.text(j, i, f"{mean:.0f}±{std:.0f}",
                            ha="center", va="center", fontsize=8)
                else:        # digit columns — show mean only
                    ax.text(j, i, f"{mean:.0f}%",
                            ha="center", va="center", fontsize=9)

        plt.colorbar(im, label="Accuracy %")
        plt.tight_layout()
        figures.append(fig)

    return figures


def plot_merge_comparison_both_datasets(files_grouped, title=""):
    
    # Get baselines from first file
    first_file = list(files_grouped.values())[0][0]
    with open(RESULTS_DIR / first_file, "rb") as f:
        first_data = pickle.load(f)

    baselines = {
        "big_model":   {
            "fashion": np.mean(first_data["specialist_results"]["big_model"]["fashion"])   * 100,
            "mnist":   np.mean(first_data["specialist_results"]["big_model"]["mnist"])     * 100,
        },
        "small_model": {
            "fashion": np.mean(first_data["specialist_results"]["small_model"]["fashion"]) * 100,
            "mnist":   np.mean(first_data["specialist_results"]["small_model"]["mnist"])   * 100,
        },
    }

    group_names = list(files_grouped.keys())
    n_groups    = len(group_names)

    # Compute means and stds per group per dataset
    fashion_means, fashion_stds = [], []
    mnist_means,   mnist_stds   = [], []

    for group, filenames in files_grouped.items():
        fashion_vals, mnist_vals = [], []
        for filename in filenames:
            with open(RESULTS_DIR / filename, "rb") as f:
                data = pickle.load(f)
            fashion_vals.extend(data["results"]["Merged Model"]["fashion"])
            mnist_vals.extend(data["results"]["Merged Model"]["mnist"])

        fashion_means.append(np.mean(fashion_vals) * 100)
        fashion_stds.append(np.std(fashion_vals)   * 100)
        mnist_means.append(np.mean(mnist_vals)     * 100)
        mnist_stds.append(np.std(mnist_vals)       * 100)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    x     = np.arange(n_groups)
    width = 0.35

    # Fashion bars
    bars_fashion = ax.bar(
        x - width/2, fashion_means, width,
        yerr=fashion_stds, capsize=4,
        color="#5b9bd5", edgecolor="white",
        label="Fashion"
    )

    # MNIST bars
    bars_mnist = ax.bar(
        x + width/2, mnist_means, width,
        yerr=mnist_stds, capsize=4,
        color="#ed7d31", edgecolor="white",
        label="MNIST"
    )

    # Baselines — fashion
    ax.axhline(baselines["big_model"]["fashion"],
               linestyle="--", color="#5b9bd5", linewidth=1.5,
               label=f"Big Model / Fashion ({baselines['big_model']['fashion']:.1f}%)")
    ax.axhline(baselines["small_model"]["fashion"],
               linestyle=":",  color="#5b9bd5", linewidth=1.5,
               label=f"Small Model / Fashion ({baselines['small_model']['fashion']:.1f}%)")

    # Baselines — mnist
    ax.axhline(baselines["big_model"]["mnist"],
               linestyle="--", color="#ed7d31", linewidth=1.5,
               label=f"Big Model / MNIST ({baselines['big_model']['mnist']:.1f}%)")
    ax.axhline(baselines["small_model"]["mnist"],
               linestyle=":",  color="#ed7d31", linewidth=1.5,
               label=f"Small Model / MNIST ({baselines['small_model']['mnist']:.1f}%)")

    # Value labels
    for bar, mean, std in zip(bars_fashion, fashion_means, fashion_stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=8)

    for bar, mean, std in zip(bars_mnist, mnist_means, mnist_stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def create_merged_accuracy_table_mixed(files, experiment_names, column_names):
    if len(files) != len(experiment_names):
        raise ValueError("files and experiment_names must have the same length")

    rows = []

    first_path = RESULTS_DIR / files[0]
    with open(first_path, "rb") as f:
        first_data = pickle.load(f)

    specialist_names = list(first_data["specialist_results"].keys())
    result_names     = list(first_data["results"].keys())

    def format_entry(entry):
        if isinstance(entry, dict):
            all_values = np.concatenate([entry[k] for k in entry.keys()])
        else:
            all_values = np.array(entry)
        return f"{all_values.mean()*100:.2f} ± {all_values.std(ddof=1)*100:.2f}%"

    for filename, experiment_name in zip(files, experiment_names):
        path = RESULTS_DIR / filename
        with open(path, "rb") as f:
            data = pickle.load(f)

        row = [experiment_name]

        for name in specialist_names:
            row.append(format_entry(data["specialist_results"][name]))

        for name in result_names:
            row.append(format_entry(data["results"][name]))

        rows.append(row)

    num_columns = len(column_names)
    fig, ax = plt.subplots(figsize=(12, 1))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=column_names,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.9)

    for col in range(num_columns):
        table[(0, col)].set_facecolor("#4C72B0")
        table[(0, col)].set_text_props(color="white", weight="bold")

    return fig

def create_3x3_experiment_table(base_name):
    
    merge_methods    = ["replace", "average", "fisher"]
    neuron_selection = ["fisher_bad", "random", "fisher_top"]
    
    col_labels = ["", "Worst Fisher", "Random", "Best Fisher"]
    row_labels = ["Replace", "Average Merge", "Fisher Merge"]

    def fmt(values):
        arr = np.array(values)
        return f"{arr.mean()*100:.1f}±{arr.std(ddof=1)*100:.1f}%"

    def build_matrix(dataset):
        table_data = []
        for merge in merge_methods:
            row = []
            for selection in neuron_selection:
                filename = f"{base_name}_{merge}_{selection}.pkl"
                path = RESULTS_DIR / filename
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    row.append(fmt(data["results"]["Merged Model"][dataset]))
                except FileNotFoundError:
                    row.append("N/A")
            table_data.append(row)
        return [[label] + row for label, row in zip(row_labels, table_data)]

    def draw_table(rows, title):
        num_columns = len(col_labels)
        num_rows    = len(rows)

        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.axis("off")
        ax.set_title(title, fontweight="bold", fontsize=11, pad=10)

        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.4, 2.2)
        table.auto_set_column_width(col=list(range(num_columns)))

        for col in range(num_columns):
            table[(0, col)].set_facecolor("#2c3e50")
            table[(0, col)].set_text_props(color="white", weight="bold")

        for row in range(1, num_rows + 1):
            table[(row, 0)].set_facecolor("#4C72B0")
            table[(row, 0)].set_text_props(color="white", weight="bold")

        colors = ["#EAF2F8", "#EAF7EA", "#FFF4E5"]
        for row in range(1, num_rows + 1):
            for col in range(1, num_columns):
                table[(row, col)].set_facecolor(colors[(row - 1) % len(colors)])

        plt.tight_layout(pad=0.1)
        plt.show()
        plt.close()

    draw_table(build_matrix("fashion"), "Fashion Accuracy — Merged Model")
    draw_table(build_matrix("mnist"),   "MNIST Accuracy — Merged Model")

def create_3x3_experiment_table_switch_pos(base_name):
    
    merge_methods    = ["replace", "fisher"]
    neuron_selection = ["fisher_bad", "random", "fisher_top"]
    col_labels       = ["", "Worst Fisher", "Random", "Best Fisher"]
    row_labels       = ["Replace", "Fisher Merge"]

    filenames = {
        ("replace", "fisher_bad"): f"{base_name}_replace_fisher_bad_all_layers_switch_pos.pkl",
        ("replace", "random"):     f"{base_name}_replace_random_switch_pos.pkl",
        ("replace", "fisher_top"): f"{base_name}_replace_fisher_top_all_layers_switch_pos.pkl",
        ("fisher",  "fisher_bad"): f"{base_name}_fisher_fisher_bad_all_layers_switch_pos.pkl",
        ("fisher",  "random"):     f"{base_name}_fisher_random_switch_pos.pkl",
        ("fisher",  "fisher_top"): f"{base_name}_fisher_fisher_top_all_layers_switch_pos.pkl",
    }

    def fmt(values):
        arr = np.array(values)
        return f"{arr.mean()*100:.1f}±{arr.std(ddof=1)*100:.1f}%"

    def build_matrix(dataset):
        table_data = []
        for merge in merge_methods:
            row = []
            for selection in neuron_selection:
                path = RESULTS_DIR / filenames[(merge, selection)]
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    row.append(fmt(data["results"]["Merged Model"][dataset]))
                except FileNotFoundError:
                    row.append("N/A")
            table_data.append(row)
        return [[label] + row for label, row in zip(row_labels, table_data)]

    def draw_table(rows, title):
        num_columns = len(col_labels)
        num_rows    = len(rows)

        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.axis("off")
        ax.set_title(title, fontweight="bold", fontsize=11, pad=10)

        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.4, 2.2)
        table.auto_set_column_width(col=list(range(num_columns)))

        for col in range(num_columns):
            table[(0, col)].set_facecolor("#2c3e50")
            table[(0, col)].set_text_props(color="white", weight="bold")

        for row in range(1, num_rows + 1):
            table[(row, 0)].set_facecolor("#4C72B0")
            table[(row, 0)].set_text_props(color="white", weight="bold")

        colors = ["#EAF2F8", "#EAF7EA", "#FFF4E5"]
        for row in range(1, num_rows + 1):
            for col in range(1, num_columns):
                table[(row, col)].set_facecolor(colors[(row - 1) % len(colors)])

        plt.tight_layout(pad=0.1)
        plt.show()
        plt.close()

    draw_table(build_matrix("fashion"), "Fashion Accuracy — Merged Model")
    draw_table(build_matrix("mnist"),   "MNIST Accuracy — Merged Model")
