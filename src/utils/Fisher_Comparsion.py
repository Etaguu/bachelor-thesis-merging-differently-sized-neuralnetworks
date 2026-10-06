import torch
import matplotlib.pyplot as plt


def analyze_fisher(fisher, plot=True):
    #Analyze a full Fisher Information Matrix.

    # Prepare matrix
    fisher = fisher.detach().float().cpu()

    # Numerical symmetrization
    fisher = (fisher + fisher.T) / 2

    # Basic statistics

    diagonal = torch.diag(fisher)
    diagonal_matrix = torch.diag(diagonal)
    off_diagonal = fisher - diagonal_matrix

    symmetry_error = torch.norm(fisher - fisher.T).item()

    off_diagonal_ratio = (
        torch.norm(off_diagonal) /
        torch.norm(fisher)
    ).item()


    # Print results

    print("Fisher Analysis ")

    print(f"Shape:                 {tuple(fisher.shape)}")

    print("\nBasic statistics:")
    print(f"Min:                   {fisher.min().item():.6e}")
    print(f"Max:                   {fisher.max().item():.6e}")
    print(f"Mean:                  {fisher.mean().item():.6e}")

    print("\nSymmetry:")
    print(f"Symmetry error:        {symmetry_error:.6e}")

    print("\nDiagonal:")
    print(f"Diagonal min:          {diagonal.min().item():.6e}")
    print(f"Diagonal max:          {diagonal.max().item():.6e}")
    print(f"Diagonal mean:         {diagonal.mean().item():.6e}")

    print("\nOff-diagonal:")
    print(f"Off-diagonal ratio:    {off_diagonal_ratio:.6f}")

    # Plots

    if plot:

        # Fisher heatmap
        plt.figure(figsize=(7, 6))

        plt.imshow(
            fisher.numpy(),
            aspect="auto"
        )

        plt.colorbar(label="Fisher value")
        plt.xlabel("Parameter")
        plt.ylabel("Parameter")
        plt.title("Full Fisher Information Matrix")

        plt.tight_layout()
        plt.show()




    return {
        "shape": tuple(fisher.shape),
        "symmetry_error": symmetry_error,
        "min": fisher.min().item(),
        "max": fisher.max().item(),
        "mean": fisher.mean().item(),
        "diagonal": diagonal,
        "diagonal_mean": diagonal.mean().item(),
        "off_diagonal_ratio": off_diagonal_ratio,
    }
