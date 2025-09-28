import matplotlib.pyplot as plt
import pandas as pd

def plot_stokes_results(model_nums):
    """
    Plots the results of the Stokes problem for multiple mesh refinements.

    This function reads CSV files containing simulation results for different mesh refinements
    and plots the drag force (Fd), lift force (Fl), lift-to-drag ratio (Fl/Fd), and the objective function (J)
    as a function of the angle of attack.

    :param model_nums: List of integers representing the mesh refinement levels (e.g., [1, 2, 3, 4]).
    """
    dataset = []
    for model_num in model_nums:
        path = f"outputs/boeing_results_{model_num}.csv"
        dataset.append(pd.read_csv(path, index_col=0))

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    plt.subplot(2, 2, 1)
    for i in range(len(model_nums)):
        data = dataset[i]
        plt.plot(data["angle"], data["fd"], label = f"Mesh Refinement #{model_nums[i]}")
    plt.legend()
    plt.title("Fd")

    plt.subplot(2, 2, 2)
    for i in range(len(model_nums)):
        data = dataset[i]
        plt.plot(data["angle"], data["fl"], label = f"Mesh Refinement #{model_nums[i]}")
    plt.legend()
    plt.title("Fl")

    plt.subplot(2, 2, 3)
    for i in range(len(model_nums)):
        data = dataset[i]
        plt.plot(data["angle"], data["fl"] / data["fd"], label = f"Mesh Refinement #{model_nums[i]}")
    plt.legend()
    plt.title("Fl / Fd")

    plt.subplot(2, 2, 4)
    for i in range(len(model_nums)):
        data = dataset[i]
        plt.plot(data["angle"], data["j"], label = f"Mesh Refinement #{model_nums[i]}")
    plt.legend()
    plt.title("J")

    plt.show()


if __name__ == "__main__":
    """
    This script reads the results from CSV files for different mesh refinement levels
    and generates plots for drag force, lift force, lift-to-drag ratio, and the objective function.
    """
    model_nums = [1,2,3,4]
    plot_stokes_results(model_nums)