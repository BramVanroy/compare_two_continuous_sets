from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

""" Calculates metrics between two continuous datasets. """


def compare_sets(set_a, set_b):
    ttest = stats.ttest_ind(set_a, set_b)
    pearsonr = stats.pearsonr(set_a, set_b)
    cs = cosine_similarity(np.array(set_a).reshape(1, -1), np.array(set_b).reshape(1, -1)).tolist()[0][0]

    mse = mean_squared_error(set_a, set_b)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(set_a, set_b)

    return ttest, pearsonr, cs, mse, rmse, mae


def get_basic_info(set_a, set_b):
    min_vals = np.min(set_a), np.min(set_b)
    max_vals = np.max(set_a), np.max(set_b)
    mean_vals = np.mean(set_a), np.mean(set_b)
    median_vals = np.median(set_a), np.median(set_b)
    std_vals = np.std(set_a), np.std(set_b)

    return min_vals, max_vals, mean_vals, median_vals, std_vals


def plot_sets(set_a, set_b, title=None):
    # Scatter plot
    # Dist plot
    ax = sns.scatterplot(set_a, set_b)
    print(ax.get_xlim())
    plt.plot([0, ax.get_ylim()[1]], [0, ax.get_ylim()[1]], color='black', linewidth=2)
    ax.set_xlabel('Real values')
    ax.set_ylabel('Predicted values')
    if title is not None:
        plt.title(title)

    plt.ylim(ax.get_ylim()[0], ax.get_xlim()[1])

    fname_scatter = title.replace(' ', '-') + '-scatter.png' if title is not None else 'scatter.png'

    plt.savefig(fname_scatter)
    plt.show()

    # Dist plot
    sns.distplot(set_a, label='Real values')
    sns.distplot(set_b, label='Predicted values')
    plt.legend()
    if title is not None:
        plt.title(title)

    fname_dist = title.replace(' ', '-') + '-dist.png' if title is not None else 'dist.png'

    plt.savefig(fname_dist)
    plt.show()

    # Box plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if title is not None:
        fig.title(title)
    sns.boxplot(set_a, orient='v', ax=ax1)
    sns.boxplot(set_b, orient='v', ax=ax2)

    ax1.set_xlabel = 'Set a'
    ax2.set_xlabel = 'Set b'

    fname_box = title.replace(' ', '-') + '-boxplot.png' if title is not None else 'boxplot.png'
    plt.savefig(fname_box)
    plt.show()


def print_set_info(size, mins, maxs, means, medians, stds, ttest, pearsonr, cs, mse, rmse, mae):
    title = f"Data set size: {size:,}"
    print(title)
    print("=" * len(title))
    print("      \tSet a \tSet b")
    print(f"min   \t{mins[0]:.4f}\t{mins[1]:.4f}")
    print(f"max   \t{maxs[0]:.4f}\t{maxs[1]:.4f}")
    print(f"mean  \t{means[0]:.4f}\t{means[1]:.4f}")
    print(f"median\t{medians[0]:.4f}\t{medians[1]:.4f}")
    print(f"std   \t{stds[0]:.4f}\t{stds[1]:.4f}\n")

    ttest_p = '< .01' if ttest.pvalue < 0.01 else f"{ttest.pvalue:.4f}"
    print(f"T-test:\n\t- statistic: {ttest.statistic:.4f}\n\t- p-value {ttest_p}\n")
    pearson_p = '< .01' if pearsonr[1] < 0.01 else f"{pearsonr[1]:.4f}"
    print(f"Pearson's r:\n\t- correlation: {pearsonr[0]:.4f}\n\t- p-value {pearson_p}\n")
    print(f"Cosine similarity: {cs:.4f}")
    print(f"Mean square error: {mse:.4f}")
    print(f"Root mean square error: {rmse:.4f}")
    print(f"Mean absolute error: {mae:.4f}\n")


def read_data(file_a, file_b):
    set_a = []
    set_b = []

    for fname, s in zip([file_a, file_b], [set_a, set_b]):
        pfname = Path(fname).resolve()
        with open(pfname, 'r', encoding='utf-8') as fhin:
            # memory-lenient for large files
            for line in fhin:
                line = line.strip()
                s.append(line)

    initial_size = len(set_a)
    # set_a and set_b are np arrays
    set_a, set_b = reject_none(set_a, set_b)

    print(f"Initial dataset: {initial_size}. Without None values: {len(set_a)}")

    return set_a, set_b


def reject_none(set_a, set_b):
    """ Remove None string values from dataset. If it occurs in one set, remove the same index in the other.

        :returns two numpy arrays

    """
    set_a = np.array(set_a)
    set_b = np.array(set_b)

    # Get indices where x=='None', merge those idxs
    none_idxs = np.union1d(np.nonzero(set_a == 'None'),
                           np.nonzero(set_b == 'None'))
    # Delete all 'None' indices from both arrays and cast to float
    try:
        set_a_no_none = np.delete(set_a, none_idxs).astype(float)
    except ValueError as e:
        raise ValueError(f"Error in set a: {e}")

    try:
        set_b_no_none = np.delete(set_b, none_idxs).astype(float)
    except ValueError as e:
        raise ValueError(f"Error in set b: {e}")

    return set_a_no_none, set_b_no_none


def main(file_a, file_b):
    print(f"Comparing two datasets:\n\t- {file_a}\n\t- {file_b}\n")
    set_a, set_b = read_data(file_a, file_b)

    if set_a.shape[0] != set_b.shape[0]:
        raise ValueError('Sets must be the same size.')

    plot_sets(set_a, set_b)
    dataset_stats = get_basic_info(set_a, set_b)
    comparison_stats = compare_sets(set_a, set_b)

    print_set_info(set_a.shape[0],
                   *dataset_stats,
                   *comparison_stats)


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Compare two datasets with continuous data. Reads data from one'
                                                 ' datapoint per file.')
    parser.add_argument('file_a', help='Path to the first input file (real values).')
    parser.add_argument('file_b', help='Path to the second input file (predicted values).')

    args = parser.parse_args()

    main(args.file_a, args.file_b)
