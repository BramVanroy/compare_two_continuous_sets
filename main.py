from pathlib import Path
import argparse
from statistics import mean, median
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

""" Calculates metrics between two continuous datasets. """


def do_test(i, j):
    ttest = stats.ttest_ind(i, j)
    cs = cosine_similarity(np.array(i).reshape(1, -1), np.array(j).reshape(1, -1)).tolist()[0][0]

    rmse = math.sqrt(mean_squared_error(i, j))
    mae = mean_absolute_error(i, j)

    i_norm, j_norm = get_norm_sets(i, j)
    rmse_norm = math.sqrt(mean_squared_error(i_norm, j_norm))
    mae_norm = mean_absolute_error(i_norm, j_norm)

    return ttest, cs, rmse, rmse_norm, mae, mae_norm


def get_basic_info(i, j):
    min_vals = min(i), min(j)
    max_vals = max(i), max(j)
    mean_vals = mean(i), mean(j)
    median_vals = median(i), median(j)

    return min_vals, max_vals, mean_vals, median_vals


def get_norm_sets(i, j):
    norm_i, norm_j = [], []
    for i_i, j_i in zip(i, j):
        norm_j.append(j_i/i_i)
        norm_i.append(1)

    return norm_i, norm_j


def plot_sets(i, j, title):
    sns.distplot(i, label='Set i')
    sns.distplot(j, label='Set j')
    plt.title(title)
    plt.legend()

    filename = title.replace(' ', '-') + '.png'
    plt.savefig(filename)
    plt.show()


def print_set_info(size, min_val, max_val, mean_val, median_val, ttest, cs, rmse, rmse_norm, mae, mae_norm, s='all'):
    title = f"Data set size ({s}): {str(size)}"
    print(title)
    print("=" * len(title))
    print("      \tSet i \tSet j")
    print(f"min   \t{min_val[0]:.4f}\t{min_val[1]:.4f}")
    print(f"max   \t{max_val[0]:.4f}\t{max_val[1]:.4f}")
    print(f"mean  \t{mean_val[0]:.4f}\t{mean_val[1]:.4f}")
    print(f"median\t{median_val[0]:.4f}\t{median_val[1]:.4f}\n")

    p = '< .01' if ttest.pvalue < 0.01 else f"{ttest.pvalue:.4f}"
    print(f"T-test ({s}):\n\t- statistic: {ttest.statistic:.4f}\n\t- p-value {p}\n")
    print(f"Cosine similarity ({s}): {cs:.4f}")
    print(f"Mean squared error ({s}): {rmse:.4f} ({rmse_norm:.2f}%)")
    print(f"Mean absolute error ({s}): {mae:.4f} ({mae_norm:.2f}%)\n")


def read_data(dir_i, dir_j):
    i_set, j_set = [], []

    for d, s in zip([dir_i, dir_j], [i_set, j_set]):
        for pfin in sorted(list(d.glob('*.cross'))):
            with open(str(pfin), 'r', encoding='utf-8') as fhin:
                cross = float(fhin.readline().strip())
                s.append(cross)

    return i_set, j_set


def reject_outliers_p(i, j, p=0.05):
    i_idxs, j_idxs = set(), set()

    for idx, data in enumerate([i, j]):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        cutoff = int(math.ceil((len(data) * p) / 2))

        min_idxs = np.argpartition(data, cutoff)[:cutoff]
        max_idxs = np.argpartition(data, -cutoff)[-cutoff:]
        merged_idxs = np.concatenate((min_idxs, max_idxs))

        if idx == 0:
            i_idxs = set(merged_idxs)
        else:
            j_idxs = set(merged_idxs)

    no_outliers_idxs = i_idxs.union(j_idxs)

    i_no_outliers = [i_i for idx, i_i in enumerate(i) if idx not in no_outliers_idxs]
    j_no_outliers = [j_i for idx, j_i in enumerate(j) if idx not in no_outliers_idxs]

    return i_no_outliers, j_no_outliers


def main(path_i, path_j):
    set_i, set_j = read_data(path_i, path_j)

    if len(set_i) != len(set_j):
        raise ValueError('Sets must be the same size.')

    plot_sets(set_i, set_j, 'with outliers')
    min_all, max_all, mean_all, median_all = get_basic_info(set_i, set_j)
    ttest_all, cs_all, rmse_all, rmse_all_norm, mae_all, mae_all_norm = do_test(set_i, set_j)

    print_set_info(len(set_i), min_all, max_all, mean_all, median_all, ttest_all,
                   cs_all, rmse_all, rmse_all_norm, mae_all, mae_all_norm)

    # NO OUTLIERS
    set_i_no_o, set_j_no_o = reject_outliers_p(set_i, set_j)

    plot_sets(set_i_no_o, set_j_no_o, 'without outliers')
    min_no_o, max_no_o, mean_no_o, median_no_o = get_basic_info(set_i_no_o, set_j_no_o)
    ttest_no_o, cs_no_o, rmse_no_o, rmse_no_o_norm, mae_no_o, mae_no_o_norm = do_test(set_i_no_o, set_j_no_o)

    print_set_info(len(set_i_no_o), min_no_o, max_no_o, mean_no_o, median_no_o, ttest_no_o,
                   cs_no_o, rmse_no_o, rmse_no_o_norm, mae_no_o, mae_no_o_norm, s='without outliers')


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Compare two datasets with continuous data. Reads data from one'
                                                 ' datapoint per file.')
    parser.add_argument('-i', '--input_dir_i', required=True,
                        help="Path to input dir i. Must contain files with one number per file.")
    parser.add_argument('-j', '--input_dir_j', required=True,
                        help="Path to input dir i. Must contain files with one number per file.")

    args = parser.parse_args()

    d_i = Path(args.input_dir_i).resolve()
    d_j = Path(args.input_dir_j).resolve()

    main(d_i, d_j)
