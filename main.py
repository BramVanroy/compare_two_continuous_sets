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
    pearsonr = stats.pearsonr(i, j)
    cs = cosine_similarity(np.array(i).reshape(1, -1), np.array(j).reshape(1, -1)).tolist()[0][0]

    rmse = math.sqrt(mean_squared_error(i, j))
    mae = mean_absolute_error(i, j)

    i_norm, j_norm = get_norm_sets(i, j)
    rmse_norm = math.sqrt(mean_squared_error(i_norm, j_norm))
    mae_norm = mean_absolute_error(i_norm, j_norm)

    return ttest, pearsonr, cs, rmse, rmse_norm, mae, mae_norm


def get_basic_info(i, j):
    min_vals = min(i), min(j)
    max_vals = max(i), max(j)
    mean_vals = mean(i), mean(j)
    median_vals = median(i), median(j)
    std_vals = np.std(i), np.std(j)

    return min_vals, max_vals, mean_vals, median_vals, std_vals


def get_norm_sets(i, j):
    norm_i, norm_j = [], []
    for i_i, j_i in zip(i, j):
        i_i += 1
        j_i += 1
        norm_j.append(j_i*100/i_i)
        norm_i.append(100)

    return norm_i, norm_j


def plot_sets(i, j, title):
    sns.distplot(i, label='Set i')
    sns.distplot(j, label='Set j')
    plt.title(title)
    plt.legend()

    filename = title.replace(' ', '-') + '-dist.png'
    plt.savefig(filename)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.boxplot(i, orient='v', ax=ax1)
    sns.boxplot(j, orient='v', ax=ax2)

    ax1.set_xlabel = 'Set i'
    ax2.set_xlabel = 'Set j'
    plt.title(title)

    filename = title.replace(' ', '-') + '-boxplot.png'
    plt.savefig(filename)
    plt.show()


def print_set_info(size, min_val, max_val, mean_val, median_val, std_val, ttest, pearsonr, cs, rmse, rmse_norm, mae, mae_norm, s='all'):
    title = f"Data set size ({s}): {str(size)}"
    print(title)
    print("=" * len(title))
    print("      \tSet i \tSet j")
    print(f"min   \t{min_val[0]:.4f}\t{min_val[1]:.4f}")
    print(f"max   \t{max_val[0]:.4f}\t{max_val[1]:.4f}")
    print(f"mean  \t{mean_val[0]:.4f}\t{mean_val[1]:.4f}")
    print(f"median\t{median_val[0]:.4f}\t{median_val[1]:.4f}")
    print(f"std   \t{std_val[0]:.4f}\t{std_val[1]:.4f}\n")

    ttest_p = '< .01' if ttest.pvalue < 0.01 else f"{ttest.pvalue:.4f}"
    print(f"T-test ({s}):\n\t- statistic: {ttest.statistic:.4f}\n\t- p-value {ttest_p}\n")
    pearson_p = '< .01' if pearsonr[1] < 0.01 else f"{pearsonr[1]:.4f}"
    print(f"Pearson r ({s}):\n\t- correlation: {pearsonr[0]:.4f}\n\t- p-value {pearson_p}\n")
    print(f"Cosine similarity ({s}): {cs:.4f}")
    print(f"Root mean square error ({s}): {rmse:.4f} ({rmse_norm:.2f}%)")
    print(f"Mean absolute error ({s}): {mae:.4f} ({mae_norm:.2f}%)\n")


def read_data(dir_i, dir_j):
    i_set, j_set = [], []

    for d, s in zip([dir_i, dir_j], [i_set, j_set]):
        for pfin in sorted(list(d.glob('*.cross'))):
            with open(str(pfin), 'r', encoding='utf-8') as fhin:
                cross = fhin.readline().strip()
                s.append(cross)

    initial_size = len(i_set)

    i_set, j_set = reject_none(i_set, j_set)

    i_set, j_set = list(map(float, i_set)), list(map(float, j_set))


    print(f"Initial dataset: {initial_size}. Without None values: {len(i_set)}")

    return i_set, j_set


def reject_none(i, j):
    if not isinstance(i, np.ndarray):
        i = np.array(i)
    if not isinstance(j, np.ndarray):
        j = np.array(j)

    i_none_idxs = set(np.nonzero(i == 'None')[0].tolist())
    j_none_idxs = set(np.nonzero(j == 'None')[0].tolist())
    none_idxs = i_none_idxs.union(j_none_idxs)

    i_no_none = np.delete(i, list(none_idxs))
    j_no_none = np.delete(j, list(none_idxs))

    return i_no_none, j_no_none


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

    i_no_outliers = np.delete(i, list(no_outliers_idxs))
    j_no_outliers = np.delete(j, list(no_outliers_idxs))

    return i_no_outliers, j_no_outliers


def main(path_i, path_j, outlier_percentage):
    set_i, set_j = read_data(path_i, path_j)

    if len(set_i) != len(set_j):
        raise ValueError('Sets must be the same size.')

    plot_sets(set_i, set_j, 'with outliers')
    min_all, max_all, mean_all, median_all, std_all = get_basic_info(set_i, set_j)
    ttest_all, pearsonr_all, cs_all, rmse_all, rmse_all_norm, mae_all, mae_all_norm = do_test(set_i, set_j)

    print_set_info(len(set_i), min_all, max_all, mean_all, median_all, std_all,
                   ttest_all, pearsonr_all, cs_all, rmse_all, rmse_all_norm, mae_all, mae_all_norm)

    # NO OUTLIERS
    set_i_no_o, set_j_no_o = reject_outliers_p(set_i, set_j, p=outlier_percentage)

    plot_sets(set_i_no_o, set_j_no_o, 'without outliers')
    min_no_o, max_no_o, mean_no_o, median_no_o, std_no_o = get_basic_info(set_i_no_o, set_j_no_o)
    ttest_no_o, pearsonr_no_o, cs_no_o, rmse_no_o, rmse_no_o_norm, mae_no_o, mae_no_o_norm = do_test(set_i_no_o, set_j_no_o)

    print_set_info(len(set_i_no_o), min_no_o, max_no_o, mean_no_o, median_no_o, std_no_o,
                   ttest_no_o, pearsonr_no_o, cs_no_o, rmse_no_o, rmse_no_o_norm, mae_no_o, mae_no_o_norm, s='without outliers')


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Compare two datasets with continuous data. Reads data from one'
                                                 ' datapoint per file.')
    parser.add_argument('-i', '--input_dir_i', required=True,
                        help="Path to input dir i. Must contain files with one number per file.")
    parser.add_argument('-j', '--input_dir_j', required=True,
                        help="Path to input dir i. Must contain files with one number per file.")

    parser.add_argument('-p', '--outlier_percentage', default=0.05,
                        help="Remove some percentage of the highest and lowest data to avoid outliers.")

    args = parser.parse_args()

    d_i = Path(args.input_dir_i).resolve()
    d_j = Path(args.input_dir_j).resolve()

    main(d_i, d_j, float(args.outlier_percentage))
