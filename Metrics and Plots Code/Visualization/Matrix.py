import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
rcParams['font.family'] = 'monospace'
import numpy as np
import os

from Util import Globals, Parser


def plot(dataset_ids):
    os.makedirs('plots/matrices', exist_ok=True)
    # Plot AR matrix
    weighted_ar_matrix, unweighted_ar_matrix, technique_acronyms = make_ar_matrices(dataset_ids)
    print('war')
    plot_matrix(weighted_ar_matrix, dataset_ids, technique_acronyms, 'war')
    print('uar')
    plot_matrix(unweighted_ar_matrix, dataset_ids, technique_acronyms, 'uar')

    # Plot CT matrix
    print('ct')
    ct_matrix, technique_acronyms = make_ct_matrix(dataset_ids)
    plot_matrix(ct_matrix, dataset_ids, technique_acronyms, 'ct')

    # Plot RPC matrix
    print('rpc')
    rpc_matrix, technique_acronyms = make_rpc_matrix(dataset_ids)
    plot_matrix(rpc_matrix, dataset_ids, technique_acronyms, 'rpc')


def plot_matrix(matrix, dataset_ids, technique_acronyms, metric_id):

    fig = plt.figure()
    ax = fig.add_subplot(111)


    if metric_id == 'uar' or metric_id == 'war':
        mat = ax.matshow(matrix, cmap=plt.cm.gist_gray)
    else:
        mat = ax.matshow(matrix, cmap=plt.cm.gist_gray_r)  # Invert colormap for instability

    # Ticks, labels and grids
    # ax.set_xticklabels(dataset_ids, rotation='vertical')
    ax.set_xticklabels(['' for i in range(len(dataset_ids))], rotation='vertical')
    ax.set_xticks(range(len(dataset_ids)), minor=False)
    ax.set_yticklabels(technique_acronyms)
    ax.set_yticks(range(len(technique_acronyms)), minor=False)
    ax.set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
    ax.set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
    plt.grid(which='minor', color='#999999', linestyle='-', linewidth=1)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    # Add the text
    x_start = 0.0
    x_end = len(dataset_ids)
    y_start = 0.0
    y_end = len(technique_acronyms)

    jump_x = (x_end - x_start) / (2.0 * len(dataset_ids))
    jump_y = (y_end - y_start) / (2.0 * len(technique_acronyms))
    x_positions = np.linspace(start=x_start-0.5, stop=x_end-0.5, num=len(dataset_ids), endpoint=False)
    y_positions = np.linspace(start=y_start-0.5, stop=y_end-0.5, num=len(technique_acronyms), endpoint=False)

    # for y_index, y in enumerate(y_positions):
    #     for x_index, x in enumerate(x_positions):
    #         label = "{0:.3f}".format(matrix[y_index, x_index]).lstrip('0')
    #         text_x = x + jump_x
    #         text_y = y + jump_y
    #         ax.text(text_x, text_y, label, color='black', ha='center', va='center', fontsize=9)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)

    plt.rc('text', usetex=True)

    if metric_id == 'uar':
        ax.set_title(r'$\rho$', size=24, fontdict={'family':'serif'}, loc='left')
    elif metric_id == 'war':
        ax.set_title(r'$\rho_W$', size=24, fontdict={'family':'serif'}, loc='left')
    elif metric_id == 'ct':
        ax.set_title(r'$\sigma_{CT}$', size=24, fontdict={'family':'serif'}, loc='left')
    elif metric_id == 'rpc':
        ax.set_title(r'$\sigma_{RP}$', size=24, fontdict={'family':'serif'}, loc='left')

    plt.colorbar(mat, cax=cax)
    fig.tight_layout()
    fig.savefig('plots/matrices/matrix-'+ metric_id +'.png', dpi=250)
    # plt.show()


def make_ct_matrix(dataset_ids):
    technique_ids = []
    all_means = []
    for dataset_id in dataset_ids:
        ct_df = Parser.read_ct_metric(dataset_id)

        dataset_means = np.array([])
        technique_list = sorted(ct_df)
        if len(technique_ids) == 0:
            technique_acronyms = [Globals.acronyms[d] for d in technique_list]

        for i, technique_id in enumerate(technique_list):
            technique_means = []
            for revision in range(int(len(ct_df[technique_id].columns) / 2)):
                df = ct_df[technique_id]
                r_col = 'r_' + str(revision)
                b_col = 'b_' + str(revision)

                diff = df[[r_col, b_col]].max(axis=1) - df[b_col]
                diff = diff.dropna()
                if len(diff) > 0:
                    diff_mean = diff.mean()
                else:
                    diff_mean = 0

                technique_means.append(diff_mean)

            dataset_means = np.append(dataset_means, np.mean(technique_means))
        all_means.append(dataset_means)

    return np.array(all_means).transpose(), technique_acronyms  # Transpose matrix so each row is a technique and each column a dataset


def make_rpc_matrix(dataset_ids):

    technique_ids = []
    all_means = []
    for dataset_id in dataset_ids:
        rpc_df = Parser.read_rpc_metric(dataset_id)

        dataset_means = np.array([])
        technique_list = sorted(rpc_df)
        if len(technique_ids) == 0:
            technique_acronyms = [Globals.acronyms[d] for d in technique_list]

        for i, technique_id in enumerate(technique_list):
            technique_means = []
            for revision in range(int(len(rpc_df[technique_id].columns) / 2)):
                df = rpc_df[technique_id]
                r_col = 'r_' + str(revision)
                b_col = 'b_' + str(revision)

                diff = df[[r_col, b_col]].max(axis=1) - df[b_col]
                diff = diff.dropna()
                if len(diff) > 0:
                    diff_mean = diff.mean()
                else:
                    diff_mean = 0

                technique_means.append(diff_mean)

            dataset_means = np.append(dataset_means, np.mean(technique_means))
        all_means.append(dataset_means)

    return np.array(all_means).transpose(), technique_acronyms  # Transpose matrix so each row is a technique and each column a dataset


def make_ar_matrices(dataset_ids):

    technique_ids = []
    weighted_means = []
    unweighted_means = []
    for dataset_id in dataset_ids:
        ar_df = Parser.read_aspect_ratios(dataset_id)

        weighted_dataset_means = np.array([])
        unweighted_dataset_means = np.array([])
        technique_list = sorted(ar_df)
        if len(technique_ids) == 0:
            technique_acronyms = [Globals.acronyms[d] for d in technique_list]

        for i, technique_id in enumerate(technique_list):
            weighted_technique_means = []
            unweighted_technique_means = []
            for revision in range(int(len(ar_df[technique_id].columns) / 2)):
                w_col = 'w_' + str(revision)
                ar_col = 'ar_' + str(revision)

                u_avg = ar_df[technique_id][ar_col].mean(axis=0)
                w_avg = np.average(ar_df[technique_id][ar_col].dropna(), weights=ar_df[technique_id][w_col].dropna())

                weighted_technique_means.append(w_avg)
                unweighted_technique_means.append(u_avg)

            weighted_dataset_means = np.append(weighted_dataset_means, np.mean(weighted_technique_means))
            unweighted_dataset_means = np.append(unweighted_dataset_means, np.mean(unweighted_technique_means))
        weighted_means.append(weighted_dataset_means)
        unweighted_means.append(unweighted_dataset_means)

    return np.array(weighted_means).transpose(), np.array(unweighted_means).transpose(), technique_acronyms  # Transpose matrices so each row is a technique and each column a dataset
