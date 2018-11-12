import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import math
import os

from Util import Globals


def plot_weighted_ar(values, dataset_id):
    nrow = 7
    ncol = 2
    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(10, 22))
    # fig.suptitle('Aspect Ratios', fontsize=14)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.95)

    technique_ids = sorted(values)
    for i in range(len(technique_ids)):
        ax = fig.axes[i]
        technique = technique_ids[i]

        statistics_list = []
        # ax.set_title(technique)
        ax.set_title(Globals.acronyms[technique])
        print('.', end='')

        n_revisions = int(len(values[technique].columns)/2)
        for revision in range(n_revisions):
            w_col = 'w_' + str(revision)
            ar_col = 'ar_' + str(revision)
            df = values[technique][[w_col, ar_col]]
            # df = rectangles[technique][revision][['rx', 'ry', 'rw', 'rh']]
            # df = df.dropna(axis=0, subset=['rx'])
            # df['ar'] = df[['rw', 'rh']].min(axis=1) / df[['rw', 'rh']].max(axis=1)

            df = df.sort_values(by=ar_col)
            df[w_col] = df[w_col].cumsum()

            f = fq = median = tq = nf = float("nan")
            for i, row in df.iterrows():
                if row[w_col] >= 0.05 and math.isnan(f):
                    f = row[ar_col]
                if row[w_col] >= 0.25 and math.isnan(fq):
                    fq = row[ar_col]
                if row[w_col] >= 0.5 and math.isnan(median):
                    median = row[ar_col]
                if row[w_col] >= 0.75 and math.isnan(tq):
                    tq = row[ar_col]
                if row[w_col] >= 0.95 and math.isnan(nf):
                    nf = row[ar_col]

            item = {}
            item["med"] = median
            item["q1"] = fq
            item["q3"] = tq
            item["whislo"] = f
            item["whishi"] = nf
            item["fliers"] = []
            if median != math.nan:
                statistics_list.append(item)

        statistics_list.sort(key=lambda x: -x['med'])
        bp = ax.bxp(statistics_list, showfliers=False, patch_artist=True, widths=1);
        styleBoxplot(bp, fig, ax, n_revisions)
        ax.set_ylim(ymin=0, ymax=1)
    os.makedirs('plots/boxplots/w_ar/', exist_ok=True)
    fig.savefig('plots/boxplots/w_ar/' + dataset_id + '-warbp.png')
    # plt.show()
    return None

def plot_unweighted_ar(values, dataset_id):
    nrow = 7
    ncol = 2
    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(10, 22))
    # fig.suptitle('Aspect Ratios', fontsize=14)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.95)

    technique_ids = sorted(values)
    for i in range(len(technique_ids)):
        ax = fig.axes[i]
        data = []
        technique = technique_ids[i]

        # ax.set_title(technique)
        ax.set_title(Globals.acronyms[technique])
        print('.', end='')

        n_revisions = int(len(values[technique].columns)/2)
        for revision in range(n_revisions):
            ar_col = 'ar_' + str(revision)
            ar = values[technique][ar_col].dropna().values
            if len(ar) > 0:
                data.append(ar)

        data.sort(key=lambda x: -np.median(x))
        bp = ax.boxplot(data, whis=[5, 95], showfliers=False, patch_artist=True, widths=1);
        styleBoxplot(bp, fig, ax, len(data))
        ax.set_ylim(ymin=0, ymax=1)

    os.makedirs('plots/boxplots/u_ar/', exist_ok=True)
    fig.savefig('plots/boxplots/u_ar/' + dataset_id + '-uarbp.png')
    # plt.show()
    return None


def plot_instability(values, dataset_id, metric_id):
    nrow = 7
    ncol = 2
    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(10, 22))
    # fig.delaxes(axs[5, 1])
    fig.tight_layout()

    technique_ids = sorted(values)
    for i in range(len(technique_ids)):
        ax = fig.axes[i]
        technique = technique_ids[i]
        # ax.set_title(technique)
        ax.set_title(Globals.acronyms[technique])

        print('.', end='')

        data = []
        for revision in range(int(len(values[technique].columns)/2)):
            # df = values[technique]
            r_col = 'r_' + str(revision)
            b_col = 'b_' + str(revision)

            diff = values[technique][[r_col, b_col]].max(axis=1) - values[technique][b_col]
            diff_values = diff.dropna().values
            # display(df)
            if len(diff_values) > 0:
                data.append(diff_values)

        data.sort(key=lambda x: -np.median(x))
        bp = ax.boxplot(data, whis=[5, 95], showfliers=False, patch_artist=True, widths=1);

        ax.set_ylim(ymin=-0, ymax=1)
        ax.set_yticks([0, .25, .5, .75, 1], minor=False)
        ax.set_yticklabels([0, .25, .5, .75, 1], fontdict=None, minor=False)
        styleBoxplot(bp, fig, ax, len(data))

    os.makedirs('plots/boxplots/' + metric_id + '/', exist_ok=True)
    fig.savefig('plots/boxplots/' + metric_id + '/' + dataset_id + '-' + metric_id + 'bp.png')
    return None


def styleBoxplot(bp, fig, ax, n_revisions):
    def get_ax_size(fig, ax):
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        width *= fig.dpi
        height *= fig.dpi
        return width, height

    for box in bp['boxes']:
        # change outline color
        box.set(color='#1b9e77',
                linewidth=0,
                path_effects=[pe.Stroke(linewidth=0.1, foreground='#1b9e77'), pe.Normal()],
                facecolor='#1b9e77')
        box.set_zorder(10)
    for i, median in enumerate(bp['medians']):
        median.set(color='#000000',
                   linewidth=2,
                   solid_capstyle="butt",
                   ms=(get_ax_size(fig, ax)[0]) / (n_revisions))
        median.set_zorder(11)
        # median.set_xdata([i + 1 - 0.3, i + 1 + 0.3])
    for whisker in bp['whiskers']:
        whisker.set(color='#CCCCCC',
                    linestyle='-',
                    solid_capstyle="butt")
        whisker.set_path_effects([pe.PathPatchEffect(edgecolor='#CCCCCC',
                                                     linewidth=((get_ax_size(fig, ax)[0]) / (n_revisions)) * 1.08,
                                                     facecolor='black')])
    for cap in bp['caps']:
        cap.set(color='#FFFFFF', linewidth=0)

    # Set only 3 ticks on x
    ax.set_xticks([1, n_revisions / 2, n_revisions], minor=False)
    ax.set_xticklabels([1, int(n_revisions / 2), n_revisions], fontdict=None, minor=False)
    # ax.set_xticklabels(["", "", ""], fontdict=None, minor=False)

    # Remove extra spines and ticks
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.spines['left'].set_zorder(100)
    ax.tick_params(axis='x', which='both', top='off', direction='out')
    ax.tick_params(axis='y', which='both', right='off', left='on', direction='out')
