import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'monospace'
import numpy as np
import pandas as pd
import os
from adjustText import adjust_text

from Util import Globals, Parser


def plot(dataset_ids, draw_data, draw_labels):
    averages = collect_averages(dataset_ids)

    fig = plt.figure(figsize=(10,10))
    plt.xlabel('Stability')
    plt.ylabel('Visual quality')
    texts = []
    for technique in sorted(Globals.acronyms):
        df = averages[(averages['technique'] == Globals.acronyms[technique])].dropna(axis=0)
        colors = [Globals.tech_to_color[technique] for i in range(len(df['label'].values))]
        labels = Globals.acronyms[technique]

        x_mean = df['inst'].mean()
        y_mean = df['ar'].mean()
        #plt.scatter(df['inst'], df['ar'], s=60, c=colors, label=labels, edgecolors=None)

        for i, point in df.iterrows():
            x_line = [x_mean, point['inst']]
            y_line = [y_mean, point['ar']]

            if draw_data:
                plt.plot(x_line, y_line, c=colors[0], zorder=1, alpha=0.3)
            if draw_labels:
                plt.text(point['inst'], point['ar'], str(int(i / len(Globals.acronyms))), color='black', ha='center', va='center', fontsize=7)

        if draw_data:
            plt.scatter(x_mean, y_mean, s=80, c=colors, label=labels, linewidth=1, zorder=10)

        t = plt.text(x_mean, y_mean, Globals.acronyms[technique], ha='center', va='center', zorder=11,
                     fontsize=14, fontweight='bold')
        texts.append(t)
    adjust_text(texts, force_points=0.2, force_text=0.2, expand_points=(1,1), expand_text=(1,1))

    plt.xlim(xmin=0) #xmax=0.35)
    plt.ylim(ymin=0, ymax=1)
    plt.legend(loc=4)


    os.makedirs('plots/scatter', exist_ok=True)
    if draw_data and draw_labels:
        print("plots/scatter/scatter-p+l.png")
        # fig.savefig("scatter/scatter-p+l.svg")
        fig.savefig("plots/scatter/scatter-p+l.png", dpi=500)
    elif draw_data and not draw_labels:
        print("plots/scatter/scatter-p.png")
        # fig.savefig("scatter/scatter-p.svg")
        fig.savefig("plots/scatter/scatter-p.png", dpi=500)
    elif draw_labels and not draw_data:
        print("plots/scatter/scatter-l.png")
        # fig.savefig("scatter/scatter-l.svg")
        fig.savefig("plots/scatter/scatter-l.png", dpi=500)

    # plt.show()
    return None


def collect_averages(dataset_ids):
    results = []
    for dataset_id in dataset_ids:
        inst_dict = instability_average(dataset_id)
        ar_dict = ar_average(dataset_id)

        technique_list = sorted(ar_dict)
        for i, technique in enumerate(technique_list):
            results.append([dataset_id, Globals.acronyms[technique], i, ar_dict[technique], inst_dict[technique]])

    df = pd.DataFrame(results, columns=['dataset', 'technique', 'label', 'ar', 'inst'])
    return df


def instability_average(dataset_id):
    ct_df = Parser.read_ct_metric(dataset_id)
    rpc_df = Parser.read_rpc_metric(dataset_id)

    means = {}

    technique_list = sorted(rpc_df)
    for i, technique_id in enumerate(technique_list):
        technique_means = []
        for revision in range(int(len(rpc_df[technique_id].columns) / 2)):
            r_col = 'r_' + str(revision)
            b_col = 'b_' + str(revision)
            diff = rpc_df[technique_id][[r_col, b_col]].max(axis=1) - rpc_df[technique_id][b_col]
            ct_mean = diff.dropna().mean()

            r_col = 'r_' + str(revision)
            b_col = 'b_' + str(revision)
            diff = ct_df[technique_id][[r_col, b_col]].max(axis=1) - ct_df[technique_id][b_col]
            rpc_mean = diff.dropna().mean()

            technique_means.append((ct_mean + rpc_mean) / 2)

        means[technique_id] = np.mean(technique_means)

    return means


def ar_average(dataset_id):
    ar_df = Parser.read_aspect_ratios(dataset_id)
    means = {}

    technique_list = sorted(ar_df)
    for i, technique_id in enumerate(technique_list):
        technique_means = []
        for revision in range(int(len(ar_df[technique_id].columns) / 2)):
            # df = ar_df[technique_id]
            w_col = 'w_' + str(revision)
            ar_col = 'ar_' + str(revision)

            u_avg = ar_df[technique_id][ar_col].mean(axis=0)
            w_avg = np.average(ar_df[technique_id][ar_col].dropna(), weights=ar_df[technique_id][w_col].dropna())

            technique_means.append((u_avg + w_avg)/2)

        means[technique_id] = np.mean(technique_means)

    return means
