import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'monospace'
from matplotlib import collections as mc
import numpy as np
import math
import os

from Util import Globals

def plot(dataframes, dataset_id):
    nrow = 4
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(8, 8))
    plt.setp(axs.flat, aspect=1.0, adjustable='box-forced')

    # fig.suptitle('Aspect Ratios', fontsize=14)
    # fig.subplots_adjust(top=0.95)
    cmap = ['#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']
    cmap = [matplotlib.colors.hex2color(c) for c in cmap]

    technique_ids = sorted(list(Globals.acronyms.keys()))
    for i, technique_id in enumerate(technique_ids):
        ax = fig.axes[i]
        technique = technique_ids[i]

        # ax.set_title(technique)
        ax.set_title(Globals.acronyms[technique])
        print('.', end='')

        centroids = {}
        for df in dataframes[technique_id]:
            for index, row in df.iterrows():
                c_x = row['rx'] + row['rw'] / 2
                c_y = row['ry'] + row['rh'] / 2
                if index in centroids:
                    centroids[index].append((c_x, c_y))  # Append tuple
                else:
                    centroids[index] = [(c_x, c_y)]  # Initialize list for a new entry

        lines = []
        colors = []
        for key, centroid_list in centroids.items():
            for i in range(len(centroid_list) - 1):
                a = centroid_list[i]
                b = centroid_list[i + 1]

                xs = np.linspace(a[0], b[0], 9)
                ys = np.linspace(a[1], b[1], 9)
                points = np.array([xs, ys]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                for n, s in enumerate(segments):
                    lines.append(s)  # Add line segment
                    alpha = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) / math.sqrt(1000**2 + 1000**2)  # Assuming we are plotting the treemap in a 1000x1000 pixel frame
                    # alpha = alpha / math.sqrt(len(centroid_list))

                    colors.append((*cmap[n], alpha))  # Set color for line segment

                    # colors.append((0, 0, 0, alpha))  # Set color for line segment

        lc = mc.LineCollection(lines, colors=colors, linewidths=1)
        ax.add_collection(lc)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        # ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    os.makedirs('plots/directed-trails', exist_ok=True)
    fig.savefig('plots/directed-trails/' + dataset_id + '-dt.png', dpi=200)

    return None