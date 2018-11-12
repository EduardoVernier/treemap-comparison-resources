import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from matplotlib.patches import Rectangle
rcParams['font.family'] = 'monospace'
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
import math
import os
from scipy import stats

from Util import Globals

def make_colors(vals, min_dens, max_dens, cmap, log):
    if log:
        norm = LogNorm(vmin=vals.min()*2, vmax=vals.max()**1.5)
    else:
        norm = Normalize(vmin=min_dens, vmax=max_dens)
    return [cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(val) for val in vals]


def plot(dataframes, dataset_id):
    nrow = 4
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(8, 8))
    plt.setp(axs.flat, aspect=1.0, adjustable='box-forced')

    # fig.suptitle('Aspect Ratios', fontsize=14)
    # fig.subplots_adjust(top=0.95)

    max_dens_app = 0
    min_dens_app = 0
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

        # Place a bunch of points along each line
        points_x = []
        points_y = []
        for key, centroid_list in centroids.items():
            for i in range(len(centroid_list) - 1):
                a = centroid_list[i]
                b = centroid_list[i + 1]
                line_len = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
                if line_len > 20:
                    n_steps = int(line_len / 20)
                    points_x.extend(np.linspace(a[0], b[0], n_steps))
                    points_y.extend(np.linspace(a[1], b[1], n_steps))

        sample_size = 5000 if len(points_x) > 5000 else len(points_x)
        print(technique_id)
        if len(points_x) == 0:
            matrix = pd.DataFrame([[0, 0.1, 0.3, 0.9], [0, 0.1, 0.1, 0.3]]).as_matrix()
        else:
            matrix = pd.DataFrame([points_x, points_y]).sample(sample_size, axis=1, random_state=1).as_matrix()

        grid_x = []
        grid_y = []
        for i in range(100):
            for j in range(100):
                grid_x.append(i*10)
                grid_y.append(j*10)

        dens = stats.gaussian_kde(matrix)
        grid = pd.DataFrame([grid_x, grid_y]).as_matrix()
        dens_pt = dens(grid)

        if max_dens_app == 0:
            max_dens_app = 1.2 * max(dens_pt)
            min_dens_app = min(dens_pt)

        colours = make_colors(dens_pt, min_dens_app, max_dens_app, 'inferno', False)
        # ax.add_patch(Rectangle((0, 0), 100, 100, color='black', alpha=1, zorder=0))
        ax.scatter(grid[0], grid[1], marker='s', color=colours, s=4, alpha=1, linewidths=0, edgecolors='red', zorder=1)

        # lc = mc.LineCollection(lines, colors=colors, linewidths=1)
        # ax.add_collection(lc)
        # ax.scatter(points_x, points_y, color='k', s=2, alpha=.1)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)


    fig.tight_layout()
    os.makedirs('plots/trail-heatmap', exist_ok=True)
    fig.savefig('plots/trail-heatmap/' + dataset_id + '-hm.png', dpi=500)

    return None