import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm
from scipy import stats
from mpl_toolkits.axes_grid1 import ImageGrid
import os

from Util import Globals


def plot_real_vs_baseline(values, dataset_id, base_metric, log):
    nrow = 7
    ncol = 2
    fig = plt.figure(1, (8, 14))
    fig.subplots_adjust(left=0.08, right=0.99, top=.99, bottom=.01)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrow, ncol),
                     axes_pad=0.05,
                     share_all=True,
                     label_mode="1")

    technique_list = sorted(values)
    for i, technique_id in enumerate(technique_list):
        # print(technique_id)
        ax = grid[i]
        #ax.set_title(technique_id)

        full_df = values[technique_id]
        if full_df.columns[0] == 'r_0':
            real_series = full_df.iloc[:, ::2].stack().values.clip(0,1)
            baseline_series = full_df.iloc[:, 1::2].stack().values.clip(0,1)
        else:
            real_series = full_df.iloc[:, 1::2].stack().values.clip(0, 1)
            baseline_series = full_df.iloc[:, ::2].stack().values.clip(0, 1)

        baseline_series += np.random.normal(0, .001, baseline_series.shape) # Adding a little bit of noise to the baseline
        # Plot all points in black with alpha
        ax.scatter(real_series, baseline_series, color='k', s=2, alpha=.1)

        # Plot points with color density
        sample_size = 5000 if len(real_series) > 5000 else len(real_series)
        matrix = pd.DataFrame([real_series, baseline_series]).sample(sample_size, axis=1).as_matrix()
        dens = stats.gaussian_kde(matrix)
        dens_pt = dens(matrix)
        colours = make_colors(dens_pt, 'inferno', log)
        ax.scatter(matrix[0], matrix[1], color=colours, s=2, alpha=.25)
        ax.text(.97, .9, Globals.acronyms[technique_id], horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

        ax.set_xlim(xmin=0, xmax=1)
        ax.set_ylim(ymin=0, ymax=.3)
        plt.axis('equal')

    colormap_mode_str = 'log' if log else 'linear'
    os.makedirs('plots/kde/' + base_metric + '/', exist_ok=True)
    fig.savefig('plots/kde/' + base_metric + '/' + dataset_id + '-' + base_metric + '-' + colormap_mode_str + '-kde' + '.png', bbox_inches='tight', dpi=300)
    # plt.draw()
    # plt.show()
    return None


def make_colors(vals, cmap, log):
    if log:
        norm = LogNorm(vmin=vals.min()*2, vmax=vals.max()**1.5)
    else:
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
    return [cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(val) for val in vals]
