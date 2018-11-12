import sys

from Util import Parser
from Util import Metrics
from Visualization import Boxplots
from Visualization import KDE
from Visualization import Matrix
from Visualization import Scatter
from Visualization import SimpleTrails
from Visualization import DirectedTrails
from Visualization import TrailHeatmap

action = sys.argv[1]


if action == 'cache-metrics':
    dataset_id = sys.argv[2]
    Metrics.compute_and_cache_metrics(dataset_id)

elif action == 'kde-ct':
    dataset_id = sys.argv[2]
    ct_values = Parser.read_ct_metric(dataset_id)
    KDE.plot_real_vs_baseline(ct_values, dataset_id, 'ct', True)
    print('---')
    KDE.plot_real_vs_baseline(ct_values, dataset_id, 'ct', False)
    print('---')

elif action == 'kde-rpc':
    dataset_id = sys.argv[2]
    rpc_values = Parser.read_rpc_metric(dataset_id)
    KDE.plot_real_vs_baseline(rpc_values, dataset_id, 'rpc', True)
    print('---')
    KDE.plot_real_vs_baseline(rpc_values, dataset_id, 'rpc', False)
    print('---')

elif action == 'boxplots':
    dataset_id = sys.argv[2]
    ar_values = Parser.read_aspect_ratios(dataset_id)
    Boxplots.plot_unweighted_ar(ar_values, dataset_id)
    print('---')
    Boxplots.plot_weighted_ar(ar_values, dataset_id)
    print('---')

    ct_values = Parser.read_ct_metric(dataset_id)
    Boxplots.plot_instability(ct_values, dataset_id, 'ct')
    print('---')

    rpc_values = Parser.read_rpc_metric(dataset_id)
    Boxplots.plot_instability(rpc_values, dataset_id, 'rpc')
    print('---')

elif action == 'scatter':
    dataset_ids = sys.argv[2:]
    Scatter.plot(dataset_ids, True, False)
    # Scatter.plot(dataset_ids, False, True)
    # Scatter.plot(dataset_ids, True, True)

elif action == "matrix":
    dataset_ids = sys.argv[2:]
    Matrix.plot(dataset_ids)

elif action == "simple-trails":
    dataset_id = sys.argv[2]
    dataframes = Parser.parse_dataset(dataset_id)
    SimpleTrails.plot(dataframes, dataset_id)

elif action == "directed-trails":
    dataset_id = sys.argv[2]
    dataframes = Parser.parse_dataset(dataset_id)
    DirectedTrails.plot(dataframes, dataset_id)

elif action == "trail-heatmap":
    dataset_id = sys.argv[2]
    dataframes = Parser.parse_dataset(dataset_id)
    TrailHeatmap.plot(dataframes, dataset_id)