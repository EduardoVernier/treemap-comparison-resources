import os
import re
import pandas as pd

base_path = '/home/eduardo/Desktop/outputNewClassifications/'
metric_path = '/home/eduardo/PycharmProjects/treemap-metrics/metric_results/'


def parse_dataset(dataset_id):
    # Separate baseline dirs from real dirs
    rectangles_dirs = [d[0] for d in os.walk(base_path)]
    rectangles_dirs.sort()
    baseline_dirs = [d for d in rectangles_dirs if dataset_id in d and "baseLine" in d]
    real_dirs = [d for d in rectangles_dirs if dataset_id in d and "baseLine" not in d]

    technique_ids = []
    rectangles = {}
    for i in range(len(real_dirs)):
        # Match technique name from path
        match = re.match(base_path + '(\w+)', real_dirs[i])
        technique_id = ''
        if match:
            technique_id = match.group(1)
            technique_ids.append(technique_id)
        else:
            print("Invalid path: ", real_dirs[i])

        # Match file paths
        real_rectangles_paths = [os.path.join(real_dirs[i], f) for f in os.listdir(real_dirs[i]) if
                                 os.path.isfile(os.path.join(real_dirs[i], f))]
        real_rectangles_paths = natural_sort(real_rectangles_paths)

        baseline_rectangles_paths = [os.path.join(baseline_dirs[i], f) for f in os.listdir(baseline_dirs[i]) if
                                     os.path.isfile(os.path.join(baseline_dirs[i], f))]
        baseline_rectangles_paths = natural_sort(baseline_rectangles_paths)

        dataframes = []
        # Real i matches with Baseline i-1
        # Revision 0 doesn't have a Baseline, but we'll need the Real layout for Aspect Ratio statistics

        real_df = pd.read_csv(real_rectangles_paths[0], header=None)
        column_names = ['id', 'rx', 'ry', 'rw', 'rh']
        real_df.columns = column_names
        real_df.set_index('id', inplace=True)
        dataframes.append(real_df)

        # Read remaining revisions
        for j in range(len(real_rectangles_paths) - 1):
            # print("joining ", real_rectangles_paths[j+1], baseline_rectangles_paths[j])
            # Read Real
            column_names = ['id', 'rx', 'ry', 'rw', 'rh']
            real_df = pd.read_csv(real_rectangles_paths[j + 1], header=None)
            real_df.columns = column_names
            real_df.set_index('id', inplace=True)
            # Read Baseline
            # print(baseline_rectangles_paths[j])
            column_names = ['id', 'bx', 'by', 'bw', 'bh']
            baseline_df = pd.read_csv(baseline_rectangles_paths[j], header=None)
            baseline_df.columns = column_names
            baseline_df.set_index('id', inplace=True)

            # Join the two tables. Remove items that don't have real coordinates (inflated or heuristic fragment)
            df = pd.concat([real_df, baseline_df], axis=1, join='outer')
            df = df.dropna(axis=0, subset=['rx'])

            dataframes.append(df)

        rectangles[technique_id] = dataframes

    return rectangles


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def read_ct_metric(dataset_id):
    ct_values = {}

    paths = [os.path.join(metric_path, f) for f in os.listdir(metric_path)
             if os.path.isfile(os.path.join(metric_path, f)) and '-' + dataset_id + '-' in f]

    for path in paths:
        # Match technique name from path
        match = re.match(metric_path + '(\w+)-' + dataset_id +'-ct', path)
        if match:
            # Read csvs from matched paths
            technique_id = match.group(1)
            ct_values[technique_id] = pd.read_csv(path, index_col='id')

    return ct_values


def read_rpc_metric(dataset_id):
    rpc_values = {}

    paths = [os.path.join(metric_path, f) for f in os.listdir(metric_path)
             if os.path.isfile(os.path.join(metric_path, f)) and '-' + dataset_id + '-' in f]

    for path in paths:
        # Match technique name from path
        match = re.match(metric_path + '(\w+)-' + dataset_id +'-rpc', path)
        if match:
            # Read csvs from matched paths
            technique_id = match.group(1)
            rpc_values[technique_id] = pd.read_csv(path, index_col='id')

    return rpc_values


def read_aspect_ratios(dataset_id):
    ar_values = {}

    paths = [os.path.join(metric_path, f) for f in os.listdir(metric_path)
             if os.path.isfile(os.path.join(metric_path, f)) and '-' + dataset_id + '-' in f]

    for path in paths:
        # Match technique name from path
        match = re.match(metric_path + '(\w+)-' + dataset_id +'-ar', path)
        if match:
            # Read csvs from matched paths
            technique_id = match.group(1)
            ar_values[technique_id] = pd.read_csv(path, index_col='id')

    return ar_values
