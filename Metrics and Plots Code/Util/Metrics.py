import math
import os.path

import numpy as np
import pandas as pd

from Util import Parser


def compute_and_cache_metrics(dataset_id):
    # Returns a dict (one entry for each technique) with lists of dataframes
    dataframes = Parser.parse_dataset(dataset_id)

    for technique, df_list in dataframes.items():
        print(technique, end='')
        # Compute aspect ratios and weight of cells
        ar_cache_path = 'metric_results/' + technique + '-' + dataset_id + '-ar.csv'
        ar_df = pd.DataFrame()
        if not os.path.isfile(ar_cache_path):
            for revision, df in enumerate(df_list):
                weight = compute_relative_weight(df, revision)
                ar = compute_aspect_ratio(df, revision)
                ar_df = pd.merge(ar_df, weight, how='outer', left_index=True, right_index=True)
                ar_df = pd.merge(ar_df, ar, how='outer', left_index=True, right_index=True)
            ar_df.fillna(0, inplace=True)
            ar_df.to_csv(ar_cache_path, index_label='id')

        # Compute Corner Travel (real and baseline)
        ct_cache_path = 'metric_results/' + technique + '-' + dataset_id + '-ct.csv'
        ct_df = pd.DataFrame()
        if not os.path.isfile(ct_cache_path):
            for revision in range(len(df_list) - 1):
                r0 = df_list[revision][['rx', 'ry', 'rw', 'rh']].dropna(axis=0, subset=['rx'])
                r1 = df_list[revision + 1][['rx', 'ry', 'rw', 'rh']].dropna(axis=0, subset=['rx'])
                b1 = df_list[revision + 1][['bx', 'by', 'bw', 'bh']].dropna(axis=0, subset=['bx'])
                ct = corner_travel_values(r0, r1, b1, revision)
                ct_df = pd.merge(ct_df, ct, how='outer', left_index=True, right_index=True)
            ct_df.fillna(0, inplace=True)
            ct_df.to_csv(ct_cache_path, index_label='id')

        # Compute Relative Position Change metric
        rpc_cache_path = 'metric_results/' + technique + '-' + dataset_id + '-rpc.csv'
        rpc_df = pd.DataFrame()
        if not os.path.isfile(rpc_cache_path):
            for revision in range(len(df_list) - 1):
                real = relative_position_change_wrapper(df_list[revision][['rx', 'ry', 'rw', 'rh']],
                                                        df_list[revision + 1][['rx', 'ry', 'rw', 'rh']])

                baseline = relative_position_change_wrapper(df_list[revision][['rx', 'ry', 'rw', 'rh']],
                                                            df_list[revision + 1][['bx', 'by', 'bw', 'bh']])

                df_temp = pd.DataFrame({'r_' + str(revision): real, 'b_' + str(revision): baseline})
                df_temp.sort_index(axis=1, ascending=False, inplace=True)
                rpc_df = pd.merge(rpc_df, df_temp, how='outer', left_index=True, right_index=True)
            rpc_df.fillna(0, inplace=True)
            rpc_df.to_csv(rpc_cache_path, index_label='id')

        print(' done.')
    return None


def compute_aspect_ratio(df, revision):
    temp_df = (df[['rw', 'rh']].min(axis=1) / df[['rw', 'rh']].max(axis=1)).to_frame()
    temp_df.columns = ['ar_' + str(revision)]
    return temp_df


def compute_relative_weight(df, revision):
    total_area = (df['rx'] + df['rw']).max() * (df['ry'] + df['rh']).max()
    temp_df = ((df['rw'] * df['rh']) / total_area).to_frame()
    temp_df.columns = ['w_' + str(revision)]
    return temp_df

#####################################################
# Corner travel based matrix
def point_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def corner_travel(*args):
    x1, y1, w1, h1, x2, y2, w2, h2 = args
    if math.isnan(w1):
        return math.nan
        # 2 times the hypotenuse -- growth from center
        #return 2 * math.sqrt(w2 ** 2 + h2 ** 2)
    elif math.isnan(w2):
        #return 2 * math.sqrt(w1 ** 2 + h1 ** 2)
        return math.nan
    else:
        return point_distance(x1, y1, x2, y2) \
               + point_distance(x1 + w1, y1, x2 + w2, y2) \
               + point_distance(x1, y1 + h1, x2, y2 + h2) \
               + point_distance(x1 + w1, y1 + h1, x2 + w2, y2 + h2)


def corner_travel_values(df0, df1, bl1, revision):
    # Normalize by 4 * hypotenuse
    base_width = (df0['rx'] + df1['rw']).max()
    base_height = (df0['ry'] + df1['rh']).max()
    norm = 4 * math.sqrt(base_width ** 2 + base_height ** 2)

    # Create dataframes with real coordinates and real-baseline coordinates
    dfr = pd.merge(df0, df1, how='outer', left_index=True, right_index=True)
    dfb = pd.merge(df0, bl1, how='outer', left_index=True, right_index=True)

    r_m = dfr.apply(lambda r: corner_travel(*list(r)) / norm, axis=1)
    b_m = dfb.apply(lambda r: corner_travel(*list(r)) / norm, axis=1)

    # Additions and removals have baseline = 0 and real = 0
    results = pd.concat([r_m, b_m], axis=1, join='outer')
    results = results.dropna(axis=0, how='any')
    results.columns = ['r_' + str(revision), 'b_' + str(revision)]
    # This is the diff metric
    # results['ct_'+str(revision)] = results['real'] - results['baseline']
    # results['ct_'+str(revision)] = results[['ct_'+str(revision)]].clip(0, 1)
    # display(results)
    return results


#####################################################
# Relative Position Change Metric from 'Stable Treemaps via Local Moves' 2017
#  - Contains a list of the rectangles in the current and in the new iteration. Only those rectangles present in both are there
#    a object in the list should contain "x1","x2","y1","y2" values
def relative_position_change_wrapper(df1, df2):
    df = pd.merge(df1, df2, how='inner', left_index=True, right_index=True)  # Retain only rows in both sets
    df.columns = ['x1', 'y1', 'w1', 'h1', 'x2', 'y2', 'w2', 'h2']

    df['w1'] = df['x1'] + df['w1']
    df['h1'] = df['y1'] + df['h1']

    df['w2'] = df['x2'] + df['w2']
    df['h2'] = df['y2'] + df['h2']

    # Coords from 1st revision and coords from 2nd revision
    df.columns = ['x11', 'y11', 'x12', 'y12', 'x21', 'y21', 'x22', 'y22']

    scores = get_relative_score(df)
    return scores


def get_relative_score(df):
    m = df.as_matrix()
    N = len(m)
    scores = pd.Series(np.zeros(N), index=df.index)

    revision_stability = 0
    for i in range(N):
        item_stability = 0
        for j in range(N):
            if i != j:
                old_percentage = getRelativePositions(m[i][0], m[i][2], m[i][1], m[i][3],
                                                      m[j][0], m[j][2], m[j][1], m[j][3])
                new_percentage = getRelativePositions(m[i][4], m[i][6], m[i][5], m[i][7],
                                                      m[j][4], m[j][6], m[j][5], m[j][7])
                pair_stability = getQuadrantStability(old_percentage, new_percentage)
                item_stability += pair_stability
                revision_stability += pair_stability
        if N > 1:
            scores.iloc[i] = (item_stability / (N - 1))
        else:
            scores.iloc[i] = 0
    # revision_stability = revision_stability / (pow(N, 2) - N)
    return scores


def getQuadrantStability(percentagesOld, percentagesNew):
    stability = 0
    for i in range(0, 8):
        oldPercentage = percentagesOld[i]
        newPercentage = percentagesNew[i]
        stability += abs(oldPercentage - newPercentage) / 2
    return stability


# gets the relative positions per quadrant from r1 to r2
def getRelativePositions(x11, x12, y11, y12, x21, x22, y21, y22):
    E = 0
    NE = 0
    N = 0
    NW = 0
    W = 0
    SW = 0
    S = 0
    SE = 0

    if (x21 >= x12):
        # Strictly east
        if (y21 < y11):
            # at least partially in NE
            # get the percentage that r2 is in NE
            NE = (y11 - y21) / (y22 - y21)
            NE = min(1, NE)

        if (y22 > y12):
            # at least partiall in SE
            SE = (y22 - y12) / (y22 - y21)
            SE = min(1, SE)

            # remainder is in east
            E = 1 - NE - SE
    elif (x22 <= x11):
        # strictly west
        if (y21 < y11):
            # at least partially in NW
            # get the percentage that r2 is in NW
            NW = (y11 - y21) / (y22 - y21)
            NW = min(1, NW)

        if (y22 > y12):
            # at least partiall in SW
            SW = (y22 - y12) / (y22 - y21)
            SW = min(1, SW)

            # remainder is in west
            W = 1 - NW - SW
    elif (y22 <= y11):
        # strictly North
        if (x21 < x11):
            # at least partially in NW
            # get the percentage that r2 is in NW
            NW = (x11 - x21) / (x22 - x21)
            NW = min(1, NW)

        if (x22 > x12):
            # at least partiall in SW
            NE = (x22 - x12) / (x22 - x21)
            NE = min(1, NE)

            # remainder is in west
            N = 1 - NW - NE
    else:
        # strictly south
        if (x21 < x11):
            # at least partially in SW
            # get the percentage that r2 is in NW
            SW = (x11 - x21) / (x22 - x21)
            SW = min(1, SW)

        if (x22 > x12):
            # at least partiall in SE
            SE = (x22 - x12) / (x22 - x21)
            SE = min(1, SE)

            # remainder is in west
            S = 1 - SW - SE

    quadrant = []
    quadrant.append(E)
    quadrant.append(NE)
    quadrant.append(N)
    quadrant.append(NW)
    quadrant.append(W)
    quadrant.append(SW)
    quadrant.append(S)
    quadrant.append(SE)
    return quadrant