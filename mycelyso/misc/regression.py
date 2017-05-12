# -*- coding: utf-8 -*-
"""
documentation
"""

import numpy as np
from scipy.stats import linregress


def find_linear_window(
        x,
        y,
        begin=float('nan'),
        end=float('nan'),
        window=0.1,
        condition=('rvalue', 'gt', 0.95),
        return_begin_end=False,
        return_nan_if_impossible=True
):
    data = np.c_[x, y]

    data[:, 1][~np.isfinite(data[:, 1])] = 0.0

    if type(window) == float:
        window = int(window * data.shape[0])

    if window < 3:
        window = 3

    pairs = [[i, min(i+window, data.shape[0])] for i in range(0, data.shape[0], window)]
    results = [[b, e, linregress(data[b:e, 0], data[b:e, 1])] for b, e in pairs]

    condition_check = (lambda r: getattr(r, condition[0]) > condition[2]) if condition[1] == 'gt'\
        else (lambda r: getattr(r, condition[0]) < condition[2])

    filtered_results = [[b, e] for b, e, r in results if condition_check(r)]

    if filtered_results:
        if return_nan_if_impossible:

            regression = results[0][2]

            # noinspection PyProtectedMember,PyProtectedMember
            regression = regression._replace(**{k: float('nan') for k in regression._asdict().keys()})

            if return_begin_end:
                return float('nan'), float('nan'), regression
            else:
                return regression

    if np.isnan(begin):
        begin = filtered_results[0][0]

    if np.isnan(end):
        end = filtered_results[-1][1]

    filtered_data = data[begin:end, :]

    regression = linregress(filtered_data[:, 0], filtered_data[:, 1])

    if return_begin_end:
        return begin, end, regression
    else:
        return regression


def prepare_optimized_regression(x, y):
    condition = ('rvalue', 'gt', 0.9)
    begin, end, regression = find_linear_window(x, y, condition=condition, return_begin_end=True)
    # noinspection PyProtectedMember
    regression = regression._asdict()
    regression['begin_index'] = begin
    regression['end_index'] = end

    if end == len(x):
        end -= 1

    if (begin != begin or end != end) or begin < 0 or end >= len(x):
        regression['begin'] = regression['end'] = -1
    else:
        regression['begin'] = x[begin]
        regression['end'] = x[end]
    return regression

