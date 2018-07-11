# -*- coding: utf-8 -*-
"""
The regression modules contains some helpers to perform linear fits on data with non-linear begins or ends.
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
    """
    Tries to find a continuous window in x/y which (mostly) follows a linear relation subject to condition.
    
    If window is a float, it is seen as relative length of the input lists.
    Linear regressions will be performed on each window, then the windows will be filtered by the condition
    (eg that they have a rvalue better than 0.95). Then the range between the first and the last window to
    follow these conditions will be used to perform the overall regression.
    
    See also :py:func:`scipy.stats.linregress`
    
    :param x: Input data, independent value
    :param y: Input data, dependent value
    :param begin: 
    :param end: 
    :param window: Window, either 
    :param condition: Condition to check, a tuple of three. The first must be a key of a 
            linear regression result object, the second either 'gt' or 'lt', and the third the value to compare. 
    :param return_begin_end: If true, return the found range as well
    :param return_nan_if_impossible: If True, return NaN if no suitable region was found, otherwise throws RuntimeError
    :return: 
    
    """
    data = np.c_[x, y]

    data[:, 1][~np.isfinite(data[:, 1])] = np.log(np.finfo(data.dtype).eps)

    if type(window) == float:
        window = int(window * data.shape[0])

    if window < 3:
        window = 3

    pairs = [[i, min(i+window, data.shape[0])] for i in range(0, data.shape[0], window)]
    with np.errstate(all='ignore'):
        results = [[b, e, linregress(data[b:e, 0], data[b:e, 1])] for b, e in pairs]

    condition_check = (lambda r: getattr(r, condition[0]) > condition[2]) if condition[1] == 'gt'\
        else (lambda r: getattr(r, condition[0]) < condition[2])

    filtered_results = [[b, e] for b, e, r in results if condition_check(r)]

    if not filtered_results:
        if return_nan_if_impossible:

            regression = results[0][2]

            # noinspection PyProtectedMember,PyProtectedMember
            regression = regression._replace(**{k: float('nan') for k in regression._asdict().keys()})

            if return_begin_end:
                return float('nan'), float('nan'), regression
            else:
                return regression
        else:
            raise RuntimeError('No optimal range for regression found.')

    if np.isnan(begin):
        begin = filtered_results[0][0]

    if np.isnan(end):
        end = filtered_results[-1][1]

    filtered_data = data[begin:end, :]

    with np.errstate(all='ignore'):
        regression = linregress(filtered_data[:, 0], filtered_data[:, 1])

    if return_begin_end:
        return begin, end, regression
    else:
        return regression


def prepare_optimized_regression(x, y):
    """
    First finds an optimal window using :py:func:`find_linear_window`, than performs a linear regression.
    
    :param x: independent variable 
    :param y: dependent variable
    :return: 
    
    >>> x = np.linspace(1, 100, 100)
    >>> y = x * 5 + 10
    >>> y[0:10] = 0  # break our nice linear curve
    >>> prepare_optimized_regression(x, y)
    OrderedDict([('slope', 5.0), ('intercept', 10.0), ('rvalue', 0.9999999999999999), ('pvalue', 0.0), \
('stderr', 7.942345602646859e-09), ('begin_index', 10), ('end_index', 100), ('begin', 11.0), ('end', 100.0)])
    """
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

