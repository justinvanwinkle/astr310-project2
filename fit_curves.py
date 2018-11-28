# from sklearn.metrics import mean_squared_error
import numpy as np
import json
import time

import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import chi2


def data_in_tooth(x, center, max_time, transit_time):
    start = center - max_time / 2 - transit_time
    end = center + max_time / 2 + transit_time

    return np.logical_and(x >= start, x <= end).any()


def tooth_function(x, mag, center, max_time, transit_time):
    y = np.zeros(x.shape)
    start = center - max_time / 2 - transit_time
    end = center + max_time / 2 + transit_time
    y += np.logical_and(x >= start, x <= start + transit_time) * (
        (mag / transit_time) * (x - start))
    y += np.logical_and(
        x >= start + transit_time, x <= start + transit_time + max_time) * mag
    y += np.logical_and(x >= end - transit_time, x <= end) * (
        (mag / transit_time) * (end - x))

    return y


def synthetic_eclipsing_binary(dates,
                               period,
                               offset,
                               seperation,
                               baseline,
                               max_time,
                               transit_time,
                               mag1,
                               mag2):
    mod_dates = np.fmod(dates + offset * period, period)
    center1 = period / 2 - seperation * period
    center2 = period / 2 + seperation * period

    data = np.full_like(dates, baseline)
    data += tooth_function(
        mod_dates, mag1, center1, max_time * period, transit_time * period)
    data += tooth_function(
        mod_dates, mag2, center2, max_time * period, transit_time * period)

    return data


def chi_square(A, B, err):
    chi_square = ((A - B)**2 / err**2).sum()
    return chi_square


def p_value(chi_square, df):
    p_value = chi2.sf(chi_square, df)
    return p_value


def fit_curve(dates, mags, magerrs, verbose=False):
    def minimizee(params):
        synthetic_mags = synthetic_eclipsing_binary(dates, *params)
        return chi_square(mags, synthetic_mags, magerrs)

    res = differential_evolution(
        minimizee,
        [(1, 120),  # period
         (0, 1),   # offset fraction
         (.0001, .25),   # seperation fraction
         (0, 20),   # baseline
         (.00001, .3),        # max_time fraction
         (.00001, .1),         # transit_time fraction
         (0, 1),    # mag1
         (0, 1)],   # mag2
        maxiter=1000,
        popsize=400,
        tol=.000000001,
        disp=verbose)

    return res.fun, res.x


def flat_curve(dates, usual):
    return np.ones(dates.shape) * usual


def fit_flat(dates, mags, magerrs, verbose=False):
    def minimizee(params):
        synthetic_mags = flat_curve(dates, *params)
        return chi_square(mags, synthetic_mags, magerrs)

    res = differential_evolution(
        minimizee,
        [(-10, 30)],
        disp=verbose)

    return res.fun, res.x


def analyze_file(fn, check=False):
    start_time = time.time()

    dates = []
    mags = []
    magerrs = []

    with open(fn) as f:
        for line in f:
            vals = line.split()
            if 'null' in vals:
                continue
            day = float(vals[0])
            mag = float(vals[1])
            magerr = float(vals[2])
            dates.append(day)
            mags.append(mag)
            magerrs.append(magerr)

    if not dates:
        return

    notes = dict(fn=fn)

    dates = np.array(dates)
    mags = np.array(mags)
    magerrs = np.array(magerrs)
    # dates = np.linspace(0, 1000, 5000)
    # mags = np.ones(5000) * 500
    # mags += tooth_function(dates, 250, 100, 50, 50)
    # mags += tooth_function(dates, -250, 800, 50, 50)
    # magerrs = np.ones(5000)
    if check:
        plt.errorbar(dates, mags, yerr=magerrs, ecolor='red', fmt='o')
        plt.title(fn)
        plt.show()
        plt.clf()

    data_points = dates.size

    chi2_flat, params = fit_flat(dates, mags, magerrs, check)
    p_flat = p_value(chi2_flat, data_points - 1)
    notes['chi2_flat'] = round(chi2_flat, 6)
    notes['p_flat'] = round(p_flat, 6)

    if p_flat < .80 or check:
        chi2_curve, params = fit_curve(dates, mags, magerrs, check)
        p_curve = p_value(chi2_curve, data_points - 8)
        notes['chi2_curve'] = round(chi2_curve, 6)
        notes['curve_params'] = params.tolist()
        notes['p_curve'] = round(p_curve, 6)

        if check:
            period = params[0]
            offset = params[1]

            synthetic_dates = np.linspace(dates.min(), dates.max(), 10000)
            synthetic_mags = synthetic_eclipsing_binary(synthetic_dates,
                                                        *params)

            plt.scatter(np.fmod(synthetic_dates + offset, period),
                        synthetic_mags,
                        color='grey')
            plt.errorbar(np.fmod(dates + offset, period),
                         mags,
                         yerr=magerrs,
                         ecolor='red',
                         fmt='o')
            plt.title(fn)
            plt.gca().invert_yaxis()
            plt.show()
            plt.clf()

    notes['processing_time'] = round(time.time() - start_time, 2)
    print(json.dumps(notes))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze data file(s).')
    parser.add_argument('fns',
                        nargs='+',
                        help="files to process")
    parser.add_argument('--check',
                        action='store_true',
                        help="evaluate and plot")
    args = parser.parse_args()

    for fn in args.fns:
        try:
            analyze_file(fn, args.check)
        except Exception as e:
            print(e)
