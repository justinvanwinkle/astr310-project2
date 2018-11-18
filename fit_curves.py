# from sklearn.metrics import mean_squared_error
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import basinhopping


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


def synthetic_eclipsing_binary(dates, period, offset, baseline, max_time, transit_time, mag1, mag2):
    mod_dates = np.fmod(dates + offset, period)
    center1 = period / 4
    center2 = 3 * period / 4

    data = np.full_like(dates, baseline)
    data += tooth_function(mod_dates, mag1, center1, max_time, transit_time)
    data += tooth_function(mod_dates, mag2, center2, max_time, transit_time)

    return data


def rms(A, B, err):
    return ((A - B)**2 / err).sum()


def fit_curve(dates, mags, magerrs):
    params, pcov = curve_fit(synthetic_eclipsing_binary, dates, mags, sigma=magerrs)


def flat_curve(dates, usual):
    return np.ones(dates.shape) * usual


def analyze_file(fn):
    print(fn)
    dates = []
    mags = []
    magerrs = []

    with open(fn) as f:
        for line in f:
            vals = line.split()
            day = float(vals[0])
            mag = float(vals[1])
            magerr = float(vals[2])
            dates.append(day)
            mags.append(mag)
            magerrs.append(magerr)

    dates = np.array(dates)
    mags = np.array(mags)
    magerrs = np.array(magerrs)
    # dates = np.linspace(0, 1000, 5000)
    # mags = np.ones(5000) * 500
    # mags += tooth_function(dates, 250, 100, 50, 50)
    # mags += tooth_function(dates, -250, 800, 50, 50)
    # magerrs = np.ones(5000)

    fit_curve(dates, mags, magerrs)
    plt.errorbar(dates, mags, yerr=magerrs, ecolor='red', fmt='o')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze data file(s).')
    parser.add_argument('fns',
                        nargs='+',
                        help="files to process")
    args = parser.parse_args()

    for fn in args.fns:
        analyze_file(fn)
