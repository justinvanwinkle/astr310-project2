# from sklearn.metrics import mean_squared_error
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution


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
    def minimizee(params):
        synthetic_mags = synthetic_eclipsing_binary(dates, *params)
        return rms(mags, synthetic_mags, magerrs)

    max_period = dates.max() - dates.min()
    mean_mag = mags.mean()
    res = differential_evolution(
        minimizee,
        [(.1, max_period), (0, max_period/2), (0, mean_mag * 2), (.00001, 10), (.00001, 2), (-mean_mag/5, 0), (-mean_mag/5, 0)],
        disp=False)

    return res.fun, res.x


def flat_curve(dates, usual):
    return np.ones(dates.shape) * usual


def analyze_file(fn):
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

    rms, params = fit_curve(dates, mags, magerrs)
    period = params[0]

    synthetic_dates = np.linspace(dates.min(), dates.max(), 5000)
    synthetic_mags = synthetic_eclipsing_binary(synthetic_dates, *params)

    print(rms, fn)

    # plt.errorbar(np.fmod(dates, period), mags, yerr=magerrs, ecolor='red', fmt='o')
    # plt.plot(np.fmod(synthetic_dates, period), synthetic_mags)
    # plt.show()
    # plt.clf()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze data file(s).')
    parser.add_argument('fns',
                        nargs='+',
                        help="files to process")
    args = parser.parse_args()

    for fn in args.fns:
        analyze_file(fn.strip())
