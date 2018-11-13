# from sklearn.metrics import mean_squared_error
import numpy as np

import matplotlib.pyplot as plt


def tooth_function(x, mag, center, max_time, transit_time):
    y = np.zeros(x.shape)
    start = center - max_time / 2 - transit_time
    end = center + max_time / 2 + transit_time
    y += np.logical_and(x >= start, x <= start + transit_time) * (
        (mag / transit_time) * (x - start))
    y += np.logical_and(
        x >= start + transit_time, x <= start + transit_time + max_time) * (mag)
    y += np.logical_and(x >= end - transit_time, x <= end) * (
        (mag / transit_time) * (end - x))

    return y


def eclipsing_binary(dates, period, offset, usual, transit_time, dip1, dip2):
    pass


def flat_curve(dates, usual):
    return np.ones(dates.shape) * usual


def analyze_file(fn):
    print(fn)
    days = []
    mags = []
    magerrs = []

    with open(fn) as f:
        for line in f:
            vals = line.split()
            day = float(vals[0])
            mag = float(vals[1])
            magerr = float(vals[2])
            days.append(day)
            mags.append(mag)
            magerrs.append(magerr)

    days = np.array(days)
    mags = np.array(mags)
    magerrs = np.array(magerrs)
    plt.errorbar(days, mags, yerr=magerrs, ecolor='red', fmt='o')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('fns',
                        nargs='+',
                        help="files to process")
    args = parser.parse_args()

    for fn in args.fns:
        analyze_file(fn)
