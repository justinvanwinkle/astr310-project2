from sklearn.metrics import mean_squared_error
import numpy as np


def eclipsing_binary(dates, period, offset, usual, transit_time, dip1, dip2):
    pass


def flat_curve(dates, usual):
    return np.ones(dates.shape) * usual


def analyze_file(fn):
    days = []
    mags = []
    magerrs = []

    with open(fn) as f:
        for line in f:
            day, mag, magerr = [float(x) for x in line.split()[:3]]
            days.append(day)
            mags.append(mags)
            magerrs.append(magerr)

    days = np.array(days)
    mags = np.array(mags)
    magerrs = np.array(magerrs)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('fns',
                        nargs='+',
                        help="files to process")
    args = parser.parse_args()

    for fn in args.fns:
        analyze_file(fn)
