import sys
import os
import numpy
import matplotlib
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import argparse as ap
import numpy as np
import re

def extract_floats(string):
    matches = re.findall('[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?',string)
    mlist = list(map(float,matches))
    if len(mlist) > 2:
        return mlist[1:]
    else:
        return mlist

def get_value(path):
    with open(path,'r') as f:
        val = float(f.readline())
    # return val
    return val*33.37

def make_dict(files):

    results = {}
    for dfile in files:
        print(dfile)
        period, radius = extract_floats(dfile)
        frac_absorb = get_value(dfile)
        # if ((radius+.08) * 2) > period:
        #     print('##################')
        #     print('Invalid dimensions')
        #     print('Period: %f'%period)
        #     print('Core Radius: %f'%radius)
        #     print('Total Diameter: %f'%((radius+.08)*2))
        #     print('##################')
        #     frac_absorb = None
        # if frac_absorb is not None and frac_absorb < 0:
        #     print('##################')
        #     print('Absorbance less than zero!')
        #     print('Period: %f'%period)
        #     print('Core Radius: %f'%radius)
        #     print('Total Diameter: %f'%((radius+.08)*2))
        #     print('##################')
        #     frac_absorb = None
        if period in results:
            results[period].append((radius, frac_absorb))
        else:
            results[period] = [(radius, frac_absorb)]
    return results

def make_plot(path):
    radii = np.arange(60,220,10)
    periods = np.arange(280, 700, 20)
    print(radii)
    print(periods)

    glob_str = os.path.join(path,'**/fractional_absorbtion.dat')
    frac_files = glob.glob(glob_str, recursive=True)

    results = make_dict(frac_files)
    key = list(results.keys())[0]
    rows = len(periods)
    cols = len(radii)

    arr = np.zeros((rows,cols))
    row = 0
    for period, data in sorted(results.items()):
        print(period)
        # print('Data Unsorted: %s'%str(data))
        data = sorted(data, key=lambda tup: tup[0])
        # print('Data Sorted: %s'%str(data))
        _radii, absorb_vals = zip(*data)
        _radii = list(_radii)
        absorb_vals = list(absorb_vals)
        # print(_radii)
        while len(_radii) < len(radii):
            _radii.append(None)
            absorb_vals.append(None)
        # print(_radii)
        print(absorb_vals)
        arr[row, :] = absorb_vals 
        row += 1
    print(arr)
    print('Maximum: %f'%np.nanmax(arr))
    sorted_inds = np.argsort(arr, axis=None)
    flat_ind = -1
    ind_tup = np.unravel_index(sorted_inds[flat_ind], arr.shape)
    val = arr[ind_tup[0], ind_tup[1]]
    while np.isnan(val):
        flat_ind -= 1
        ind_tup = np.unravel_index(sorted_inds[flat_ind], arr.shape)
        val = arr[ind_tup[0], ind_tup[1]]
    max_period = periods[ind_tup[0]]
    max_rad = radii[ind_tup[1]]
    ind_tup2 = np.unravel_index(sorted_inds[flat_ind-1], arr.shape)
    max_period2 = periods[ind_tup2[0]]
    max_rad2 = radii[ind_tup2[1]]
    fig, ax = plt.subplots()
    # mat = ax.matshow(arr, aspect='auto', origin='lower')
    mat = ax.matshow(arr, aspect='auto')
    raw_inds = ax.get_yticks()
    raw_inds = list(range(int(raw_inds[0]), int(raw_inds[-1]), 2))
    ticks = []
    labels = [] 
    for ind in raw_inds:
        if 0 <= ind < len(periods):
            ticks.append(ind)
            labels.append(periods[int(ind)])
        # else:
        #     labels.append('')
    # print(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    raw_inds = ax.get_xticks()
    ticks = []
    labels = [] 
    for ind in raw_inds:
        if 0 <= ind < len(radii):
            ticks.append(ind)
            labels.append(radii[int(ind)])
        # else:
        #     labels.append('')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.plot(ind_tup[1], ind_tup[0], 'kx', markersize=12, markeredgewidth=4)
    ax.plot(ind_tup2[1], ind_tup2[0], 'kx', markersize=12, markeredgewidth=4)
    ax.set_xlabel('Radius [nm]')
    ax.xaxis.tick_bottom()
    # ax.xaxis.set_label_position('top')
    ax.set_ylabel('Array Period [nm]')
    cb = fig.colorbar(mat, ax=ax)
    cb.set_label(r"$J_{ph}$ [mA/cm$^2$]")
    plt.savefig('test_map.pdf')
    # plt.show()
    plt.close(fig)

    # fig, ax = plt.subplots()
    # start_ind = (-1,-2,-3,-4,-5,-6,-7)
    # for ind in start_ind:
    #     data = np.diagonal(arr, ind)
    #     # print(data)
    #     ax.plot(data, label='Diagonal %i'%ind)
    # plt.legend(loc='best')
    # plt.savefig('test_diags.pdf')
    # plt.close(fig)

    # fig, ax = plt.subplots()
    # for rad_ind in range(len(radii)):
    #     data = arr[:, rad_ind]
    #     ax.plot(periods, data, label='Radius = %f'%radii[rad_ind])
    # plt.xlabel('Period [nm]')
    # plt.ylabel('Integrated Absorbance')
    # plt.legend(loc='best')
    # plt.savefig('test_radii.pdf')
    # plt.close(fig)

    # fig, ax = plt.subplots()
    # for rad_ind in range(2, len(periods)):
    #     data = arr[rad_ind, :]
    #     # print(data.shape)
    #     # print(radii.shape)
    #     # print(radii)
    #     ax.plot(radii, data, label='Period = %f'%periods[rad_ind])
    # plt.xlabel('Radius [nm]')
    # plt.ylabel('Integrated Absorbance')
    # plt.legend(loc='best')
    # plt.savefig('test_periods.pdf')
    # plt.close(fig)

    # fig, ax = plt.subplots()
    # for rad_ind in range(2, len(periods)-6):
    #     data = arr[rad_ind, :]
    #     print(data.shape)
    #     print(radii.shape)
    #     print(radii)
    #     ax.plot(radii, data, label='Period = %f'%periods[rad_ind])
    # plt.xlabel('Radius [nm]')
    # plt.ylabel('Integrated Absorbance')
    # plt.legend(loc='best')
    # plt.savefig('test_periods_beginning.pdf')
    # plt.close(fig)

def main():

    parser = ap.ArgumentParser(description="""Makes heatmap of integrated
                               absorbance with period and radius """)
    parser.add_argument('path',type=str,help="Path to sweep dir")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        raise ValueError("Supplied path doesn't exist")

    make_plot(args.path)

if __name__ == '__main__':
    main()

