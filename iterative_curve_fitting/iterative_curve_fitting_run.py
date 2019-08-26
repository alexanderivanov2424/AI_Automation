"""
Iterative Curve Fitting Algorithm Example

Uses files in given directories to perform iterative curve fitting
and saves results.

Independent code. Does not rely on dataGrid object

"""


import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from iterative_curve_fitting.iterative_curve_fitting import fit_curves_to_data,save_data_to_csv

path_to_files = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C"
path_to_save_dir = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/params"




"""
Noise Level
Peaks below this intensity will not be considered
"""
NOISE = 5


"""
Max Curve Fit
Maximum number of curve that should be fit to a block
"""
max_curves=30

"""
Smallest allowable block
"""
MIN_BLOCK_SIZE = 20


"""
If false plots are saved but not displayed
"""
SHOW_PLOTS = False




try:
    os.stat(path_to_save_dir)
except:
    print("Creating directory: " + path_to_save_dir)
    os.mkdir(path_to_save_dir)

files = os.listdir(path_to_files)


for file in files:
    if not os.path.splitext(file)[1] == ".csv":
        print("Skipping: " + file)
        continue
    print("Reading File: " + file)

    try:
        data_array = np.array(pd.read_csv(os.path.join(path_to_files, file),header=None))
    except:
        print("## Bad Formatting in file: " + file)
        continue
    try:
        data_array[0].astype(np.float)
    except:
        data_array = data_array[1:]
    data = data_array.astype(np.float)
    #load X and Y data for a diffraction pattern
    X = data[:,0]
    Y = data[:,1]

    print("## Fitting ...")
    dict = fit_curves_to_data(X,Y,1.5,1.81,noise=NOISE,max_curves=max_curves,min_block_size=MIN_BLOCK_SIZE)

    param_list = dict['curve_params']
    change_points = dict['change_points']
    residuals = dict['residuals']
    voigt = dict['profile']
    block_curves = dict['block curves']
    block_fits = dict['block fits']
    fit = dict['fit']


    fig = plt.figure(figsize=(20,10))

    # plot blocks
    [plt.axvline(X[c],color="red") for c in change_points]

    #block block curves
    plt.plot(X,Y,color="black")

    #plot curves in each block
    for block_curve in block_curves:
        plt.plot(*block_curve,color="orange")


    #plot block residuals
    combined_resid = 0
    for resid in residuals:
        combined_resid += np.sum(np.abs(resid[1])) #sum y axis in residuals
        plt.plot(*resid,color="green")
    print("## Integrated Residual: ", combined_resid)

    #plot fit for each block
    for block_fit in block_fits:
        plt.plot(*block_fit,color="blue")

    #plot fit curve
    #(different from combined block fits at change points)
    #plt.plot(X,[fit(x) for x in X],color="red")

    #plot peak points
    for params in param_list:
        lim = .05
        #if params[2] < lim and params[3] < lim:
        x = params[1]
        y = fit(x)
        plt.plot([x],[y],'ro')

    from matplotlib.lines import Line2D

    plt.title(os.path.splitext(file)[0])
    L1 = mpatches.Patch(color='black', label='Data')
    L2 = mpatches.Patch(color='blue', label='Curve fit')
    L3 = mpatches.Patch(color='green', label='Residual')
    L4 = mpatches.Patch(color='red', label='Blocks')
    L5 = mpatches.Patch(color='orange', label='Individual Curves')


    plt.legend(handles=[L1,L2,L3,L4,L5])



    plt.savefig(os.path.join(path_to_save_dir,os.path.splitext(file)[0] + ".png"))
    print("## Plot saved to: " + os.path.join(path_to_save_dir,os.path.splitext(file)[0] + "_plot.png"))

    save_data_to_csv(os.path.join(path_to_save_dir,os.path.splitext(file)[0] + "_params.csv"),dict)
    print("## Params saved to: " + os.path.join(path_to_save_dir,os.path.splitext(file)[0] + "_params.csv"))

    if SHOW_PLOTS:
        plt.show()

    plt.cla()
    plt.close()
