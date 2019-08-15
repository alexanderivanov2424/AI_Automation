

import matplotlib.pyplot as plt
import numpy as np

from peak_fitting.itterative_curve_fitting import fit_curves_to_data,save_data_to_csv
from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
dataGrid = DataGrid_TiNiSn_500C()
#81 is concerning
#151 is bad
for loc in range(1,dataGrid.size,10):
    X = dataGrid.data_at_loc(loc)[:,0]
    Y = dataGrid.data_at_loc(loc)[:,1]

    dict = fit_curves_to_data(X,Y,1.5,1.81)

    save_data_to_csv("/home/sasha/Desktop/itterative_curve_fitting_save_test/params_" + str(loc) + ".csv",dict)

    param_list = dict['curve_params']
    change_points = dict['change_points']
    residuals = dict['residuals']
    voigt = dict['profile']
    block_curves = dict['block curves']
    block_fits = dict['block fits']

    # plot blocks
    [plt.axvline(X[c],color="red") for c in change_points]

    #block block curves
    plt.plot(X,Y,color="black")

    #plot curves in each block
    for block_curve in block_curves:
        plt.plot(*block_curve,color="orange")

    #plot block residuals
    for resid in residuals:
        plt.plot(*resid,color="green")

    #plot fit for each block
    for block_fit in block_fits:
        plt.plot(*block_fit,color="blue")


    plt.title(str(loc))
    plt.savefig("/home/sasha/Desktop/itterative_curve_fitting_save_test/plot_" + str(loc) + ".png")
    plt.show()
