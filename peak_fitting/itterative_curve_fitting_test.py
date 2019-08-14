

import matplotlib.pyplot as plt
import numpy as np

from peak_fitting.itterative_curve_fitting import fit_curves_to_data
from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
dataGrid = DataGrid_TiNiSn_500C()
#81 is concerning
for loc in range(81,dataGrid.size,10):
    X = dataGrid.data_at_loc(loc)[:,0]
    Y = dataGrid.data_at_loc(loc)[:,1]

    dict = fit_curves_to_data(X,Y)
    param_list = dict['curve_params']
    change_points = dict['change_points']
    voigt = dict['profile']
    [plt.axvline(X[c],color="red") for c in change_points]

    plt.plot(X,Y)
    for params in param_list:
        curve = lambda x : voigt(x,*params)
        plt.plot(X,[curve(x) for x in X])
    plt.show()
