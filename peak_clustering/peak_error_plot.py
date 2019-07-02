
from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import csv_to_dict

from scipy.signal import find_peaks



"""
Load Data and Peak Data
"""
dataGrid = DataGrid_TiNiSn_500C()

data_dir = "/home/sasha/Desktop/TiNiSn_500C_PeakData_0.5/"
#data_dir = "/home/sasha/Desktop/peakTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
peakGrid = DataGrid(data_dir,regex)


save_path = "/home/sasha/Desktop/"

mistakes = csv_to_dict(save_path,"peak_errors")




for grid_location in peakGrid.data.keys():
    fig = plt.figure(figsize =(17,9))
    mistakes[grid_location] = []
    X = dataGrid.data_at_loc(grid_location)[:,0]
    Y = dataGrid.data_at_loc(grid_location)[:,1]

    for peak_x in peakGrid.data_at_loc(grid_location)[:,1]:
        i = (np.abs(X - peak_x)).argmin()
        plt.plot([X[i]],[Y[i]],"x",color='black')
    #loaded as strings so need to be converted
    for loc in eval(mistakes[str(grid_location)]):
        plt.plot([X[loc]],[Y[loc]],'o',color='red')
    plt.plot(X,Y,color='blue')
    plt.show()
