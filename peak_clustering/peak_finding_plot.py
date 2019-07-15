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



data_dir = "/home/sasha/Desktop/peakTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkg_curveParams.csv"""
curveGrid = DataGrid(data_dir,regex)

"""
Smoothing function  (not used)
"""
def smooth(list,k):
    smooth = []
    for i in range(len(list)):
        a = max(i-int(k/2),0)
        b = min(i+int(k/2),len(list)-1)
        smooth.append(sum(list[a:b+1])/(b-a))
    return smooth

smooth_stack = lambda l,k,n : smooth(l,k) if n == 1 else smooth(smooth_stack(l,k,n-1),k)



for grid_location in range(1,dataGrid.size+1):
    fig = plt.figure(figsize =(17,9))
    X = dataGrid.data_at_loc(grid_location)[:,0]
    Y = dataGrid.data_at_loc(grid_location)[:,1]
    Slope = [(Y[i]-Y[i+1])/(X[i] - X[i+1])/100 for i in range(len(X)-1)]


    for peak_x in peakGrid.data_at_loc(grid_location)[:,1]:
        i = (np.abs(X - peak_x)).argmin()
        plt.plot([X[i]],[Y[i]],"x",color='black')

    for peak_x in curveGrid.data_at_loc(grid_location)[:,2]:
        i = (np.abs(X - peak_x)).argmin()
        plt.plot([X[i]],[Y[i]+50],"x",color='red')
    plt.plot(X,Y,color='blue')
    #plt.plot(X[:-1],Slope,color='green')
    #plt.plot(X,[0 for i in X],color='black')
    plt.title(grid_location)
    plt.show()
