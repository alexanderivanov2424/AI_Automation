

from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
from peak_fitting.fit_data_sample import fit_curves_to_data, get_peak_indices

import matplotlib.pyplot as plt
import matplotlib

from scipy.optimize import curve_fit

import numpy as np
import math
import statistics

"""
######################
Plotting found peaks on the diffraction patterns
######################
"""

"""
Load Data and Peak Data
"""
dataGrid = DataGrid_TiNiSn_500C()

data_dir = "/home/sasha/Desktop/TiNiSn_500C_PeakData_0.5/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
peakBBAGrid = DataGrid(data_dir,regex)

data_dir = "/home/sasha/Desktop/peakTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkg_curveParams.csv"""
curveBBAGrid = DataGrid(data_dir,regex)


draw_peaks = False
draw_curves = False
draw_min_peaks = False

percent_trim = 3
percent_trim = percent_trim / 100

#load Min block peak data
peakMinGrid = DataGrid(data_dir,regex)
data_dir = "/home/sasha/Desktop/peakData_temp/"
for loc in range(1,dataGrid.size+1):
    file = data_dir + str(loc) + ".txt"
    try:
        peaks = eval(open(file).read())
    except:
        print("Generating " + str(loc))
        X = dataGrid.data_at_loc(loc)[:,0]
        Y = dataGrid.data_at_loc(loc)[:,1]

        #curve_params = fit_curves_to_data(X,Y)
        peaks = get_peak_indices(X,Y) ##Use Min blocks to find peaks
        open(file,'w+').write(str(peaks))
    peakMinGrid.data[loc] = peaks


background_start = 1.5
background_end = 1.7

def fourier(x, *a):
    ret = a[0] * np.cos(a[1] * x) + a[2]
    return ret

def fourier(x, *a):
    ret = a[0] * np.cos(10/x)
    for deg in range(1, len(a)):
        ret += a[deg] * np.cos(10*(deg+1) /x)
    return ret

for loc in dataGrid.data.keys():
    X = dataGrid.data_at_loc(loc)[:,0]
    Y = dataGrid.data_at_loc(loc)[:,1]
    s = np.argmin(np.abs(X - background_start))
    e = np.argmin(np.abs(X - background_end))
    X_trim = dataGrid.data_at_loc(loc)[s:e,0]
    Y_trim = dataGrid.data_at_loc(loc)[s:e,1]

    #fit fourier curve
    popt, pcov = curve_fit(lambda x,a,b,c,d : a*np.cos(b*x + c) + d, X_trim, Y_trim,[1,10,0,20])


    #create resid
    resid = Y_trim - fourier(X_trim, *popt)
    plt.plot(X_trim,Y_trim)
    plt.plot(X_trim, fourier(X_trim, *popt))


    sd = statistics.stdev(resid) * 3

    plt.fill_between(X,Y+sd,Y-sd,facecolor='red', alpha=0.5)
    plt.plot(X,Y,color="blue")

    if draw_peaks:
        max_p = np.max(peakBBAGrid.data_at_loc(loc)[:,2])
        for x in peakBBAGrid.data_at_loc(loc)[:,1]:
            i = np.argmin(np.abs(X - x))
            if Y[i]/max_p > percent_trim:
                plt.plot(X[i],Y[i],marker="o",color="red")
            else:
                plt.plot(X[i],Y[i],marker="o",color="black")
    if draw_curves:
        max_p = 0
        for x in curveBBAGrid.data_at_loc(loc)[:,2]:
            i = np.argmin(np.abs(X - x))
            max_p = max(max_p,Y[i])
        for x in curveBBAGrid.data_at_loc(loc)[:,2]:
            i = np.argmin(np.abs(X - x))
            if Y[i]/max_p > percent_trim:
                plt.plot(X[i],Y[i],marker="o",color="red")
            else:
                plt.plot(X[i],Y[i],marker="o",color="black")

    if draw_min_peaks:
        max_p = 0
        for x in curveBBAGrid.data_at_loc(loc)[:,2]:
            i = np.argmin(np.abs(X - x))
            max_p = max(max_p,Y[i])
        for i in peakMinGrid.data_at_loc(loc)[:]:
            if Y[i]/max_p > percent_trim:
                plt.plot(X[i],Y[i],marker="o",color="red")
            else:
                plt.plot(X[i],Y[i],marker="o",color="black")
    plt.show()
