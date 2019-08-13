

from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
from peak_fitting.fit_data_sample import fit_curves_to_data, get_peak_indices

import matplotlib.pyplot as plt
import matplotlib

from scipy.optimize import curve_fit
from scipy.special import wofz

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
draw_curves = True
draw_min_peaks = False


#load Min block peak data
peakMinGrid = DataGrid(data_dir,regex)
data_dir = "/home/sasha/Desktop/MinBlockCurveParams/"
for loc in range(1,dataGrid.size+1):
    file = data_dir + str(loc) + ".txt"
    #peaks = eval(open(file).read())

    try:
        peaks = eval(open(file).read())
    except:
        print("Generating " + str(loc))
        X = dataGrid.data_at_loc(loc)[:,0]
        Y = dataGrid.data_at_loc(loc)[:,1]

        #curve_params = fit_curves_to_data(X,Y)
        peaks = get_peak_indices(X,Y).tolist()#[:,0] ##Use Min blocks to find peaks
        open(file,'w+').write(str(peaks))
    peakMinGrid.data[loc] = peaks


background_start = 1.5
background_end = 1.8

def fourier(x, *a):
    ret = a[0] * np.cos(a[1] * x) + a[2]
    return ret

def fourier(x, *a):
    ret = a[0] * np.cos(10/x)
    for deg in range(1, len(a)):
        ret += a[deg] * np.cos(10*(deg+1) /x)
    return ret

def voigt_shift(x,amp,cen,alpha,gamma,shift,slope):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi) + slope * x + shift

def plot_curve(curve,params,range):
    #plt.plot(range,[np.sqrt(curve(x,*params)) for x in range],color="green")
    plt.plot(range,[curve(x,*params) for x in range],color="green")


for loc in dataGrid.data.keys():
    loc = 77
    X = dataGrid.data_at_loc(loc)[:,0]
    Y = dataGrid.data_at_loc(loc)[:,1]
    s = np.argmin(np.abs(X - background_start))
    e = np.argmin(np.abs(X - background_end))
    X_trim = dataGrid.data_at_loc(loc)[s:e,0]
    Y_trim = dataGrid.data_at_loc(loc)[s:e,1]

    #fit fourier curve
    popt, pcov = curve_fit(lambda x,a,b,c,d : fourier(x,a,b,c,d), X_trim, Y_trim,[30,5,0,20])


    #create resid
    resid = Y_trim - fourier(X_trim, *popt)
    #plt.plot(X_trim,Y_trim)
    plt.plot(X_trim, fourier(X_trim, *popt))

    #resid = [1,1,1,1]
    sd = statistics.stdev(resid) * 3

    #plt.fill_between(X,Y+sd,Y-sd,facecolor='red', alpha=0.5)
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
        for x in curveBBAGrid.data_at_loc(loc):
            i = np.argmin(np.abs(X - x[2]))
            if x[4] > sd:
                pass
                #plt.plot(X[i],Y[i],marker="o",color="red")
            else:
                pass
                #plt.plot(X[i],Y[i],marker="o",color="black")
        #skip = 0
        for x in curveBBAGrid.data_at_loc(loc):
            #skip += 1
            #if skip == 2 or skip == 4:
            #    continue
            if x[5] < .1 and x[6] < .1:
                print(x[6])
                plot_curve(voigt_shift,[x[4],x[2],x[5],x[6],0,0],np.linspace(x[2]-.2,x[2]+.2,60))

    if draw_min_peaks:
        max_p = np.max([x[1] for x in peakMinGrid.data_at_loc(loc)])
        for x in peakMinGrid.data_at_loc(loc):
            i = int(x[0])
            plot_curve(voigt_shift,x[1:],np.linspace(x[2]-.1,x[2]+.1,60))
            if x[1] > sd:
                plt.plot(X[i],Y[i],marker="o",color="red")
            else:
                plt.plot(X[i],Y[i],marker="o",color="black")
    plt.show()
