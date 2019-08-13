from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
from peak_fitting.fit_data_sample import fit_curves_to_data, get_peak_indices


from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz

import matplotlib.pyplot as plt
import numpy as np


"""
Load Data
"""
dataGrid = DataGrid_TiNiSn_500C()


data_dir = "/home/sasha/Desktop/peakTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkg_curveParams.csv"""
peakGrid = DataGrid(data_dir,regex)


def line(x,slope,shift):
    return slope*x + shift

def gaussian(x, amp, cen, sig):
    return amp * np.exp(-(x-cen)**2 / (2*sig**2))/(sig * np.sqrt(2*np.pi))

def gaussian_shift(x, amp, cen, sig, shift,slope):
    return amp * np.exp(-(x-cen)**2 / (2*sig**2))/(sig * np.sqrt(2*np.pi)) + slope * x + shift

def voigt(x, amp,cen,alpha,gamma):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)

def voigt_shift(x,amp,cen,alpha,gamma,shift,slope):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi) + slope * x + shift

def plot_curve(curve,params,range):
    plt.plot(range,[curve(x,*params) for x in range],color="green")


for loc in range(1,dataGrid.size):
    X = dataGrid.data_at_loc(loc)[:,0]
    Y = dataGrid.data_at_loc(loc)[:,1]

    curve_params = fit_curves_to_data(X,Y)
    peaks = get_peak_indices(X,Y)[:,0]
    BBA_peaks = peakGrid.data_at_loc(loc)[:,2]

    plt.plot(X,Y,color="blue")

    """
    for peak in BBA_peaks:
        p = np.argmin(np.abs(X-peak))
        plt.plot(X[p],Y[p],'o',color="black")
    for p in peaks:
        plt.plot(X[p],Y[p],'x',color="red")
    """
    """
    for params in curve_params:
        cen = params[1]
        sig = .1
        range = np.linspace(cen-sig,cen+sig,60)
        plt.plot(range,[voigt_shift(x,*params) for x in range])
    """

    plt.show()


x = np.linspace(-10,10,101)
y = gaussian(x, 2.33, 0.21, 1.51) + random.normal(0, 0.2, len(x))

init_vals = [1, 0, 1]  # for [amp, cen, wid]
best_vals, covar = curve_fit(voigt, x, y)#, p0=init_vals)
print(best_vals)

plt.plot(x,y)
plt.plot(x,[voigt(i,*best_vals) for i in x])
plt.show()



params0 = [a0, n0, m0]
args = (x, y, (E0, E1, E2), (T0, T1, T2), (n1, n1+n2))
result = scipy.optimize.leastsq(leastsq_function, params0, args=args)
