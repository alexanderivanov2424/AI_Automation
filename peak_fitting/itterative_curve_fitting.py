

from scipy.optimize import curve_fit, leastsq
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.special import wofz


"""
X - x positions of data points
Y - y positions of data points
profile - the profile shape function used for fitting / optimization

returns dictionary of parameters
params - curve params
peaks - peak locations
I - peak intensities
blocks - list of blocks
"""

def voigt(x, amp,cen,alpha,gamma):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)

def multi_voigt(x,*params):
    sum = 0.0
    for i in range(0,len(params)-1,4):
        sum += voigt(x,params[i],params[i+1],params[i+2],params[i+3])
    return sum

def is_noise(data):
    sd = np.std(data)
    mean = np.mean(data)
    dev = np.where(np.abs(data-mean)/sd > 2)[0]
    #more than 2 points are statistically significant
    if len(dev) > 2:
        return False
    return True

def fit_curves_to_data(X,Y):
    local_minima,_ = find_peaks(np.amax(Y) - Y)
    change_points = list(local_minima) + [len(X)-1]
    median = np.median(Y)
    print(median)
    i = 0
    while i < len(change_points)-2:
        i+=1
        if change_points[i+1] - change_points[i] < 30 or Y[change_points[i+1]] > median:
            change_points.pop(i+1)
    curve_params = []
    for i,c in enumerate(change_points[:-1]):
        block_X = X[change_points[i]:change_points[i+1]]
        block_Y = Y[change_points[i]:change_points[i+1]]
        #try:

        curve_params = curve_params + fit_curves_to_block(block_X,block_Y)
        #except:
        #    print("Failed to fit peak")
    return curve_params
    #return fit_curves_to_block(X,Y,0)

def fit_guess_curve_to_block(X,Y):
    cen = X[np.argmax(Y)] #peak center
    B = (X[-1] - X[0]) * .01 # 1% of block width
    bounds = ([0,cen-B,.001,.001],[np.amax(Y),cen+B,.1,.1])
    params,_ = curve_fit(voigt,X,Y,bounds=bounds)
    return params

def fit_curves_to_block(X,Y):
    all_params = np.array([])
    resid = Y
    num_curves = 0
    while not is_noise(resid) and num_curves < 3:
        num_curves += 1
        params = fit_guess_curve_to_block(X,resid)
        #all_params = np.append(all_params,params)
        print(params)
        p0 = all_params
        #opt_params,_ = curve_fit(multi_voigt,X,resid,p0=p0)
        curve = lambda x : multi_voigt(x,all_params)
        resid = np.array([Y[i] - curve(x) for i,x in enumerate(X)])

        plt.plot(X,Y)
        plt.plot(X,resid)
        plt.plot(X,[curve(x) for x in X])
        plt.show()
        if is_noise(resid):
            return [params] #+ fit_curves_to_block(X,[Y[i] - curve(x) for i,x in enumerate(X)])
    return []
    #sub_param_list = fit_curves_to_block(X,resid)
    #join all params
    for sub_params in sub_param_list:
        params = np.append(params,sub_params)

    p0 = list(params)
    opt_params,_ = curve_fit(voigt,X,Y,bounds=bounds)
    return []



from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C

dataGrid = DataGrid_TiNiSn_500C()
for loc in range(1,dataGrid.size):
    X = dataGrid.data_at_loc(loc)[:,0]
    Y = dataGrid.data_at_loc(loc)[:,1]

    param_list = fit_curves_to_data(X,Y)
    plt.plot(X,Y)
    for params in param_list:
        curve = lambda x : voigt(x,*params)
        plt.plot(X,[curve(x) for x in X])
    plt.show()
