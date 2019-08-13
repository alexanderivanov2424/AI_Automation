"""
Itterative curve fitting on blocks


ISSUE:
Bad noise detection


"""

from scipy.optimize import curve_fit, leastsq
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.special import wofz



def voigt(x, amp,cen,alpha,gamma):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)

def multi_voigt(x,*params):
    sum = 0.0
    for i in range(0,len(params)-1,4):
        sum += voigt(x,params[i],params[i+1],params[i+2],params[i+3])
    return sum

def is_noise(data):
    #R = 1 / (len(data) * np.sum(np.square(data)) - np.square(np.sum(data)))
    #print(R)
    sd = np.std(data)
    mean = np.mean(data)
    dev = np.where(np.abs(data-mean)/sd > 2)[0]
    #more than k points are statistically significant
    if len(dev) > 1:#k
        return False
    return True

def fit_curves_to_data(X,Y):
    local_minima,_ = find_peaks(np.amax(Y) - Y)
    change_points = list(local_minima) + [len(X)-1]
    median = np.median(Y)
    i = 0
    while i < len(change_points)-2:
        i+=1
        if change_points[i+1] - change_points[i] < 20:
            change_points.pop(i+1)

    for c in change_points:
        #print(Y[c])
        if Y[c] > median:
            change_points.remove(c)

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
    bounds = ([0,cen-B,0,0],[np.amax(Y),cen+B,.01,.01])
    #plt.plot(X,Y)
    #curve = lambda x : voigt(x,np.amax(Y)/20,cen,.01,.01)
    #plt.plot(X,[curve(x) for x in X])
    #plt.show()
    params,_ = curve_fit(voigt,X,Y,bounds=bounds,maxfev=2000)
    #plt.plot(X,Y)
    #curve = lambda x : voigt(x,*params)
    #plt.plot(X,[curve(x) for x in X])
    #plt.show()
    #print(params)
    return params

def fit_curves_to_block(X,Y):
    all_params = np.array([])
    resid = Y.copy()
    num_curves = 0

    plt.plot(X,Y)
    plt.show()

    while not is_noise(resid):
        print("loop")
        num_curves += 1
        params = fit_guess_curve_to_block(X,resid)
        all_params = np.append(all_params,params)
        p0 = all_params

        #plt.plot(X,resid)
        #curve = lambda x : voigt(x,*params)
        #plt.plot(X,[curve(x) for x in X])
        #plt.show()

        try:
            all_params,_ = curve_fit(multi_voigt,X,Y,p0=p0,maxfev=2000)
            curve = lambda x : multi_voigt(x,*all_params)
            resid = np.array([Y[i] - curve(x) for i,x in enumerate(X)])
        except:
            break
        #plt.plot(X,Y)
        #plt.plot(X,[curve(x) for x in X])
        #plt.plot(X,resid)
        #plt.show()

    return [[all_params[i],all_params[i+1],all_params[i+2],all_params[i+3]] for i in range(0,len(all_params-1),4)]

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
