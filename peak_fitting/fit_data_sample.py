"""
Algorithm to fit curves to a data sample and return parameters.

return curve parameters (parameters of the voigt profile)

"""


from scipy.optimize import curve_fit, leastsq
from scipy.signal import find_peaks
from scipy.special import wofz
import numpy as np
import matplotlib.pyplot as plt


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


def get_peak_indices(X,Y):
    slope = np.gradient(Y)
    #plt.plot(X,Y)
    #plt.plot(X,slope)
    #plt.show()
    #slope_minima,_ = find_peaks(np.amax(slope) - slope)

    #peaks = slope_minima[slope[slope_minima] > 1000]
    peaks = []
    param_list = fit_curves_to_data(X,Y)
    for curve in param_list:
        peaks.append([np.argmin(np.abs(X-curve[1])),*curve])
    return np.array(peaks)

def fit_curves_to_data(X,Y):
    #base_line_params, _ = curve_fit(line,X,Y)
    #base_line = lambda x : line(x,*base_line_params)
    #plot_curve(line,base_line_params,X)
    G = np.gradient(Y)
    local_minima,_ = find_peaks(np.amax(Y) - Y)
    #local_maxima,_ = find_peaks(Y)

    #local_minima = [*local_minima,*slope_minima]

    #change_points = [i for i in local_minima]# if Y[i] < base_line(X[i])]
    change_points = list(local_minima) + [len(X)-1]
    #for i,c in enumerate(change_points[:-1]):
    #    if change_points[i+1] - change_points[i] > 30:
    #        change_points.insert(i+1,int((change_points[i+1] + change_points[i])/2))

    #major_peaks = [i for i in local_maxima if Y[i] > base_line(X[i])]
    #[plt.axvline(X[i],color = "red",linestyle="--") for i in change_points]

    def smooth_on_range(Y,index_range,k):
        smooth = Y.copy()
        for i in index_range:
            a = max(i-int(k/2),0)
            b = min(i+int(k/2),len(Y)-1)
            smooth[i] = (sum(Y[a:b+1])/(b-a))
        Y[index_range] = smooth[index_range]

    def subtract_data_voigt(params):
        curve_flat = lambda x : voigt(x,*params[0:4])
        #curve = lambda x : voigt_shift(x,*params)
        for i in range(len(X)):
            Y[i] = Y[i] - curve_flat(X[i])

        smooth_range = np.where(curve_flat(X) > 5)[0]
        #print(smooth_range)
        #val = np.sum(Y[smooth_range]) / len(smooth_range)
        #Y[smooth_range] = val
        smooth_on_range(Y,smooth_range[-2:2],10)
        smooth_on_range(Y,smooth_range,10)


    """
    curve_params = []
    for i in major_peaks:
        ps = (X[-1] - X[0]) * .001
        bounds = ([0,X[i]-ps,0,0],[np.amax(Y),X[i]+ps,.01,.01])
        params,_ = curve_fit(voigt,X,Y,bounds=bounds)
        #print(params)
        curve_params.append(params)
        plt.plot(X,Y)
        plt.plot(X,[voigt(x,*params) for x in X],color="orange")
        #plt.show()
        #subtract_data_voigt(params)
        #plt.plot(X,Y,color="black")
        plt.show()
    """

    curve_params = []
    for i,c in enumerate(change_points[:-1]):
        block_X = X[change_points[i]:change_points[i+1]]
        block_Y = Y[change_points[i]:change_points[i+1]]


        try:
            params, split_loc = fit_curves_to_block(block_X,block_Y)

            #bounds = ([0,block_X[0],0,0],[np.amax(block_Y),block_X[-1],10,10])
            #params,_ = curve_fit(voigt,block_X,block_Y,bounds=bounds)


            #bounds = ([0,block_X[0],0,0,-10,0],[np.amax(block_Y),block_X[-1],10,10,10,1000])
            #params,_ = curve_fit(voigt_shift,block_X,block_Y,p0 = [p for p in params] + [0,0],bounds=bounds)
            curve_params.append(params)
            #curve_params = curve_params + fit_curves_to_block_optimize(block_X,block_Y)
        except:
            print("Failed to fit peak")
    return curve_params

def fit_curves_to_block(block_X,block_Y):
    param_list = []
    cen = block_X[np.argmax(block_Y)]
    B = (block_X[-1] - block_X[0]) * .01

    bounds = ([0,cen-B,0,0],[np.amax(block_Y),cen+B,.1,.1])
    params,_ = curve_fit(voigt,block_X,block_Y,bounds=bounds)

    bounds = ([0,cen-B,0,0,-10,0],[np.amax(block_Y),cen+B,1,1,10,1000])
    params,_ = curve_fit(voigt_shift,block_X,block_Y,p0 = [p for p in params] + [0,0],bounds=bounds,maxfev = 2000)

    function = lambda x : voigt_shift(x,*params)
    resid = block_Y - function(block_X)
    variance = np.var(resid)
    if variance > 1 and len(block_Y) > 15:
        return (params,np.argmax(resid))
    return (params,-1)


def fit_curves_to_block_optimize(X,Y):
    N = 1
    def leastsq_voigt(params):
        fit = np.zeros(Y.shape)
        for i in range(N):
            amp = params[i*4 + 0]
            cen = params[i*4 + 1]
            alpha = params[i*4 + 2]
            gamma = params[i*4 + 3]
            function = lambda x : voigt(x,amp,cen,alpha,gamma)
            fit[:] = fit[:] + function(X[:])
        return Y-fit

    def leastsq_voigt_shift(params):
        fit = np.zeros(Y.shape)
        for i in range(N):
            amp = params[i*6 + 0]
            cen = params[i*6 + 1]
            alpha = params[i*6 + 2]
            gamma = params[i*6 + 3]
            slope = params[i*6 + 4]
            shift = params[i*6 + 5]
            function = lambda x : voigt_shift(x,amp,cen,alpha,gamma,slope,shift)
            fit[:] = fit[:] + function(X[:])
        return Y-fit

    params0 = []
    error = 100
    while error > 10:
        params0 = [np.mean(Y),np.mean(X),.0001,.0001] * N
        result,_ = leastsq(leastsq_voigt, params0)
        params0 = list(result) + [0,0]
        result,_ = leastsq(leastsq_voigt_shift, params0)
        error = np.mean(leastsq_voigt_shift(list(result)))
        N+=1
        if len(Y) < 6*N:
            break
    print("Fit " + str(N-1) + " curves")
    param_list = [params0[i:i+6] for i in range(0,len(params0),6)]
    return param_list
