"""
Itterative curve fitting on blocks

Blocks:
    local minima below median.
    merge blocks smaller than min-size

Itterative Curve Fitting:
    max 4 curves
    stop if residual equivalent to noise
    joint curve optimization

ISSUES:
    Bad noise detection
    Misses peaks in high density


"""

from scipy.optimize import curve_fit, leastsq
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.special import wofz


from statsmodels.stats.diagnostic import acorr_ljungbox

"""
Voigt Profile used for fitting
"""
def voigt(x, amp,cen,alpha,gamma):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)

"""
Combination of multiple Voigt profiles
"""
def multi_voigt(x,*params):
    sum = 0.0
    for i in range(0,len(params)-1,4):
        sum += voigt(x,params[i],params[i+1],params[i+2],params[i+3])
    return sum

"""
Check for if a set of data is noise
Tests if there are statistically significant points
"""
def is_noise(data):
    sd = np.std(data)
    mean = np.mean(data)
    dev = np.where(np.abs(data-mean)/sd > 2)[0]
    #more than k points are statistically significant
    if len(dev) > 2:#k
        return False
    return True

"""
Fit curves to diffraction pattern.

return curve parameters
"""
def fit_curves_to_data(X,Y):
    local_minima,_ = find_peaks(np.amax(Y) - Y)
    change_points = list(local_minima) + [len(X)-1]
    median = np.median(Y)
    i = 0
    while i < len(change_points)-2:
        i+=1
        if change_points[i+1] - change_points[i] < 20:
            change_points.pop(i+1)
            i-=1

    i = 0
    while i < len(change_points)-2:
        i+=1
        if Y[change_points[i]] > median:
            change_points.remove(change_points[i])
            i-=1


    #fit curves to every block and produce combined curve param list
    curve_params = []
    for i,c in enumerate(change_points[:-1]):
        block_X = X[change_points[i]:change_points[i+1]]
        block_Y = Y[change_points[i]:change_points[i+1]]
        curve_params = curve_params + fit_curves_to_block(block_X,block_Y)

    curve_centers = []
    curve_I = []
    for curve_p in curve_params:
        curve = lambda x : voigt(x,*curve_p)
        curve_centers.append(curve_p[1])
        curve_I.append(curve(curve_p[1]))
    #create return dictionary
    dict = {}
    dict['curve_params'] = curve_params
    dict['change_points'] = change_points
    dict['Q'] = curve_centers
    dict['I'] = curve_I
    dict['profile'] = voigt
    return dict

"""
Fit a single Voigt profile to data
return curve parameters.
"""
def fit_guess_curve_to_block(X,Y):
    cen = X[np.argmax(Y)] #peak center
    B = (X[-1] - X[0]) * .01 # 1% of block width
    bounds = ([0,cen-B,0,0],[np.amax(Y),cen+B,.01,.01])
    params,_ = curve_fit(voigt,X,Y,bounds=bounds,maxfev=2000)
    return params

"""
Itteratively fit curves to a block in the diffraction pattern
"""
def fit_curves_to_block(X,Y):
    all_params = np.array([])
    resid = Y.copy()
    num_curves = 0
    #fit up to 4 curves
    while num_curves <= 3:
        num_curves += 1
        #fit initial guess curve to residual
        params = fit_guess_curve_to_block(X,resid)
        all_params = np.append(all_params,params)

        # try to optimize all curves together for better fit
        p0 = all_params
        k = len(all_params)//4
        bounds = ([0,np.min(X),0,0] * k,[np.max(Y),np.max(X),.03,.03]*k)
        try:
            all_params,_ = curve_fit(multi_voigt,X,Y,p0=p0,bounds=bounds,maxfev=2000)
            #recalculate residual
            curve = lambda x : multi_voigt(x,*all_params)
            resid = np.array([Y[i] - curve(x) for i,x in enumerate(X)])

        except:
            #if fit fails return peak params as is
            break
        #if residual after optimization is noise finish
        if is_noise(resid):
            break

    #return parameters as a list of 4-lists where each 4-list is a curve
    return [[all_params[i],all_params[i+1],all_params[i+2],all_params[i+3]] for i in range(0,len(all_params-1),4)]
