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
import csv

#unused import
#from statsmodels.stats.diagnostic import acorr_ljungbox

"""
Voigt Profile used for fitting
"""
def voigt(x, amp,cen,alpha,gamma,c):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)+c

"""
Combination of multiple Voigt profiles
"""
def multi_voigt(x,*params):
    sum = 0.0
    for i in range(0,len(params)-1,5):
        sum += voigt(x,params[i],params[i+1],params[i+2],params[i+3],params[i+4])
    return sum

"""
Check for if a set of data is noise
Tests if there are statistically significant points
"""
def is_noise(data,noise_threshold):
    #residual based on amplitude noise
    return np.max(np.abs(data)) < noise_threshold
    #noise based on Standard Deviation
    """
    sd = np.std(data)
    mean = np.mean(data)
    dev = np.where(np.abs(data-mean)/sd > 2)[0]
    #more than k points are statistically significant
    if len(dev) > 2:#k
        return False
    return True
    """
"""
Saves dictionary output from fit_curves_to_data to a csv
Note: only saves peak parameters
"""
def save_data_to_csv(full_file_name,dict):
    curve_params = dict['curve_params']
    csv_data = [['I','Q','alpha','gamma']] + curve_params
    with open(full_file_name,"w+") as csv_file:
        csvWriter = csv.writer(csv_file,delimiter=',')
        csvWriter.writerows(csv_data)

"""
Calculate the background noise in an interval (inclusive)
used for determining noise in residual

"""
def calculate_background_noise(X,Y,background_start,background_end):
    i = np.argmin(np.abs(X-background_start))
    j = np.argmin(np.abs(X-background_end))
    X_background = X[i:j+1]
    Y_background = Y[i:j+1]
    cosine_curve = lambda x,a,b,c,d : a*np.cos(b*x + c) + d
    popt,_ = curve_fit(cosine_curve, X_background, Y_background)
    resid = Y_background - cosine_curve(X_background,*popt)
    sd = np.std(resid)
    return sd*3
"""
Fit curves to diffraction pattern.

return curve parameters
"""
def fit_curves_to_data(X,Y,background_start,background_end):
    noise_threshold = calculate_background_noise(X,Y,background_start,background_end)

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
    resids = []
    block_curves = []
    block_fits = []
    for i,c in enumerate(change_points[:-1]):
        block_X = X[change_points[i]:change_points[i+1]+1]
        block_Y = Y[change_points[i]:change_points[i+1]+1]
        block_params = fit_curves_to_block(block_X,block_Y,noise_threshold)
        curve_params = curve_params + block_params

        for curve_p in block_params:
            curve_t = lambda x : voigt(x,*curve_p)
            block_curves.append((block_X,[curve_t(x) for x in block_X]))


        curve = lambda x : multi_voigt(x,*[p for params in block_params for p in params])
        resid = np.array([block_Y[i] - curve(x) for i,x in enumerate(block_X)])
        resids.append((block_X,resid))
        block_fits.append((block_X,[curve(x) for x in block_X]))



    #optimize full spectra
    """
    full_params = np.array([])
    for P in curve_params:
        full_params = np.append(full_params,P)
    p0 = full_params
    k = len(full_params)//5
    eps = .000000001
    bounds0 = []
    bounds1 = []

    #Bounds for optimization
    #seems like with any bounds it either takes forever or does nothing.
    for i in range(k):
        bounds0 = bounds0 + [p0[i*5],p0[i*5+1],p0[i*5+2],p0[i*5+3],0]
        bounds1 = bounds1 + [p0[i*5]+eps,p0[i*5+1]+eps,p0[i*5+2]+eps,p0[i*5+3]+eps,100]
        #bounds = ([0,np.min(X),0,0,0] * k,[np.max(Y),np.max(X),5,5,np.max(Y)]*k)
    bounds = (bounds0,bounds1)
    full_params,_ = curve_fit(multi_voigt,X,Y,p0=p0,bounds=bounds,maxfev=2000,xtol=.00005)
    #recalculate residual
    curve = lambda x : multi_voigt(x,*full_params)
    resid = np.array([Y[i] - curve(x) for i,x in enumerate(X)])
    plt.plot(X,Y)
    plt.plot(X,[curve(x) for x in X])
    plt.plot(X,resid)
    plt.show()
    """

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
    dict['residuals'] = resids
    dict['block curves'] = block_curves
    dict['block fits'] = block_fits
    return dict

"""
Fit a single Voigt profile to data
return curve parameters.
"""
def fit_guess_curve_to_block(X,Y):
    cen = X[np.argmax(Y)] #peak center
    B = (X[-1] - X[0]) * 1 # 1% of block width
    p0 = [np.max(Y)/100,cen,.01,.01,0]
    bounds = ([0,cen-B,0,0,0],[np.max(Y),cen+B,5,5,np.amax(Y)])
    params,_ = curve_fit(voigt,X,Y,p0=p0,bounds=bounds,maxfev=2000)
    return params

"""
Itteratively fit curves to a block in the diffraction pattern
"""
def fit_curves_to_block(X,Y,noise_threshold):
    all_params = np.array([])
    resid = Y.copy()
    num_curves = 0
    #fit up to 4 curves
    while not is_noise(resid,noise_threshold):#num_curves <= 4:
        num_curves += 1
        #fit initial guess curve to residual
        params = fit_guess_curve_to_block(X,resid)
        all_params = np.append(all_params,params)

        # try to optimize all curves together for better fit
        p0 = all_params
        k = len(all_params)//5
        bounds = ([0,np.min(X),0,0,0] * k,[np.max(Y),np.max(X),5,5,np.max(Y)]*k)
        try:
            all_params,_ = curve_fit(multi_voigt,X,Y,p0=p0,bounds=bounds,maxfev=2000)
            #recalculate residual
            curve = lambda x : multi_voigt(x,*all_params)
            resid = np.array([Y[i] - curve(x) for i,x in enumerate(X)])

        except:
            #if fit fails return peak params as is
            break
        #if residual after optimization is noise finish
        if is_noise(resid,noise_threshold):
            break

    #return parameters as a list of 4-lists where each 4-list is a curve
    return [[all_params[i],all_params[i+1],all_params[i+2],all_params[i+3],all_params[i+4]] for i in range(0,len(all_params-1),5)]
