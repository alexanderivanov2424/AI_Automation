"""
Iterative curve fitting on blocks

Blocks:
    local minima below median.
    merge blocks smaller than min-size

Iterative Curve Fitting:
    Iteratively fit guess curves to the residual.
    Optimize all curves together to produce new residual
    Stop if residual below to noise threshold

ISSUES:
    Bad noise detection

"""

from scipy.optimize import curve_fit, leastsq
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.special import wofz
import csv


"""
Voigt Profile used for fitting

NOTE: When adjusting the number of parameters code needs to be modified.
change the NUM_PARAMS variable respectively and adjust code at the ## PARAM tags

specifically the fit_curves_to_block, fit_guess_curve_to_block, and save_data_to_csv
"""
NUM_PARAMS = 5

def voigt(x, amp,cen,alpha,gamma,c): ##PARAM
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)+c

"""
Combination of multiple Voigt profiles
"""
def multi_voigt(x,*params):
    sum = 0.0
    for i in range(0,len(params)-1,NUM_PARAMS):
        sum += voigt(x,*params[i:i+NUM_PARAMS])
    return sum

"""
Check for if a set of data is noise
Tests if there are statistically significant points
"""
def is_noise(data,noise_threshold):
    #residual based on amplitude noise
    return np.max(np.abs(data)) < noise_threshold

"""
Saves dictionary output from fit_curves_to_data to a csv
Note: only saves peak parameters
Note: full_file_name includes the path to the file, its name, and the .csv at the end
"""
def save_data_to_csv(full_file_name,dict): ## PARAM
    curve_params = dict['curve_params']
    for params in curve_params:
        #calculate peak Intensity
        curve = lambda x : voigt(x,*params)
        params[0] = curve(params[1]) - params[4]
        #good approximation of FWHM
        FG = 2*params[2] * np.sqrt(2*np.log(2))
        FL= 2*params[3]
        FWHM = .5346*FL + np.sqrt(.2166*FL*FL + FG * FG)
        params.append(FWHM)
    csv_data = [['I','Q','alpha','gamma','c','width']] + curve_params
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
    popt,_ = curve_fit(cosine_curve, X_background, Y_background,maxfev=2000)
    resid = Y_background - cosine_curve(X_background,*popt)
    sd = np.std(resid)
    return sd*3


"""
Fit curves to diffraction pattern.

return curve parameters
"""
def fit_curves_to_data(X,Y,background_start=None,background_end=None,noise=None,max_curves=30,min_block_size=20):
    if noise == None:
        noise_threshold = calculate_background_noise(X,Y,background_start,background_end)
    else:
        noise_threshold = noise
    local_minima,_ = find_peaks(np.amax(Y) - Y)
    change_points = list(local_minima) + [len(X)-1]
    median = np.median(Y)

    #make sure change points are further apart than the min_block_size
    i = 0
    while i < len(change_points)-2:
        i+=1
        if change_points[i+1] - change_points[i] < min_block_size:
            change_points.pop(i+1)
            i-=1

    #make sure no change points are below the median
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
        block_params = fit_curves_to_block(block_X,block_Y,noise_threshold,max_curves)

        if isinstance(block_params,tuple):
            #block with no peaks (fit a line)
            a = block_params[0]
            b = block_params[1]
            curve_params = curve_params + []

            block_curves.append((block_X,[a*x +b for x in block_X]))

            resid = np.array([block_Y[i] - a*x - b for i,x in enumerate(block_X)])
            resids.append((block_X,resid))
            block_fits.append((block_X,[a*x +b for x in block_X]))
        else:
            #block with peaks (multi voigt)
            curve_params = curve_params + block_params

            for curve_p in block_params:
                curve_t = lambda x : voigt(x,*curve_p)
                block_curves.append((block_X,[curve_t(x) for x in block_X]))


            curve = lambda x : multi_voigt(x,*[p for params in block_params for p in params])
            resid = np.array([block_Y[i] - curve(x) for i,x in enumerate(block_X)])
            resids.append((block_X,resid))
            block_fits.append((block_X,[curve(x) for x in block_X]))

    #function for the full curve fit
    def fit(x):
        loc = np.argmin(np.abs(X - x))
        block = 0
        for i,c in enumerate(change_points):
            if c > loc:
                block = i - 1
                break
        block_index =np.argmin(np.abs(block_fits[block][0] - x))
        return block_fits[block][1][block_index]

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
    dict['fit'] = fit
    return dict

"""
Fit a single Voigt profile to data
return curve parameters.
"""
def fit_guess_curve_to_block(X,Y):
    cen = X[np.argmax(Y)] #peak center
    B = (X[-1] - X[0]) * 1 # 1% of block width
    try:
        p0 = [np.max(Y)/100,cen,.01,.01,0]## PARAM
        bounds = ([0,cen-B,0,0,0],[np.max(Y),cen+B,2,2,np.amax(Y)])## PARAM
        params,_ = curve_fit(voigt,X,Y,p0=p0,bounds=bounds,maxfev=2000)
    except:
        try:
            p0 = [np.max(Y)/100,cen,.01,.01,0]## PARAM
            bounds = ([0,cen-B,0,0,0],[np.max(Y),cen+B,5,5,np.amax(Y)])## PARAM
            params,_ = curve_fit(voigt,X,Y,p0=p0,bounds=bounds,maxfev=2000)
        except:
            p0 = [np.max(Y)/100,cen,2,2,np.amax(Y)/100]## PARAM
            bounds = ([0,cen-B,2,2,0],[np.max(Y),cen+B,5,5,np.amax(Y)])## PARAM
            params,_ = curve_fit(voigt,X,Y,p0=p0,bounds=bounds,maxfev=2000)
    return params

"""
Fit a line to given data and check if residual is withing noise
Used to identify if a given block is purely noise and doesn't need a peak
"""
def is_background_line(X,Y,noise_threshold):
    line = lambda x,a,b : a*x + b
    params,_ = curve_fit(line,X,Y)
    line_fit = lambda x : line(x,*params)
    return np.max(np.abs(Y - line_fit(X))) < noise_threshold

def fit_line_params(X,Y):
    line = lambda x,a,b : a*x + b
    params,_ = curve_fit(line,X,Y)
    return (params[0],params[1])


"""
Iteratively fit curves to a block in the diffraction pattern
"""
def fit_curves_to_block(X,Y,noise_threshold,max_curves):
    all_params = np.array([])
    resid = Y.copy()

    #if the data can be fit by a line don't fit any curves
    #this means the fit for the block will be constant 0
    if is_background_line(X,Y,noise_threshold):
        return fit_line_params(X,Y) #return tuple with line params

    num_curves = 0
    while not is_noise(resid,noise_threshold):
        num_curves += 1
        if num_curves > max_curves:
            #exit if the maximum number of curves has been fit
            break
        #fit initial guess curve to residual
        params = fit_guess_curve_to_block(X,resid)
        all_params = np.append(all_params,params)

        # try to optimize all curves together for better
        p0 = all_params
        k = len(all_params)//NUM_PARAMS
        bounds = ([0,np.min(X),0,0,0] * k,[np.max(Y),np.max(X),5,5,np.max(Y)]*k)## PARAM
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

    #return parameters as a list of 5-element-lists where each 5-element-list is
    #the params for a curve
    return [list(all_params[i:i+NUM_PARAMS]) for i in range(0,len(all_params-1),NUM_PARAMS)]
