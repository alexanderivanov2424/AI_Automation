
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

data_dir = "/home/sasha/Desktop/peakTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkg_curveParams.csv"""
peakGrid = DataGrid(data_dir,regex)


save_path = "/home/sasha/Desktop/python/peak_error/"

mistakes = csv_to_dict(save_path,"peak_errors")

w = 5
h = 5
fig, ax = plt.subplots(w, h)

skip = (w*h)*1
i = 0
for key in mistakes.keys():
    loc = int(key)
    m_list = eval(mistakes[key])
    X = dataGrid.data_at_loc(loc)[:,0]
    Y = dataGrid.data_at_loc(loc)[:,1]
    peaks = peakGrid.data_at_loc(loc)[:,1]
    dips,_ = find_peaks(max(Y) - Y)
    for m in m_list:
        if skip > 0:
            skip -= 1
            continue
        # Nearest Peak Found
        peak_index = np.argmin(np.abs(peaks-X[m]))
        P = np.argmin(np.abs(X - peaks[peak_index]))

        #Find Local Minima
        if len(np.where(dips-m > 0)[0]) == 0:
            D = len(peaks)-1
        else:
            D = np.where(dips-m > 0)[0][0]

        #Take local minimu between the mistake and peak if possible
        D_1 = dips[max(D-1,0)]
        D_2 = dips[min(D,len(dips)-1)]
        if min(P,m) < D_1 and D_1 < max(P,m):
            D = D_1
        elif min(P,m) < D_2 and D_2 < max(P,m):
            D = D_2
        else:
            D = D_1

        axis = ax[i%w,int(i/w)]
        axis.plot([X[D]],[Y[D]],'o',color="green")
        axis.plot([X[P]],[Y[P]],'o',color="black")
        axis.plot([X[m]],[Y[m]],'o',color="red")
        #Title is the percent height of the missed peak compared to
        #the nearest found peak and local min
        axis.title.set_text("% " + str(100 * (Y[m] - Y[D])/(Y[P] - Y[D])))
        start = min(D_1,P)-10
        end = max(D_2,P)+10
        axis.plot(X[start:end],Y[start:end])
        #Only plot the first w * h mistakes
        i+=1
        if i >= w * h:
            break
    if i >= w * h:
        break
k = .05
plt.subplots_adjust(left=k, bottom=k, right=1-k, top=1-k, wspace=.3, hspace=.35)
plt.show()



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
    mistakes[grid_location] = []
    X = dataGrid.data_at_loc(grid_location)[:,0]
    Y = dataGrid.data_at_loc(grid_location)[:,1]
    Slope = [(Y[i]-Y[i+1])/(X[i] - X[i+1])/100 for i in range(len(X)-1)]


    for peak_x in peakGrid.data_at_loc(grid_location)[:,1]:
        i = (np.abs(X - peak_x)).argmin()
        plt.plot([X[i]],[Y[i]],"x",color='black')
    #loaded as strings so need to be converted
    for loc in eval(mistakes[str(grid_location)]):
        plt.plot([X[loc]],[Y[loc]],'o',color='red')
    plt.plot(X,Y,color='blue')
    #plt.plot(X[:-1],Slope,color='green')
    #plt.plot(X,[0 for i in X],color='black')
    plt.title(grid_location)
    plt.show()
