"""
Manual peak selection script.

Used with peak_error_plot

"""


from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import dict_to_csv

from scipy.signal import find_peaks



"""
Load Data and Peak Data
"""
dataGrid = DataGrid_TiNiSn_500C()

data_dir = "/home/sasha/Desktop/TiNiSn_500C_PeakData_0.5/"
#data_dir = "/home/sasha/Desktop/peakTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
peakGrid = DataGrid(data_dir,regex)


save_path = "/home/sasha/Desktop/python/peak_error/peak_errors.csv"


def selectPeak(event):
    """
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    """
    global cur
    cur = nearest_index(event.xdata)

def key_press(event):
    global mistakes, cur, grid_location, next
    if event.key == 'p':
        mistakes[grid_location] = mistakes[grid_location] + [cur]
        cur = 0
    if event.key == 'right':
        cur += 1
    if event.key == 'left':
        cur -= 1
    if event.key == 'z':
        mistakes[grid_location] = mistakes[grid_location][0:-1]
    if event.key == 'n':
        next = True


fig = plt.figure(figsize =(17,9))
cid = fig.canvas.mpl_connect('button_press_event', selectPeak)
fig.canvas.mpl_connect('key_press_event', key_press)


mistakes = {}
grid_location = 1
cur = 0
next = False
def nearest_index(x):
    return (np.abs(X - x)).argmin()

for grid_location in range(1,dataGrid.size+1):
    mistakes[grid_location] = []
    X = dataGrid.data_at_loc(grid_location)[:,0]
    Y = dataGrid.data_at_loc(grid_location)[:,1]
    next = False
    while not next:
        plt.cla()
        plt.title(str(grid_location))
        plt.plot(X,Y,color='blue')
        for peak_x in peakGrid.data_at_loc(grid_location)[:,1]:
            i = (np.abs(X - peak_x)).argmin()
            plt.plot([X[i]],[Y[i]],"x",color='black')
        for loc in mistakes[grid_location]:
            plt.plot([X[loc]],[Y[loc]],'x',color='red')
        plt.plot([X[cur]],[Y[cur]],"x",color='red')
        plt.draw()
        plt.pause(.1)
    #print(X[mistakes[grid_location]])
dict_to_csv(mistakes,save_path)
