'''
Script for plotting spectra at several data values to
visually compare differences.







'''


from data_loading.data_grid import DataGrid


from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math






regex_500 = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSub_1D.csv"""
regex_600 = """TiNiSn_600C_Y20190219_14x14_t65_(?P<num>.*?)_bkgdSub_1D.csv"""


data_path ="/path/to/data/here/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
data_path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"

dataGrid = DataGrid(data_path,regex_500)

# grid locations to plot
locations = [32,33,34,55]

#how much to shift each grid location vertically
#(makes it easier to see peaks)
shifts = []


def similarity_vector(A,B):
    return np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B)

#cosine similarity function using two grid positions
def similarity(d1,d2):
    a = dataGrid.data[d1][:,1]
    b = dataGrid.data[d2][:,1]
    return similarity_vector(a,b)


for i,k in enumerate(locations):
    y = dataGrid.data[k][:,1]
    if len(shifts) == len(locations):
        y = y + shifts[i]
    x = dataGrid.data[k][:,0]
    plt.plot(x,y,label=str(k))
plt.legend()


for i in range(len(locations)):
    for j in range(i+1,len(locations)):
        print("Similarity " + str(locations[i]) + "," + str(locations[j]) + ": " + str(similarity(locations[i],locations[j])))

plt.show()
