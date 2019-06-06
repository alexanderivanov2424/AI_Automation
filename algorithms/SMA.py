#Similarity from measurement Avereges

from data_loading.data_grid import DataGrid
from utils.utils import plotDataGrid, interpolateData, similarity, interpolateDataAvg
from utils.utils import getSimilarityMatrix, clipSimilarityMatrix
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import imageio
import numpy as np
import time

#seed algorithm
seed = 0
np.random.seed(seed)

#set up DataGrid object
path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
dataGrid = DataGrid(path)

#set up array to store plots
video = []
file_name = "SMA-" + str(seed)


#set up the visuals
fig, ax = plt.subplots(nrows=2, ncols=3)
canvas = FigureCanvasAgg(fig)
true_data = clipSimilarityMatrix(getSimilarityMatrix(dataGrid.get_data_array(),dataGrid))
ax[1,2].imshow(true_data)
text = ax[1,1].text(0, 0, "", fontsize=8)

#initialize variables
exp_data = np.zeros(true_data.shape)
old_x = 1
old_y = 1


#CONSTANTS
blur_const = 4
NUMBER_OF_SAMPLES = 50

#DATA STRUCTURES
M = np.empty(shape=(dataGrid.size,dataGrid.data_length))
S = set()

#cosine similarity function using two grid positions
def get_similarity(d1,d2):
    if np.linalg.norm(M[d1]) == 0.:
        print(d1)
    if np.linalg.norm(M[d2]) == 0.:
        print(d2)
    return similarity(M[d1],M[d2])

#Used to convert blurred similarity matrix to probability distribution
def convertTo1D(G):
    ret = np.empty(dataGrid.size)
    for i in range(dataGrid.size):
        x,y = dataGrid.coord(i+1)
        ret[i] = G[x-1][y-1]
    return ret

#Setting up Timer and time record
times = []
total_timer = 0.
timer = time.time()

def start_time():
    global timer
    timer = time.time()

def stop_time():
    global total_timer, timer
    total_timer += time.time() - timer
    timer = time.time()

def get_time():
    global total_timer
    t = total_timer
    total_timer = 0
    return t


#__________________________________________________
# START


# INITIAL SAMPLES
C_list = np.random.choice(range(1,dataGrid.size+1), 3)
for C in C_list:
    M[C-1] = dataGrid.data_at_loc(C)[:,1]
    S.add(C)

while len(S) < NUMBER_OF_SAMPLES:

    start_time()

    dissim = getSimilarityMatrix(interpolateDataAvg(M),dataGrid)
    blurred = gaussian_filter(dissim, sigma=blur_const)
    flat = convertTo1D(blurred)
    if  np.sum(flat) == 0:
        Distribution = np.full(shape=(dataGrid.size),fill_value = 1/dataGrid.size)
    else:
        Distribution = flat / np.sum(flat)

    stop_time()
    #Plotting
    ax[0,0].imshow(dissim)
    ax[0,1].imshow(blurred)
    ax[0,2].imshow(exp_data)

    measured_points = np.zeros(dataGrid.dims)
    for s in S:
        x,y = dataGrid.coord(s)
        measured_points[x-1,y-1] = 1
    ax[1,0].imshow(measured_points)
    start_time()

    #Note: cells numbering starts at 1
    data_range = range(1,dataGrid.size+1)

    cells = np.random.choice(data_range, 1, p=Distribution)
    while cells[0] in S:
        cells = np.random.choice(data_range, 1, p=Distribution)
    C = cells[0]


    M[C-1] = dataGrid.data_at_loc(C)[:,1] #"taking a measurement"
    S.add(C)

    stop_time()
    #Additional Plotting
    next_x,next_y = dataGrid.coord(C)
    sct_next = ax[0,1].scatter(next_y-1,next_x-1,s=15,c='red')
    sct_old = ax[0,1].scatter(old_y-1,old_x-1,s=15,c='purple')

    times = [get_time()] + times
    s = "Avg Sample Time: \n"
    s += str(float(sum(times)/len(times))) + "\n"
    s += "Mean Squared Error: \n"
    s += str(float(np.square(np.subtract(exp_data, true_data)).mean())) + "\n"
    s += "L2 Distance: \n"
    s += str(float(np.sum(np.square(np.subtract(exp_data, true_data))))) + "\n"
    s += "L1 Distance: \n"
    s+= str(float(np.sum(np.abs(np.subtract(exp_data, true_data))))) + "\n"
    text.set_text(s)


    full_data = interpolateData(M,3,dataGrid)
    exp_data = clipSimilarityMatrix(getSimilarityMatrix(full_data,dataGrid))

    plt.draw()
    canvas.draw()

    #saving frame to video
    frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    w,h = canvas.get_width_height()
    frame = np.reshape(frame,(h,w,3))
    video.append(frame)

    sct_next.remove()
    sct_old.remove()
    old_x = next_x
    old_y = next_y


# END
#__________________________________________________



#save video as file_name
imageio.mimwrite("/home/sasha/Desktop/python/videos/" + file_name + ".mp4", video, fps=2)


print("Finished Sampling")
print("_________________")


full_data = interpolateData(M,4,dataGrid)
exp_data = clipSimilarityMatrix(getSimilarityMatrix(full_data,dataGrid))

print("Mean Squared Error: ")
print(np.square(np.subtract(exp_data, true_data)).mean())
print()

print("L2 Distance: ")
print(np.sum(np.square(np.subtract(exp_data, true_data))))
print()

print("L1 Distance: ")
print(np.sum(np.abs(np.subtract(exp_data, true_data))))
print()
