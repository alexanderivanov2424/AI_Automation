
from data_loading.data_grid import DataGrid
from utils.utils import plotDataGrid, interpolateData, similarity
from utils.utils import getSimilarityMatrix, clipSimilarityMatrix
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import imageio
import numpy as np
import time




#folder with data files
path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
dataGrid = DataGrid(path)

video = []

fig, ax = plt.subplots(nrows=2, ncols=3)
canvas = FigureCanvasAgg(fig)


true_data = clipSimilarityMatrix(getSimilarityMatrix(dataGrid.get_data_array(),dataGrid))
ax[1,2].imshow(true_data)
text = ax[1,1].text(0, 0, "", fontsize=8)
exp_data = np.zeros(true_data.shape)
old_x = 0
old_y = 0

k = 0.1

power = 20

blur_const = 4

M = np.empty(shape=(dataGrid.size,dataGrid.data_length))
G = np.full(shape=dataGrid.size,fill_value = k)
S = set()


#cosine similarity function using two grid positions
def get_similarity(d1,d2):
    if np.linalg.norm(M[d1]) == 0.:
        print(d1)
    if np.linalg.norm(M[d2]) == 0.:
        print(d2)
    return similarity(M[d1],M[d2])


def blur(G):
    T = np.zeros(shape=dataGrid.dims)
    for i,v in enumerate(G):
        x,y = dataGrid.coord(i)
        T[x-1][y-1] = v
    T_blurred = gaussian_filter(T, sigma=blur_const)
    final = np.empty(shape=G.shape)
    for i,v in enumerate(G):
        x,y = dataGrid.coord(i)
        final[i] = T_blurred[x-1][y-1]
    return final


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

while len(S) < 50:
    start_time()
    blurred = blur(np.power(G,power))
    G_norm = blurred / np.sum(blurred)

    stop_time()

    plotDataGrid(ax[0,0],np.power(G,power),dataGrid)
    plotDataGrid(ax[0,1],G_norm,dataGrid)
    ax[0,2].imshow(exp_data)

    measured_points = np.zeros(dataGrid.dims)
    for s in S:
        x,y = dataGrid.coord(s)
        measured_points[x-1,y-1] = 1
    ax[1,0].imshow(measured_points)

    start_time()

    #Note: cells numbering starts at 1
    data_range = range(1,dataGrid.size+1)

    cells = np.random.choice(data_range, 1, p=G_norm)
    while cells[0] in S:
        cells = np.random.choice(data_range, 1, p=G_norm)

    C = cells[0]
    M[C-1] = dataGrid.data_at_loc(C)[:,1] #"taking a measurement"
    S.add(C)

    stop_time()
    next_x,next_y = dataGrid.coord(C)
    sct_next = ax[0,1].scatter(next_y-1,next_x-1,s=15,c='red')
    sct_old = ax[0,1].scatter(old_y-1,old_x-1,s=15,c='purple')
    start_time()

    sim_list = []
    for K in dataGrid.neighbors(C).values():
        if not K in S:
            M[K-1] = dataGrid.data_at_loc(K)[:,1]  #"taking a measurement"
            S.add(K)
        sim_list = [get_similarity(C-1,K-1)] + sim_list

    G[C-1] = max(sim_list)


    stop_time()
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


    full_data = interpolateData(M,4,dataGrid)
    exp_data = clipSimilarityMatrix(getSimilarityMatrix(full_data,dataGrid))


    plt.draw()
    canvas.draw()

    #print("saving frame")
    frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    w,h = canvas.get_width_height()
    frame = np.reshape(frame,(h,w,3))
    video.append(frame)


    sct_next.remove()
    sct_old.remove()
    old_x = next_x
    old_y = next_y


imageio.mimwrite("/home/sasha/Desktop/python/videos/PSG.mp4", video, fps=2)


print("Finished Sampling")
print("_________________")


full_data = interpolateData(M,dataGrid)
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
