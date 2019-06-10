#Probabilistic Similarity Gradient


from data_loading.data_grid_TiNiSn import DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C

from utils.utils import plotDataGrid,trim_outside_grid, interpolateData, similarity
from utils.utils import getDissimilarityMatrix, clipSimilarityMatrix, dict_to_csv
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import imageio
import numpy as np
import time


import argparse

parser = argparse.ArgumentParser(description='Run Probabilistic Similarity Gradient Simulation')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='seed algorithm')
parser.add_argument('-k','--G_init', type=float, default=0.1,
                    help='initial value for similarities')
parser.add_argument('-p','--power', type=int, default=20,
                    help='y=x^p scale for similarities')
parser.add_argument('-b','--blur', type=int, default=4,
                    help='sigma value for gaussian blur')
parser.add_argument('-N', type=int, default=50,
                    help='number of samples')
parser.add_argument('-v','--video', action='store_true',
                    help="Save training video")
parser.add_argument('--graphics', action='store_true',
                    help="Show plot real time")
parser.add_argument('--delay', type=float, default=0.001,
                    help='delay between video frames')
parser.set_defaults(video=False)
parser.set_defaults(graphics=False)
args = parser.parse_args()


#seed algorithm
seed = args.seed
np.random.seed(seed)

#set up DataGrid object
dataGrid = DataGrid_TiNiSn_500C()
true_data = clipSimilarityMatrix(getDissimilarityMatrix(dataGrid.get_data_array(),dataGrid))


#set up array to store plots
if args.video:
    video = []
    data_log = {}
    file_name = "PSG-" + str(seed)

#set up the visuals
if args.video or args.graphics:
    fig = plt.figure(num='Probabilistic Similarity Gradient')
    ax = fig.subplots(nrows=2, ncols=3)
    [[x.axis('off') for x in y] for y in ax]
    [[x.set_ylim(-1,15) for x in y] for y in ax]
    ax[0,0].title.set_text('Dissimilarity\nMatrix')
    ax[0,1].title.set_text('Sampling\nProbability')
    ax[0,2].title.set_text('Interpolated\nMeasurements')
    ax[1,0].title.set_text('Measurements')
    ax[1,2].title.set_text('True Data')

    fig.tight_layout()
    ax[1,2].imshow(trim_outside_grid(true_data,dataGrid))
    text = ax[1,1].text(0, 0, "", fontsize=10)

#initialize variables
exp_data = np.zeros(true_data.shape) #experimental data
old_x = 1
old_y = 1


#CONSTANTS
k = args.G_init
power = args.power
blur_const = args.blur
NUMBER_OF_SAMPLES = args.N

#DATA STRUCTURES
M = np.empty(shape=(dataGrid.size,dataGrid.data_length))
G = np.full(shape=dataGrid.size,fill_value = k)
S = set()


#cosine similarity function using two grid positions
def get_similarity(d1,d2):
    return similarity(M[d1],M[d2])

#define blur function to be used on flattened arrays
def blur(G):
    T = np.zeros(shape=dataGrid.dims)
    for i,v in enumerate(G):
        x,y = dataGrid.coord(i+1)
        T[x-1][y-1] = v
    T_blurred = gaussian_filter(T, sigma=blur_const)
    final = np.empty(shape=G.shape)
    for i,v in enumerate(G):
        x,y = dataGrid.coord(i+1)
        final[i] = T_blurred[x-1][y-1]
    return final

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

# MAIN LOOP
i = 0
while len(S) < NUMBER_OF_SAMPLES:
    i += 1
    start_time()
    # Create Probability Distribution
    blurred = blur(np.power(G,power))
    G_norm = blurred / np.sum(blurred)

    #Select a random cell to measure
    data_range = range(1,dataGrid.size+1) #Note: cells numbering starts at 1
    cells = np.random.choice(data_range, 1, p=G_norm)
    while cells[0] in S:
        cells = np.random.choice(data_range, 1, p=G_norm)
    C = cells[0]


    stop_time()
    # Plotting
    if args.video or args.graphics:
        plotDataGrid(ax[0,0],np.power(G,power),dataGrid)
        plotDataGrid(ax[0,1],G_norm,dataGrid)
        ax[0,2].imshow(trim_outside_grid(exp_data,dataGrid))

        measured_points = np.full(dataGrid.dims,.1)
        for s in S:
            x,y = dataGrid.coord(s)
            measured_points[x-1,y-1] = 1
        ax[1,0].imshow(measured_points)

        next_x,next_y = dataGrid.coord(C)
        sct_next = ax[0,1].scatter(next_y-1,next_x-1,s=15,c='red')
        sct_old = ax[0,1].scatter(old_y-1,old_x-1,s=15,c='purple')
    start_time()

    #Take a measurement at C
    M[C-1] = dataGrid.data_at_loc(C)[:,1]
    S.add(C)

    #Take aditional measurements as necesary and calculate Dis(C)
    sim_list = []
    for K in dataGrid.neighbors(C).values():
        if not K in S:
            M[K-1] = dataGrid.data_at_loc(K)[:,1]  #"taking a measurement"
            S.add(K)
        sim_list = [get_similarity(C-1,K-1)] + sim_list

    G[C-1] = max(sim_list)


    stop_time()

    #Additional Plotting
    if args.video or args.graphics:
        mse = float(np.square(np.subtract(exp_data, true_data)).mean())
        l2 = float(np.sum(np.square(np.subtract(exp_data, true_data))))
        l1 = float(np.sum(np.abs(np.subtract(exp_data, true_data))))
        if args.video:
            data_log[i] = {'mse':mse,"l2":l2,"l1":l1}

        times = [get_time()] + times
        s = "Avg Sample Time: \n"
        s += str(float(sum(times)/len(times))) + "\n"
        s += "Mean Squared Error: \n"
        s += str(mse) + "\n"
        s += "L2 Distance: \n"
        s += str(l2) + "\n"
        s += "L1 Distance: \n"
        s+= str(l1) + "\n"
        text.set_text(s)

    #plotting graphics to screen
    if args.graphics:
        plt.draw()
        plt.pause(args.delay)

    #saving frame to video
    if args.video:
        fig.canvas.draw()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        w,h = fig.canvas.get_width_height()
        frame = np.reshape(frame,(h,w,3))
        video.append(frame)


    full_data = interpolateData(M,4,dataGrid)
    exp_data = clipSimilarityMatrix(getDissimilarityMatrix(full_data,dataGrid))

    #resetting scatter plot and points
    if args.video or args.graphics:
        sct_next.remove()
        sct_old.remove()
        old_x = next_x
        old_y = next_y



# END
#__________________________________________________


#save video as file_name
if args.video:
    video_path = "/home/sasha/Desktop/python/videos/"
    imageio.mimwrite(video_path + file_name + ".mp4", video, fps=2)
    data_path = "/home/sasha/Desktop/python/logs/"
    dict_to_csv(data_log,data_path,file_name)
    print("Video saved to " + video_path)
    print("Data log save to " + data_path)


#leave plot open
if args.graphics:
    plt.show()

print()
print("Finished Sampling")
print("_________________")


full_data = interpolateData(M,4,dataGrid)
exp_data = clipSimilarityMatrix(getDissimilarityMatrix(full_data,dataGrid))


mse = float(np.square(np.subtract(exp_data, true_data)).mean())
l2 = float(np.sum(np.square(np.subtract(exp_data, true_data))))
l1 = float(np.sum(np.abs(np.subtract(exp_data, true_data))))

print("Mean Squared Error: ")
print(str(mse) + "\n")
print("L2 Distance: ")
print(str(l2) + "\n")
print("L1 Distance: ")
print(str(l1) + "\n")
