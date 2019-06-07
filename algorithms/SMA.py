#Similarity from measurement Avereges

from data_loading.data_grid import DataGrid
from utils.utils import trim_outside_grid, interpolateData, similarity, interpolateDataAvg
from utils.utils import getDissimilarityMatrix, clipSimilarityMatrix, dict_to_csv

from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import imageio
import numpy as np
import time

import argparse

parser = argparse.ArgumentParser(description='Run Probabilistic Similarity Gradient Simulation')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='seed algorithm')
parser.add_argument('-b','--blur', type=int, default=4,
                    help='sigma value for gaussian blur')
parser.add_argument('-N', type=int, default=50,
                    help='number of samples')
parser.add_argument('-v','--video', action='store_true',
                    help="Save training video")
parser.add_argument('--graphics', action='store_true',
                    help="Show plot real time")
parser.add_argument('--delay', type=float, default=0.5,
                    help='delay between video frames')
parser.set_defaults(video=False)
parser.set_defaults(graphics=False)
args = parser.parse_args()


#seed algorithm
seed = args.seed
np.random.seed(seed)

#set up DataGrid object
path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSub_1D.csv"""
dataGrid = DataGrid(data_path,regex)
true_data = clipSimilarityMatrix(getDissimilarityMatrix(dataGrid.get_data_array(),dataGrid))



#set up array to store plots
if args.video:
    video = []
    data_log = {}
    file_name = "PSG-" + str(seed)

#set up the visuals
if args.video or args.graphics:
    fig = plt.figure()
    ax = fig.subplots(nrows=2, ncols=3)
    [[x.axis('off') for x in y] for y in ax]
    fig.tight_layout()

    ax[1,2].imshow(trim_outside_grid(true_data,dataGrid))
    text = ax[1,1].text(0, 0, "", fontsize=8)


#initialize variables
exp_data = np.zeros(true_data.shape)
old_x = 1
old_y = 1


#CONSTANTS
blur_const = args.blur
NUMBER_OF_SAMPLES = args.N

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

i = 0
while len(S) < NUMBER_OF_SAMPLES:
    i += 1
    start_time()

    dissim = getDissimilarityMatrix(interpolateDataAvg(M),dataGrid)
    blurred = gaussian_filter(dissim, sigma=blur_const)
    flat = convertTo1D(blurred)
    if  np.sum(flat) == 0:
        Distribution = np.full(shape=(dataGrid.size),fill_value = 1/dataGrid.size)
    else:
        Distribution = flat / np.sum(flat)

    stop_time()
    #Plotting
    if args.video or args.graphics:
        ax[0,0].imshow(trim_outside_grid(dissim,dataGrid))
        ax[0,1].imshow(trim_outside_grid(blurred,dataGrid))
        ax[0,2].imshow(trim_outside_grid(exp_data,dataGrid))

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
    if args.video or args.graphics:
        next_x,next_y = dataGrid.coord(C)
        sct_next = ax[0,1].scatter(next_y-1,next_x-1,s=15,c='red')
        sct_old = ax[0,1].scatter(old_y-1,old_x-1,s=15,c='purple')

        mse = float(np.square(np.subtract(exp_data, true_data)).mean())
        l2 = float(np.sum(np.square(np.subtract(exp_data, true_data))))
        l1 = float(np.sum(np.abs(np.subtract(exp_data, true_data))))
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
