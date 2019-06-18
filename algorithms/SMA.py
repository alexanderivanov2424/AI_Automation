#Similarity from measurement Averages

from data_loading.data_grid_TiNiSn import DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
from algorithms.similarity_metrics.similarity import getSimilarityClass



from utils.plotvis import PlotVisualizer
from utils.timer import Timer
from utils.utils import interpolateDataCubic,interpolateDataAvg, getDissimilarityMatrix

from scipy.ndimage.filters import gaussian_filter

import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Run Probabilistic Similarity Gradient Simulation')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='seed algorithm')
parser.add_argument('-b','--blur', type=int, default=2,
                    help='sigma value for gaussian blur')
parser.add_argument('-p','--power', type=int, default=20,
                    help='y=x^p scale for probability')
parser.add_argument('-N', type=int, default=50,
                    help='number of samples')
parser.add_argument('-v','--video', action='store_true',
                    help="Save training video")
parser.add_argument('--graphics', action='store_true',
                    help="Show plot real time")
parser.add_argument('--delay', type=float, default=0.001,
                    help='additional delay for onscreen graphics')
parser.set_defaults(video=False)
parser.set_defaults(graphics=False)
args = parser.parse_args()


#seed algorithm
seed = args.seed
np.random.seed(seed)

#set up DataGrid object
dataGrid = DataGrid_TiNiSn_500C()

#set up Similarity Metric object
similarity_metric = getSimilarityClass('cosine')

#initialize variables
true_data = dataGrid.get_data_array() #true data set
exp_data = np.empty(true_data.shape) #experimental data
old_x = 1
old_y = 1


#set up the visuals
if args.video or args.graphics:
    plotVisualizer = PlotVisualizer('Similarity from Measurement Averages',(2,3), dataGrid)
    plotVisualizer.set_title(0,0,'Measurement Removed\nDis-Matrix')
    plotVisualizer.set_title(0,1,'Sampling\nProbability')
    plotVisualizer.set_title(0,2,'Interpolated\nMeasurements')
    plotVisualizer.set_title(1,0,'Measurements')
    plotVisualizer.set_title(1,2,'True Data')
    plotVisualizer.plot_measurement(true_data,similarity_metric,1,2)

if args.video:
    plotVisualizer.with_save("SMA-" + str(seed))


#CONSTANTS
blur_const = args.blur
power_const = args.power
NUMBER_OF_SAMPLES = args.N

#DATA STRUCTURES
M = np.zeros(shape=(dataGrid.size,dataGrid.data_length))
S = set()

#cosine similarity function using two grid positions
def get_similarity(d1,d2):
    return similarity_metric.similarity(M[d1],M[d2])

#Used to convert blurred similarity matrix to probability distribution
def convertTo1D(G):
    ret = np.empty(dataGrid.size)
    for i in range(dataGrid.size):
        x,y = dataGrid.coord(i+1)
        ret[i] = G[x-1][y-1]
    return ret

#Setting up Timer and time record
time = Timer()

#__________________________________________________
# START


# INITIAL SAMPLES
C_list = np.random.choice(range(1,dataGrid.size+1), 3)
for C in C_list:
    M[C-1] = dataGrid.data_at_loc(C)[:,1]
    S.add(C)

#Main Loop

while len(S) < NUMBER_OF_SAMPLES:
    time.start()
    exp_data = interpolateDataCubic(M,dataGrid)
    dissim = getDissimilarityMatrix(exp_data,similarity_metric,dataGrid)

    dissim_removed = dissim.copy()
    for s in S:
        x,y = dataGrid.coord(s)
        dissim_removed[x-1][y-1] = 0
    blurred = gaussian_filter(dissim_removed, sigma=blur_const)
    blurred = np.power(blurred,power_const)
    flat = convertTo1D(blurred)


    Distribution = flat / np.sum(flat)

    data_range = range(1,dataGrid.size+1)

    cells = np.random.choice(data_range, 1, p=Distribution)
    while cells[0] in S:
        cells = np.random.choice(data_range, 1, p=Distribution)
    C = cells[0]

    time.stop()
    #Plotting
    if args.video or args.graphics:
        #plot grids
        plotVisualizer.plot_grid(dissim_removed,0,0)
        plotVisualizer.plot_grid(blurred,0,1)
        plotVisualizer.plot_grid(dissim,0,2)

        #plot locations sampled so far
        measured_points = np.zeros(dataGrid.dims)
        for s in S:
            measured_points[tuple(x-y for x, y in zip(dataGrid.coord(s), (1,1)))] = 1
        plotVisualizer.plot_grid(measured_points,1,0)

        #plot current and next measurement
        next_x,next_y = dataGrid.coord(C)
        plotVisualizer.point(0,1,next_y-1,next_x-1,s=15,color='red')
        plotVisualizer.point(0,1,old_y-1,old_x-1,s=15,color='purple')

    time.start()

    #Take a measurement at C
    M[C-1] = dataGrid.data_at_loc(C)[:,1]
    S.add(C)

    time.stop()
    time.get_time()

    if args.video or args.graphics:
        plotVisualizer.plot_text(time.list(),true_data,exp_data,1,1)

    #plotting graphics to screen
    if args.graphics:
        plotVisualizer.show(args.delay)
    #saving frame to video
    if args.video:
        plotVisualizer.save_frame()

    #resetting scatter plot and points
    if args.video or args.graphics:
        plotVisualizer.reset_axis(0,1)
        old_x = next_x
        old_y = next_y


# END
#__________________________________________________



#save video as file_name
if args.video:
    video_path = "/home/sasha/Desktop/python/videos/"
    data_path = "/home/sasha/Desktop/python/logs/"
    plotVisualizer.save_to_paths(video_path,data_path)
    print("Video saved to " + video_path)
    print("Data log save to " + data_path)


#Display final stats of simulation
print()
print("Finished Sampling")
print("_________________")

exp_data = interpolateDataCubic(M,dataGrid)

mse = float(np.square(np.subtract(exp_data, true_data)).mean())
l2 = float(np.sum(np.square(np.subtract(exp_data, true_data))))
l1 = float(np.sum(np.abs(np.subtract(exp_data, true_data))))

print("Mean Squared Error: ")
print(str(mse) + "\n")
print("L2 Distance: ")
print(str(l2) + "\n")
print("L1 Distance: ")
print(str(l1) + "\n")

#leave plot open
if args.graphics:
    plotVisualizer.show_plot()
