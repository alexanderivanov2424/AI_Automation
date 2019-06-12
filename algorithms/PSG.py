#Probabilistic Similarity Gradient


from data_loading.data_grid_TiNiSn import DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C

from utils.plotvis import PlotVisualizer
from utils.timer import Timer
from utils.utils import interpolateData, similarity


from scipy.ndimage.filters import gaussian_filter

import numpy as np


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
true_data = dataGrid.get_data_array()


if args.video or args.graphics:
    plotVisualizer = PlotVisualizer('Probabilistic Similarity Gradient',(2,3))
    plotVisualizer.set_title(0,0,'Dissimilarity\nMatrix')
    plotVisualizer.set_title(0,1,'Sampling\nProbability')
    plotVisualizer.set_title(0,2,'Interpolated\nMeasurements')
    plotVisualizer.set_title(1,0,'Measurements')
    plotVisualizer.set_title(1,2,'True Data')
    plotVisualizer.plot_measurement(true_data,dataGrid,1,2)
    #plotVisualizer.start()

if args.video:
    plotVisualizer.with_save("PSG-" + str(seed))

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
time = Timer()



#__________________________________________________
# START

# MAIN LOOP

while len(S) < NUMBER_OF_SAMPLES:
    time.start()
    # Create Probability Distribution
    blurred = blur(np.power(G,power))
    G_norm = blurred / np.sum(blurred)

    #Select a random cell to measure
    data_range = range(1,dataGrid.size+1) #Note: cells numbering starts at 1
    cells = np.random.choice(data_range, 1, p=G_norm)
    while cells[0] in S:
        cells = np.random.choice(data_range, 1, p=G_norm)
    C = cells[0]


    time.stop()
    # Plotting
    if args.video or args.graphics:
        #plot grids
        plotVisualizer.plot_grid(np.power(G,power),dataGrid,0,0)
        plotVisualizer.plot_grid(G_norm,dataGrid,0,1)
        plotVisualizer.plot_measurement(exp_data,dataGrid,0,2)

        #plot locations sampled so far
        measured_points = np.zeros(dataGrid.dims)
        for s in S:
            x,y = dataGrid.coord(s)
            measured_points[x-1,y-1] = 1
        plotVisualizer.plot_grid(measured_points,dataGrid,1,0)

        #plot current and next measurement
        next_x,next_y = dataGrid.coord(C)
        plotVisualizer.point(0,1,next_y-1,next_x-1,s=15,color='red')
        plotVisualizer.point(0,1,old_y-1,old_x-1,s=15,color='purple')

    time.start()

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


    time.stop()

    #update time list
    times = time.list()

    #Additional Plotting
    if args.video or args.graphics:
        plotVisualizer.plot_text(times,true_data,exp_data,1,1)

    #plotting graphics to screen
    if args.graphics:
        plotVisualizer.show(args.delay)

    #saving frame to video
    if args.video:
        plotVisualizer.save_frame()


    exp_data = interpolateData(M,4,dataGrid)

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


print()
print("Finished Sampling")
print("_________________")


exp_data = interpolateData(M,4,dataGrid)

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
