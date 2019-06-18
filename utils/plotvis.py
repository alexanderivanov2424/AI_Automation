'''
PLOT VISUALIZER

Object used to visualise algorithm steps.

Renders grayscale grids, measurement arrays, and text.
Handles saving data and video file.
'''
from data_loading.data_grid import DataGrid

from utils.utils import trim_outside_grid
from utils.utils import getDissimilarityMatrix, dict_to_csv

import matplotlib.pyplot as plt
import matplotlib
import imageio
import numpy as np
import os



class PlotVisualizer:

    def __init__(self, name, dims, dataGrid, titles=None):

        """
        Initialize plots

        # name - figure name
        # dims - dimensions of subplots
        # dataGrid - dataGrid object to be used for rendering
        # titles - [optional] titles of subplots. can be set via method

        """

        #plot dimensions
        self.dims = dims
        self.dataGrid = dataGrid
        self.fig = plt.figure(num=name)
        self.ax = self.fig.subplots(nrows=dims[0], ncols=dims[1])

        #hide axis labels
        [[x.axis('off') for x in y] for y in self.ax]

        #set axis limits
        [[x.set_ylim(-1,15) for x in y] for y in self.ax]
        [[x.set_xlim(-1,15) for x in y] for y in self.ax]
        if not titles==None:
            [[self.ax[x,y].title.set_text(titles[x,y]) for y in range(dims[1])] for x in range(dims[0])]

        self.fig.tight_layout()
        self.text = [[None for y in range(dims[1])] for x in range(dims[0])]
        self.scatter = [[[] for y in range(dims[1])] for x in range(dims[0])]
        self.save = False

    def check(self,r,c):
        if r < 0 or self.dims[0] <= r:
            print("Invalid row for subplot: " + str(r))
            sys.exit()
        if c < 0 or self.dims[1] <= c:
            print("Invalid col for subplot: " + str(c))
            sys.exit()

    def set_title(self,r,c,title):
        """
        Set title of specific subplot
        """
        self.check(r,c)
        self.ax[r,c].title.set_text(title)
        self.fig.tight_layout()

    def point(self,r,c,x,y,s,color):
        """
        Plot a point in a subplot
        """
        self.check(r,c)
        self.scatter[r][c].append(self.ax[r,c].scatter(x,y,s=s,c=color))

    def reset_axis(self,r,c):
        """
        Reset points in a subplot
        """
        self.check(r,c)
        [x.set_visible(False) for x in self.scatter[r][c]]

    def with_save(self,file_name):
        """
        Toggle saving video and data files
        This only records video and data but does not save
        """
        self.save = True
        self.video = []
        self.data_log = {}
        self.file_name = file_name
        self.step = 0

    def save_frame(self):
        """
        Saves a specific frame to the video
        """
        if not self.save:
            print("# WARNING:\tSaving not set\n\t\tRun \"with_save(\'file_name\')\"")
            print()
            return
        self.fig.canvas.draw()
        plt.draw()
        frame = np.fromstring(self.fig.canvas.tostring_rgb(), dtype='uint8')
        w,h = self.fig.canvas.get_width_height()
        frame = np.reshape(frame,(h,w,3))
        self.video.append(frame)


    def save_to_paths(self,video_path,data_path):
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        imageio.mimwrite(os.path.join(video_path, self.file_name +".mp4"), self.video, fps=2)
        print("Video saved to " + video_path)

        if not os.path.exists(data_path):
            os.makedirs(data_path)
        dict_to_csv(self.data_log,os.path.join(data_path,self.file_name + ".csv"))
        print("Data log save to " + data_path)

    def show_plot(self):
        plt.show()

    def show(self,delay):
        plt.draw()
        plt.pause(delay)


    def plot_grid(self,grid,r,c):
        self.check(r,c)
        if len(grid.shape) == 1:
            G = np.zeros(shape=self.dataGrid.dims)
            for i,v in enumerate(grid):
                x,y = self.dataGrid.coord(i+1)
                G[x-1][y-1] = v
            G = trim_outside_grid(G,self.dataGrid)
        else:
            G = trim_outside_grid(grid,self.dataGrid)
        self.ax[r,c].imshow(G)


    def plot_measurement(self,measurements,metric,r,c):
        self.check(r,c)
        dis_matrix = getDissimilarityMatrix(measurements,metric,self.dataGrid)
        self.ax[r,c].imshow(trim_outside_grid(dis_matrix,self.dataGrid))


    def plot_text(self,times,true_data,exp_data,r,c):
        self.check(r,c)
        if self.text[r][c] == None:
            self.text[r][c] = self.ax[r,c].text(0, 0, "", fontsize=10)
        if len(times) == 0 :
            avg_time = 0
        else:
            avg_time = float(sum(times)/len(times))
        mse = float(np.square(np.subtract(exp_data, true_data)).mean())
        l2 = float(np.sum(np.square(np.subtract(exp_data, true_data))))
        l1 = float(np.sum(np.abs(np.subtract(exp_data, true_data))))
        if self.save:
            self.data_log[self.step] = {'mse':mse,"l2":l2,"l1":l1}
            self.step += 1

        s = "Avg Sample Time: \n"
        s += str(avg_time) + "\n"
        s += "Mean Squared Error: \n"
        s += str(mse) + "\n"
        s += "L2 Distance: \n"
        s += str(l2) + "\n"
        s += "L1 Distance: \n"
        s+= str(l1) + "\n"
        self.text[r][c].set_text(s)
