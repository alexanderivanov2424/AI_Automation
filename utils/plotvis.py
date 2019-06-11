



class PlotVisualizer:

    def __init__(self, name, dims, titles):

        self.dims = dims
        self.fig = plt.figure(num=name)
        self.ax = fig.subplots(nrows=dims[0], ncols=dims[1])

        #hide axis labels
        [[x.axis('off') for x in y] for y in self.ax]

        #set axis limits
        [[x.set_ylim(-1,15) for x in y] for y in self.ax]
        [[x.set_xlim(-1,15) for x in y] for y in self.ax]

        [[self.ax[x,y].title.set_text(titles[x,y]) for y in range(dims[1])] for x in range(dims[0])]
        self.fig.tight_layout()
        self.text = None

    def with_save(file_name):
        self.save = True
        self.video = []
        self.data_log = {}
        self.file_name = file_name

    def save_frame():
        self.fig.canvas.draw()
        plt.draw()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        w,h = fig.canvas.get_width_height()
        frame = np.reshape(frame,(h,w,3))
        self.video.append(frame)

    def show(delay):
        plt.draw()
        plt.pause(delay)


    def plot_grid(self,grid,data_grid,r,c):
        G = trim_outside_grid(grid,data_grid)
        self.ax[r,c].imshow(G)


    def plot_measurement(self,measurements,data_grid,r,c):
        self.ax[r,c].imshow(getDissimilarityMatrix(measurements,data_grid))


    def plot_text(self,times,true,exp,r,c):
        if self.text = None:
            self.text = self.ax[r,c].text(0, 0, "", fontsize=10)
        avg_time = float(sum(times)/len(times))
        mse = float(np.square(np.subtract(exp_data, true_data)).mean())
        l2 = float(np.sum(np.square(np.subtract(exp_data, true_data))))
        l1 = float(np.sum(np.abs(np.subtract(exp_data, true_data))))
        if self.save:
            self.data_log[i] = {'mse':mse,"l2":l2,"l1":l1}

        s = "Avg Sample Time: \n"
        s += str(avg_time) + "\n"
        s += "Mean Squared Error: \n"
        s += str(mse) + "\n"
        s += "L2 Distance: \n"
        s += str(l2) + "\n"
        s += "L1 Distance: \n"
        s+= str(l1) + "\n"
        self.text.set_text(s)
