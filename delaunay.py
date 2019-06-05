from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import random



class DelaunayChooser:

    def __init__(self, points):
        self.points = points
        self.tri = Delaunay(self.points,incremental=True)

        self.data = Data(5)


    def add_points(self,pnt):
        self.points = np.append(self.points,pnt,axis = 0)
        self.tri.add_points(pnt)
        self.update_data()

    def next_point(self, eps):
        if(random.random() < eps):
            loc = np.where(self.data.areas == np.amax(self.data.areas))
            r = [0]
            if(len(loc[0]) > 1):
                r = np.random.randint(0,len(loc[0])-1,1)
            return (self.points[loc[0][r]] + self.points[loc[1][r]] + self.points[loc[2][r]])/3
        else:
            loc = np.where(self.data.lengths == np.amax(self.data.lengths))
            r = [0]
            if(len(loc[0]) > 1):
                r = np.random.randint(0,len(loc[0])-1,1)
            return (self.points[loc[0][r]] + self.points[loc[1][r]])/2

    def update_data(self):
        if(len(self.points) >= self.data.size):
            self.data.grow(2 * len(self.points))
        else:
            self.data.grow(len(self.points))
        for s in self.tri.simplices:
            a = self.length(s[0],s[1])
            b = self.length(s[1],s[2])
            c = self.length(s[0],s[2])
            self.data.lengths[s[0]][s[1]] = a
            self.data.lengths[s[1]][s[2]] = b
            self.data.lengths[s[0]][s[2]] = c
            self.data.areas[s[0]][s[1]][s[2]] = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)
            #print(self.data.areas[s[0]][s[1]][s[2]])

    def length(self, p1, p2):
        return np.sqrt(np.sum((self.points[p1] - self.points[p2])**2))

    def plot(self,ax):
        ax.triplot(self.points[:,0],self.points[:,1],self.tri.simplices.copy())



class Data:

    def __init__(self,init_size):
        self.size = init_size
        self.lengths = np.empty(shape=(init_size,init_size))
        self.areas = np.empty(shape=(init_size,init_size,init_size))

    def grow(self,new_size):

        self.lengths = np.zeros((new_size,new_size))
        self.areas = np.zeros((new_size,new_size,new_size))

        self.size = new_size
