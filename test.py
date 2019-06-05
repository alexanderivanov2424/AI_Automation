from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import euclidean
from delaunay import DelaunayChooser
import matplotlib.pyplot as plt
import numpy as np
import time

points = np.random.rand(1,2);
points = np.append(points,[[0,0],[0,1],[1,0],[1,1]], axis = 0)

vor = Voronoi(points,incremental=True)


fig = plt.figure();
ax = fig.add_subplot(111)

d = DelaunayChooser(np.array([[0,0],[0,1],[1,0],[1,1],[.5,.5]]))

t = 0
times = np.array([])

for i in range(200):
    pnt = np.random.rand(1,2)
    t = time.time()
    d.add_points(d.next_point(0))
    times = np.append(times,time.time()-t)


    #voronoi_plot_2d(vor,ax,show_vertices=False)

    #plt.plot(points[:,0],points[:,1],'o')
    #plt.pause(1)

print(np.sum(times)/len(times))
print(np.sum(times))
d.plot(ax)
d.plot(ax)
plt.show()

'''
1) Delaunay triangles
2) edge lengths
3) edge weights
4) triangle ares with herons formula
'''

#plt.triplot(points[:,0],points[:,1],tri.simplices.copy())

#plt.plot(vor.vertices[:,0],vor.vertices[:,1],'o')
