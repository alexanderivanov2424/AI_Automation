from data_grid import DataGrid


from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


def cosineClustering(num_clusters,dataGrid):
    global cluster_dict
    if (num_clusters,"c") in cluster_dict.keys():
        return cluster_dict[(num_clusters,"c")]
    #cosine similarity function using two grid positions
    def similarity_vector(A,B):
        return np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B)
    def similarity(d1,d2):
        a = dataGrid.data[d1][:,1]
        b = dataGrid.data[d2][:,1]
        return similarity_vector(a,b)

    D = np.ones(shape=(dataGrid.size,dataGrid.size))
    for x in range(dataGrid.size):
        for y in range(dataGrid.size):
            D[x,y] = 1 - similarity(x+1,y+1)

    def get_cluster_grids(i,prev_agg=None,prev_points=None,prev_hue=None):
        #cluster the data based on similarity metric
        agg = AgglomerativeClustering(n_clusters=i, compute_full_tree = True, affinity='precomputed',linkage='complete')
        agg.fit(D)

        #cluster colors
        hues = [float(float(x)/float(i)) for x in range(1,i+1)]

        #make the clusters colors match up
        if not prev_agg==None and not prev_points==None and not prev_hue==None:
            dict = {}
            for x,l in enumerate(agg.labels_):
                dict[l] = prev_agg.labels_[x]
            cluster_split = [0,0]
            for index1,v1 in dict.items():
                for index2,v2 in dict.items():
                    if not index1 == index2 and v1 ==v2:
                        cluster_split = [index1,index2]
            cluster_split.sort()

            hues_new = np.zeros(i)
            for x in range(i):
                if x in cluster_split:
                    delta = float(cluster_split.index(x)*2-1)/float(i)
                    hues_new[x] = prev_hue[dict[x]] + delta/2
                else:
                    hues_new[x] = prev_hue[dict[x]]
            hues = hues_new

        #data grouped by cluster
        grouped_data = [[] for x in range(i)]
        for loc,val in enumerate(agg.labels_):
            grouped_data[val].append(dataGrid.data_at_loc(loc+1)[:,1])
        #average spectra for each cluster
        averages = [np.mean(x,axis=0) for x in grouped_data]


        #cluster grid to show where clusters fall
        cluster_grid = np.ones(shape = (dataGrid.dims[0],dataGrid.dims[1],3))
        #cluster grid with similarity to cluster average as brightness
        cluster_grid_scale = np.ones(shape = (dataGrid.dims[0],dataGrid.dims[1],3))
        for val in dataGrid.grid_locations:
            x,y = dataGrid.coord(val)
            cluster = agg.labels_[val-1]
            similarity = similarity_vector(dataGrid.data_at_loc(val)[:,1],averages[cluster])
            #adjusting the similarity gradient
            similarity = math.pow(similarity,power)
            cluster_grid_scale[y-1][dataGrid.dims[0]-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,similarity])
            cluster_grid[y-1][dataGrid.dims[0]-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])

        # x, y locations of spectra closest to average
        points_x = [-1 for x in range(i)]
        points_y = [-1 for x in range(i)]
        # grid location of averages
        points_loc = [-1 for x in range(i)]

        #find grid locations closest to average
        for loc in dataGrid.grid_locations:
            cluster = agg.labels_[loc-1]
            cur_x = points_x[cluster]
            cur_y = points_y[cluster]
            if cur_x == -1:
                x,y = dataGrid.coord(loc)
                points_x[cluster] = x
                points_y[cluster] = y
                points_loc[cluster] = loc
                continue
            sim_cur = similarity_vector(dataGrid.data_at(cur_x,cur_y)[:,1],averages[cluster])
            sim_new = similarity_vector(dataGrid.data_at_loc(loc)[:,1],averages[cluster])
            if sim_cur < sim_new:
                x,y = dataGrid.coord(loc)
                points_x[cluster] = x
                points_y[cluster] = y
                points_loc[cluster] = loc

        return agg, cluster_grid, cluster_grid_scale, points_x, points_y, points_loc, agg.labels_, hues
    return get_cluster_grids(num_clusters)


def peakReductionClustering(num_clusters,dataGrid,curveGrid):
    global cluster_dict

    if (num_clusters,"p") in cluster_dict.keys():
        return cluster_dict[(num_clusters,"p")]
    #Create Point Cloud
    SCALE = 100
    def to_point(x,y,p):
        return [(x-1)/dataGrid.dims[0],(y-1)/dataGrid.dims[1],SCALE*float(p)/5]
    peaks = []
    for k in curveGrid.data.keys():
        x,y = curveGrid.coord(k)
        [peaks.append(to_point(x,y,p)) for p in curveGrid.data_at_loc(k)[:,2]]

    #Cluster Point Cloud
    X = np.array(peaks)
    clustering = DBSCAN(eps=0.25, min_samples=5).fit(X)
    C = len(set(clustering.labels_).difference(set([-1])))

    #reduce dimentions
    M = np.zeros(shape=(curveGrid.size,C))
    for k in curveGrid.data.keys():
        x,y = curveGrid.coord(k)
        V = np.zeros(shape=C)
        for i,p in enumerate(curveGrid.data_at_loc(k)[:,2]):
            loc = clustering.labels_[peaks.index(to_point(x,y,p))]
            if loc == -1:
                continue
            M[k-1,loc] = 1#peakGrid.data_at_loc(k)[i,3]

    pca = PCA(n_components = 'mle',svd_solver='full').fit_transform(M)
    #pca = PCA(n_components = 20,svd_solver='full').fit_transform(M)

    def get_cluster_grids(i):
        agg = AgglomerativeClustering(n_clusters=i).fit(pca)

        hues = [float(float(x)/float(i)) for x in range(1,i+1)]

        cluster_centers = np.zeros(i)
        #sum vectors in each cluster
        cluster_sums = np.zeros(shape=(i,len(pca[0])))
        for val in dataGrid.grid_locations:
            cluster = agg.labels_[val-1]
            cluster_sums[cluster] = cluster_sums[cluster] + pca[0]

        #divide by cluster size to get average point
        for cluster in range(0,i):
            count = np.count_nonzero(agg.labels_==cluster)
            cluster_sums[cluster] = cluster_sums[cluster] / count
        #find cluster centers
        for cluster,center_v in enumerate(cluster_sums):
            cluster_v = pca[np.where(agg.labels_==cluster)[0]]
            index = np.argmin(np.sum(np.square(cluster_v - center_v),axis=1))
            cluster_centers[cluster] = np.where(agg.labels_==cluster)[0][index] + 1

        cluster_grid = np.ones(shape = (dataGrid.dims[0],dataGrid.dims[1],3))
        for val in dataGrid.grid_locations:
            x,y = dataGrid.coord(val)
            cluster = agg.labels_[val-1]
            cluster_grid[y-1][dataGrid.dims[0]-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])

        return cluster_grid, cluster_centers,agg.labels_
    return get_cluster_grids(num_clusters)


def peakCountGrid(peakGrid):
    peak_max_counts = np.zeros(i)
    for val in peakGrid.grid_locations:
        cluster = agg.labels_[val-1]
        peak_max_counts[cluster] = max(peak_max_counts[cluster],len(peakGrid.data_at_loc(val)[:,1]))

    peak_grid = np.ones(shape =(peakGrid.dims[0],peakGrid.dims[1],3))
    for val in peakGrid.grid_locations:
        x,y = peakGrid.coord(val)
        cluster = agg.labels_[val-1]
        k = len(peakGrid.data_at_loc(val)[:,1])/peak_max_counts[cluster]
        peak_grid[y-1][peakGrid.dims[0]-x] = matplotlib.colors.hsv_to_rgb([1,1,k])
    return peak_grid


def peakFWHMGrid(peakGrid):
    width_max = 0
    for val in peakGrid.grid_locations:
        width_max = max(width_max,np.nanmax(peakGrid.data_at_loc(val)[:,2].astype(np.float)))

    width_grid = np.ones(shape =(peakGrid.dims[0],peakGrid.dims[1]))
    width_grid.fill(np.nan)
    for val in peakGrid.grid_locations:
        x,y = peakGrid.coord(val)
        cluster = agg.labels_[val-1]
        k = np.nanmax(peakGrid.data_at_loc(val)[:,2].astype(np.float))
        width_grid[y-1][peakGrid.dims[0]-x] = k
    return width_grid
