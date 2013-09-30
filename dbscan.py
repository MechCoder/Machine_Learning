from scipy import io
import numpy as np
import pylab
import sys

# Hoping that there is a maximum of 8 clusters
colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo']

# Assuming input is in the form of .mat file. This can be
# changed later.
data = io.loadmat(sys.argv[1])['X']
minpts = int(sys.argv[2])
eps = float(sys.argv[3])
inum, fnum = data.shape
final_cluster = []
noise_cluster = []

def flatten_check(ind_):
    r"""
    Check if the required index is already present in the
    final_cluster.
    """
    for x in final_cluster:
        for y in x:
            if y == ind_:
                return True

def rQuery(point, point_index):
    r"""
    Returns
    1. Indices of those points, that are a distance lesser than eps,
       from a given point so that it need not be computed again.
    2. The number of such points.
    """
    npoints = []
    npoints_index = []
    count = 0
    for dind, dist in enumerate(data[point_index + 1: ] - point):  # Vectorisation
        if sum(dist**2) < eps:
            count += 1
            npoints_index.append(point_index + 1 + dind)
    return npoints_index, count

def expandCluster(cpoint, cpindex, npoints_index):
    r"""
    1. cpoint is the point where the cluster starts.
    2. cpindex is the index of the cluster point in the bigger
       dataset data
    3. npoints_indices is the list of indices of neighbouring points
       whose distance from the cluster point is lesser than a given value.
    """
    ncluster = []
    ncluster.append(cpindex)

    # Check in the neighbouring region of the cluster point
    for point_ind, nind in enumerate(npoints_index):
        # If it hasn't been visited yet, mark it as visited.
        if not dbs[nind]:
            dbs[nind] = 1
            nnpoints_index, count = rQuery(data[nind], -1)
            if count >= minpts:
                newpoints = set(nnpoints_index) - set(npoints_index)
                npoints_index.extend(newpoints)
        # Check if nind is there in final_cluster
        if not flatten_check(nind):
            ncluster.append(nind)

    return ncluster

dbs = np.zeros(inum)
for point_index, point in enumerate(dbs):
    if not point:  # Unchecked
        cpoint = data[point_index]
        # So that distances are not calculated multiple times.
        npoints_index, count = rQuery(cpoint, point_index)
        if count >= minpts: 
            final_cluster.append(expandCluster(cpoint, point_index, npoints_index))    
        else:
            noise_cluster.append(cpoint)
        dbs[point_index] = 1


for fcli, temp in enumerate(final_cluster):
    pylab.plot(data[np.array(temp)].T[0], data[np.array(temp)].T[1], colors[fcli], markersize=5) 
pylab.show()
