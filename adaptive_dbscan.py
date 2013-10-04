from scipy import io
import numpy as np
import pylab
import sys


r"""
References:
1. http://en.wikipedia.org/wiki/DBSCAN
2. Research on Adaptive Parameters Determination in
DBSCAN Algorithm by Hongfang Zhou, Peng Wang, Hongyan Li
"""
# Preprocessing
data = io.loadmat(sys.argv[1])['X']  # Change this if input is not .mat
dset, fnum = data.shape
dist = np.zeros([dset, dset])  # To store the sorted DDM
sort_arr = []  # To store the indices of the sorted arrays
# To store the closest index of each array in sort_arr for which distance
# is greatest of the nearest neighbours but just lesser than eps
rQuery = []
final_cluster = []  # To store the final cluster
visit_array = np.zeros(dset)  # Checking if a point has been visited or not.

def flatten_check(x):
    r"""
    Checking if x is present in final_cluster
    """
    for i in final_cluster:
        for j in i:
            if j == x:
                return True

def distance(x, y):
    r"""
    Returns distance without square root
    """
    return sum((x - y)**2)

def expandCluster(fpind, npind):
    r"""
    fpind is the index of the point which starts the cluster.
    npind is the index of the farthest nearest neighbour which is still
    less than eps.
    It basically works this way,
    1] For all the neighbouring points of the cluster point.
    2] A list of other neighbouring points are taken which are less than eps.
    3] If the number is greater than minpts, they are added to initial neighbouring
       points, which builds the cluster.
    Finally all points are added to the cluster.
    """
    cluster = []
    cluster.append(fpind)

    neighborPts = list(sort_arr[fpind][1 : npind + 1])
    for point in neighborPts:
        if not visit_array[point]:
            visit_array[point] = 1
            nbind = rQuery[point]
            if nbind >= minpts:
                temp = list(sort_arr[point][1: nbind + 1])
                neighborPts.extend(set(temp) - set(neighborPts) - set([fpind]))
        if not flatten_check(point):
            cluster.append(point)
    return cluster

# Determining distance distribution matrix
# Basically distance of each point from the other n - 1 points are stored in
# sorted order rowwise
# sort_arr stores the indices of the sorted array rowwise.

eps = minpts = 0
for i in xrange(dset):
    for j in xrange(i, dset):
        dist[i][j] = dist[j][i] = distance(data[i], data[j])
    sort_arr.append(np.argsort(dist[i]))  # Storing indices of the sorted arrays.
    dist[i] = dist[i][sort_arr[i]]  # Sorting distances according to the indices.

    # The mean of the fourth nearest neighbour gives eps.
    eps += dist[i][5]

eps = eps/dset

# Finding minimum points, which is given by the mean of the points in eps neighbourhood 
for i in xrange(dset):
    j = 1
    while 1:
        if dist[i][j] > eps:
            minpts += j - 1
            rQuery.append(j - 1)
            break
        j += 1

minpts = minpts/dset

# Real DBSCAN starts from here.s
for pind, point in enumerate(visit_array):
    if not point:  # Unvisited
        nbind = rQuery[pind]
        if not nbind < minpts:
            final_cluster.append(expandCluster(pind, nbind))
        visit_array[point] = 1

print eps, minpts
