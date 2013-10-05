from scipy import io
import numpy as np
import pylab
import sys

# Hoping that there is a maximum of 8 clusters
colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo']

# Function which returns the index of the nearest cluster of the nearest dataset.
def nearestCentroid(i, initial_centroids):
    count = 0
    min_ = sum((i - initial_centroids[0])**2)
    for index, element in enumerate(initial_centroids[1: ]):
        testmin = sum((i - element)**2)
        if testmin < min_:
            min_ = testmin
            count =  index + 1
    return count 

def kcluster(in_data, K, delta):
    r"""
    Function that outputs the final cluster. K is the number of clusters
    and delts is the minimum error
    """
    global data

    data = in_data
    fnum = data.shape[1]
    tset = data.shape[0]
    maxf = np.amax(data, axis=0)
    minf = np.amin(data, axis=0)
    initial_centroids = np.zeros([K, fnum])
    next_centroids = np.zeros([K, fnum])

    # Initialising random clusters
    for i in xrange(fnum):
        initial_centroids[:, i] = np.random.uniform(maxf[i], minf[i], size=K)

    tdelta = 10
    while tdelta > delta:
        cmat = dict([(str(i), []) for i in range(K)])

        # Storing indexes of the nearest clusters of the dataset.
        for daind, dlist in enumerate(data):
            c = nearestCentroid(dlist, initial_centroids)
            cmat[str(int(c))].append(daind)

        for cind, cm in cmat.items():
            temp = data[cm]
            next_centroids[int(cind)] = sum(temp)/len(temp)
        tdelta = np.amax((next_centroids - initial_centroids)/initial_centroids)
        initial_centroids = next_centroids.copy()
    return cmat
