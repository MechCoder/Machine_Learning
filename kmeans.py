from scipy import io
import numpy as np
import pylab
import sys

# Hoping that there is a maximum of 8 clusters
colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo']

# Assuming input is in the form of a mat file, which is bad.
data = io.loadmat(sys.argv[1])['X']
K = int(sys.argv[2])
delta = float(sys.argv[-1])
fnum = data.shape[1]
tset = data.shape[0]

datat = data.T  # Returns features rowwise.
if fnum == 2:
    pylab.plot(datat[0], datat[1], 'ro')
    pylab.show()

centroid = np.zeros([K, fnum])
maxf = np.amax(data, axis=0)
minf = np.amin(data, axis=0)

initial_centroids = np.zeros([K, fnum])
next_centroids = np.zeros([K, fnum])

# Initialising random clusters
for i in xrange(fnum):
    initial_centroids[:, i] = np.random.uniform(maxf[i], minf[i], size=K)

# Function which returns the index of the nearest cluster of the nearest dataset.
def nearestCentroid(i):
    count = 0
    min_ = sum((i - initial_centroids[0])**2)
    for index, element in enumerate(initial_centroids[1: ]):
        testmin = sum((i - element)**2)
        if testmin < min_:
            min_ = testmin
            count =  index + 1
    return count 

tdelta = 10
while tdelta > delta:
    cmat = [[] for i in range(K)]

    # Storing indexes of the nearest clusters of the dataset.
    for daind, dlist in enumerate(data):
        c = nearestCentroid(dlist)
        cmat[int(c)].append(daind)
    cmat = np.asarray(cmat)  # Converting to numpy array for easier manipulation.
    pylab.plot(initial_centroids.T[0], initial_centroids.T[1], 'ro', markersize=20)
    for cind, cm in enumerate(cmat):
        temp = data[cm]
        if fnum == 2:
             pylab.plot(temp.T[0], temp.T[1], colors[cind])
        next_centroids[cind] = sum(temp)/len(temp)
    if fnum == 2:
        pylab.show()
    tdelta = np.amax((next_centroids - initial_centroids)/initial_centroids)
    initial_centroids = next_centroids.copy()
