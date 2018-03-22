#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np
# from cpython.array import array, clone
from cython.view cimport array as cvarray
cimport cython
import csv
import time
# from cpython.array cimport array, clone
from libc.math cimport *

cdef double [:] zNormalize(double [:] my_time_series):
    cdef int index = 0
    cdef int length = my_time_series.shape[0]
    # print(length)
    cdef double[:,] data = np.zeros(length) # last bit of python doesnt matter only called once



    cdef double mean = _sum(my_time_series, length) / length

    while index < length:
        data[index] = (my_time_series[index] - mean) ** 2
        index += 1
    cdef double sum_squares = _sum(data, length)
    cdef double deviation = (sum_squares/ length) ** .5

    index = 0
    while index < length:
        my_time_series[index] = (my_time_series[index] - mean) / deviation
        index += 1

    return my_time_series

cdef double _sum(double[:,] item, int size) nogil:
    cdef int i = 0
    cdef double ret = 0.0
    for i in range(size):
        ret += item[i]
    return ret

# adapted from Mueen and Keogh 2012
# Q query: a list of numbers. The motif to be matched to.
# filePathT: A the path pointing to a csv containing one number per row.
# sample_rate: an integer. If you wish to sample every nth value of the time series, set this to n. Note that the matching motif much also be collected at this same sampling rate.
# normalize: If true, compare distaces between z-normalized query and test-data. If false compare absolute distance
# returns the first index of the closest fit, nn
cpdef double[:,] find_matches(double[:,] T, double[:,] Q, int knn, int sample_rate=1):
    # t = thing to search in len length. q is query, len m
    # thor modifications: T is regular np array, use T[count] instead of the .next function

    cdef double bestSoFar = float("inf")
    cdef int length = T.shape[0]
    cdef int count = 0  # the 'step' in the time series
    Q = zNormalize(Q)
    cdef int m = Q.shape[0]
    cdef double[:,] X = np.zeros(m)
    cdef double ex = 0
    cdef double ex2 = 0

    cdef double[:,] nn = np.zeros(knn)
    cdef double[:,] nn_dist = np.zeros(knn)
    cdef int j = 0

    cdef double dist = 0
    cdef double u
    cdef double sdev
    cdef int knn_pos = 0
    cdef int knn_pos_plus = 0
    if knn > 1:
        knn_pos_plus = 1
    cdef int loc
    cdef int i
    cdef double tnext
    tnext = T[count]
    hasNext = True
    # return nn
    while hasNext:  # can choose to limit the depth you look into the TS with "and count < 100000:"
        count += 1
        if count % sample_rate == 0:
            i = count / sample_rate % m
            i = int(i)
            X[i] = tnext  # circular buffer to store current subsequence.
            ex = ex + X[i]  # iteratively sum up values to use for the mean
            ex2 = ex2 + X[i] ** 2  # sum up squared values to use for sdev

            if count >= m - 1:
                # print("entering distance calc")
                u = ex / m  # u is mu, or the mean

                sdev = abs(ex2 / m - u ** 2) ** (0.5)
                j = 0
                dist = 0

                # compare Q and T[i]
                while j < m and dist < bestSoFar:

                    dist = dist + (Q[j] - (X[(i + j) % m] - u) / sdev) ** 2


                    j = j + 1
                # print("got distance "+str(dist)) # can get NANs here!
                if dist < bestSoFar:
                    loc = count - m  # count gives the end of the matched motif. Move m spaces back to finds its head.

                  # print("inserting best" +str(loc)+ " "+ str(knn_pos))
                    nn[knn_pos] = loc
                    nn_dist[knn_pos] = dist
                    bestSoFar = nn_dist[knn_pos_plus]
                    if bestSoFar == 0 :
                        bestSoFar = 9999999999999
                    knn_pos += 1
                    knn_pos_plus += 1
                    knn_pos = knn_pos % knn
                    knn_pos_plus = knn_pos_plus % knn


                # keep the mean and sdev and moving averages.
                ex = ex - X[(i + 1) % m]
                ex2 = ex2 - X[(i + 1) % m] ** 2
        if count < length:
            tnext = T[count]
        else:
            hasNext = False
          # print("finished executing")
    return nn  #  list of knn positions