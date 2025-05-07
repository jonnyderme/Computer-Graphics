# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 1st Assignment
# Deirmentzoglou Ioannis AEM : 10015

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as plt2
import time

def vector_interp(p1,p2,V1,V2,coord,dim):
    """""
    :parameter V1,V2: values of the vectors V1,V2
    :parameter p1,p2: coordinates of vectors V1,V2
    :parameter coord: takes the value of x or y, if dim = 1 or dim = 2
    :parameter dim: dimension along which the interpolation takes place
    :returns V: value of vector with coordinates x,y
    """""

    x1,y1 = p1
    x2,y2 = p2

    if dim == 1:
        x = coord
        if x > max(x1,x2):
            print("Given coordinates are not in range [x1,x2]")
            exit(1)
        if abs(x2-x1) < 1e-3:
            if abs(x-x1) < abs(x-x2):
                return x1
            else:
                return x2
        v1Coeff = np.abs(x2 - x) / np.abs(x2 - x1)
        v2Coeff = np.abs(x - x1) / np.abs(x2 - x1)
        V1 = np.array(V1)
        V2 = np.array(V2)
        V = V1*v1Coeff + V2*v2Coeff

        return V


    elif dim == 2:
        y = coord
        if y > max(y1,y2):
            print("Given coordinates are not in range [y1,y2]")
            exit(1)

        if abs(y2-y1) < 1e-3:
            if abs(y-y1) < abs(y-y2):
                return y1
            else:
                return y2
        v1Coeff = np.abs(y2 - y) / np.abs(y2 - y1)
        v2Coeff = np.abs(y - y1) / np.abs(y2 - y1)

        V1 = np.array(V1)
        V2 = np.array(V2)
        V = V1*v1Coeff + V2*v2Coeff

        return V

