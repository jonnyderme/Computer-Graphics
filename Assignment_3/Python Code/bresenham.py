# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 1st Assignment
# Deirmentzoglou Ioannis AEM : 10015

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as plt2
import time

def bresenhamAlgorithm(vertexPoint1, vertexPoint2, axis):
    """""
    Implements the bresenham line drawing algorithm for two vertices
    :parameter vertexPoint1: coordinates [x,y] of vertex point 1
    :parameter vertexPoint2: coordinates [x,y] of vertex point 2
    :parameter axis: the axis along which the line is calculated
    :returns edgePixels: (M,2) array that contains the coordinates [x,y] of the pixels that belong to the line
    """""

    # Bresenham in y axis
    if axis == 1:
        # Find starting and ending point
        if vertexPoint1[1] <= vertexPoint2[1]:
            x0, y0 = vertexPoint1
            x1, y1 = vertexPoint2
        else:
            x0, y0 = vertexPoint2
            x1, y1 = vertexPoint1

        # Compute deltaX and deltaY
        deltaY = 2 * (y1 - y0)
        deltaX = 2 * np.abs(x1 - x0)
        f = -deltaX + deltaY / 2
        x = x0
        y = y0
        edgePixels = np.array([x, y])
        for y in range(y0 + 1, y1):
            if f < 0:
                if x0 < x1:
                    x = x + 1
                else:
                    x = x - 1
                f = f + deltaY
            f = f - deltaX
            edgePixels = np.vstack((edgePixels, np.array([x, y])))
        edgePixels = np.vstack((edgePixels, np.array([x1, y1])))

    # Bresenham in x axis
    elif axis == 0:
        # Find starting and ending point
        if vertexPoint1[0] < vertexPoint2[0]:
            x0, y0 = vertexPoint1
            x1, y1 = vertexPoint2
        else:
            x0, y0 = vertexPoint2
            x1, y1 = vertexPoint1

        # Compute deltaX and deltaY
        deltaX = 2 * (x1 - x0)
        deltaY = 2 * np.abs(y1 - y0)
        f = -deltaY + deltaX / 2
        x = x0
        y = y0
        edgePixels = np.array([x, y])
        for x in range(x0 + 1, x1):
            if f < 0:
                if y0 < y1:
                    y = y + 1
                else:
                    y = y - 1
                f = f + deltaX
            f = f - deltaY
            edgePixels = np.vstack((edgePixels, np.array([x, y])))
        edgePixels = np.vstack((edgePixels, np.array([x1, y1])))
    return edgePixels

def fillLine(vertexPoint1, vertexPoint2):
    """""
     Draws the line for two given vertices
    :param vertexpPoint1: coordinates [x,y] of vertex point 1
    :param vertexPoint2: coordinates [x,y] of vertex point 2
    :returns edgePoints: (M,2) array that contains the coordinates [x,y] of the pixels that belong to the line
    """""
    vertexPoint1 = np.array(vertexPoint1)
    vertexPoint2 = np.array(vertexPoint2)

    # If the two points are in the same line
    if vertexPoint1[0] == vertexPoint2[0]:
        x = vertexPoint1[0]
        start = min(vertexPoint1[1], vertexPoint2[1])
        end = max(vertexPoint1[1], vertexPoint2[1])

        edgePoints = np.array([x, start])
        for y in range(start + 1, end + 1):
            edgePoints = np.vstack((edgePoints, np.array([x, y])))

    # If the two points are in the same column
    if vertexPoint1[1] == vertexPoint2[1]:
        y = vertexPoint1[1]
        start = min(vertexPoint1[0], vertexPoint2[0])
        end = max(vertexPoint1[0], vertexPoint2[0])

        edgePoints = np.array([start, y])
        for x in range(start + 1, end + 1):
            edgePoints = np.vstack((edgePoints, np.array([x, y])))

    # If the two points are neither in the same column and line, perform bresenham in x or y axis, depending on the slope
    # If slope < 1 -> bresenham in y axis, else -> bresenham in x axis
    else:
        # Find slope
        slope = (vertexPoint1[0] - vertexPoint2[0]) / (vertexPoint2[1] - vertexPoint1[1])
        if np.abs(slope) < 1:
            # Bresenham in y axis (axis = 1)
            edgePoints = bresenhamAlgorithm(vertexPoint1, vertexPoint2, axis=1)
        else:
            # Bresenham in x axis
            edgePoints = bresenhamAlgorithm(vertexPoint1, vertexPoint2, axis=0)

    return edgePoints          

