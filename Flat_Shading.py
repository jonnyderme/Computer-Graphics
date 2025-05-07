# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 1st Assignment
# Deirmentzoglou Ioannis AEM : 10015

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as plt2
import time

from bresenham import bresenhamAlgorithm , fillLine

def f_shading(img, vertices, vcolors):
    """""
    :img: the already existing image with its triangles
    :vertices: Integer array 3x2 that contains the coordinates of triangle's vertices in each line.
    :vcolors: Array 3x3 that contains the color of triangle's vertex in each line with RGB values[0,1]
    :updated_img: Array MxNx3 that contains the RGB values for all the parts of the
    :triangle and the already existing triangles
    :returns updated_img: MxNx3 array containing the RGB values for all the parts of the triangle and the already existing triangles
    """""

    # Initialize variables x_k_min, x_k_max, y_k_min, y_k_max for filling algorithm
    x_k_min = np.zeros(3)
    x_k_max = np.zeros(3)
    y_k_min = np.zeros(3)
    y_k_max = np.zeros(3)

    # Find the x_k_min, x_k_max, y_k_min, y_k_max for every edge
    for k in range(3):
        if k == 2:
            x_k_min[k] = min(vertices[k][0], vertices[k - 2][0])
            x_k_max[k] = max(vertices[k][0], vertices[k - 2][0])
            y_k_min[k] = min(vertices[k][1], vertices[k - 2][1])
            y_k_max[k] = max(vertices[k][1], vertices[k - 2][1])
        else:
            x_k_min[k] = min(vertices[k][0], vertices[k + 1][0])
            x_k_max[k] = max(vertices[k][0], vertices[k + 1][0])
            y_k_min[k] = min(vertices[k][1], vertices[k + 1][1])
            y_k_max[k] = max(vertices[k][1], vertices[k + 1][1])

    # The smallest ymin and the biggest ymax
    y_min = min(y_k_min)
    y_max = max(y_k_max)

    # Declare the coordinates of every vertex point
    vertexPoint1 = vertices[0]
    vertexPoint2 = vertices[1]
    vertexPoint3 = vertices[2]

    # Colors (RGB - Channels) for every vertex
    v1Color = vcolors[0]
    v2Color = vcolors[1]
    v3Color = vcolors[2]

    # Calculate the pixels of every edge including the coordinates of 2 vertices
    edge1 = fillLine(vertexPoint1, vertexPoint2)
    edge2 = fillLine(vertexPoint2, vertexPoint3)
    edge3 = fillLine(vertexPoint3, vertexPoint1)

    # Check if there is horizontal line on top or bottom
    topHorizontalLine = 0
    bottomHorizontalLine = 0

    if y_k_min[0] == y_k_min[1] and y_k_min[1] == y_k_min[2]:
        bottomHorizontalLine = 1

    if y_k_max[0] == y_k_max[1] and y_k_max[1] == y_k_max[2]:
        topHorizontalLine = 1

    # Sort the pixels of every edge
    if edge1.ndim > 1 :
       e1Ind = np.argsort(edge1[:,0])
       edge1 = edge1[e1Ind]
    else :
      edge1 = np.array([edge1])

    if edge2.ndim > 1 :
       e2Ind = np.argsort(edge2[:,0])
       edge2 = edge2[e2Ind]
    else :
       edge2 = np.array([edge2])

    if edge3.ndim > 1 :
       e3Ind = np.argsort(edge3[:,0])
       edge3 = edge3[e3Ind]
    else :
      edge3 = np.array([edge3])

    edge1 = list(edge1)
    edge2 = list(edge2)
    edge3 = list(edge3)

    # Find the average colors based on the colors of the triangle's vertices
    meanColor = [0, 0, 0]

    meanColor[0] = (vcolors[0][0] + vcolors[1][0] + vcolors[2][0])/ 3
    meanColor[1] = (vcolors[0][1] + vcolors[1][1] + vcolors[2][1])/ 3
    meanColor[2] = (vcolors[0][2] + vcolors[1][2] + vcolors[2][2])/ 3

    # Fill the pixels of every edge with the mean color of the triangle
    for point in edge1:
            img[point[1]][point[0]] = meanColor
    for point in edge2:
            img[point[1]][point[0]] = meanColor
    for point in edge3:
            img[point[1]][point[0]] = meanColor

    # Find bottom and top Scanline
    bottomScanline =  min(edge1[0][0], edge2[0][0], edge3[0][0])
    topScanline    =  max(edge1[-1][0], edge2[-1][0], edge3[-1][0])

    # Scanline Algorithm
    for scanline in range(bottomScanline, topScanline + 1):
        activePoints = []
        for point in edge1:
            if point[0] == scanline:
                activePoints.append(point)
        for point in edge2:
            if point[0] == scanline:
                activePoints.append(point)
        for point in edge3:
            if point[0] == scanline:
                activePoints.append(point)

         # If only one point is found (it will be a vertex--already colored), move to the next iteration
        if len(activePoints) == 1:
           continue
        else:
            # Find the minimum and maximum column indices among the active points
            min_col_active = np.array(activePoints).min(axis=0)[1]
            max_col_active = np.array(activePoints).max(axis=0)[1]

            # Fill the pixels between the minimum and maximum column indices with the mean color
            activePoints  =np.array(activePoints)
            for i in range(min_col_active, max_col_active):
              if np.array([i, scanline]) in activePoints:
                    if not np.allclose(img[i][scanline], meanColor):
                       img[i][scanline] = meanColor
                    continue
              else:
                  img[i][scanline] = meanColor

    updated_img = img
    return updated_img

