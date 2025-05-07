# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 1st Assignment
# Deirmentzoglou Ioannis AEM : 10015

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as plt2
import time

from vectors_interpolation import vector_interp
from bresenham import bresenhamAlgorithm , fillLine
from Flat_Shading import f_shading 
from Gouraud_Shading import g_shading

def render_img( faces, vertices, vcolors, depth, shading):
    """""
    :canvas:   (M,N,3) array created below to represent the canvas
    :vertices: (K,2) array that contains the [x,y] coordinates of each vertix. K is the number of vertices
    :faces:    (L,3) array that contains 3 indexes to the vertices array that define the vertices of the triangles. L is the number of triangles
    :vcolors:  (K,3) array that contains the [r,g,b] values of each vertix. K is the number of vertices
    :depth:    (K,1) array that contains the depth values for each vertix. K is the number of vertices
    :shading: is equal either to "f" or "g" and specifies the shading method
    :returns img: the updated img after rendering
    """""

    # Set up canvas with white background and resolution 512x512
    M = 512
    N = 512
    img = [[[1.0 for i in range(3)] for j in range(M)] for k in range(N)]

    triangleColors = []
    triangleDepths = []

    # Iterate over each triangle in the mesh in order to find the mean deapth and sort the depths of triangles
    for triangle in faces:
        # Determine the index of the current triangle in the faces list
        index = faces.index(triangle)
        # Convert the vertices of the triangle to 2D coordinates
        faces[index] = [vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]]

         # Calculate the triangle depth as the mean of the depths of the vertices
        newDepth = (depth[triangle[0]] + depth[triangle[1]] + depth[triangle[2]]) / 3
        triangleColors.append([vcolors[triangle[0]], vcolors[triangle[1]], vcolors[triangle[2]]])
        triangleDepths.append(newDepth)

    zipped = zip(triangleDepths, faces, triangleColors)
    # Get the indexes after sorting the triangle depth array with descending order
    triangleDepths, faces, triangleColors = zip(*sorted(zipped, key=lambda x: -x[0]))

    # Paint the triangles with flat or gouraud method based on shading
    for triangle in faces:
      if shading == 'f'  :
         img = f_shading(img, triangle, triangleColors[faces.index(triangle)])
      elif shading == 'g' :
         img = g_shading(img, triangle, triangleColors[faces.index(triangle)])


    return img

