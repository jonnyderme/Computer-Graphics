# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 3rd Assignment
# Deirmentzoglou Ioannis AEM : 10015

import numpy as np

from calcNormalsB1 import calculate_normals
from IlluminationCalculationA import light
from bresenham import bresenhamAlgorithm , fillLine

### 1st Assignment functions used 
### vector_interp, bresenhamAlgorithm , fillLine
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

### G_shading function from the first Assignment
def g_shading(img, vertices, vcolors):
    """""
    :img: the already existing image with its triangles
    :vertices: Integer array 3x2 that contains the coordinates of triangle's vertices in each line.
    :vcolors: Array 3x3 that contains the color of triangle's vertex in each line with RGB values[0,1]
    :updated_img: Array MxNx3 that contains the RGB values for all the parts of the
    :triangle and the already existing triangles
    :returns updated_img: MxNx3 array containing the RGB values for all the parts of the triangle and the already existing triangles
    """""
    vertices = vertices.tolist()
    vcolors = vcolors.tolist()

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
    
    for point in edge1:
            # Check if the current point is a vertex of the triangle
            vertBool = False
            for vert in vertices:
               if np.array_equal(point,vert) :
                  vertBool = False
               else :
                   vertBool = True

            if  vertBool :
                # If gradient > 1
                if abs(vertices[0][0] - vertices[1][0]) > abs(vertices[0][1] - vertices[1][1]):
                    img[point[1]][point[0]] = vector_interp(vertices[0], vertices[1], v1Color, v2Color, point[0], dim=1)
                else:
                    img[point[1]][point[0]] = vector_interp(vertices[0], vertices[1], v1Color, v2Color, point[1], dim=2)
            else:
                  verticesList = list(vertices)
                  index = verticesList.index(list(point))
                  img[point[1]][point[0]] = vcolors[index]


    for point in edge2:
           vertBool = False
           for vert in vertices:
               if np.array_equal(point,vert) :
                  vertBool = False
               else :
                   vertBool = True

           if vertBool :
                # If gradient > 1
                if abs(vertices[1][0] - vertices[2][0]) > abs(vertices[1][1] - vertices[2][1]):
                    img[point[1]][point[0]] = vector_interp(vertices[1], vertices[2], v2Color, v3Color, point[0], dim = 1)
                else:
                    img[point[1]][point[0]] = vector_interp(vertices[1], vertices[2], v2Color, v3Color, point[1], dim = 2)
           else:
                verticesList = list(vertices)
                index = verticesList.index(list(point))
                img[point[1]][point[0]] = vcolors[index]


    for point in edge3:
           vertBool = False
           for vert in vertices :
               if np.array_equal(point,vert) :
                  vertBool = False
               else :
                   vertBool = True

           if vertBool :
                if abs(vertices[2][0] - vertices[0][0]) > abs(vertices[2][1] - vertices[0][1]):
                    # If gradient > 1
                    img[point[1]][point[0]] = vector_interp(vertices[2], vertices[0], v3Color, v1Color, point[0], dim = 1)
                else:
                    img[point[1]][point[0]] = vector_interp(vertices[2], vertices[0], v3Color, v1Color, point[1], dim = 2)
            #c)
           else:
                verticesList = list(vertices)
                index = verticesList.index(list(point))
                img[point[1]][point[0]] = vcolors[index]



    # Find bottom and top Scanline
    bottomScanline =  min(edge1[0][0], edge2[0][0], edge3[0][0])
    topScanline    =  max(edge1[-1][0], edge2[-1][0], edge3[-1][0])

    # List of active points and update it for each new scanline
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

        # If there is only one active point, skip to the next iteration
        if len(activePoints) == 1:
            continue
        else:
            # From the list of active points, get the minimum value by column
            min_col_active = np.array(activePoints).min(axis=0)[1]

            # From the list of active points, get the maximum value by column
            max_col_active = np.array(activePoints).max(axis=0)[1]

            # Draw the active points between the boundary columns
            activePoints  =np.array(activePoints)
            for i in range(min_col_active, max_col_active):
              if np.array([i, scanline]) in activePoints :
                    img[i][scanline] = vector_interp([min_col_active,scanline], [max_col_active,scanline] ,
                                                           img[min_col_active][scanline], img[max_col_active][scanline], i, dim=1)
                    continue
              else:
                  img[i][scanline] = vector_interp([min_col_active,scanline], [max_col_active,scanline] ,
                                                           img[min_col_active][scanline], img[max_col_active][scanline], i, dim=1)


    updated_img = img
    return updated_img


##** Gouraud Shading Function **##
def shade_gouraud(vertsp, vertsn, vertsc, bcoords, cam_pos, ka, kd, ks, n, lpos, lint, lamb, X, lightOption):
    """
    Apply Gouraud shading to a triangle based on its vertices and normal vectors.

    Parameters:
    vertsp (array): Array of vertex positions (3x3).
    vertsn (array): Array of vertex normals (3x3).
    vertsc (array): Array of vertex colors (3x3).
    bcoords (array): Barycentric coordinates.
    cam_pos (array): Camera position.
    ka (array): Ambient reflection coefficient.
    kd (array): Diffuse reflection coefficient.
    ks (array): Specular reflection coefficient.
    n (float): Specular exponent.
    lpos (array): Light positions.
    lint (array): Light intensities.
    lamb (float): Ambient light intensity.
    X (array): The image being shaded.
    lightOption (str): Lighting option ('Ambient', 'Diffuse', 'Specular', 'Combined').

    Returns:
    Y (array): The shaded image.
    """

    # Store the vertex normals
    vertsnHor = vertsn

    # Initialize an array to hold the colors of the vertices
    verts_colors = np.zeros((3, 3))

    # Compute the color at each vertex using the lighting model
    verts_colors[0,:] = light(bcoords, vertsnHor[0,:], vertsc[0,:], cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
    verts_colors[1,:] = light(bcoords, vertsnHor[1,:], vertsc[1,:], cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
    verts_colors[2,:] = light(bcoords, vertsnHor[2,:], vertsc[2,:], cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
    

    # Convert the vertex colors to a list
    vcolors = verts_colors.tolist()

    # Apply Gouraud shading to the image
    Y = g_shading(X, vertsp[:,:2], verts_colors)

    return Y




def interpolate_normal(x1, x2, x, N1, N2):
    """
    Interpolates the normal vector at point x based on the normal vectors at points x1 and x2.

    Parameters:
    x1 (float): Coordinate of the first interpolation point.
    x2 (float): Coordinate of the second interpolation point.
    x (float): Coordinate of the point where interpolation is desired.
    N1 (np.ndarray): Normal vector at the first interpolation point.
    N2 (np.ndarray): Normal vector at the second interpolation point.

    Returns:
    np.ndarray: Normalized interpolated normal vector at point x.
    """
    # Handle case where x1 and x2 are very close to each other to avoid division by zero
    if abs(x1 - x2) < 1e-3:
        return N1
    
    # Use linear interpolation formula for normal vectors: N = lambda * N1 + (1 - lambda) * N2
    lambdaP = (x2 - x) / (x2 - x1)
    N = lambdaP * N1 + (1 - lambdaP) * N2
    
    # Normalize the interpolated normal vector
    normalized_N = N / np.linalg.norm(N)
    
    return normalized_N

def interpolate_color(x1, x2, x, color1, color2):
    """
    Interpolates between two colors c1 and c2 based on the position x within the range [x1, x2].
    
    Parameters:
    x1 (float): The start point of the interpolation range.
    x2 (float): The end point of the interpolation range.
    x (float): The specific point within the range [x1, x2] at which to interpolate.
    c1 (list): The color at the start point x1, represented as a list of RGB values.
    c2 (list): The color at the end point x2, represented as a list of RGB values.
    
    Returns:
    np.ndarray: The interpolated color as an array of RGB values.
    """
    
    # Calculate the interpolation factor lambda, where lambda = (x2 - x) / (x2 - x1)
    lambdaFactor = (x2 - x) / (x2 - x1)
    
    # Interpolate between the two colors using the calculated lambda
    interpColor = lambdaFactor * np.array(color1) + (1 - lambdaFactor) * np.array(color2)
    
    return interpColor


###*** Phong Shading Function ***## 

def shade_phong(vertsp, vertsn, vertsc, bcoords, cam_pos, ka, kd, ks, n, lpos, lint, lamb, X, lightOption) : 
    
    vertsnHor = vertsn
    
    verts_colors = np.zeros((3, 3))
    verts_colors[0,:] = light(bcoords, vertsnHor[0,:], vertsc[0,:], cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
    verts_colors[1,:] = light(bcoords, vertsnHor[1,:], vertsc[1,:], cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
    verts_colors[2,:] = light(bcoords, vertsnHor[2,:], vertsc[2,:], cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
    
    img = X 
    vertices = vertsp[:,:2]
    vertices = vertices.tolist()
    vcolors = verts_colors 
    
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

    for point in edge1:
             # Check if the current point is a vertex of the triangle
            vertBool = False
            for vert in vertices:
               if np.array_equal(point,vert) :
                  vertBool = False
               else :
                   vertBool = True

            if  vertBool :
                # If gradient > 1
                if abs(vertices[0][0] - vertices[1][0]) > abs(vertices[0][1] - vertices[1][1]):
                    interpN = interpolate_normal(vertices[0][0], vertices[1][0], point[0], vertsn[0], vertsn[1])
                    vertColor = interpolate_color(vertices[0][0], vertices[1][0], point[0], vertsc[0].tolist(), vertsc[1].tolist())
                    img[point[1]][point[0]] = light(bcoords, interpN, vertColor, cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
              
                else:
                    interpN = interpolate_normal(vertices[0][1], vertices[1][1], point[1], vertsn[0], vertsn[1])
                    vertColor = interpolate_color(vertices[0][1], vertices[1][1], point[1], vertsc[0].tolist(), vertsc[1].tolist())
                    img[point[1]][point[0]] = light(bcoords, interpN, vertColor, cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
                    
            else:
                  verticesList = list(vertices)
                  index = verticesList.index(list(point))
                  img[point[1]][point[0]] = light(bcoords, vertsn[index, :], vertsc[index, :], cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)


    for point in edge2:
           vertBool = False
           for vert in vertices:
               if np.array_equal(point,vert) :
                  vertBool = False
               else :
                   vertBool = True

           if vertBool :
                # If gradient > 1
                if abs(vertices[1][0] - vertices[2][0]) > abs(vertices[1][1] - vertices[2][1]):
                    interpN = interpolate_normal(vertices[1][0], vertices[2][0], point[0], vertsn[1], vertsn[2])
                    vertColor = interpolate_color(vertices[1][0], vertices[2][0], point[0], vertsc[1].tolist(), vertsc[2].tolist())
                    img[point[1]][point[0]] = light(bcoords, interpN, vertColor, cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
              
                else:
                    interpN = interpolate_normal(vertices[1][1], vertices[2][1], point[1], vertsn[1], vertsn[2])
                    vertColor = interpolate_color(vertices[1][1], vertices[2][1], point[1], vertsc[1].tolist(), vertsc[2].tolist())
                    img[point[1]][point[0]] = light(bcoords, interpN, vertColor, cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
                    
           else:
                  verticesList = list(vertices)
                  index = verticesList.index(list(point))
                  img[point[1]][point[0]] = light(bcoords, vertsn[index, :], vertsc[index, :], cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)

    for point in edge3:
           vertBool = False
           for vert in vertices :
               if np.array_equal(point,vert) :
                  vertBool = False
               else :
                   vertBool = True

           if vertBool :
                # If gradient > 1
                if abs(vertices[2][0] - vertices[0][0]) > abs(vertices[2][1] - vertices[0][1]):
                    interpN = interpolate_normal(vertices[2][0], vertices[0][0], point[0], vertsn[2], vertsn[0])
                    vertColor = interpolate_color(vertices[2][0], vertices[0][0], point[0], vertsc[2].tolist(), vertsc[0].tolist())
                    img[point[1]][point[0]] = light(bcoords, interpN, vertColor, cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
              
                else:
                    interpN = interpolate_normal(vertices[2][1], vertices[0][1], point[1], vertsn[2], vertsn[0])
                    vertColor = interpolate_color(vertices[2][1], vertices[0][1], point[1], vertsc[2].tolist(), vertsc[0].tolist())
                    img[point[1]][point[0]] = light(bcoords, interpN, vertColor, cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)
                    
           else:
                  verticesList = list(vertices)
                  index = verticesList.index(list(point))
                  img[point[1]][point[0]] = light(bcoords, vertsn[index, :], vertsc[index, :], cam_pos, ka, kd, ks, n, lpos, lint, lamb, lightOption)




    # Find bottom and top Scanline
    bottomScanline =  min(edge1[0][0], edge2[0][0], edge3[0][0])
    topScanline    =  max(edge1[-1][0], edge2[-1][0], edge3[-1][0])

    # List of active points and update it for each new scanline
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

        # If there is only one active point, skip to the next iteration
        if len(activePoints) == 1:
            continue
        else:
            # From the list of active points, get the minimum value by column
            min_col_active = np.array(activePoints).min(axis=0)[1]

            # From the list of active points, get the maximum value by column
            max_col_active = np.array(activePoints).max(axis=0)[1]

            # Draw the active points between the boundary columns
            activePoints  =np.array(activePoints)
            for i in range(min_col_active, max_col_active):
              if np.array([i, scanline]) in activePoints :
                    img[i][scanline] = interpolate_color(min_col_active, max_col_active, i, X[min_col_active][scanline], X[max_col_active][scanline])
                    continue
              else:
                  img[i][scanline] = interpolate_color(min_col_active, max_col_active, i, X[min_col_active][scanline], X[max_col_active][scanline])


    Y = img
    return Y 