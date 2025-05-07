# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 1st Assignment
# Deirmentzoglou Ioannis AEM : 10015

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plt2
import time
import math
from PIL import Image

from calcNormalsB1 import calculate_normals
from IlluminationCalculationA import light
from bresenham import bresenhamAlgorithm , fillLine
from renderObjectB2 import shade_gouraud, shade_phong


def create_tuples(*items):
    return tuple(item for item in items)


def world2view(pts: np.ndarray, R: np.ndarray, c0: np.ndarray) :
    # Implements a world-to-view transform, i.e. transforms the specified
    # points to the coordinate frame of a camera. The camera coordinate frame
    # is specified rotation (w.r.t. the world frame) and its point of reference
    # (w.r.t. to the world frame).

    pts = np.array(pts)   
    pts, R, c0 = create_tuples(pts, R, c0)

    if pts.ndim == 1:
        pts = (pts - c0)
    else:
        pts = np.array([pts[i, :] - c0 for i in range(pts.shape[0])])
    
    # Apply rotation to pts
    rotated_pts = np.dot(R, pts.T)

    # Translate points by c0
    translated_pts = rotated_pts 

    # Return transformed points
    return translated_pts.T

def lookat(eye: np.ndarray, up: np.ndarray, target: np.ndarray) :
    # Calculate the camera's view matrix (i.e., its coordinate frame transformation specified
    # by a rotation matrix R, and a translation vector t).
    # :return a tuple containing the rotation matrix R (3 x 3) and a translation vector
    # t (1 x 3)
    
    # Calculate the unit vectors of the camera 
    vectorCK = np.array(target) - np.array(eye)

    # Calculate the unit vector in the direction of vectorCK
    c_z = vectorCK / np.linalg.norm(vectorCK)
    tCoord = np.array(up - np.dot(up.T, c_z) * c_z)
    
    c_y = tCoord / np.linalg.norm(tCoord)
    
    # Compute the cross product of c_y and c_z to get the third axis
    c_x = np.cross(c_y, c_z)
    c_x, c_y, c_z = create_tuples(c_x, c_y, c_z)    
    
    # Construct the rotation matrix
    R = np.vstack((c_x, c_y, c_z))
    # Set the translation vector
    t = eye 

    return R, t

def perspective_project(pts: np.ndarray, focal: float, R: np.ndarray, t: np.ndarray):
   # Project the specified 3d points pts on the image plane, according to a pinhole
   # perspective projection model
   
   # Transform coordinates from WCS to CCS
   pts = world2view(pts, R, t)

   # Project from 3D to 2D and calculate the depth
   depth = pts[:, 2]
   x_perspective = (focal / depth) * pts[:, 0]
   y_perspective = (focal / depth) * pts[:, 1]
   pts_2D_perspective = np.vstack((x_perspective, y_perspective)) #    pts_2D_perspective = np.vstack((y_perspective, x_perspective))

   
   return pts_2D_perspective.T, depth

def rasterize(pts_2d: np.ndarray, plane_w: int, plane_h: int, res_w: int, res_h: int) -> np.ndarray:
    # Rasterize the incoming 2d points from the camera plane to image pixel coordinates
   
    # Convert pts_2d to floating-point array to allow addition of float
    pts_2d = pts_2d.astype(float)
    pts_2dRast = np.zeros((len(pts_2d), 2))

    # Calculate scaling factors
    scale_w = res_w / plane_w
    scale_h = res_h / plane_h
    
    # Apply the scaling factors to the coordinates
    for i in range(len(pts_2d)):
        pts_2dRast[i, 0] = np.around((pts_2d[i, 0] + plane_w / 2) * scale_w)
        pts_2dRast[i, 1] = np.around((-pts_2d[i, 1] + plane_h / 2) * scale_h)


    pts_2dRast = np.round(pts_2dRast).astype(int)

    return pts_2dRast

    
### Texture shading Function 
def texture_shading(img, vertices, triangle_uvs, texture_map):
    """
    :img: Array MxNx3 that contains the RGB values for the entire image.
    :vertices: Integer array 3x2 that contains the coordinates of the triangle's vertices in each row.
    :triangle_uvs: Array 3x2 that contains the UV coordinates for each vertex of the triangle.
    :texture_map: Array MxMx3 that contains the RGB values of the texture.
    :returns updated_img: MxNx3 array containing the RGB values for all parts of the triangle and the already existing image.
    """
    p1, p2, p3 = vertices

    vertices = vertices.tolist()

    # Initialize variables x_k_min, x_k_max, y_k_min, y_k_max for the filling algorithm
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

    # Calculate the pixels of every edge including the coordinates of 2 vertices
    edge1 = fillLine(vertexPoint1, vertexPoint2)
    edge2 = fillLine(vertexPoint2, vertexPoint3)
    edge3 = fillLine(vertexPoint3, vertexPoint1)

    # Check if there is a horizontal line on top or bottom
    topHorizontalLine = 0
    bottomHorizontalLine = 0

    if y_k_min[0] == y_k_min[1] and y_k_min[1] == y_k_min[2]:
        bottomHorizontalLine = 1

    if y_k_max[0] == y_k_max[1] and y_k_max[1] == y_k_max[2]:
        topHorizontalLine = 1

    # Sort the pixels of every edge
    if edge1.ndim > 1:
        e1Ind = np.argsort(edge1[:, 0])
        edge1 = edge1[e1Ind]
    else:
        edge1 = np.array([edge1])

    if edge2.ndim > 1:
        e2Ind = np.argsort(edge2[:, 0])
        edge2 = edge2[e2Ind]
    else:
        edge2 = np.array([edge2])

    if edge3.ndim > 1:
        e3Ind = np.argsort(edge3[:, 0])
        edge3 = edge3[e3Ind]
    else:
        edge3 = np.array([edge3])

    edge1 = list(edge1)
    edge2 = list(edge2)
    edge3 = list(edge3)

    for point in edge1:
        lambda1, lambda2, lambda3 = barycentric_coordinates(point, p1, p2, p3)
        uv = interpolate_uv(triangle_uvs[0], triangle_uvs[1], triangle_uvs[2], lambda1, lambda2, lambda3)
        color = bilerp(uv, texture_map)
        img[point[1]][point[0]] = color

    for point in edge2:
        lambda1, lambda2, lambda3 = barycentric_coordinates(point, p1, p2, p3)
        uv = interpolate_uv(triangle_uvs[0], triangle_uvs[1], triangle_uvs[2], lambda1, lambda2, lambda3)
        color = bilerp(uv, texture_map)
        img[point[1]][point[0]] = color

    for point in edge3:
        lambda1, lambda2, lambda3 = barycentric_coordinates(point, p1, p2, p3)
        uv = interpolate_uv(triangle_uvs[0], triangle_uvs[1], triangle_uvs[2], lambda1, lambda2, lambda3)
        color = bilerp(uv, texture_map)
        img[point[1]][point[0]] = color

    # Find bottom and top Scanline
    bottomScanline = min(edge1[0][0], edge2[0][0], edge3[0][0])
    topScanline = max(edge1[-1][0], edge2[-1][0], edge3[-1][0])

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
            activePoints = np.array(activePoints)
            for i in range(min_col_active, max_col_active):
                if np.array([i, scanline]) in activePoints:
                    point = np.array([scanline, i])
                    lambda1, lambda2, lambda3 = barycentric_coordinates(point, p1, p2, p3)
                    uv = interpolate_uv(triangle_uvs[0], triangle_uvs[1], triangle_uvs[2], lambda1, lambda2, lambda3)
                    color = bilerp(uv, texture_map)
                    img[point[1]][point[0]] = color
                    continue
                else:
                    point = np.array([scanline, i])
                    lambda1, lambda2, lambda3 = barycentric_coordinates(point, p1, p2, p3)
                    uv = interpolate_uv(triangle_uvs[0], triangle_uvs[1], triangle_uvs[2], lambda1, lambda2, lambda3)
                    color = bilerp(uv, texture_map)
                    img[point[1]][point[0]] = color

    updated_img = img
    return updated_img


def shade_texture(vertsp, vertsn, bcoords, cam_pos, ka, kd, ks, n, lpos, lint, lamb, X, lightOption, triangle_uvs, texture_map):
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

    # Apply Gouraud shading to the imag
    Y = texture_shading(X, vertsp[:,:2] , triangle_uvs, texture_map)

    return Y



def bilerp(uv, texture_map):
    """
    Perform bilinear interpolation to retrieve the color corresponding to 
    the given 2D UV coordinates from the texture map.

    Parameters:
    uv (list or tuple): 2D coordinates (u, v) in the range [0, 1].
    texture_map (numpy array): The M x M x 3 texture image.

    Returns:
    The interpolated color at the given UV coordinates.
    """
    u, v = uv

    if math.isnan(u) or math.isnan(v): 
        u, v = 0, 0 
        
    height, width, _ = texture_map.shape

    # Convert u, v from range [0, 1] to range [0, width-1] and [0, height-1]
    x = u * (width - 1)
    y = v * (height - 1)

    # Find the coordinates of the four surrounding pixels
    x0 = int(np.floor(x))         
    x1 = min(x0 + 1, width - 1)   
    y0 = int(np.floor(y))         
    y1 = min(y0 + 1, height - 1)  

    # Calculate the distances from the point to the four surrounding pixels
    dx = x - x0  # Distance from the point to the left pixel in x direction
    dy = y - y0  # Distance from the point to the top pixel in y direction

    # Get the colors of the four surrounding pixels
    texelColor00 = texture_map[y0, x0]  # Color at (x0, y0)
    texelColor10 = texture_map[y0, x1]  # Color at (x1, y0)
    texelColor01 = texture_map[y1, x0]  # Color at (x0, y1)
    texelColor11 = texture_map[y1, x1]  # Color at (x1, y1)

    # Perform bilinear interpolation
    # Combine the colors weighted by the distances
    texelColor = (
        texelColor00 * (1 - dx) * (1 - dy) +  # Top-left weight
        texelColor10 * dx * (1 - dy) +        # Top-right weight
        texelColor01 * (1 - dx) * dy +        # Bottom-left weight
        texelColor11 * dx * dy                # Bottom-right weight
    )

    return texelColor


def interpolate_uv(uv1, uv2, uv3, w1, w2, w3):
    """
    Interpolate UV coordinates for a point inside a triangle using barycentric weights.

    Parameters:
    uv1, uv2, uv3: UV coordinates of the triangle vertices.
    w1, w2, w3: Barycentric weights of the point inside the triangle.

    Returns:
    Interpolated UV coordinates.
    """
    return uv1 * w1 + uv2 * w2 + uv3 * w3

def barycentric_coordinates(p, p1, p2, p3):
    # Vector from P1 to P2 and P3
    p1p2 = p2 - p1
    p1p3 = p3 - p1
    p1p = p - p1
    p2p = p2 - p
    p3p = p3 - p

    # Cross product magnitudes
    area_triangle = np.linalg.norm(np.cross(p1p2, p1p3))
    lambda1 = np.linalg.norm(np.cross(p2p, p3p)) / area_triangle
    lambda2 = np.linalg.norm(np.cross(p1p, p3p)) / area_triangle
    lambda3 = 1 - lambda1 - lambda2

    return lambda1, lambda2, lambda3

def render_object_texture(shader, focal, eye, lookatP, up, bg_color, M, N, H, W, verts, vert_colors, faces, 
                  ka, kd, ks, n, lpos, lint, lamb, lightOption, uvs, face_uv_indices, texture_map):
    """
   Render a 3D object using either Gouraud or Phong or Texture shading.

   Parameters:
   - shader: Type of shading to use ('gouraud', 'phong', or 'texture').
   - focal: Focal length for the perspective projection.
   - eye: Position of the camera/eye.
   - lookatP: Point where the camera is looking at.
   - up: Up vector for the camera orientation.
   - bg_color: Background color of the canvas.
   - M: Width of the canvas.
   - N: Height of the canvas.
   - H: Height of the viewing window.
   - W: Width of the viewing window.
   - verts: List of vertex coordinates.
   - vert_colors: List of vertex colors.
   - faces: List of faces.
   - ka: Ambient reflection coefficient.
   - kd: Diffuse reflection coefficient.
   - ks: Specular reflection coefficient.
   - n: Phong coefficient.
   - lpos: Position of the light source.
   - lint: Intensity of the light source.
   - lamb: Ambient light intensity.
   - lightOption: Lighting options.
   - uvs: Texture coordinates for vertices.
   - face_uv_indices: Indices of texture coordinates for each face.
   - texture_map: Texture map for the object.

   Returns:
   - img: The rendered image as a 2D array.
   """
    # Calculation of normal vectors for each face of the object
    normalVectors = calculate_normals(verts, faces)
    
    # Calculation of rotation matrix R and translation vector t based on the camera's eye position, up vector, and lookat point
    target = lookatP
    R, t = lookat(eye, up, target)
    
    # Projection of the 3D vertices onto a 2D plane using perspective projection
    verts2d, depth = perspective_project(verts, focal, R, t)
    
    # Transformation of the projected vertices to screen space coordinates
    verts2d = rasterize(verts2d, W, H, N, M)
    
    # Conversion of the vertex coordinates to integers
    verts2d = np.array(verts2d).astype(int)
    
    # Calculation of the average depth of each face and sorting of the faces by depth (back to front)
    new_depth = np.array(np.mean(depth[faces], axis=1))
    facesSorted = list(np.flip(np.argsort(new_depth)))
    facesUVSorted = list(np.flip(np.argsort(new_depth)))
    
    # Creation of a canvas with the background color, with dimensions MxN
    canva = [[[bg_color[i] for i in range(3)] for j in range(M)] for k in range(N)]
    X = canva 

    # Application of Gouraud shading if the shader type is 'gouraud'
    if shader == 'gouraud':
        for triangle in facesSorted:
            verts_idx = faces[triangle]
            triangle_vertsc = np.array(vert_colors[verts_idx])
            triangle_vertsp = np.array(verts2d[verts_idx])
            bcoords = np.mean(verts[verts_idx], axis=0)
            
            # Application of Gouraud shading to the triangle
            img = shade_gouraud(triangle_vertsp, normalVectors[verts_idx], triangle_vertsc, bcoords, eye, ka, kd, ks, n, lpos, lint, lamb, X, lightOption)

    # Application of Phong shading if the shader type is 'phong'
    if shader == 'phong':
        for triangle in facesSorted:
            verts_idx = faces[triangle]
            triangle_vertsc = np.array(vert_colors[verts_idx])
            triangle_vertsp = np.array(verts2d[verts_idx])
            bcoords = np.mean(verts[verts_idx], axis=0)
            
            # Application of Phong shading to the triangle
            img = shade_phong(triangle_vertsp, normalVectors[verts_idx], triangle_vertsc, bcoords, eye, ka, kd, ks, n, lpos, lint, lamb, X, lightOption)

    if shader == 'texture' : 
        # Perform shading and texturing
        for triangle in facesSorted:
            verts_idx = faces[triangle]
            uv_idx = face_uv_indices[triangle]
            triangle_vertsp = np.array(verts2d[verts_idx])
            triangle_uvs = np.array(uvs[uv_idx])
            bcoords = np.mean(verts[verts_idx], axis=0)
            
            # Apply texture shading to the triangle
            img = shade_texture(triangle_vertsp, normalVectors[verts_idx], bcoords, eye, ka, kd, ks, n, lpos, lint, lamb, X, lightOption, triangle_uvs, texture_map)
        


    
    return img



start = time.time()
data = np.load(r'C:\Users\User\Desktop\Computer_Graphics\Assignment_3\hw3\h3.npy', allow_pickle=True).tolist()

# Transpose the vertices, vertex colors, and face indices arrays
verts = np.array(data['verts']).T  # 3D vertices
vertex_colors = np.array(data['vertex_colors']).T  # 3D vertex colors
face_indices = np.array(data['face_indices']).T  # 3D face indices


# Camera parameters
cam_eye = np.array(data['cam_eye']) 
cam_up = np.array(data['cam_up']) 
cam_lookat = np.array(data['cam_lookat']) 

# Material properties
ka = np.array(data['ka'])  # Ambient reflection coefficient
kd = np.array(data['kd'])  # Diffuse reflection coefficient
ks = np.array(data['ks'])  # Specular reflection coefficient

# Specular exponent
n = np.array(data['n'])  # Specular exponent

# Light positions and intensities
lpos = np.array(data['light_positions'])  # Light positions
lint = np.array(data['light_intensities'])  # Light intensities

# Ambient light intensity
Ia = np.array(data['Ia'])  # Ambient light intensity

# Image and camera dimensions
M = np.array(data['M'])  # Image width
N = np.array(data['N'])  # Image height
W = np.array(data['W'])  # Camera width
H = np.array(data['H'])  # Camera height

# Background color
bg_color = np.array(data['bg_color'])  # Background color (white)

focal = np.array(data['focal'])  # Focal length


# Load texture map
image_path = r'C:\Users\User\Desktop\Computer_Graphics\Assignment_3\hw3\cat_diff.png'
image = Image.open(image_path)
# Convert the image to a NumPy array
texture_map = np.array(image)
texture_map = texture_map.astype(float)/255
plt.figure(0)
plt.imshow(texture_map)
plt.title('Cat diff image')
plt.show()
# uv coordinates 
uvs = np.array(data['uvs']).T
face_uv_indices = np.array(data['face_uv_indices']).T

end = time.time()
print('Data management execution time : ', end - start)

# Light option setting
lightOption = 'Combined'

# All 3 lighting sources used 
light_positions = lpos 
light_intensities = lint

start = time.time()
img = render_object_texture('texture', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption, uvs, face_uv_indices, texture_map)
end = time.time()
plt.figure(1)
plt.imshow(img)
plt.title('Texture Shading')
plt.show()
print('Figure 1 rendering execution time : ', end - start)
# #################################



