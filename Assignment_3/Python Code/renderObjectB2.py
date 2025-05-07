# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 1st Assignment
# Deirmentzoglou Ioannis AEM : 10015

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plt2
import time

from calcNormalsB1 import calculate_normals
from GouraudPhongShadingB3B4 import shade_gouraud, shade_phong

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



def render_object(shader, focal, eye, lookatP, up, bg_color, M, N, H, W, verts, vert_colors, faces, 
                  ka, kd, ks, n, lpos, lint, lamb, lightOption):
    """
   Render a 3D object using either Gouraud or Phong shading.

   Parameters:
   - shader: Type of shading to use ('gouraud' or 'phong').
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

    return img

