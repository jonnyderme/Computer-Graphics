# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 1st Assignment
# Deirmentzoglou Ioannis AEM : 10015

# All functions 

import numpy as np

from vectors_interpolation import vector_interp
from bresenham import bresenhamAlgorithm , fillLine
from Flat_Shading import f_shading 
from Gouraud_Shading import g_shading
from Render import render_img
#from demoChangeCS import lookat, perspective_project, world2view, create_tuples



## A. 
class Transform:
    # Interface for performing affine transformations
    def __init__(self):
        # Initialize a Transform object with identity matrix
        self.mat = np.eye(4)

    def rotate(self, theta: float, u: np.ndarray) -> None:
        # Rotate the transformation matrix
        # Calculate rotation matrix using Rodrigues' rotation formula
        ux, uy, uz = u / np.linalg.norm(u)
        
        b1 = np.array([(1 - np.cos(theta)) * ux ** 2 + np.cos(theta),
                       (1 - np.cos(theta)) * ux * uy - np.sin(theta) * uz,
                       (1 - np.cos(theta)) * ux * uz + np.sin(theta) * uy])
        
        b2 = np.array([(1 - np.cos(theta)) * ux * uy + np.sin(theta) * uz,
                       (1 - np.cos(theta)) * uy ** 2 + np.cos(theta),
                       (1 - np.cos(theta)) * uy * uz - np.sin(theta) * ux])
        
        b3 = np.array([(1 - np.cos(theta)) * ux * uz - np.sin(theta) * uy, 
                       (1 - np.cos(theta)) * uy * uz - np.sin(theta) * ux, 
                       (1 - np.cos(theta)) * uz ** 2 + np.cos(theta)])
        
     
        R = np.eye(4) 
        R[:3, :3] = np.vstack((b1, b2, b3))
        
        # Update transformation matrix
        self.mat = np.dot(R, self.mat)
        

    def translate(self, t: np.ndarray) :#-> None:
        # Translate the transformation matrix.
        # Create translation matrix
        T = np.eye(4)
        T[:3, 3] = t
        # Update transformation matrix
        self.mat = np.dot(T, self.mat)
       

    def transform_pts(self, pts: np.ndarray) -> np.ndarray:
        # Transform the specified points
        # according to our current matrix
        
        pts = pts.T
        if pts.ndim > 1:
            # Add a column of ones to pts for homogeneous coordinates
            pts_homogenius = np.hstack((pts, np.ones((pts.shape[0], 1))))
        else:
            pts_homogenius = np.append(pts, 1)
        
        # Apply transformation matrix
        transformed_pts = np.dot(self.mat, pts_homogenius.T).T

        if transformed_pts.ndim > 1:
            transformed_pts = np.delete(transformed_pts, 3, 1)
        else:
            transformed_pts = np.array(transformed_pts[:3])
        
        return transformed_pts



    
## C.       
def create_tuples(*items):
    return tuple(item for item in items)

def world2view(pts: np.ndarray, R: np.ndarray, c0: np.ndarray) :
    # Implements a world-to-view transform, i.e. transforms the specified
    # points to the coordinate frame of a camera. The camera coordinate frame
    # is specified rotation (w.r.t. the world frame) and its point of reference
    # (w.r.t. to the world frame).
    
    pts = np.array(pts)
    # Ensure pts is in the correct shape
    if pts.shape[0] == 3:
        pts = pts.T
    
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





## D. 
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
    R = np.vstack((c_x, c_y, c_z)).T
    # Set the translation vector
    t = eye 
    
    return R, t





## E. 
def perspective_project(pts: np.ndarray, focal: float, R: np.ndarray, t: np.ndarray):
   # Project the specified 3d points pts on the image plane, according to a pinhole
   # perspective projection model
 
   # Transform coordinates from WCS to CCS
   pts = world2view(pts, R, t)
   # Project from 3D to 2D and calculate the depth
   depth = pts[:, 2]
   x_perspective = (focal / depth) * pts[:, 1]
   y_perspective = (focal / depth) * pts[:, 0]
   pts_2D_perspective = np.vstack((y_perspective, x_perspective))
   
   return pts_2D_perspective.T, depth





## F.
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
        pts_2dRast[i, 0] = np.around((-pts_2d[i, 0] + plane_w / 2) * scale_w)
        pts_2dRast[i, 1] = np.around((pts_2d[i, 1] + plane_h / 2) * scale_h)


    pts_2dRast = np.round(pts_2dRast).astype(int)

    return pts_2dRast





## G. 
def render_object(v_pos, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target):
    # Render the specified object from the specified camera
    
    # Calculate rotation matrix R and translation vector
    R, t = lookat(eye, up, target)
    
    # Calculate 2d projections and depth
    pts_2d, depth = perspective_project(v_pos, focal, R, t)
    depth = depth.tolist()

    # Rasterize 2d perspective projectected points 
    pts_2d = rasterize(pts_2d, plane_w, plane_h, res_w, res_h)
    pts_2d = np.array(pts_2d).astype(int)
    pts_2d = pts_2d.tolist()
    
    
    t_pos_idx = t_pos_idx.tolist()
    v_clr = v_clr.tolist()
    
    faces = t_pos_idx
    vertices = pts_2d
    vcolors = v_clr
    
    # Initialize image and set white background
    img = 255 * np.ones([res_h,res_w,3], dtype=np.uint8)
    
    # Render image with gouraud shading
    img = render_img(faces, vertices, vcolors, depth, "g")
   
    return img
