# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 1st Assignment
# Deirmentzoglou Ioannis AEM : 10015

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as plt2
import time

from bresenham import bresenhamAlgorithm , fillLine
from Flat_Shading import f_shading 
from Gouraud_Shading import g_shading
from Render import render_img
from allFunctions import Transform, world2view, create_tuples, lookat, perspective_project, rasterize, render_object

# Load the data from hw1.py
start = time.time()
data = np.load(r'C:\Users\t.deirmentzoglou\Desktop\Computer_Graphics\Assignment_2\hw2.npy', allow_pickle=True)

v_pos = data[()]['v_pos']
v_clr = data[()]['v_clr']

t_pos_idxData = data[()]['t_pos_idx']
t_pos_idx = t_pos_idxData

# The coordinates of the center of the camera
eyeData = data[()]['eye']
eye = eyeData.reshape(-1)

# The coordinates of the up vector of the camera
upData = data[()]['up']
up = upData.reshape(-1)

# The coordinates of the target point
targetData = data[()]['target']
target = targetData.reshape(-1)

# The focal length of the lens
focal = data[()]['focal']


# Image and camera height and width
plane_w = data[()]['plane_w']
plane_h = data[()]['plane_h']
res_w = data[()]['res_w']
res_h = data[()]['res_h']

# The angle of the rotation
theta_0 = data[()]['theta_0']
# Axis along which the rotation is performed
rot_axis_0 = data[()]['rot_axis_0']


# The displacement vectors
t_0 = data[()]['t_0']
t_1 = data[()]['t_1']


# Iterate over the variables and print their names, types, and shapes
for var_name in ['v_pos', 'v_clr', 't_pos_idx', 'eye', 'up', 'target', 'focal', 
                 'plane_w', 'plane_h', 'res_w', 'res_h', 'theta_0', 'rot_axis_0', 
                 't_0', 't_1']:
    var = data[()][var_name]
    if isinstance(var, np.ndarray):
        print(f"Name: {var_name},\nShape: {var.shape}\n")
    else:
        print(f"Name: {var_name}\n")
        
end = time.time()
print('Data management execution time : ', end - start)




# Step 0 -Initial Position
start = time.time()
img = render_object(v_pos, v_clr, t_pos_idx,  plane_h, plane_w, res_h, res_w, focal, eye, up, target)
end = time.time()

# Plot and Save Image
plt.figure(1)
plt.imshow(img)
plt.title('Step 0--Initial Position')
plt.show()
plt2.imsave('Initial_Position.jpg',np.array(img))
print('Figure 1 rendering execution time : ', end - start)


# Step A -- Image Rotation by theta_0 and the rotation axis rot_axis_0
transformObject = Transform()
transformObject.rotate(theta_0, rot_axis_0)

start = time.time()
v_pos = transformObject.transform_pts(v_pos)
img = render_object(v_pos, v_clr, t_pos_idx,  plane_h, plane_w, res_h, res_w, focal, eye, up, target)
end = time.time()

# Plot and Save Image
plt.figure(2)
plt.imshow(img)
plt.title('Step A--Image rotated by theta_0 ')
plt.show()
plt2.imsave('Step_A_Image.jpg',np.array(img))
print('Figure 2 rendering execution time : ', end - start)



# Step B -- Displace image by t_0
start = time.time()
transformObject.translate(t_0)
v_pos = transformObject.transform_pts(v_pos.T)
img = render_object(v_pos, v_clr, t_pos_idx,  plane_h, plane_w, res_h, res_w, focal, eye, up, target)
end = time.time()

# Plot and Save Image
plt.figure(3)
plt.imshow(img)
plt.title('Step B--Image displaced by t_0')
plt.show()
plt2.imsave('Step_B_Image.jpg',np.array(img))
print('Figure 3 rendering execution time : ', end - start)



# Step C -- Displace image by t_1
start = time.time()
transformObject.translate(t_1)
v_pos = transformObject.transform_pts(v_pos.T)
img = render_object(v_pos, v_clr, t_pos_idx,  plane_h, plane_w, res_h, res_w, focal, eye, up, target)
end = time.time()

# Plot and Save Image
plt.figure(4)
plt.imshow(img)
plt.title('Step C--Image displaced by t_1')
plt.show()
plt2.imsave('Step_C_Image.jpg',np.array(img))
print('Figure 4 rendering execution time : ', end - start)