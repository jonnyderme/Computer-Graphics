# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 1st Assignment
# Deirmentzoglou Ioannis AEM : 10015

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as plt2
import time


from renderObjectB2 import render_object  

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

# Phong coefficient 
n = np.array(data['n'])  

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

end = time.time()
print('Data management execution time : ', end - start)



###### Gouraud Shading Experiments #######



#### 1. 3 Sources 
### Gouraud Shading Combined Lighting ###

# Light option setting
lightOption = 'Combined'

# All 3 lighting sources used 
light_positions = lpos 
light_intensities = lint

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(1)
plt.imshow(img)
plt.title('Gouraud Shading Combined Lighting - 3 Sources')
plt.show()
plt2.imsave('Gouraud_Shading_Combined_Lighting_3_Sources.png',np.array(img))
print('Figure 1 rendering execution time : ', end - start)
#################################


### Gouraud Shading Ambient Lighting ###

# Light option setting
lightOption1 = 'Ambient'

# All 3 lighting sources used 
light_positions = lpos 
light_intensities = lint

img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors,
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption1)
end = time.time()
plt.figure(2)
plt.imshow(img)
plt.title('Gouraud Shading Ambient Lighting - 3 Sources')
plt.show()
plt2.imsave('Gouraud_Shading_Ambient_Lighting_3_Sources.png',np.array(img))
print('Figure 2 rendering execution time : ', end - start)
#################################


### Gouraud Shading Diffusion Lighting ###

# Light option setting
lightOption = 'Diffusion'

# All 3 lighting sources used 
light_positions = lpos 
light_intensities = lint

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(3)
plt.imshow(img)
plt.title('Gouraud Shading Diffusion Lighting - 3 Sources')
plt.show()
plt2.imsave('Gouraud_Shading_Diffusion_Lighting_3_Sources.png',np.array(img))
print('Figure 3 rendering execution time : ', end - start)
#################################


### Gouraud Shading Specular Lighting ###

# Light option setting
lightOption = 'Specular'

# All 3 lighting sources used 
light_positions = lpos 
light_intensities = lint

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(4)
plt.imshow(img)
plt.title('Gouraud Shading Specular Lighting - 3 Sources')
plt.show()
plt2.imsave('Gouraud_Shading_Specular_Lighting_3_Sources.png',np.array(img))
print('Figure 4 rendering execution time : ', end - start)


#### 2. 1 Source : 1st  ####
### Gouraud Shading Combined Lighting ###

# Light option setting
lightOption = 'Combined'

# 1st lighting source used 
light_positions = lpos[0,:] 
light_intensities = lint[0,:]

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(5)
plt.imshow(img)
plt.title('Gouraud Shading Combined Lighting - 1st Source')
plt.show()
plt2.imsave('Gouraud_Shading_Combined_Lighting_1st_Source.png',np.array(img))
print('Figure 5 rendering execution time : ', end - start)
#################################


### Gouraud Shading Ambient Lighting ###

# Light option setting
lightOption = 'Ambient'

# 1st lighting source used 
light_positions = lpos[0,:] 
light_intensities = lint[0,:]

img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors,
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(6)
plt.imshow(img)
plt.title('Gouraud Shading Ambient Lighting - 1st Source')
plt.show()
plt2.imsave('Gouraud_Shading_Ambient_Lighting_1st_Source.png',np.array(img))
print('Figure 6 rendering execution time : ', end - start)
#################################


### Gouraud Shading Diffusion Lighting ###

# Light option setting
lightOption = 'Diffusion'

# 1st lighting source used 
light_positions = lpos[0,:] 
light_intensities = lint[0,:]

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(7)
plt.imshow(img)
plt.title('Gouraud Shading Diffusion Lighting - 1st Source')
plt.show()
plt2.imsave('Gouraud_Shading_Diffusion_Lighting_1st_Source.png',np.array(img))
print('Figure 7 rendering execution time : ', end - start)
#################################


### Gouraud Shading Specular Lighting ###

# Light option setting
lightOption = 'Specular'

# 1st lighting source used 
light_positions = lpos[0,:] 
light_intensities = lint[0,:]

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(8)
plt.imshow(img)
plt.title('Gouraud Shading Specular Lighting - 1st Source')
plt.show()
plt2.imsave('Gouraud_Shading_Specular_Lighting_1st_Source.png',np.array(img))
print('Figure 8 rendering execution time : ', end - start)


#### 2. 1 Source : 2nd  ####
### Gouraud Shading Combined Lighting ###

# Light option setting
lightOption = 'Combined'

# 2nd lighting source used 
light_positions = lpos[1,:] 
light_intensities = lint[1,:]

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(9)
plt.imshow(img)
plt.title('Gouraud Shading Combined Lighting - 2nd Source')
plt.show()
plt2.imsave('Gouraud_Shading_Combined_Lighting_2nd_Source.png',np.array(img))
print('Figure 9 rendering execution time : ', end - start)
#################################


### Gouraud Shading Ambient Lighting ###

# Light option setting
lightOption = 'Ambient'

# 2nd lighting source used 
light_positions = lpos[1,:] 
light_intensities = lint[1,:]

img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors,
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(10)
plt.imshow(img)
plt.title('Gouraud Shading Ambient Lighting - 2nd Source')
plt.show()
plt2.imsave('Gouraud_Shading_Ambient_Lighting_2nd_Source.png',np.array(img))
print('Figure 10 rendering execution time : ', end - start)
#################################


### Gouraud Shading Diffusion Lighting ###

# Light option setting
lightOption = 'Diffusion'

# 2nd lighting source used 
light_positions = lpos[1,:] 
light_intensities = lint[1,:]

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(11)
plt.imshow(img)
plt.title('Gouraud Shading Diffusion Lighting - 2nd Source')
plt.show()
plt2.imsave('Gouraud_Shading_Diffusion_Lighting_2nd_Source.png',np.array(img))
print('Figure 11 rendering execution time : ', end - start)
#################################


### Gouraud Shading Specular Lighting ###

# Light option setting
lightOption = 'Specular'

# 2nd lighting source used 
light_positions = lpos[1,:] 
light_intensities = lint[1,:]

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(12)
plt.imshow(img)
plt.title('Gouraud Shading Specular Lighting - 2nd Source')
plt.show()
plt2.imsave('Gouraud_Shading_Specular_Lighting_2nd_Source.png',np.array(img))
print('Figure 12 rendering execution time : ', end - start)



#### 3. 1 Source : 3rd  ####
### Gouraud Shading Combined Lighting ###

# Light option setting
lightOption = 'Combined'

# 3rd lighting source used 
light_positions = lpos[2,:] 
light_intensities = lint[2,:]

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(13)
plt.imshow(img)
plt.title('Gouraud Shading Combined Lighting - 3rd Source')
plt.show()
plt2.imsave('Gouraud_Shading_Combined_Lighting_3rd_Source.png',np.array(img))
print('Figure 13 rendering execution time : ', end - start)
#################################


### Gouraud Shading Ambient Lighting ###

# Light option setting
lightOption = 'Ambient'

# 3rd lighting source used 
light_positions = lpos[2,:] 
light_intensities = lint[2,:]

img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors,
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(14)
plt.imshow(img)
plt.title('Gouraud Shading Ambient Lighting - 3rd Source')
plt.show()
plt2.imsave('Gouraud_Shading_Ambient_Lighting_3rd_Source.png',np.array(img))
print('Figure 14 rendering execution time : ', end - start)
#################################


### Gouraud Shading Diffusion Lighting ###

# Light option setting
lightOption = 'Diffusion'

# 3rd lighting source used 
light_positions = lpos[2,:] 
light_intensities = lint[2,:]

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(15)
plt.imshow(img)
plt.title('Gouraud Shading Diffusion Lighting - 3rd Source')
plt.show()
plt2.imsave('Gouraud_Shading_Diffusion_Lighting_3rd_Source.png',np.array(img))
print('Figure 15 rendering execution time : ', end - start)
#################################


### Gouraud Shading Specular Lighting ###

# Light option setting
lightOption = 'Specular'

# 3rd lighting source used 
light_positions = lpos[2,:] 
light_intensities = lint[2,:]

start = time.time()
img = render_object('gouraud', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(16)
plt.imshow(img)
plt.title('Gouraud Shading Specular Lighting - 3rd Source')
plt.show()
plt2.imsave('Gouraud_Shading_Specular_Lighting_3rd_Source.png',np.array(img))
print('Figure 16 rendering execution time : ', end - start)

#####################################################################################
###******************************************************************************####

##### Phong Shading Experiments #######


#### 1. 3 Sources 
### Phong Shading Combined Lighting ###

# Light option setting
lightOption = 'Combined'

# All 3 lighting sources used 
light_positions = lpos 
light_intensities = lint

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(17)
plt.imshow(img)
plt.title('Phong Shading Combined Lighting - 3 Sources')
plt.show()
plt2.imsave('Phong_Shading_Combined_Lighting_3_Sources.png',np.array(img))
print('Figure 17 rendering execution time : ', end - start)
#################################


### Phong Shading Ambient Lighting ###

# Light option setting
lightOption1 = 'Ambient'

# All 3 lighting sources used 
light_positions = lpos 
light_intensities = lint

img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors,
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption1)
end = time.time()
plt.figure(18)
plt.imshow(img)
plt.title('Phong Shading Ambient Lighting - 3 Sources')
plt.show()
plt2.imsave('Phong_Shading_Ambient_Lighting_3_Sources.png',np.array(img))
print('Figure 18 rendering execution time : ', end - start)
#################################


### Phong Shading Diffusion Lighting ###

# Light option setting
lightOption = 'Diffusion'

# All 3 lighting sources used 
light_positions = lpos 
light_intensities = lint

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(19)
plt.imshow(img)
plt.title('Phong Shading Diffusion Lighting - 3 Sources')
plt.show()
plt2.imsave('Phong_Shading_Diffusion_Lighting_3_Sources.png',np.array(img))
print('Figure 19 rendering execution time : ', end - start)
#################################


### Phong Shading Specular Lighting ###

# Light option setting
lightOption = 'Specular'

# All 3 lighting sources used 
light_positions = lpos 
light_intensities = lint

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(20)
plt.imshow(img)
plt.title('Phong Shading Specular Lighting - 3 Sources')
plt.show()
plt2.imsave('Phong_Shading_Specular_Lighting_3_Sources.png',np.array(img))
print('Figure 20 rendering execution time : ', end - start)


#### 2. 1 Source : 1st  ####
### Phong Shading Combined Lighting ###

# Light option setting
lightOption = 'Combined'

# 1st lighting source used 
light_positions = lpos[0,:] 
light_intensities = lint[0,:]

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(21)
plt.imshow(img)
plt.title('Phong Shading Combined Lighting - 1st Source')
plt.show()
plt2.imsave('Phong_Shading_Combined_Lighting_1st_Source.png',np.array(img))
print('Figure 21 rendering execution time : ', end - start)
#################################


### Phong Shading Ambient Lighting ###

# Light option setting
lightOption = 'Ambient'

# 1st lighting source used 
light_positions = lpos[0,:] 
light_intensities = lint[0,:]

img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors,
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(22)
plt.imshow(img)
plt.title('Phong Shading Ambient Lighting - 1st Source')
plt.show()
plt2.imsave('Phong_Shading_Ambient_Lighting_1st_Source.png',np.array(img))
print('Figure 22 rendering execution time : ', end - start)
#################################


### Gouraud Shading Diffusion Lighting ###

# Light option setting
lightOption = 'Diffusion'

# 1st lighting source used 
light_positions = lpos[0,:] 
light_intensities = lint[0,:]

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(23)
plt.imshow(img)
plt.title('Phong Shading Diffusion Lighting - 1st Source')
plt.show()
plt2.imsave('Phong_Shading_Diffusion_Lighting_1st_Source.png',np.array(img))
print('Figure 23 rendering execution time : ', end - start)
#################################


### Phong Shading Specular Lighting ###

# Light option setting
lightOption = 'Specular'

# 1st lighting source used 
light_positions = lpos[0,:] 
light_intensities = lint[0,:]

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(24)
plt.imshow(img)
plt.title('Phong Shading Specular Lighting - 1st Source')
plt.show()
plt2.imsave('Phong_Shading_Specular_Lighting_1st_Source.png',np.array(img))
print('Figure 24 rendering execution time : ', end - start)


#### 2. 1 Source : 2nd  ####
### Phong Shading Combined Lighting ###

# Light option setting
lightOption = 'Combined'

# 2nd lighting source used 
light_positions = lpos[1,:] 
light_intensities = lint[1,:]

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(25)
plt.imshow(img)
plt.title('Phong Shading Combined Lighting - 2nd Source')
plt.show()
plt2.imsave('Phong_Shading_Combined_Lighting_2nd_Source.png',np.array(img))
print('Figure 25 rendering execution time : ', end - start)
#################################


### Phong Shading Ambient Lighting ###

# Light option setting
lightOption = 'Ambient'

# 2nd lighting source used 
light_positions = lpos[1,:] 
light_intensities = lint[1,:]

img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors,
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(26)
plt.imshow(img)
plt.title('Phong Shading Ambient Lighting - 2nd Source')
plt.show()
plt2.imsave('Phong_Shading_Ambient_Lighting_2nd_Source.png',np.array(img))
print('Figure 26 rendering execution time : ', end - start)
#################################


### Phong Shading Diffusion Lighting ###

# Light option setting
lightOption = 'Diffusion'

# 2nd lighting source used 
light_positions = lpos[1,:] 
light_intensities = lint[1,:]

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(27)
plt.imshow(img)
plt.title('Phong Shading Diffusion Lighting - 2nd Source')
plt.show()
plt2.imsave('Phong_Shading_Diffusion_Lighting_2nd_Source.png',np.array(img))
print('Figure 27 rendering execution time : ', end - start)
#################################


### Phong Shading Specular Lighting ###

# Light option setting
lightOption = 'Specular'

# 2nd lighting source used 
light_positions = lpos[1,:] 
light_intensities = lint[1,:]

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(28)
plt.imshow(img)
plt.title('Phong Shading Specular Lighting - 2nd Source')
plt.show()
plt2.imsave('Phong_Shading_Specular_Lighting_2nd_Source.png',np.array(img))
print('Figure 28 rendering execution time : ', end - start)



#### 3. 1 Source : 3rd  ####
### Phong Shading Combined Lighting ###

# Light option setting
lightOption = 'Combined'

# 3rd lighting source used 
light_positions = lpos[2,:] 
light_intensities = lint[2,:]

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(29)
plt.imshow(img)
plt.title('Phong Shading Combined Lighting - 3rd Source')
plt.show()
plt2.imsave('Phong_Shading_Combined_Lighting_3rd_Source.png',np.array(img))
print('Figure 29 rendering execution time : ', end - start)
#################################


### Phong Shading Ambient Lighting ###

# Light option setting
lightOption = 'Ambient'

# 3rd lighting source used 
light_positions = lpos[2,:] 
light_intensities = lint[2,:]

img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors,
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(30)
plt.imshow(img)
plt.title('Phong Shading Ambient Lighting - 3rd Source')
plt.show()
plt2.imsave('Phong_Shading_Ambient_Lighting_3rd_Source.png',np.array(img))
print('Figure 30 rendering execution time : ', end - start)
#################################


### Phong Shading Diffusion Lighting ###

# Light option setting
lightOption = 'Diffusion'

# 3rd lighting source used 
light_positions = lpos[2,:] 
light_intensities = lint[2,:]

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(31)
plt.imshow(img)
plt.title('Phong Shading Diffusion Lighting - 3rd Source')
plt.show()
plt2.imsave('Phong_Shading_Diffusion_Lighting_3rd_Source.png',np.array(img))
print('Figure 31 rendering execution time : ', end - start)
#################################


### Gouraud Shading Specular Lighting ###

# Light option setting
lightOption = 'Specular'

# 3rd lighting source used 
light_positions = lpos[2,:] 
light_intensities = lint[2,:]

start = time.time()
img = render_object('phong', focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, 
                    face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lightOption)
end = time.time()
plt.figure(32)
plt.imshow(img)
plt.title('Phong Shading Specular Lighting - 3rd Source')
plt.show()
plt2.imsave('Phong_Shading_Specular_Lighting_3rd_Source.png',np.array(img))
print('Figure 32 rendering execution time : ', end - start)