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
from Render import render_img


# Load the data from hw1.py
start = time.time()
data = np.load(r'C:\Users\User\Desktop\Computer_Graphics\Assignment_1\hw1.npy', allow_pickle=True)
vcolors = data[()]['vcolors'].tolist()
faces = data[()]['faces'].tolist()
depth = data[()]['depth'].tolist()
vertices = data[()]['vertices'].astype(int).tolist()
end = time.time()
print('Data management execution time : ', end - start)

# Render
start = time.time()
img = render_img(faces, vertices, vcolors, depth,'g')
end = time.time()
print('Rendering execution time : ', end - start)

plt.imshow(img, interpolation='nearest')
plt.title('Gouraud Shading Shading')
plt.show()
plt2.imsave('Shade_Gouraud_Image.png',np.array(img))


