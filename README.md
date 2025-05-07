# 💻 Computer Graphics

---

### Assignments for "Computer Graphics" Coursework (2023)
Assignment for the "Computer Graphics" Course  
Faculty of Engineering, AUTh  
School of Electrical and Computer Engineering  
Electronics and Computers Department

📚 *Course:* Computer Graphics
🏛️ *Faculty:* AUTh - School of Electrical and Computer Engineering  
📅 *Semester:* 8th Semester, 2023–2024


---

# 📌 Assignment 1: Triangle Filling
## 📝 Assignment Overview

This assignment focuses on implementing triangle rasterization and color shading techniques using linear interpolation. The objective is to simulate the coloring of 3D objects projected in 2D, using both **Flat** and **Gouraud shading** techniques.

### 🧩 Main Goals:
- Develop a linear interpolation function.
- Implement a triangle filling algorithm.
- Apply both **Flat** and **Gouraud shading** to render colored triangles.
- Integrate the above techniques in an object rendering function.

---

### 📚 Contents

### 1. `vector_interp` Function
Calculates a vector’s value at a given point between two known values using linear interpolation.

### 2. Triangle Filling Functions
- `f_shading`: Colors the triangle with the average color of its vertices (Flat shading).
- `g_shading`: Colors the triangle with interpolated colors across the surface (Gouraud shading).

### 3. `render_img` Function
Draws multiple triangles that form a 3D object projection, handling:
- Vertex data
- Color per vertex
- Depth buffer
- Shading type selection

---


# 📌 Assignment 2: Transformations and Projections
## 📝 Assignment Overview

This repository contains the implementation for the second assignment of the Computer Graphics course at AUTH. The goal of this assignment is to build a minimal 3D rendering pipeline that transforms, projects, and renders 3D objects onto a 2D screen using fundamental linear algebra operations and camera modeling techniques.

---

## 📚 Theoretical Overview

This assignment focuses on several core concepts in 3D computer graphics:

### ✅ 1. Affine Transformations
Affine transformations, including **rotation** and **translation**, are essential to manipulate 3D models in a virtual space. These are represented using homogeneous coordinates and 4×4 transformation matrices. The `Transform` class encapsulates this functionality, maintaining a cumulative transformation matrix.

### ✅ 2. Camera Coordinate System
The **camera view** is modeled through a coordinate system transformation:
- `world2view()` shifts and rotates the world space into the view (camera) space using a rotation matrix and a translation vector.
- `lookat()` generates a view matrix that orients the camera using three key vectors: `eye`, `target`, and `up`.

### ✅ 3. Perspective Projection
Using a **pinhole camera model**, 3D points are projected to a 2D plane through the `perspective_project()` function. The projection simulates how humans perceive depth and perspective.

### ✅ 4. Rasterization
The `rasterize()` function converts camera-plane coordinates to discrete image pixels, accounting for screen resolution and aspect ratio. This step is necessary to visualize the projected 2D image.

### ✅ 5. Rendering Pipeline
Finally, `render_object()` acts as the orchestrator that:
1. Transforms world coordinates to camera space.
2. Projects points onto the 2D image plane.
3. Converts them to pixel coordinates.
4. Colors the visible vertices using vertex colors.

---

## 💻 Implemented Functions

- `Transform` class
  - `.rotate(theta, u)` — Rotate around axis `u` by angle `theta`
  - `.translate(t)` — Translate by vector `t`
  - `.transform_pts(pts)` — Apply current transformation matrix to points
- `world2view(pts, R, c0)` — World-to-camera coordinate transformation
- `lookat(eye, up, target)` — Computes camera orientation (rotation and translation)
- `perspective_project(pts, focal, R, t)` — Projects 3D points to 2D
- `rasterize(pts_2d, plane_w, plane_h, res_w, res_h)` — Maps projected points to image pixels
- `render_object(...)` — Complete rendering pipeline from 3D object to 2D image

---

## 📁 Repository Structure

---
# 🎥 Assignment 3: Viewing
## 📝 Assignment Overview

This repository contains the implementation for the third assignment of the Computer Graphics course at AUTH. The goal of this task is to extend the rendering pipeline with **illumination modeling**, **shading**, and a complete virtual camera setup. We implement both **Gouraud** and **Phong** shading to simulate realistic lighting effects on 3D surfaces.

---

## 📚 Theoretical Overview

This assignment builds on the affine transformations and projections from Assignments 1 and 2, introducing **lighting**, **shading**, and a complete rendering pipeline.

### ✅ 1. Illumination Model (Phong Lighting)
The `light()` function implements the **Phong lighting model**, which simulates realistic lighting using three components:
- **Ambient lighting** (`ka`): Uniform light scattered in the scene.
- **Diffuse reflection** (`kd`): Light scattered from rough surfaces depending on the angle between light direction and normal.
- **Specular reflection** (`ks`): Mirror-like reflection based on the viewer's direction and the surface normal.

The function supports **multiple point light sources**, each contributing additively to the color at a surface point.

### ✅ 2. Surface Normals
Using the `calculate_normals()` function, we compute **per-vertex normals** from triangle faces. These normals are critical for shading:
- Normals are computed using the **right-hand rule**.
- Each vertex normal is an average of the normals of adjacent faces.

### ✅ 3. Complete Rendering Pipeline
The `render_object()` function ties everything together:
1. Transforms object geometry to camera space.
2. Projects the geometry to a 2D image plane.
3. Computes per-pixel color using **Gouraud** or **Phong** shading.
4. Applies **z-buffering** to ensure correct visibility ordering.
5. Generates the final 2D image as `img`.

### ✅ 4. Shading Models
- **Gouraud shading**: Computes lighting at vertices and interpolates colors across the triangle.
- **Phong shading**: Interpolates normals across the triangle and computes lighting per pixel for smoother, more realistic highlights.

---

## 💻 Implemented Functions

- `light(point, normal, vcolor, cam_pos, ka, kd, ks, n, lpos, lint)` — Computes light intensity at a point using Phong lighting.
- `calculate_normals(verts, faces)` — Calculates vertex normals from triangle geometry.
- `render_object(...)` — Renders a 3D object with selected shading technique and full lighting model.

---

## 📁 Repository Structure
