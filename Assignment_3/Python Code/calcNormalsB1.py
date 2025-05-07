# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 3rd Assignment
# Deirmentzoglou Ioannis AEM : 10015

import numpy as np

def calculate_normals(verts, faces):
    """
    Calculate normal vectors for each vertex in a triangle.

    Parameters:
    - verts: List of vertex coordinates.
    - faces: List of faces, each defined by indices into the verts list.
    
    Returns:
    - normals: Array of normalized normal vectors for each vertex.
    """
    # Initialization of an array to store the normal vectors for each vertex
    normals = np.zeros(verts.shape)
    
    # Iteration through each face (triangle) in the list of faces
    for triangle in faces:
        # Getting the indices of the vertices that make up the triangle
        index = verts[triangle]
        
        # Extraction of the vertices A, B, and C of the triangle
        A = index[0]
        B = index[1]
        C = index[2]
        
        # Computation of the edge vectors AB and AC
        AB = B - A
        AC = C - A
        
        # Normalization of the edge vectors AB and AC
        norm_AB = np.linalg.norm(AB)
        norm_AC = np.linalg.norm(AC)
        
        if norm_AB != 0 and norm_AC != 0:
            nAB = AB / norm_AB
            nAC = AC / norm_AC
        
            # Computation of the normal vector using the cross product of AB and AC
            outer_AB_AC = np.cross(nAB, nAC)
        
            # Addition of the computed normal vector to the normals of the vertices that make up the triangle
            normals[triangle] += outer_AB_AC

    # Normalization of the normal vectors for each vertex
    for i in range(len(normals)):
        norm = np.linalg.norm(normals[i])
        if norm != 0:
            normals[i] /= norm
    
    return normals
