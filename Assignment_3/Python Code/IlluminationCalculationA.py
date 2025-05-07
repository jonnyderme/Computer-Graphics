# Aristotle University of Thessaloniki (AUTH)
# Electrical and Computer Engineering (ECE)
# Computer Graphics 1st Assignment
# Deirmentzoglou Ioannis AEM : 10015

import numpy as np
import sys



def light(point, normal, vcolor, cam_pos, ka, kd, ks, n, lpos, lint, I_a, lightOption):
    """
    Calculate the illumination of a point on a surface with Phong-type material.

    Parameters:
    point (np.ndarray): 1×3 vector of surface point coordinates.
    normal (np.ndarray): 1×3 vector of the normal vector coordinates at the point.
    vcolor (np.ndarray): 1×3 vector with the color components of the point.
    cam_pos (np.ndarray): 1×3 vector of the observer's coordinates.
    ka (float): Coefficient of ambient light.
    kd (float): Diffuse reflection coefficient.
    ks (float): Specular reflection coefficient.
    n (int): Phong constant for specular reflection.
    lpos (np.ndarray): N×3 matrix of positions of point light sources.
    lint (np.ndarray): N×3 matrix of intensities of point light sources.
    I_a (np.ndarray): 1×3 vector of ambient light intensity.
    lightOption (str): Lighting option ('Combined', 'Ambient', 'Diffusion', 'Specular').

    Returns:
    np.ndarray: 1×3 vector of the trichromatic radiation intensity reflected from the point.
    """
    
    # Calculate ambient light intensity
    I_ambient = ka * np.array(I_a)
    
    # Initialize diffuse and specular intensity components
    I_diffuse = np.zeros(3)
    I_specular = np.zeros(3)
    
    # If there is only one light source
    if lpos.ndim == 1:
        SP = lpos - point  # Vector from surface point to light source
        L = SP / np.linalg.norm(SP)  # Normalized light direction vector
        L = np.reshape(L, (3, ))
        
        cosa = np.dot(normal, L)
        cosa = max(cosa, 0)  # Ensure non-negative cosine

        # Calculate diffuse reflection
        I_diff = lint * kd * cosa
        I_diffuse = vcolor * I_diff

        # Calculate specular reflection
        CP = cam_pos - point  # Vector from surface point to camera
        V = CP / np.linalg.norm(CP)  # Normalized view direction vector
        V = np.reshape(V, (3, ))
        
        R = 2 * normal * cosa - L  # Reflection vector
        cosb_a = np.dot(R, V)
        cosb_a = max(cosb_a, 0)  # Ensure non-negative cosine
        
        I_spec = lint * ks * (cosb_a ** n)
        I_specular = vcolor * I_spec
        
    # If there are multiple light sources
    else:
        for i in range(lpos.shape[0]):
            SP = lpos[i] - point  # Vector from surface point to light source
            L = SP / np.linalg.norm(SP)  # Normalized light direction vector
            L = np.reshape(L, (3, ))
            
            cosa = np.dot(normal, L)
            cosa = max(cosa, 0)  # Ensure non-negative cosine

            # Calculate diffuse reflection
            I_diff = lint[i] * kd * cosa
            I_diffuse += vcolor * I_diff

            # Calculate specular reflection
            CP = cam_pos - point  # Vector from surface point to camera
            V = CP / np.linalg.norm(CP)  # Normalized view direction vector
            V = np.reshape(V, (3, ))
            
            R = 2 * normal * cosa - L  # Reflection vector
            cosb_a = np.dot(R, V)
            cosb_a = max(cosb_a, 0)  # Ensure non-negative cosine
            
            I_spec = lint[i] * ks * (cosb_a ** n)
            I_specular += vcolor * I_spec
    
    # Combine all intensities based on the selected lighting option
    if lightOption == 'Combined':   
        I_total = I_ambient + I_diffuse + I_specular
    elif lightOption == 'Ambient':
        I_total = I_ambient
    elif lightOption == 'Diffusion':
        I_total = I_diffuse
    elif lightOption == 'Specular':
        I_total = I_specular
    else:
        print("Incorrect light option given. You can only use: 'Combined', 'Ambient', 'Diffusion', or 'Specular'")
        return None
    
    # Ensure the intensity values are clamped between 0 and 1
    I_total = np.clip(I_total, 0, 1)
    
    return I_total
