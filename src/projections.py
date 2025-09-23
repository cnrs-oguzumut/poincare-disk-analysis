"""
Stereographic projection and metric transformation functions.

This module contains functions for:
- Stereographic projection from disk coordinates to metric tensors
- Centering transformations using gamma parameter
- Inverse projections for coordinate recovery
"""

import numpy as np
import math

def Cij_from_stereographic_projection_tr(x, y):
    """
    Apply stereographic projection with centering transformation.
    
    Parameters:
    -----------
    x, y : array_like
        Stereographic coordinates on the Poincaré disk
        
    Returns:
    --------
    c11, c22, c12 : array_like
        Centered metric tensor components
    c11t, c22t, c12t : array_like
        Original (non-centered) metric tensor components
    """
    # Define centering transformation matrix
    gamma = (4 / 3) ** (1 / 4)
    H = gamma * np.array([[math.sqrt(2 + math.sqrt(3))/2, math.sqrt(2 - math.sqrt(3))/2], 
                         [math.sqrt(2 - math.sqrt(3))/2., math.sqrt(2 + math.sqrt(3))/2]])
    H=np.eye(2)
    
    # Stereographic projection
    t = 2. / (1. - x ** 2. - y ** 2.)
    c11 = t * (1. + x) - 1.
    c22 = t * (1. - x) - 1.
    c12 = t * y
    
    # Apply centering transformation: ctr = H_t @ csq @ H
    H_inv = np.linalg.inv(H)
    H_t_inv = np.transpose(H_inv)
    H_t = np.transpose(H)
    
    csq = np.array([[c11, c12], [c12, c22]])
    ctr = np.zeros_like(csq)
    
    for i in range(len(csq)):
        for j in range(len(csq[0])):
            for k in range(len(H_t)):
                for l in range(len(H_t[0])):
                    ctr[i, j] += H_t[i, k] * csq[k, l] * H[l, j]
    
    return ctr[0,0], ctr[1,1], ctr[0,1], c11, c22, c12

def Cij_from_stereographic_projection(x, y):
    """
    Standard stereographic projection without centering.
    
    Parameters:
    -----------
    x, y : array_like
        Stereographic coordinates
        
    Returns:
    --------
    c11, c22, c12 : array_like
        Metric tensor components
    """
    t = 2. / (1. - x**2 - y**2)
    
    c11 = t * (1. + x) - 1.
    c22 = t * (1. - x) - 1.
    c12 = t * y
    
    return c11, c22, c12

def stereographic_projection_from_Cij_2D(c11, c22, c12):
    """
    Inverse stereographic projection from metric components to coordinates.
    
    Parameters:
    -----------
    c11, c22, c12 : array_like
        Metric tensor components
        
    Returns:
    --------
    x, y : array_like
        Stereographic coordinates
    """
    t = 2. / (2. + c11 + c22)
    
    x = t * (c11 - c22) / 2.
    y = t * c12
    
    return x, y

def Cij_from_orthogonal_projection(x, y):
    """
    Orthogonal projection alternative (for comparison).
    
    Parameters:
    -----------
    x, y : array_like
        Projection coordinates
        
    Returns:
    --------
    c11, c22, c12 : array_like
        Metric tensor components
    """
    c11 = x
    c22 = -x
    c12 = y
    
    return c11, c22, c12