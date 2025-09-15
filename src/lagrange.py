"""
Lagrange reduction algorithms for metric tensors.

This module implements Lagrange reduction to map metric tensors
to the fundamental domain defined by:
- C₁₂ ≥ 0
- C₂₂ ≥ C₁₁  
- 2C₁₂ ≤ C₁₁
"""

import numpy as np

def lagrange_reduction(c11, c22, c12, verbose=True):
    """
    Perform Lagrange reduction on metric tensor components.
    
    Parameters:
    -----------
    c11, c22, c12 : array_like
        Metric tensor components
    verbose : bool
        Print progress information
        
    Returns:
    --------
    c11_reduced, c22_reduced, c12_reduced : array_like
        Reduced metric tensor components
    iterations : int
        Number of reduction iterations performed
    """
    # Initialize boolean array and iteration count
    need_reduction = np.where(
        np.logical_or(
            np.logical_or(c12 < 0., c22 < c11), 
            2.*c12 > c11
        ), True, False
    )
    reduction_iter = 0
    
    while np.any(need_reduction):
        # Lagrange reduction steps
        c12 = np.where(c12 < 0., -1.*c12, c12)
        c11, c22 = np.where(c22 < c11, c22, c11), np.where(c22 < c11, c11, c22)
        c22, c12 = np.where(2.*c12 > c11, c22 + c11 - 2.*c12, c22), \
                   np.where(2.*c12 > c11, c12 - c11, c12)
        
        # Update iteration count
        reduction_iter += 1
        
        # Update boolean array
        need_reduction = np.where(
            np.logical_or(
                np.logical_or(c12 < 0., c22 < c11), 
                2.*c12 > c11
            ), True, False
        )
        
        # Verbose output
        if verbose and (np.mod(reduction_iter, 100) == 0):
            print(f"Iteration {reduction_iter}: C₁₁ range [{np.nanmin(c11):.6f}, {np.nanmax(c11):.6f}]")
    
    return c11, c22, c12, reduction_iter

def meet_Cij_conditions(c):
    """
    Check if metric components satisfy Lagrange reduction conditions.
    
    Parameters:
    -----------
    c : array_like
        Metric components [c11, c22, c12]
        
    Returns:
    --------
    bool
        True if conditions are met
    """
    epsilon = 1.e-16
    
    if c[2] < -epsilon:  # c12 < 0
        return False
    if c[1] < c[0] - epsilon:  # c22 < c11
        return False
    if 2.*c[2] > c[0] + epsilon:  # 2*c12 > c11
        return False
        
    return True

# Lagrange reduction matrices (for advanced usage)
lag_m1 = np.array([[1., 0.], [0., -1.]])
lag_m2 = np.array([[0., 1.], [1., 0.]])
lag_m3 = np.array([[1., -1.], [0., 1.]])  # horizontal shear
lag_m4 = np.array([[1., 0], [-1., 1.]])   # vertical shear
lag_m3_n = np.array([[1., 1.], [0., 1.]]) # negative horizontal shear
lag_m4_n = np.array([[1., 0], [1., 1.]])  # negative vertical shear