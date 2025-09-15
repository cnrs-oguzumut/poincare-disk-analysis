"""
Energy density calculations for different lattice types.

This module computes strain energy density using interatomic potentials
for square and triangular lattices.
"""

import numpy as np

def interatomic_phi0_from_Cij(c11, c22, c12, lattice):
    """
    Compute strain energy density using interatomic potential.
    
    Parameters:
    -----------
    c11, c22, c12 : array_like
        Metric tensor components
    lattice : str
        Lattice type ('square' or 'triangular')
        
    Returns:
    --------
    phi0 : array_like
        Strain energy density
    """
    # Potential parameters
    r1 = 1.
    r2 = 1.425
    b1 = 8.
    b2 = 8.
    a = 1.
    c1 = 2. * a
    
    # Lattice-specific parameters
    if lattice == 'triangular':
        c2 = 0.
    elif lattice == 'square':
        c2 = c1
    else:
        raise ValueError("Lattice must be 'triangular' or 'square'")
    
    # Calculation parameters
    scl = 1.0661
    cut = 2.5
    
    phi0 = np.zeros(np.shape(c11))
    
    # Sum over lattice neighbors
    for s in range(-3, 3):
        for l in range(-3, 3):
            if (s != 0) or (l != 0):
                # Distance calculation
                r = scl * np.sqrt((s**2) * c11 + 2.*s*l*c12 + (l**2) * c22)
                
                # Potential calculation
                tmp = np.where(
                    r > cut, 
                    0., 
                    a/(r)**12. - c1*np.exp(-b1*(r-r1)**2.) - c2*np.exp(-b2*(r-r2)**2.)
                )
                
                phi0 += tmp
    
    phi0 = 0.5 * phi0
    
    return phi0

def convert_data_SymLog(data, const):
    """
    Convert data to symmetric logarithmic scale.
    
    Applies log scaling for both positive and negative values
    with linearization near zero.
    
    Reference: J B W Webber - A bi-symmetric log transformation 
    for wide-range data (2013) Measurement Science and Technology 24
    
    Parameters:
    -----------
    data : array_like
        Input data
    const : float
        Constant defining the linearized 'near zero' region
        
    Returns:
    --------
    array_like
        Transformed data
    """
    return np.sign(data) * np.log10(1. + np.absolute(data) / const)

def CfromE(e1, e2):
    """
    Calculate metric tensor from lattice vectors.
    
    Parameters:
    -----------
    e1, e2 : array_like
        Lattice vectors
        
    Returns:
    --------
    Cij : ndarray
        2x2 metric tensor
    """
    Cij = np.zeros([2, 2])
    Cij[0, 0] = e1[0]*e1[0] + e1[1]*e1[1]
    Cij[0, 1] = e1[0]*e2[0] + e1[1]*e2[1]
    Cij[1, 0] = e1[0]*e2[0] + e1[1]*e2[1]
    Cij[1, 1] = e2[0]*e2[0] + e2[1]*e2[1]
    return Cij