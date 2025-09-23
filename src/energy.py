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
    cut = 3.5
    
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



def conti_phi0_from_Cij(c11, c22, c12, lattice):
    """
    Calculate interatomic energy from elastic constants.
    
    Parameters:
    -----------
    c11, c22, c12 : float or array
        Elastic constants
    lattice : str
        Lattice type ('square' or 'triangular')
    
    Returns:
    --------
    energy : float or array
        Computed energy values
    """
    
    # Material parameters (you may need to adjust these based on your system)
    if lattice == 'square':
        K = 4.0  # Adjust based on your material
        beta = -0.25 # Adjust based on your material
        burgers = 1.0  # Burgers vector magnitude
    elif lattice == 'triangular':
        K = 1.0  # Different values for triangular lattice
        beta = 4.0
        burgers = 1.0
    else:
        raise ValueError("lattice must be 'square' or 'triangular'")
    
    """
    Calculate energy for scalar or array inputs of c11, c22, c12
    
    Parameters:
    c11, c22, c12: float or numpy array
    K, burgers, beta: float
    """

    """
    Direct translation from your clean Mathematica code.
    This should be MUCH more reliable than the C++ mess!
    """
    
    # Parameters from your Mathematica code
    beta = -0.25
    K = 1
    
    # Helper functions (direct translation from Mathematica)
    def sqrtdetC(c11, c22, c12):
        return np.sqrt(c11*c22 - c12**2)
    
    def detC(c11, c22, c12):
        return c11*c22 - c12**2
    
    def trC(c11, c22, c12):
        return c11 + c22
    
    def y1(c11, c22, c12):
        return (c11 - c22) / sqrtdetC(c11, c22, c12)
    
    def y2(c11, c22, c12):
        return (c11 + c22 - 4*c12) / sqrtdetC(c11, c22, c12)
    
    def y3(c11, c22, c12):
        return (c11 + c22 - c12) / sqrtdetC(c11, c22, c12)
    
    def I1(c11, c22, c12):
        return y3(c11, c22, c12) / 3
    
    def I2(c11, c22, c12):
        return (y1(c11, c22, c12)**2 + y2(c11, c22, c12)**2/3) / 4
    
    def I3(c11, c22, c12):
        return (y1(c11, c22, c12)**2 * y2(c11, c22, c12) - 
                y2(c11, c22, c12)**3/9)
    
    def ksi1(c11, c22, c12):
        return (I1(c11, c22, c12)**4 * I2(c11, c22, c12) - 
                41 * I2(c11, c22, c12)**3 / 99 +
                7 * I1(c11, c22, c12) * I2(c11, c22, c12) * I3(c11, c22, c12) / 66 +
                I3(c11, c22, c12)**2 / 1056)
    
    def ksi3(c11, c22, c12):
        return (4 * I2(c11, c22, c12)**3 / 11 + 
                I1(c11, c22, c12)**3 * I3(c11, c22, c12) -
                8 * I1(c11, c22, c12) * I2(c11, c22, c12) * I3(c11, c22, c12) / 11 +
                17 * I3(c11, c22, c12)**2 / 528)
    
    # Ensure numerical stability
    det = detC(c11, c22, c12)
    det = np.where(det <= 0, 1e-10, det)  # Prevent issues with negative determinant
    
    # Main energy function (ksi)
    energy = (beta * ksi1(c11, c22, c12) + 
              ksi3(c11, c22, c12) +
              K * (det - np.log(det)))
    
    return energy


