"""
Energy density calculations for different lattice types.

This module computes strain energy density using interatomic potentials
for square and triangular lattices.
"""

import numpy as np


import numpy as np


def square_energy(r):
    """
    Calculate energy for square lattice potential.
    
    Parameters:
    -----------
    r : array_like
        Distance
        
    Returns:
    --------
    energy : array_like
        Potential energy
    """
    # Constants
    a = 1.0
    c1 = 2.0 * a
    c2 = 2.0 * a
    b1 = 8.0
    r1 = 1.0
    r2 = 1.425
    
    # Energy calculation
    repulsive_term = a / r**12
    attractive_term1 = c1 * np.exp(-b1 * (r - r1)**2)
    attractive_term2 = c2 * np.exp(-b1 * (r - r2)**2)
    
    return repulsive_term - attractive_term1 - attractive_term2


def square_energy_der(r):
    """
    Calculate derivative of energy for square lattice potential.
    
    Parameters:
    -----------
    r : array_like
        Distance
        
    Returns:
    --------
    derivative : array_like
        dE/dr
    """
    # Constants
    a = 1.0
    c1 = 2.0 * a
    c2 = 2.0 * a
    b1 = 8.0
    r1 = 1.0
    r2 = 1.425
    
    # Derivative terms
    repulsive_der = -12.0 * a / r**13
    attractive_der1 = 2.0 * b1 * c1 * (r - r1) * np.exp(-b1 * (r - r1)**2)
    attractive_der2 = 2.0 * b1 * c2 * (r - r2) * np.exp(-b1 * (r - r2)**2)
    
    return repulsive_der + attractive_der1 + attractive_der2

def lennard_jones_energy_v3(r):
    """
    Lennard-Jones energy variant 3.
    
    Parameters:
    -----------
    r : array_like
        Distance(s)
    
    Returns:
    --------
    energy : array_like
        Potential energy
    """
    # Constants
    sigma = 1.0
    rc = 1.86602540378444  # sigma*((sqrt(3)+2)/2)
    A = -4.47124703217530
    B = -90.2632082644540
    C2p0 = -5530.88442798764
    C2p1 = 20688.7278150076
    C2p2 = -34486.6459222351
    C2p3 = 32704.6608372787
    C2p4 = -18879.7974113434
    C2p5 = 6595.54667916880
    C2p6 = -1286.11734180526
    C2p7 = 107.717810684010
    
    energy = np.zeros_like(r)
    
    # Region 1: r < sigma
    # U1 = -2/r^4 + r^(-8)
    mask1 = r < sigma
    r4 = r[mask1]**4
    r8 = r[mask1]**8
    energy[mask1] = -2.0 / r4 + 1.0 / r8
    
    # Region 2: sigma <= r < rc
    # U2 = A/r^8 - B/r^4 + polynomial
    mask2 = (r >= sigma) & (r < rc)
    r_m2 = r[mask2]
    r2 = r_m2**2
    r3 = r_m2**3
    r4 = r_m2**4
    r5 = r_m2**5
    r6 = r_m2**6
    r7 = r_m2**7
    r8 = r_m2**8
    
    energy[mask2] = (A / r8 - B / r4 +
                     C2p0 + C2p1 * r_m2 + C2p2 * r2 + C2p3 * r3 +
                     C2p4 * r4 + C2p5 * r5 + C2p6 * r6 + C2p7 * r7)
    
    # Region 3: r >= rc
    # U3 = 0 (already initialized to zero)
    
    return energy

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
    scl = 1.0659
    cut = 3.5
    
    phi0 = np.zeros(np.shape(c11))
    
    # Sum over lattice neighbors
    for s in range(-9, 9):
        for l in range(-9, 9):
            if (s != 0) or (l != 0):
                # Distance calculation
                r = scl * np.sqrt((s**2) * c11 + 2.*s*l*c12 + (l**2) * c22)
                
                # Potential calculation
                tmp = np.where(
                    r > cut, 
                    0., 
                    a/(r)**12. - c1*np.exp(-b1*(r-r1)**2.) - c2*np.exp(-b2*(r-r2)**2.)
                )
                #tmp = lennard_jones_energy_v3(r)
                # tmp = square_energy(r)
                phi0 += tmp
    
    phi0 = 0.5 * phi0
    
    return phi0



import numpy as np

def deformation_gradient_from_metric(metric):
    """
    Calculate deformation gradient F from the metric tensor C = F^T @ F
    using eigendecomposition.
    
    Parameters:
    -----------
    metric : ndarray, shape (..., 2, 2) or (2, 2)
        Right Cauchy-Green deformation tensor C = F^T @ F
        Can handle arrays of metric tensors
    
    Returns:
    --------
    F : ndarray
        Deformation gradient tensor
    """
    # Check if we have a single metric or an array of metrics
    original_shape = metric.shape
    
    if len(original_shape) == 2:
        # Single metric tensor case
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        
        # Ensure positive eigenvalues (metric should be positive definite)
        if np.any(eigenvalues < 0):
            raise ValueError("Metric tensor has negative eigenvalues - not physical")
        
        # Calculate F = Q @ sqrt(Lambda) @ Q^T
        sqrt_lambda = np.diag(np.sqrt(eigenvalues))
        F = eigenvectors @ sqrt_lambda @ eigenvectors.T
    else:
        # Array of metric tensors case
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        
        # Ensure positive eigenvalues
        if np.any(eigenvalues < 0):
            raise ValueError("Metric tensor has negative eigenvalues - not physical")
        
        # Calculate F = Q @ sqrt(Lambda) @ Q^T for each point
        # eigenvalues shape: (..., 2)
        # eigenvectors shape: (..., 2, 2)
        sqrt_lambda = np.sqrt(eigenvalues)
        
        # F = eigenvectors @ diag(sqrt_lambda) @ eigenvectors.T
        # Using einsum for efficient computation
        F = np.einsum('...ij,...j,...kj->...ik', 
                      eigenvectors, sqrt_lambda, eigenvectors)
    
    return F


def interatomic_stress_from_Cij(c11, c22, c12, M, lattice, F=None):
    """
    Compute Cauchy stress components from interatomic potential.
    
    Parameters:
    -----------
    c11, c22, c12 : array_like
        Metric tensor components (can be scalars or arrays)
    M : array_like
        Transformation matrix (2x2 or (..., 2, 2))
    lattice : str
        Lattice type ('square' or 'triangular')
    F : array_like, optional
        Deformation gradient (2x2 or (..., 2, 2)). If provided, skips eigen decomposition.
        
    Returns:
    --------
    sigma_11, sigma_22, sigma_12 : array_like
        Cauchy stress tensor components
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
    cut = 16.5

    # Determine if inputs are arrays or scalars
    is_array = isinstance(c11, np.ndarray)
    
    # Calculate deformation gradient if not provided
    if F is None:
        if is_array:
            # Construct metric tensor array: shape (..., 2, 2)
            shape = c11.shape
            metric = np.zeros(shape + (2, 2))
            metric[..., 0, 0] = c11
            metric[..., 0, 1] = c12
            metric[..., 1, 0] = c12
            metric[..., 1, 1] = c22
        else:
            # Scalar case: construct simple 2x2 matrix
            metric = np.array([[c11, c12],
                               [c12, c22]])
        
        # Calculate deformation gradient from metric tensor (does eigen decomposition)
        F = deformation_gradient_from_metric(metric)
    
    # Calculate Jacobian (determinant of F)
    J = np.linalg.det(F)
    
    # Initialize 2nd Piola-Kirchhoff stress components (dφ/dc_ij)
    S_11 = np.zeros(np.shape(c11))
    S_22 = np.zeros(np.shape(c11))
    S_12 = np.zeros(np.shape(c11))
    S_21 = np.zeros(np.shape(c11))
    
    # Sum over lattice neighbors to compute dφ/dc_ij
    for s in range(-9, 9):
        for l in range(-9, 9):
            if (s != 0) or (l != 0):
                # Distance calculation
                r = scl * np.sqrt((s**2) * c11 + 2.*s*l*c12 + (l**2) * c22)
                
                # Derivative of potential with respect to r
                dphi_dr = 0.5*square_energy_der(r)/r
                
                # Chain rule: dφ/dc_ij = (dφ/dr) * (dr/dc_ij)
                S_11 += dphi_dr * 0.5*(scl*scl * s**2)
                S_22 += dphi_dr * 0.5*(scl*scl * l**2)
                S_12 += dphi_dr * (scl*scl * s*l)
                S_21 += dphi_dr * (scl*scl * s*l)
            
    S_12 *= 0.5 
    S_21 *= 0.5 
    
    if is_array:
        # Construct 2nd Piola-Kirchhoff stress tensor array
        shape = c11.shape
        S = np.zeros(shape + (2, 2))
        S[..., 0, 0] = S_11
        S[..., 0, 1] = S_12
        S[..., 1, 0] = S_12
        S[..., 1, 1] = S_22
        
        # Transform to Cauchy stress: σ = 2 * F * M * S * M^T * F^T
        #FSF = 2 * np.einsum('...ia,...ab,...bc,...dc,...ed->...ie', F, M, S, M, F)
        FSF = 2 * np.einsum('...ab,...bc,...dc->...ad', M, S, M)

        sigma = FSF
        
        # Extract components
        sigma_11 = sigma[..., 0, 0]
        sigma_22 = sigma[..., 1, 1]
        sigma_12 = sigma[..., 0, 1]
    else:
        # Scalar case
        S = np.array([[S_11, S_12],
                      [S_12, S_22]])
        
        # Transform to Cauchy stress
        sigma = 2 * (F @ M) @ S @ (F @ M).T
        
        # Extract components
        sigma_11 = sigma[0, 0]
        sigma_22 = sigma[1, 1]
        sigma_12 = sigma[0, 1]
    
    return sigma_11, sigma_22, sigma_12



def interatomic_stress_from_Binv(b_inv_11, b_inv_22, b_inv_12, lattice):
    """
    Compute Cauchy stress components from interatomic potential using B^{-1}.
    
    Parameters:
    -----------
    b_inv_11, b_inv_22, b_inv_12 : array_like
        Components of B^{-1} = (F·F^T)^{-1} (Eulerian/spatial frame)
    lattice : str
        Lattice type ('square' or 'triangular')
        
    Returns:
    --------
    sigma_11, sigma_22, sigma_12 : array_like
        Cauchy stress tensor components (directly, no transformation needed!)
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
    cut = 6.5

    # Determine if inputs are arrays or scalars
    is_array = isinstance(b_inv_11, np.ndarray)
    
    # Invert B^{-1} to get B for distance calculations
    if is_array:
        shape = b_inv_11.shape
        B_inv = np.zeros(shape + (2, 2))
        B_inv[..., 0, 0] = b_inv_11
        B_inv[..., 0, 1] = b_inv_12
        B_inv[..., 1, 0] = b_inv_12
        B_inv[..., 1, 1] = b_inv_22
        
        B = np.linalg.inv(B_inv)
        b11 = B[..., 0, 0]
        b22 = B[..., 1, 1]
        b12 = B[..., 0, 1]
    else:
        # Compute B from B^{-1} using 2x2 inverse formula
        det_Binv = b_inv_11 * b_inv_22 - b_inv_12**2
        b11 = b_inv_22 / det_Binv
        b22 = b_inv_11 / det_Binv
        b12 = -b_inv_12 / det_Binv
    
    # Initialize stress derivatives w.r.t. B (dψ/db_ij)
    T_11 = np.zeros(np.shape(b_inv_11))
    T_22 = np.zeros(np.shape(b_inv_11))
    T_12 = np.zeros(np.shape(b_inv_11))
    
    # Sum over lattice neighbors to compute dψ/db_ij
    # Distances in current configuration: r^2 = s^2*b11 + 2*s*l*b12 + l^2*b22
    for s in range(-16, 16):
        for l in range(-16, 16):
            if (s != 0) or (l != 0):
                # Distance calculation using B (spatial metric)
                r = scl * np.sqrt((s**2) * b11 + 2.*s*l*b12 + (l**2) * b22)
                
                # Derivative of potential with respect to r
                dphi_dr = 0.5 * square_energy_der(r) / r
                
                # Chain rule: dψ/db_ij = (dψ/dr) * (dr/db_ij)
                T_11 += dphi_dr * 0.5 * (scl*scl * s**2)
                T_22 += dphi_dr * 0.5 * (scl*scl * l**2)
                T_12 += dphi_dr * (scl*scl * s*l)
    
    T_12 *= 0.5
    
    # Convert from dψ/dB to dψ/d(B^{-1}) using chain rule
    # ∂ψ/∂(B^{-1}) = -B · (∂ψ/∂B) · B
    if is_array:
        T = np.zeros(shape + (2, 2))
        T[..., 0, 0] = T_11
        T[..., 0, 1] = T_12
        T[..., 1, 0] = T_12
        T[..., 1, 1] = T_22
        
        # Transform: S_Binv = -B · T · B
        S_Binv = -np.einsum('...ab,...bc,...cd->...ad', B, T, B)
        
        # Cauchy stress (NO transformation needed - already in spatial frame!)
        sigma = 2 * S_Binv
        
        # Extract components
        sigma_11 = sigma[..., 0, 0]
        sigma_22 = sigma[..., 1, 1]
        sigma_12 = sigma[..., 0, 1]
    else:
        T = np.array([[T_11, T_12],
                      [T_12, T_22]])
        
        B_mat = np.array([[b11, b12],
                          [b12, b22]])
        
        # Transform: S_Binv = -B · T · B
        S_Binv = -B_mat @ T @ B_mat
        
        # Cauchy stress
        sigma = 2 * S_Binv
        
        # Extract components
        sigma_11 = sigma[0, 0]
        sigma_22 = sigma[1, 1]
        sigma_12 = sigma[0, 1]
    
    return sigma_11, sigma_22, sigma_12

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
        K = 1.0  # Adjust based on your material
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


