#!/usr/bin/env python3
"""
Main script for PoincarÃ© disk analysis with centered metrics.

This script performs the complete workflow:
1. Generate stereographic projection coordinates
2. Apply centering transformation
3. Perform Lagrange reduction
4. Compute strain energy density
5. Visualize results with fundamental domain boundaries

Added functionality:
- Batch processing of triangles_*.dat files
- F matrix reading and C tensor computation
- Scatter plot generation on Poincaré disk
"""

import time
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

from src.projections import Cij_from_stereographic_projection_tr, Cij_from_stereographic_projection
from src.lagrange import lagrange_reduction, lagrange_reduction_with_matrix
from src.elastic_reduction_square import reduction_elastic_square
from src.elastic_reduction_v2 import elasticReduction
from src.energy import interatomic_phi0_from_Cij, conti_phi0_from_Cij,interatomic_stress_from_Cij,convert_data_SymLog,interatomic_stress_from_Binv
# from src.plotting import (poincare_plot_energy_with_precise_boundaries, 
#                           poincare_plot_energy_with_f_matrices,
#                           read_f_matrix_from_file)
from src.plotting import poincare_plot_energy_with_precise_boundaries
from src.plotting import (poincare_plot_energy_with_f_matrices,
                          read_f_matrix_from_file,
                          poincare_plot_path,
                          project_c_to_poincare,
                          compute_c_tensor_from_f)
from src.enhanced_plotting import poincare_plot_energy_with_fundamental_domains
from src.visualization_3d import create_3d_energy_surface, make_browser_interactive_plots

from pathlib import Path


def shear_stress_analysis(alpha_range, lattice='square', apply_lagrange=False, normalize=False, use_binv=False):
    """
    Apply shear deformation F = [[1, α], [0, 1]] and analyze σ₁₂ response.
    
    Can compute stress from either C (Lagrangian) or B^{-1} (Eulerian).
    
    Parameters:
    -----------
    alpha_range : array_like
        Range of shear parameter values to test
    lattice : str
        Lattice type ('square' or 'triangular')
    apply_lagrange : bool
        Whether to apply Lagrange reduction to metric tensor
    normalize : bool
        Whether to normalize stress values
    use_binv : bool
        If True, compute stress from B^{-1} (Eulerian)
        If False, compute stress from C (Lagrangian) with M transformation
        
    Returns:
    --------
    alpha_values : np.ndarray
        Array of alpha values
    sigma_12 : np.ndarray
        Shear stress component for each alpha
    results_dict : dict
        Dictionary containing all computed quantities
    """
    print("Starting shear stress analysis...")
    print(f"Using {'B^{-1} (Eulerian)' if use_binv else 'C (Lagrangian)'} formulation")
    
    alpha_values = np.asarray(alpha_range)
    N = len(alpha_values)
    
    # Initialize storage
    c11_array = np.zeros(N)
    c22_array = np.zeros(N)
    c12_array = np.zeros(N)
    c11_red_array = np.zeros(N)
    c22_red_array = np.zeros(N)
    c12_red_array = np.zeros(N)
    
    b_inv_11_array = np.zeros(N)
    b_inv_22_array = np.zeros(N)
    b_inv_12_array = np.zeros(N)
    
    sigma_11_array = np.zeros(N)
    sigma_22_array = np.zeros(N)
    sigma_12_array = np.zeros(N)
    f1_array = np.zeros((N, 2))
    f2_array = np.zeros((N, 2))
    m_matrices = []
    
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    
    print(f"Computing stress for {N} alpha values...")
    
    for i, alpha in enumerate(alpha_values):
        # Shear deformation gradient F = [[1, α], [0, 1]]
        F_app = np.array([[1.0, alpha],
                          [0.0, 1.0]])
        
        # Deformed basis vectors: f_i = F @ e_i
        f1 = F_app @ e1
        f2 = F_app @ e2
        
        f1_array[i] = f1
        f2_array[i] = f2
        
        # Construct F matrix
        F = np.column_stack([f1, f2])
        
        # ===== Compute C = F^T·F (Lagrangian) =====
        c11 = np.dot(f1, f1)
        c22 = np.dot(f2, f2)
        c12 = np.dot(f1, f2)
        
        c11_array[i] = c11
        c22_array[i] = c22
        c12_array[i] = c12
        
        # ===== Compute B^{-1} = (F·F^T)^{-1} (Eulerian) =====
        B = F @ F.T
        B_inv = np.linalg.inv(B)
        
        b_inv_11 = B_inv[0, 0]
        b_inv_22 = B_inv[1, 1]
        b_inv_12 = B_inv[0, 1]
        
        b_inv_11_array[i] = b_inv_11
        b_inv_22_array[i] = b_inv_22
        b_inv_12_array[i] = b_inv_12
        
        # ===== Apply Lagrange reduction if requested =====
        if apply_lagrange:
            # Reduce C to get transformation matrix m
            c11_red, c22_red, c12_red, m_matrix, third_cond, iters = \
                lagrange_reduction_with_matrix(c11, c22, c12)
            
            m_matrices.append(m_matrix)
            
            # Store reduced C
            c11_red_array[i] = c11_red
            c22_red_array[i] = c22_red
            c12_red_array[i] = c12_red
            
            # Transform B^{-1}: B̃^{-1} = m^{-T}·B^{-1}·m^{-1}
            m_inv = np.linalg.inv(m_matrix)
            B_inv_mat = np.array([[b_inv_11, b_inv_12],
                                  [b_inv_12, b_inv_22]])
            B_inv_tilde = m_inv.T @ B_inv_mat @ m_inv
            
            b_inv_11_red = B_inv_tilde[0, 0]
            b_inv_22_red = B_inv_tilde[1, 1]
            b_inv_12_red = B_inv_tilde[0, 1]
        else:
            # No reduction
            c11_red, c22_red, c12_red = c11, c22, c12
            b_inv_11_red, b_inv_22_red, b_inv_12_red = b_inv_11, b_inv_22, b_inv_12
            m_matrix = np.eye(2)
            m_matrices.append(m_matrix)
            
            c11_red_array[i] = c11_red
            c22_red_array[i] = c22_red
            c12_red_array[i] = c12_red
        
        # ===== Compute stress (choose formulation) =====
        if use_binv:
            # Eulerian formulation: use B^{-1}
            sigma_11, sigma_22, sigma_12 = interatomic_stress_from_Binv(
                b_inv_11_red, b_inv_22_red, b_inv_12_red, lattice
            )
        else:
            # Lagrangian formulation: use C with M transformation
            sigma_11, sigma_22, sigma_12 = interatomic_stress_from_Cij(
                c11_red, c22_red, c12_red, m_matrix, lattice, F=F_app
            )
        
        sigma_11_array[i] = sigma_11
        sigma_22_array[i] = sigma_22
        sigma_12_array[i] = sigma_12
    
    print(f"Stress range: σ₁₂ ∈ [{np.nanmin(sigma_12_array):.2e}, {np.nanmax(sigma_12_array):.2e}]")
    
    # Normalize if requested
    if normalize:
        sigma_12_normalized = (sigma_12_array - np.nanmean(sigma_12_array)) / np.nanstd(sigma_12_array)
    else:
        sigma_12_normalized = sigma_12_array
    
    # Pack results
    results_dict = {
        'c11': c11_array,
        'c22': c22_array,
        'c12': c12_array,
        'c11_red': c11_red_array,
        'c22_red': c22_red_array,
        'c12_red': c12_red_array,
        'b_inv_11': b_inv_11_array,
        'b_inv_22': b_inv_22_array,
        'b_inv_12': b_inv_12_array,
        'sigma_11': sigma_11_array,
        'sigma_22': sigma_22_array,
        'sigma_12_raw': sigma_12_array,
        'sigma_12_normalized': sigma_12_normalized,
        'f1': f1_array,
        'f2': f2_array,
        'm_matrices': m_matrices,
        'lattice': lattice,
        'method': 'B_inverse' if use_binv else 'C_with_M'
    }
    
    return alpha_values, sigma_22_array, results_dict

def plot_shear_stress_response(alpha_range, lattice='square', apply_lagrange=False, 
                               normalize=False, figsize=(14, 10)):
    """
    Plot comprehensive shear stress analysis results.
    
    Parameters:
    -----------
    alpha_range : array_like
        Range of shear parameter values
    lattice : str
        Lattice type ('square' or 'triangular')
    apply_lagrange : bool
        Whether to apply Lagrange reduction
    normalize : bool
        Whether to normalize stress
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    results_dict : dict
        All computed quantities
    """
    import matplotlib.pyplot as plt
    
    # Perform analysis
    alpha, sigma_12, results = shear_stress_analysis(
        alpha_range, lattice, apply_lagrange, normalize
    )
    
    # Create figure with subplots (similar layout to generate_background_energy visualizations)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Plot 1: σ₁₂ vs α (MAIN RESULT)
    axes[0, 0].plot(alpha, sigma_12, 'b-', linewidth=2.5, label='σ₁₂')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Shear parameter α', fontsize=13)
    axes[0, 0].set_ylabel('σ₁₂ (Shear stress)', fontsize=13)
    axes[0, 0].set_title(f'Shear Stress Response - {lattice} lattice', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: All stress components
    axes[0, 1].plot(alpha, results['sigma_11'], 'r-', label='σ₁₁', linewidth=2)
    axes[0, 1].plot(alpha, results['sigma_22'], 'g-', label='σ₂₂', linewidth=2)
    axes[0, 1].plot(alpha, results['sigma_12'], 'b-', label='σ₁₂', linewidth=2)
    axes[0, 1].set_xlabel('Shear parameter α', fontsize=13)
    axes[0, 1].set_ylabel('Stress components', fontsize=13)
    axes[0, 1].set_title('All Stress Components', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 3: Metric tensor components
    axes[0, 2].plot(alpha, results['c11'], 'r--', label='C₁₁', linewidth=1.5, alpha=0.5)
    axes[0, 2].plot(alpha, results['c22'], 'g--', label='C₂₂', linewidth=1.5, alpha=0.5)
    axes[0, 2].plot(alpha, results['c12'], 'b--', label='C₁₂', linewidth=1.5, alpha=0.5)
    if apply_lagrange:
        axes[0, 2].plot(alpha, results['c11_red'], 'r-', label='C₁₁ (red)', linewidth=2)
        axes[0, 2].plot(alpha, results['c22_red'], 'g-', label='C₂₂ (red)', linewidth=2)
        axes[0, 2].plot(alpha, results['c12_red'], 'b-', label='C₁₂ (red)', linewidth=2)
    axes[0, 2].set_xlabel('Shear parameter α', fontsize=13)
    axes[0, 2].set_ylabel('Metric components', fontsize=13)
    axes[0, 2].set_title('Metric Tensor Evolution', fontsize=14)
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Deformed basis vector f₁
    axes[1, 0].plot(alpha, results['f1'][:, 0], 'r-', label='f₁ˣ', linewidth=2)
    axes[1, 0].plot(alpha, results['f1'][:, 1], 'b-', label='f₁ʸ', linewidth=2)
    axes[1, 0].set_xlabel('Shear parameter α', fontsize=13)
    axes[1, 0].set_ylabel('f₁ components', fontsize=13)
    axes[1, 0].set_title('Deformed Vector f₁', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 5: Deformed basis vector f₂
    axes[1, 1].plot(alpha, results['f2'][:, 0], 'r-', label='f₂ˣ', linewidth=2)
    axes[1, 1].plot(alpha, results['f2'][:, 1], 'b-', label='f₂ʸ', linewidth=2)
    axes[1, 1].set_xlabel('Shear parameter α', fontsize=13)
    axes[1, 1].set_ylabel('f₂ components', fontsize=13)
    axes[1, 1].set_title('Deformed Vector f₂', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 6: Stress vs C₁₂ (phase space view)
    axes[1, 2].scatter(results['c12'], sigma_12, c=alpha, cmap='viridis', s=20)
    cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
    cbar.set_label('α', fontsize=12)
    axes[1, 2].set_xlabel('C₁₂ (metric component)', fontsize=13)
    axes[1, 2].set_ylabel('σ₁₂ (shear stress)', fontsize=13)
    axes[1, 2].set_title('Stress-Strain Phase Space', fontsize=14)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes, results


# Simple usage function
def quick_shear_plot(alpha_min=-0.5, alpha_max=0.5, n_points=100, lattice='square', save_path='shear_stress.png'):
    """Quick plot of σ₁₂ vs α with minimal options."""
    import matplotlib.pyplot as plt
    
    print(f"Creating alpha range: [{alpha_min}, {alpha_max}] with {n_points} points")
    alpha = np.linspace(alpha_min, alpha_max, n_points)
    
    print(f"Running shear stress analysis...")
    _, sigma_12, _ = shear_stress_analysis(alpha, lattice=lattice)
    print(f"sigma_12 range: [{np.min(sigma_12):.3e}, {np.max(sigma_12):.3e}]")
    
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alpha, sigma_12, 'b-', linewidth=2.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Shear parameter α', fontsize=14)
    ax.set_ylabel('σ₁₂ (Shear stress)', fontsize=14)
    ax.set_title(f'Shear Stress Response - {lattice} lattice', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {save_path}")
    plt.close()
    
    return alpha, sigma_12

def get_transformed_domain(x_, y_, m):
    """
    Apply fundamental domain conditions {0 < C̃₁₁ ≤ C̃₂₂, 0 ≤ C̃₁₂ ≤ C̃₁₁/2} 
    where C̃ = m^T C m
    """
    # Get original metric components
    c11, c22, c12 = Cij_from_stereographic_projection(x_, y_)
    
    # Transform: C̃ = m^T C m
    c11_tilde = m[0,0]**2 * c11 + 2*m[0,0]*m[1,0]*c12 + m[1,0]**2 * c22
    c12_tilde = m[0,0]*m[0,1]*c11 + (m[0,0]*m[1,1] + m[0,1]*m[1,0])*c12 + m[1,0]*m[1,1]*c22
    c22_tilde = m[0,1]**2 * c11 + 2*m[0,1]*m[1,1]*c12 + m[1,1]**2 * c22
    
    # Apply fundamental domain conditions to transformed metric
    keep = (c11_tilde > 0) & (c11_tilde <= c22_tilde) & (c12_tilde >= 0) & (c12_tilde <= c11_tilde/2) & (x_**2 + y_**2 < 1)
    
    return keep


def generate_background_energy(disc=1000, lattice='square'):
    """
    Generate the background energy configuration for the Poincaré disk.
    
    Parameters:
    -----------
    disc : int
        Discretization resolution
    lattice : str
        Lattice type ('square' or 'triangular')
        
    Returns:
    --------
    config : numpy.ndarray
        Energy configuration for background
    pvmin, pvmax : float
        Color scale limits
    """
    print("Generating background energy configuration...")
    
    # Generate stereographic projection coordinates
    size_of_disk = .999
    x, y = np.linspace(-size_of_disk, size_of_disk, num=disc, endpoint=True), np.linspace(-size_of_disk, size_of_disk, num=disc, endpoint=True)
    x_, y_ = np.meshgrid(x, y)
    size_of_disk2 =   .999
    # Mask values outside Poincaré disk
    x_ = np.where(x_**2 + y_**2 - (size_of_disk2)**2 >= 1.e-6, np.nan, x_)
    y_ = np.where(x_**2 + y_**2 - (size_of_disk2)**2 >= 1.e-6, np.nan, y_)


    # size_of_disk = .999
    # x, y = np.linspace(-size_of_disk, size_of_disk, num=disc, endpoint=True), np.linspace(-size_of_disk, size_of_disk, num=disc, endpoint=True)
    # x_, y_ = np.meshgrid(x, y)

    # Constraint 1: Inside Poincaré disk
    mask = x_**2 + y_**2 >= 1.

    # buffer = 0.1  # adjust this value to control distance from boundaries

    # # Constraint 2: C12 >= 0 (requires y >= 0), with buffer
    # mask |= y_ < buffer

    # # Constraint 3: C12 <= C11 (requires (x+1)^2 + (y-1)^2 >= 1), with buffer
    # mask |= (x_ + 1)**2 + (y_ - 1)**2 < 1.0 + buffer

    # # Constraint 4: C12 <= C22 (requires (x-1)^2 + (y-1)^2 >= 1), with buffer
    # mask |= (x_ - 1)**2 + (y_ - 1)**2 < 1.0 + buffer


    # For y >= 0: exclude circles of radius 2 centered at (-1, 2) and (1, 2)
    mask_upper = (y_ >= 0) & (((x_ + 1)**2 + (y_ - 2)**2 < 4) | ((x_ - 1)**2 + (y_ - 2)**2 < 4))

    # For y < 0: exclude circles of radius 2 centered at (-1, -2) and (1, -2)
    mask_lower = (y_ < 0) & (((x_ + 1)**2 + (y_ + 2)**2 < 4) | ((x_ - 1)**2 + (y_ + 2)**2 < 4))

    # Combine with existing constraints
    mask |= mask_upper | mask_lower



    # Keep original domain
    # keep_original = (x_ <= 0) & (y_ >= 0) & ((x_ + 1)**2 + (y_ - 1)**2 >= 1) & ((x_ + 1)**2 + (y_ - 2)**2 >= 4) & (x_**2 + y_**2 < 1)
    # keep_transformed = ((x_ + 1)**2 + (y_ - 1)**2 >= 1) & ((x_ + 1)**2 + (y_ - 2)**2 <= 4) & ((x_ - 1)**2 + (y_ - 2)**2 >= 4) & (x_**2 + y_**2 < 1)
    # keep_transformed2 = ((x_ + 1)**2 + (y_ + 1)**2 >= 1) & ((x_ + 1)**2 + (y_ + 2)**2 <= 4) & ((x_ - 1)**2 + (y_ + 2)**2 >= 4) & (x_**2 + y_**2 < 1)

    # m = np.array([[1, 0], [0, 1]])
    # keep_transformed = get_transformed_domain(x_, y_, m)



    # Keep transformed domain  
    # keep_transformed = (y_ <= 0) & ((x_ + 1)**2 + (y_ + 1)**2 >= 1) & ((x_ - 1)**2 + (y_ + 2)**2 >= 4) & (x_**2 + y_**2 < 1)

    # Mask everything except these two domains
    # mask |= ~(  keep_transformed )


    # Apply mask
    # x_ = np.where(mask, np.nan, x_)
    # y_ = np.where(mask, np.nan, y_)

    # Apply stereographic projection with centering transformation
    c11, c22, c12, c11t, c22t, c12t = Cij_from_stereographic_projection_tr(x_, y_,lattice)
    
    # Perform Lagrange reduction on centered metrics
    #c11_reduced, c22_reduced, c12_reduced, iterations = lagrange_reduction(c11, c22, c12, verbose=False)
    c11_red, c22_red, c12_red, m_matrix, third_cond, iters = lagrange_reduction_with_matrix(c11, c22, c12)

    #c11_reduced, c22_reduced, c12_reduced = c11, c22, c12
    # Compute strain energy density
    #phi0 = interatomic_phi0_from_Cij(c11_reduced, c22_reduced, c12_reduced, lattice)
    #phi0 = conti_phi0_from_Cij(c11_reduced, c22_reduced, c12_reduced, lattice)
    # Create identity matrix for transformation (since we are using reduced metrics)
    shape = c11.shape
    id_matrix = np.tile(np.eye(2), shape + (1, 1))

    # Compute strain energy density using REDUCED metrics
    #sigma_11, sigma_22, sigma_12 = interatomic_stress_from_Cij(c11_red, c22_red, c12_red, id_matrix, lattice)
    
    #phi0  = sigma_12

    phi0 = interatomic_phi0_from_Cij(c11, c22, c12, lattice)
    phi0_normalized = phi0
    
    
    
    # Normalize energy data
    phi0_normalized = (phi0 - np.nanmin(phi0)) / (np.nanmax(phi0) - np.nanmin(phi0))
    #phi0_normalized = (phi0 - np.nanmean(phi0)) / np.nanstd(phi0)
    phi0 = phi0_normalized
    print(f"Background energy range: {np.nanmin(phi0_normalized):.2e} to {np.nanmax(phi0_normalized):.2e}")

    
    #phi0_normalized = (phi0 - np.nanmean(phi0)) / np.nanstd(phi0)
    #phi0_normalized = sigma_12

    #phi0_normalized = phi0
    # # Option 1: Clip at percentile thresholds
    # lower_percentile = 50   # Remove bottom 1%
    # upper_percentile = 90  # Remove top 1%

    # # Set outliers to NaN instead of clipping
    # lower_percentile = 0   # Remove bottom 1%
    # upper_percentile = 100  # Remove top 1%

    # lower_thresh = np.nanpercentile(phi0, lower_percentile)
    # upper_thresh = np.nanpercentile(phi0, upper_percentile)

    # phi0_clipped = phi0.copy()
    # phi0_clipped[(phi0 < lower_thresh) | (phi0 > upper_thresh)] = np.nan

    # phi0_normalized = phi0_clipped    

    # Convert to symmetric log scale
    c_scale = 1e-19  # for interatomic phi0
    c_scale = 1e-19  # for interatomic phi0
    #c_scale = np.median(phi0_normalized[phi0_normalized > 0])
    config = convert_data_SymLog(phi0_normalized, c_scale)
    #config = phi0_normalized
    print(f"Background energy range: {np.nanmin(config):.2e} to {np.nanmax(config):.2e}")
    
    pvmin = np.nanmin(config)
    pvmax = np.nanmax(config)
    
    return config, pvmin, pvmax

def process_triangle_files(folder_path, output_folder='figures', lattice='square', stability_file=None, start_file=None, n_stability_copies=1):
    """
    Process all triangle files in a folder and generate Poincaré disk plots.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing triangles_*.dat files
    output_folder : str
        Output folder for PNG files
    lattice : str
        Lattice type for energy calculation
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate background energy configuration once
    config, pvmin, pvmax = generate_background_energy(lattice=lattice)
    
    # Check if input folder exists
    if not os.path.exists(folder_path):
        print(f"Input folder does not exist: {folder_path}")
        print("Generating plot with background energy only (no scatter points).")
        
        # Create output name for background-only plot
        output_name = os.path.join(output_folder, 'poincare_background_only')
        
        # Create plot without scatter points (pass empty list)
        poincare_plot_energy_with_f_matrices(config, [], output_name, pvmin, pvmax, stability_file=stability_file)
        print(f"Saved background-only plot: {output_name}.png")
        return
    
    print(f'Batch processing triangle files from: {folder_path}')
    
    # Find all triangle files
    pattern = os.path.join(folder_path, 'triangles_*.dat')
    
    # Sort files naturally
    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
        
    triangle_files = sorted(glob.glob(pattern), key=natural_sort_key)
    
    # Filter files if start_file is specified
    if start_file:
        found_start = False
        resumed_files = []
        for f in triangle_files:
            if os.path.basename(f) == start_file:
                found_start = True
            if found_start:
                resumed_files.append(f)
        
        if not found_start:
            print(f"Warning: Start file {start_file} not found in {folder_path}. Processing all files.")
        else:
            triangle_files = resumed_files
            print(f"Resuming from {start_file}. {len(triangle_files)} files remaining.")
    
    if not triangle_files:
        print(f"No triangles_*.dat files found in {folder_path} to process.")
        return
    
    print(f"Found {len(triangle_files)} triangle files to process")
    
    # Process each file
    for i, filename in enumerate(triangle_files):
        print(f"\nProcessing file {i+1}/{len(triangle_files)}: {os.path.basename(filename)}")
        
        # Read F matrices from file
        f_matrices = read_f_matrix_from_file(filename)
        
        if not f_matrices:
            print(f"  Skipping {filename} - no valid data found")
            continue
        
        print(f"  Read {len(f_matrices)} F matrices")
        
        # Generate output name
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_name = os.path.join(output_folder, f'poincare_{base_name}')
        
        # Create scatter plot
        # poincare_plot_energy_with_f_matrices(config, f_matrices, output_name, pvmin, pvmax, plot_mode="scatter", stability_file=stability_file)
        poincare_plot_energy_with_f_matrices(config, f_matrices, output_name, pvmin, pvmax, plot_mode="density", stability_file=stability_file, n_stability_copies=n_stability_copies)
        print(f"  Saved: {output_name}.png")


def save_full_grid_to_xyz(x_grid, y_grid, data_grid, filename="poincare_heatmap_data.xyz"):
    """Saves a full grid. Coordinates are kept, data uses NaN for masking."""
    # Flatten using 'C' order to maintain row-major alignment
    x_flat = x_grid.flatten(order='C')
    y_flat = y_grid.flatten(order='C')
    z_flat = data_grid.flatten(order='C')
    
    combined = np.column_stack((x_flat, y_flat, z_flat))
    
    # Save as space-separated XYZ
    np.savetxt(filename, combined, fmt='%.8e', header='X Y Z', comments='')
    print(f"Successfully exported {len(z_flat)} points to {filename}")

def main_standard():
    print('Lagrange reduction and Poincaré disk projection')
    start_time = time.time()
    os.makedirs('figures', exist_ok=True)
    
    # Parameters
    disc = 1000 
    lattice = 'square'
    symmetry_projection = 'square'  # 'square' or 'triangular' 

    
    # 1. Generate Coordinates
    # x_, y_ are for the math (will have NaNs)
    # x_grid, y_grid are for the FILE (must be clean)
    x = np.linspace(-.999, .999, num=disc, endpoint=True)
    y = np.linspace(-.999, .999, num=disc, endpoint=True)
    x_grid, y_grid = np.meshgrid(x, y)
    
    
    # Create math mask
    x_math = np.where(x_grid**2 + y_grid**2 - (.999)**2 >= 1.e-6, np.nan, x_grid)
    y_math = np.where(x_grid**2 + y_grid**2 - (.999)**2 >= 1.e-6, np.nan, y_grid)
    
    # 2. Run your existing math
    c11, c22, c12, c11t, c22t, c12t = Cij_from_stereographic_projection_tr(x_math, y_math, lattice)
    c11_reduced, c22_reduced, c12_reduced, iterations = lagrange_reduction(c11, c22, c12)
    phi0 = conti_phi0_from_Cij(c11_reduced, c22_reduced, c12_reduced, lattice)
    
    # 3. Normalize and Log Scale
    phi0_min = np.nanmin(phi0)
    phi0_max = np.nanmax(phi0)
    phi0_normalized = (phi0 - phi0_min) / (phi0_max - phi0_min)
    
    c_scale = 1e-13
    config = convert_data_SymLog(phi0_normalized, c_scale)
    #Apply the mask while data is still a 2D matrix
    # Points where radius > 0.999 become NaN
    mask_2d = (x_grid**2 + y_grid**2) > (0.999**2 + 1e-6)
    config_masked = np.where(mask_2d, np.nan, config)

    
    # 5. Save the full grid
    # SAVE using a rock-solid column stack
    # .ravel() is faster and safer for aligned flattening
    combined = np.column_stack((
        x_grid.ravel(), 
        y_grid.ravel(), 
        config_masked.ravel()
    ))
    
    np.savetxt("poincare_heatmap_data.xyz", combined, fmt='%.8e')    
    print(f'Analysis completed in {time.time() - start_time:.2f}s')
    #exit(0)



   
    
    #create_3d_interactive_energy_surface(x_, y_, phi0_normalized, 'square', 'file.pdf')
    #make_browser_interactive_plots(x_, y_, phi0_normalized, phi0_normalized, view_radius=1.)

    #create_3d_energy_surface(x_, y_, phi0_normalized, 'square', 'figures/3d_square_lattice.pdf')
    #make_browser_interactive_plots(x_, y_, config, config)
    #make_browser_interactive_plots(x_, y_, phi0_normalized, phi0_normalized, view_radius=0.3)
    
    # Generate visualization
    print("Creating visualization...")
    saving_index = 1
    output_name = f'./figures/poincare_disk_analysis_{saving_index}'
    stability_file="./stability_boundary_original.dat"
    
    poincare_plot_energy_with_precise_boundaries(
        config, 
        output_name, 
        np.nanmin(config), 
        0.8 * np.nanmax(config),
        disc,stability_file,
        lattice=symmetry_projection
    )
    
    output_name = f'./figures/poincare_newdisk_analysis_{saving_index}'

    poincare_plot_energy_with_fundamental_domains(
        config, 
        output_name, 
        np.nanmin(config), 
         0.8*np.nanmax(config),
        disc,
        shape_flag=symmetry_projection,  # or "triangle"
        stability_file=stability_file)





    # Normalize energy data
    phi0_normalized = (phi0 - np.nanmin(phi0)) / (np.nanmax(phi0) - np.nanmin(phi0))
    phi0_normalized = (phi0 - np.nanmin(phi0)) 
    
    # Convert to symmetric log scale
    c_scale = 1e-19  # for interatomic phi0
    c_scale = 1e-2  # for conti phi0
    c_scale = 1.55e-5
    c_scale = 1e-2

    config = convert_data_SymLog(phi0_normalized, c_scale)
    print(f"Energy range 2: {np.nanmin(phi0_normalized):.2e} to {np.nanmax(phi0_normalized):.2e}")
    #exit(0)
    
    #create_3d_interactive_energy_surface(x_, y_, phi0_normalized, 'square', 'file.pdf')
    #make_browser_interactive_plots(x_, y_, phi0_normalized, phi0_normalized, view_radius=1.)



    end_time = time.time()
    print(f'Analysis completed successfully!')
    print(f'Output saved as: {output_name}.pdf')
    print(f'Total execution time: {end_time - start_time:.4f} seconds')

def process_path_mode(alpha_min=-2.0, alpha_max=2.0, n_steps=100, theta_deg=0.0, output_folder='./figures'):
    """
    Generate path visualization for shear deformation with rotation.
    
    Parameters:
    -----------
    alpha_min, alpha_max : float
        Range of shear amplitude
    n_steps : int
        Number of points along the path
    theta_deg : float
        Rotation angle in degrees
    output_folder : str
        Output directory for figures
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Generating path for shear deformation...")
    print(f"  Alpha range: [{alpha_min}, {alpha_max}]")
    print(f"  Rotation angle: {theta_deg}°")
    print(f"  Number of steps: {n_steps}")
    
    # Generate background energy
    config, pvmin, pvmax = generate_background_energy(lattice='square')
    
    # Rotation angle in radians
    theta = np.deg2rad(theta_deg)
    
    # Define rotated vectors
    # a' = R * [0, 1]^T = [-sin(theta), cos(theta)]^T
    # n' = R * [1, 0]^T = [cos(theta), sin(theta)]^T
    a_prime = np.array([-np.sin(theta), np.cos(theta)])
    n_prime = np.array([np.cos(theta), np.sin(theta)])
    
    # Generate alpha values
    alphas = np.linspace(alpha_min, alpha_max, n_steps)
    
    # Storage for paths (image coordinates)
    paths = {
        'Original': [],
        'Lagrange': [],
        'Elastic_V1': [],
        'Elastic_V2': []
    }
    
    disc = 1000  # Same as background energy grid
    
    # Storage for comparison
    comparison_data = []
    
    for alpha in alphas:
        # Calculate F = I + alpha * (a' otimes n')
        # Outer product: a' \otimes n' gives a 2x2 matrix
        F = np.eye(2) + alpha * np.outer(a_prime, n_prime)
        
        # Calculate C = F^T * F
        C = F.T @ F
        c11, c22, c12 = C[0, 0], C[1, 1], C[0, 1]
        
        # Original path
        x_stereo, y_stereo = project_c_to_poincare(c11, c22, c12)
        if x_stereo**2 + y_stereo**2 < 0.999**2:
            x_img = (x_stereo + 0.999) * (disc - 1) / (2 * 0.999)
            y_img = (y_stereo + 0.999) * (disc - 1) / (2 * 0.999)
            paths['Original'].append((x_img, y_img))
        
        # Lagrange reduced
        try:
            c11_lag, c22_lag, c12_lag, _ = lagrange_reduction(c11, c22, c12, verbose=False)
            x_stereo_lag, y_stereo_lag = project_c_to_poincare(c11_lag, c22_lag, c12_lag)
            if x_stereo_lag**2 + y_stereo_lag**2 < 0.999**2:
                x_img_lag = (x_stereo_lag + 0.999) * (disc - 1) / (2 * 0.999)
                y_img_lag = (y_stereo_lag + 0.999) * (disc - 1) / (2 * 0.999)
                paths['Lagrange'].append((x_img_lag, y_img_lag))
        except:
            pass
        
        # Elastic V1 (reduction_elastic_square)
        C_v1 = None
        try:
            C_mat = np.array([[c11, c12], [c12, c22]])
            C_v1 = reduction_elastic_square(C_mat)
            c11_v1, c22_v1, c12_v1 = C_v1[0,0], C_v1[1,1], C_v1[0,1]
            x_stereo_v1, y_stereo_v1 = project_c_to_poincare(c11_v1, c22_v1, c12_v1)
            if x_stereo_v1**2 + y_stereo_v1**2 < 0.999**2:
                x_img_v1 = (x_stereo_v1 + 0.999) * (disc - 1) / (2 * 0.999)
                y_img_v1 = (y_stereo_v1 + 0.999) * (disc - 1) / (2 * 0.999)
                paths['Elastic_V1'].append((x_img_v1, y_img_v1))
        except Exception as e:
            print(f"  Warning: Elastic V1 failed at alpha={alpha}: {e}")
        
        # Elastic V2 (elasticReduction)
        C_v2 = None
        try:
            C_mat = np.array([[c11, c12], [c12, c22]])
            C_v2, label, depth = elasticReduction(C_mat)
            c11_v2, c22_v2, c12_v2 = C_v2[0,0], C_v2[1,1], C_v2[0,1]
            x_stereo_v2, y_stereo_v2 = project_c_to_poincare(c11_v2, c22_v2, c12_v2)
            if x_stereo_v2**2 + y_stereo_v2**2 < 0.999**2:
                x_img_v2 = (x_stereo_v2 + 0.999) * (disc - 1) / (2 * 0.999)
                y_img_v2 = (y_stereo_v2 + 0.999) * (disc - 1) / (2 * 0.999)
                paths['Elastic_V2'].append((x_img_v2, y_img_v2))
        except Exception as e:
            print(f"  Warning: Elastic V2 failed at alpha={alpha}: {e}")

        # Comparison of Elastic V1 and V2
        if C_v1 is not None and C_v2 is not None:
            # We also get the original (before reduction) metrics from c11, c22, c12
            diff = np.abs(C_v1 - C_v2)
            comparison_data.append((alpha, c11, c22, c12, 
                                    C_v1[0,0], C_v1[1,1], C_v1[0,1], 
                                    C_v2[0,0], C_v2[1,1], C_v2[0,1],
                                    diff[0,0], diff[1,1], diff[0,1]))
    
    # Save comparison data to file
    if comparison_data:
        comp_file = os.path.join(output_folder, 'elastic_comparison.dat')
        with open(comp_file, 'w') as f_out:
            header = ("# alpha | "
                     "C11_orig C22_orig C12_orig | "
                     "C11_v1 C22_v1 C12_v1 | "
                     "C11_v2 C22_v2 C12_v2 | "
                     "diff11 diff22 diff12\n")
            f_out.write(header)
            for entry in comparison_data:
                # alpha + 3 orig + 3 v1 + 3 v2 + 3 diff = 13 values
                line = f"{entry[0]:12.6f} "
                line += f"{entry[1]:12.6f} {entry[2]:12.6f} {entry[3]:12.6f} "
                line += f"{entry[4]:12.6f} {entry[5]:12.6f} {entry[6]:12.6f} "
                line += f"{entry[7]:12.6f} {entry[8]:12.6f} {entry[9]:12.6f} "
                line += f"{entry[10]:12.8e} {entry[11]:12.8e} {entry[12]:12.8e}\n"
                f_out.write(line)
        print(f"Elastic reduction comparison saved to: {comp_file}")
    
    # Generate plot
    output_name = os.path.join(output_folder, f'poincare_path_theta_{int(theta_deg)}')
    poincare_plot_path(config, paths, output_name, pvmin, pvmax)
    
    print(f"Path visualization saved to: {output_name}.png")
    print(f"  Original path points: {len(paths['Original'])}")
    print(f"  Lagrange path points: {len(paths['Lagrange'])}")
    print(f"  Elastic V1 path points: {len(paths['Elastic_V1'])}")
    print(f"  Elastic V2 path points: {len(paths['Elastic_V2'])}")

def main():
    """Main execution function with mode selection."""
    print('PoincarÃ© Disk Analysis Tool')
    print('=' * 50)
    quick_shear_plot(alpha_min=-2.3, alpha_max=2.3, n_points=200)
    
    # Configuration - Change these as needed
    mode = 'standard'  # 'standard', 'batch', or 'path'
    folder_path = './triangle_data'  # Path to triangle files folder (for batch mode)
    output_folder = './figures'
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    lattice = 'square'  # or 'triangular'
    
    if mode == 'path':
        print('Running in PATH mode - shear deformation trajectory analysis')
        process_path_mode(alpha_min=-5.0, alpha_max=5.0, n_steps=400, theta_deg=28, output_folder=output_folder)
        
    elif mode == 'batch':
        print('Running in BATCH mode - processing triangle files')
        stability_file = "./stability_boundary_original.dat"
        
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} not found")
            process_triangle_files(folder_path, output_folder, lattice, stability_file=stability_file, start_file=None, n_stability_copies=5)
        else:
            start_time = time.time()
            process_triangle_files(folder_path, output_folder, lattice, stability_file=stability_file, start_file=None, n_stability_copies=5)
            end_time = time.time()
            print(f'\nBatch processing completed!')
            print(f'Output saved in: {output_folder}/')
            print(f'Total execution time: {end_time - start_time:.2f} seconds')
            
    elif mode == 'standard':
        print('Running in STANDARD mode - single analysis')
        main_standard()
    
    else:
        print(f"Unknown mode: {mode}")
        print("Please set mode to 'standard' or 'batch'")

if __name__ == "__main__":
    main()

