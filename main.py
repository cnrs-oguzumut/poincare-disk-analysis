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

from src.projections import Cij_from_stereographic_projection_tr
from src.lagrange import lagrange_reduction
from src.energy import interatomic_phi0_from_Cij, conti_phi0_from_Cij,convert_data_SymLog
from src.plotting import (poincare_plot_energy_with_precise_boundaries, 
                          poincare_plot_energy_with_f_matrices,
                          read_f_matrix_from_file)
from src.visualization_3d import create_3d_energy_surface, make_browser_interactive_plots

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
    x, y = np.linspace(-.999, .999, num=disc, endpoint=True), np.linspace(-.999, .999, num=disc, endpoint=True)
    x_, y_ = np.meshgrid(x, y)
    
    # Mask values outside Poincaré disk
    x_ = np.where(x_**2 + y_**2 - (.999)**2 >= 1.e-6, np.nan, x_)
    y_ = np.where(x_**2 + y_**2 - (.999)**2 >= 1.e-6, np.nan, y_)
    
    # Apply stereographic projection with centering transformation
    c11, c22, c12, c11t, c22t, c12t = Cij_from_stereographic_projection_tr(x_, y_)
    
    # Perform Lagrange reduction on centered metrics
    c11_reduced, c22_reduced, c12_reduced, iterations = lagrange_reduction(c11, c22, c12, verbose=False)
    
    # Compute strain energy density
    phi0 = interatomic_phi0_from_Cij(c11_reduced, c22_reduced, c12_reduced, lattice)
    
    # Normalize energy data
    phi0_normalized = (phi0 - np.nanmin(phi0)) / (np.nanmax(phi0) - np.nanmin(phi0))
    
    # Convert to symmetric log scale
    c_scale = 1e-19  # for interatomic phi0
    config = convert_data_SymLog(phi0_normalized, c_scale)
    
    pvmin = np.nanmin(config)
    pvmax = 0.8 * np.nanmax(config)
    
    return config, pvmin, pvmax

def process_triangle_files(folder_path, output_folder='figures', lattice='square'):
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
    print(f'Batch processing triangle files from: {folder_path}')
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all triangle files
    pattern = os.path.join(folder_path, 'triangles_*.dat')
    triangle_files = sorted(glob.glob(pattern))
    
    if not triangle_files:
        print(f"No triangles_*.dat files found in {folder_path}")
        return
    
    print(f"Found {len(triangle_files)} triangle files to process")
    
    # Generate background energy configuration once
    config, pvmin, pvmax = generate_background_energy(lattice=lattice)
    
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
        poincare_plot_energy_with_f_matrices(config, f_matrices, output_name, pvmin, pvmax)
        print(f"  Saved: {output_name}.png")

def main_standard():
    """Standard analysis workflow (original functionality)."""
    print('Lagrange reduction and PoincarÃ© disk projection with centered metrics')
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs('figures', exist_ok=True)
    
    # Parameters
    disc = 2000  # Poincaré disk discretization
    lattice = 'square'  # or 'triangular'
    
    # Generate stereographic projection coordinates
    print("Generating stereographic projection coordinates...")
    x, y = np.linspace(-.999, .999, num=disc, endpoint=True), np.linspace(-.999, .999, num=disc, endpoint=True)
    x_, y_ = np.meshgrid(x, y)
    
    # Mask values outside PoincarÃ© disk
    x_ = np.where(x_**2 + y_**2 - (.999)**2 >= 1.e-6, np.nan, x_)
    y_ = np.where(x_**2 + y_**2 - (.999)**2 >= 1.e-6, np.nan, y_)
    
    # Apply stereographic projection with centering transformation
    print("Applying stereographic projection with centering transformation...")
    c11, c22, c12, c11t, c22t, c12t = Cij_from_stereographic_projection_tr(x_, y_)
    
    # Perform Lagrange reduction on centered metrics
    print("Performing Lagrange reduction...")
    c11_reduced, c22_reduced, c12_reduced, iterations = lagrange_reduction(c11, c22, c12)
    print(f"Lagrange reduction completed after {iterations} iterations")
    
    # Compute strain energy density
    print("Computing strain energy density...")
    phi0 = conti_phi0_from_Cij(c11_reduced, c22_reduced, c12_reduced, lattice)
    print(f"Energy range 1: {np.nanmin(phi0):.2e} to {np.nanmax(phi0):.2e}")

    
    # Normalize energy data
    phi0_normalized = (phi0 - np.nanmin(phi0)) / (np.nanmax(phi0) - np.nanmin(phi0))
    phi0_normalized = (phi0 - np.nanmin(phi0)) 
    
    # Convert to symmetric log scale
    c_scale = 1e-19  # for interatomic phi0
    c_scale = 1e-2  # for conti phi0

    config = convert_data_SymLog(phi0_normalized, c_scale)
    print(f"Energy range 2: {np.nanmin(phi0_normalized):.2e} to {np.nanmax(phi0_normalized):.2e}")
    #exit(0)
    
    #create_3d_interactive_energy_surface(x_, y_, phi0_normalized, 'square', 'file.pdf')
    make_browser_interactive_plots(x_, y_, phi0_normalized, phi0_normalized, view_radius=1.)

    #create_3d_energy_surface(x_, y_, phi0_normalized, 'square', 'figures/3d_square_lattice.pdf')
    #make_browser_interactive_plots(x_, y_, config, config)
    #make_browser_interactive_plots(x_, y_, phi0_normalized, phi0_normalized, view_radius=0.3)
    
    # Generate visualization
    print("Creating visualization...")
    saving_index = 1
    output_name = f'./figures/poincare_disk_analysis_{saving_index}'
    
    poincare_plot_energy_with_precise_boundaries(
        config, 
        output_name, 
        np.nanmin(config), 
        0.8 * np.nanmax(config),
        disc
    )
    
    end_time = time.time()
    print(f'Analysis completed successfully!')
    print(f'Output saved as: {output_name}.pdf')
    print(f'Total execution time: {end_time - start_time:.4f} seconds')

def main():
    """Main execution function with mode selection."""
    print('PoincarÃ© Disk Analysis Tool')
    print('=' * 50)
    
    # Configuration - Change these as needed
    mode = 'standard'  # 'standard' or 'batch'
    folder_path = '/Users/usalman/programming/FEM_2D/factorized/lattice_triangulation/simulation_test2/triangle_data'  # Path to triangle files folder
    output_folder = './figures'
    lattice = 'square'  # or 'triangular'
    
    if mode == 'batch':
        print('Running in BATCH mode - processing triangle files')
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} does not exist")
            print("Please update the folder_path variable or create the folder")
            print("Switching to standard mode...")
            main_standard()
        else:
            start_time = time.time()
            process_triangle_files(folder_path, output_folder, lattice)
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