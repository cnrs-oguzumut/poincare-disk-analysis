#!/usr/bin/env python3
"""
Main script for Poincaré disk analysis with centered metrics.

This script performs the complete workflow:
1. Generate stereographic projection coordinates
2. Apply centering transformation
3. Perform Lagrange reduction
4. Compute strain energy density
5. Visualize results with fundamental domain boundaries
"""

import time
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from src.projections import Cij_from_stereographic_projection_tr
from src.lagrange import lagrange_reduction
from src.energy import interatomic_phi0_from_Cij, convert_data_SymLog
from src.plotting import poincare_plot_energy_with_precise_boundaries

def main():
    """Main execution function."""
    print('Lagrange reduction and Poincaré disk projection with centered metrics')
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs('figures', exist_ok=True)
    
    # Parameters
    disc = 1000  # Poincaré disk discretization
    lattice = 'square'  # or 'triangular'
    
    # Generate stereographic projection coordinates
    print("Generating stereographic projection coordinates...")
    x, y = np.linspace(-.999, .999, num=disc, endpoint=True), np.linspace(-.999, .999, num=disc, endpoint=True)
    x_, y_ = np.meshgrid(x, y)
    
    # Mask values outside Poincaré disk
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
    phi0 = interatomic_phi0_from_Cij(c11_reduced, c22_reduced, c12_reduced, lattice)
    
    # Normalize energy data
    phi0_normalized = (phi0 - np.nanmin(phi0)) / (np.nanmax(phi0) - np.nanmin(phi0))
    
    # Convert to symmetric log scale
    c_scale = 1e-19  # for interatomic phi0
    config = convert_data_SymLog(phi0_normalized, c_scale)
    
    # Generate visualization
    print("Creating visualization...")
    saving_index = 1
    output_name = f'./figures/poincare_disk_analysis_{saving_index}'
    
    poincare_plot_energy_with_precise_boundaries(
        config, 
        output_name, 
        np.nanmin(config), 
        0.8 * np.nanmax(config)
    )
    
    end_time = time.time()
    print(f'Analysis completed successfully!')
    print(f'Output saved as: {output_name}.pdf')
    print(f'Total execution time: {end_time - start_time:.4f} seconds')

if __name__ == "__main__":
    main()