#!/usr/bin/env python3
"""
Basic usage example for Poincaré disk analysis.

This example demonstrates the core functionality with minimal parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.projections import Cij_from_stereographic_projection_tr
from src.lagrange import lagrange_reduction
from src.energy import interatomic_phi0_from_Cij, convert_data_SymLog
from src.plotting import poincare_plot_energy_with_precise_boundaries

def basic_example():
    """Run a basic analysis example."""
    print("Running basic Poincaré disk analysis example...")
    
    # Create output directory
    os.makedirs('examples/output', exist_ok=True)
    
    # Small grid for quick example
    disc = 500
    
    # Generate coordinates
    x, y = np.linspace(-.999, .999, num=disc, endpoint=True), \
           np.linspace(-.999, .999, num=disc, endpoint=True)
    x_, y_ = np.meshgrid(x, y)
    
    # Mask outside disk
    x_ = np.where(x_**2 + y_**2 - (.999)**2 >= 1.e-6, np.nan, x_)
    y_ = np.where(x_**2 + y_**2 - (.999)**2 >= 1.e-6, np.nan, y_)
    
    # Apply projections
    c11, c22, c12, c11t, c22t, c12t = Cij_from_stereographic_projection_tr(x_, y_)
    
    # Lagrange reduction
    c11_red, c22_red, c12_red, iterations = lagrange_reduction(c11, c22, c12, verbose=False)
    print(f"Completed Lagrange reduction in {iterations} iterations")
    
    # Compute energy
    phi0 = interatomic_phi0_from_Cij(c11_red, c22_red, c12_red, 'square')
    phi0_norm = (phi0 - np.nanmin(phi0)) / (np.nanmax(phi0) - np.nanmin(phi0))
    config = convert_data_SymLog(phi0_norm, 1e-19)
    
    # Create plot
    output_name = 'examples/output/basic_example'
    poincare_plot_energy_with_precise_boundaries(
        config, output_name, np.nanmin(config), 0.8 * np.nanmax(config)
    )
    
    print(f"Example completed! Output saved as {output_name}.pdf")

if __name__ == "__main__":
    basic_example()