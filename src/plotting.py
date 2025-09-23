"""
Visualization functions for PoincarÃ© disk analysis.

This module contains all plotting functions for visualizing energy landscapes
and fundamental domain boundaries on the PoincarÃ© disk.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .projections import Cij_from_stereographic_projection_tr, stereographic_projection_from_Cij_2D

def read_f_matrix_from_file(filename):
    """
    Read F matrix components from C++ triangle output file.
    
    Parameters:
    -----------
    filename : str
        Path to the triangle data file
        
    Returns:
    --------
    f_matrices : list of dict
        List containing F matrix data for each element
    """
    f_matrices = []
    
    try:
        with open(filename, 'r') as file:
            for line_num, line in enumerate(file, 1):
                # Skip comment lines starting with #
                if line.strip().startswith('#'):
                    continue
                
                # Skip empty lines
                if not line.strip():
                    continue
                    
                parts = line.strip().split()
                if len(parts) >= 5:  # At least elem_idx + 4 F components
                    try:
                        elem_idx = int(parts[0])
                        f11 = float(parts[1])
                        f12 = float(parts[2])
                        f21 = float(parts[3])
                        f22 = float(parts[4])
                        
                        # Store as matrix and additional info
                        f_matrix = np.array([[f11, f12], [f21, f22]])
                        f_matrices.append({
                            'elem_idx': elem_idx,
                            'F': f_matrix,
                            'F11': f11, 'F12': f12, 'F21': f21, 'F22': f22
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line {line_num} in {filename}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return []
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []
    
    return f_matrices

def compute_c_tensor_from_f(f_matrix):
    """
    Compute C = F^T · F from deformation gradient.
    
    Parameters:
    -----------
    f_matrix : numpy.ndarray
        2x2 F matrix (deformation gradient)
        
    Returns:
    --------
    c_matrix : numpy.ndarray
        2x2 C tensor (right Cauchy-Green deformation tensor)
    c11, c22, c12 : float
        Components of C tensor
    """
    # Compute right Cauchy-Green deformation tensor: C = F^T · F
    F_transpose = f_matrix.T
    c_matrix = F_transpose @ f_matrix
    
    # Extract components
    c11 = c_matrix[0, 0]
    c12 = c_matrix[0, 1]  # Note: C12 = C21 since C is symmetric
    c22 = c_matrix[1, 1]
    
    return c_matrix, c11, c22, c12

def project_c_to_poincare(c11, c22, c12):
    """
    Project C tensor components to Poincaré disk coordinates.
    
    Parameters:
    -----------
    c11, c22, c12 : float
        C tensor components
        
    Returns:
    --------
    x_stereo, y_stereo : float
        Stereographic coordinates for Poincaré disk
    """
    try:
        # Use inverse stereographic projection
        x_stereo, y_stereo = stereographic_projection_from_Cij_2D(c11, c22, c12)
        
        # Ensure coordinates stay within valid range for disk
        radius = np.sqrt(x_stereo**2 + y_stereo**2)
        if radius >= 0.999:
            scale_factor = 0.998 / radius
            x_stereo *= scale_factor
            y_stereo *= scale_factor
            
    except (ValueError, RuntimeError):
        # Handle degenerate cases
        x_stereo, y_stereo = 0.0, 0.0
    
    return x_stereo, y_stereo

def poincare_plot_energy_with_f_matrices(config, f_matrices, name, pvmin, pvmax):
    """
    Plot energy on Poincaré disk with F matrix points as scatter overlay.
    
    Parameters:
    -----------
    config : array_like
        Background energy configuration
    f_matrices : list
        F matrix data from read_f_matrix_from_file()
    name : str
        Output filename (without extension)
    pvmin, pvmax : float
        Color scale limits
    """
    fig = plt.figure(figsize=(12., 12.))
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = fig.add_subplot(111)
    
    disc = 1000
    colmap = matplotlib.cm.RdYlBu_r
    
    # Plot background energy
    m = ax.imshow(config, origin='lower', interpolation='none',
                  cmap=colmap, vmin=pvmin, vmax=pvmax)
    
    R = 0.5 * disc
    
    # Process F matrices and convert to scatter points
    if f_matrices:
        x_points = []
        y_points = []
        
        # Define the same H matrix used in projections.py
        gamma = (4/3)**(1/4)
        H = gamma * np.array([[np.sqrt(2 + np.sqrt(3))/2, np.sqrt(2 - np.sqrt(3))/2], 
                             [np.sqrt(2 - np.sqrt(3))/2, np.sqrt(2 + np.sqrt(3))/2]])
        H = np.linalg.inv(H)
        H_t = np.transpose(H)
        
        for f_data in f_matrices:
            # Compute C tensor from F matrix
            c_matrix, c11, c22, c12 = compute_c_tensor_from_f(f_data['F'])
            
            # Apply centering transformation: ctr = H^T · C · H
            csq = np.array([[c11, c12], [c12, c22]])
            ctr = H_t @ csq @ H
            c11_centered = ctr[0, 0]
            c22_centered = ctr[1, 1] 
            c12_centered = ctr[0, 1]
            
            # Project to Poincaré coordinates (using centered metrics)
            x_stereo, y_stereo = project_c_to_poincare(c11_centered, c22_centered, c12_centered)
            
            # Convert to image coordinates
            x_img = (x_stereo + 0.999) * (disc - 1) / (2 * 0.999)
            y_img = (y_stereo + 0.999) * (disc - 1) / (2 * 0.999)
            
            # Check if point is within the disk
            if x_stereo**2 + y_stereo**2 < 0.999**2:
                x_points.append(x_img)
                y_points.append(y_img)
        
        # Plot F matrix points as scatter
        if x_points:
            ax.scatter(x_points, y_points, c='white', s=15, alpha=0.9, 
                      edgecolors='black', linewidth=0.5, zorder=10)

    # Plot the edge of Poincaré disk
    circle_x = R * np.cos(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    circle_y = R * np.sin(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    plt.plot(circle_x, circle_y, 'k-', lw=1.1)
    
    # Add fundamental domain boundaries
    try:
        x_stereo = np.linspace(-.999, .999, num=disc, endpoint=True)
        y_stereo = np.linspace(-.999, .999, num=disc, endpoint=True)
        x_grid, y_grid = np.meshgrid(x_stereo, y_stereo)
        
        c11_grid, c22_grid, c12_grid, _, _, _ = Cij_from_stereographic_projection_tr(x_grid, y_grid)
        
        # Mask values outside the disk
        mask = x_grid**2 + y_grid**2 < 0.999**2
        c11_grid = np.where(mask, c11_grid, np.nan)
        c22_grid = np.where(mask, c22_grid, np.nan)
        c12_grid = np.where(mask, c12_grid, np.nan)
        
        # Convert coordinates to image coordinates
        x_img_grid = (x_grid + 0.999) * (disc - 1) / (2 * 0.999)
        y_img_grid = (y_grid + 0.999) * (disc - 1) / (2 * 0.999)
        
        # Plot boundaries
        ax.contour(x_img_grid, y_img_grid, c12_grid, levels=[0], 
                  colors=['brown'], linewidths=1, alpha=0.7)
        ax.contour(x_img_grid, y_img_grid, c12_grid - c11_grid, levels=[0], 
                  colors=['black'], linewidths=1, alpha=0.7)
        ax.contour(x_img_grid, y_img_grid, c12_grid - c22_grid, levels=[0], 
                  colors=['gray'], linewidths=1, alpha=0.7)
        
        # Square lattice fundamental domain boundaries: Dsq = {2|C12| ≤ min(C11,C22)}
        # Calculate min(C11, C22)
        min_c11_c22 = np.minimum(c11_grid, c22_grid)
        
        # Plot 2|C12| = min(C11,C22), which gives us two boundaries:
        # 2*C12 = min(C11,C22) and -2*C12 = min(C11,C22)
        
        # Boundary: 2*C12 - min(C11,C22) = 0
        ax.contour(x_img_grid, y_img_grid, 2*c12_grid - min_c11_c22, levels=[0],
                colors=['black'], linewidths=1.5, alpha=0.8, linestyles='--')
        
        # Boundary: -2*C12 - min(C11,C22) = 0 (equivalent to 2*C12 + min(C11,C22) = 0)
        ax.contour(x_img_grid, y_img_grid, -2*c12_grid - min_c11_c22, levels=[0],
                colors=['black'], linewidths=1.5, alpha=0.8, linestyles='--')
    except:
        pass
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.75%", pad=0.1)
    plt.colorbar(mappable=m, cax=cax)
    
    ax.axis('off')
    fig.savefig(name + '.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

def poincare_plot_energy_with_precise_boundaries(config, name, pvmin, pvmax, disc):
    """
    Plot energy on PoincarÃ© disk with precise fundamental domain boundaries.
    
    Uses contour lines for accurate boundary visualization with centered labels.
    
    Parameters:
    -----------
    config : array_like
        Energy configuration data (2D array)
    name : str
        Output filename (without extension)
    pvmin, pvmax : float
        Color scale limits
    """
    fig = plt.figure(figsize=(12., 12.))
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = fig.add_subplot(111)
    
    colmap = matplotlib.cm.RdYlBu_r
    
    m = ax.imshow(config, origin='lower', interpolation='none', 
                 cmap=colmap, vmin=pvmin, vmax=pvmax)
    R = 0.5 * disc
    
    # Plot the edge of PoincarÃ© disk
    circle_x = R * np.cos(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    circle_y = R * np.sin(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    plt.plot(circle_x, circle_y, 'k-', lw=1.1)
    
    # Create grid for contour calculation
    x_stereo = np.linspace(-.999, .999, num=disc, endpoint=True)
    y_stereo = np.linspace(-.999, .999, num=disc, endpoint=True)
    x_grid, y_grid = np.meshgrid(x_stereo, y_stereo)
    
    # Convert to metric components using centering transformation
    c11_grid, c22_grid, c12_grid, c11t, c22t, c12t = Cij_from_stereographic_projection_tr(x_grid, y_grid)
    
    # Mask values outside the disk
    mask = x_grid**2 + y_grid**2 < 0.999**2
    c11_grid = np.where(mask, c11_grid, np.nan)
    c22_grid = np.where(mask, c22_grid, np.nan)
    c12_grid = np.where(mask, c12_grid, np.nan)
    
    # Convert coordinates to image coordinates (0 to disc-1)
    x_img = (x_grid + 0.999) * (disc - 1) / (2 * 0.999)
    y_img = (y_grid + 0.999) * (disc - 1) / (2 * 0.999)
    
    # Plot boundary Câ‚â‚‚ = 0
    try:
        contour1 = ax.contour(x_img, y_img, c12_grid, levels=[0], 
                             colors=['red'], linewidths=2)
        path = contour1.collections[0].get_paths()[0]
        vertices = path.vertices
        mid_idx = len(vertices) // 2
        x_pos, y_pos = vertices[mid_idx]
        ax.clabel(contour1, manual=[(x_pos, y_pos)], inline=True, 
                 fontsize=10, fmt='$C_{12}=0$')
    except:
        pass
    
    # Plot boundary Câ‚â‚‚ = Câ‚â‚
    try:
        diff_c11 = c12_grid - c11_grid
        contour2 = ax.contour(x_img, y_img, diff_c11, levels=[0], 
                             colors=['yellow'], linewidths=2)
        path = contour2.collections[0].get_paths()[0]
        vertices = path.vertices
        mid_idx = len(vertices) // 2
        x_pos, y_pos = vertices[mid_idx]
        ax.clabel(contour2, manual=[(x_pos, y_pos)], inline=True, 
                 fontsize=10, fmt='$C_{12}=C_{11}$')
    except:
        pass
    
    # Plot boundary Câ‚â‚‚ = Câ‚‚â‚‚
    try:
        diff_c22 = c12_grid - c22_grid
        contour3 = ax.contour(x_img, y_img, diff_c22, levels=[0], 
                             colors=['gray'], linewidths=2)
        path = contour3.collections[0].get_paths()[0]
        vertices = path.vertices
        mid_idx = len(vertices) // 2
        x_pos, y_pos = vertices[mid_idx]
        ax.clabel(contour3, manual=[(x_pos, y_pos)], inline=True, 
                 fontsize=10, fmt='$C_{12}=C_{22}$')
    except:
        pass
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.75%", pad=0.1)
    plt.colorbar(mappable=m, cax=cax)
    ax.axis('off')
    
    fig.savefig(name + '.pdf', bbox_inches='tight')
    plt.close(fig)

def poincare_plot_energy(config, name, pvmin, pvmax):
    """
    Basic energy plot on PoincarÃ© disk without boundaries.
    
    Parameters:
    -----------
    config : array_like
        Energy configuration data
    name : str
        Output filename (without extension)
    pvmin, pvmax : float
        Color scale limits
    """
    fig = plt.figure(figsize=(12., 12.))
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = fig.add_subplot(111)
    
    disc = 1000
    colmap = matplotlib.cm.RdYlBu_r
    m = ax.imshow(config, origin='lower', interpolation='none', 
                 cmap=colmap, vmin=pvmin, vmax=pvmax)
    R = 0.5 * disc
    
    # Plot disk boundary
    circle_x = R * np.cos(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    circle_y = R * np.sin(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    plt.plot(circle_x, circle_y, 'k-', lw=1.1)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.75%", pad=0.1)
    plt.colorbar(mappable=m, cax=cax)
    ax.axis('off')
    
    fig.savefig(name + '.pdf', bbox_inches='tight')
    plt.close(fig)

def poincare_plot_scatter(config, px, py, name, pvmin, pvmax):
    """
    Plot energy with scatter points overlay.
    
    Parameters:
    -----------
    config : array_like
        Background energy configuration
    px, py : array_like
        Scatter point coordinates
    name : str
        Output filename
    pvmin, pvmax : float
        Color scale limits
    """
    fig = plt.figure(figsize=(12., 12.))
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = fig.add_subplot(111)
    
    disc = 1000
    colmap = matplotlib.cm.RdYlBu_r
    m = ax.imshow(config, origin='lower', interpolation='none', 
                 cmap=colmap, vmin=pvmin, vmax=pvmax)
    R = 0.5 * disc
    
    # Plot scatter points
    axx = R * px + R - 1
    ayy = R * py + R - 1
    ax.plot(axx, ayy, marker='o', color='w', markeredgecolor='k', 
           markersize=.5, lw=0, linestyle='None')
    
    # Plot disk boundary
    circle_x = R * np.cos(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    circle_y = R * np.sin(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    plt.plot(circle_x, circle_y, 'k-', lw=1.1)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.75%", pad=0.1)
    plt.colorbar(mappable=m, cax=cax)
    ax.axis('off')
    
    fig.savefig(name + '.pdf', bbox_inches='tight')
    plt.close(fig)