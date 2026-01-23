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
from matplotlib.colors import LogNorm
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

from .projections import Cij_from_stereographic_projection_tr, stereographic_projection_from_Cij_2D

from src.lagrange import lagrange_reduction, lagrange_reduction_with_matrix

from src.elastic_reduction_square import reduction_elastic_square
from src.elastic_reduction_v2 import elasticReduction

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

def compute_c_tensor_from_f(f_matrix, remove_vol_eff=True):
    """
    Compute C = F^T · F from deformation gradient.
    
    Parameters:
    -----------
    f_matrix : numpy.ndarray
        2x2 F matrix (deformation gradient)
    remove_vol_eff : bool, default True
        If True, divides F by its determinant to remove volume effects.
        
    Returns:
    --------
    c_matrix : numpy.ndarray
        2x2 C tensor (right Cauchy-Green deformation tensor)
    c11, c22, c12 : float
        Components of C tensor
    """
    if remove_vol_eff:
        det_f = np.linalg.det(f_matrix)
        if np.abs(det_f) > 1e-12:
            f_matrix = f_matrix / det_f
            
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


def plot_reference_lattice_points(ax, disc, project_c_to_poincare):
    """
    Plot reference lattice points on Poincaré disk.
    
    Parameters:
    -----------
    ax : matplotlib axes
        Axes object to plot on
    disc : int
        Discretization parameter for coordinate conversion
    project_c_to_poincare : function
        Function to project metric to Poincaré coordinates
    """
    H = np.eye(2) 
    H_t = np.transpose(H)
    
    # 1. Identity metric (square lattice)
    C11_id, C22_id, C12_id = 1.0, 1.0, 0.0
    cid_matrix = np.array([[C11_id, C12_id], [C12_id, C22_id]])
    ctr_id = H_t @ cid_matrix @ H
    x_stereo_id, y_stereo_id = project_c_to_poincare(ctr_id[0, 0], ctr_id[1, 1], ctr_id[0, 1])
    x_img_id = (x_stereo_id + 0.999) * (disc - 1) / (2 * 0.999)
    y_img_id = (y_stereo_id + 0.999) * (disc - 1) / (2 * 0.999)
    if x_stereo_id**2 + y_stereo_id**2 < 0.999**2:
        ax.plot(x_img_id, y_img_id, marker='s', markersize=12, color='red', 
                markeredgecolor='black', markeredgewidth=2, zorder=15)
    
    # 2. Triangular metric (positive C12)
    gamma = (4/3)**(1/4)
    C11_id, C22_id, C12_id = gamma**2, gamma**2, gamma**2/2
    cid_matrix = np.array([[C11_id, C12_id], [C12_id, C22_id]])
    ctr_id = H_t @ cid_matrix @ H
    x_stereo_id, y_stereo_id = project_c_to_poincare(ctr_id[0, 0], ctr_id[1, 1], ctr_id[0, 1])
    x_img_id = (x_stereo_id + 0.999) * (disc - 1) / (2 * 0.999)
    y_img_id = (y_stereo_id + 0.999) * (disc - 1) / (2 * 0.999)
    if x_stereo_id**2 + y_stereo_id**2 < 0.999**2:
        ax.plot(x_img_id, y_img_id, marker='^', markersize=12, color='green', 
                markeredgecolor='black', markeredgewidth=2, zorder=15)
    
    # 3. Triangular metric (negative C12)
    C11_id, C22_id, C12_id = gamma**2, gamma**2, -gamma**2/2
    cid_matrix = np.array([[C11_id, C12_id], [C12_id, C22_id]])
    ctr_id = H_t @ cid_matrix @ H
    x_stereo_id, y_stereo_id = project_c_to_poincare(ctr_id[0, 0], ctr_id[1, 1], ctr_id[0, 1])
    x_img_id = (x_stereo_id + 0.999) * (disc - 1) / (2 * 0.999)
    y_img_id = (y_stereo_id + 0.999) * (disc - 1) / (2 * 0.999)
    if x_stereo_id**2 + y_stereo_id**2 < 0.999**2:
        ax.plot(x_img_id, y_img_id, marker='^', markersize=12, color='blue', 
                markeredgecolor='black', markeredgewidth=2, zorder=15)


    # 4. Sheared metric (square lattice)
    C11_id, C22_id, C12_id = 1.0, 2.0, 1.0
    cid_matrix = np.array([[C11_id, C12_id], [C12_id, C22_id]])
    ctr_id = H_t @ cid_matrix @ H
    x_stereo_id, y_stereo_id = project_c_to_poincare(ctr_id[0, 0], ctr_id[1, 1], ctr_id[0, 1])
    x_img_id = (x_stereo_id + 0.999) * (disc - 1) / (2 * 0.999)
    y_img_id = (y_stereo_id + 0.999) * (disc - 1) / (2 * 0.999)
    if x_stereo_id**2 + y_stereo_id**2 < 0.999**2:
        ax.plot(x_img_id, y_img_id, marker='s', markersize=12, color='blue', 
                markeredgecolor='black', markeredgewidth=2, zorder=15)
    

    # 5. Sheared metric (square lattice)
    C11_id, C22_id, C12_id = 2.0, 1.0, 1.0
    cid_matrix = np.array([[C11_id, C12_id], [C12_id, C22_id]])
    ctr_id = H_t @ cid_matrix @ H
    x_stereo_id, y_stereo_id = project_c_to_poincare(ctr_id[0, 0], ctr_id[1, 1], ctr_id[0, 1])
    x_img_id = (x_stereo_id + 0.999) * (disc - 1) / (2 * 0.999)
    y_img_id = (y_stereo_id + 0.999) * (disc - 1) / (2 * 0.999)
    if x_stereo_id**2 + y_stereo_id**2 < 0.999**2:
        ax.plot(x_img_id, y_img_id, marker='s', markersize=12, color='blue', 
                markeredgecolor='black', markeredgewidth=2, zorder=15)



def poincare_plot_energy_with_f_matrices(config, f_matrices, name, pvmin, pvmax, lattice="square", plot_mode="scatter", stability_file=None, n_stability_copies=1):
    """
    Plot energy on Poincaré disk with F matrix points as scatter or density overlay.
    
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
    lattice : str
        Lattice type ('square' or 'triangular')
    plot_mode : str
        'scatter' for dots (default), 'density' for binned heatmap
    """
    fig = plt.figure(figsize=(12., 12.))
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = fig.add_subplot(111)
    
    disc = 1000
    colmap = matplotlib.cm.RdYlBu_r
    #colmap = matplotlib.cm.seismic
    
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
        H=np.eye(2)  # For square lattice, H is identity
        H = np.linalg.inv(H)
        H_t = np.transpose(H)
        
        for f_data in f_matrices:
            # Compute C tensor from F matrix
            c_matrix, c11, c22, c12 = compute_c_tensor_from_f(f_data['F'])
            
            # Use new elastic reduction v2
            Cr = np.array([[c11, c12], [c12, c22]])
            #csq, label, depth = elasticReduction(Cr) # elias
            csq = reduction_elastic_square(Cr)
            #csq = Cr
            
            # Apply centering transformation: ctr = H^T · C · H
            # csq is the reduced C matrix
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
        
        if x_points:
            if plot_mode == 'density':
                # Binned representation
                bins = 400
                # Use image coordinates limits: 0 to disc-1
                heatmap, xedges, yedges = np.histogram2d(x_points, y_points, bins=bins, 
                                                       range=[[0, disc], [0, disc]])
                
                # Mask zeros for transparency
                heatmap = np.ma.masked_where(heatmap == 0, heatmap)
                
                # Plot density overlay
                # Note: histogram2d returns H[x, y], pcolormesh needs X, Y, C
                # X, Y are edges.
                X, Y = np.meshgrid(xedges, yedges)
                
                # Plot with separate colormap (e.g., Greens or Purples to contrast with RdYlBu)
                # Use vmin=1 to ensure empty bins are transparent/bottom of scale (log(0) is undefined)
                # Let vmax auto-scale keying on the peak density of the current frame
                density_max = np.max(heatmap)
                if density_max > 0:
                   # Use LogNorm to make low-density regions visible
                   density_mesh = ax.pcolormesh(X, Y, heatmap.T, cmap='Greys', alpha=0.8, zorder=10, 
                                norm=LogNorm(vmin=1, vmax=density_max))
                else:
                   density_mesh = ax.pcolormesh(X, Y, heatmap.T, cmap='Greys', alpha=0.8, zorder=10)
                
            else:
                # Standard scatter plot
                ax.scatter(x_points, y_points, c='white', s=15, alpha=0.9, 
                          edgecolors='black', linewidth=0.5, zorder=10)
            
    #plot_reference_lattice_points(ax, disc, project_c_to_poincare)


    # Plot the edge of Poincaré disk
    circle_x = R * np.cos(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    circle_y = R * np.sin(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    plt.plot(circle_x, circle_y, 'k-', lw=1.1)
    
    # Add fundamental domain boundaries
    try:
        x_stereo = np.linspace(-.999, .999, num=disc, endpoint=True)
        y_stereo = np.linspace(-.999, .999, num=disc, endpoint=True)
        x_grid, y_grid = np.meshgrid(x_stereo, y_stereo)
        
        c11_grid, c22_grid, c12_grid, _, _, _ = Cij_from_stereographic_projection_tr(x_grid, y_grid, lattice)
        
        # Mask values outside the disk
        m_disk = x_grid**2 + y_grid**2 < 0.999**2
        c11_grid = np.where(m_disk, c11_grid, np.nan)
        c22_grid = np.where(m_disk, c22_grid, np.nan)
        c12_grid = np.where(m_disk, c12_grid, np.nan)
        
        # Convert coordinates to image coordinates
        x_img_grid = (x_grid + 0.999) * (disc - 1) / (2 * 0.999)
        y_img_grid = (y_grid + 0.999) * (disc - 1) / (2 * 0.999)
        
        # Center Line: C12 = 0
        ax.contour(x_img_grid, y_img_grid, c12_grid, levels=[0], 
                  colors=['black'], linewidths=1.0, alpha=0.9, linestyles='-')

        # Fundamental domain boundaries: Dsq = {2|C12| ≤ min(C11,C22)}
        min_c11_c22 = np.minimum(c11_grid, c22_grid)
        
        # Boundary: -2*C12 - min(C11,C22) = 0
        ax.contour(x_img_grid, y_img_grid, -2*c12_grid - min_c11_c22, levels=[0],
                        colors=['black'], linewidths=2.5, alpha=0.9, linestyles='-')
    
        # Boundary: 2*C12 - min(C11,C22) = 0
        ax.contour(x_img_grid, y_img_grid, 2*c12_grid - min_c11_c22, levels=[0],
                        colors=['black'], linewidths=2.5, alpha=0.9, linestyles='-')
    
    except Exception as e:
        print(f"Warning: Could not plot fundamental domain boundaries: {e}")

    # Plot stability boundary if provided
    if stability_file is not None:
        try:
            # Read the data file (skip header line starting with #)
            data = np.loadtxt(stability_file)
            c11_data = data[:, 0]
            c22_data = data[:, 1]
            c12_data = data[:, 2]

            # Apply stability criterion: only show the segment inside the fundamental domain
            stability_mask = 2*np.abs(c12_data) <= np.minimum(c11_data, c22_data)
            c11_data = c11_data[stability_mask]
            c22_data = c22_data[stability_mask]
            c12_data = c12_data[stability_mask]
            
            # Define transformation matrices for copies
            transformations = [
                (np.array([[1, 0], [0, 1]]), r'$\det \mathbf{Q} = 0$')          # Identity (original)
            ]
            
            if n_stability_copies > 1:
                # Add first 4 non-trivial copies as used in enhanced_plotting
                transformations.extend([
                    (np.array([[1, 1], [0, 1]]), "m=[1,1;0,1]"),
                    (np.array([[1, -1], [0, 1]]), "m=[1,-1;0,1]"),
                    (np.array([[1, 0], [1, 1]]), "m=[1,0;1,1]"),
                    (np.array([[1, 0], [-1, 1]]), "m=[1,0;-1,1]")
                ])
                # We limit to n_stability_copies total (including original)
                transformations = transformations[:n_stability_copies]

            # Store the original filtered data to apply transformations to
            c11_orig = c11_data.copy()
            c22_orig = c22_data.copy()
            c12_orig = c12_data.copy()

            # Helper to find connected segments and keep only the longest
            def find_longest_segment(x, y, threshold=0.1):
                if len(x) == 0: return [], []
                segments = []
                cur_x, cur_y = [x[0]], [y[0]]
                for i in range(1, len(x)):
                    if np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2) < threshold:
                        cur_x.append(x[i])
                        cur_y.append(y[i])
                    else:
                        segments.append((cur_x, cur_y))
                        cur_x, cur_y = [x[i]], [y[i]]
                segments.append((cur_x, cur_y))
                longest = max(segments, key=lambda s: len(s[0]))
                return np.array(longest[0]), np.array(longest[1])

            # Define color list for copies
            colors = ['red', 'darkblue', 'brown', 'green', 'orange']

            for idx, (mat, label) in enumerate(transformations):
                # Apply transformation: C' = mat^T @ C @ mat
                m11, m12 = mat[0, 0], mat[0, 1]
                m21, m22 = mat[1, 0], mat[1, 1]
                
                c11_t = m11**2 * c11_orig + 2*m11*m21*c12_orig + m21**2 * c22_orig
                c22_t = m12**2 * c11_orig + 2*m12*m22*c12_orig + m22**2 * c22_orig
                c12_t = m11*m12*c11_orig + (m11*m22 + m12*m21)*c12_orig + m21*m22*c22_orig
                
                # Convert (C11, C22, C12) to stereographic coordinates
                x_stereo_data, y_stereo_data = stereographic_projection_from_Cij_2D(c11_t, c22_t, c12_t)
                
                x_stereo_data, y_stereo_data = find_longest_segment(x_stereo_data, y_stereo_data)

                # Smooth the boundary points
                if len(x_stereo_data) >= 4:
                    try:
                        # Pre-smooth to remove noise before interpolation
                        x_stereo_smooth = gaussian_filter1d(x_stereo_data, sigma=2)
                        y_stereo_smooth = gaussian_filter1d(y_stereo_data, sigma=2)
                        
                        # Spline interpolation
                        tck, u = splprep([x_stereo_smooth, y_stereo_smooth], s=0, k=3)
                        u_new = np.linspace(0, 1, 1000)
                        x_stereo_data, y_stereo_data = splev(u_new, tck)
                    except Exception as e:
                        if idx == 0: print(f"Warning: Spline interpolation failed for {label}: {e}")

                # Convert to image coordinates
                x_img_data = (x_stereo_data + 0.999) * (disc - 1) / (2 * 0.999)
                y_img_data = (y_stereo_data + 0.999) * (disc - 1) / (2 * 0.999)
                
                # Plot the stability boundary with higher zorder
                color = colors[idx % len(colors)]
                alpha_val = 1.0 if idx == 0 else 0.7
                
                ax.plot(x_img_data, y_img_data, color=color, linestyle='-', linewidth=3.0, label=label, alpha=alpha_val, zorder=25)
            
            ax.legend(loc='lower left', fontsize=8, framealpha=0.5)
            
        except Exception as e:
            print(f"Warning: Could not plot stability boundary: {e}")
    
    # Add colorbar
    # Add colorbars as small insets in the top corners (outside the disk)
    
    # Background Energy (Top Left)
    # Position: [x, y, width, height] in axes coordinates
    cax_energy = ax.inset_axes([0.02, 0.95, 0.25, 0.015]) 
    cbar_energy = plt.colorbar(mappable=m, cax=cax_energy, orientation="horizontal")
    # cbar_energy.set_label("Energy", fontsize=8, labelpad=-25, y=1.5)
    cax_energy.text(0.5, 0.5, "Energy", transform=cax_energy.transAxes, 
                   ha='center', va='center', fontsize=7, fontweight='bold', color='black')
    cbar_energy.ax.xaxis.set_ticks_position('bottom')
    cbar_energy.ax.tick_params(labelsize=7)

    # Density (Top Right)
    if 'density_mesh' in locals():
        cax_density = ax.inset_axes([0.75, 0.95, 0.24, 0.015])
        cbar_density = plt.colorbar(mappable=density_mesh, cax=cax_density, orientation="horizontal")
        # cbar_density.set_label("Density", fontsize=8, labelpad=-25, y=1.5)
        cax_density.text(0.5, 0.5, "Density", transform=cax_density.transAxes, 
                       ha='center', va='center', fontsize=7, fontweight='bold', color='black')
        cbar_density.ax.xaxis.set_ticks_position('bottom')
        cbar_density.ax.tick_params(labelsize=7)
    
    ax.axis('off')
    #fig.savefig(name + '.png', bbox_inches='tight', dpi=300)
    fig.savefig(name + '.png', bbox_inches='tight', dpi=600, transparent=True)

    #fig.savefig(name + '.pdf', bbox_inches='tight', dpi=300)

    plt.close(fig)

def poincare_plot_energy_with_precise_boundaries(config, name, pvmin, pvmax, disc,stability_file=None,lattice="square"):
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
    c11_grid, c22_grid, c12_grid, c11t, c22t, c12t = Cij_from_stereographic_projection_tr(x_grid, y_grid,lattice)
    
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


    if stability_file is not None:
        try:
            # Read the data file (skip header line starting with #)
            data = np.loadtxt(stability_file)
            c11_data = data[:, 0]
            c22_data = data[:, 1]
            c12_data = data[:, 2]
            
            # Convert (C11, C22, C12) to stereographic coordinates
            # You need the inverse transformation here
            # This is a placeholder - you'll need to implement the inverse
            x_stereo_data, y_stereo_data = stereographic_projection_from_Cij_2D(c11_data, c22_data, c12_data)

            
            # Convert to image coordinates
            x_img_data = (x_stereo_data + 0.999) * (disc - 1) / (2 * 0.999)
            y_img_data = (y_stereo_data + 0.999) * (disc - 1) / (2 * 0.999)
            
            # Plot the stability boundary
            ax.plot(x_img_data, y_img_data, 'mo-', linewidth=2.5, markersize=3,
                   label='Stability boundary (det=0)', alpha=0.8)
            ax.legend(loc='upper right', fontsize=10)
            
        except Exception as e:
            print(f"Warning: Could not plot stability boundary: {e}")


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
def poincare_plot_path(config, c_paths, name, pvmin, pvmax, lattice="square"):
    """
    Plot energy on Poincaré disk with overlaid paths for different reductions.
    
    Parameters:
    -----------
    config : array_like
        Background energy configuration
    c_paths : dict
        Dictionary of paths, where keys are labels (e.g., 'Original', 'Lagrange') 
        and values are lists of (x_img, y_img) coordinates.
    name : str
        Output filename
    pvmin, pvmax : float
        Color scale limits
    lattice : str
        Lattice type
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
    
    # Define colors for different paths
    path_colors = {
        'Original': 'white',
        'Lagrange': 'black',
        'Elastic_V1': 'red',
        'Elastic_V2': 'cyan'
    }
    
    path_styles = {
        'Original': '-',
        'Lagrange': '--',
        'Elastic_V1': '--',  # Standard dashed
        'Elastic_V2': ':'    # Dotted (different density)
    }
    
    path_widths = {
        'Original': 2.5,
        'Lagrange': 2.5,
        'Elastic_V1': 1.0,
        'Elastic_V2': 3.0
    }

    # Plot paths
    if c_paths:
        for label, points in c_paths.items():
            if not points:
                continue
                
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            
            color = path_colors.get(label, 'red')
            
            # Use scatter for elastic reductions, lines for others
            if label in ['Elastic_V1', 'Elastic_V2']:
                # Default settings
                marker = 'o'
                edge_color = 'black'
                size = 12
                alpha_val = 0.9
                
                if label == 'Elastic_V1':
                    marker = 'o'
                    edge_color = 'red'
                    size = 30        # Significantly larger
                    alpha_val = 0.6   # More transparent to act as a background/halo
                
                elif label == 'Elastic_V2':
                    marker = 's'
                    edge_color = 'blue'
                    size = 15         # Smaller to sit inside the triangle
                    alpha_val = .5   # Fully opaque to stand out
                
                # Plotting with the specific conditional values
                ax.scatter(x_vals, y_vals, 
                        facecolors='none', 
                        edgecolors=edge_color, 
                        marker=marker, 
                        s=size, 
                        alpha=alpha_val, 
                        linewidths=2,
                        label=label.replace('_', ' '), 
                        zorder=10)


            else:
                # Plot as line for Original and Lagrange
                style = path_styles.get(label, '-')
                width = path_widths.get(label, 2.5)
                
                ax.plot(x_vals, y_vals, color=color, linestyle=style, 
                       linewidth=width, label=label.replace('_', ' '), alpha=0.9, zorder=5)
                
                # Add start and end markers for line paths
                if len(x_vals) > 0:
                    ax.plot(x_vals[0], y_vals[0], 'o', color=color, markersize=10, 
                           markeredgecolor='black', markeredgewidth=0.5, zorder=6)
                    ax.plot(x_vals[-1], y_vals[-1], 's', color=color, markersize=10, 
                           markeredgecolor='black', markeredgewidth=0.5, zorder=6)

    # Plot the edge of Poincaré disk
    circle_x = R * np.cos(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    circle_y = R * np.sin(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    plt.plot(circle_x, circle_y, 'k-', lw=1.1)

    # Add legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add colorbars (Corner Insets as per recent style)
    # Background Energy (Top Left)
    cax_energy = ax.inset_axes([0.02, 0.95, 0.25, 0.015]) 
    cbar_energy = plt.colorbar(mappable=m, cax=cax_energy, orientation="horizontal")
    # cbar_energy.set_label("Energy", fontsize=8, labelpad=-25, y=1.5)
    cax_energy.text(0.5, 0.5, "Energy", transform=cax_energy.transAxes, 
                   ha='center', va='center', fontsize=7, fontweight='bold', color='black')
    cbar_energy.ax.xaxis.set_ticks_position('bottom')
    cbar_energy.ax.tick_params(labelsize=7)
    
    ax.axis('off')
    fig.savefig(name + '.png', bbox_inches='tight', dpi=300)
    fig.savefig(name + '.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
