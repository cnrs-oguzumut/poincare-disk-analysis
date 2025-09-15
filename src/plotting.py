"""
Visualization functions for Poincaré disk analysis.

This module contains all plotting functions for visualizing energy landscapes
and fundamental domain boundaries on the Poincaré disk.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .projections import Cij_from_stereographic_projection_tr

def poincare_plot_energy_with_precise_boundaries(config, name, pvmin, pvmax):
    """
    Plot energy on Poincaré disk with precise fundamental domain boundaries.
    
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
    
    disc = 1000
    colmap = matplotlib.cm.RdYlBu_r
    m = ax.imshow(config, origin='lower', interpolation='none', 
                 cmap=colmap, vmin=pvmin, vmax=pvmax)
    R = 0.5 * disc
    
    # Plot the edge of Poincaré disk
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
    
    # Plot boundary C₁₂ = 0
    try:
        contour1 = ax.contour(x_img, y_img, c12_grid, levels=[0], 
                             colors=['black'], linewidths=2)
        path = contour1.collections[0].get_paths()[0]
        vertices = path.vertices
        mid_idx = len(vertices) // 2
        x_pos, y_pos = vertices[mid_idx]
        ax.clabel(contour1, manual=[(x_pos, y_pos)], inline=True, 
                 fontsize=10, fmt='$C_{12}=0$')
    except:
        pass
    
    # Plot boundary C₁₂ = C₁₁
    try:
        diff_c11 = c12_grid - c11_grid
        contour2 = ax.contour(x_img, y_img, diff_c11, levels=[0], 
                             colors=['black'], linewidths=2)
        path = contour2.collections[0].get_paths()[0]
        vertices = path.vertices
        mid_idx = len(vertices) // 2
        x_pos, y_pos = vertices[mid_idx]
        ax.clabel(contour2, manual=[(x_pos, y_pos)], inline=True, 
                 fontsize=10, fmt='$C_{12}=C_{11}$')
    except:
        pass
    
    # Plot boundary C₁₂ = C₂₂
    try:
        diff_c22 = c12_grid - c22_grid
        contour3 = ax.contour(x_img, y_img, diff_c22, levels=[0], 
                             colors=['black'], linewidths=2)
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
    Basic energy plot on Poincaré disk without boundaries.
    
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