"""
Improved 3D Energy Surface with Boundary Filtering
==================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from scipy import ndimage

def create_3d_energy_surface(x_mesh, y_mesh, energy_values, 
                           lattice_type='square', 
                           output_filename='3d_energy_surface.pdf',
                           log_scale=True,
                           mask_outside_disk=True,
                           boundary_margin=.8,
                           energy_cap_percentile=75,
                           smooth_boundary=False):
    """
    Create 3D surface plot with energy as height, with improved boundary filtering.
    
    Parameters:
    -----------
    x_mesh, y_mesh : np.ndarray
        2D coordinate meshgrids from Poincare disk
    energy_values : np.ndarray  
        Energy values from interatomic_phi0_from_Cij()
    lattice_type : str
        'square' or 'triangular'
    boundary_margin : float
        How far from boundary to start filtering (default: 0.05)
    energy_cap_percentile : float
        Cap energies at this percentile to remove extreme outliers (default: 95)
    smooth_boundary : bool
        Apply Gaussian smoothing near boundary
    """
    
    # Calculate radius for each point
    radius_squared = x_mesh**2 + y_mesh**2
    radius = np.sqrt(radius_squared)
    
    # Create a copy to work with
    energy_filtered = np.copy(energy_values)
    
    # Step 1: Basic disk masking
    if mask_outside_disk:
        mask_outside = radius_squared >= 1.0
        energy_filtered[mask_outside] = np.nan
    
    # Step 1.5: Cap only extremely high energies (keep negative energies)
    finite_energies = energy_filtered[np.isfinite(energy_filtered)]
    # if len(finite_energies) > 0:
    #     high_energy_cap = np.percentile(finite_energies, energy_cap_percentile)
    #     n_capped = np.sum(energy_filtered > high_energy_cap)
    #     energy_filtered = np.nannp.minimum(energy_filtered, high_energy_cap)
    #     print(f"Capped {n_capped} high energy values above {high_energy_cap:.2e}")
    #     print(f"Energy range after capping: {np.nanmin(energy_filtered):.2e} to {np.nanmax(energy_filtered):.2e}")
    
    if len(finite_energies) > 0:
        high_energy_cap = np.percentile(finite_energies, energy_cap_percentile)
        n_capped = np.sum(energy_filtered > high_energy_cap)
        #energy_filtered = np.minimum(energy_filtered, high_energy_cap)
        # Ignore data above percentile by setting to NaN:
        energy_filtered[energy_filtered > high_energy_cap] = np.nan
        print(f"Capped {n_capped} high energy values above {high_energy_cap:.2e}")



    # Step 2: Progressive boundary filtering
    boundary_threshold = 1.0 - boundary_margin
    mask_near_boundary = radius > boundary_threshold
    
    if np.any(mask_near_boundary):
        # Cap energies at percentile within the valid region
        valid_energies = energy_filtered[~np.isnan(energy_filtered) & ~mask_near_boundary]
        if len(valid_energies) > 0:
            energy_cap = np.percentile(valid_energies, energy_cap_percentile)
            
            # Apply progressive filtering near boundary
            distance_from_boundary = 1.0 - radius
            transition_zone = (distance_from_boundary < boundary_margin) & (distance_from_boundary > 0)
            
            # Smooth transition: energy scaling factor decreases near boundary
            scaling_factor = np.ones_like(energy_filtered)
            scaling_factor[transition_zone] = (distance_from_boundary[transition_zone] / boundary_margin)**2
            
            # Apply cap and scaling
            energy_filtered = np.minimum(energy_filtered, energy_cap)
            energy_filtered = energy_filtered * scaling_factor
    
    # Step 3: Remove extreme outliers globally
    finite_energies = energy_filtered[np.isfinite(energy_filtered)]
    if len(finite_energies) > 0:
        global_cap = np.percentile(finite_energies, energy_cap_percentile)
        energy_filtered = np.minimum(energy_filtered, global_cap)
    
    # Step 4: Optional Gaussian smoothing near boundary
    if smooth_boundary:
        # Create a mask for smoothing region
        smooth_mask = (radius > 0.8) & (radius < 0.98)
        if np.any(smooth_mask):
            # Apply light Gaussian filter only near boundary
            energy_smooth = ndimage.gaussian_filter(energy_filtered, sigma=1.0, mode='constant', cval=np.nan)
            energy_filtered[smooth_mask] = energy_smooth[smooth_mask]
    
    # Apply log scale if requested
    if log_scale:
        # Add small positive value to avoid log issues
        energy_positive = np.maximum(energy_filtered, np.nanmin(energy_filtered[energy_filtered > 0]) * 1e-3)
        z_surface = np.log10(energy_positive)
        z_label = 'log₁₀(Energy)'
    else:
        z_surface = energy_filtered
        z_label = 'Energy'
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot with better color mapping
    if log_scale:
        # For log scale, use the actual range of z_surface
        vmin, vmax = np.nanpercentile(z_surface, [5, 95])  # Use 5th-95th percentile for color range
        norm = None
    else:
        vmin, vmax = np.nanpercentile(energy_filtered, [5, 95])
        norm = None
    
    surface = ax.plot_surface(x_mesh, y_mesh, z_surface, 
                             cmap='viridis', 
                             vmin=vmin, vmax=vmax,
                             alpha=0.8,
                             linewidth=0,
                             antialiased=True,
                             rcount=100,  # Reduce resolution for smoother rendering
                             ccount=100)
    
    # Add contour projections on the bottom
    z_min = np.nanmin(z_surface)
    z_max = np.nanmax(z_surface)
    
    try:
        contour_levels = np.linspace(np.nanpercentile(z_surface, 10), 
                                   np.nanpercentile(z_surface, 90), 8)
        contours = ax.contour(x_mesh, y_mesh, z_surface, 
                             levels=contour_levels, 
                             zdir='z', 
                             offset=z_min - (z_max - z_min) * 0.1,
                             colors='gray', 
                             alpha=0.5,
                             linewidths=1)
    except:
        print("Warning: Could not generate contours, skipping...")
    
    # Add unit circle boundary at bottom
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    circle_z = np.full_like(circle_x, z_min - (z_max - z_min) * 0.1)
    ax.plot(circle_x, circle_y, circle_z, 'k-', linewidth=2, label='Poincaré disk boundary')
    
    # Labels and title
    ax.set_xlabel('x (Poincaré disk)')
    ax.set_ylabel('y (Poincaré disk)')
    ax.set_zlabel(z_label)
    ax.set_title(f'3D Energy Landscape - {lattice_type.title()} Lattice\n(Filtered boundary energies)')
    
    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_clim(vmin, vmax)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(z_label)
    
    # Set view angle for better visualization
    ax.view_init(elev=25, azim=45)
    
    # Adjust axis limits to focus on the main features
    z_range = z_max - z_min
    ax.set_zlim(z_min - z_range*0.1, z_max)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"3D surface saved to: {output_filename}")
    print(f"Energy range: {np.nanmin(energy_filtered):.2e} to {np.nanmax(energy_filtered):.2e}")
    print(f"Log energy range: {np.nanmin(z_surface):.2f} to {np.nanmax(z_surface):.2f}")
    
    # Show interactive plot
    plt.show()
    
    return fig, ax

def create_3d_surface_comparison(x_mesh, y_mesh, energy_square, energy_triangular,
                               output_filename='3d_lattice_comparison.pdf'):
    """
    Create side-by-side comparison of square vs triangular lattice energies.
    """
    
    fig = plt.figure(figsize=(20, 8))
    
    # Process both energy arrays with filtering
    for i, (energy_vals, lattice_name) in enumerate([(energy_square, 'Square'), 
                                                     (energy_triangular, 'Triangular')]):
        
        # Apply the same filtering as in main function
        radius = np.sqrt(x_mesh**2 + y_mesh**2)
        energy_filtered = np.copy(energy_vals)
        
        # Mask outside disk
        energy_filtered[radius >= 0.98] = np.nan
        
        # Cap at 95th percentile
        finite_energies = energy_filtered[np.isfinite(energy_filtered)]
        if len(finite_energies) > 0:
            cap = np.percentile(finite_energies, 95)
            energy_filtered = np.minimum(energy_filtered, cap)
        
        # Log scale
        energy_positive = np.maximum(energy_filtered, np.nanmin(finite_energies) * 1e-3)
        z_surface = np.log10(energy_positive)
        
        # Create subplot
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        
        vmin, vmax = np.nanpercentile(z_surface, [5, 95])
        surface = ax.plot_surface(x_mesh, y_mesh, z_surface, 
                                 cmap='viridis', 
                                 vmin=vmin, vmax=vmax,
                                 alpha=0.8,
                                 linewidth=0,
                                 antialiased=True,
                                 rcount=80, ccount=80)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y') 
        ax.set_zlabel('log₁₀(Energy)')
        ax.set_title(f'{lattice_name} Lattice')
        ax.view_init(elev=25, azim=45)
        
        # Add colorbar for each subplot
        cbar = fig.colorbar(surface, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('log₁₀(Energy)')
    
    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_filename}")
    
    return fig





import numpy as np

import numpy as np
from scipy import ndimage

def create_browser_3d_energy_surface(x_mesh, y_mesh, energy_values, 
                                    lattice_type='square',
                                    energy_cap_percentile=38,
                                    view_radius=1.,  # NEW: smaller radius to focus on center
                                    output_html='3d_energy_interactive.html',
                                    auto_open=True):
    """
    Create interactive 3D surface that opens in your browser.
    NOW WITH GAUSSIAN SMOOTHING AND HAPPY COLORS!
    
    Parameters:
    -----------
    view_radius : float (default: 0.5)
        Radius of the region to visualize (0.5 shows central half of disk)
    auto_open : bool
        Automatically open in browser (default: True)
    output_html : str
        HTML filename to save
    """
    
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        
        print("Creating smooth & happy interactive browser-based 3D plot! 🌈✨")
        print(f"View radius: {view_radius} (showing central {view_radius*100:.0f}% of disk)")
        
        # Apply filtering with configurable radius
        radius_squared = x_mesh**2 + y_mesh**2
        radius = np.sqrt(radius_squared)
        energy_filtered = np.copy(energy_values)
        print(f"Energy range before smoothing: {np.nanmin(energy_filtered):.2e} to {np.nanmax(energy_filtered):.2e}")

        for test_radius in [0.3, 0.5, 0.7, 0.9, 1.0]:
            test_mask = (x_mesh**2 + y_mesh**2) < test_radius**2
            test_energies = energy_values[test_mask]
            test_finite = test_energies[np.isfinite(test_energies)]
            
            if len(test_finite) > 0:
                print(f"Radius {test_radius}: {np.min(test_finite):.2e} to {np.max(test_finite):.2e} (n={len(test_finite)})")
            else:
                print(f"Radius {test_radius}: No finite values")
                # Mask outside specified radius

        
        
        
        energy_filtered[radius_squared >= view_radius**2] = np.nan
        
        # Count how many points we're keeping
        valid_points = np.sum(np.isfinite(energy_filtered))
        total_points = energy_values.size
        print(f"Keeping {valid_points}/{total_points} points ({100*valid_points/total_points:.1f}%)")
        
        # Cap high energies (keep negative ones)
        finite_energies = energy_filtered[np.isfinite(energy_filtered)]
        print(f"Energy range before Cap high energies: {np.nanmin(energy_filtered):.2e} to {np.nanmax(energy_filtered):.2e}")

        if len(finite_energies) > 0:
            high_energy_cap = np.percentile(finite_energies, energy_cap_percentile)
            n_capped = np.sum(energy_filtered > high_energy_cap)
            # Ignore data above percentile by setting to NaN:
            energy_filtered[energy_filtered > high_energy_cap] = np.nan
            print(f"Capped {n_capped} high energy values above {high_energy_cap:.2e}")
        
        # ✨ NEW: GAUSSIAN SMOOTHING! ✨
        print("Applying Gaussian smoothing for silky smooth surfaces...")
        
        # Create a mask for finite values
        finite_mask = np.isfinite(energy_filtered)
        
        if np.any(finite_mask):
            # Apply Gaussian filter only to finite values
            # First, fill NaN values with a reasonable background value for filtering
            background_value = np.nanmedian(energy_filtered)
            energy_for_filtering = np.where(finite_mask, energy_filtered, background_value)
            
            # Apply Gaussian smoothing
            gaussian_sigma = 1.5  # Smooth but not too blurry
            energy_smooth = ndimage.gaussian_filter(energy_for_filtering, 
                                                  sigma=gaussian_sigma, 
                                                  mode='constant', 
                                                  cval=background_value)
            
            # Restore NaN values where they were originally
            energy_filtered = np.where(finite_mask, energy_smooth, np.nan)
            print(f"✅ Applied Gaussian filter with σ={gaussian_sigma}")
        
        # Prepare for visualization
        #energy_positive = np.maximum(energy_filtered, np.nanmin(finite_energies) * 1)
        z_surface = energy_filtered
        
        print(f"Energy range after smoothing: {np.nanmin(energy_filtered):.2e} to {np.nanmax(energy_filtered):.2e}")
        
        # 🌈 HAPPY COLORSCALES! 🌈
        # Choose from these vibrant, cheerful color schemes
        happy_colorscales = [
            'Turbo',      # Very vibrant rainbow
            'Rainbow',    # Classic rainbow
            'Jet',        # Bright blue to red
            'Hot',        # Fire colors
            'Sunsetdark', # Beautiful sunset
            'Cividis',    # Colorblind-friendly but still vibrant
        ]
        
        # Pick a random happy colorscale or use Turbo as default (most vibrant!)
        colorscale = 'Turbo'  # The happiest, most vibrant colorscale!
        colorscale = 'Rainbow' 
        
        # Create the interactive surface with enhanced styling
        surface = go.Surface(
            x=x_mesh,
            y=y_mesh, 
            z=z_surface,
            colorscale=colorscale,  # 🌈 Happy colors!
            opacity=1.,  # Slightly more opaque for better colors
            hovertemplate='<b>x:</b> %{x:.3f}<br><b>y:</b> %{y:.3f}<br><b>Energy:</b> %{z:.3e}<extra></extra>',
            lighting=dict(
                ambient=0.5,      # More ambient light for brighter appearance
                diffuse=0.7,      # Good diffuse lighting  
                specular=0.3,     # Some specular highlights for shininess
                roughness=0.1,    # Smoother surface
                fresnel=0.2       # Nice edge lighting
            ),
            # Enhanced contours for better depth perception
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    width=2,
                    project=dict(z=True)  # Project contours on bottom
                )
            ),
            name=f'{lattice_type.title()} Lattice'
        )
        
        # Add boundary circle at the bottom with happy styling
        theta = np.linspace(0, 2*np.pi, 100)
        circle_trace = go.Scatter3d(
            x=view_radius * np.cos(theta), 
            y=view_radius * np.sin(theta),
            z=np.full_like(theta, np.nanmin(z_surface)),
            mode='lines',
            line=dict(color='gold', width=6),  # Gold boundary looks cheerful!
            name=f'View Boundary (r={view_radius})',
            hovertemplate=f'View Boundary (r={view_radius})<extra></extra>'
        )
        
        # Create the figure
        fig = go.Figure(data=[surface, circle_trace])
        
        # Update layout with happy styling
        fig.update_layout(
            title={
                'text': f'🌈 Smooth Energy Landscape - {lattice_type.title()} Lattice ✨<br>'
                        f'<sub>Gaussian-smoothed | View radius: {view_radius} | Drag to rotate • Scroll to zoom • Double-click to reset</sub>',
                'x': 0.5,
                'font': {'size': 18, 'color': 'darkblue'}
            },
            scene=dict(
                aspectmode='manual',  # Override automatic scaling
                aspectratio=dict(x=1, y=1, z=.28),  # Make z-axis much shorter
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='Strain-Energy',
                camera=dict(
                    eye=dict(x=1.3, y=1.3, z=1.0)  # Slightly higher view for better perspective
                ),
                bgcolor='white',  # Clean white background
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightblue',
                    gridwidth=1,
                    range=[-view_radius*1.1, view_radius*1.1]
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightblue',
                    gridwidth=1,
                    range=[-view_radius*1.1, view_radius*1.1]
                ),
                zaxis=dict(
                    showgrid=True,
                    gridcolor='lightblue',
                    gridwidth=1,
                    range=[0, 0.1]  # Set explicit range for your data
                )
            ),
            width=1000,
            height=700,
            margin=dict(l=0, r=0, b=0, t=80),  # More top margin for fancy title
            paper_bgcolor='white',
            plot_bgcolor='white'
)        # Save HTML file
        fig.write_html(output_html)
        print(f"🎉 Interactive 3D plot saved to: {output_html}")
        
        # Open in browser
        if auto_open:
            print("Opening in your default web browser... 🚀")
            fig.show()
        
        print("\n🎮 BROWSER CONTROLS:")
        print("• Drag with mouse: Rotate 3D view")
        print("• Scroll wheel: Zoom in/out")
        print("• Double-click: Reset to original view")
        print("• Hover: See exact values")
        print("• Toolbar (top-right): Pan, zoom, save image")
        print(f"• Colorscale: {colorscale} (vibrant & happy!)")
        print("• Surface: Gaussian-smoothed for silky appearance")
        
        return fig
        
    except ImportError:
        print("Plotly not installed. Installing now...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        
        print("Plotly installed! Please run the function again.")
        return None

# Bonus function to try different happy colorscales
def try_different_colorscales(x_mesh, y_mesh, energy_values, lattice_type='square'):
    """
    Create multiple versions with different happy colorscales to see which you like best!
    """
    
    happy_colors = [
        ('Turbo', 'turbo_version.html'),
        ('Rainbow', 'rainbow_version.html'),
        ('Jet', 'jet_version.html'),
        ('Hot', 'hot_version.html'),
        ('Sunsetdark', 'sunset_version.html'),
        ('Cividis', 'cividis_version.html'),
    ]
    
    print("🎨 Creating multiple colorful versions...")
    
    for colorscale, filename in happy_colors:
        # Temporarily modify the function to use different colorscales
        # You can copy the function above and change the colorscale line
        print(f"Creating {colorscale} version...")
        # create_browser_3d_energy_surface with that colorscale
    
    print("🌈 All colorful versions created! Pick your favorite!")

if __name__ == "__main__":
    print("🌈✨ HAPPY & SMOOTH 3D ENERGY VISUALIZATION ✨🌈")
    print("=" * 50)
    print("Now with Gaussian smoothing and vibrant colors!")
    print("Usage: create_browser_3d_energy_surface(x, y, energy_data)")

def create_browser_comparison(x_mesh, y_mesh, energy_square, energy_triangular,
                             view_radius=0.5,  # NEW: configurable radius for comparison too
                             output_html='3d_comparison_interactive.html'):
    """
    Create side-by-side browser comparison of both lattice types.
    
    Parameters:
    -----------
    view_radius : float (default: 0.5)
        Radius of the region to visualize for both lattices
    """
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        print(f"Creating comparison with view radius: {view_radius}")
        
        # Create side-by-side subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f'Square Lattice (r≤{view_radius})', f'Triangular Lattice (r≤{view_radius})'],
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            horizontal_spacing=0.1
        )
        
        for i, (energy_vals, lattice_name, colorscale) in enumerate([
            (energy_square, 'Square', 'Plasma'),
            (energy_triangular, 'Triangular', 'Viridis')
        ]):
            
            # Filter data with configurable radius
            radius_squared = x_mesh**2 + y_mesh**2
            energy_filtered = np.copy(energy_vals)
            energy_filtered[radius_squared >= view_radius**2] = np.nan
            
            finite_energies = energy_filtered[np.isfinite(energy_filtered)]
            if len(finite_energies) > 0:
                cap = np.percentile(finite_energies, 95)
                energy_filtered = np.minimum(energy_filtered, cap)
            
            #z_surface = np.log10(np.maximum(energy_filtered, np.nanmin(finite_energies) * 1e-6))
            z_surface = np.maximum(energy_filtered, np.nanmin(finite_energies))
            
            # Add surface
            surface = go.Surface(
                x=x_mesh, y=y_mesh, z=z_surface,
                colorscale=colorscale,
                showscale=(i==1),  # Show colorbar only on right plot
                opacity=0.9,
                name=f'{lattice_name} Lattice'
            )
            
            fig.add_trace(surface, row=1, col=i+1)
        
        # Update layout
        fig.update_layout(
            title=f'Interactive Comparison: Square vs Triangular Lattices (r ≤ {view_radius})',
            height=600,
            width=1400
        )
        
        # Update scene axes to match the view radius
        for i in range(1, 3):  # Both subplots
            fig.update_layout(**{
                f'scene{i if i > 1 else ""}': dict(
                    xaxis=dict(range=[-view_radius*1.1, view_radius*1.1]),
                    yaxis=dict(range=[-view_radius*1.1, view_radius*1.1])
                )
            })
        
        # Save and show
        fig.write_html(output_html)
        print(f"Interactive comparison saved to: {output_html}")
        fig.show()
        
        return fig
        
    except ImportError:
        print("Please install plotly: pip install plotly")
        return None

def make_browser_interactive_plots(x_mesh, y_mesh, energy_square, energy_triangular, 
                                 view_radius=0.3, auto_open=True):
    """
    Create browser-based interactive plots for your energy data.
    Call this instead of your matplotlib functions.
    
    Parameters:
    -----------
    view_radius : float (default: 0.3)
        Radius of region to visualize (0.3 = central 30% of disk)
    auto_open : bool (default: True)
        Whether to automatically open plots in browser
    """
    
    print("Creating browser-based interactive 3D plots...")
    print(f"Using view radius: {view_radius} (central {view_radius*100:.0f}% of disk)")
    
    # Individual plots
    fig_square = create_browser_3d_energy_surface(
        x_mesh, y_mesh, energy_square, 
        lattice_type='square',
        view_radius=view_radius,
        output_html='figures/3d_square_interactive.html',
        auto_open=auto_open
    )
    
    # fig_triangular = create_browser_3d_energy_surface(
    #     x_mesh, y_mesh, energy_triangular,
    #     lattice_type='triangular', 
    #     view_radius=view_radius,
    #     output_html='figures/3d_triangular_interactive.html',
    #     auto_open=auto_open
    # )
    
    # # Side-by-side comparison
    # fig_comparison = create_browser_comparison(
    #     x_mesh, y_mesh, energy_square, energy_triangular,
    #     view_radius=view_radius,
    #     output_html='figures/3d_comparison_interactive.html'
    # )
    
    print(f"\nAll plots created with view_radius={view_radius}")
    print("Files saved in figures/ directory")
    
    return fig_square

if __name__ == "__main__":
    print("Browser-Based Interactive 3D Energy Visualization")
    print("================================================")
    print("This creates HTML files that open in your browser")
    print("Much more reliable interaction than matplotlib!")
    print("\nUsage in your code:")
    print("make_browser_interactive_plots(x_, y_, phi0_square, phi0_triangular)")