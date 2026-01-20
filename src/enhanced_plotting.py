"""
Enhanced plotting module that combines energy visualization with 
fundamental domain boundaries from python_version.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .projections import Cij_from_stereographic_projection_tr, stereographic_projection_from_Cij_2D

def projection1(c11, c22, c12):
    """Project metric onto disk (from python_version.py)"""
    d_t = (c12/c22)*(c12/c22) + (1.0/c22 + 1.0)*(1.0/c22 + 1.0)
    x1_t = (c12/c22)*(c12/c22) + (1.0/c22)*(1.0/c22) - 1.0
    y1_t = 2*c12/c22
    x_t = x1_t/d_t
    y_t = y1_t/d_t
    return(x_t, y_t)

def mapping_c(c11, c22, c12, mt):
    """Map metric using transformation matrix (from python_version.py)"""
    c_mat = np.array([[c11, c12], [c12, c22]])
    c_mapped = np.matmul(np.matmul(np.transpose(mt), c_mat), mt)
    return c_mapped

def setup_fundamental_domain_boundary():
    """Setup boundary data (from python_version.py)"""
    N_steps = 100
    c1_list = np.zeros([3*(N_steps + 1) + 1, 3])
    
    ## 1st boundary
    t_min = 0.0001
    t_max = 1.0
    for ind_b in range(N_steps+1):
        t_temp = t_min + ind_b*(t_max - t_min)/N_steps
        c1_list[ind_b, 0] = t_temp
        c1_list[ind_b, 1] = 1.0/t_temp
        c1_list[ind_b, 2] = 0.0
    
    ## 2nd boundary
    t_min = 0.0001
    t_max = 1.0/2.0
    for ind_b in range(N_steps+1):
        t_temp = t_min + ind_b*(t_max - t_min)/N_steps
        c1_list[N_steps + 1 + ind_b, 0] = math.sqrt(t_temp*t_temp + 1)
        c1_list[N_steps + 1 + ind_b, 1] = math.sqrt(t_temp*t_temp + 1)
        c1_list[N_steps + 1 + ind_b, 2] = t_temp
    
    ## 3rd boundary
    t_min = math.sqrt(3.0/4.0)
    t_max = 1000.0
    for ind_b in range(N_steps+1):
        t_temp = t_min + (ind_b*ind_b*ind_b)*(t_max - t_min)/((N_steps*N_steps*N_steps))
        c1_list[2*(N_steps + 1) + ind_b, 0] = 1.0/t_temp
        c1_list[2*(N_steps + 1) + ind_b, 1] = 1.0/(t_temp*4.0) + t_temp
        c1_list[2*(N_steps + 1) + ind_b, 2] = 1.0/(t_temp*2.0)
    
    ## Closing the loop
    c1_list[np.size(c1_list, 0)-1, :] = c1_list[0, :]
    
    ## Point at the center of the region
    c1_center = np.zeros(3)
    c1_center[0] = 1.0 + 0.2
    c1_center[1] = 2.0
    c1_center[2] = 0.3
    detC = c1_center[0]*c1_center[1] - c1_center[2]*c1_center[2]
    c1_center = c1_center/math.sqrt(detC)
    
    return c1_list, c1_center, N_steps

def plot_fundamental_domain_boundaries(ax, shape_flag="square"):
    """
    Plot fundamental domain boundaries on existing axis
    
    Parameters:
    -----------
    ax : matplotlib axis
        Existing plot axis
    shape_flag : str
        "square" or "triangle" mode
    """
    # Setup H matrix based on shape_flag
    g_temp = (4.0/3.0)**(1.0/4.0)
    
    if shape_flag == "square":
        H_tri = np.eye(2)
        r_array = np.eye(2)
    else:  # shape_flag == "triangle" 
        H_tri = (g_temp)*np.array([[1.0, 1.0/2.0],[0.0, math.sqrt(3.0)/2.0]])
        r_array = np.array([[math.sqrt(3.0)/2.0, -1.0/2.0], [1.0/2.0, math.sqrt(3.0)/2.0]])
    
    H_inv = np.linalg.inv(H_tri)
    
    # Get boundary data
    c1_list, c1_center, N_steps = setup_fundamental_domain_boundary()
    
    # Transformation matrices (from python_version.py)
    m1_t = np.array([[1, 0],[0, 1]])
    m2_t = np.array([[0, -1],[1, 1]])
    m3_t = np.array([[1, 1],[1, 0]])
    m4_t = np.array([[-1, 0],[1, 1]])
    m5_t = np.array([[1, 1],[0, -1]])
    m6_t = np.array([[0, 1],[1, 0]])
    
    def BinaryParallel(ind_temp):
        lvl_max = 20
        bin_ind = np.zeros(lvl_max, dtype=int)
        r_temp = int(ind_temp)
        r_counter = 0
        while r_temp>0:
            bin_ind[r_counter] = int(r_temp%2)
            r_temp = r_temp - r_temp%2
            r_temp = int(r_temp/2)
            r_counter = r_counter + 1
        return bin_ind

    def MAPandPLOT_to_coordinates(c_list_temp, mt):
        """Convert boundary to image coordinates"""
        xy_mapped = np.zeros([np.size(c_list_temp, 0), 2])
        
        for ind_m in range(np.size(c_list_temp, 0)):
            c_mapped = mapping_c(c_list_temp[ind_m, 0], c_list_temp[ind_m, 1], c_list_temp[ind_m, 2], mt)
            c_mapped = np.matmul(np.matmul(np.transpose(H_inv), c_mapped), H_inv)
            xy_mapped[ind_m, 0], xy_mapped[ind_m, 1] = projection1(c_mapped[0, 0], c_mapped[1, 1], c_mapped[0, 1])
            xy_mapped[ind_m, :] = np.matmul(r_array, xy_mapped[ind_m, :])
            
        # Convert from [-1,1] coordinates to image coordinates [0, disc-1]
        #xy_img = np.zeros_like(xy_mapped)
        # xy_img[:, 0] = (xy_mapped[:, 0] + 0.999) * (disc - 1) / (2 * 0.999)
        # xy_img[:, 1] = (xy_mapped[:, 1] + 0.999) * (disc - 1) / (2 * 0.999)
        
        return xy_mapped

    # Generate all transformation matrices and plot boundaries
    level = 4
    boundary_color = 'white'
    boundary_alpha = 0.8
    
    for ind_l in range(level):
        if ind_l == 0:
            NUM_elastic = 1
        else:
            NUM_elastic = int(3*2**(ind_l-1))
            
        for ind_s in range(6):
            if ind_s==0:
                mt = np.matmul(m1_t, m1_t)
            elif ind_s==1:
                mt = np.matmul(m4_t, m1_t)
            elif ind_s==2:
                mt = np.matmul(m5_t, m1_t)
            elif ind_s==3:
                mt = np.matmul(m6_t, m1_t)
            elif ind_s==4:
                mt = np.matmul(m4_t, m5_t)
            elif ind_s==5:
                mt = np.matmul(m4_t, m6_t)
            
            if ind_l==0:
                n_copy = 1
            else:
                n_copy = 3
                
            for ind_q in range(n_copy):
                for ind_nj in range(NUM_elastic):
                    mt2 = np.eye(2)
                    parallel = BinaryParallel(ind_nj)
                    for ind_n in range(ind_l-1):
                        mt2 = np.matmul(mt2, m3_t)
                        if parallel[ind_n]==0:
                            mt2 = np.matmul(mt2, m5_t)
                        elif parallel[ind_n]==1:
                            mt2 = np.matmul(mt2, m6_t)
                    if ind_l==0:
                        m_sh = m1_t
                    else:
                        if ind_q==0:
                            m_sh = np.matmul(m3_t, m1_t)
                        elif ind_q==1:
                            m_sh = np.matmul(m3_t, m4_t)
                        elif ind_q==2:
                            m_sh = np.matmul(m3_t, m6_t)
                    
                    mt2 = np.matmul(mt2, m_sh)
                    mt_f = np.matmul(mt, mt2)
                    
                    # Convert boundaries to image coordinates
                    xy_mapped = MAPandPLOT_to_coordinates(c1_list, mt_f)
                    
                    # Plot different boundary segments with different line styles
                    line_styles = [':', '--', '-']
                    line_widths = [1.0, 1.0, 1.5]
                    
                    # 1st boundary
                    ax.plot(xy_mapped[0:N_steps, 0], xy_mapped[0:N_steps, 1], 
                           c=boundary_color, linewidth=line_widths[0], 
                           linestyle=line_styles[0], alpha=boundary_alpha)
                    
                    # 2nd boundary  
                    ax.plot(xy_mapped[(N_steps+1):(2*N_steps), 0], 
                           xy_mapped[(N_steps+1):(2*N_steps), 1], 
                           c=boundary_color, linewidth=line_widths[1], 
                           linestyle=line_styles[1], alpha=boundary_alpha)
                    
                    # 3rd boundary
                    ax.plot(xy_mapped[(2*N_steps+1):(3*N_steps), 0], 
                           xy_mapped[(2*N_steps+1):(3*N_steps), 1], 
                           c=boundary_color, linewidth=line_widths[2], 
                           linestyle=line_styles[2], alpha=boundary_alpha)

def plot_shape_at_matrix(ax, matrix, color='red', size=0.03, shape='square', shape_flag="square"):
    """
    Plot a square or triangle at the location corresponding to a transformation matrix
    
    Parameters:
    -----------
    ax : matplotlib axis
        Plot axis
    matrix : numpy.ndarray
        2x2 transformation matrix
    disc : int
        Discretization parameter for coordinate conversion
    color : str
        Color of the shape
    size : float
        Size of the shape (in image coordinates)
    shape : str
        'square' or 'triangle' to choose the shape type
    shape_flag : str
        "square" or "triangle" mode for coordinate system
    """
    # Setup H matrix and rotation based on shape_flag
    g_temp = (4.0/3.0)**(1.0/4.0)
    
    if shape_flag == "square":
        H_tri = np.eye(2)
        r_array = np.eye(2)
    else:  # shape_flag == "triangle" 
        H_tri = (g_temp)*np.array([[1.0, 1.0/2.0],[0.0, math.sqrt(3.0)/2.0]])
        r_array = np.array([[math.sqrt(3.0)/2.0, -1.0/2.0], [1.0/2.0, math.sqrt(3.0)/2.0]])
    
    H_inv = np.linalg.inv(H_tri)
    
    # Convert matrix to a point in the Poincare disk
    matrix_transformed = np.matmul(np.matmul(np.transpose(H_inv), matrix), H_inv)
    x_pos, y_pos = projection1(matrix_transformed[0, 0], matrix_transformed[1, 1], matrix_transformed[0, 1])
    
    # Apply rotation array
    xy_pos = np.matmul(r_array, np.array([x_pos, y_pos]))
    
    # Use mathematical coordinates directly
    x_img = xy_pos[0]
    y_img = xy_pos[1]

    # Keep size as-is (it's already in mathematical coordinates)
    size_img = size    

    if shape == 'triangle':
        # Plot triangle - equilateral triangle centered at (x_img, y_img)
        triangle_x = [x_img, 
                      x_img - size_img*math.cos(math.pi/6), 
                      x_img + size_img*math.cos(math.pi/6), 
                      x_img]
        triangle_y = [y_img + size_img, 
                      y_img - size_img*math.sin(math.pi/6), 
                      y_img - size_img*math.sin(math.pi/6), 
                      y_img + size_img]
        ax.fill(triangle_x, triangle_y, color=color, alpha=.7, edgecolor='black', linewidth=1, zorder=10)
    else:  # shape == 'square' (default)
        # Plot square
        square_x = [x_img-size_img, x_img+size_img, x_img+size_img, x_img-size_img, x_img-size_img]
        square_y = [y_img-size_img, y_img-size_img, y_img+size_img, y_img+size_img, y_img-size_img]
        ax.fill(square_x, square_y, color=color, alpha=.7, edgecolor='black', linewidth=1, zorder=10)
    
    return x_img, y_img

def generate_transformation_matrices():
    """Generate transformation matrices (from python_version.py)"""
    transformation_matrices = []
    
    # Transformation matrices
    m1_t = np.array([[1, 0],[0, 1]])
    m2_t = np.array([[0, -1],[1, 1]])
    m3_t = np.array([[1, 1],[1, 0]])
    m4_t = np.array([[-1, 0],[1, 1]])
    m5_t = np.array([[1, 1],[0, -1]])
    m6_t = np.array([[0, 1],[1, 0]])
    
    def BinaryParallel(ind_temp):
        lvl_max = 20
        bin_ind = np.zeros(lvl_max, dtype=int)
        r_temp = int(ind_temp)
        r_counter = 0
        while r_temp>0:
            bin_ind[r_counter] = int(r_temp%2)
            r_temp = r_temp - r_temp%2
            r_temp = int(r_temp/2)
            r_counter = r_counter + 1
        return bin_ind

    level = 4
    for ind_l in range(level):
        if ind_l == 0:
            NUM_elastic = 1
        else:
            NUM_elastic = int(3*2**(ind_l-1))
        for ind_s in range(6):
            if ind_s==0:
                mt = np.matmul(m1_t, m1_t)
            elif ind_s==1:
                mt = np.matmul(m4_t, m1_t)
            elif ind_s==2:
                mt = np.matmul(m5_t, m1_t)
            elif ind_s==3:
                mt = np.matmul(m6_t, m1_t)
            elif ind_s==4:
                mt = np.matmul(m4_t, m5_t)
            elif ind_s==5:
                mt = np.matmul(m4_t, m6_t)
            
            if ind_l==0:
                n_copy = 1
            else:
                n_copy = 3
            for ind_q in range(n_copy):
                for ind_nj in range(NUM_elastic):
                    mt2 = np.eye(2)
                    parallel = BinaryParallel(ind_nj)
                    for ind_n in range(ind_l-1):
                        mt2 = np.matmul(mt2, m3_t)
                        if parallel[ind_n]==0:
                            mt2 = np.matmul(mt2, m5_t)
                        elif parallel[ind_n]==1:
                            mt2 = np.matmul(mt2, m6_t)
                    if ind_l==0:
                        m_sh = m1_t
                    else:
                        if ind_q==0:
                            m_sh = np.matmul(m3_t, m1_t)
                        elif ind_q==1:
                            m_sh = np.matmul(m3_t, m4_t)
                        elif ind_q==2:
                            m_sh = np.matmul(m3_t, m6_t)
                    
                    mt2 = np.matmul(mt2, m_sh)
                    mt_f = np.matmul(mt, mt2)
                    
                    transformation_matrices.append(mt_f.copy())
    
    return transformation_matrices

def plot_squares_and_triangles(ax, shape_flag="square"):
    """
    Plot squares and triangles at transformation matrix positions
    
    Parameters:
    -----------
    ax : matplotlib axis
        Plot axis
    disc : int
        Discretization parameter
    shape_flag : str
        "square" or "triangle" mode
    """
    # Get transformation matrices
    transformation_matrices = generate_transformation_matrices()
    
    # Plot identity matrix square first
    plot_shape_at_matrix(ax, np.eye(2), color='black', size=0.016, shape='square', shape_flag=shape_flag)
    
    # Plot squares using transformation matrices
    base_square_matrix = np.eye(2)
    for i, mt in enumerate(transformation_matrices):
        transformed_matrix = mt.T @ base_square_matrix @ mt
        plot_shape_at_matrix(ax, transformed_matrix, color='black', size=0.016, shape='square', shape_flag=shape_flag)

    # Plot triangles using specific triangle F matrix
    g_temp = (4.0/3.0)**(1.0/4.0)
    triangle_F_base = g_temp * np.array([[1, 1/2], [0, math.sqrt(3.0)/2.0]])
    base_triangle_matrix = triangle_F_base.T @ triangle_F_base

    for i, mt in enumerate(transformation_matrices):
        transformed_matrix = mt.T @ base_triangle_matrix @ mt
        plot_shape_at_matrix(ax, transformed_matrix, color='black', size=0.02, shape='triangle', shape_flag=shape_flag)

def poincare_plot_energy_with_fundamental_domains(config, name, pvmin, pvmax, disc=1000, shape_flag="square", stability_file=None):
    """
    Plot energy on Poincaré disk with fundamental domain boundaries and square/triangle points
    
    Parameters:
    -----------
    config : array_like
        Energy configuration data (2D array)
    name : str
        Output filename (without extension)
    pvmin, pvmax : float
        Color scale limits
    disc : int
        Discretization parameter
    shape_flag : str
        "square" or "triangle" mode for boundaries
    stability_file : str, optional
        Path to stability boundary data file
    """
    from scipy.interpolate import splprep, splev
    
    energy_filtered = np.copy(config)
    finite_energies = energy_filtered[np.isfinite(energy_filtered)]
    energy_cap_percentile = 100

    if len(finite_energies) > 0:
        high_energy_cap = np.percentile(finite_energies, energy_cap_percentile)
        n_capped = np.sum(energy_filtered > high_energy_cap)
        energy_filtered = np.minimum(energy_filtered, high_energy_cap)
        print(f"Capped {n_capped} high energy values above {high_energy_cap:.2e}")

    config = energy_filtered
    # pvmin = np.nanmin(config)
    # pvmax =  np.nanmax(config)
     
    # pvmin = np.nanmin(config)
    # pvmax = 0.03 * np.nanmax(config)


    fig = plt.figure(figsize=(12., 12.))
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = fig.add_subplot(111)
    
    colmap = matplotlib.cm.RdYlBu_r
    m = ax.imshow(config, origin='lower', interpolation='none', 
                 cmap=colmap, vmin=pvmin, vmax=pvmax, extent=[-0.999, 0.999, -0.999, 0.999])
    R = 1
    
    # Plot the edge of Poincaré disk
    circle_x = R * np.cos(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    circle_y = R * np.sin(np.linspace(0., 2.*np.pi, 360, endpoint=False)) + R - 1
    ax.plot(circle_x, circle_y, 'k-', lw=2.0)
    
    # Plot fundamental domain boundaries
    plot_fundamental_domain_boundaries(ax, shape_flag)
    
    # Plot squares and triangles
    plot_squares_and_triangles(ax, shape_flag)

    # === STABILITY BOUNDARY WITH INTERPOLATION ===
    if stability_file is not None:
        try:
            # **ADD THIS FLAG AT THE TOP**
            USE_ALL_POINTS = True  # Set to True to use all points without filtering
            
            # Read the data file (skip header line starting with #)
            data = np.loadtxt(stability_file)
            c11_data = data[:, 0]
            c22_data = data[:, 1]
            c12_data = data[:, 2]
            
            # Apply stability criterion
            # stability_mask = np.logical_and(
            #     c12_data >= 0,
            #     c12_data <= np.minimum(c11_data, c22_data)
            # )  
            stability_mask = 2*np.abs(c12_data) <= np.minimum(c11_data, c22_data) 



            # Filter the data
            c11_data = c11_data[stability_mask]
            c22_data = c22_data[stability_mask]
            c12_data = c12_data[stability_mask]
            
            print(f"Loaded {len(c11_data)} stability boundary points from file")
            
            # === IDENTIFY AND REMOVE ISOLATED SEGMENTS ===
            # Group points into connected segments
            def find_contour_segments(x, y, distance_threshold=100):
                """Split points into connected segments"""
                if len(x) == 0:
                    return []
                
                segments = []
                current_segment_x = [x[0]]
                current_segment_y = [y[0]]
                
                for i in range(1, len(x)):
                    dist = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
                    
                    if dist < distance_threshold:
                        # Point is connected to previous
                        current_segment_x.append(x[i])
                        current_segment_y.append(y[i])
                    else:
                        # Start new segment
                        if len(current_segment_x) > 1:
                            segments.append((np.array(current_segment_x), 
                                        np.array(current_segment_y)))
                        current_segment_x = [x[i]]
                        current_segment_y = [y[i]]
                
                # Add last segment
                if len(current_segment_x) > 1:
                    segments.append((np.array(current_segment_x), 
                                np.array(current_segment_y)))
                
                return segments
            
            # Define transformation matrices
            # transformations = [
            #     (np.array([[1, 0], [0, 1]]), "Original"),           # Identity (original)
            #     (np.array([[1, 1], [0, 1]]), "m=[1,1;0,1]"),       # m = [1, 1; 0, 1]
            #     (np.array([[1, -1], [0, 1]]), "m=[1,-1;0,1]"),     # m = [1, -1; 0, 1]
            #     (np.array([[1, 0], [1, 1]]), "m=[1,0;1,1]"),       # m = [1, 0; 1, 1]
            #     (np.array([[1, 0], [-1, 1]]), "m=[1,0;-1,1]")      # m = [1, 0; -1, 1]
            # ]
            transformations = [
                (np.array([[1, 0], [0, 1]]), r'$\det \mathbf{Q} = 0$'),          # Identity (original)
            ]

            if shape_flag == "triangular":
                gamma = (4/3)**(1/4)
                H = gamma * np.array([[np.sqrt(2 + np.sqrt(3))/2, np.sqrt(2 - np.sqrt(3))/2],
                                    [np.sqrt(2 - np.sqrt(3))/2, np.sqrt(2 + np.sqrt(3))/2]])
                H = np.linalg.inv(H)
                print("Using triangle shape_flag: applying H transformation")
                transformations.append((np.array([[1, 1/2], [0, math.sqrt(3)/2]]), "Triangle F"))
                
                # Compose transformations with H
                transformations = [(mat @ H, label) for mat, label in transformations]

            colors = ['black', 'blue', 'red', 'green', 'orange']
            
            # Plot each transformation
            for (mat, label), color in zip(transformations, colors):
                # Apply transformation: C' = mat^T @ C @ mat
                m11, m12 = mat[0, 0], mat[0, 1]
                m21, m22 = mat[1, 0], mat[1, 1]
                
                c11_t = m11**2 * c11_data + 2*m11*m21*c12_data + m21**2 * c22_data
                c22_t = m12**2 * c11_data + 2*m12*m22*c12_data + m22**2 * c22_data
                c12_t = m11*m12*c11_data + (m11*m22 + m12*m21)*c12_data + m21*m22*c22_data
                
                # Convert to stereographic coordinates
                x_stereo_data, y_stereo_data = stereographic_projection_from_Cij_2D(
                    c11_t, c22_t, c12_t)

                # Apply rotation
                theta = np.radians(0)
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                x_img_data = x_stereo_data * cos_theta - y_stereo_data * sin_theta
                y_img_data = x_stereo_data * sin_theta + y_stereo_data * cos_theta
                
                # === DATA VALIDATION AND CLEANING ===
                valid_mask = np.isfinite(x_img_data) & np.isfinite(y_img_data)
                x_img_data = x_img_data[valid_mask]
                y_img_data = y_img_data[valid_mask]
                
                print(f"{label}: After removing invalid values: {len(x_img_data)} points")
                
                # === CONDITIONAL SEGMENT FILTERING ===
                if not USE_ALL_POINTS:
                    # === SPLIT INTO SEGMENTS AND KEEP ONLY THE LONGEST ===
                    segments = find_contour_segments(x_img_data, y_img_data, 
                                                    distance_threshold=0.05)
                    
                    if len(segments) > 0:
                        # Find longest segment
                        segment_lengths = [len(seg[0]) for seg in segments]
                        longest_idx = np.argmax(segment_lengths)
                        
                        x_img_data = segments[longest_idx][0]
                        y_img_data = segments[longest_idx][1]
                        
                        print(f"{label}: Found {len(segments)} segments, kept longest with {len(x_img_data)} points")
                    else:
                        print(f"{label}: No valid segments found")
                        continue
                else:
                    print(f"{label}: Using all {len(x_img_data)} points (no segment filtering)")
                
                # Remove duplicate consecutive points
                if len(x_img_data) > 1:
                    diffs = np.sqrt(np.diff(x_img_data)**2 + np.diff(y_img_data)**2)
                    keep_mask = np.concatenate([[True], diffs > 1e-10])
                    x_img_data = x_img_data[keep_mask]
                    y_img_data = y_img_data[keep_mask]
                    print(f"{label}: After removing duplicates: {len(x_img_data)} points")
                
                interpolation = True
                # === INTERPOLATE THESE POINTS ===
                if len(x_img_data) >= 4 and interpolation == True:
                    from scipy.ndimage import gaussian_filter1d
                    
                    # Pre-smooth
                    x_img_data = gaussian_filter1d(x_img_data, sigma=5, mode='wrap')
                    y_img_data = gaussian_filter1d(y_img_data, sigma=5, mode='wrap')
                    
                    dist_to_start = np.sqrt((x_img_data[0] - x_img_data[-1])**2 + 
                                        (y_img_data[0] - y_img_data[-1])**2)
                    is_closed = dist_to_start < 0.1
                    
                    try:
                        tck, u = splprep([x_img_data, y_img_data], s=0.0001, k=3, per=is_closed)
                        u_new = np.linspace(0, 1, 2000)
                        x_smooth, y_smooth = splev(u_new, tck)
                        
                        ax.plot(x_smooth, y_smooth, '-', color=color, linewidth=2,
                            label=f'Stability boundary ({label})', alpha=0.8)
                        
                        print(f"✓ Plotted {label} with {len(x_smooth)} points")
                        
                    except Exception as e:
                        print(f"{label}: Spline interpolation failed: {e}")
                        ax.plot(x_img_data, y_img_data, 'o-', color=color, markersize=1, 
                            linewidth=1, label=f'Stability boundary ({label})', alpha=0.8)
                else:
                    ax.plot(x_img_data, y_img_data, 'o-', color=color, markersize=3,
                        label=f'Stability boundary ({label})', alpha=0.8)
                    print(f"Warning: {label} only {len(x_img_data)} points")
            
            ax.legend(loc='upper right', fontsize=10)
            
        except Exception as e:
            print(f"Warning: Could not plot stability boundary: {e}")
            import traceback
            traceback.print_exc()

                # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.75%", pad=0.1)
    cbar = plt.colorbar(mappable=m, cax=cax)
    cbar.set_label('Strain-Energy Density', fontsize=12)
    
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=17, pad=2)
    ax.tick_params(axis='both', which='minor', labelsize=11, pad=1)
    ax.set_xlabel('p', fontsize=24)
    ax.set_ylabel('q', fontsize=24)
    
    # Save both PDF and PNG
    fig.savefig(name + '.pdf', bbox_inches='tight', dpi=800)
    fig.savefig(name + '.png', bbox_inches='tight', dpi=800)
    
    plt.close(fig)
    
def create_combined_visualization():
    """
    Example function showing how to use the combined visualization
    """
    from src.projections import Cij_from_stereographic_projection_tr
    from src.lagrange import lagrange_reduction
    from src.energy import conti_phi0_from_Cij, convert_data_SymLog
    
    print("Generating combined energy and boundary visualization...")
    
    # Parameters
    disc = 1000
    lattice = 'square'
    shape_flag = 'square'  # or 'triangle'
    
    # Generate stereographic projection coordinates
    x, y = np.linspace(-.999, .999, num=disc, endpoint=True), np.linspace(-.999, .999, num=disc, endpoint=True)
    x_, y_ = np.meshgrid(x, y)
    
    # Mask values outside Poincaré disk
    x_ = np.where(x_**2 + y_**2 - (.999)**2 >= 1.e-6, np.nan, x_)
    y_ = np.where(x_**2 + y_**2 - (.999)**2 >= 1.e-6, np.nan, y_)
    
    # Apply stereographic projection with centering transformation
    c11, c22, c12, c11t, c22t, c12t = Cij_from_stereographic_projection_tr(x_, y,shape_flag)
    
    # Perform Lagrange reduction
    c11_reduced, c22_reduced, c12_reduced, iterations = lagrange_reduction(c11, c22, c12, verbose=False)
    
    # Compute strain energy density
    phi0 = conti_phi0_from_Cij(c11_reduced, c22_reduced, c12_reduced, lattice)
    
    # Normalize and convert to symmetric log scale
    phi0_normalized = (phi0 - np.nanmin(phi0))
    c_scale = 1e-2
    config = convert_data_SymLog(phi0_normalized, c_scale)
    
    pvmin = np.nanmin(config)
    pvmax = 0.8 * np.nanmax(config)
    
    # Create the combined plot
    output_name = './figures/poincare_energy_with_boundaries'
    poincare_plot_energy_with_fundamental_domains(config, output_name, pvmin, pvmax, disc, shape_flag)
    
    print(f"Combined visualization saved as: {output_name}.pdf and {output_name}.png")

if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)
    create_combined_visualization()