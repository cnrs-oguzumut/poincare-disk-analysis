###########################

## importing libraries 

import time

import math as m

import matplotlib

matplotlib.use('Agg')



import numpy as np

import matplotlib.pyplot as plt



from scipy.ndimage import gaussian_filter

from mpl_toolkits.axes_grid1 import make_axes_locatable



from math import sqrt

import math



import os

import sys			     	



#...................................................................................................



def poincare_plot_scatter(config, px, py, name, pvmin, pvmax):

    

    fig = plt.figure(figsize=(12.,12.))

    plt.clf()

    plt.rc('text', usetex=True)

    plt.rc('font', family='serif')

    ax = fig.add_subplot(111)

    disc=1000

    colmap = matplotlib.cm.RdYlBu_r

    m=ax.imshow(config, origin='lower',interpolation='none', cmap = colmap, vmin = pvmin, vmax=pvmax)

    R = 0.5 * disc

    axx = R*px + R-1

    ayy = R*py + R-1

    ax.plot(axx, ayy , marker='o', color='w', markeredgecolor='k', markersize=.5, lw=0,linestyle='None')

    # edge of Poincare disk

    plt.plot(R * np.cos(np.linspace(0.,2.*np.pi,360,endpoint=False)) + R-1, 

             R * np.sin(np.linspace(0.,2.*np.pi,360,endpoint=False)) + R-1, 'k-', lw=1.1)

    

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="3.75%", pad=0.1)

    plt.colorbar(mappable=m, cax=cax)

    ax.axis('off')

    



    fig.savefig(name +'.pdf', bbox_inches='tight')

    plt.close(fig)#'''





def poincare_plot_scatter_hist(config, px, py, name, pvmin, pvmax):





	disc=1000

	plt.clf()

	plt.rc('text', usetex=True)

	plt.rc('font', family='serif')

	#     ax = fig.add_subplot(111)

	#     

	R = 0.5 * disc

	import matplotlib.colors as mcolors

	from matplotlib.colors import LogNorm

	axx = R*px + R-1

	ayy = R*py + R-1



	bins=100

	h, xedges, yedges, image = plt.hist2d(axx, ayy, bins=(bins, bins),norm=LogNorm(vmin=1, vmax=1000))

	xcenters = (xedges[:-1] + xedges[1:]) / 2

	ycenters = (yedges[:-1] + yedges[1:]) / 2

	

	fig = plt.figure(figsize=(12.,12.))

	

	colmap = matplotlib.cm.RdYlBu_r

	m=plt.imshow(config, origin='lower',interpolation='none', cmap = colmap, vmin = pvmin, vmax=pvmax)



# 	plt.hist2d(axx, ayy, bins=(bins, bins), cmap=matplotlib.cm.seismic,alpha=.9,lw=0,linestyle='None',norm=LogNorm(vmin=1, vmax=1000))

# 	cbar = plt.colorbar()

# 	cbar.set_label('counts', labelpad=-40, y=1.05, rotation=0)

# 	cbar.ax.set_ylabel('counts')



	plt.pcolormesh(xcenters, ycenters, h)



	fig.savefig(name +'.pdf', bbox_inches='tight')

	plt.close(fig)#’''





#...................................................................................................



def convert_data_SymLog(map, const):

	'''Convert data to a SymLog data scale 

    (log scales for both positive and negative values and linearized near zero)

    

	Reference : 	J B W Webber - A bi-symmetric log transformation for wide-range data (2013)

					Measurement Science and Technology 24'''

                    

	# const: constant defining the linearized 'near zero region'

    

	return np.sign(map)*np.log10(1. + np.absolute(map)/const)



#...................................................................................................

def Cij_from_orthogonal_projection(x, y):

    

    c11 =  x

    c22 = -x

    c12 =  y

    

    return c11, c22, c12





#...................................................................................................

def Cij_from_stereographic_projection(x, y): 

    

    t = 2./(1.-x**2.-y**2.)

    

    c11 = t*(1.+x)-1.

    c22 = t*(1.-x)-1.

    c12 = t*y

    

    return c11, c22, c12

 #...................................................................................................

def stereographic_projection_from_Cij_2D(c11, c22, c12):



    t = 2./(2.+c11+c22)



    x = t * (c11-c22) / 2.

    y = t * c12



    return x, y   



#...................................................................................................



def interatomic_phi0_from_Cij(c11, c22, c12, lattice):

    

    r1 = 1.

    r2 = 1.425

    b1 = 8.

    b2 = 8.

    a  = 1.

    c1 = 2.*a

    

    if(lattice == 'triangular'): c2 = 0.

    if(lattice == 'square'    ): c2 = c1

    

    scl  = 1.0661

    cut  = 2.5

    rc=cut

    phi0 = np.zeros(np.shape(c11))

    

    for s in range(-3,3):

        for l in range(-3,3):

            if( (s!=0) or (l!=0) ):

                r = scl * np.sqrt( (s**2.)*c11 + 2.*s*l*c12 + (l**2.)*c22 )

#                 tmp = np.where(r > cut, 0., a/(r)**12. - c1*np.exp(-b1*(r-r1)**2.) - c2*np.exp(-b2*(r-r2)**2.) )

                tmp = np.where(r > cut, 0.,-4*(pow(rc,-12) - pow(rc,-6)) + 4*(pow(r,-12) - pow(r,-6)) -  4*(-12/pow(rc,13) + 6/pow(rc,7))*(-rc + r) )



                phi0 += tmp

    phi0=0.5*phi0

    

    return phi0





## Calculate the metric from lattice vectors

def CfromE(e1, e2):

    Cij = np.zeros([2,2])

    Cij[0,0] = e1[0]*e1[0] + e1[1]*e1[1]

    Cij[0,1] = e1[0]*e2[0] + e1[1]*e2[1]

    Cij[1,0] = e1[0]*e2[0] + e1[1]*e2[1]

    Cij[1,1] = e2[0]*e2[0] + e2[1]*e2[1]

    return Cij





def FfromF(f11,f22,f12,f21):

    F = np.zeros([2,2])

    F[0,0] = f11

    F[0,1] = f12

    F[1,0] = f21

    F[1,1] = f22

    return F



    

def fromctodisk(c11,c22,c12,detc):

    yy=np.sqrt(detc)/c22

    xx=np.sqrt(detc)*c12/c22

    boh2=0

    y=(2*xx/(np.power(xx,2)+np.power(1+yy,2)))

    x=((-1+np.power(xx,2)+np.power(yy,2))/(np.power(xx,2)+np.power(1+yy,2)))  

    ray=np.sqrt(np.power(x,2)+np.power(y,2))

    if x>0:

        boh2=np.atan(y/x)  

    elif x<0 and y>=0:

        boh2=np.atan(y/x)+1*np.pi

    elif x<0 and y<0:

        boh2=np.atan(y/x)-1*np.pi

    elif x==0 and y>0:

        boh2=0.5*np.pi

    elif x==0 and y<0:

        boh2=-0.5*np.pi      

    angle=boh2

    return ray,angle

    

def get_simplices(self, vertex):

    "Find all simplices this `vertex` belongs to"

    visited = set()

    queue = [self.vertex_to_simplex[vertex]]

    while queue:

        simplex = queue.pop()

        for i, s in enumerate(self.neighbors[simplex]):

            if self.simplices[simplex][i] != vertex and s != -1 and s not in visited:

                queue.append(s)

        visited.add(simplex)

    return np.array(list(visited))



# matrices mi \in GL(2,Z) for the 3 possible stages of a Lagrange reduction step on the metric components

lag_m1 = np.array([[1., 0.],[0.,-1.]])

lag_m2 = np.array([[0., 1.],[1., 0.]])







lag_m3 = np.array([[1.,-1.],[0., 1.]]) #horizontal shear



lag_m4 = np.array([[1.,0],[-1., 1.]]) #vertical shear

lag_m3_n = np.array([[1.,1.],[0., 1.]])  # - horizontal shear

lag_m4_n = np.array([[1.,0],[1., 1.]]) # - vertical shear







epsilon = 1.e-16



#..........................................................................................

def eps_lt(x, y):

	return x < y 

#..........................................................................................

#..........................................................................................

def eps_gt(x, y):

	return eps_lt(y, x)



#..........................................................................................

def meet_Cij_conditions(c):

#..........................................................................................



	if(eps_lt(   c[2],    0)): return False

	if(eps_lt(   c[1], c[0])): return False

	if(eps_gt(2.*c[2], c[0])): return False

	

	return True



#..........................................................................................

def lagrange_Cij_reduction_step(c, m):

#..........................................................................................

	s1=0

	s2=0

	

	

	if(eps_lt(   c[2],    0)): 

		c[2] = -1 * c[2]

		s1=1

# 		m = np.dot(m, lag_m1)

	if(eps_lt(   c[1], c[0])): 

		c[0], c[1] = c[1], c[0]

		s2=1

# 		m = np.dot(m, lag_m2)

	if(eps_gt(2.*c[2], c[0])):

# 		c[1], c[2] = c[1] + c[0] - 2.*c[2], c[2] - c[0]

		a =  c[1] + c[0] - 2.*c[2]

		b =   c[2] - c[0]

		c[1] = a

		c[2] = b

		if(s1==0 and s2==0):

			m = np.dot(m, lag_m3)

		if(s1==1 and s2==0):

			m = np.dot(m, lag_m3_n)

		if(s1==0 and s2==1):

			m = np.dot(m, lag_m4)

		if(s1==1 and s2==1):

			m = np.dot(m, lag_m4_n)

		

	return c, m

    



#..........................................................................................

def condition_skip(ix,iy,nx,ny):

#..........................................................................................

	i=0

	if(ix==0  or ix == nx-1 or iy==0  or iy == ny-1): i=1

# 	if(iy < 173 or iy > 235 ): i=1

# 	if(ix < 173 or ix > 235 ): i=1

# 	cut = 100

# 	if(iy < ny/2-cut or iy > ny/2+cut): i=1

# 	if(ix < nx/2-cut or ix > nx/2+cut): i=1

    

	

	

	return i



#..........................................................................................



def condition_skip_large(ix,iy,nx,ny):

#..........................................................................................

	i=0

	if(ix==0  or ix == nx-1 or iy==0  or iy == ny-1): i=1

# 	if(iy < 150 or iy > 250 ): i=1

# 	if(ix < 150 or ix > 250 ): i=1

# 	cut = 100

# 	if(iy < ny/2-cut or iy > ny/2+cut): i=1

# 	if(ix < nx/2-cut or ix > nx/2+cut): i=1

    

	

	

	return i



#..........................................................................................



def lagrange_ei_reduction_step(e1, e2):

	i=0

	if(np.linalg.norm(e1) > np.linalg.norm(e2) ):

		e1, e2 = e2, e1

		i=i+1

	if(np.dot(e1, e2) < 0 ):

		e2 = -e2

	if(  np.linalg.norm(e1-e2)< np.linalg.norm(e2)  ):

		e2 = e1 - e2

		i=i+1

	

	return e1, e2,i





def LetticeReduction(e1,e2):

### original vectors

# 	if(np.linalg.norm(e2) > np.linalg.norm(e1)):

# 		while np.linalg.norm(e2) > np.linalg.norm(e2 - e1) or np.linalg.norm(e2) > np.linalg.norm(e2 + e1):

# 			if np.linalg.norm(e2 - e1) < np.linalg.norm(e2 + e1):

# 				e2 = (e2 - e1)

# 			else:

# 				e2 = (e2 + e1)

# 		while np.linalg.norm(e1) > np.linalg.norm(e1 - e2) or np.linalg.norm(e1) > np.linalg.norm(e2 + e1):

# 			if np.linalg.norm(e1 - e2) < np.linalg.norm(e2 + e1):

# 				e1 = (e1 - e2)

# 			else:

# 				e1 = (e2 + e1)

# 

# 	else:

# 		### modify vector e1

# 		while np.linalg.norm(e1) > np.linalg.norm(e1 - e2) or np.linalg.norm(e1) > np.linalg.norm(e2 + e1):

# 

# 			if np.linalg.norm(e1 - e2) < np.linalg.norm(e2 + e1):

# 				e1 = (e1 - e2)

# 			else:

# 				e1 = (e2 + e1)

# 

# 		### modify vector e2

# 		while np.linalg.norm(e2) > np.linalg.norm(e2 - e1) or np.linalg.norm(e2) > np.linalg.norm(e2 + e1):

# 

# 			if np.linalg.norm(e2 - e1) < np.linalg.norm(e2 + e1):

# 				e2 = (e2 - e1)

# 			else:

# 				e2 = (e2 + e1)

	if(np.linalg.norm(e2)>np.linalg.norm(e1)):

# 		print('H')

		### modify vector e2

		while (np.linalg.norm(e2)>np.linalg.norm(e2-e1)) | (np.linalg.norm(e2)>np.linalg.norm(e2+e1)):

			if (np.linalg.norm(e2-e1)<np.linalg.norm(e2+e1)):

				e2_new=(e2-e1)

				if abs(e2_new[1]) > abs(e2_new[0]):

					e2 = e2_new

				else:

					break

			else:

				e2_new=e2+e1

				if abs(e2_new[1]) > abs(e2_new[0]):

					e2 = e2_new

				else:

					break

		### modify vector e1

		while (np.linalg.norm(e1)>np.linalg.norm(e1-e2)) | (np.linalg.norm(e1)>np.linalg.norm(e2+e1)):

			if (np.linalg.norm(e1-e2)<np.linalg.norm(e2+e1)):

				e1_new=(e1-e2)

				if abs(e1_new[0]) > abs(e1_new[1]):

					e1 = e1_new

				else:

					break

			else:

				e1_new=e2+e1

				if abs(e1_new[0]) > abs(e1_new[1]):

					e1 = e1_new

				else:

					break

	else:

		### modify vector e1

		while (np.linalg.norm(e1)>np.linalg.norm(e1-e2)) | (np.linalg.norm(e1)>np.linalg.norm(e2+e1)):

			if (np.linalg.norm(e1-e2)<np.linalg.norm(e2+e1)):

				e1_new=(e1-e2)

				if abs(e1_new[0]) > abs(e1_new[1]):

					e1 = e1_new

				else:

					break

			else:

				e1_new=e2+e1

				if abs(e1_new[0]) > abs(e1_new[1]):

					e1 = e1_new

				else:

					break

		### modify vector e2

		while (np.linalg.norm(e2)>np.linalg.norm(e2-e1)) | (np.linalg.norm(e2)>np.linalg.norm(e2+e1)):

			if (np.linalg.norm(e2-e1)<np.linalg.norm(e2+e1)):

				e2_new=(e2-e1)

				if abs(e2_new[1]) > abs(e2_new[0]):

					e2 = e2_new

				else:

					break

			else:

				e2_new=e2+e1

				if abs(e2_new[1]) > abs(e2_new[0]):

					e2 = e2_new

				else:

					break

		  

	### reduced vectors

	

	return e1,e2

#..........................................................................................

def meet_ei_conditions(e1, e2):



	if(np.linalg.norm(e1) > np.linalg.norm(e2) ): return True

	if(np.dot(e1, e2) < 0 ): return True

	if(  np.linalg.norm(e1-e2)< np.linalg.norm(e2)  ): return True

	

	return False

#..........................................................................................



#..........................................................................................

def ei_lagrange_reduction(e1, e2):



	e1_0  = e1

	e2_0  = e2

	def_C = np.array([[np.dot(e1,e1), np.dot(e1,e2)], [np.dot(e1,e2), np.dot(e2,e2)]])

	count = 0

	i=0

	

	while(meet_ei_conditions(e1, e2)):

		e1, e2,i = lagrange_ei_reduction_step(e1, e2)

		count +=1

	

	red_C = np.array([[np.dot(e1,e1), np.dot(e1,e2)], [np.dot(e1,e2), np.dot(e2,e2)]])

	m = mij_from_ei_and_red_ei(e1_0, e2_0, e1, e2)

# 	verif = np.max(np.abs(red_C - np.dot(np.T, np.dot(def_C, m))))

	return e1, e2,m,i

# 	if(eps_eq(verif, 0)):

# 		return red_C, m

# 	else:

# 		print('reduction failed at ...:', c11, c22, c12, ' ; verif = ', verif)

# 		exit()



def mij_from_ei_and_red_ei(e1, e2, red_e1, red_e2):



	m11 = (e2[1]*red_e1[0] - e2[0]*red_e1[1]) / (e1[0]*e2[1] - e2[0]*e1[1])

	m21 = (e1[0]*red_e1[1] - e1[1]*red_e1[0]) / (e1[0]*e2[1] - e2[0]*e1[1])

	m12 = (e2[1]*red_e2[0] - e2[0]*red_e2[1]) / (e1[0]*e2[1] - e2[0]*e1[1])

	m22 = (e1[0]*red_e2[1] - e1[1]*red_e2[0]) / (e1[0]*e2[1] - e2[0]*e1[1])

	

	m = np.array([[m11, m12],[m21, m22]])

	

	return m









##############################

start_time = time.time()

nx=100

ny=100








print('Lagrange reduction and Poincaré disk projection')





# some definitions.............................................................................................................



print('Lagrange reduction and Poincaré disk projection')





# some definitions.............................................................................................................



# choose Poincare disk discretization

disc = 1000





# descretize 2D space of stereographic projection

x , y  = np.linspace(-.999, .999, num=disc, endpoint=True), np.linspace(-.999, .999, num=disc, endpoint=True)

x_, y_ = np.meshgrid(x, y)





# mask values outside Poincare disk


x_ = np.where( x_**2.+y_**2.-(.999)**2. >= 1.e-6, np.nan, x_)

y_ = np.where( x_**2.+y_**2.-(.999)**2. >= 1.e-6, np.nan, y_)





# recover metric components from stereographic projection with projection center C(-1,-1,0)

c11, c22, c12 = Cij_from_stereographic_projection(x_, y_)







# plot metric components on Poincaré disk

#     config = convert_data_SymLog(c11, 1.e-1)

#     poincare_plot(config, 'cij_ini_c11', np.nanmin(config), np.nanmax(config))

#     

#     config = convert_data_SymLog(c22, 1.e-1)

#     poincare_plot(config, 'cij_ini_c22', np.nanmin(config), np.nanmax(config))

#     

#     config = convert_data_SymLog(c12, 1.e-1)

#     poincare_plot(config, 'cij_ini_c12', np.nanmin(config), np.nanmax(config))











# Lagrange reduction...........................................................................................................



# initialize boolean array and iteraction count 

need_reduction = np.where( np.logical_or( np.logical_or(c12 < 0., c22 < c11), 2.*c12 > c11 ), True, False )

reduction_iter = 0



while(np.any(need_reduction)):



    # Lagrange reduction steps

    c12      = np.where(c12 < 0.    , -1.*c12           , c12)

    c11, c22 = np.where(c22 < c11   ,     c22           , c11), np.where(   c22 < c11, c11    , c22)

    c22, c12 = np.where(2.*c12 > c11, c22 + c11 - 2.*c12, c22), np.where(2.*c12 > c11, c12-c11, c12)



    # update iteration count

    reduction_iter += 1



    # update boolean array

    need_reduction = np.where( np.logical_or( np.logical_or(c12 < 0., c22 < c11), 2.*c12 > c11 ), True, False )



    # verbose

    if(np.mod(reduction_iter, 100)==0): print(reduction_iter, np.nanmin(c11), np.nanmax(c11))









# import metric components from calculation..................................................................................



# 	z     = np.loadtxt("./umut.dat", skiprows=0,usecols=2)

# 	x     = np.loadtxt("./umut.dat", skiprows=0,usecols=0)

# 	y     = np.loadtxt("./umut.dat", skiprows=0,usecols=1)



z= 0 #metrics[:,0,1]

x= 1 #metrics[:,0,0]

y= 1 #metrics[:,1,1]





# recover coordinates on Poincaré disk

px, py = stereographic_projection_from_Cij_2D(x, y, z)



# print min and max of projection coordinates

# print(min(px), max(px))

# print(min(py), max(py))







# compute strain energy density................................................................................................



# choose 2D lattice

#lattice = 'triangular'

lattice = 'triangular'



# choose energy computation

#     phi0 = polynomial_phi0_from_Cij(c11, c22, c12, lattice)

phi0 = interatomic_phi0_from_Cij(c11, c22, c12, lattice)



# transforme data for clarity..................................................................................................

phi0   = (phi0 - np.nanmin(phi0))/(np.nanmax(phi0) - np.nanmin(phi0)) 



c = 1e-12 # for polynomial  phi0

c = 1e-19 # for interatomic phi0



config = convert_data_SymLog(phi0, c)
saving_index=1


name='./figure/cij_phi0_px_py_%s' % (saving_index,) + '.pdf'

poincare_plot_scatter(config, px, py, name, np.nanmin(config), 0.8*np.nanmax(config))



# name2 = './figure_histogram/movie_histo_short_new'+'%s' % (saving_index,)

# poincare_plot_scatter_hist(config, px, py, name2, np.nanmin(config), 0.8*np.nanmax(config))



# 	import matplotlib.colors as mcolors

# 	from matplotlib.colors import LogNorm

# 

# 	plt.figure(figsize=(20,20))

# # 	cbar = plt.colorbar()

# # 	

# # 	cbar.set_label('counts', labelpad=-40, y=1.05, rotation=0)

# # 	cbar.ax.set_ylabel('counts')

# 	bins=100

# 	plt.hist2d(px, py, bins=(bins, bins), cmap=matplotlib.cm.RdYlBu_r,alpha=.9,lw=0,linestyle='None',norm=LogNorm(vmin=1, vmax=1000))

# 	cbar = plt.colorbar()

# 	cbar.set_label('counts', labelpad=-40, y=1.05, rotation=0)

# 	cbar.ax.set_ylabel('counts')

# 

# 	plt.savefig('movie_histo_short_new'+'%s' % (saving_index,)+'.png', format='png', dpi=1200)

# 	plt.cla()

	



# 	end_time = time.time()

# 	print ('Total time: %.4f s' %(end_time - start_time))

##############################





