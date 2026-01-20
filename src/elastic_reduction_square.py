import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os
import sys
import warnings
matplotlib.use('Agg')



epsilon = 1.e-16


# ..........................................................................................
def eps_lt(x, y):
    return x < y 


def eps_gt(x, y):
    return eps_lt(y, x)
    
def meet_Cij_conditions(c):

    if (eps_lt(c[2], 0)): return False
    if (eps_lt(c[1], c[0])): return False
    if (eps_gt(2. * c[2], c[0])): return False

    return True


def lagrange_Cij_reduction_step_original_v3(c, m):
    s1=0
    s2=0
    
    m1 = np.array([[1., 0.],[0., 1.]])
    m2 = np.array([[1., 0.],[0., 1.]])
    m3 = np.array([[1., 0.],[0., 1.]])


    lag_m1 = np.array([[1., 0.], [0., -1.]])
    lag_m2 = np.array([[0., 1.], [1., 0.]])
    
    lag_m3 = np.array([[1., -1.], [0., 1.]])  # horizontal shear

    
    if(eps_lt(   c[2],    0)): 
        c[2] = -1 * c[2]
        m = np.dot(m, lag_m1)
        m1 = lag_m1
    if(eps_lt(   c[1], c[0])): 
        c[0], c[1] = c[1], c[0]
        m = np.dot(m, lag_m2)
        m2 = lag_m2
    if(eps_gt(2.*c[2], c[0])):
#         c[1], c[2] = c[1] + c[0] - 2.*c[2], c[2] - c[0]
        a =  c[1] + c[0] - 2.*c[2]
        b =   c[2] - c[0]
        c[1] = a
        c[2] = b
        m = np.dot(m, lag_m3)
        m3 = lag_m3
    
    
    m_cap =   np.matmul(m1,m2)
    m_cap =   np.matmul(m_cap,m3)
    m_cap =   np.matmul(m_cap,m2)
    m_cap =   np.matmul(m_cap,m1)
    

    
        
    return c,m,m_cap


def reduction_elastic_square(cij):
          
    c = np.zeros(3)
    c[0] = cij[0,0]
    c[1] = cij[1,1]
    c[2] = cij[0,1]
    m = np.array([[1., 0.],[0., 1.]])
    m_cap = np.array([[1., 0.],[0., 1.]])
    m_cap_temp = np.array([[1., 0.],[0., 1.]])
    metrico= np.array([[c[0], c[2]],[c[2], c[1]]])
    metric= np.array([[c[0], c[2]],[c[2], c[1]]])
    
    logi=False
    while(not logi):
        c2, m, m_cap_temp   = lagrange_Cij_reduction_step_original_v3(c, m)
        logi=meet_Cij_conditions(c2)
        m_cap = np.matmul(m_cap,m_cap_temp)
        metric=np.matmul(np.matmul(np.transpose(m_cap),metrico),m_cap)
        c[0] = metric[0,0]
        c[1] = metric[1,1]
        c[2] = metric[0,1]
    
    cij_2 = np.array([[1., 0.],[0., 1.]])
    cij_2[0,0] = c[0]
    cij_2[1,1] = c[1]
    cij_2[0,1] = c[2]
    cij_2[1,0] = c[2]

    
    return cij_2

# Cr = np.empty((2, 2))
# Cr[0,0] = 1
# Cr[1,1] = 10.2717
# Cr[0,1] =  -3.04315
# Cr[1,0] =  -3.04315
# print(Cr)

# Ct=reduction_elastic_square(Cr)
# print(Ct)


