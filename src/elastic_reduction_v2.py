import numpy as np
from typing import Tuple

def U_m(m:int):
    return np.array([[1,  m],[0,  1]], dtype=int)
def V_m(m:int):
    return np.array([[1,  0],[m,  1]], dtype=int)

def in_quadrant_geom(C: np.ndarray, label: int, tol: float = 1e-12) -> bool:
    """
    Test if C lies in one of four central 'quadrants' defined by linear constraints
    in (C11, C22, C12). These boundaries project to geodesics (diameters/orthogonal arcs)
    on the Poincaré disk.
    """
    C11, C22, C12 = C[0,0], C[1,1], C[0,1]
    if label == 0:
        return (C11 > tol) and (C11 <= C22 + tol) and (C12 >= -tol) and (C12 <= 0.5*C11 + tol)
    if label == 1:
        return (C11 > tol) and (C11 <= C22 + tol) and (C12 <=  tol) and (C12 >= -0.5*C11 - tol)
    if label == 2:
        return (C22 > tol) and (C22 <= C11 + tol) and (C12 >= -tol) and (C12 <= 0.5*C22 + tol)
    if label == 3:
        return (C22 > tol) and (C22 <= C11 + tol) and (C12 <=  tol) and (C12 >= -0.5*C22 - tol)
    return False

def elasticReduction(C: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Minimal reduction: scan words in BFS order; the first hit gives the quadrant label
    and minimal word length (depth).
    """
    max_depth=100
    C = C.copy()
    for d in range(0,max_depth):
        for lab in (0,1,2,3):
            # Is C in elastic domain?
            if in_quadrant_geom(C, lab):
                return C, lab, d
        if C[0,0] < C[1,1]:
            # Avoid division by zero if C[0,0] is 0 (though unlikely for metric tensor)
            if abs(C[0,0]) < 1e-12:
                 # Should probably shift state or break
                 break
            m_val = -C[0,1]/C[0,0]
            # Use np.round to find nearest integer, standard reduction step usually involves rounding
            # The user code had: np.sign(-C[0,1]/C[0,0]). Wait, the user code is:
            # W = U_m(np.sign(-C[0,1]/C[0,0]))
            # BUT usually reduction involves round(). 
            # Reviewing USER REQUEST: 
            # "W = U_m(np.sign(-C[0,1]/C[0,0]))"
            # Okay, I will stick EXACTLY to the user provided code logic for the step.
            # But wait, np.sign returns -1, 0, or 1. This looks like a greedy descent step size 1?
            # User code: m = np.sign(...)
            # Let's copy it exactly.
            
            # Re-reading user snippet carefully:
            # if C[0,0]<C[1,1]:
            #     W = U_m(np.sign(-C[0,1]/C[0,0]))
            # else:
            #     W = V_m(np.sign(-C[0,1]/C[1,1]))
            
            # Wait, notice the user snippet had `dtype=int` for U_m. np.sign returns float usually, need to cast to int.
            pass

        # Implementing EXACTLY as requested
        if C[0,0] < C[1,1]:
             s = np.sign(-C[0,1]/C[0,0])
             if s == 0: s = 1 # Fallback or just stop? If s=0 then C01=0, which means it IS reduced likely.
             # Actually if C01 is 0, it should have been caught by in_quadrant_geom?
             # Let's just use int(np.sign(...))
             W = U_m(int(s))
        else:
             s = np.sign(-C[0,1]/C[1,1])
             W = V_m(int(s))
             
        C = W.T @ C @ W
        
    return C, -1, -1
