#==============================================================================
#  spec1D - Vertical and radial soil consolidation using the spectral method
#  Copyright (C) 2013  Rohan T. Walker (rtrwalker@gmail.com)
# 
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see http://www.gnu.org/licenses/gpl.html.
#==============================================================================
"""Equations for one dimesnional spectral consolidation
    
"""

from __future__ import division

def m_func(i, boundary=0):
    """Sine series eigenvalue of boundary value problem on [0, 1]
    
    
    `i` th eigenvalue, M,  of f(x) = sin(M*x) that satisfies:
        f(0) = 0, f(1) = 0; for `boundary` = 0 i.e. PTPB
        f(0) = 0; f'(1) = 0; for `boundary` = 1 i.e. PTIB
        
    Parameters
    ----------
    i : int
        eigenvalue  in series to return
    boundary : {0, 1}, optional
        boundary condition. 
        For 'Pervious Top Pervious Bottom (PTPB)', boundary = 0
        For 'Pervious Top Impervious Bottom (PTIB)', boundary = 1
         
    Returns
    -------
    out : float
        returns the `i` th eigenvalue
        
    """    
    from math import pi    
    
    if boundary not in {0, 1}:
        raise ValueError('boundary = %s; must be 0 or 1.' % (boundary))
        
    return pi * (i + 1 - boundary / 2.0)
    
def make_gam(m, mvt, mvb, zt, zb):
    """Create and populate the gam matrix for spec1d
        
    
    The gam matrix depends on depth dependance of volume compressibility (mv)
        
    Parameters
    ----------
    m : list of float
        eigenvlaues of BVP. generate with spec1d.m_func
    mvt : list of float
        volume compressibility at top of each layer
    mvb : list of float
        volume compressibility at bottom of each layer        
    zt : list of float
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : list of float
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
             
    Returns
    -------
    gam : numpy.ndarray
        returns the square symmetric gam matrix, size determined by size of `m`
        
    See Also
    --------
    mfunc : used to generate 'm' input parameter
    
    Notes
    -----
    The :math:`\\mathbf{\\Gamma}` matrix arises when integrating the depth 
    dependant volume compressibility (:math:`m_v` against the spectral basis 
    functions:
    
    .. math:: \\mathbf{\\Gamma}_{i,j}=\\int_{0}^1{\\frac{m_v}{\\overline{m}_v}\\phi_i\\phi_j\\,dZ}
    
    """
    
    from numpy import zeros
    from math import sin, cos
    
    neig = len(m)
    nlayers = len(zt)
    
    gam = zeros([neig, neig], float)        
    for layer in range(nlayers):
        for i in range(neig):
            gam[i, i] += -mvb[layer]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*cos(m[i]*zb[layer])**2 + mvb[layer]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*cos(m[i]*zt[layer])**2 - 2*mvb[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) + 2*mvb[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) + mvb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zb[layer]**2*sin(m[i]*zb[layer])**2 + mvb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zb[layer]**2*cos(m[i]*zb[layer])**2 - mvb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zt[layer]**2*sin(m[i]*zt[layer])**2 - mvb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zt[layer]**2*cos(m[i]*zt[layer])**2 + mvt[layer]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*cos(m[i]*zb[layer])**2 - mvt[layer]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*cos(m[i]*zt[layer])**2 + 2*mvt[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) - 2*mvt[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) - mvt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zb[layer]**2*sin(m[i]*zb[layer])**2 - mvt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zb[layer]**2*cos(m[i]*zb[layer])**2 + mvt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zt[layer]**2*sin(m[i]*zt[layer])**2 + mvt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zt[layer]**2*cos(m[i]*zt[layer])**2 - 2*zb[layer]*mvt[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) + 2*zb[layer]*mvt[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) + 2*zb[layer]*mvt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zb[layer]*sin(m[i]*zb[layer])**2 + 2*zb[layer]*mvt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zb[layer]*cos(m[i]*zb[layer])**2 - 2*zb[layer]*mvt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zt[layer]*sin(m[i]*zt[layer])**2 - 2*zb[layer]*mvt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zt[layer]*cos(m[i]*zt[layer])**2 + 2*zt[layer]*mvb[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) - 2*zt[layer]*mvb[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) - 2*zt[layer]*mvb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zb[layer]*sin(m[i]*zb[layer])**2 - 2*zt[layer]*mvb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zb[layer]*cos(m[i]*zb[layer])**2 + 2*zt[layer]*mvb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zt[layer]*sin(m[i]*zt[layer])**2 + 2*zt[layer]*mvb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*zt[layer]*cos(m[i]*zt[layer])**2
        for i in range(neig-1):
            for j in range(i + 1, neig):
                gam[i, j] += mvb[layer]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) - mvb[layer]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) - mvb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) + mvb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) + 2*mvb[layer]*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) - 2*mvb[layer]*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) + mvb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) - mvb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) + mvb[layer]*m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) - mvb[layer]*m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) + mvb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) - mvb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) - mvb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) + mvb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) - mvt[layer]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) + mvt[layer]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) + mvt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) - mvt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) - 2*mvt[layer]*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) + 2*mvt[layer]*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) - mvt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) + mvt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) - mvt[layer]*m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) + mvt[layer]*m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) - mvt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) + mvt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) + mvt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) - mvt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) - zb[layer]*mvt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) + zb[layer]*mvt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) + zb[layer]*mvt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) - zb[layer]*mvt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) + zb[layer]*mvt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) - zb[layer]*mvt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) - zb[layer]*mvt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) + zb[layer]*mvt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) + zt[layer]*mvb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) - zt[layer]*mvb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) - zt[layer]*mvb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) + zt[layer]*mvb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) - zt[layer]*mvb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) + zt[layer]*mvb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) + zt[layer]*mvb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) - zt[layer]*mvb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])                
                
    #gam is symmetric
    for i in range(neig - 1):        
        for j in range(i + 1, neig):
            gam[j, i] = gam[i, j]                
    
    return gam

    
