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
    
    
    `i`th eigenvalue, M,  of f(x) = sin(M*x) that satisfies:
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
         returns the `i'th eigenvalue
         
    """    
    from math import pi    
    
    if boundary not in {0, 1}:
        raise ValueError('boundary = %s; must be 0 or 1.' % (boundary))
        
    return pi * (i + 1 - boundary / 2.0)
    
def make_gam(m, mvt, mvb, zt, zb):
    import numpy
    from math import sin, cos
    
    neig = len(m)
    nlayers = len(zt)
    
    gam = numpy.zeros([neig, neig], float)        
    for layer in range(nlayers):
        for i in range(neig):
            gam[i, i] += -cos(zb[layer]*m[i])**2*mvb[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + cos(zb[layer]*m[i])**2*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + cos(zt[layer]*m[i])**2*mvb[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - cos(zt[layer]*m[i])**2*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + 2*m[i]*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])*mvb[layer]*zt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - 2*m[i]*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])*zb[layer]*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - 2*m[i]*cos(zt[layer]*m[i])*sin(zt[layer]*m[i])*mvb[layer]*zt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + 2*m[i]*cos(zt[layer]*m[i])*sin(zt[layer]*m[i])*zb[layer]*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - 2*zb[layer]*m[i]*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])*mvb[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + 2*zb[layer]*m[i]*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - 2*zb[layer]*m[i]**2*sin(zb[layer]*m[i])**2*mvb[layer]*zt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + 2*zb[layer]*m[i]**2*sin(zb[layer]*m[i])**2*zb[layer]*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - 2*zb[layer]*m[i]**2*cos(zb[layer]*m[i])**2*mvb[layer]*zt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + 2*zb[layer]*m[i]**2*cos(zb[layer]*m[i])**2*zb[layer]*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + zb[layer]**2*m[i]**2*sin(zb[layer]*m[i])**2*mvb[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - zb[layer]**2*m[i]**2*sin(zb[layer]*m[i])**2*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + zb[layer]**2*m[i]**2*cos(zb[layer]*m[i])**2*mvb[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - zb[layer]**2*m[i]**2*cos(zb[layer]*m[i])**2*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + 2*zt[layer]*m[i]*cos(zt[layer]*m[i])*sin(zt[layer]*m[i])*mvb[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - 2*zt[layer]*m[i]*cos(zt[layer]*m[i])*sin(zt[layer]*m[i])*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + 2*zt[layer]*m[i]**2*sin(zt[layer]*m[i])**2*mvb[layer]*zt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - 2*zt[layer]*m[i]**2*sin(zt[layer]*m[i])**2*zb[layer]*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + 2*zt[layer]*m[i]**2*cos(zt[layer]*m[i])**2*mvb[layer]*zt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - 2*zt[layer]*m[i]**2*cos(zt[layer]*m[i])**2*zb[layer]*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - zt[layer]**2*m[i]**2*sin(zt[layer]*m[i])**2*mvb[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + zt[layer]**2*m[i]**2*sin(zt[layer]*m[i])**2*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) - zt[layer]**2*m[i]**2*cos(zt[layer]*m[i])**2*mvb[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1) + zt[layer]**2*m[i]**2*cos(zt[layer]*m[i])**2*mvt[layer]*(4*m[i]**2*zb[layer] - 4*m[i]**2*zt[layer])**(-1)
        for i in range(neig-1):
            for j in range(i + 1, neig):
                gam[i, j] += sin(zb[layer]*m[j])*m[i]**2*sin(zb[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - sin(zb[layer]*m[j])*m[i]**2*sin(zb[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + sin(zb[layer]*m[j])*m[i]**3*cos(zb[layer]*m[i])*mvb[layer]*zt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - sin(zb[layer]*m[j])*m[i]**3*cos(zb[layer]*m[i])*zb[layer]*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - sin(zt[layer]*m[j])*m[i]**2*sin(zt[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + sin(zt[layer]*m[j])*m[i]**2*sin(zt[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - sin(zt[layer]*m[j])*m[i]**3*cos(zt[layer]*m[i])*mvb[layer]*zt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + sin(zt[layer]*m[j])*m[i]**3*cos(zt[layer]*m[i])*zb[layer]*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + 2*m[j]*cos(zb[layer]*m[j])*m[i]*cos(zb[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - 2*m[j]*cos(zb[layer]*m[j])*m[i]*cos(zb[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - m[j]*cos(zb[layer]*m[j])*m[i]**2*sin(zb[layer]*m[i])*mvb[layer]*zt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + m[j]*cos(zb[layer]*m[j])*m[i]**2*sin(zb[layer]*m[i])*zb[layer]*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - 2*m[j]*cos(zt[layer]*m[j])*m[i]*cos(zt[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + 2*m[j]*cos(zt[layer]*m[j])*m[i]*cos(zt[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + m[j]*cos(zt[layer]*m[j])*m[i]**2*sin(zt[layer]*m[i])*mvb[layer]*zt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - m[j]*cos(zt[layer]*m[j])*m[i]**2*sin(zt[layer]*m[i])*zb[layer]*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + m[j]**2*sin(zb[layer]*m[j])*sin(zb[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - m[j]**2*sin(zb[layer]*m[j])*sin(zb[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - m[j]**2*sin(zb[layer]*m[j])*m[i]*cos(zb[layer]*m[i])*mvb[layer]*zt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + m[j]**2*sin(zb[layer]*m[j])*m[i]*cos(zb[layer]*m[i])*zb[layer]*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - m[j]**2*sin(zt[layer]*m[j])*sin(zt[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + m[j]**2*sin(zt[layer]*m[j])*sin(zt[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + m[j]**2*sin(zt[layer]*m[j])*m[i]*cos(zt[layer]*m[i])*mvb[layer]*zt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - m[j]**2*sin(zt[layer]*m[j])*m[i]*cos(zt[layer]*m[i])*zb[layer]*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + m[j]**3*cos(zb[layer]*m[j])*sin(zb[layer]*m[i])*mvb[layer]*zt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - m[j]**3*cos(zb[layer]*m[j])*sin(zb[layer]*m[i])*zb[layer]*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - m[j]**3*cos(zt[layer]*m[j])*sin(zt[layer]*m[i])*mvb[layer]*zt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + m[j]**3*cos(zt[layer]*m[j])*sin(zt[layer]*m[i])*zb[layer]*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - zb[layer]*sin(zb[layer]*m[j])*m[i]**3*cos(zb[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + zb[layer]*sin(zb[layer]*m[j])*m[i]**3*cos(zb[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + zb[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**2*sin(zb[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - zb[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**2*sin(zb[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + zb[layer]*m[j]**2*sin(zb[layer]*m[j])*m[i]*cos(zb[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - zb[layer]*m[j]**2*sin(zb[layer]*m[j])*m[i]*cos(zb[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - zb[layer]*m[j]**3*cos(zb[layer]*m[j])*sin(zb[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + zb[layer]*m[j]**3*cos(zb[layer]*m[j])*sin(zb[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + zt[layer]*sin(zt[layer]*m[j])*m[i]**3*cos(zt[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - zt[layer]*sin(zt[layer]*m[j])*m[i]**3*cos(zt[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - zt[layer]*m[j]*cos(zt[layer]*m[j])*m[i]**2*sin(zt[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + zt[layer]*m[j]*cos(zt[layer]*m[j])*m[i]**2*sin(zt[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - zt[layer]*m[j]**2*sin(zt[layer]*m[j])*m[i]*cos(zt[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + zt[layer]*m[j]**2*sin(zt[layer]*m[j])*m[i]*cos(zt[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) + zt[layer]*m[j]**3*cos(zt[layer]*m[j])*sin(zt[layer]*m[i])*mvb[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1) - zt[layer]*m[j]**3*cos(zt[layer]*m[j])*sin(zt[layer]*m[i])*mvt[layer]*(m[i]**4*zb[layer] - m[i]**4*zt[layer] - 2*m[j]**2*m[i]**2*zb[layer] + 2*m[j]**2*m[i]**2*zt[layer] + m[j]**4*zb[layer] - m[j]**4*zt[layer])**(-1)                
                
    #gam is symmetric
    for i in range(neig - 1):        
        for j in range(i + 1, neig):
            gam[j, i] = gam[i, j]                
    
    return gam

    
