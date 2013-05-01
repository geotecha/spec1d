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
"""The basis_function module contains functions for forming the basis 
    functions and eigenvalues for spectral methods
    
    """

def eigenvalues_of_sine(i, boundary=0):
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