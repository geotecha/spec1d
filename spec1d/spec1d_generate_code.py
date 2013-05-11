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
"""use sympy to generate code for generating spectral method matrix 
subroutines"""

from __future__ import division

import sympy

from sympy import Symbol, sin, cos, integrate, diff, assume
 
from sympy.tensor import IndexedBase, Idx


def linear(x, x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1) * (x-x1) + y1
    
def phi(m,z):
    return sin(m*z)


def generate_gam_code():
    m = IndexedBase('m')
    mvt = IndexedBase('mvt')
    mvb = IndexedBase('mvb')
    zt = IndexedBase('zt')
    zb = IndexedBase('zb')
    z = Symbol('z')
    
    i = Idx('i')
    j = Idx('j')
    layer = Idx('layer')
    
    
    mi = Symbol('mi')
    mj = Symbol('mj')
    ztop = Symbol('ztop')
    zbot = Symbol('zbot')
    mvtop = Symbol('mvtop')
    mvbot = Symbol('mvbot')
        
    mv = linear(z, ztop, mvtop, zbot, mvbot)
    
    #fdiag = integrate(mv * phi(m[i], z) * phi(m[i], z), z)
    fdiag = integrate(mv * phi(mi, z) * phi(mi, z), z)
    fdiag = fdiag.subs([
                        [ztop, zt[layer]], 
                        [mvtop, mvt[layer]],
                        [zbot, zb[layer]],
                        [mvbot, mvb[layer]]
                        ])
    fdiag = fdiag.subs([[mi,m[i]],[mj,m[j]]])                        
    fdiag = fdiag.subs(z, zb[layer]) - fdiag.subs(z, zt[layer])
    
    #foff = integrate(phi(m[i], z) * phi(m[j], z), z)
    foff = integrate(mv * phi(mi, z) * phi(mj, z), z)
    foff = foff.subs([
                        [ztop, zt[layer]], 
                        [mvtop, mvt[layer]],
                        [zbot, zb[layer]],
                        [mvbot, mvb[layer]]
                        ]) 
    foff = foff.subs([[mi,m[i]],[mj,m[j]]])    
    foff = foff.subs(z, zb[layer])-foff.subs(z, zt[layer])
    
    text = """def make_gam(m, mvt, mvb, zt, zb):
    import numpy
    from math import sin, cos
    
    neig = len(m)
    nlayers = len(zt)
    
    gam = numpy.zeros([neig, neig], float)        
    for layer in range(nlayers):
        for i in range(neig):
            gam[i, i] += %s
        for i in range(neig-1):
            for j in range(i + 1, neig):
                gam[i, j] += %s                
                
    #gam is symmetric
    for i in range(neig - 1):        
        for j in range(i + 1, neig):
            gam[j, i] = gam[i, j]                
    
    return gam"""
    
        
    fn = text % (fdiag, foff)
        
    return fn

def generate_psi_code():
    m = IndexedBase('m')
    kvt = IndexedBase('kvt')
    kvb = IndexedBase('kvb')
    kht = IndexedBase('kht')
    khb = IndexedBase('khb')
    ett = IndexedBase('ett')
    etb = IndexedBase('etb')
    zt = IndexedBase('zt')
    zb = IndexedBase('zb')
    dTv = Symbol('z')
    dTh = Symbol('z')
    dT = Symbol('z')
    
    z = Symbol('z')
    
    i = Idx('i')
    j = Idx('j')
    layer = Idx('layer')
    
#    fdiag = integrate(diff(phi(m[i], z),z,2) * phi(m[i], z), z)-
#    fdiag = #fdiag.subs(z, zb[layer]) - fdiag.subs(z, zt[layer])
#    
#    foff = #integrate(phi(m[i], z) * phi(m[j], z), z)
#    foff = #fdiag.subs(z, zb[layer])-fdiag.subs(z, zt[layer])
    
    text = """def make_psi(m, kvt, kvb, kht, khb, ett, etb, 
                           zt, zb, dTv, dTh = 0.0, dT = 1.0):
        import numpy
        neig = len(m)
        nlayers = len(zt)
        
        psi = numpy.zeros([neig, neig], float)        
        for layer in range(nlayer):
            for i in range(neig - 1):
                psi[i, i] += %s
                for j in range(i + 1, neig):
                    psi[i, j] += %s                
                    
        #psi is symmetric
        for i in range(neig - 1):        
            for j in range(i + 1, neig):
                psi[j, i] = psi[i, j]                
        
        return psi"""
    
        
    fn = text % (fdiag, foff)
        
    return fn


if __name__ == '__main__':
    print(generate_gam_code())
    print
    #print(generate_psi_code())
    
#==============================================================================
# from sympy import symbols
# from sympy.utilities.codegen import codegen
# from sympy import Eq
# n,m = symbols('n m', integer=True)
# A = IndexedBase('A')
# x = IndexedBase('x')
# y = IndexedBase('y')
# i = Idx('i', m)
# j = Idx('j', n)
#  
# [(c_name, c_code), (h_name, c_header)] = \
# codegen(name_expr=('matrix_vector', Eq(y[i], A[i, j]*x[j])), language = "F95",prefix = "file", header=False,
#  
# empty=False)
# print c_name
# print c_code
# print h_name
# print c_header
#==============================================================================









