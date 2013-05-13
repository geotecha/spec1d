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

from sympy import Symbol, sin, cos, integrate, diff, var, symbols, Function, simplify
 
from sympy.tensor import IndexedBase, Idx


def linear(x, x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1) * (x-x1) + y1
    
def phi(m,z):
    return sin(m*z)

def string_to_IndexedBase(s):
    return IndexedBase(s)
        
def create_layer_sympy_var_and_maps(layer_prop=['z','kz','kh','et', 'mv','surz','vacz']):    
    #http://www.daniweb.com/software-development/python/threads/111526/setting-a-string-as-a-variable-name    
    m = IndexedBase('m')
    i = Idx('i')
    j = Idx('j')
    var('z, mi, mj')
    layer = Idx('layer')

    suffix={'t':'top','b': 'bot'}        
    prop_map = {}
    linear_expressions ={}
    
    prop_map['mi'] = m[i]
    prop_map['mj'] = m[j]
    
    for prop in layer_prop:            
        for s1, s3 in suffix.iteritems():
            vars()[prop + s1] = string_to_IndexedBase(prop + s1)            
            var(prop + s3)                        
            prop_map[prop + s3] = vars()[prop + s1][layer]        
        linear_expressions[prop]=linear(z, ztop, eval(prop+suffix['t']), zbot, eval(prop+suffix['b']))
    return (prop_map, linear_expressions)
    
def generate_gam_code():
    
    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','kz','kh','et', 'mv','surz','vacz'])
    
    fdiag = integrate(p['mv'] * phi(mi, z) * phi(mi, z), z)    
    fdiag = fdiag.subs(z, mp['zbot']) - fdiag.subs(z, mp['ztop'])
    fdiag = fdiag.subs(mp)
    
    foff = integrate(p['mv'] * phi(mj, z) * phi(mi, z), z)  
    foff = foff.subs(z, mp['zbot']) - foff.subs(z, mp['ztop'])
    foff = foff.subs(mp)
    
    text = """def make_gam(m, mvt, mvb, zt, zb):
    import numpy.zeros
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


#def runtest():
    
    
if __name__ == '__main__':
    print(generate_gam_code())    
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









