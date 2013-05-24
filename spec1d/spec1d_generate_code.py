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

def linear(x, x1, y1, x2, y2):
    """Interpolation between two points.

    """
    return (y2 - y1)/(x2 - x1) * (x-x1) + y1
    
def phi(m,z):
    """phi function

    """
    
    return sympy.sin(m*z)

def string_to_IndexedBase(s):
    """turn string into sympy.tensor.IndexedBase

    """
    
    return sympy.tensor.IndexedBase(s)
        
def create_layer_sympy_var_and_maps(layer_prop=['z','kv','kh','et', 'mv',
                                                'surz','vacz']):
    """Create sympy variables and maps for use with integrating \
    to generate 1d spectral equations.
    
    Each x in layer prop will get a 'top', and 'bot', suffix
    Each 'xtop' will get mapped to 'xt[layer]', each 'xbot' to 'xb[layer] 
    and added to `prop_map` dict.
    Each s in layer prop will get a linear representation added to the 
    `linear_expression` dict:
        x = (xbot-xtop)/(zbot-ztop)*(z-zbot) +xtop
    `prop_map` will also get a 'mi' to m[i], and mj to m[j] map.
    'z, mi, mj' will become global variables
            
    Parameters
    ----------
    layer_prop : list of str, optional
        label for properties that vary in a layer
    
         
    Returns
    -------
    prop_map : dict
        maps the string version of a variable to the sympy.tensor.IndexedBase 
        version e.g. prop_map['kvtop'] = kvt[layer]
    linear_expressions: dict
        maps the string version of a variable to an expression describing how 
        that varibale varies linearly within a layer    
        
    Examples
    --------
    >>> prop_map, linear_expressions = create_layer_sympy_var_and_maps(layer_prop=['
z','kv'])
    >>> prop_map
    {'kvtop': kvt[layer], 'mi': m[i], 'mj': m[j], 'zbot': zb[layer], 'ztop': zt[laye
    r], 'kvbot': kvb[layer]}
    >>> linear_expressions
    {'z': z, 'kv': kvtop + (kvbot - kvtop)*(z - ztop)/(zbot - ztop)}

    """
    #http://www.daniweb.com/software-development/python/threads/111526/setting-a-string-as-a-variable-name    
    m = sympy.tensor.IndexedBase('m')
    i = sympy.tensor.Idx('i')
    j = sympy.tensor.Idx('j')
    layer = sympy.tensor.Idx('layer')
    
    sympy.var('z, mi, mj')
    suffix={'t':'top','b': 'bot'}        
    prop_map = {}
    linear_expressions ={}
    
    prop_map['mi'] = m[i]
    prop_map['mj'] = m[j]
    
    for prop in layer_prop:            
        for s1, s3 in suffix.iteritems():
            vars()[prop + s1] = string_to_IndexedBase(prop + s1)            
            sympy.var(prop + s3)                        
            prop_map[prop + s3] = vars()[prop + s1][layer]        
        linear_expressions[prop]=linear(z, ztop, eval(prop+suffix['t']), zbot, eval(prop+suffix['b']))
    return (prop_map, linear_expressions)
    
def generate_gam_code():
    """Perform integrations and output a function that will generate gam (without docstring).
    
    Paste the resulting code (at least the loops) into make_gam.

    """
    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','mv'])
    
    fdiag = sympy.integrate(p['mv'] * phi(mi, z) * phi(mi, z), z)    
    fdiag = fdiag.subs(z, mp['zbot']) - fdiag.subs(z, mp['ztop'])
    fdiag = fdiag.subs(mp)
    
    foff = sympy.integrate(p['mv'] * phi(mj, z) * phi(mi, z), z)  
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
    """Perform integrations and output the function that will generate psi (without docstring).

    Paste the resulting code (at least the loops) into make_psi.
    """
    sympy.var('dTv, dTh, dT')    
    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','kv','kh','et'])
        
    fdiag = dTh / dT * sympy.integrate(p['kh'] * p['et'] * phi(mi, z) * phi(mi, z), z)
    fdiag += dTv / dT * (
        sympy.integrate(p['kv'] * sympy.diff(phi(mi, z), z, 2) * phi(mi,z), z))
    fdiag -= dTv / dT * (sympy.diff(p['kv'], z) * sympy.diff(phi(mi, z), z) * phi(mi, z))             
        # note the negative for the diff (kv) is because the step function 
        #at the top and bottom of the layer yields a dirac function that is 
        #positive at ztop and negative at zbot. It works because definite 
        #integral of f between ztop and zbot is F(ztop)- F(zbot). 
        #i.e. I've created the indefininte integral such that when i sub 
        #in ztop and zbot in the next step i get the correct contribution 
        #for the step functions at ztop and zbot            
    fdiag = fdiag.subs(z, mp['zbot']) - fdiag.subs(z, mp['ztop'])
    fdiag = fdiag.subs(mp)
    
    foff = dTh / dT * sympy.integrate(p['kh'] * p['et'] * phi(mj, z) * phi(mi, z), z)
    foff += dTv / dT * (
        sympy.integrate(p['kv'] * sympy.diff(phi(mj, z), z, 2) * phi(mi,z), z))
    foff -= dTv / dT * (sympy.diff(p['kv'], z) * sympy.diff(phi(mj, z), z) * phi(mi, z))
                     
    foff = foff.subs(z, mp['zbot']) - foff.subs(z, mp['ztop'])
    foff = foff.subs(mp)
    
    text = """def make_psi(m, kvt, kvb, kht, khb, ett, etb, zt, zb, dTv, dTh, dT = 1.0):
    import numpy.zeros
    from math import sin, cos
    
    neig = len(m)
    nlayers = len(zt)
    
    psi = numpy.zeros([neig, neig], float)        
    for layer in range(nlayers):
        for i in range(neig):
            psi[i, i] += %s
        for i in range(neig-1):
            for j in range(i + 1, neig):
                psi[i, j] += %s                
                
    #psi is symmetric
    for i in range(neig - 1):        
        for j in range(i + 1, neig):
            psi[j, i] = psi[i, j]                
    #one day I'll know exacly why it's -1 * psi
    return psi"""
    
        
    fn = text % (fdiag, foff)
        
    return fn



    
    
if __name__ == '__main__':
    print(generate_gam_code())
    print '#'*65
    print(generate_psi_code())
    pass
        
    
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









