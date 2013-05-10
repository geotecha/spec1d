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
"""test routines for the basis_function module
Shows a few approaches to testing using the 
spec1d.basis_functions.eigenvalues_of_sine function as an example

"""

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
import numpy as np

import spec1d.basis_functions


### start Method 1: global vars 
###     (not recommended as it uses global variables)
PTPB = None
PTIB = None

def setup():
    """setup fn for eigenvalues_of_sine tests using global variables"""
    global PTPB, PTIB
    PTPB = [3.14159, 6.28319, 9.42478, 12.56637, 15.70796, 18.84956, 21.99115]
    PTIB = [1.57080, 4.71239, 7.85398, 10.99557, 14.13717, 17.27876, 20.42035]

def teardown():
    """teardown fn for eigenvalues_of_sine tests using global variables"""
    global PTPB, PTIB
    PTPB = None
    PTIB = None

@with_setup(setup, teardown)    
def test_eigenvalues_of_sine_bc_0v1():
    """eigenvalues_of_sine tests using global vars, i = 0, boundary = 0"""       
    m0 = spec1d.basis_functions.eigenvalues_of_sine(0, 0)    
    assert_almost_equal(m0, PTPB[0], 5)
    
@with_setup(setup)
def test_eigenvalues_of_sine_bc_1v1():
    """eigenvalues_of_sine tests using global vars, i = 0, boundary = 1"""        
    m0 = spec1d.basis_functions.eigenvalues_of_sine(0, 1)    
    assert_almost_equal(m0, PTIB[0], 5)
### end Method 1

### start Method 2: self contained
###     (ok for very simply tests but will lead to repeated data)

def test_eigenvalues_of_sine_bc_0v2():
    """eigenvalues_of_sine tests, self contained, i = 0, boundary = 0"""        
    m0 = spec1d.basis_functions.eigenvalues_of_sine(0, 0)    
    assert_almost_equal(m0, 3.14159, 5)
    
def test_eigenvalues_of_sine_bc_1v2():        
    """eigenvalues_of_sine tests, self contained, i = 0, boundary = 1"""
    m0 = spec1d.basis_functions.eigenvalues_of_sine(0, 1)    
    assert_almost_equal(m0, 1.57080, 5)
### end Method 2


### start Method 3: classes
###     (better than 1 and 2 when fixtures are needed)
class test_eigenvalues_of_sine(object):
    """A suite of tests for spec1d.basis_functions.eigenvalues_of_sine
    Shows two approaches: individual methods and looping through a list
    
    """
    
    def __init__(self):        
        self.PTPB = [3.14159, 6.28319, 9.42478, 12.56637, 
                     15.70796, 18.84956, 21.99115]
        self.PTIB = [1.57080, 4.71239, 7.85398, 10.99557, 
                     14.13717, 17.27876, 20.42035]
        
                     
        self.cases = [ #used for generator example
            [(0, 0), 3.14159],
            [(0, 1), 1.57080],
            [(1, 0), 6.28319],
            [(1, 1), 4.71239],
            [(np.array(range(7)), 0), self.PTPB],
            [(np.array(range(7)), 0), self.PTIB],
            ] #then you canjust add more cases
    def test_bc0(self):
        """test i = 0, boundary = 0"""
        m0 = spec1d.basis_functions.eigenvalues_of_sine(0,0)    
        assert_almost_equal(m0, self.PTPB[0], 5)
        
    def test_bc1(self): 
        """test i = 0, boundary = 1"""
        m0 = spec1d.basis_functions.eigenvalues_of_sine(0,1)    
        assert_almost_equal(m0, self.PTIB[0], 5)

    def test_numpy(self):
        """test a numpy array as input to i; i = range(7), boundary = 0"""
        x = np.array(range(7))
        y0 = spec1d.basis_functions.eigenvalues_of_sine(x,0)
        assert np.allclose(y0,self.PTPB)
    
    
    ### a generator example
    def test_cases(self):
        """loop through and test cases with numpy.allclose"""
        for fixture,result in self.cases:
            m = spec1d.basis_functions.eigenvalues_of_sine(*fixture)
            yield np.allclose, m, result
            
    
            
### end Method 3        

def test_eigenvalues_of_sine_bad_boundary():
    """eigenvalues_of_sine fail test, self contained, i = 2, boundary = 1.5"""
    assert_raises(ValueError,
                  spec1d.basis_functions.eigenvalues_of_sine , 2, 1.5)