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
"""test routines for the spec1d module
Shows a few approaches to testing using the 
m_func function as an example

"""

from nose import with_setup
from nose.tools.trivial import assert_almost_equal, assert_raises, ok_
import numpy as np

from spec1d.spec1d import m_func
from spec1d.spec1d import make_gam

### start Method 1: global vars 
###     (not recommended as it uses global variables)
PTPB = None
PTIB = None

def setup():
    """setup fn for m_func tests using global variables"""
    global PTPB, PTIB
    PTPB = [3.14159, 6.28319, 9.42478, 12.56637, 15.70796, 18.84956, 21.99115]
    PTIB = [1.57080, 4.71239, 7.85398, 10.99557, 14.13717, 17.27876, 20.42035]

def teardown():
    """teardown fn for m_func tests using global variables"""
    global PTPB, PTIB
    PTPB = None
    PTIB = None

@with_setup(setup, teardown)    
def test_m_func_bc_0v1():
    """m_func tests using global vars, i = 0, boundary = 0"""       
        
    m0 = m_func(0, 0)    
    assert_almost_equal(m0, PTPB[0], 5)
    
@with_setup(setup)
def test_m_func_bc_1v1():
    """m_func tests using global vars, i = 0, boundary = 1"""        
    m0 = m_func(0, 1)    
    assert_almost_equal(m0, PTIB[0], 5)
### end Method 1

### start Method 2: self contained
###     (ok for very simply tests but will lead to repeated data)

def test_m_func_bc_0v2():
    """m_func tests, self contained, i = 0, boundary = 0"""        
    m0 = m_func(0, 0)    
    assert_almost_equal(m0, 3.14159, 5)
    
def test_m_func_bc_1v2():        
    """m_func tests, self contained, i = 0, boundary = 1"""
    m0 = m_func(0, 1)    
    assert_almost_equal(m0, 1.57080, 5)
### end Method 2


### start Method 3: classes
###     (better than 1 and 2 when fixtures are needed)
class test_m_func(object):
    """A suite of tests for m_func
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
            [(np.array(range(7)), 1), self.PTIB],
            ] #then you canjust add more cases
                
    def test_bc0(self):
        """test i = 0, boundary = 0"""
        m0 = m_func(0,0)    
        assert_almost_equal(m0, self.PTPB[0], 5)
        
    def test_bc1(self): 
        """test i = 0, boundary = 1"""
        m0 = m_func(0,1)    
        assert_almost_equal(m0, self.PTIB[0], 5)

    def test_numpy(self):
        """test a numpy array as input to i; i = range(7), boundary = 0"""
        x = np.array(range(7))
        y0 = m_func(x,0)
        assert np.allclose(y0,self.PTPB)
    
    ### a generator example
    def test_cases(self):
        """loop through and test m_func cases with numpy.allclose"""
        for fixture,result in self.cases:
            m = m_func(*fixture)            
            check = np.allclose(m, result)
            msg = """\
failed m_func.test_cases, case:
%s
m:
%s
expected:
%s""" % (fixture, m, result)
            yield ok_, check, msg
    
            
### end Method 3        

def test_m_func_bad_boundary():
    """m_func fail test, self contained, i = 2, boundary = 1.5"""
    assert_raises(ValueError, m_func , 2, 1.5)
    



class test_make_gam(object):
    """A suite of tests for the make_gam function        
    """
    
    def __init__(self):        
        self.PTPB = m_func(np.arange(2),0)
        self.PTIB = m_func(np.arange(2),1)
        self.gam_isotropic = np.array([[0.5, 0], [0, 0.5]])
        
        self.cases = [            
            [{'m': self.PTIB, 'zt': [0], 'zb': [1], 'mvt': [1], 'mvb': [1]},
             self.gam_isotropic],
            [{'m': self.PTPB, 'zt': [0], 'zb': [1], 'mvt': [1], 'mvb': [1]},
             self.gam_isotropic],
             
            [{'m': self.PTIB, 'zt': [0], 'zb': [1], 'mvt': [2], 'mvb': [2]},
             self.gam_isotropic * 2],
            [{'m': self.PTPB, 'zt': [0], 'zb': [1], 'mvt': [2], 'mvb': [2]},
             self.gam_isotropic * 2],
             
            [{'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [1, 1], 'mvb': [1, 1]},
             self.gam_isotropic],
            [{'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [1, 1], 'mvb': [1, 1]},
             self.gam_isotropic], 
            
            [{'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [2, 2], 'mvb': [2, 2]},
             self.gam_isotropic * 2],
            [{'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [2, 2], 'mvb': [2, 2]},
             self.gam_isotropic * 2],
             
            [{'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [1, 0.5], 'mvb': [1, 0.5]},
             np.array([[0.274317, 0.052295], [0.052295, 0.3655915]])],
            [{'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [1, 0.5], 'mvb': [1, 0.5]},
             np.array([[0.326612767905, 0.0912741609272], [0.0912741609272, 0.368920668216]])],

            #from speccon vba : debug.Print ("[[" & gammat(1,1) & ", " & gammat(1,2) & "],[" & gammat(2,1) & ", " & gammat(2,2) & "]]")
            [{'m': self.PTIB, 'zt': [0], 'zb': [1], 'mvt': [1], 'mvb': [2]},
             np.array([[0.851321183642337, -0.101321183642336],[-0.101321183642336, 0.761257909293592]])],
            [{'m': self.PTPB, 'zt': [0], 'zb': [1], 'mvt': [1], 'mvb': [2]},
             np.array([[0.750000000000001, -9.00632743487448E-02],[-9.00632743487448E-02, 0.750000000000001]])]
            ]
        
                             
    
        
    ### a generator example
    def test_cases(self):   
        """loop through and test make_gam cases with numpy.allclose"""
        for fixture, result in self.cases:            
            gam = make_gam(**fixture)
            check = np.allclose(gam, result)
            msg = """\
failed test_make_gam, case:
%s
gam:
%s
expected:
%s""" % (fixture, gam, result)
            yield ok_, check, msg