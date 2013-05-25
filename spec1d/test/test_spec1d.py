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

from spec1d.spec1d import m_func, make_gam, make_psi

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
            
            ['mv const, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'mvt': [1], 'mvb': [1]}, 
             self.gam_isotropic],
            ['mv const, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'mvt': [1], 'mvb': [1]},
             self.gam_isotropic],
            
            ['mv const *2, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'mvt': [2], 'mvb': [2]}, 
             self.gam_isotropic * 2],
            ['mv const *2, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'mvt': [2], 'mvb': [2]},
             self.gam_isotropic * 2],
             
            ['mv const, 2 layers, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [1, 1], 'mvb': [1, 1]}, 
             self.gam_isotropic],
            ['mv const, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [1, 1], 'mvb': [1, 1]},
             self.gam_isotropic], 
            
            ['mv const*2, 2 layers, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [2, 2], 'mvb': [2, 2]}, 
             self.gam_isotropic * 2],
            ['mv const*2, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [2, 2], 'mvb': [2, 2]},
             self.gam_isotropic * 2],
            #mv 2 layers, const within each layer 
            ['mv 2 layers, const within each layer, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [1, 0.5], 'mvb': [1, 0.5]}, 
             np.array([[0.274317, 0.052295], [0.052295, 0.3655915]])],
            ['mv 2 layers, const within each layer, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'mvt': [1, 0.5], 'mvb': [1, 0.5]},
             np.array([[0.326612767905, 0.0912741609272], [0.0912741609272, 0.368920668216]])],
            
            
            #from speccon vba : debug.Print ("[[" & gammat(1,1) & ", " & gammat(1,2) & "],[" & gammat(2,1) & ", " & gammat(2,2) & "]]")
            ['mv linear within one layer, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'mvt': [1], 'mvb': [2]}, 
             np.array([[0.851321183642337, -0.101321183642336],[-0.101321183642336, 0.761257909293592]])],
            ['mv linear within one layer, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'mvt': [1], 'mvb': [2]},
             np.array([[0.750000000000001, -9.00632743487448E-02],[-9.00632743487448E-02, 0.750000000000001]])]
            ]
        
                             
    
        
    ### a generator example
    def test_cases(self):   
        """loop through and test make_gam cases with numpy.allclose"""
        for desc, fixture, result in self.cases:            
            gam = make_gam(**fixture)
            check = np.allclose(gam, result)
            msg = """\
failed test_make_gam, case: %s
%s
gam:
%s
expected:
%s""" % (desc, fixture, gam, result)
            yield ok_, check, msg
            
            
            
class test_make_psi(object):
    """A suite of tests for the make_psi function        
    """
    
    def __init__(self):        
        self.PTPB = m_func(np.arange(2),0)
        self.PTIB = m_func(np.arange(2),1)
        
        self.psi_iso_v_only_PTIB = 0.5 * np.array([[(np.pi/2.0)**2.0, 0], [0, (3.0*np.pi/2.0)**2.0]])
        self.psi_iso_v_only_PTPB = 0.5 * np.array([[(np.pi)**2.0, 0], [0, (2.0*np.pi)**2.0]])
        self.psi_kh_iso = np.array([[0.5, 0], [0, 0.5]])
        
        self.cases = [     
            #kv
            ['kv const, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [1], 'kvb': [1],
              'kht': [1], 'khb':[1], 'ett': [0], 'etb': [0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0},
             self.psi_iso_v_only_PTIB], 
            ['kv const, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [1], 'kvb': [1],
              'kht': [1], 'khb':[1], 'ett': [0], 'etb': [0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0},
             self.psi_iso_v_only_PTPB],
            
            ['kv const*2, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [2], 'kvb': [2],
              'kht': [1], 'khb':[1], 'ett': [0], 'etb': [0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0}, 
             self.psi_iso_v_only_PTIB*2], 
            ['kv const*2, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [2], 'kvb': [2],
              'kht': [1], 'khb':[1], 'ett': [0], 'etb': [0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0},
             self.psi_iso_v_only_PTPB*2],
            
            ['kv const, dTv*2, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [1], 'kvb': [1],
              'kht': [1], 'khb':[1], 'ett': [0], 'etb': [0], 'dT': 1.0, 'dTv': 2.0, 'dTh': 1.0}, 
             self.psi_iso_v_only_PTIB * 2], 
            ['kv const, dTv*2, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [1], 'kvb': [1],
              'kht': [1], 'khb':[1], 'ett': [0], 'etb': [0], 'dT': 1.0, 'dTv': 2.0, 'dTh': 1.0},
             self.psi_iso_v_only_PTPB * 2],
            
            ['kv const, dT*0.5, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [1], 'kvb': [1],
              'kht': [1], 'khb':[1], 'ett': [0], 'etb': [0], 'dT': 0.5, 'dTv': 1.0, 'dTh': 1.0}, 
             self.psi_iso_v_only_PTIB * 2], 
            ['kv const, dT*0.5, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [1], 'kvb': [1],
              'kht': [1], 'khb':[1], 'ett': [0], 'etb': [0], 'dT': 0.5, 'dTv': 1.0, 'dTh': 1.0},
             self.psi_iso_v_only_PTPB * 2], 
            
            ['kv const, 2 layers, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [1, 1], 'kvb': [1, 1],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [0, 0], 'etb': [0, 0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0},
             self.psi_iso_v_only_PTIB], 
            ['kv const, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [1, 1], 'kvb': [1, 1],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [0, 0], 'etb': [0, 0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0},
             self.psi_iso_v_only_PTPB],
            
            ['kv const*2, 2 layers, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [2, 2], 'kvb': [2, 2],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [0, 0], 'etb': [0, 0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0},
             self.psi_iso_v_only_PTIB * 2], 
            ['kv const*2, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [2, 2], 'kvb': [2, 2],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [0, 0], 'etb': [0, 0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0},
             self.psi_iso_v_only_PTPB * 2], 
            
            ['2 layers, kv const within eachlayer, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [1, 2], 'kvb': [1, 2],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [0, 0], 'etb': [0, 0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0},
             np.array([[1.60044185963,  -1.466671155],[-1.466671155, 18.4577561084]])], 
            ['2 layers, kv const within eachlayer, PTPB',
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [1, 2], 'kvb': [1, 2],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [0, 0], 'etb': [0, 0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0},
             np.array([[ 7.43403806325, -2.37230488791], [-2.37230488791,  33.0766501659]])], 
             
            # from speccon debug.Print ("[[" & psimat(1,1) & ", " & psimat(1,2) & "],[" & psimat(2,1) & ", " & psimat(2,2) & "]]") 
            ['kv linear within one layer, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [1], 'kvb': [2],
              'kht': [1], 'khb':[1], 'ett': [0], 'etb': [0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0},
             np.array([[1.60055082520425, -0.749999999999957],[-0.749999999999957, 16.4049574268382]])], 
            ['kv linear within one layer, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [1], 'kvb': [2],
              'kht': [1], 'khb':[1], 'ett': [0], 'etb': [0], 'dT': 1.0, 'dTv': 1.0, 'dTh': 1.0},
             np.array([[7.40220330081701, -2.22222222222222],[-2.22222222222222, 29.6088132032681]])], 

            #kh
            ['kh const, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso], 
            ['kh const, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso], 
            
            ['kh const*2, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [2], 'khb':[2], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso*2], 
            ['kh const*2, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [2], 'khb':[2], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso*2],
             
            ['kh const,dTh*2 PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 2.0},
             self.psi_kh_iso*2], 
            ['kh const, dTh*2 PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 2.0},
             self.psi_kh_iso*2],
             
            ['kh const,dT*0.5 PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 0.5, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso*2], 
            ['kh const, dT*0.5 PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 0.5, 'dTv': 0, 'dTh': 1},
             self.psi_kh_iso*2],
            
            ['kh const, 2 layers PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [0, 0], 'kvb': [0, 0],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [1, 1], 'etb': [1, 1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso], 
            ['kh const, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [0, 0], 'kvb': [0, 0],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [1, 1], 'etb': [1, 1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso],
             
            ['kh 2 layers, const within each layer, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [0, 0], 'kvb': [0, 0],
              'kht': [1, 2], 'khb':[1, 2], 'ett': [1, 1], 'etb': [1, 1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             np.array([[ 0.951365345728, -0.104590881539], [-0.104590881539,  0.768817023874]])], 
            ['kh 2 layers, const within each layer, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [0, 0], 'kvb': [0, 0],
              'kht': [1, 2], 'khb':[1, 2], 'ett': [1, 1], 'etb': [1, 1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             np.array([[ 0.846774464189, -0.182548321854], [-0.182548321854, 0.762158663568]])],             
            
            ['kh linear within 1 layer, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[2], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             np.array([[0.851321183642337, -0.101321183642336],[-0.101321183642336, 0.761257909293592]])], 
            ['kh linear within 1 layer, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[2], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             np.array([[0.750000000000001, -9.00632743487448E-02],[-9.00632743487448E-02, 0.750000000000001]])], 
            
            #et, originally Walker Phd kh and et were lumped together.  
            #Therefore once separated varying one parameter while keeping 
            #the other constant, then changing which parameter alters should
            #yiueld the same result. Hence reusing the kh results.
            ['et const, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso], 
            ['et const, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso], 
            
            ['et const*2, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [2], 'khb':[2], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso*2], 
            ['et const*2, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [2], 'etb': [2], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso*2],
             
            ['et const,dTh*2 PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 2.0},
             self.psi_kh_iso*2], 
            ['et const, dTh*2 PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 1.0, 'dTv': 0, 'dTh': 2.0},
             self.psi_kh_iso*2],
             
            ['et const,dT*0.5 PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 0.5, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso*2], 
            ['et const, dT*0.5 PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [1], 'dT': 0.5, 'dTv': 0, 'dTh': 1},
             self.psi_kh_iso*2],
            
            ['et const, 2 layers PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [0, 0], 'kvb': [0, 0],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [1, 1], 'etb': [1, 1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso], 
            ['et const, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [0, 0], 'kvb': [0, 0],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [1, 1], 'etb': [1, 1], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso],
             
            ['et 2 layers, const within each layer, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [0, 0], 'kvb': [0, 0],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [1, 2], 'etb': [1, 2], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             np.array([[ 0.951365345728, -0.104590881539], [-0.104590881539,  0.768817023874]])], 
            ['et 2 layers, const within each layer, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'kvt': [0, 0], 'kvb': [0, 0],
              'kht': [1, 1], 'khb':[1, 1], 'ett': [1, 2], 'etb': [1, 2], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             np.array([[ 0.846774464189, -0.182548321854], [-0.182548321854, 0.762158663568]])], 
             
            ['et linear within 1 layer, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [2], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             np.array([[0.851321183642337, -0.101321183642336],[-0.101321183642336, 0.761257909293592]])], 
            ['et linear within 1 layer, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [1], 'khb':[1], 'ett': [1], 'etb': [2], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             np.array([[0.750000000000001, -9.00632743487448E-02],[-9.00632743487448E-02, 0.750000000000001]])], 
            
            
            
            #mixed
            ['kh const, et const cancel out, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [0.5], 'khb':[0.5], 'ett': [2], 'etb': [2], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso], 
            ['kh const, et const cancel out, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'kvt': [0], 'kvb': [0],
              'kht': [0.5], 'khb':[0.5], 'ett': [2], 'etb': [2], 'dT': 1.0, 'dTv': 0, 'dTh': 1.0},
             self.psi_kh_iso]
             
             #would be nice to have a test for both et and kh varying 
             #linearly at the same time. I think I'll just have to trust the
             #sympy integrations.  At least if the above tests pass I'll have 
             #some confidence.

            ]
        
                             
    
        
    ### a generator example
    def test_cases(self):   
        """loop through and test make_psi cases with numpy.allclose"""
        for desc, fixture, result in self.cases:            
            psi = make_psi(**fixture)
            check = np.allclose(psi, result)
            msg = """\
failed test_make_psi, case: %s
%s
psi:
%s
expected:
%s""" % (desc, fixture, psi, result)
            yield ok_, check, msg            