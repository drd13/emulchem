#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `emulchem` package."""


import unittest


from emulchem import ChemistryEmulator
from emulchem import RadexEmulator

class TestEmulchem(unittest.TestCase):
    """Tests for `emulchem` package."""

    def setUp(self):
        self.CO= ChemistryEmulator("CO")
        self.CS= ChemistryEmulator("CS")

    #def tearDown(self):
    #    self.CO.dispose()
    #    self.CS.dispose()

    def test_chem_emulator1(self):
        val = self.CO.get_prediction(radfield=50,zeta=50,temperature=10,density=10**4,av=10,metallicity=1)
        true_val = 8.081735e-05
        self.assertTrue(true_val-10e-10<val<true_val+10e-10)



    def test_chem_emulator2(self):
        val = self.CS.get_prediction(radfield=50,zeta=50,temperature=10,density=10**4,av=10,metallicity=1)
        true_val = 6.260397e-08
        self.assertTrue(true_val-10e-10<val<true_val+10e-10)

    def test__chem_bounds1(self):
         self.assertRaises(Exception,self.CO.get_prediction,radfield=0.1,zeta=50,temperature=10,density=10**4,av=10,metallicity=1)     
         #self.assertRaises(Exception,self.CO.get_prediction(radfield=20,zeta=50,temperature=5000,density=10**4,av=10,metallicity=1))        
         #self.assertRaises(Exception,self.CO.get_prediction(radfield=10,zeta=50,temperature=10,density=10**4,av=10,metallicity=-5))
        

if __name__=="__main__":
    unittest.main()
