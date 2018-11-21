#!/usr/bin/env python3

import unittest
import numpy as np
import functions as F
import newton

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0*x + 6.0

        # Setting maxiter to 2 b/c we're guessing the actual root
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(-2.0)
        # Equality should be exact if we supply *the* root, ergo
        # assertEqual rather than assertAlmostEqual
        self.assertEqual(x, -2.0)

    def testQuadratic(self):
        # f(x) = x^2 - 2x - 3
        f = F.Polynomial([-3,-2,1])
        solver = newton.Newton(f,tol=1.e-15,maxiter=200)
        x = solver.solve(3.3)
        self.assertAlmostEqual(x,3.0)
       

    def testQuadratic2(self):
        # y : x*x + 6*x + 9
        f = F.Polynomial([9,6,1])
        _Df = F.Polynomial([6,2])
        solver = newton.Newton(f, tol=1.e-15, maxiter=200,Df=_Df)
        x = solver.solve(-2.0)
        self.assertAlmostEqual(x, -3.0)
        
    def testExponential(self):
        # f(x) = e^x
        f = lambda x: np.exp(x)
        solver = newton.Newton(f,tol=1.e-15,maxiter=50)
        x = solver.solve(1.0)
        self.assertEqual(x,float('inf'))
#EXPONENTIAL IS NOT WORKING SO WELL

    def testZeroDerivative(self):
        f = F.Polynomial([-1,0,1])
        solver = newton.Newton(f,tol=1.e-15,maxiter=200)
        x = solver.solve(0.0)
        self.assertAlmostEqual(x,1.0)

    def test2d(self):
        f = lambda x: np.matrix([[x[0,0]-x[1,0]+7],[x[0,0]+x[1,0]-15]])
        solver = newton.Newton(f, tol=1.e-6, maxiter=30)
        x0 = np.matrix([[3],[2]])
        x = solver.solve(x0)
        np.testing.assert_array_almost_equal(x, np.matrix([[4.], [11.]]))

if __name__ == "__main__":
    unittest.main()

    
