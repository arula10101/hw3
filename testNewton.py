#!/usr/bin/env python3

import unittest
import numpy as np
import functions as F
import newton

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0*x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(-2.0)
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
       
    def testZeroDerivative(self):
        f = F.Polynomial([-1,0,1])
        solver = newton.Newton(f,tol=1.e-15,maxiter=200)
        x = solver.solve(0.0)
        self.assertAlmostEqual(x,1.0)

    def test2D(self):
        f = lambda x: np.matrix([[x[0,0]-x[1,0]+7],[x[0,0]+x[1,0]-15]])
        solver = newton.Newton(f, tol=1.e-6, maxiter=30)
        x0 = np.matrix([[3],[2]])
        x = solver.solve(x0)
        np.testing.assert_array_almost_equal(x, np.matrix([[4.], [11.]]))

    def testRadius(self):
        f = lambda x: x/(x**2 + 1.)
        solver = newton.Newton(f, tol=1.e-15, maxiter=30, max_radius=1)
        x = solver.solve(0.4)
        self.assertAlmostEqual(x, 0.)
        
    def testStep(self):
        f = lambda x : np.cos(x)
        solver = newton.Newton(f,tol=1.e-15, maxiter=1)
        x0 = 3
        x1 = solver.step(x0)
        x2 = solver.step(x0, f(x0))
        self.assertEqual(x1, x2)

    def testTangent(self):
        f = lambda x: x*(1-x)
        solver = newton.Newton(f,tol=1.e-15,maxiter=200)
        x = solver.solve(0.5)
        self.assertAlmostEqual(x,1.0)

    def testNoRoot(self):
        print("Expect this test to return maxiter error")
        f = lambda x: x**2 + 1.0
        solver = newton.Newton(f,tol=1.e-15,maxiter=200)
        x = solver.solve(2.0)
        print("Expect this test to return maxiter error")

    def testBoundRoot(self):
        print("Initial guess has to be within (-1/sqrt(2) , 1/sqrt(2))")
        f = lambda x: x*np.exp(-x**2)
        solver = newton.Newton(f,tol=1.e-15,maxiter=200,max_radius=1./np.sqrt(2.))
        x = solver.solve(0.05)
        self.assertAlmostEqual(x,0.0)

    def test2Danalytic(self):
        f = F.Polynomial([-9,0,1])
        _Df = lambda x: 2*x
        solver = newton.Newton(f,tol=1.e-15,maxiter=200,Df=_Df)
        x = solver.solve(-4.0)
        self.assertAlmostEqual(x,-3.0)

if __name__ == "__main__":
    unittest.main()

    
