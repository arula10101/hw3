#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt

import functions as F

class TestFunctions(unittest.TestCase):
    def test_ApproxJacobian1D(self):
        for slope in range(-10,10):
            for x0 in range(-10,10):
                f = lambda x: slope*x + 5.0
                dx = 1.e-3
                Df_x = F.approximateJacobian(f, x0, dx)
                # If x and f are scalar-valued, Df_x should be, too
                self.assertTrue(np.isscalar(Df_x))
                self.assertAlmostEqual(Df_x, slope)

    def test_ApproxJacobian2D(self):
        # numpy matrices can also be initialized with strings. The
        # semicolon separates rows; spaces (or commas) delimit entries
        # within a row.
        A = np.matrix("1.0 2.0; 3.0 4.0")

        def f(x):
            # The * operator for numpy matrices is overloaded to mean
            # matrix-multiplication, rather than elementwise
            # multiplication as it does for numpy arrays
            return A * x

        # The vector-valued function f defined above is the following:
        # if we let u = f(x), then
        #
        # u1 = x1 + 2 x2
        # u2 = 3 x1 + 4 x2
        #
        # The Jacobian of this function is constant and exactly equal
        # to the matrix A. approximateJacobian should thus return
        # something pretty close to A.

        x0 = np.matrix("5.0; 6.0")
        dx = 1.e-6
        Df_x = F.approximateJacobian(f, x0, dx)

        # Make sure approximateJA
        self.assertEqual(Df_x.shape, (2,2))
        # numpy arrays and matrices vectorize comparisons. So if a & b
        # are arrays, the expression a==b will itself be an array of
        # booleans. But an array of booleans does not itself evaluate
        # to a clean boolean (this is an exception to the general
        # Python rule that "every object can be interpreted as a
        # boolean"), so normal assert statements will break. We need
        # array-specific assert statements found in numpy.testing
        npt.assert_array_almost_equal(Df_x, A)


    def testApproxJacobianND(self):
        for n in range(10):
            A = np.random.random((n,n))
            x0 = np.random.random((n,1))
            f = lambda x: np.dot(A,x)
            dx = 1.e-6
            Df_x = F.approximateJacobian(f,x0,dx)
            self.assertEqual(Df_x.shape,(n,n))
            np.testing.assert_array_almost_equal(Df_x,A)


    def test_Polynomial(self):
        # p(x) = x^2 + 5x + 4
        p = F.Polynomial([4, 5, 1])
        # linspace(a, b, N) produces N equally spaced values from a to
        # b, including both a & b.
        for x in np.linspace(-2,2,11):
            self.assertAlmostEqual(p(x), 4 + 5*x + x**2)
        
        # It would have been more Pythonic to feed the whole array x into p
        # (the way the Polynomial class is written, it's agnostic
        # about whether x is a scalar or an array) and compare with an
        # npt assertion, as follows:
        #
        # p = F.Polynomial([4, 5, 1])
        # npt.assert_array_almost_equal(p(x), 4 + 5*x + x**2)
        #
        # But I though this would be more legible.

    def testTrigonometricFunctions(self):
        function = F.TrigonometricFunctions(a=2,b=2,c=4)
        test_x  = [ 0., 5., -2., 200, 5000]
        test_fx = [ 6., 2.6494758, 1.3491114, 3.2277807, 2.3334039]
        for test in range(len(test_x)):
            np.testing.assert_array_almost_equal(function.f(test_x[test]),test_fx[test],decimal=4)

        

if __name__ == '__main__':
    unittest.main()



