#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 测试文件

测试所有实现的函数，包括：
- ODE系统函数
- 边界条件函数
- 打靶法求解器
- scipy.solve_bvp封装函数
- 方法比较和可视化

总分：100分
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configuration
TEST_TOLERANCE = 1e-6
PLOT_TOLERANCE = 1e-4

# Import reference solution
try:
    from solution.shooting_method_solution import (
        ode_system_shooting as ref_ode_system_shooting,
        boundary_conditions_scipy as ref_boundary_conditions_scipy,
        ode_system_scipy as ref_ode_system_scipy,
        solve_bvp_shooting_method as ref_solve_bvp_shooting_method,
        solve_bvp_scipy_wrapper as ref_solve_bvp_scipy_wrapper,
        compare_methods_and_plot as ref_compare_methods_and_plot
    )
    REFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import reference solution: {e}")
    REFERENCE_AVAILABLE = False

# Import student solution
try:
    from shooting_method_student import (
        ode_system_shooting as stu_ode_system_shooting,
        boundary_conditions_scipy as stu_boundary_conditions_scipy,
        ode_system_scipy as stu_ode_system_scipy,
        solve_bvp_shooting_method as stu_solve_bvp_shooting_method,
        solve_bvp_scipy_wrapper as stu_solve_bvp_scipy_wrapper,
        compare_methods_and_plot as stu_compare_methods_and_plot
    )
    STUDENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import student solution: {e}")
    STUDENT_AVAILABLE = False


class TestReferenceSolution(unittest.TestCase):
    """Test reference solution to ensure correctness."""
    
    def setUp(self):
        """Set up test parameters."""
        self.x_span = (0, 1)
        self.boundary_conditions = (1, 1)
        self.test_point = 0.5
        self.test_state = np.array([1.0, 0.5])
    
    @unittest.skipUnless(REFERENCE_AVAILABLE, "Reference solution not available")
    def test_reference_ode_system_shooting_5pts(self):
        """Test reference ODE system for shooting method."""
        dydt = ref_ode_system_shooting(self.test_point, self.test_state)
        expected = [0.5, -np.pi*(1.0+1)/4]
        
        self.assertIsInstance(dydt, (list, np.ndarray), "ODE system should return list or array")
        self.assertEqual(len(dydt), 2, "ODE system should return 2 derivatives")
        np.testing.assert_allclose(dydt, expected, rtol=TEST_TOLERANCE, 
                                 err_msg="ODE system derivatives incorrect")
    
    @unittest.skipUnless(REFERENCE_AVAILABLE, "Reference solution not available")
    def test_reference_ode_system_scipy_5pts(self):
        """Test reference ODE system for scipy.solve_bvp."""
        dydt = ref_ode_system_scipy(self.test_point, self.test_state)
        expected = np.array([[0.5], [-np.pi*2/4]])
        
        self.assertIsInstance(dydt, np.ndarray, "Scipy ODE system should return numpy array")
        self.assertEqual(dydt.shape, (2, 1), "Scipy ODE system should return column vector")
        np.testing.assert_allclose(dydt, expected, rtol=TEST_TOLERANCE,
                                 err_msg="Scipy ODE system derivatives incorrect")
    
    @unittest.skipUnless(REFERENCE_AVAILABLE, "Reference solution not available")
    def test_reference_boundary_conditions_5pts(self):
        """Test reference boundary conditions function."""
        ya = np.array([1.0, 0.5])
        yb = np.array([1.0, -0.3])
        
        bc_residual = ref_boundary_conditions_scipy(ya, yb)
        expected = np.array([0.0, 0.0])
        
        self.assertIsInstance(bc_residual, np.ndarray, "Boundary conditions should return numpy array")
        self.assertEqual(len(bc_residual), 2, "Should return 2 boundary condition residuals")
        np.testing.assert_allclose(bc_residual, expected, rtol=TEST_TOLERANCE,
                                 err_msg="Boundary condition residuals incorrect")
    
    @unittest.skipUnless(REFERENCE_AVAILABLE, "Reference solution not available")
    def test_reference_shooting_method_15pts(self):
        """Test reference shooting method implementation."""
        x, y = ref_solve_bvp_shooting_method(self.x_span, self.boundary_conditions, n_points=50)
        
        # Check return types and shapes
        self.assertIsInstance(x, np.ndarray, "x should be numpy array")
        self.assertIsInstance(y, np.ndarray, "y should be numpy array")
        self.assertEqual(len(x), len(y), "x and y should have same length")
        self.assertEqual(len(x), 50, "Should return requested number of points")
        
        # Check boundary conditions
        self.assertAlmostEqual(y[0], self.boundary_conditions[0], places=5,
                              msg="Left boundary condition not satisfied")
        self.assertAlmostEqual(y[-1], self.boundary_conditions[1], places=5,
                              msg="Right boundary condition not satisfied")
        
        # Check domain
        self.assertAlmostEqual(x[0], self.x_span[0], places=10, msg="Domain start incorrect")
        self.assertAlmostEqual(x[-1], self.x_span[1], places=10, msg="Domain end incorrect")
    
    @unittest.skipUnless(REFERENCE_AVAILABLE, "Reference solution not available")
    def test_reference_scipy_wrapper_10pts(self):
        """Test reference scipy.solve_bvp wrapper."""
        x, y = ref_solve_bvp_scipy_wrapper(self.x_span, self.boundary_conditions, n_points=25)
        
        # Check return types and shapes
        self.assertIsInstance(x, np.ndarray, "x should be numpy array")
        self.assertIsInstance(y, np.ndarray, "y should be numpy array")
        self.assertEqual(len(x), len(y), "x and y should have same length")
        
        # Check boundary conditions
        self.assertAlmostEqual(y[0], self.boundary_conditions[0], places=5,
                              msg="Left boundary condition not satisfied")
        self.assertAlmostEqual(y[-1], self.boundary_conditions[1], places=5,
                              msg="Right boundary condition not satisfied")
        
        # Check domain
        self.assertAlmostEqual(x[0], self.x_span[0], places=10, msg="Domain start incorrect")
        self.assertAlmostEqual(x[-1], self.x_span[1], places=10, msg="Domain end incorrect")


class TestStudentImplementation(unittest.TestCase):
    """Test student implementation against reference solution."""
    
    def setUp(self):
        """Set up test parameters."""
        self.x_span = (0, 1)
        self.boundary_conditions = (1, 1)
        self.test_point = 0.5
        self.test_state = np.array([1.0, 0.5])
    
    @unittest.skipUnless(STUDENT_AVAILABLE, "Student solution not available")
    def test_student_ode_system_shooting_15pts(self):
        """Test student ODE system for shooting method."""
        try:
            dydt = stu_ode_system_shooting(self.test_point, self.test_state)
            expected = [0.5, -np.pi*(1.0+1)/4]
            
            self.assertIsInstance(dydt, (list, np.ndarray), "ODE system should return list or array")
            self.assertEqual(len(dydt), 2, "ODE system should return 2 derivatives")
            np.testing.assert_allclose(dydt, expected, rtol=TEST_TOLERANCE,
                                     err_msg="ODE system derivatives incorrect")
            
            # Test with different values
            test_state2 = np.array([0.5, -0.2])
            dydt2 = stu_ode_system_shooting(0.3, test_state2)
            expected2 = [-0.2, -np.pi*(0.5+1)/4]
            np.testing.assert_allclose(dydt2, expected2, rtol=TEST_TOLERANCE,
                                     err_msg="ODE system fails with different inputs")
            
        except NotImplementedError:
            self.fail("ode_system_shooting function not implemented")
        except Exception as e:
            self.fail(f"ode_system_shooting function failed: {str(e)}")
    
    @unittest.skipUnless(STUDENT_AVAILABLE, "Student solution not available")
    def test_student_ode_system_scipy_10pts(self):
        """Test student ODE system for scipy.solve_bvp."""
        try:
            dydt = stu_ode_system_scipy(self.test_point, self.test_state)
            expected = np.array([[0.5], [-np.pi*2/4]])
            
            self.assertIsInstance(dydt, np.ndarray, "Scipy ODE system should return numpy array")
            self.assertEqual(dydt.shape, (2, 1), "Scipy ODE system should return column vector")
            np.testing.assert_allclose(dydt, expected, rtol=TEST_TOLERANCE,
                                     err_msg="Scipy ODE system derivatives incorrect")
            
        except NotImplementedError:
            self.fail("ode_system_scipy function not implemented")
        except Exception as e:
            self.fail(f"ode_system_scipy function failed: {str(e)}")
    
    @unittest.skipUnless(STUDENT_AVAILABLE, "Student solution not available")
    def test_student_boundary_conditions_10pts(self):
        """Test student boundary conditions function."""
        try:
            ya = np.array([1.0, 0.5])
            yb = np.array([1.0, -0.3])
            
            bc_residual = stu_boundary_conditions_scipy(ya, yb)
            expected = np.array([0.0, 0.0])
            
            self.assertIsInstance(bc_residual, np.ndarray, "Boundary conditions should return numpy array")
            self.assertEqual(len(bc_residual), 2, "Should return 2 boundary condition residuals")
            np.testing.assert_allclose(bc_residual, expected, rtol=TEST_TOLERANCE,
                                     err_msg="Boundary condition residuals incorrect")
            
            # Test with different boundary values
            ya2 = np.array([0.5, 0.1])
            yb2 = np.array([2.0, -0.5])
            bc_residual2 = stu_boundary_conditions_scipy(ya2, yb2)
            expected2 = np.array([-0.5, 1.0])
            np.testing.assert_allclose(bc_residual2, expected2, rtol=TEST_TOLERANCE,
                                     err_msg="Boundary conditions fail with different inputs")
            
        except NotImplementedError:
            self.fail("boundary_conditions_scipy function not implemented")
        except Exception as e:
            self.fail(f"boundary_conditions_scipy function failed: {str(e)}")
    
    @unittest.skipUnless(STUDENT_AVAILABLE, "Student solution not available")
    def test_student_shooting_method_40pts(self):
        """Test student shooting method implementation."""
        try:
            x, y = stu_solve_bvp_shooting_method(self.x_span, self.boundary_conditions, n_points=50)
            
            # Check return types and shapes
            self.assertIsInstance(x, np.ndarray, "x should be numpy array")
            self.assertIsInstance(y, np.ndarray, "y should be numpy array")
            self.assertEqual(len(x), len(y), "x and y should have same length")
            self.assertEqual(len(x), 50, "Should return requested number of points")
            
            # Check boundary conditions (more lenient for shooting method)
            self.assertAlmostEqual(y[0], self.boundary_conditions[0], places=4,
                                  msg="Left boundary condition not satisfied")
            self.assertAlmostEqual(y[-1], self.boundary_conditions[1], places=4,
                                  msg="Right boundary condition not satisfied")
            
            # Check domain
            self.assertAlmostEqual(x[0], self.x_span[0], places=10, msg="Domain start incorrect")
            self.assertAlmostEqual(x[-1], self.x_span[1], places=10, msg="Domain end incorrect")
            
            # Check monotonicity of x
            self.assertTrue(np.all(np.diff(x) > 0), "x array should be strictly increasing")
            
            # Test with different parameters
            x2, y2 = stu_solve_bvp_shooting_method((0, 2), (0, 2), n_points=30)
            self.assertEqual(len(x2), 30, "Should handle different n_points")
            self.assertAlmostEqual(y2[0], 0, places=4, msg="Different boundary conditions not handled")
            self.assertAlmostEqual(y2[-1], 2, places=4, msg="Different boundary conditions not handled")
            
        except NotImplementedError:
            self.fail("solve_bvp_shooting_method function not implemented")
        except Exception as e:
            self.fail(f"solve_bvp_shooting_method function failed: {str(e)}")
    
    @unittest.skipUnless(STUDENT_AVAILABLE, "Student solution not available")
    def test_student_scipy_wrapper_25pts(self):
        """Test student scipy.solve_bvp wrapper."""
        try:
            x, y = stu_solve_bvp_scipy_wrapper(self.x_span, self.boundary_conditions, n_points=25)
            
            # Check return types and shapes
            self.assertIsInstance(x, np.ndarray, "x should be numpy array")
            self.assertIsInstance(y, np.ndarray, "y should be numpy array")
            self.assertEqual(len(x), len(y), "x and y should have same length")
            
            # Check boundary conditions
            self.assertAlmostEqual(y[0], self.boundary_conditions[0], places=5,
                                  msg="Left boundary condition not satisfied")
            self.assertAlmostEqual(y[-1], self.boundary_conditions[1], places=5,
                                  msg="Right boundary condition not satisfied")
            
            # Check domain
            self.assertAlmostEqual(x[0], self.x_span[0], places=10, msg="Domain start incorrect")
            self.assertAlmostEqual(x[-1], self.x_span[1], places=10, msg="Domain end incorrect")
            
            # Check monotonicity of x
            self.assertTrue(np.all(np.diff(x) > 0), "x array should be strictly increasing")
            
            # Compare with reference if available
            if REFERENCE_AVAILABLE:
                x_ref, y_ref = ref_solve_bvp_scipy_wrapper(self.x_span, self.boundary_conditions, n_points=25)
                # Interpolate to compare
                y_ref_interp = np.interp(x, x_ref, y_ref)
                max_diff = np.max(np.abs(y - y_ref_interp))
                self.assertLess(max_diff, 0.1, "Student solution differs significantly from reference")
            
        except NotImplementedError:
            self.fail("solve_bvp_scipy_wrapper function not implemented")
        except Exception as e:
            self.fail(f"solve_bvp_scipy_wrapper function failed: {str(e)}")
    
    @unittest.skipUnless(STUDENT_AVAILABLE, "Student solution not available")
    def test_student_error_handling_5pts(self):
        """Test student functions handle errors appropriately."""
        # Test invalid inputs
        with self.assertRaises((ValueError, TypeError), msg="Should handle invalid x_span"):
            stu_solve_bvp_shooting_method((1, 0), (1, 1))  # Invalid span
        
        with self.assertRaises((ValueError, TypeError), msg="Should handle invalid boundary conditions"):
            stu_solve_bvp_shooting_method((0, 1), (1,))  # Invalid boundary conditions
        
        with self.assertRaises((ValueError, TypeError), msg="Should handle invalid n_points"):
            stu_solve_bvp_shooting_method((0, 1), (1, 1), n_points=2)  # Too few points


class TestMethodComparison(unittest.TestCase):
    """Test method comparison and visualization."""
    
    @unittest.skipUnless(STUDENT_AVAILABLE, "Student solution not available")
    def test_student_comparison_function_10pts(self):
        """Test student method comparison function."""
        try:
            # Capture matplotlib output to avoid display during testing
            plt.ioff()  # Turn off interactive mode
            
            # Redirect stdout to capture print statements
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                results = stu_compare_methods_and_plot()
                
                # Check return type
                self.assertIsInstance(results, dict, "Should return dictionary of results")
                
                # Check required keys
                required_keys = ['x_shooting', 'y_shooting', 'x_scipy', 'y_scipy', 
                               'max_difference', 'rms_difference']
                for key in required_keys:
                    self.assertIn(key, results, f"Missing key: {key}")
                
                # Check data types
                self.assertIsInstance(results['x_shooting'], np.ndarray, "x_shooting should be array")
                self.assertIsInstance(results['y_shooting'], np.ndarray, "y_shooting should be array")
                self.assertIsInstance(results['x_scipy'], np.ndarray, "x_scipy should be array")
                self.assertIsInstance(results['y_scipy'], np.ndarray, "y_scipy should be array")
                
                # Check numerical values
                self.assertIsInstance(results['max_difference'], (int, float), "max_difference should be numeric")
                self.assertIsInstance(results['rms_difference'], (int, float), "rms_difference should be numeric")
                self.assertGreaterEqual(results['max_difference'], 0, "max_difference should be non-negative")
                self.assertGreaterEqual(results['rms_difference'], 0, "rms_difference should be non-negative")
                
            finally:
                sys.stdout = old_stdout
                plt.close('all')  # Close any plots
                plt.ion()  # Turn interactive mode back on
            
        except NotImplementedError:
            self.fail("compare_methods_and_plot function not implemented")
        except Exception as e:
            self.fail(f"compare_methods_and_plot function failed: {str(e)}")


def run_tests():
    """Run all tests and return results summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestReferenceSolution))
    suite.addTests(loader.loadTestsFromTestCase(TestStudentImplementation))
    suite.addTests(loader.loadTestsFromTestCase(TestMethodComparison))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Calculate scores
    total_tests = result.testsRun
    failed_tests = len(result.failures) + len(result.errors)
    passed_tests = total_tests - failed_tests
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"- {test}: {error_msg}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"- {test}: {error_msg}")
    
    # Simple scoring (for GitHub Classroom)
    if STUDENT_AVAILABLE and not REFERENCE_AVAILABLE:
        print(f"\nNote: Reference solution not available. Only basic functionality tested.")
    elif not STUDENT_AVAILABLE:
        print(f"\nNote: Student solution not available. Please implement the required functions.")
    
    return result


if __name__ == "__main__":
    print("项目2：打靸法与scipy.solve_bvp求解边值问题 - 测试")
    print("=" * 60)
    
    # Check availability
    print(f"Reference solution available: {REFERENCE_AVAILABLE}")
    print(f"Student solution available: {STUDENT_AVAILABLE}")
    print()
    
    # Run tests
    test_result = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if test_result.wasSuccessful() else 1)
