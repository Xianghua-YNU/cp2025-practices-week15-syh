#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 参考答案

本项目实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

作者：教学团队
创建日期：2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


def ode_system_shooting(y, t=None):
    """
    Define the ODE system for shooting method.
    
    The second-order ODE: u'' + π²/4 * (u+1) = 0
    Convert to first-order system:
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π²/4 * (y1+1)
    
    Args:
        y (array or float): State vector [y1, y2] where y1=u, y2=u' OR time t
        t (float or array, optional): Independent variable (time/position) OR state vector
    
    Returns:
        list: Derivatives [y1', y2']
    
    Note: This function can handle both (y, t) and (t, y) parameter orders
    for compatibility with different solvers and test cases.
    """
    # Handle both (y, t) and (t, y) parameter orders
    if isinstance(y, (int, float)) and hasattr(t, '__len__'):
        # Called as (t, y) - swap parameters
        t, y = y, t
    elif t is None:
        # Called with single argument, assume it's y and t is not needed
        pass
    
    return [y[1], -np.pi*(y[0]+1)/4]


def boundary_conditions_scipy(ya, yb):
    """
    Define boundary conditions for scipy.solve_bvp.
    
    Boundary conditions: u(0) = 1, u(1) = 1
    ya[0] should equal 1, yb[0] should equal 1
    
    Args:
        ya (array): Values at left boundary [u(0), u'(0)]
        yb (array): Values at right boundary [u(1), u'(1)]
    
    Returns:
        array: Boundary condition residuals
    """
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    """
    Define the ODE system for scipy.solve_bvp.
    
    Note: scipy.solve_bvp uses (x, y) parameter order, different from odeint
    
    Args:
        x (float): Independent variable
        y (array): State vector [y1, y2]
    
    Returns:
        array: Derivatives as column vector
    """
    return np.vstack((y[1], -np.pi*(y[0]+1)/4))


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    """
    Solve boundary value problem using shooting method.
    
    Algorithm:
    1. Guess initial slope m1
    2. Solve IVP with initial conditions [u(0), m1]
    3. Check if u(1) matches boundary condition
    4. If not, adjust slope using secant method and repeat
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of discretization points
        max_iterations (int): Maximum iterations for shooting
        tolerance (float): Convergence tolerance
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # Validate input parameters
    if len(x_span) != 2 or x_span[1] <= x_span[0]:
        raise ValueError("x_span must be a tuple (x_start, x_end) with x_end > x_start")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple (u_left, u_right)")
    if n_points < 10:
        raise ValueError("n_points must be at least 10")
    
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    # Setup domain
    x = np.linspace(x_start, x_end, n_points)
    
    # Initial guess for slope
    m1 = -1.0  # First guess
    y0 = [u_left, m1]  # Initial conditions [u(0), u'(0)]
    
    # Solve with first guess
    sol1 = odeint(ode_system_shooting, y0, x)
    u_end_1 = sol1[-1, 0]  # u(x_end) with first guess
    
    # Check if first guess is good enough
    if abs(u_end_1 - u_right) < tolerance:
        return x, sol1[:, 0]
    
    # Second guess using linear scaling
    m2 = m1 * u_right / u_end_1 if abs(u_end_1) > 1e-12 else m1 + 1.0
    y0[1] = m2
    sol2 = odeint(ode_system_shooting, y0, x)
    u_end_2 = sol2[-1, 0]  # u(x_end) with second guess
    
    # Check if second guess is good enough
    if abs(u_end_2 - u_right) < tolerance:
        return x, sol2[:, 0]
    
    # Iterative improvement using secant method
    for iteration in range(max_iterations):
        # Secant method to find better slope
        if abs(u_end_2 - u_end_1) < 1e-12:
            # Avoid division by zero
            m3 = m2 + 0.1
        else:
            m3 = m2 + (u_right - u_end_2) * (m2 - m1) / (u_end_2 - u_end_1)
        
        # Solve with new guess
        y0[1] = m3
        sol3 = odeint(ode_system_shooting, y0, x)
        u_end_3 = sol3[-1, 0]
        
        # Check convergence
        if abs(u_end_3 - u_right) < tolerance:
            return x, sol3[:, 0]
        
        # Update for next iteration
        m1, m2 = m2, m3
        u_end_1, u_end_2 = u_end_2, u_end_3
    
    # If not converged, return best solution with warning
    print(f"Warning: Shooting method did not converge after {max_iterations} iterations.")
    print(f"Final boundary error: {abs(u_end_3 - u_right):.2e}")
    return x, sol3[:, 0]


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    Solve boundary value problem using scipy.solve_bvp.
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of initial mesh points
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # Validate input parameters
    if len(x_span) != 2 or x_span[1] <= x_span[0]:
        raise ValueError("x_span must be a tuple (x_start, x_end) with x_end > x_start")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple (u_left, u_right)")
    if n_points < 5:
        raise ValueError("n_points must be at least 5")
    
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    # Setup initial mesh
    x_init = np.linspace(x_start, x_end, n_points)
    
    # Initial guess: linear interpolation between boundary values
    y_init = np.zeros((2, x_init.size))
    y_init[0] = u_left + (u_right - u_left) * (x_init - x_start) / (x_end - x_start)
    y_init[1] = (u_right - u_left) / (x_end - x_start)  # Constant slope guess
    
    # Solve using scipy.solve_bvp
    try:
        sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x_init, y_init)
        
        if not sol.success:
            raise RuntimeError(f"scipy.solve_bvp failed: {sol.message}")
        
        # Generate solution on fine mesh
        x_fine = np.linspace(x_start, x_end, 100)
        y_fine = sol.sol(x_fine)[0]
        
        return x_fine, y_fine
        
    except Exception as e:
        raise RuntimeError(f"Error in scipy.solve_bvp: {str(e)}")


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    Compare shooting method and scipy.solve_bvp, generate comparison plot.
    
    Args:
        x_span (tuple): Domain for the problem
        boundary_conditions (tuple): Boundary values (left, right)
        n_points (int): Number of points for plotting
    
    Returns:
        dict: Dictionary containing solutions and analysis
    """
    print("Solving BVP using both methods...")
    
    try:
        # Solve using shooting method
        print("Running shooting method...")
        x_shoot, y_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
        
        # Solve using scipy.solve_bvp
        print("Running scipy.solve_bvp...")
        x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points//2)
        
        # Interpolate scipy solution to shooting method grid for comparison
        y_scipy_interp = np.interp(x_shoot, x_scipy, y_scipy)
        
        # Calculate differences
        max_diff = np.max(np.abs(y_shoot - y_scipy_interp))
        rms_diff = np.sqrt(np.mean((y_shoot - y_scipy_interp)**2))
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Main comparison plot
        plt.subplot(2, 1, 1)
        plt.plot(x_shoot, y_shoot, 'b-', linewidth=2, label='Shooting Method')
        plt.plot(x_scipy, y_scipy, 'r--', linewidth=2, label='scipy.solve_bvp')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Comparison of BVP Solution Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mark boundary points
        plt.plot([x_span[0], x_span[1]], [boundary_conditions[0], boundary_conditions[1]], 
                'ko', markersize=8, label='Boundary Conditions')
        plt.legend()
        
        # Difference plot
        plt.subplot(2, 1, 2)
        plt.plot(x_shoot, y_shoot - y_scipy_interp, 'g-', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('Difference (Shooting - scipy)')
        plt.title(f'Solution Difference (Max: {max_diff:.2e}, RMS: {rms_diff:.2e})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print analysis
        print("\nSolution Analysis:")
        print(f"Maximum difference: {max_diff:.2e}")
        print(f"RMS difference: {rms_diff:.2e}")
        print(f"Shooting method points: {len(x_shoot)}")
        print(f"scipy.solve_bvp points: {len(x_scipy)}")
        
        # Verify boundary conditions
        print(f"\nBoundary condition verification:")
        print(f"Shooting method: u({x_span[0]}) = {y_shoot[0]:.6f}, u({x_span[1]}) = {y_shoot[-1]:.6f}")
        print(f"scipy.solve_bvp: u({x_span[0]}) = {y_scipy[0]:.6f}, u({x_span[1]}) = {y_scipy[-1]:.6f}")
        print(f"Target: u({x_span[0]}) = {boundary_conditions[0]}, u({x_span[1]}) = {boundary_conditions[1]}")
        
        return {
            'x_shooting': x_shoot,
            'y_shooting': y_shoot,
            'x_scipy': x_scipy,
            'y_scipy': y_scipy,
            'max_difference': max_diff,
            'rms_difference': rms_diff,
            'boundary_error_shooting': [abs(y_shoot[0] - boundary_conditions[0]), 
                                      abs(y_shoot[-1] - boundary_conditions[1])],
            'boundary_error_scipy': [abs(y_scipy[0] - boundary_conditions[0]), 
                                   abs(y_scipy[-1] - boundary_conditions[1])]
        }
        
    except Exception as e:
        print(f"Error in method comparison: {str(e)}")
        raise


# Test functions for development and debugging
def test_ode_system():
    """
    Test the ODE system implementation.
    """
    print("Testing ODE system...")
    
    # Test point
    t_test = 0.5
    y_test = np.array([1.0, 0.5])
    
    # Test shooting method ODE system
    dydt = ode_system_shooting(y_test, t_test)
    expected = [0.5, -np.pi*(1.0+1)/4]
    print(f"ODE system (shooting): dydt = {dydt}")
    print(f"Expected: {expected}")
    assert np.allclose(dydt, expected), "Shooting ODE system test failed"
    
    # Test scipy ODE system
    dydt_scipy = ode_system_scipy(t_test, y_test)
    expected_scipy = np.array([[0.5], [-np.pi*2/4]])
    print(f"ODE system (scipy): dydt = {dydt_scipy.flatten()}")
    print(f"Expected: {expected_scipy.flatten()}")
    assert np.allclose(dydt_scipy, expected_scipy), "Scipy ODE system test failed"
    
    print("ODE system tests passed!")


def test_boundary_conditions():
    """
    Test the boundary conditions implementation.
    """
    print("Testing boundary conditions...")
    
    ya = np.array([1.0, 0.5])  # Left boundary
    yb = np.array([1.0, -0.3])  # Right boundary
    
    bc_residual = boundary_conditions_scipy(ya, yb)
    expected = np.array([0.0, 0.0])  # Both boundaries should be satisfied
    print(f"Boundary condition residuals: {bc_residual}")
    print(f"Expected: {expected}")
    assert np.allclose(bc_residual, expected), "Boundary conditions test failed"
    
    print("Boundary conditions test passed!")


def test_shooting_method():
    """
    Test the shooting method implementation.
    """
    print("Testing shooting method...")
    
    x_span = (0, 1)
    boundary_conditions = (1, 1)
    
    x, y = solve_bvp_shooting_method(x_span, boundary_conditions, n_points=50)
    
    # Check boundary conditions
    assert abs(y[0] - boundary_conditions[0]) < 1e-6, "Left boundary condition not satisfied"
    assert abs(y[-1] - boundary_conditions[1]) < 1e-6, "Right boundary condition not satisfied"
    
    print(f"Shooting method: u(0) = {y[0]:.6f}, u(1) = {y[-1]:.6f}")
    print("Shooting method test passed!")


def test_scipy_method():
    """
    Test the scipy.solve_bvp wrapper.
    """
    print("Testing scipy.solve_bvp wrapper...")
    
    x_span = (0, 1)
    boundary_conditions = (1, 1)
    
    x, y = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=20)
    
    # Check boundary conditions
    assert abs(y[0] - boundary_conditions[0]) < 1e-6, "Left boundary condition not satisfied"
    assert abs(y[-1] - boundary_conditions[1]) < 1e-6, "Right boundary condition not satisfied"
    
    print(f"scipy.solve_bvp: u(0) = {y[0]:.6f}, u(1) = {y[-1]:.6f}")
    print("scipy.solve_bvp wrapper test passed!")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题 - 参考答案")
    print("=" * 60)
    
    # Run all tests
    print("Running unit tests...")
    test_ode_system()
    test_boundary_conditions()
    test_shooting_method()
    test_scipy_method()
    print("All unit tests passed!\n")
    
    # Run method comparison
    print("Running method comparison...")
    results = compare_methods_and_plot()
    
    print("\n项目2完成！所有功能已实现并测试通过。")
