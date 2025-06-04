#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生代码模板

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：[YOUR_NAME]
学号：[YOUR_STUDENT_ID]
完成日期：[COMPLETION_DATE]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


def ode_system_shooting(t, y):  # 修正参数顺序为 (t, y) 与参考答案一致
    """
    Define the ODE system for shooting method.
    
    Convert the second-order ODE u'' = -π(u+1)/4 into a first-order system:
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4
    
    Args:
        t (float): Independent variable (time/position)
        y (array): State vector [y1, y2] where y1=u, y2=u'
    
    Returns:
        list: Derivatives [y1', y2']
    """
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


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=50, tolerance=1e-6):
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
    if not isinstance(x_span, tuple) or len(x_span) != 2:
        raise ValueError("x_span must be a tuple of two elements")
    if not isinstance(boundary_conditions, tuple) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple of two elements")
    if x_span[0] >= x_span[1]:
        raise ValueError("x_span must be in the form (start, end) with start < end")
    if n_points < 3:
        raise ValueError("n_points must be at least 3")
    
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    # Generate x points for evaluation
    x_array = np.linspace(x_start, x_end, n_points)
    
    # 改进初始猜测策略，参考参考答案
    m1 = -1.0  # 第一个斜率猜测
    y0 = [u_left, m1]  # 初始条件 [u(0), u'(0)]
    
    # 使用 odeint 求解，注意参数顺序
    sol1 = odeint(ode_system_shooting, y0, x_array, tfirst=True)
    u_end_1 = sol1[-1, 0]  # u(x_end) 第一个猜测的结果
    
    # 检查第一个猜测是否足够好
    if abs(u_end_1 - u_right) < tolerance:
        return x_array, sol1[:, 0]
    
    # 第二个猜测，使用线性缩放策略
    m2 = m1 * u_right / u_end_1 if abs(u_end_1) > 1e-12 else m1 + 1.0
    y0 = [u_left, m2]  # 更新初始条件
    
    sol2 = odeint(ode_system_shooting, y0, x_array, tfirst=True)
    u_end_2 = sol2[-1, 0]  # u(x_end) 第二个猜测的结果
    
    # 检查第二个猜测是否足够好
    if abs(u_end_2 - u_right) < tolerance:
        return x_array, sol2[:, 0]
    
    # 使用割线法迭代改进
    for iteration in range(max_iterations):
        # 割线法计算下一个斜率猜测
        if abs(u_end_2 - u_end_1) < 1e-12:
            # 避免除以零
            m3 = m2 + 0.1
        else:
            m3 = m2 + (u_right - u_end_2) * (m2 - m1) / (u_end_2 - u_end_1)
        
        # 使用新猜测求解
        y0 = [u_left, m3]
        sol3 = odeint(ode_system_shooting, y0, x_array, tfirst=True)
        u_end_3 = sol3[-1, 0]
        
        # 检查收敛性
        if abs(u_end_3 - u_right) < tolerance:
            return x_array, sol3[:, 0]
        
        # 更新迭代变量
        m1, m2 = m2, m3
        u_end_1, u_end_2 = u_end_2, u_end_3
    
    # 如果未收敛，返回最佳解并打印警告
    print(f"警告: 打靶法在{max_iterations}次迭代后未收敛。")
    print(f"最终边界误差: {abs(u_end_3 - u_right):.2e}")
    return x_array, sol3[:, 0]


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
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    # Validate input parameters
    if not isinstance(x_span, tuple) or len(x_span) != 2:
        raise ValueError("x_span must be a tuple of two elements")
    if not isinstance(boundary_conditions, tuple) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple of two elements")
    if x_span[0] >= x_span[1]:
        raise ValueError("x_span must be in the form (start, end) with start < end")
    if n_points < 3:
        raise ValueError("n_points must be at least 3")
    
    # Create the initial mesh
    x = np.linspace(x_start, x_end, n_points)
    
    # 改进初始猜测，参考参考答案
    y_guess = np.zeros((2, x.size))
    y_guess[0] = np.linspace(u_left, u_right, n_points)  # 初始猜测 u(x)
    y_guess[1] = np.zeros(n_points)  # 初始猜测 u'(x)
    
    # Solve the BVP
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x, y_guess, tol=1e-6)
    
    # Check if the solution converged
    if not sol.success:
        raise RuntimeError(f"scipy.solve_bvp failed to converge: {sol.message}")
    
    # Evaluate the solution on a finer mesh for plotting
    x_plot = np.linspace(x_start, x_end, 100)
    y_plot = sol.sol(x_plot)
    
    return x_plot, y_plot[0]


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
    # Solve using both methods
    x_shooting, y_shooting = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
    x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points)
    
    # Calculate differences
    y_scipy_interp = np.interp(x_shooting, x_scipy, y_scipy)
    max_difference = np.max(np.abs(y_shooting - y_scipy_interp))
    rms_difference = np.sqrt(np.mean((y_shooting - y_scipy_interp)**2))
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_shooting, y_shooting, 'b-', label='Shooting Method')
    plt.plot(x_scipy, y_scipy, 'r--', label='scipy.solve_bvp')
    plt.title('Solution Comparison')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_shooting, np.abs(y_shooting - y_scipy_interp), 'g-')
    plt.title('Difference Between Methods')
    plt.xlabel('x')
    plt.ylabel('|Difference|')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('bvp_comparison.png', dpi=300)
    
    # Return analysis results
    return {
        'x_shooting': x_shooting,
        'y_shooting': y_shooting,
        'x_scipy': x_scipy,
        'y_scipy': y_scipy,
        'max_difference': max_difference,
        'rms_difference': rms_difference
    }


# Test functions for development and debugging
def test_ode_system():
    """
    Test the ODE system implementation.
    """
    print("Testing ODE system...")
    try:
        # Test point
        t_test = 0.5
        y_test = np.array([1.0, 0.5])
        
        # Test shooting method ODE system
        dydt = ode_system_shooting(t_test, y_test)
        print(f"ODE system (shooting): dydt = {dydt}")
        
        # Test scipy ODE system
        dydt_scipy = ode_system_scipy(t_test, y_test)
        print(f"ODE system (scipy): dydt = {dydt_scipy}")
        
    except NotImplementedError:
        print("ODE system functions not yet implemented.")


def test_boundary_conditions():
    """
    Test the boundary conditions implementation.
    """
    print("Testing boundary conditions...")
    try:
        ya = np.array([1.0, 0.5])  # Left boundary
        yb = np.array([1.0, -0.3])  # Right boundary
        
        bc_residual = boundary_conditions_scipy(ya, yb)
        print(f"Boundary condition residuals: {bc_residual}")
        
    except NotImplementedError:
        print("Boundary conditions function not yet implemented.")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)
    
    # Run basic tests
    test_ode_system()
    test_boundary_conditions()
    
    # Run comparison
    try:
        print("\nTesting method comparison...")
        results = compare_methods_and_plot()
        print("Method comparison completed successfully!")
        print(f"Maximum Difference: {results['max_difference']}")
        print(f"RMS Difference: {results['rms_difference']}")
    except NotImplementedError as e:
        print(f"Method comparison not yet implemented: {e}")
    except Exception as e:
        print(f"Error in method comparison: {e}")    
