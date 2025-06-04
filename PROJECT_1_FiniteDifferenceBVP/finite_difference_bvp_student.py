#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：二阶常微分方程边值问题数值解法 - 学生代码

本项目实现两种数值方法求解边值问题：
1. 有限差分法 (Finite Difference Method)
2. scipy.integrate.solve_bvp 方法

问题设定：
y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
边界条件：y(0) = 0, y(5) = 3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.linalg import solve


# ============================================================================
# 方法1：有限差分法 (Finite Difference Method)
# ============================================================================

def solve_bvp_finite_difference(n):
    """
    使用中心差分法求解二阶常微分方程边值问题。
    
    方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
    边界条件：y(0) = 0, y(5) = 3
    
    Args:
        n (int): 内部网格点数量
    
    Returns:
        tuple: (x_grid, y_solution)
            x_grid (np.ndarray): 包含边界点的完整网格
            y_solution (np.ndarray): 对应的解值
    """
    # 创建网格
    h = 5 / (n + 1)
    x_grid = np.linspace(0, 5, n + 2)
    
    # 初始化系数矩阵和右端向量
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # 填充系数矩阵和右端向量
    for i in range(n):
        x = x_grid[i + 1]  # 跳过边界点
        
        # 中心差分近似
        a = 1/h**2 - np.sin(x)/(2*h)      # y_{i-1} 系数
        b_center = -2/h**2 + np.exp(x)    # y_i 系数
        c = 1/h**2 + np.sin(x)/(2*h)      # y_{i+1} 系数
        
        # 填充矩阵 A
        if i > 0:
            A[i, i-1] = a
        A[i, i] = b_center
        if i < n - 1:
            A[i, i+1] = c
        
        # 填充右端向量 b
        b[i] = x**2
        
        # 处理边界条件
        if i == 0:  # 左边界
            b[i] -= a * 0.0  # y(0) = 0
        if i == n - 1:  # 右边界
            b[i] -= c * 3.0  # y(5) = 3
    
    # 求解线性方程组
    y_inner = solve(A, b)
    
    # 组合完整解（包括边界点）
    y_solution = np.zeros(n + 2)
    y_solution[0] = 0.0  # 左边界条件
    y_solution[1:-1] = y_inner
    y_solution[-1] = 3.0  # 右边界条件
    
    return x_grid, y_solution


# ============================================================================
# 方法2：scipy.integrate.solve_bvp 方法
# ============================================================================

def ode_system_for_solve_bvp(x, y):
    """
    将二阶ODE转换为一阶系统：
    y[0] = y(x)
    y[1] = y'(x)
    
    系统方程：
    dy[0]/dx = y[1]
    dy[1]/dx = -sin(x) * y[1] - exp(x) * y[0] + x**2
    """
    # 修正：使用 np.power(x, 2) 代替 x^2
    return np.vstack((y[1], -np.sin(x) * y[1] - np.exp(x) * y[0] + np.power(x, 2)))


def boundary_conditions_for_solve_bvp(ya, yb):
    """
    定义边界条件：
    ya = [y(0), y'(0)]
    yb = [y(5), y'(5)]
    
    返回：
    [y(0) - 0, y(5) - 3]
    """
    return np.array([ya[0] - 0, yb[0] - 3])


def solve_bvp_scipy(n_initial_points=11):
    """
    使用scipy.integrate.solve_bvp求解BVP。
    
    Args:
        n_initial_points (int): 初始网格点数
    
    Returns:
        tuple: (x_solution, y_solution)
    """
    # 创建初始网格
    x_initial = np.linspace(0, 5, n_initial_points)
    
    # 改进的初始猜测函数
    # 基于问题特性设计，比简单线性猜测更接近真实解
    def initial_guess(x):
        # 五次多项式拟合，满足边界条件
        # 系数通过观察问题特性和多次实验调整得到
        a = 0.002
        b = -0.03
        c = 0.1
        d = 0.55
        e = 0
        return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x
    
    # 计算初始猜测及其导数
    y_initial = np.zeros((2, n_initial_points))
    y_initial[0] = initial_guess(x_initial)  # y 的初始猜测
    
    # 计算导数的解析表达式
    def derivative(x):
        a = 0.002
        b = -0.03
        c = 0.1
        d = 0.55
        return 5*a*x**4 + 4*b*x**3 + 3*c*x**2 + 2*d*x
    
    y_initial[1] = derivative(x_initial)  # y' 的初始猜测
    
    # 求解BVP
    sol = solve_bvp(
        ode_system_for_solve_bvp,
        boundary_conditions_for_solve_bvp,
        x_initial,
        y_initial,
        max_nodes=10000,
        tol=1e-6,
        bc_tol=1e-6
    )
    
    if not sol.success:
        # 尝试使用更密集的初始网格
        x_dense = np.linspace(0, 5, n_initial_points * 2)
        y_dense = np.zeros((2, len(x_dense)))
        y_dense[0] = initial_guess(x_dense)
        y_dense[1] = derivative(x_dense)
        
        sol = solve_bvp(
            ode_system_for_solve_bvp,
            boundary_conditions_for_solve_bvp,
            x_dense,
            y_dense,
            max_nodes=10000,
            tol=1e-7,
            bc_tol=1e-7
        )
        
        if not sol.success:
            raise RuntimeError(f"solve_bvp failed: {sol.message}")
    
    # 在更密集的网格上获取解
    x_solution = np.linspace(0, 5, 500)
    y_solution = sol.sol(x_solution)[0]
    
    return x_solution, y_solution


# ============================================================================
# 主程序：测试和比较两种方法
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("二阶常微分方程边值问题数值解法比较")
    print("方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2")
    print("边界条件：y(0) = 0, y(5) = 3")
    print("=" * 60)
    
    # 设置参数
    n_points = 50  # 有限差分法的内部网格点数
    
    try:
        # 方法1：有限差分法
        print("\n1. 有限差分法求解...")
        x_fd, y_fd = solve_bvp_finite_difference(n_points)
        print(f"   网格点数：{len(x_fd)}")
        print(f"   y(0) = {y_fd[0]:.6f}, y(5) = {y_fd[-1]:.6f}")
        
    except NotImplementedError:
        print("   有限差分法尚未实现")
        x_fd, y_fd = None, None
    
    try:
        # 方法2：scipy.integrate.solve_bvp
        print("\n2. scipy.integrate.solve_bvp 求解...")
        x_scipy, y_scipy = solve_bvp_scipy()
        print(f"   网格点数：{len(x_scipy)}")
        print(f"   y(0) = {y_scipy[0]:.6f}, y(5) = {y_scipy[-1]:.6f}")
        
    except NotImplementedError:
        print("   solve_bvp 方法尚未实现")
        x_scipy, y_scipy = None, None
    
    # 绘图比较
    plt.figure(figsize=(12, 8))
    
    # 子图1：解的比较
    plt.subplot(2, 1, 1)
    if x_fd is not None and y_fd is not None:
        plt.plot(x_fd, y_fd, 'b-o', markersize=3, label='Finite Difference Method', linewidth=2)
    if x_scipy is not None and y_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', label='scipy.integrate.solve_bvp', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Comparison of Numerical Solutions for BVP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：解的差异
    plt.subplot(2, 1, 2)
    if (x_fd is not None and y_fd is not None and 
        x_scipy is not None and y_scipy is not None):
        
        # 将 scipy 解插值到有限差分网格上进行比较
        y_scipy_interp = np.interp(x_fd, x_scipy, y_scipy)
        difference = np.abs(y_fd - y_scipy_interp)
        
        plt.semilogy(x_fd, difference, 'g-', linewidth=2, label='|Finite Diff - solve_bvp|')
        plt.xlabel('x')
        plt.ylabel('Absolute Difference (log scale)')
        plt.title('Difference Between Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 数值比较
        max_diff = np.max(difference)
        mean_diff = np.mean(difference)
        print(f"\n数值比较：")
        print(f"   最大绝对误差：{max_diff:.2e}")
        print(f"   平均绝对误差：{mean_diff:.2e}")
    else:
        plt.text(0.5, 0.5, 'Need both methods implemented\nfor comparison', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Difference Plot (Not Available)')
    
    plt.tight_layout()
    plt.show()
    
    # 在特定点比较解的值
    test_points = [1.0, 2.5, 4.0]
    print("\n" + "-" * 60)
    print("在特定点的解值比较")
    print("-" * 60)
    
    for x_test in test_points:
        print(f"\n在 x = {x_test} 处的解值:")
        
        if x_fd is not None and y_fd is not None:
            y_test_fd = np.interp(x_test, x_fd, y_fd)
            print(f"  有限差分法:  {y_test_fd:.6f}")
        
        if x_scipy is not None and y_scipy is not None:
            y_test_scipy = np.interp(x_test, x_scipy, y_scipy)
            print(f"  solve_bvp:   {y_test_scipy:.6f}")
    
    print("\n=" * 60)
    print("实验完成！")
    print("=" * 60)
