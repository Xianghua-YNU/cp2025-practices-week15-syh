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
    dy[1]/dx = -sin(x) * y[1] - exp(x) * y[0] + x^2
    """
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
    
    # 简化的初始猜测 - 线性插值
    y_initial = np.zeros((2, n_initial_points))
    y_initial[0] = np.linspace(0, 3, n_initial_points)  # y 的初始猜测（线性插值）
    y_initial[1] = np.ones(n_initial_points) * 0.6      # y' 的初始猜测（常数）
    
    # 求解BVP，只保留基本参数
    sol = solve_bvp(
        ode_system_for_solve_bvp,
        boundary_conditions_for_solve_bvp,
        x_initial,
        y_initial,
    )
    
    # 检查求解是否成功
    if not sol.success:
        # 尝试使用更密集的初始网格
        x_dense = np.linspace(0, 5, n_initial_points * 5)
        y_dense = np.zeros((2, len(x_dense)))
        y_dense[0] = np.linspace(0, 3, len(x_dense))
        y_dense[1] = np.ones(len(x_dense)) * 0.6
        
        sol = solve_bvp(
            ode_system_for_solve_bvp,
            boundary_conditions_for_solve_bvp,
            x_dense,
            y_dense,
        )
        
        if not sol.success:
            # 最终尝试：使用分段线性插值作为初始猜测
            key_points = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            key_values = np.array([0.0, 0.5, 1.2, 1.8, 2.5, 3.0])
            y_piecewise = np.interp(x_dense, key_points, key_values)
            
            # 分段导数
            y_piecewise_deriv = np.zeros_like(x_dense)
            for i in range(1, len(key_points)):
                mask = (x_dense >= key_points[i-1]) & (x_dense <= key_points[i])
                slope = (key_values[i] - key_values[i-1]) / (key_points[i] - key_points[i-1])
                y_piecewise_deriv[mask] = slope
            
            sol = solve_bvp(
                ode_system_for_solve_bvp,
                boundary_conditions_for_solve_bvp,
                x_dense,
                np.vstack((y_piecewise, y_piecewise_deriv)),
            )
            
            if not sol.success:
                raise RuntimeError(f"solve_bvp 求解失败: {sol.message}")
    
    # 在500个点的网格上获取解，与测试要求一致
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
    n_points = 100  # 离散点数
    
    print(f"\n求解区间: [0, 5]")
    print(f"边界条件: y(0) = 0, y(5) = 3")
    print(f"离散点数: {n_points}")
    
    # ========================================================================
    # 方法1：有限差分法
    # ========================================================================
    print("\n" + "-" * 60)
    print("方法1：有限差分法 (Finite Difference Method)")
    print("-" * 60)
    
    try:
        x_fd, y_fd = solve_bvp_finite_difference(n_points - 2)  # 减去边界点
        print("有限差分法求解成功！")
    except Exception as e:
        print(f"有限差分法求解失败: {e}")
        x_fd, y_fd = None, None
    
    # ========================================================================
    # 方法2：scipy.integrate.solve_bvp
    # ========================================================================
    print("\n" + "-" * 60)
    print("方法2：scipy.integrate.solve_bvp")
    print("-" * 60)
    
    try:
        x_scipy, y_scipy = solve_bvp_scipy(n_points)
        print("solve_bvp 求解成功！")
    except Exception as e:
        print(f"solve_bvp 求解失败: {e}")
        x_scipy, y_scipy = None, None
    
    # ========================================================================
    # 结果可视化与比较
    # ========================================================================
    print("\n" + "-" * 60)
    print("结果可视化与比较")
    print("-" * 60)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制两种方法的解
    if x_fd is not None and y_fd is not None:
        plt.plot(x_fd, y_fd, 'b-', linewidth=2, label='Finite Difference Method', alpha=0.8)
    
    if x_scipy is not None and y_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', linewidth=2, label='scipy solve_bvp', alpha=0.8)
    
    # 标记边界条件
    plt.scatter([0, 5], [0, 3], color='red', s=100, zorder=5, label='Boundary Conditions')
    
    # 图形美化
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y(x)', fontsize=12)
    plt.title(r"BVP Solution: $y'' + \sin(x)y' + e^x y = x^2$, $y(0)=0$, $y(5)=3$", 
              fontsize=14, pad=20)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    # ========================================================================
    # 数值结果比较
    # ========================================================================
    print("\n" + "-" * 60)
    print("数值结果比较")
    print("-" * 60)
    
    # 在几个特定点比较解的值
    test_points = [1.0, 2.5, 4.0]
    
    for x_test in test_points:
        print(f"\n在 x = {x_test} 处的解值:")
        
        if x_fd is not None and y_fd is not None:
            # 插值得到测试点的值
            y_test_fd = np.interp(x_test, x_fd, y_fd)
            print(f"  有限差分法:  {y_test_fd:.6f}")
        
        if x_scipy is not None and y_scipy is not None:
            y_test_scipy = np.interp(x_test, x_scipy, y_scipy)
            print(f"  solve_bvp:   {y_test_scipy:.6f}")
    
    print("\n" + "=" * 60)
    print("求解完成！")
    print("=" * 60)
