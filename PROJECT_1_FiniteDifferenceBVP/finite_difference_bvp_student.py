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
    
    # 基于问题特性设计的初始猜测函数
    # 使用分段二次函数，在不同区间使用不同参数
    def initial_guess(x):
        y = np.zeros_like(x)
        for i, xi in enumerate(x):
            if xi <= 1.5:
                # 左侧部分 - 二次函数，满足 y(0)=0
                a = 0.2
                y[i] = a * xi**2
            elif xi <= 3.5:
                # 中间部分 - 二次函数，平滑连接左右段
                a = -0.1
                b = 0.7
                c = -0.375
                y[i] = a * xi**2 + b * xi + c
            else:
                # 右侧部分 - 二次函数，满足 y(5)=3
                a = 0.12
                b = -1.1
                c = 3.25
                y[i] = a * xi**2 + b * xi + c
        return y
    
    # 计算初始猜测及其导数
    y_initial = np.zeros((2, n_initial_points))
    y_initial[0] = initial_guess(x_initial)  # y 的初始猜测
    
    # 计算导数的解析表达式
    def derivative(x):
        dydx = np.zeros_like(x)
        for i, xi in enumerate(x):
            if xi <= 1.5:
                a = 0.2
                dydx[i] = 2 * a * xi
            elif xi <= 3.5:
                a = -0.1
                b = 0.7
                dydx[i] = 2 * a * xi + b
            else:
                a = 0.12
                b = -1.1
                dydx[i] = 2 * a * xi + b
        return dydx
    
    y_initial[1] = derivative(x_initial)  # y' 的初始猜测
    
    # 尝试使用不同的求解策略
    strategies = [
        {"method": "initial", "x": x_initial, "y": y_initial},
        {"method": "dense_grid", "x": np.linspace(0, 5, n_initial_points * 5), "y": None},
        {"method": "piecewise_linear", "x": np.linspace(0, 5, n_initial_points * 10), "y": None},
    ]
    
    # 为密集网格生成初始猜测
    for strategy in strategies[1:]:
        if strategy["y"] is None:
            # 插值初始猜测到新网格
            y_dense = np.zeros((2, len(strategy["x"])))
            y_dense[0] = np.interp(strategy["x"], x_initial, y_initial[0])
            
            # 计算导数的初始猜测
            dx = np.diff(strategy["x"])
            dy = np.diff(y_dense[0])
            y_dense[1][1:-1] = (dy[1:] + dy[:-1]) / (dx[1:] + dx[:-1])
            y_dense[1][0] = (y_dense[0][1] - y_dense[0][0]) / dx[0]
            y_dense[1][-1] = (y_dense[0][-1] - y_dense[0][-2]) / dx[-1]
            
            strategy["y"] = y_dense
    
    # 尝试多种求解策略
    last_error = None
    for strategy in strategies:
        try:
            sol = solve_bvp(
                ode_system_for_solve_bvp,
                boundary_conditions_for_solve_bvp,
                strategy["x"],
                strategy["y"],
            )
            
            if sol.success:
                # 在500个点的网格上获取解，与测试要求一致
                x_solution = np.linspace(0, 5, 500)
                y_solution = sol.sol(x_solution)[0]
                
                # 验证边界条件
                if not np.allclose(y_solution[0], 0.0, atol=1e-7):
                    print(f"警告: 左边界条件未完全满足: y(0) = {y_solution[0]:.8f}")
                if not np.allclose(y_solution[-1], 3.0, atol=1e-7):
                    print(f"警告: 右边界条件未完全满足: y(5) = {y_solution[-1]:.8f}")
                
                return x_solution, y_solution
            else:
                last_error = sol.message
                print(f"策略 {strategy['method']} 求解失败: {sol.message}")
                
        except Exception as e:
            last_error = str(e)
            print(f"策略 {strategy['method']} 执行出错: {e}")
    
    # 如果所有策略都失败，抛出异常
    raise RuntimeError(f"所有求解策略都失败: {last_error}")


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
    
    # 求解有限差分法
    print("\n" + "-" * 60)
    print("方法1：有限差分法")
    print("-" * 60)
    try:
        x_fd, y_fd = solve_bvp_finite_difference(n_points - 2)
        print(f"有限差分法求解成功，网格点数: {len(x_fd)}")
        print(f"边界条件验证: y(0)={y_fd[0]:.6f}, y(5)={y_fd[-1]:.6f}")
    except Exception as e:
        print(f"有限差分法求解失败: {e}")
        x_fd, y_fd = None, None
    
    # 求解scipy方法
    print("\n" + "-" * 60)
    print("方法2：scipy.integrate.solve_bvp")
    print("-" * 60)
    try:
        x_scipy, y_scipy = solve_bvp_scipy(n_points)
        print(f"scipy方法求解成功，网格点数: {len(x_scipy)}")
        print(f"边界条件验证: y(0)={y_scipy[0]:.6f}, y(5)={y_scipy[-1]:.6f}")
    except Exception as e:
        print(f"scipy方法求解失败: {e}")
        x_scipy, y_scipy = None, None
    
    # 结果可视化
    plt.figure(figsize=(12, 8))
    
    if x_fd is not None:
        plt.plot(x_fd, y_fd, 'b-', linewidth=2, label='Finite Difference')
    
    if x_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', linewidth=2, label='scipy.solve_bvp')
    
    plt.scatter([0, 5], [0, 3], color='red', s=100, zorder=5, label='Boundary Conditions')
    
    plt.title(r"边值问题数值解比较: $y'' + \sin(x)y' + e^x y = x^2$", fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y(x)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    # 数值结果比较
    print("\n" + "-" * 60)
    print("数值结果比较")
    print("-" * 60)
    
    test_points = [1.0, 2.5, 4.0]
    for x_test in test_points:
        print(f"\n在 x = {x_test} 处的解值:")
        
        if x_fd is not None:
            y_fd_test = np.interp(x_test, x_fd, y_fd)
            print(f"  有限差分法: {y_fd_test:.6f}")
        
        if x_scipy is not None:
            y_scipy_test = np.interp(x_test, x_scipy, y_scipy)
            print(f"  solve_bvp: {y_scipy_test:.6f}")
    
    print("\n" + "=" * 60)
    print("求解完成！")
    print("=" * 60)
