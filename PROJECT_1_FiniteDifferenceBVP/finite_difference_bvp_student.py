"""
Module: Advanced BVP Solver
File: advanced_bvp_solver.py
Description: 二阶常微分方程边值问题的高级求解器

本模块实现了两种数值方法求解边值问题：
1. 有限差分法 (Finite Difference Method)
2. 基于scipy的打靶法 (Shooting Method with scipy)

求解的边值问题：
y''(x) + sin(x)*y'(x) + exp(x)*y(x) = x^2
边界条件：y(0) = 0, y(5) = 3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import minimize_scalar
from typing import Tuple

# 定义问题常量
X_START, Y_START = 0.0, 0.0  # 左边界条件
X_END, Y_END = 5.0, 3.0      # 右边界条件


class BVPSolver:
    """边值问题求解器基类"""
    
    def __init__(self, ode_func, bc_func, x_range=(X_START, X_END), y_bc=(Y_START, Y_END)):
        """
        初始化边值问题求解器
        
        Args:
            ode_func: ODE系统函数
            bc_func: 边界条件函数
            x_range: 求解区间
            y_bc: 边界条件值
        """
        self.ode_func = ode_func
        self.bc_func = bc_func
        self.x_start, self.x_end = x_range
        self.y_start, self.y_end = y_bc
        
    def solve(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """求解边值问题，返回(x, y)数组"""
        raise NotImplementedError("子类必须实现solve方法")


class FiniteDifferenceSolver(BVPSolver):
    """有限差分法求解器"""
    
    def solve(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用有限差分法求解BVP
        
        Args:
            n_points: 内部网格点数量
        
        Returns:
            (x_grid, y_solution): 网格点和对应的解
        """
        # 创建网格
        h = (self.x_end - self.x_start) / (n_points + 1)
        x_grid = np.linspace(self.x_start, self.x_end, n_points + 2)
        
        # 构建线性方程组
        A = np.zeros((n_points, n_points))
        b = np.zeros(n_points)
        
        # 填充系数矩阵和右端向量
        for i in range(n_points):
            x_i = x_grid[i + 1]  # 内部点
            
            # 中心差分近似
            a = 1/h**2 - np.sin(x_i)/(2*h)
            b_center = -2/h**2 + np.exp(x_i)
            c = 1/h**2 + np.sin(x_i)/(2*h)
            
            # 填充矩阵
            if i > 0:
                A[i, i-1] = a
            A[i, i] = b_center
            if i < n_points - 1:
                A[i, i+1] = c
            
            # 填充右端向量
            b[i] = x_i**2
            
            # 处理边界条件
            if i == 0:
                b[i] -= a * self.y_start
            if i == n_points - 1:
                b[i] -= c * self.y_end
        
        # 求解线性方程组
        y_inner = np.linalg.solve(A, b)
        
        # 组合完整解
        y_solution = np.zeros(n_points + 2)
        y_solution[0] = self.y_start
        y_solution[1:-1] = y_inner
        y_solution[-1] = self.y_end
        
        return x_grid, y_solution


class ScipyBVPSolver(BVPSolver):
    """基于scipy的BVP求解器"""
    
    def solve(self, n_initial_points: int = 11, use_shooting: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用scipy的solve_bvp或打靶法求解BVP
        
        Args:
            n_initial_points: 初始网格点数
            use_shooting: 是否使用打靶法
            
        Returns:
            (x_solution, y_solution): 解的网格点和对应值
        """
        if use_shooting:
            return self._solve_with_shooting(n_initial_points)
        else:
            return self._solve_with_solve_bvp(n_initial_points)
    
    def _solve_with_solve_bvp(self, n_initial_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """使用scipy.integrate.solve_bvp求解"""
        # 创建初始网格
        x_initial = np.linspace(self.x_start, self.x_end, n_initial_points)
        
        # 基于物理特性的智能初始猜测
        y_initial = np.zeros((2, n_initial_points))
        
        # 设计分段初始猜测
        for i, x in enumerate(x_initial):
            # 左侧区域：使用抛物线形状
            if x < (self.x_start + self.x_end) / 3:
                y_initial[0, i] = self.y_start + (self.y_end - self.y_start) * (x / self.x_end)**2
                y_initial[1, i] = 2 * (self.y_end - self.y_start) * x / (self.x_end**2)
            # 右侧区域：使用线性插值
            else:
                slope = (self.y_end - self.y_start) / (self.x_end - self.x_start)
                y_initial[0, i] = self.y_start + slope * (x - self.x_start)
                y_initial[1, i] = slope
        
        # 求解BVP
        solution = solve_bvp(
            self.ode_func, 
            self.bc_func, 
            x_initial, 
            y_initial,
            tol=1e-8,
            max_nodes=10000
        )
        
        if not solution.success:
            raise RuntimeError(f"solve_bvp failed: {solution.message}")
        
        # 在密集网格上获取解
        x_solution = np.linspace(self.x_start, self.x_end, 500)
        y_solution = solution.sol(x_solution)[0]
        
        return x_solution, y_solution
    
    def _solve_with_shooting(self, n_initial_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """使用打靶法求解BVP"""
        def objective_function(slope: float) -> float:
            """目标函数：寻找合适的初始斜率使y(x_end)接近y_end"""
            sol = solve_ivp(
                self.ode_func,
                [self.x_start, self.x_end],
                [self.y_start, slope],
                dense_output=True,
                rtol=1e-8,
                atol=1e-8
            )
            return (sol.sol(self.x_end)[0] - self.y_end)**2
        
        # 优化初始斜率
        result = minimize_scalar(
            objective_function,
            bounds=(-10, 10),
            method='bounded',
            options={'xatol': 1e-8}
        )
        
        if not result.success:
            raise RuntimeError(f"Shooting method failed: {result.message}")
        
        # 使用最优斜率求解IVP
        optimal_slope = result.x
        sol = solve_ivp(
            self.ode_func,
            [self.x_start, self.x_end],
            [self.y_start, optimal_slope],
            dense_output=True,
            rtol=1e-8,
            atol=1e-8
        )
        
        # 在密集网格上获取解
        x_solution = np.linspace(self.x_start, self.x_end, 500)
        y_solution = sol.sol(x_solution)[0]
        
        return x_solution, y_solution


# 定义ODE系统
def ode_system(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    定义ODE系统：
    y[0] = y(x)
    y[1] = y'(x)
    
    返回：[dy/dx, dy'/dx]
    """
    return np.vstack([
        y[1],
        -np.sin(x) * y[1] - np.exp(x) * y[0] + x**2
    ])


# 定义边界条件
def boundary_conditions(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
    """
    定义边界条件：
    ya = [y(x_start), y'(x_start)]
    yb = [y(x_end), y'(x_end)]
    
    返回：[y(x_start) - y_start, y(x_end) - y_end]
    """
    return np.array([
        ya[0] - Y_START,
        yb[0] - Y_END
    ])


# 主函数
def main():
    """主函数：演示不同方法求解BVP"""
    print("=" * 80)
    print("二阶常微分方程边值问题求解演示")
    print("方程: y''(x) + sin(x)*y'(x) + exp(x)*y(x) = x^2")
    print(f"边界条件: y({X_START}) = {Y_START}, y({X_END}) = {Y_END}")
    print("=" * 80)
    
    # 设置求解参数
    num_points = 100
    print(f"\n离散点数: {num_points}")
    
    # 创建求解器
    fd_solver = FiniteDifferenceSolver(ode_system, boundary_conditions)
    scipy_solver = ScipyBVPSolver(ode_system, boundary_conditions)
    
    # 使用有限差分法求解
    print("\n" + "-" * 60)
    print("方法1：有限差分法")
    print("-" * 60)
    
    try:
        x_fd, y_fd = fd_solver.solve(num_points - 2)
        print("有限差分法求解成功！")
    except Exception as e:
        print(f"有限差分法求解失败: {e}")
        x_fd, y_fd = None, None
    
    # 使用scipy求解
    print("\n" + "-" * 60)
    print("方法2：scipy.integrate.solve_bvp")
    print("-" * 60)
    
    try:
        x_scipy, y_scipy = scipy_solver.solve(num_points)
        print("solve_bvp求解成功！")
    except Exception as e:
        print(f"solve_bvp求解失败: {e}")
        print("尝试使用打靶法...")
        try:
            x_scipy, y_scipy = scipy_solver.solve(num_points, use_shooting=True)
            print("打靶法求解成功！")
        except Exception as e2:
            print(f"打靶法求解失败: {e2}")
            x_scipy, y_scipy = None, None
    
    # 结果可视化
    print("\n" + "-" * 60)
    print("结果可视化")
    print("-" * 60)
    
    plt.figure(figsize=(12, 8))
    
    if x_fd is not None:
        plt.plot(x_fd, y_fd, 'b-', linewidth=2, label='Finite Difference', alpha=0.8)
    
    if x_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', linewidth=2, label='scipy solve_bvp', alpha=0.8)
    
    plt.scatter([X_START, X_END], [Y_START, Y_END], color='red', s=100, zorder=5, label='Boundary Conditions')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y(x)', fontsize=12)
    plt.title(r"BVP Solution: $y'' + \sin(x)y' + e^x y = x^2$, $y(0)=0$, $y(5)=3$", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()
    
    # 数值比较
    print("\n" + "-" * 60)
    print("数值结果比较")
    print("-" * 60)
    
    test_points = [1.0, 2.5, 4.0]
    
    for x_test in test_points:
        print(f"\n在 x = {x_test} 处的解值:")
        
        if x_fd is not None:
            y_fd_test = np.interp(x_test, x_fd, y_fd)
            print(f"  有限差分法:  {y_fd_test:.6f}")
        
        if x_scipy is not None:
            y_scipy_test = np.interp(x_test, x_scipy, y_scipy)
            print(f"  solve_bvp:   {y_scipy_test:.6f}")
    
    print("\n" + "=" * 80)
    print("求解完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
