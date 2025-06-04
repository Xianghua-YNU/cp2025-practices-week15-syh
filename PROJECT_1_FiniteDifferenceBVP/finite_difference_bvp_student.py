import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.linalg import solve

# ============================================================================
# 方法1：有限差分法 (Finite Difference Method)
# ============================================================================

def solve_bvp_finite_difference(n):
    """
    使用有限差分法求解二阶常微分方程边值问题。
    
    方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
    边界条件：y(0) = 0, y(5) = 3
    
    Args:
        n (int): 内部网格点数量
    
    Returns:
        tuple: (x_grid, y_solution)
            x_grid (np.ndarray): 包含边界点的完整网格
            y_solution (np.ndarray): 对应的解值
    """
    # Step 1: 创建网格
    h = 5.0 / (n + 1)
    x_grid = np.linspace(0, 5, n + 2)
    
    # Step 2: 构建系数矩阵 A 和右端向量 b
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Step 3: 填充矩阵 A 和向量 b
    for i in range(n):
        x_i = x_grid[i + 1]  # 内部点的 x 坐标
        
        # 系数计算
        # y''_i ≈ (y_{i+1} - 2*y_i + y_{i-1}) / h^2
        # y'_i ≈ (y_{i+1} - y_{i-1}) / (2*h)
        # 方程: y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
        # 重新整理: (1/h^2 - sin(x_i)/(2*h)) * y_{i-1} + (-2/h^2 + exp(x_i)) * y_i + (1/h^2 + sin(x_i)/(2*h)) * y_{i+1} = x_i^2
        
        coeff_left = 1.0 / h**2 - np.sin(x_i) / (2.0 * h)
        coeff_center = -2.0 / h**2 + np.exp(x_i)
        coeff_right = 1.0 / h**2 + np.sin(x_i) / (2.0 * h)
        
        # 填充矩阵 A
        if i > 0:
            A[i, i-1] = coeff_left
        A[i, i] = coeff_center
        if i < n - 1:
            A[i, i+1] = coeff_right
        
        # 填充右端向量 b
        b[i] = x_i**2
        
        # 处理边界条件
        if i == 0:  # 第一个内部点，需要考虑左边界 y_0 = 0
            b[i] -= coeff_left * 0.0
        if i == n - 1:  # 最后一个内部点，需要考虑右边界 y_{n+1} = 3
            b[i] -= coeff_right * 3.0
    
    # Step 4: 求解线性系统
    y_interior = solve(A, b)
    
    # Step 5: 组合完整解
    y_solution = np.zeros(n + 2)
    y_solution[0] = 0.0  # 左边界
    y_solution[1:-1] = y_interior  # 内部点
    y_solution[-1] = 3.0  # 右边界
    
    return x_grid, y_solution



# ============================================================================
# 方法2：scipy.integrate.solve_bvp 方法
# ============================================================================

def ode_system_for_solve_bvp(x, y):
    """
    为 scipy.integrate.solve_bvp 定义ODE系统。
    
    将二阶ODE转换为一阶系统：
    y[0] = y(x)
    y[1] = y'(x)
    
    系统方程：
    dy[0]/dx = y[1]
    dy[1]/dx = -sin(x) * y[1] - exp(x) * y[0] + x^2
    
    Args:
        x (float or array): 自变量
        y (array): 状态变量 [y, y']
    
    Returns:
        array: 导数 [dy/dx, dy'/dx]
    """
    y0 = y[0]  # y(x)
    y1 = y[1]  # y'(x)
    
    dy0_dx = y1
    dy1_dx = -np.sin(x) * y1 - np.exp(x) * y0 + x**2
    
    return np.vstack([dy0_dx, dy1_dx])

def boundary_conditions_for_solve_bvp(ya, yb):
    """
    为 scipy.integrate.solve_bvp 定义边界条件。
    
    Args:
        ya (array): 左边界处的状态 [y(0), y'(0)]
        yb (array): 右边界处的状态 [y(5), y'(5)]
    
    Returns:
        array: 边界条件残差 [y(0) - 0, y(5) - 3]
    """
    return np.array([ya[0] - 0, yb[0] - 3])

def solve_bvp_scipy(n_initial_points=11):
    """
    使用 scipy.integrate.solve_bvp 求解BVP。
    
    Args:
        n_initial_points (int): 初始网格点数
    
    Returns:
        tuple: (x_solution, y_solution)
            x_solution (np.ndarray): 解的 x 坐标数组
            y_solution (np.ndarray): 解的 y 坐标数组
    """
    # Step 1: 创建初始网格
    x_initial = np.linspace(0, 5, n_initial_points)
    
    # Step 2: 创建初始猜测
    y_initial = np.zeros((2, n_initial_points))
    y_initial[0] = np.linspace(0, 3, n_initial_points)  # y(x) 的初始猜测
    y_initial[1] = np.ones(n_initial_points) * 0.6      # y'(x) 的初始猜测
    
    # Step 3: 调用 solve_bvp
    solution = solve_bvp(ode_system_for_solve_bvp, boundary_conditions_for_solve_bvp, 
                         x_initial, y_initial)
    
    # Step 4: 提取解
    if solution.success:
        x_solution = solution.x
        y_solution = solution.y[0]  # 只取 y(x)，不要 y'(x)
        return x_solution, y_solution
    else:
        raise RuntimeError("solve_bvp failed to converge")

# ============================================================================
# 主程序：演示三种方法求解边值问题
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("二阶常微分方程边值问题求解演示")
    print("方程: y''(x) + sin(x)*y'(x) + exp(x)*y(x) = x^2")
    print("边界条件: y(0) = 0, y(5) = 3")
    print("=" * 80)
    
    # 定义问题参数
    x_start, y_start = 0.0, 0.0  # 左边界条件
    x_end, y_end = 5.0, 3.0      # 右边界条件
    num_points = 100             # 离散点数
    
    print(f"\n求解区间: [{x_start}, {x_end}]")
    print(f"边界条件: y({x_start}) = {y_start}, y({x_end}) = {y_end}")
    print(f"离散点数: {num_points}")
    
    # ========================================================================
    # 方法1：有限差分法
    # ========================================================================
    print("\n" + "-" * 60)
    print("方法1：有限差分法 (Finite Difference Method)")
    print("-" * 60)
    
    try:
        x_fd, y_fd = solve_bvp_finite_difference(num_points - 2)  # 减去边界点
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
        x_scipy, y_scipy = solve_bvp_scipy(num_points)
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
    plt.scatter([x_start, x_end], [y_start, y_end], 
               color='red', s=100, zorder=5, label='Boundary Conditions')
    
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
    
    print("\n" + "=" * 80)
    print("求解完成！")
    print("=" * 80)
