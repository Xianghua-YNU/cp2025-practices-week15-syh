"""
学生模板：平方反比引力场中的运动
文件：inverse_square_law_motion_student.py
作者：[你的名字]
日期：[完成日期]

重要：函数名称、参数名称和返回值的结构必须与参考答案保持一致！
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def derivatives(t, state_vector, gm_val):
    """
    计算状态向量 [x, y, vx, vy] 的导数。

    运动方程（直角坐标系）:
    dx/dt = vx
    dy/dt = vy
    dvx/dt = -GM * x / r^3
    dvy/dt = -GM * y / r^3
    其中 r = sqrt(x^2 + y^2)。

    参数:
        t (float): 当前时间 (solve_ivp 需要，但在此自治系统中不直接使用)。
        state_vector (np.ndarray): 一维数组 [x, y, vx, vy]，表示当前状态。
        gm_val (float): 引力常数 G 与中心天体质量 M 的乘积。

    返回:
        np.ndarray: 一维数组，包含导数 [dx/dt, dy/dt, dvx/dt, dvy/dt]。
    """
    x, y, vx, vy = state_vector
    r_cubed = (x**2 + y**2)**1.5
    
    # 计算加速度分量
    ax = -gm_val * x / r_cubed
    ay = -gm_val * y / r_cubed
    
    return np.array([vx, vy, ax, ay])

def solve_orbit(initial_conditions, t_span, t_eval, gm_val):
    """
    使用 scipy.integrate.solve_ivp 求解轨道运动问题。

    参数:
        initial_conditions (list or np.ndarray): 初始状态 [x0, y0, vx0, vy0]。
        t_span (tuple): 积分时间区间 (t_start, t_end)。
        t_eval (np.ndarray): 需要存储解的时间点数组。
        gm_val (float): GM 值 (引力常数 * 中心天体质量)。

    返回:
        scipy.integrate.OdeSolution: solve_ivp 返回的解对象。
                                     可以通过 sol.y 访问解的数组，sol.t 访问时间点。
    """
    return solve_ivp(
        fun=derivatives,
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval,
        args=(gm_val,),
        method='DOP853',
        rtol=1e-7,
        atol=1e-9
    )

def calculate_energy(state_vector, gm_val, m=1.0):
    """
    计算质点的（比）机械能。
    （比）能量 E/m = 0.5 * v^2 - GM/r

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组。
        gm_val (float): GM 值。
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比能 (E/m)。

    返回:
        np.ndarray or float: （比）机械能。
    """
    if state_vector.ndim == 1:
        x, y, vx, vy = state_vector
        r = np.sqrt(x**2 + y**2)
        v_squared = vx**2 + vy**2
        specific_energy = 0.5 * v_squared - gm_val / r
        return specific_energy * m if m != 1.0 else specific_energy
    else:
        x, y, vx, vy = state_vector.T
        r = np.sqrt(x**2 + y**2)
        v_squared = vx**2 + vy**2
        specific_energy = 0.5 * v_squared - gm_val / r
        return specific_energy * m if m != 1.0 else specific_energy

def calculate_angular_momentum(state_vector, m=1.0):
    """
    计算质点的（比）角动量 (z分量)。
    （比）角动量 Lz/m = x*vy - y*vx

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组。
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比角动量 (Lz/m)。

    返回:
        np.ndarray or float: （比）角动量。
    """
    if state_vector.ndim == 1:
        x, y, vx, vy = state_vector
        specific_Lz = x * vy - y * vx
        return specific_Lz * m if m != 1.0 else specific_Lz
    else:
        x, y, vx, vy = state_vector.T
        specific_Lz = x * vy - y * vx
        return specific_Lz * m if m != 1.0 else specific_Lz

def plot_orbits_by_energy(gm_val=1.0):
    """
    绘制不同能量下的轨道：椭圆、抛物线和双曲线
    """
    # 设置初始条件
    t_start = 0
    t_end_ellipse = 20
    t_end_parabola = 30
    t_end_hyperbola = 25
    t_eval_ellipse = np.linspace(t_start, t_end_ellipse, 1000)
    t_eval_parabola = np.linspace(t_start, t_end_parabola, 1000)
    t_eval_hyperbola = np.linspace(t_start, t_end_hyperbola, 1000)
    
    # 椭圆轨道 (E < 0)
    ic_ellipse = [1.0, 0.0, 0.0, 0.8]
    sol_ellipse = solve_orbit(ic_ellipse, (t_start, t_end_ellipse), t_eval_ellipse, gm_val)
    energy_ellipse = calculate_energy(sol_ellipse.y.T, gm_val)
    angular_momentum_ellipse = calculate_angular_momentum(sol_ellipse.y.T)
    
    # 抛物线轨道 (E ≈ 0)
    ic_parabola = [1.0, 0.0, 0.0, np.sqrt(2.0)]  # 逃逸速度
    sol_parabola = solve_orbit(ic_parabola, (t_start, t_end_parabola), t_eval_parabola, gm_val)
    energy_parabola = calculate_energy(sol_parabola.y.T, gm_val)
    angular_momentum_parabola = calculate_angular_momentum(sol_parabola.y.T)
    
    # 双曲线轨道 (E > 0)
    ic_hyperbola = [1.0, 0.0, 0.0, 1.5]
    sol_hyperbola = solve_orbit(ic_hyperbola, (t_start, t_end_hyperbola), t_eval_hyperbola, gm_val)
    energy_hyperbola = calculate_energy(sol_hyperbola.y.T, gm_val)
    angular_momentum_hyperbola = calculate_angular_momentum(sol_hyperbola.y.T)
    
    # 绘制轨道
    plt.figure(figsize=(12, 10))
    
    # 椭圆轨道
    plt.subplot(2, 2, 1)
    plt.plot(sol_ellipse.y[0], sol_ellipse.y[1], 'b-', label='椭圆轨道')
    plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
    plt.title(f'椭圆轨道 (E ≈ {energy_ellipse[0]:.4f})')
    plt.xlabel('x 坐标')
    plt.ylabel('y 坐标')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 抛物线轨道
    plt.subplot(2, 2, 2)
    plt.plot(sol_parabola.y[0], sol_parabola.y[1], 'g-', label='抛物线轨道')
    plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
    plt.title(f'抛物线轨道 (E ≈ {energy_parabola[0]:.4f})')
    plt.xlabel('x 坐标')
    plt.ylabel('y 坐标')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 双曲线轨道
    plt.subplot(2, 2, 3)
    plt.plot(sol_hyperbola.y[0], sol_hyperbola.y[1], 'r-', label='双曲线轨道')
    plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
    plt.title(f'双曲线轨道 (E ≈ {energy_hyperbola[0]:.4f})')
    plt.xlabel('x 坐标')
    plt.ylabel('y 坐标')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 所有轨道在同一张图上
    plt.subplot(2, 2, 4)
    plt.plot(sol_ellipse.y[0], sol_ellipse.y[1], 'b-', label=f'椭圆 (E≈{energy_ellipse[0]:.4f})')
    plt.plot(sol_parabola.y[0], sol_parabola.y[1], 'g-', label=f'抛物线 (E≈{energy_parabola[0]:.4f})')
    plt.plot(sol_hyperbola.y[0], sol_hyperbola.y[1], 'r-', label=f'双曲线 (E≈{energy_hyperbola[0]:.4f})')
    plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
    plt.title('不同能量的轨道对比')
    plt.xlabel('x 坐标')
    plt.ylabel('y 坐标')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # 绘制能量和角动量随时间的变化（可选）
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t_eval_ellipse, energy_ellipse, 'b-', label='椭圆轨道能量')
    plt.plot(t_eval_parabola, energy_parabola, 'g-', label='抛物线轨道能量')
    plt.plot(t_eval_hyperbola, energy_hyperbola, 'r-', label='双曲线轨道能量')
    plt.title('能量守恒验证')
    plt.xlabel('时间')
    plt.ylabel('能量')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(t_eval_ellipse, angular_momentum_ellipse, 'b-', label='椭圆轨道角动量')
    plt.plot(t_eval_parabola, angular_momentum_parabola, 'g-', label='抛物线轨道角动量')
    plt.plot(t_eval_hyperbola, angular_momentum_hyperbola, 'r-', label='双曲线轨道角动量')
    plt.title('角动量守恒验证')
    plt.xlabel('时间')
    plt.ylabel('角动量')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_orbits_by_angular_momentum(gm_val=1.0, rp=0.5, E=-0.5):
    """
    绘制固定能量下不同角动量的椭圆轨道
    """
    # 计算近心点速度
    vp = np.sqrt(2 * (E + gm_val / rp))
    
    # 设置三个不同的角动量
    Lz_values = [rp * vp * 0.8, rp * vp, rp * vp * 1.2]
    labels = [f'Lz = {Lz_values[0]:.4f}', f'Lz = {Lz_values[1]:.4f}', f'Lz = {Lz_values[2]:.4f}']
    
    # 计算对应的初始速度
    v0_values = [Lz / rp for Lz in Lz_values]
    
    # 设置积分参数
    t_start = 0
    t_end = 20
    t_eval = np.linspace(t_start, t_end, 1000)
    
    # 求解并绘制轨道
    plt.figure(figsize=(12, 10))
    
    # 计算理论轨道参数
    a = -gm_val / (2 * E)  # 半长轴
    plt.subplot(2, 2, 1)
    plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
    
    for i, v0 in enumerate(v0_values):
        # 初始条件：近心点在x轴正半轴
        initial_conditions = [rp, 0.0, 0.0, v0]
        
        # 求解轨道
        sol = solve_orbit(initial_conditions, (t_start, t_end), t_eval, gm_val)
        
        # 计算能量和角动量
        energy = calculate_energy(sol.y.T, gm_val)
        angular_momentum = calculate_angular_momentum(sol.y.T)
        
        # 计算偏心率
        e = np.sqrt(1 + (2 * energy[0] * angular_momentum[0]**2) / gm_val**2)
        
        # 绘制轨道
        plt.plot(sol.y[0], sol.y[1], label=f'{labels[i]}, e≈{e:.4f}')
        
        print(f"轨道 {i+1}:")
        print(f"  初始条件: x0={rp}, y0=0, vx0=0, vy0={v0:.4f}")
        print(f"  能量: {energy[0]:.6f}")
        print(f"  角动量: {angular_momentum[0]:.6f}")
        print(f"  偏心率: {e:.6f}")
        print(f"  半长轴: {a:.6f} (理论值)")
        print()
    
    plt.title('固定能量下不同角动量的椭圆轨道')
    plt.xlabel('x 坐标')
    plt.ylabel('y 坐标')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 绘制能量随时间的变化
    plt.subplot(2, 2, 2)
    for i, v0 in enumerate(v0_values):
        initial_conditions = [rp, 0.0, 0.0, v0]
        sol = solve_orbit(initial_conditions, (t_start, t_end), t_eval, gm_val)
        energy = calculate_energy(sol.y.T, gm_val)
        plt.plot(t_eval, energy, label=labels[i])
    
    plt.title('能量守恒验证')
    plt.xlabel('时间')
    plt.ylabel('能量')
    plt.legend()
    plt.grid(True)
    
    # 绘制角动量随时间的变化
    plt.subplot(2, 2, 3)
    for i, v0 in enumerate(v0_values):
        initial_conditions = [rp, 0.0, 0.0, v0]
        sol = solve_orbit(initial_conditions, (t_start, t_end), t_eval, gm_val)
        angular_momentum = calculate_angular_momentum(sol.y.T)
        plt.plot(t_eval, angular_momentum, label=labels[i])
    
    plt.title('角动量守恒验证')
    plt.xlabel('时间')
    plt.ylabel('角动量')
    plt.legend()
    plt.grid(True)
    
    # 绘制偏心率与角动量关系
    plt.subplot(2, 2, 4)
    Lz_range = np.linspace(Lz_values[0]*0.8, Lz_values[2]*1.2, 100)
    e_values = np.sqrt(1 + (2 * E * Lz_range**2) / gm_val**2)
    plt.plot(Lz_range, e_values, 'b-')
    
    # 标记三个轨道的偏心率
    for i, Lz in enumerate(Lz_values):
        e = np.sqrt(1 + (2 * E * Lz**2) / gm_val**2)
        plt.plot(Lz, e, 'ro', markersize=6)
        plt.annotate(f'e≈{e:.4f}', (Lz, e), textcoords="offset points", 
                     xytext=(0,10), ha='center')
    
    plt.title('偏心率与角动量关系')
    plt.xlabel('角动量 Lz')
    plt.ylabel('偏心率 e')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("平方反比引力场中的运动模拟")
    
    # 任务A：不同能量的轨道
    print("\n--- 任务A：不同能量的轨道 ---")
    plot_orbits_by_energy(gm_val=1.0)
    
    # 任务B：不同角动量的椭圆轨道
    print("\n--- 任务B：不同角动量的椭圆轨道 ---")
    plot_orbits_by_angular_momentum(gm_val=1.0, rp=0.5, E=-0.5)    
