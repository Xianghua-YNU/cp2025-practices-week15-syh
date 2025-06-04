[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/0UiTaIgw)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=19704629)
# 第15周作业：边值问题数值解法与物理建模

本周作业主要关注常微分方程边值问题的数值解法，以及物理系统的建模与仿真。通过四个项目，学生将学习并实践有限差分法、打靶法等边值问题求解方法，以及物理系统的数值模拟。

## 项目列表

### 项目1：有限差分法求解边值问题
- **目录**：[PROJECT_1_FiniteDifferenceBVP](./PROJECT_1_FiniteDifferenceBVP)
- **说明文档**：[项目说明](./PROJECT_1_FiniteDifferenceBVP/项目说明.md)
- **主要任务**：
  - 实现有限差分法求解二阶常微分方程边值问题
  - 处理非线性边值问题
  - 将高阶ODE系统转换为一阶系统
  - 封装scipy.integrate.solve_bvp方法
  - 比较不同方法的精度和效率

### 项目2：打靶法与scipy.integrate.solve_bvp比较
- **目录**：[PROJECT_2_ShootingMethod](./PROJECT_2_ShootingMethod)
- **说明文档**：[项目说明](./PROJECT_2_ShootingMethod/项目说明.md)
- **主要任务**：
  - 实现打靶法求解二阶常微分方程边值问题
  - 处理非线性边值问题
  - 使用scipy.optimize.root_scalar进行根查找
  - 与scipy.integrate.solve_bvp方法比较
  - 分析两种方法的优缺点

### 项目3：平方反比引力场中的运动模拟
- **目录**：[PROJECT_3_InverseSquareLawMotion](./PROJECT_3_InverseSquareLawMotion)
- **说明文档**：[项目说明](./PROJECT_3_InverseSquareLawMotion/项目说明.md)
- **主要任务**：
  - 数值求解平方反比引力场中质点的运动方程
  - 使用scipy.integrate.solve_ivp求解常微分方程组
  - 验证能量和角动量守恒
  - 模拟不同初始条件下的轨道
  - 可视化轨道和物理量

### 项目4：双摆动力学仿真
- **目录**：[PROJECT_4_DoublePendulumSimulation](./PROJECT_4_DoublePendulumSimulation)
- **说明文档**：[项目说明](./PROJECT_4_DoublePendulumSimulation/项目说明.md)
- **主要任务**：
  - 实现双摆系统的运动方程
  - 使用scipy.integrate.odeint求解常微分方程组
  - 计算系统能量并分析能量守恒
  - 调整数值方法参数提高计算精度
  - 创建双摆运动动画（可选）

## 提交要求

### 代码提交
- 完成所有学生模板代码的实现
- 确保代码通过基本测试用例
- 遵循Python编程规范（PEP 8）
- 包含必要的注释和文档字符串

### 实验报告
- 完成每个项目的实验报告模板
- 所有图表标注必须使用英文
- 包含完整的数值结果和分析讨论
- 提交格式：Markdown文件

### 提交截止时间
**2025年6月7日06:00 (星期六凌晨6点)**

## 评分标准

本次作业总分为 **40分**，共4个项目，每个项目10分。

### 项目评分分配
- **项目1：有限差分法求解边值问题** (10分)
- **项目2：打靶法求解边值问题** (10分)
- **项目3：引力场中的轨道运动** (10分)
- **项目4：双摆动力学仿真** (10分)

### 每个项目的评分要点
- **代码实现正确性** (6分)：函数实现正确，通过测试用例
- **数值结果准确性** (2分)：计算结果符合物理预期
- **图表质量和分析** (2分)：图表清晰美观，包含英文标注，结果分析合理

### 通用要求
- **图表标注**：所有图表的标题、轴标签、图例必须使用英文
- **代码规范**：遵循PEP 8规范，包含适当的注释和文档字符串
- **实验报告**：按模板完成，分析深入，讨论合理
- **学术诚信**：独立完成，引用资料需注明来源

## 技术支持
- Python 3.8+，依赖见 `requirements.txt`
