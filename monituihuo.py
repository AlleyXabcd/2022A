import numpy as np


# 模拟退火算法的实现
# objective 目标函数
# bounds 边界调节
# T0初始温度
# Tf终止温度
# alpha 冷却率（0到1之间）
# max_iter最大迭代次数
def simulated_annealing(objective, bounds, T0=1000, Tf=1e-5, alpha=0.99, max_iter=1000):
    # 随机初始化解
    num = len(bounds)
    current_solution = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(num)]
    current_value = objective(current_solution)

    # 设置初始解为最佳解
    best_solution = current_solution
    best_value = current_value

    # 初始温度
    temp = T0

    process_solution = []
    process_value = []

    for i in range(max_iter):
        # 生成一个新的解，通过在当前解的附近随机扰动
        new_solution = []
        for j in range(num):
            perturbation = np.random.uniform(-1000, 1000)  # 扰动服从均匀分布
            new_value_j = current_solution[j] + perturbation
            # 确保新解在范围内
            new_value_j = max(bounds[j][0], min(bounds[j][1], new_value_j))
            new_solution.append(new_value_j)

        new_value = objective(new_solution)

        process_solution.append(new_solution)
        process_value.append(new_value)

        # 计算接受新解的概率
        delta = new_value - current_value
        acceptance_probability = np.exp(-delta / temp) if delta > 0 else 1.0

        # 决定是否接受新解
        if np.random.uniform(0, 1) < acceptance_probability:
            current_solution = new_solution
            current_value = new_value

            # 如果新解是目前最好的解，则更新最佳解
            if new_value < best_value:
                best_solution = new_solution
                best_value = new_value

        # 降低温度
        temp *= alpha

        # 如果温度低于终止温度，终止算法
        if temp < Tf:
            break

    return best_solution, best_value, process_solution, process_value

