import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def ini_position(p, m1, m2, g, h, r, k, l0):
    m = m1 + m2
    # 计算圆锥部分的体积
    V = 1 / 3 * math.pi * r * r * h

    if p * g * V < m * g:
        return [h + l0 - m1 * g / k, (m * g - p * g * V) / (math.pi * r * r * p * g) + h]
    else:
        return [h + l0 - m1 * g / k, math.pow((3 * m * h * h) / (p * math.pi * r * r), 1 / 3)]


def solve_v(x, l, h, r, h2):  # 根据浮子的位移确定排水量
    if x >= l:
        return 0
    elif l > x >= l - h:
        return 1 / 3 * math.pi * r * r / (h * h) * (l - x) ** 3
    elif l - h2 - h <= x < l - h:
        return 1 / 3 * math.pi * r * r * h + math.pi * r * r * (l - x - h)
    elif x < l - h2 - h:
        return 1 / 3 * math.pi * r * r * h + math.pi * r * r * h2


def PTO(x1, x2, v1, v2, s0, h, k, l0):  # 计算PTO相关的力
    # 计算弹簧长度
    z = s0 - h + x2 - x1

    # 计算弹力 考虑方向
    F_tan = k * (z - l0)

    # c = 10000  # 直线阻尼器的阻尼系数  1.10000  2.10000*abs(v1-v2)**0.5

    c = 10000

    F_zu = -c * (v1 - v2)

    return F_tan + F_zu


# 定义微分方程
def system_of_equations(t, y, params):
    x1, v1, x2, v2 = y
    f, w, p, g, s1, h_zhui, r, h_zhu, s0, k, l0, a, m_fu, m_fujia, m_zhen = params
    v_pai = solve_v(x1, s1, h_zhui, r, h_zhu)
    F_PTO = PTO(x1, x2, v1, v2, s0, h_zhui, k, l0)

    dydt = [
        v1,
        (f * np.cos(w * t) - a * v1 + p * g * v_pai - m_fu * g + F_PTO) / (m_fu + m_fujia),
        v2,
        (-F_PTO - m_zhen * g) / m_zhen
    ]
    return dydt


def main():
    # 参数
    f = 6250  # 垂直激励力振幅
    w = 1.4005  # 波浪圆频率
    m_zhen = 2433  # 振子质量
    m_fu = 4866  # 浮子质量
    m_fujia = 1335.535  # 附加质量
    dt = 0.2  # 时间间隔
    p = 1025  # 海水密度
    g = 9.8
    h_zhui = 0.8  # 圆锥部分高度
    r = 1  # 底面圆半径
    h_zhu = 3
    k = 80000  # 弹簧刚度
    l0 = 0.5  # 弹簧原长
    a = 656.3616  # 兴波阻尼系数
    s0, s1 = ini_position(p, m_zhen, m_fu, g, h_zhui, r, k, l0)  # （平衡） s0是振子底部至圆锥顶点的距离，s1为初始时刻海平面距离圆锥顶点的距离

    T = 2 * math.pi / w  # 周期

    steps = 40 * T / dt

    # 初始条件
    y0 = [0, 0, 0, 0]
    t_span = (0, int(steps) * dt)
    t_eval = np.arange(0, int(steps) * dt, dt)

    # Define parameters for ODE
    params = (f, w, p, g, s1, h_zhui, r, h_zhu, s0, k, l0, a, m_fu, m_fujia, m_zhen)
    # 使用 solve_ivp 求解
    solution = solve_ivp(system_of_equations, t_span, y0, t_eval=t_eval, method='RK45', args=(params,))

    # 提取解
    t = solution.t
    x_fu = solution.y[0]
    v_fu = solution.y[1]
    x_zhen = solution.y[2]
    v_zhen = solution.y[3]

    # 绘制结果
    plt.plot(t, x_fu, '--', label='x_fu')
    plt.plot(t, x_zhen, '*', label='x_zhen')
    plt.xlabel('Time t')
    plt.ylabel('y1(t), y2(t)')
    plt.title('Solution of the Differential Equation System')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the results to a CSV file
    df = pd.DataFrame({'time': t, 'x_fu': x_fu, 'v_fu': v_fu, 'x_zhen': x_zhen, 'v_zhen': v_zhen})
    df.to_csv('result1-1.csv', index=False)


if __name__ == "__main__":
    main()
