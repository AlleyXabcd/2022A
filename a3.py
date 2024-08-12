import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def ini_position(p, m1, m2, g, h, r, k1, l0):
    m = m1 + m2
    # 计算圆锥部分的体积
    V = 1 / 3 * math.pi * r * r * h

    if p * g * V < m * g:
        return [h + l0 - m1 * g / k1, (m * g - p * g * V) / (math.pi * r * r * p * g) + h]
    else:
        return [h + l0 - m1 * g / k1, math.pow((3 * m * h * h) / (p * math.pi * r * r), 1 / 3)]


def solve_v(x, l, h, r, h2):  # 根据浮子的位移确定排水量
    if x >= l:
        return 0
    elif l > x >= l - h:
        return 1 / 3 * math.pi * r * r / (h * h) * (l - x) ** 3
    elif l - h2 - h <= x < l - h:
        return 1 / 3 * math.pi * r * r * h + math.pi * r * r * (l - x - h)
    elif x < l - h2 - h:
        return 1 / 3 * math.pi * r * r * h + math.pi * r * r * h2


def PTO(x1, x2, v1, v2, s0, h, k1, l0):  # 计算PTO相关的力
    # 计算弹簧长度
    z = s0 - h + x2 - x1

    # 计算弹力 考虑方向
    F_tan = k1 * (z - l0)

    c = 10000  # 直线阻尼器的阻尼系数  1.10000  2.10000*abs(v1-v2)**0.5

    F_zu = -c * (v1 - v2)

    return F_tan + F_zu


# 定义微分方程
def system_of_equations(t, y, params):
    x1, v1, x2, v2, theta1, omega1, theta2, omega2 = y
    f, w, p, g, s1, h_zhui, r, h_zhu, s0, k1, l0, a, m_fu, m_fujia, m_zhen, r_zhen, h_zhen, L, b, k_jin, k2, k_xuan, j_fujia = params
    v_pai = solve_v(x1, s1, h_zhui, r, h_zhu)
    F_PTO = PTO(x1, x2, v1, v2, s0, h_zhui, k1, l0)

    # 浮子相对转轴的转动惯量(常量)
    j_fu = 1 / 12 * m_fu * 9 / (9 + 0.8) * (3 * r ** 2 + h_zhu ** 2) + m_fu * 9 / (9 + 0.8) * (
            h_zhu / 2) ** 2 + 3 / 20 * m_fu * 0.8 / (9 + 0.8) * (r ** 2 + h_zhui ** 2 / 4) + m_fu * 0.8 / (
                   9 + 0.8) * (h_zhui / 3) ** 2

    # 振子相对转轴的转动惯量（变量）
    j_zhen = 1 / 12 * m_zhen * (3 * r_zhen ** 2 + h_zhen ** 2) + m_zhen * (s0 - h_zhui + x2 - x1 + h_zhen / 2) ** 2

    dydt = [
        v1,
        (f * np.cos(w * t) - a * v1 + p * g * v_pai - m_fu * g + F_PTO * np.cos(theta2)) / (m_fu + m_fujia),
        v2,
        (-F_PTO * np.cos(theta2) - m_zhen * g) / m_zhen,

        omega1,
        (L * np.cos(w * t) - b * omega1 - k_jin * theta1 - k2 * (theta1 - theta2) - k_xuan * (omega1 - omega2)) / (
                j_fu + j_fujia),

        omega2,

        (k2 * (theta1 - theta2) + k_xuan * (omega1 - omega2)) / j_zhen

        # 考虑重力力矩
        # (m_zhen * g * (s0 - h_zhui + x2 - x1 + h_zhen / 2) * np.sin(theta2) + k2 * (theta1 - theta2) + k_xuan * (omega1 - omega2)) / j_zhen

    ]
    return dydt


def main():
    # 参数
    f = 3640  # 垂直激励力振幅
    L = 1690  # 纵摇激励力矩振幅
    w = 1.7152  # 波浪圆频率
    m_fujia = 1028.876  # 附加质量
    j_fujia = 7001.914  # 附加转动惯量
    a = 683.4558  # 垂荡兴波阻尼系数
    b = 654.3383  # 纵摇兴波阻尼系数
    dt = 0.2  # 时间间隔

    m_zhen = 2433  # 振子质量
    m_fu = 4866  # 浮子质量
    p = 1025  # 海水密度
    g = 9.8
    h_zhui = 0.8  # 圆锥部分高度
    r = 1  # 底面圆半径
    h_zhu = 3
    k1 = 80000  # 弹簧刚度
    k2 = 250000  # 扭转弹簧刚度
    l0 = 0.5  # 弹簧原长
    r_zhen = 0.5  # 振子半径
    h_zhen = 0.5  # 振子高度

    k_jin = 8890.7  # 静水恢复力力矩系数
    k_xuan = 1000  # 旋转阻尼器的阻尼系数
    s0, s1 = ini_position(p, m_zhen, m_fu, g, h_zhui, r, k1, l0)  # （平衡） s0是振子底部至圆锥顶点的距离，s1为初始时刻海平面距离圆锥顶点的距离

    T = 2 * math.pi / w  # 周期

    steps = 40 * T / dt

    # 初始条件
    y0 = [0, 0, 0, 0, 0, 0, 0, 0]
    t_span = (0, int(steps) * dt)
    t_eval = np.arange(0, int(steps) * dt, dt)

    # Define parameters for ODE
    params = (
        f, w, p, g, s1, h_zhui, r, h_zhu, s0, k1, l0, a, m_fu, m_fujia, m_zhen, r_zhen, h_zhen, L, b, k_jin, k2, k_xuan,
        j_fujia)

    # 使用 solve_ivp 求解
    solution = solve_ivp(system_of_equations, t_span, y0, t_eval=t_eval, method='RK45', args=(params,))

    # 提取解
    t = solution.t
    x_fu = solution.y[0]
    v_fu = solution.y[1]
    x_zhen = solution.y[2]
    v_zhen = solution.y[3]
    theta_fu = solution.y[4]
    omega_fu = solution.y[5]
    theta_zhen = solution.y[6]
    omega_zhen = solution.y[7]

    # 绘制结果
    plt.plot(t, x_fu, '--', label='x_fu')
    plt.plot(t, x_zhen, '*', label='x_zhen')
    plt.xlabel('Time t')
    plt.ylabel('y1(t), y2(t)')
    plt.title('Solution of the Differential Equation System')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(t, theta_fu, '--', label='theta_fu')
    plt.plot(t, theta_zhen, '--', label='theta_zhen')
    plt.xlabel('Time t')
    plt.ylabel('y1(t), y2(t)')
    plt.title('Solution of the Differential Equation System')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the results to a CSV file
    df = pd.DataFrame(
        {'time': t, 'x_fu': x_fu, 'v_fu': v_fu, 'x_zhen': x_zhen, 'v_zhen': v_zhen, 'theta_fu': theta_fu,
         'omega_fu': omega_fu, 'theta_zhen': theta_zhen, 'omega_zhen': omega_zhen})
    df.to_csv('result1-1.csv', index=False)


if __name__ == "__main__":
    main()
