# 2_1的可视化
'''
import numpy as np
from matplotlib import pyplot as plt

from a2_1 import objective_function

c = np.arange(0,100000,1000)

result=[]
for i in c:
    result.append(-objective_function(i))

# 绘制优化过程
plt.plot(c,result)

plt.grid(True)
plt.show()
'''

# 2_2的可视化
'''
import numpy as np
from matplotlib import pyplot as plt


from a2_2 import objective_function

e = np.arange(0,100000,1000)
alpha = np.arange(0,1,0.1)

# 创建 meshgrid
E, Alpha = np.meshgrid(e, alpha)

# 计算每一对 (E, Alpha) 对应的目标函数值
Z = np.zeros_like(E)
for i in range(E.shape[0]):
    for j in range(E.shape[1]):
        Z[i, j] = -objective_function([np.array([E[i, j]]), np.array([Alpha[i, j]])])
# 绘制优化过程

ax = plt.axes(projection='3d')
ax.plot_surface(E, Alpha, Z, cmap='viridis')
plt.grid(True)
plt.show()
'''

# 4的可视化
import numpy as np
from matplotlib import pyplot as plt

from a4 import objective_function

k_zhi = np.arange(0, 100000, 1000)
k_xuan = np.arange(0, 100000, 1000)

# 创建 meshgrid
K_zhi, K_xuan = np.meshgrid(k_zhi, k_xuan)

# 计算每一对 (E, Alpha) 对应的目标函数值
p = np.zeros_like(K_zhi)
for i in range(K_zhi.shape[0]):
    for j in range(K_zhi.shape[1]):
        p[i, j] = -objective_function([np.array([K_zhi[i, j]]), np.array([K_xuan[i, j]])])
# 绘制优化过程

ax = plt.axes(projection='3d')
ax.plot_surface(K_zhi, K_xuan, p, cmap='viridis')
plt.grid(True)
plt.show()
