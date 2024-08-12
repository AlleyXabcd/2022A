import numpy as np
from matplotlib import pyplot as plt

from a2_1 import objective_function

c = np.arange(0,100000,1000)

result=[]
for i in c:
    result.append(-objective_function(i))

# 绘制优化过程
plt.plot(result)

plt.grid(True)
plt.show()