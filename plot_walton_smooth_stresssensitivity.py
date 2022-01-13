# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from model_tools import model_SW_dG, model_SW_dsdp
p = np.linspace(0, 20.e6, 100)
vs = np.ones(100)
vs[0:10] *= 200
vs[10:100] *= 400

plt.plot(p, model_SW_dG(p, 1, 1))
plt.plot(p, model_SW_dsdp(p, vs))
plt.ylabel("Velocity sensitivity to pressure (m/s / N)")
plt.xlabel("Pressure (N)")
plt.show()
