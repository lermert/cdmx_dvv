# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from model_tools import model_SW_dG
p = np.linspace(0, 20.e6, 100)
plt.plot(p, model_SW_dG(p, 1, 1))
plt.ylabel("Velocity sensitivity to pressure (m/s / N)")
plt.xlabel("Pressure (N)")
plt.show()
