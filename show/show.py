"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import sys
sys.path.append('..')
from dataset import read, sizes

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)
plt.rc("font", family="Times New Roman")

mains = read("ukdale", 2)
apps = read("ukdale", 2, 'm')

plt.figure(figsize=(5, 3), dpi=300)
plt.plot(mains[:, 0], mains[:, 1], label='mains')
plt.plot(apps[:, 0], apps[:, 1], label='apps')
plt.ylabel("Power(W)")
plt.xlabel("Time(s)")
plt.legend(loc="upper right")
plt.show()
plt.close()
