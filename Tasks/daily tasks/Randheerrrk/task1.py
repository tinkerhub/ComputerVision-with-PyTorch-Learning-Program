# -*- coding: utf-8 -*-
"""task1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KiNRo7ltyENDERowPkIDEJ1uWrDw2h4n
"""

import torch as pt
import numpy as np

print(pt.matmul(pt.from_numpy(np.random.randn(5, 3)), pt.from_numpy(np.random.randn(3, 4))))