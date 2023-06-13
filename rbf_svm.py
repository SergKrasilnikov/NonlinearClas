# -*- coding: utf-8 -*-
"""
Nonlinear classification with Support Vector Machines (SVM) approach
with Gaussian Kernel (rbf) transformation

Created on Tue Jun 13 12:56:00 2023
@author: SergKrasilnikov
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC







# Визуализация данных и разделяющей поверхности
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(x1[:, 0], x1[:, 1], c='red', label='Class 1')
plt.scatter(x2[:, 0], x2[:, 1], c='blue', label='Class 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVM with Gaussian Kernel')
plt.legend()
plt.grid(True)

# Отображение графика
plt.show()