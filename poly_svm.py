# -*- coding: utf-8 -*-
"""
Nonlinear classification with Support Vector Machines (SVM) approach
with Polynomial Kernel (poly) transformation

Created on Thu Jun  8 09:47:23 2023
@author: SergKrasilnikov
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from dbfread import DBF
import pandas as pd
import os
import geopandas as gpd

# From pandas import DataFrame
dbf_n = DBF(os.path.join(os.getcwd(), 'data', 'class_n.dbf'))
df_n = pd.DataFrame(iter(dbf_n))
dbf_s = DBF(os.path.join(os.getcwd(), 'data', 'class_s.dbf'))
df_s = pd.DataFrame(iter(dbf_s))

# Add column with weights and colors
df_n = df_n.assign(Weight=1, Color='blue')
df_s = df_s.assign(Weight=-1, Color='orange')

# Extract arrays of points and class labels
df = pd.concat([df_n[['x', 'y']], df_s[['x', 'y']]]).values
weight = pd.concat([df_n["Weight"], df_s["Weight"]]).values
color = pd.concat([df_n["Color"], df_s["Color"]]).values

# Model training SVM using kernels
# Degree of the polynomial is a parameter that determines the complexity of the
# decision boundary
model = SVC(kernel='poly', degree=15)
model.fit(df, weight)

# Grid generation for constructing a decision boundary
x_min, x_max = df[:, 0].min() - 1, df[:, 0].max() + 1
y_min, y_max = df[:, 1].min() - 1, df[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), 
                     np.arange(y_min, y_max, 0.1))

# Predict class labels for each point in the grid
line = model.predict(np.c_[xx.ravel(), yy.ravel()])
line = line.reshape(xx.shape)

# Gridline creation
# New DataFrame creation
line_coords = np.column_stack((xx.ravel(), yy.ravel(), line.ravel()))
df_line = pd.DataFrame(line_coords, columns=['x', 'y', 'line'])
df_new = pd.DataFrame(columns=['x', 'y', 'line'])
df_new.loc[ len(df_new.index )] = [df_line.iloc[0]['x'], df_line.iloc[0]['y'], 
                                   df_line.iloc[0]['line']]

# Save grid values to the line values
temp_num = 0
for i in range(len(df_line)):
    if df_line.iloc[i]['line'] != temp_num:
        #add row to end of DataFrame
        df_new.loc[ len(df_new.index )] = [df_line.iloc[i]['x'], 
                                           df_line.iloc[i]['y'], 
                                           df_line.iloc[i]['line']]
        temp_num = df_line.iloc[i]['line']
df_new = df_new.loc[df_new['x'] > df_line.iloc[0]['x']]

# Data visualization
plt.figure(figsize=(24, 16))
plt.scatter(df[:, 0], df[:, 1], c=color) # all points
plt.scatter(df_new['x'], df_new['y'], color='red', s=200) # drow border points
plt.contour(xx, yy, line, colors='green') # drow the line
#plt.contourf(xx, yy, line, alpha=0.1, cmap='bwr') # drow the grid
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x', fontsize=25)
plt.ylabel('y', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('SVM with Polynomial Kernel (poly)', fontsize=25)
plt.grid(True)

plt.show()


# EXPORT TO THE SHP
# Create a GeoDataFrame from the DataFrame
gdf = gpd.GeoDataFrame(df_new, geometry=gpd.points_from_xy(df_new['x'],
                                                            df_new['y']))
# Define the projection
crs = 'GCS_Mars_2000'
# Save the GeoDataFrame as a shapefile
gdf.to_file(os.path.join(os.getcwd(), 'data/output/', 'output.shp'),
            driver='ESRI Shapefile', crs=crs)

# EXPORT DataFrame TO CSV
df_new.to_csv(os.path.join(os.getcwd(), 'data/output/', 'line.csv'),
             index= True )
