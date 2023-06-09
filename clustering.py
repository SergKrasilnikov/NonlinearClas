# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:47:23 2023

@author: Sergey
"""

#Nonlinear classification with Support Vector Machines (SVM)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from dbfread import DBF
import pandas as pd
import os
#import geopandas as gpd
#from shapely.geometry import LineString

# From pandas import DataFrame
dbf_n = DBF(os.path.join(os.getcwd(), 'data', 'class_n.dbf'))
df_n = pd.DataFrame(iter(dbf_n))
dbf_s = DBF(os.path.join(os.getcwd(), 'data', 'class_s.dbf'))
df_s = pd.DataFrame(iter(dbf_s))

# Add column with weights and colors
df_n = df_n.assign(Weight=1, Color='blue')
df_s = df_s.assign(Weight=-1, Color='orange')

# Извлечение массивов точек и меток классов
df = pd.concat([df_n[['x', 'y']], df_s[['x', 'y']]]).values
weight = pd.concat([df_n["Weight"], df_s["Weight"]]).values
color = pd.concat([df_n["Color"], df_s["Color"]]).values

# Обучение модели SVM с полиномиальным ядром
model = SVC(kernel='poly', degree=2)
model.fit(df, weight)

# Генерация сетки для построения разделяющей поверхности
x_min, x_max = df[:, 0].min() - 1, df[:, 0].max() + 1
y_min, y_max = df[:, 1].min() - 1, df[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Создание массива со значениями только для разделительной линии
#Z = np.zeros_like(xx)
# Установка значений только для разделительной линии
#Z[:, -1] = 1

# Прогнозирование меток классов для каждой точки в сетке
line = model.predict(np.c_[xx.ravel(), yy.ravel()])
line = line.reshape(xx.shape)


# Визуализация данных и разделяющей поверхности
plt.figure(figsize=(24, 16))
plt.scatter(df[:, 0], df[:, 1], c=color)
plt.contour(xx, yy, line, colors='green')
#plt.contourf(xx, yy, Z, alpha=0.1, cmap='bwr')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x', fontsize=25)
plt.ylabel('y', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Nonlinear classification with support vector machine', fontsize=25)
plt.grid(True)

# Отображение графика
plt.show()


# Gridline creation
# New DataFrame creation
line_coords = np.column_stack((xx.ravel(), yy.ravel(), line.ravel()))
df_line = pd.DataFrame(line_coords, columns=['x', 'y', 'line'])

df_new = pd.DataFrame(columns=['x', 'y'])

for i in range(len(df_line)):
    temp_num = 0
    if df_line.iloc[i]['line'] != temp_num:
        data = {'x': df_line.iloc[i]['x'], 'y': df_line.iloc[i]['y']}
        df_new = df_new.append(data, ignore_index=True)
        temp_num = df_line.iloc[i]['line']

print(df_new)

#df_line[(df_line["y"] != param) & (df_line["line"] == -1)]
#south_border = df_line['y']

#print(df_line["x"].mean())
#count = 10
#for value in df_line:
#    print(value)
#    count -= 1
#    if count < 0:
#        break



#from dbfpy import dbf

# Specify the output DBF file path
#output_dbf_file = 'line.dbf'

# Create a new DBF file
#db = dbf.Dbf(os.path.join(os.getcwd(), 'data/output/', 'line.dbf'), new=True)

# Create fields in the DBF file based on the DataFrame columns
#for column in df_line.columns:
#    print(column)
#    db.addField((column, 'N', 10, 5))

# Write data to the DBF file
#for _, row in df_line.iterrows():
#    rec = db.newRecord()
#    for column in df_line.columns:
#        rec[column] = row[column]
#    rec.store()

# Close the DBF file
#db.close()



#df_line.to_file('export.dbf')

# Указание пути к файлу DBF
#output_dbf_file = os.path.join('data/output', 'line.dbf')

# Преобразование DataFrame в список словарей
#data = df_line.to_dict('records')

# Сохранение данных в файл DBF
#with DBF(output_dbf_file, 'w') as dbf:
#    dbf.write(data)

#from dbfpy import dbf

# Specify the output DBF file path
#output_dbf_file = 'data/output/line.dbf'

# Create a new DBF file
#db = dbf.Dbf(output_dbf_file, new=True)

# Create fields in the DBF file based on the DataFrame columns
#for column in df_line.columns:
#    db.addField((column, 'C', 50))
    
# Write data to the DBF file
#for i, row in df_line.iterrows():
#    rec = db.newRecord()
#    for column in df.columns:
#        rec[column] = str(row[column])
#    rec.store()

# Close the DBF file
#db.close()



#from shapely.geometry import Point
# Создание DataFrame с координатами линии
#df_line = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1, 2, 3, 4]})

# Создание геометрического столбца с линией
#df_line['geometry'] = LineString(df_line[['x', 'y']].values)
#df_line['geometry'] = df_line.apply(lambda row: Point(row['x'], row['y']), axis=1)


# Создание GeoDataFrame из DataFrame
#gdf_line = gpd.GeoDataFrame(df_line, geometry='geometry')
#gdf_line = gpd.GeoDataFrame(df_line, geometry='geometry')
#print(gdf_line)

# Указание пути к файлу shapefile и сохранение GeoDataFrame
#output_shapefile = os.path.join(os.getcwd(), 'data/output/', 'line.shp')
#gdf_line.to_file(output_shapefile, driver='ESRI Shapefile')

#output_shapefile = os.path.join(os.getcwd(), 'data/output/', 'line.shp')
#gdf_line.to_file(output_shapefile, driver='ESRI Shapefile')








