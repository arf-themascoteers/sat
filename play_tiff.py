from osgeo import gdal
import matplotlib.pyplot as plt

dataset = gdal.Open(r'C:/Users/Administrator/Downloads/Dataset/TZ_SOC/BD_l4.tif')
print(dataset)