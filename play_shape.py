import geopandas as gpd
#import gdal
import shapely
from osgeo import gdal

print(gdal)

gdf = gpd.read_file("C:/Users/Administrator/Downloads/Dataset/ERF/soil_carbon/sustainable intensification.shp")
print(gdf.head())