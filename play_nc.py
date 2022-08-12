import netCDF4 as nc
fn = 'C:/Users/Administrator/Downloads/Dataset/SampleNC/ERS_SSM_H_19971201_012159.nc'
ds = nc.Dataset(fn)
print(ds)