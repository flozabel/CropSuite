import numpy as np
import rasterio 
import data_tools as dt
import nc_tools as nc
import numpy as np


def read_climate_params_for_timestep(filename, extent):
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        with rasterio.open(filename) as src:
            data = np.asarray(src.read(), dtype=np.float16)
            bounds = src.bounds
            nodata = src.nodata
        data = dt.extract_domain_from_global_raster(data, extent, raster_extent=bounds)
        return data, nodata
    else:
        #y_max, x_min, y_min, x_max
        data, _ = nc.read_area_from_netcdf(filename, extent=[extent[1], extent[0], extent[3], extent[2]])
        data = np.asarray(data).transpose(2, 0, 1)
        nodata = nc.get_nodata_value(filename)
        return data, nodata



         


