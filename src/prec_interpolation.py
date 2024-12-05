import os
import numpy as np
import rasterio
import gc
import sys
try:
    import data_tools as dt
    import nc_tools as nc
except:
    from src import data_tools as dt
    from src import nc_tools as nc


def read_tif(fn):
    with rasterio.open(fn, 'r') as src:
        nodata = src.nodata
        data = src.read()
        bounds = src.bounds
    data = np.squeeze(data)
    data[data == nodata] = np.nan
    return data, bounds


def interpolate_precipitation(config_file, coarse_dem, fine_dem, precip_data, domain, land_sea_mask, full_domain = [0, 0, 0, 0]):
    """
        domain: [y_max, x_min, y_min, x_max]
    """
    interpolation_method = int(config_file['options']['precipitation_downscaling_method'])
    try:
        world_clim_data_dir = config_file['files']['worldclim_precipitation_data_dir']
    except:
        world_clim_data_dir = None

    if interpolation_method == 0:
        precip_data = precipitation_interpolation_nearestneighbour(coarse_dem, fine_dem, precip_data)
    elif interpolation_method == 1:
        precip_data = precipitation_interpolation_bilinear(fine_dem, precip_data)
    elif interpolation_method == 2:
        return
        fine_dem_full, _ = dt.load_specified_lines(config_file['files']['fine_dem'], full_domain, False)
        out_res = (abs(float(full_domain[1])) + abs(float(full_domain[3]))) / precip_data.shape[2]
        factors_file = calculate_prec_factors(fine_dem_full.shape, (int(np.round(precip_data.shape[2] * out_res)), np.shape(precip_data)[2]),
                                              world_clim_data_dir, full_domain)
        del fine_dem_full
        precip_data = precipitation_interpolation_worldclim(factors_file, fine_dem, precip_data, domain)

    precip_data[np.isnan(land_sea_mask)] = -32767
    return precip_data.astype(np.int16)

"""
def calculate_prec_factors(fine_dem_shape, coarse_dem_shape, world_clim_data_dir, full_domain):
    for idx, fn in enumerate(sorted([fname for fname in os.listdir(world_clim_data_dir) if fname.endswith('.tif') and not fname.startswith('factors')])):
        if os.path.exists(os.path.join(world_clim_data_dir, f'factors_month_{idx+1}.tif')):
            continue
        sys.stdout.write(f'     - Reading WorldClim data for month #{idx+1}                      '+'\r')
        sys.stdout.flush()

        world_clim_data, nan_value = dt.load_specified_lines(os.path.join(world_clim_data_dir, fn), full_domain, False)
        world_clim_data = world_clim_data.astype(np.float16)
        world_clim_data[world_clim_data == nan_value] = np.nan
        nan_mask = np.isnan(world_clim_data)
        world_clim_data = dt.fill_nan_nearest(world_clim_data)
        world_clim_data_coarse = dt.resize_array_mean(world_clim_data, coarse_dem_shape).astype(np.float16)
        world_clim_data_coarse = dt.resize_array_interp(world_clim_data_coarse, world_clim_data.shape)[0].astype(np.float16) #type:ignore

        # Calculate Factor
        if world_clim_data.shape[0] != fine_dem_shape[0]: #type:ignore
            dat, _ = dt.resize_array_interp(world_clim_data / world_clim_data_coarse, (fine_dem_shape[0], fine_dem_shape[1]))
            dat = dat.astype(np.float16)
        else:
            dat = (world_clim_data / world_clim_data_coarse).astype(np.float16)
        dat[nan_mask] = np.nan
        dat[(dat == np.inf) | (dat == -np.inf)] = 1
        dt.write_geotiff(world_clim_data_dir, f'factors_month_{idx+1}.tif', dat, full_domain, nodata_value=np.nan, inhibit_message=True)
        del world_clim_data, world_clim_data_coarse, dat

    sys.stdout.write(f'   -> WorldClim data read successfully                       '+'\r')
    sys.stdout.flush()

    return world_clim_data_dir
"""
def calculate_prec_factors(fine_dem_shape, coarse_dem_shape, world_clim_data_dir, full_domain):
    for idx, fn in enumerate(sorted([fname for fname in os.listdir(world_clim_data_dir) if fname.endswith('.tif') and not fname.startswith('factors')])):
        if os.path.exists(os.path.join(world_clim_data_dir, f'factors_month_{idx+1}.tif')):
            continue
        sys.stdout.write(f'     - Reading WorldClim data for month #{idx+1}                      '+'\r')
        sys.stdout.flush()

        world_clim_data, nan_value = dt.load_specified_lines(os.path.join(world_clim_data_dir, fn), full_domain, False)
        world_clim_data = world_clim_data.astype(np.float16)
        world_clim_data[world_clim_data == nan_value] = np.nan
        nan_mask = np.isnan(world_clim_data)
        world_clim_data = world_clim_data.astype(np.float16)
        world_clim_data = dt.fill_nan_nearest(world_clim_data)
        world_clim_data_coarse = dt.resize_array_mean(world_clim_data, coarse_dem_shape).astype(np.float16)
        world_clim_data_coarse = dt.resize_array_interp(world_clim_data_coarse, world_clim_data.shape)[0].astype(np.float16) #type:ignore

        # Calculate Factor
        if world_clim_data.shape[0] != fine_dem_shape[0]: #type:ignore
            dat, _ = dt.resize_array_interp(world_clim_data / world_clim_data_coarse, (fine_dem_shape[0], fine_dem_shape[1]))
            dat = dat.astype(np.float16)
        else:
            dat = (world_clim_data / world_clim_data_coarse).astype(np.float16)
        dat[nan_mask] = np.nan
        nc.write_to_netcdf(dat, os.path.join(world_clim_data_dir, f'factors_month_{idx+1}.nc'), extent=full_domain, compress=True, complevel=5, nodata_value=np.nan) #type:ignore
        #dt.write_geotiff(world_clim_data_dir, f'factors_month_{idx+1}.tif', dat, full_domain, nodata_value=np.nan, inhibit_message=True)
        del world_clim_data, world_clim_data_coarse, dat

    sys.stdout.write(f'   -> WorldClim data read successfully                       '+'\r')
    sys.stdout.flush()

    # Write to NetCDF
    #factors_file = dt.merge_geotiffs_to_multiband([os.path.join(world_clim_data_dir, f'factors_month_{idx+1}.tif') for idx in range(12)], factors_file)
    return world_clim_data_dir


def precipitation_interpolation_worldclim(factors_file, fine_dem, precip_data, domain, bias_thres=0.0001):
    covered = dt.check_geotiff_extent(factors_file, domain)
    if covered:
        factors, _ = dt.load_specified_lines(factors_file, domain)

    factors[np.isnan(factors)] = 1
    precip_data = (precip_data * 10).astype(np.int16)
    new_precip = np.empty((fine_dem.shape + (365,)), dtype=precip_data.dtype)
    for day in range(365):
        sys.stdout.write(f'     - Downscaling of precipitation data for day #{day+1}                      '+'\r')
        sys.stdout.flush()
        new_precip[..., day] = (dt.resize_array_interp(precip_data[day], fine_dem.shape))[0] * factors[int(day // (365 / 12))] #type:ignore
    
    # Ensure energy conservation by calulation of residues
    #dt.remove_residuals(new_precip, precip_data, bias_thres, 10)

    sys.stdout.write(f'   -> Downscaling of precipitation data completed successfully                       '+'\r')
    sys.stdout.flush()
    del factors, precip_data
    gc.collect()
    return new_precip


def precipitation_interpolation_nearestneighbour(coarse_dem, fine_dem, precip_data):
    ratio = fine_dem.shape[0] // coarse_dem.shape[0]
    precip_data = (precip_data * 10).astype(np.int16)
    new_precip = np.empty((fine_dem.shape + (365,)), dtype=precip_data.dtype)
    for day in range(precip_data.shape[0]):
        sys.stdout.write(f'     - Downscaling of precipitation data for day #{day+1}                      '+'\r')
        sys.stdout.flush()
        new_precip[..., day] = np.repeat(np.repeat(precip_data[day], ratio, axis=0), ratio, axis=1)
    sys.stdout.write(f'   -> Downscaling of precipitation data completed successfully                       '+'\r')
    sys.stdout.flush()
    return new_precip


def precipitation_interpolation_bilinear(fine_dem, precip_data):
    new_precip = np.empty((fine_dem.shape + (365,)), dtype=precip_data.dtype)
    precip_data = (precip_data * 10).astype(np.int16)
    for day in range(precip_data.shape[0]):
        sys.stdout.write(f'     - Downscaling of precipitation data for day #{day+1}                      '+'\r')
        sys.stdout.flush()
        new_precip[..., day] = (dt.resize_array_interp(precip_data[day], fine_dem.shape)[0]).astype(np.int16)
    sys.stdout.write(f'   -> Downscaling of precipitation data completed successfully                       '+'\r')
    sys.stdout.flush()
    return new_precip  
