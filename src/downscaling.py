import os
import numpy as np
import rasterio
import sys
try:
    import data_tools as dt
    import nc_tools as nc
    import temp_interpolation as ti
    import prec_interpolation as pi
except:
    from src import data_tools as dt
    from src import nc_tools as nc
    from src import temp_interpolation as ti
    from src import prec_interpolation as pi
from scipy.stats import linregress
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math
from skimage import transform as skt

temp = os.path.join(os.getcwd(), 'temp')
os.makedirs(temp, exist_ok=True)


### GENERAL ###

def read_timestep(filename, extent, timestep=-1):
    if timestep == -1:
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            with rasterio.open(filename) as src:
                data = np.asarray(src.read(), dtype=np.float16)
                bounds = src.bounds
                nodata = src.nodata
            data = dt.extract_domain_from_global_raster(data, extent, raster_extent=bounds)
            return data, nodata
        else:
            data, _ = nc.read_area_from_netcdf(filename, extent=[extent[1], extent[0], extent[3], extent[2]])
            data = np.asarray(data).transpose(2, 0, 1)
            nodata = nc.get_nodata_value(filename)
            return data, nodata
    else:
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            data, nodata = dt.load_specified_lines(filename, extent, all_bands=timestep+1) #type:ignore
            return data, nodata
        else:
            varname = nc.get_variable_name_from_nc(filename)
            data, _ = nc.read_area_from_netcdf(filename, extent=extent, day_range=timestep, variable=varname)
            nodata = nc.get_nodata_value(filename)
            return data, nodata


### TEMPERATURE ###

### MAIN TEMPERATURE METHOD ###

def interpolate_temperature(config_file, domain, area_name):
    """
        domain: [y_max, x_min, y_min, x_max]
    """
    interpolation_method = int(config_file['options']['temperature_downscaling_method'])
    output_dir = os.path.join(config_file['files']['output_dir']+'_downscaled', area_name)
    os.makedirs(output_dir, exist_ok=True)
    if interpolation_method == 0:
        temp_files = temperature_interpolation_nearestneighbour(config_file, domain, output_dir)
    elif interpolation_method == 1:
        temp_files = temperature_interpolation_bilinear(config_file, domain, output_dir)
    elif interpolation_method == 2:
        temp_files = worldclim_downscaling_temp(domain, config_file, output_dir)
    elif interpolation_method == 3:
        temp_files = temperature_interpolation_height_regression(config_file, domain, output_dir)
    return temp_files, True

def worldclim_downscaling_temp(extent, config_file, output_dir):
    if all(j for j in [os.path.exists(os.path.join(output_dir, f'ds_temp_{i}.nc')) for i in range(0, 365)]) :
        return [os.path.join(output_dir, f'ds_temp_{day}.nc') for day in range(0, 365)]
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    world_clim_data_dir = os.path.join(config_file['files']['worldclim_temperature_data_dir'])

    fine_resolution = dt.get_resolution_array(config_file, extent, True)

    #fine_dem, _ = dt.load_specified_lines(config_file['files']['fine_dem'], extent, False)
    #fine_resolution = fine_dem.shape
    land_sea_mask, _ = dt.load_specified_lines(config_file['files']['land_sea_mask'], extent, False)
    land_sea_mask = np.asarray(land_sea_mask).astype(np.float16)
    land_sea_mask[land_sea_mask == 0] = np.nan
    if land_sea_mask.shape[0] != fine_resolution[0]:
        land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_resolution)
    if not os.path.exists(os.path.join(climate_data_dir, 'Temp_avg.tif')):
        temp_file = os.path.join(climate_data_dir, 'Temp_avg.nc')
    else:
        temp_file = os.path.join(climate_data_dir, 'Temp_avg.tif')
    coarse_resolution = read_timestep(temp_file, extent=extent, timestep=0)[0].shape
    ext = nc.check_extent_load_file(os.path.join(world_clim_data_dir, f'factors_month_1.nc'), extent=extent)
    [os.remove(file) for i in range(1, 13) if not ext and (file := os.path.join(world_clim_data_dir, f'factors_month_{i}.nc')) and os.path.exists(file)]
    if not all(os.path.exists(os.path.join(world_clim_data_dir, f'factors_month_{band_index}.nc')) for band_index in range(1, 13)):
        world_clim_data_dir = ti.calculate_temp_factors(fine_resolution, coarse_resolution, world_clim_data_dir, extent)
    temp_data, _ = dt.load_specified_lines(temp_file, extent=extent)
    parallel_processing(temp_data, extent, fine_resolution, output_dir, 'temp', np.isnan(land_sea_mask), world_clim_data_dir)
    return [os.path.join(output_dir, f'ds_temp_{day}.nc') for day in range(0, 365)]

def worldclim_downscaling_temp_singleslice(data_slice, extent, worldclim_factors, fine_resolution, day, output_dir):
    if os.path.exists(os.path.join(output_dir, f'ds_temp_{day}.nc')):
        return
    worldclim_factors[np.isnan(worldclim_factors)] = 0
    if worldclim_factors.shape != fine_resolution:
        worldclim_factors = dt.resize_array_mean(worldclim_factors, fine_resolution)
    temp_data = skt.resize(data_slice - 273.15, fine_resolution, order=1, mode='edge', anti_aliasing=False).astype(np.float32)
    nc.write_to_netcdf(temp_data, os.path.join(output_dir, f'ds_temp_{day}.nc'), extent=extent, compress=True, complevel=9) #type:ignore

def temperature_interpolation_bilinear(config_file, extent, output_dir):
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    #fine_dem, _ = dt.load_specified_lines(config_file['files']['fine_dem'], extent, False)
    #fine_resolution = fine_dem.shape
    fine_resolution = dt.get_resolution_array(config_file, extent, True)
    land_sea_mask, _ = dt.load_specified_lines(config_file['files']['land_sea_mask'], extent, False)
    land_sea_mask = np.asarray(land_sea_mask).astype(np.float16)
    land_sea_mask[land_sea_mask == 0] = np.nan
    if land_sea_mask.shape[0] != fine_resolution[0]:
        land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_resolution)
    if not os.path.exists(os.path.join(climate_data_dir, 'Temp_avg.tif')):
        temp_file = os.path.join(climate_data_dir, 'Temp_avg.nc')
    else: temp_file = os.path.join(climate_data_dir, 'Temp_avg.tif')
    for day in range(365):
        if os.path.exists(os.path.join(output_dir, f'ds_temp_{day}.tif')) or os.path.exists(os.path.join(output_dir, f'ds_temp_{day}.nc')):
            continue
        sys.stdout.write(f'     - Downscaling of temperature data for day #{day+1}                      '+'\r')
        sys.stdout.flush()
        temp_data, temp_nodata = read_timestep(temp_file, extent=extent, timestep=day)
        temp_data[(temp_data < -100) | (temp_data > 60) |(temp_data == temp_nodata)] = np.nan
        temp_data = skt.resize(temp_data * 10, fine_resolution, order=1, mode='edge', anti_aliasing=False).astype(np.int16)
        temp_data[np.isnan(land_sea_mask)] = -32767
        nc.write_to_netcdf(temp_data, os.path.join(output_dir, f'ds_temp_{day}.nc'), extent=extent, compress=True, nodata_value=-32767) #type:ignore
    return [os.path.join(output_dir, f'ds_temp_{day}.nc') for day in range(0, 365)]

def temperature_interpolation_nearestneighbour(config_file, extent, output_dir):
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    #fine_dem, _ = dt.load_specified_lines(config_file['files']['fine_dem'], extent, False)
    #fine_resolution = fine_dem.shape
    fine_resolution = dt.get_resolution_array(config_file, extent, True)
    land_sea_mask, _ = dt.load_specified_lines(config_file['files']['land_sea_mask'], extent, False)
    land_sea_mask = np.asarray(land_sea_mask).astype(np.float16)
    land_sea_mask[land_sea_mask == 0] = np.nan
    if land_sea_mask.shape[0] != fine_resolution[0]:
        land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_resolution)
    if not os.path.exists(os.path.join(climate_data_dir, 'Temp_avg.tif')):
        temp_file = os.path.join(climate_data_dir, 'Temp_avg.nc')
    else: temp_file = os.path.join(climate_data_dir, 'Temp_avg.tif')
    for day in range(365):
        if os.path.exists(os.path.join(output_dir, f'ds_temp_{day}.tif')) or os.path.exists(os.path.join(output_dir, f'ds_temp_{day}.nc')):
            continue
        sys.stdout.write(f'     - Downscaling of temperature data for day #{day+1}                      '+'\r')
        sys.stdout.flush()
        temp_data, temp_nodata = read_timestep(temp_file, extent=extent, timestep=day)
        ratio = fine_resolution[0] // temp_data.shape[0]
        temp_data[(temp_data < -100) | (temp_data > 60) |(temp_data == temp_nodata)] = np.nan
        temp_data = (np.repeat(np.repeat(temp_data * 10, ratio, axis=0), ratio, axis=1)).astype(np.int16)
        temp_data[np.isnan(land_sea_mask)] = -32767
        nc.write_to_netcdf(temp_data, os.path.join(output_dir, f'ds_temp_{day}.nc'), extent=extent, compress=True, nodata_value=-32767) #type:ignore
    return [os.path.join(output_dir, f'ds_temp_{day}.nc') for day in range(0, 365)]

def height_regress_day(day, output_dir, temp_data, domain, window_size, coarse_dem, fine_dem, fine_resolution, use_gradient, precip_day_sum_thres, saturation_adiabatic_gradient, dryadiabatic_gradient, bias_thres, land_sea_mask):
    if os.path.exists(os.path.join(output_dir, f'ds_temp_{day}.tif')) or os.path.exists(os.path.join(output_dir, f'ds_temp_{day}.nc')):
        return None
    sys.stdout.write(f'     - Downscaling of temperature data for day #{day + 1}                      ' + '\r')
    sys.stdout.flush()
    prec_data, _ = nc.read_area_from_netcdf(os.path.join(output_dir, f'ds_prec_{day}.nc'), extent=domain)
    temp_arr = np.zeros((np.shape(fine_dem)[0], np.shape(fine_dem)[1]), dtype=np.float16)
    if window_size[0] >= temp_data.shape[0] or window_size[1] >= temp_data.shape[1]:
        window_size = (temp_data.shape[0] - 1, temp_data.shape[1] - 1)
    dem_window = np.lib.stride_tricks.sliding_window_view(coarse_dem, window_size)
    temp_window = np.lib.stride_tricks.sliding_window_view(temp_data, window_size)
    slopes = np.zeros_like(temp_data)
    offsets = np.zeros_like(temp_data)
    half_window_i, half_window_j = window_size[0] // 2, window_size[1] // 2
    for i in range(coarse_dem.shape[0]):
        if i >= dem_window.shape[0]: i = dem_window.shape[0]-1
        for j in range(coarse_dem.shape[1]):
            if j >= dem_window.shape[1]: j = dem_window.shape[1]-1
            x, y = dem_window[i, j].ravel(), temp_window[i, j].ravel()
            mask = ~np.isnan(x) & ~np.isnan(y)
            if np.sum(np.isnan(x)) >= len(x) - np.count_nonzero(np.isnan(x)): continue
            try:
                slope, offset, _, _, _ = linregress(x[mask], y[mask])
            except:
                slope, offset = 0, np.nanmean(y[mask])
            slopes[i + half_window_i, j + half_window_j] = slope
            offsets[i + half_window_i, j + half_window_j] = offset
    slopes = skt.resize(np.nan_to_num(slopes), fine_resolution, order=1, mode='edge', anti_aliasing=False)
    if use_gradient:
        grad = np.where(prec_data >= precip_day_sum_thres * 10, saturation_adiabatic_gradient, dryadiabatic_gradient).astype(np.float16)
        slopes = np.clip(slopes, -grad, grad)
    offsets = skt.resize(np.nan_to_num(offsets), fine_resolution, order=1, mode='edge', anti_aliasing=False)
    temp_arr = slopes * fine_dem + offsets
    del slopes, offsets
    temp_arr = dt.remove_residuals(temp_arr, temp_data, bias_thres, 10)
    temp_arr = (temp_arr * 10).astype(np.int16)
    temp_arr[np.isnan(land_sea_mask)] = -32767
    nc.write_to_netcdf(temp_arr, os.path.join(output_dir, f'ds_temp_{day}.nc'), extent=domain, compress=True, nodata_value=-32767) #type:ignore

def temperature_interpolation_height_regression(config_file, domain, output_dir):
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    fine_dem, _ = dt.load_specified_lines(config_file['files']['fine_dem'], domain, False)
    fine_resolution = dt.get_resolution_array(config_file, domain, True)
    if fine_resolution != fine_dem.shape:
        fine_dem, _ = dt.resize_array_interp(fine_dem, fine_resolution)
    window_size = (int(config_file['options']['downscaling_window_size']), int(config_file['options']['downscaling_window_size'])) # Size of Moving Window
    use_gradient = config_file['options']['downscaling_use_temperature_gradient']
    dryadiabatic_gradient = float(config_file['options']['downscaling_dryadiabatic_gradient']) # K/m
    saturation_adiabatic_gradient = float(config_file['options']['downscaling_saturation_adiabatic_gradient']) # K/m
    bias_thres = float(config_file['options']['downscaling_temperature_bias_threshold']) # K
    precip_day_sum_thres = float(config_file['options']['downscaling_precipitation_per_day_threshold']) * 10 # mm/day
    land_sea_mask, _ = dt.load_specified_lines(config_file['files']['land_sea_mask'], domain, False)
    land_sea_mask = np.asarray(land_sea_mask).astype(np.float16)
    land_sea_mask[land_sea_mask == 0] = np.nan
    if land_sea_mask.shape != fine_dem.shape:
        land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_dem.shape)
    if not os.path.exists(os.path.join(climate_data_dir, 'Temp_avg.tif')):
        temp_file = os.path.join(climate_data_dir, 'Temp_avg.nc')
    else:
        temp_file = os.path.join(climate_data_dir, 'Temp_avg.tif')
    coarse_dem = dt.resize_array_mean(fine_dem, read_timestep(temp_file, extent=domain, timestep=0)[0].shape)
    if temp_file.endswith('.nc'):
        temp_data, _ = dt.load_specified_lines(temp_file, extent=domain)
    else:
        with rasterio.open(temp_file, 'r') as src:
            temp_data = src.read()
            _ = src.nodata
            tbounds = src.bounds
        temp_data = dt.extract_domain_from_global_3draster(temp_data, [domain[1], domain[0], domain[3], domain[2]], [tbounds.left, tbounds.top, tbounds.right, tbounds.bottom])

    """
        ### DEBUG ###
    for day in range(365):
        height_regress_day(day, output_dir, temp_data[day], domain, window_size, coarse_dem, fine_dem, fine_resolution, use_gradient, precip_day_sum_thres, saturation_adiabatic_gradient, dryadiabatic_gradient, bias_thres, land_sea_mask)
    
    """

    area = int((domain[0] - domain[2]) * (domain[3] - domain[1]))
    worker = np.clip(int((dt.get_cpu_ram()[1] / area) * 500), 1, dt.get_cpu_ram()[0]-1)
    print(f'Using {worker} workers')

    with ProcessPoolExecutor(max_workers=worker) as executor:
        tasks = [executor.submit(height_regress_day, day, output_dir, temp_data[day], domain, window_size, coarse_dem, fine_dem, fine_resolution, use_gradient, precip_day_sum_thres, saturation_adiabatic_gradient, dryadiabatic_gradient, bias_thres, land_sea_mask) for day in range(365) if not os.path.exists(os.path.join(output_dir, f'ds_temp_{day}.nc'))]
        for future in tasks:
            future.result()
    return [os.path.join(output_dir, f'ds_temp_{day}.nc') for day in range(0, 365)]


### MAIN PROCESSING METHOD ###

def process_day_slice(args):
    current_data_slice, extent, worldclim_factors, fine_resolution, day_index, output_dir, func_type, _, landsea_mask = args
    output_file = os.path.join(output_dir, f'ds_{func_type}_{day_index}.nc')
    if os.path.exists(output_file):
        return
    if np.isnan(current_data_slice).sum() > 0:
        current_data_slice = dt.fill_nan_nearest(current_data_slice)
    data = skt.resize(current_data_slice, fine_resolution, order=1, mode='edge', anti_aliasing=False)
    if func_type == 'temp':
        data -= worldclim_factors
    elif func_type == 'prec':
        data *= worldclim_factors
    data = (data * 10).astype(np.int16)
    data[landsea_mask] = -32767
    nc.write_to_netcdf(data, output_file, extent=extent, compress=True, complevel=9, nodata_value=-32767)  # type: ignore

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def parallel_processing(current_data, extent, fine_resolution, output_dir, func_type, landsea_mask, world_clim_data_dir):
    cpu, ram = dt.get_cpu_ram()
    area = (extent[3] - extent[1]) * (extent[0] - extent[2])
    max_proc = np.clip(int(ram / ((area / 50000) * 7)), 1, cpu-1)
    print(f'\n -> Using {max_proc} cores')
    month_map = np.searchsorted(np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), np.arange(365))
    missing_days = [day for day in range(365) if not os.path.exists(os.path.join(output_dir, f'ds_{func_type}_{day}.nc'))]
    if not missing_days:
        return
    
    month, worldclim_factors, month_tasks = -1, None, []
    for day in missing_days:
        current_month = month_map[day]
        if month != current_month and month_tasks:
            try:
                print_progress_bar(month, 12, prefix='       Progress', suffix='Complete', length=50)
            except:
                pass
            if len(month_tasks) > 0:
                """
                    ### DEBUG ###

                for task in month_tasks:
                    process_day_slice(task)
                """
                with ProcessPoolExecutor(max_workers=max_proc) as executor:
                    executor.map(process_day_slice, month_tasks, chunksize=int(math.ceil(len(month_tasks) / max_proc)))
                
                month_tasks = []
        if month != current_month:
            month = current_month
            worldclim_factors = nc.read_area_from_netcdf(os.path.join(world_clim_data_dir, f'factors_month_{current_month+1}.nc'),extent=extent)[0].astype(np.float16)
            worldclim_factors[np.isnan(worldclim_factors)] = 0
            if worldclim_factors.shape != fine_resolution:
                worldclim_factors = dt.resize_array_mean(worldclim_factors, fine_resolution)
        month_tasks.append((current_data[day].astype(np.float16), extent, worldclim_factors, fine_resolution, day, output_dir, func_type, None, landsea_mask))
    if month_tasks:
        if len(month_tasks) > 0:
            with ProcessPoolExecutor(max_workers=max_proc) as executor:
                executor.map(process_day_slice, month_tasks, chunksize=int(math.ceil(len(month_tasks) / max_proc)))
    try:
        print_progress_bar(12, 12, prefix='       Progress', suffix='Complete', length=50)
        print('')
    except:
        pass


### PRECIPITATION ###

### MAIN PRECIPITATION METHOD ###


def interpolate_precipitation(config_file, domain, area_name):
    """
        domain: [y_max, x_min, y_min, x_max]
    """
    interpolation_method = int(config_file['options']['precipitation_downscaling_method'])
    output_dir = os.path.join(config_file['files']['output_dir']+'_downscaled', area_name)
    os.makedirs(output_dir, exist_ok=True)
    if interpolation_method == 0:
        prec_files = precipitation_interpolation_nearestneighbour(config_file, domain, output_dir)
    elif interpolation_method == 1:
        prec_files = precipitation_interpolation_bilinear(config_file, domain, output_dir)
    elif interpolation_method == 2:
        prec_files = worldclim_downscaling_prec(domain, config_file, output_dir)

    return prec_files, True

def worldclim_downscaling_prec(extent, config_file, output_dir):
    if all(j for j in [os.path.exists(os.path.join(output_dir, f'ds_prec_{i}.nc')) for i in range(0, 365)]) :
        return [os.path.join(output_dir, f'ds_prec_{day}.nc') for day in range(0, 365)]
    print('Downscaling')
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    world_clim_data_dir = os.path.join(config_file['files']['worldclim_precipitation_data_dir'])
    #fine_dem, _ = dt.load_specified_lines(config_file['files']['fine_dem'], extent, False)
    #fine_resolution = fine_dem.shape
    fine_resolution = dt.get_resolution_array(config_file, extent, True)
    land_sea_mask, _ = dt.load_specified_lines(config_file['files']['land_sea_mask'], extent, False)
    land_sea_mask = land_sea_mask != 0
    if land_sea_mask.shape != fine_resolution:
        land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_resolution)
    if not os.path.exists(os.path.join(climate_data_dir, 'Prec_avg.tif')): prec_file = os.path.join(climate_data_dir, 'Prec_avg.nc')
    else: prec_file = os.path.join(climate_data_dir, 'Prec_avg.tif')

    coarse_resolution = read_timestep(prec_file, extent=extent, timestep=0)[0].shape
    ext = nc.check_extent_load_file(os.path.join(world_clim_data_dir, f'factors_month_1.nc'), extent=extent)
    [os.remove(file) for i in range(1, 13) if not ext and (file := os.path.join(world_clim_data_dir, f'factors_month_{i}.nc')) and os.path.exists(file)]
    if not all(os.path.exists(os.path.join(world_clim_data_dir, f'factors_month_{band_index}.nc')) for band_index in range(1, 13)):
        world_clim_data_dir = pi.calculate_prec_factors(fine_resolution, coarse_resolution, world_clim_data_dir, extent)
    prec_data, _ = dt.load_specified_lines(prec_file, extent=extent)
    prec_thres = float(config_file['options'].get('downscaling_precipitation_per_day_threshold', 0.75))
    prec_data[prec_data < prec_thres] = 0
    parallel_processing(prec_data, extent, fine_resolution, output_dir, 'prec', np.isnan(land_sea_mask), world_clim_data_dir)
    return [os.path.join(output_dir, f'ds_prec_{day}.nc') for day in range(0, 365)]

def worldclim_downscaling_prec_singleslice(data_slice, extent, worldclim_factors, fine_resolution, day, output_dir):
    if os.path.exists(os.path.join(output_dir, f'ds_prec_{day}.nc')):
        return
    worldclim_factors[np.isnan(worldclim_factors)] = 0
    if worldclim_factors.shape != fine_resolution:
        worldclim_factors = dt.resize_array_mean(worldclim_factors, fine_resolution)
    prec_data = (skt.resize(data_slice * 3600 * 24, fine_resolution, order=1, mode='edge', anti_aliasing=False) * worldclim_factors).astype(np.float16)
    nc.write_to_netcdf(prec_data, os.path.join(output_dir, f'ds_prec_{day}.nc'), extent=extent, compress=True, complevel=9) #type:ignore

def process_precday_interp(day, prec_data, extent, prec_thres, fine_dem_shape, land_sea_mask, output_dir, prec_nodata, mode='nearest'):
    if os.path.exists(os.path.join(output_dir, f'ds_prec_{day}.tif')) or os.path.exists(os.path.join(output_dir, f'ds_prec_{day}.nc')):
        return f"Day {day + 1} skipped (already processed)."
    sys.stdout.write(f'     - Downscaling of precipitation data for day #{day + 1}                      ' + '\r')
    sys.stdout.flush()
    prec_data[prec_data < prec_thres] = 0
    prec_data[(prec_data < 0) | (prec_data == prec_nodata)] = np.nan
    prec_data *= 10
    if mode == 'bilinear':
        prec_data = skt.resize(prec_data, fine_dem_shape, order=1, mode='edge', anti_aliasing=False).astype(np.int16)
    elif mode == 'nearest':
        prec_data = (np.repeat(np.repeat(prec_data, int(fine_dem_shape[0] / prec_data.shape[0]), axis=0), int(fine_dem_shape[0] / prec_data.shape[0]), axis=1)).astype(np.int16)
    prec_data[np.isnan(land_sea_mask)] = -32767
    nc.write_to_netcdf(prec_data, os.path.join(output_dir, f'ds_prec_{day}.nc'), extent=extent, compress=True, nodata_value=-32767) #type:ignore
    return ''

def precipitation_interpolation_bilinear(config_file, extent, output_dir):
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    fine_resolution = dt.get_resolution_array(config_file, extent, True)
    land_sea_mask, _ = dt.load_specified_lines(config_file['files']['land_sea_mask'], extent, False)
    land_sea_mask = np.asarray(land_sea_mask).astype(np.float16)
    land_sea_mask[land_sea_mask == 0] = np.nan
    if land_sea_mask.shape != fine_resolution:
        land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_resolution)
    if not os.path.exists(os.path.join(climate_data_dir, 'Prec_avg.tif')):
        prec_file = os.path.join(climate_data_dir, 'Prec_avg.nc')
    else: prec_file = os.path.join(climate_data_dir, 'Prec_avg.tif')
    prec_thres = float(config_file['options'].get('downscaling_precipitation_per_day_threshold', 0.75))

    if prec_file.endswith('.nc'):
        prec_data, nodata = dt.load_specified_lines(prec_file, extent=extent)
    else:
        with rasterio.open(prec_file, 'r') as src:
            prec_data = src.read()
            nodata = src.nodata
            pbounds = src.bounds
        prec_data = dt.extract_domain_from_global_3draster(prec_data, [extent[1], extent[0], extent[3], extent[2]], [pbounds.left, pbounds.top, pbounds.right, pbounds.bottom])

    """
        ### DEBUG ###
    for day in range(365):
        process_precday_interp(day, prec_data[day], extent, prec_thres, fine_resolution, land_sea_mask, output_dir, nodata, 'bilinear')
    """

    area = int((extent[0] - extent[2]) * (extent[3] - extent[1]))
    worker = np.clip(int((dt.get_cpu_ram()[1] / area) * 1200), 1, dt.get_cpu_ram()[0]-1)
    print(f'Using {worker} workers')
    with ProcessPoolExecutor(max_workers=worker) as executor:
        tasks = [executor.submit(process_precday_interp, day, prec_data[day], extent, prec_thres, fine_resolution, land_sea_mask, output_dir, nodata, 'bilinear') for day in range(365) if not os.path.exists(os.path.join(output_dir, f'ds_prec_{day}.nc'))]
        for future in tasks:
            future.result()
    
    return [os.path.join(output_dir, f'ds_prec_{day}.nc') for day in range(0, 365)]

def precipitation_interpolation_nearestneighbour(config_file, extent, output_dir):
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    fine_resolution = dt.get_resolution_array(config_file, extent, True)
    land_sea_mask, _ = dt.load_specified_lines(config_file['files']['land_sea_mask'], extent, False)
    land_sea_mask = np.asarray(land_sea_mask).astype(np.float16)
    land_sea_mask[land_sea_mask == 0] = np.nan
    if land_sea_mask.shape[0] != fine_resolution[0]:
        land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_resolution)
    if not os.path.exists(os.path.join(climate_data_dir, 'Prec_avg.tif')):
        prec_file = os.path.join(climate_data_dir, 'Prec_avg.nc')
    else: prec_file = os.path.join(climate_data_dir, 'Prec_avg.tif')
    prec_thres = float(config_file['options'].get('downscaling_precipitation_per_day_threshold', 0.75))

    if prec_file.endswith('.nc'):
        prec_data, nodata = dt.load_specified_lines(prec_file, extent=extent)
    else:
        with rasterio.open(prec_file, 'r') as src:
            prec_data = src.read()
            nodata = src.nodata
            pbounds = src.bounds
        prec_data = dt.extract_domain_from_global_3draster(prec_data, [extent[1], extent[0], extent[3], extent[2]], [pbounds.left, pbounds.top, pbounds.right, pbounds.bottom])

    """
     # DEBUG
    for day in range(365):
        process_precday_interp(day, prec_data[day], extent, prec_thres, fine_resolution, land_sea_mask, output_dir, nodata)
    """
    area = int((extent[0] - extent[2]) * (extent[3] - extent[1]))
    worker = np.clip(int((dt.get_cpu_ram()[1] / area) * 600), 1, dt.get_cpu_ram()[0]-1)
    with ProcessPoolExecutor(max_workers=worker) as executor:
        tasks = [executor.submit(process_precday_interp, day, prec_data[day], extent, prec_thres, fine_resolution, land_sea_mask, output_dir, nodata, mode='nearest') for day in range(365)]
        for future in tasks:
            print(future.result())    
    return [os.path.join(output_dir, f'ds_prec_{day}.nc') for day in range(0, 365)]

def process_rrpcf_day(day, data, fine_resolution, landsea_mask, output_dir, crop, water, extent, method='linear'):
    if not data.shape == fine_resolution:
        order = 0 if method == 'nearest' else 1 if method == 'linear' else 3 if method == 'cubic' else 1
        dayslice = skt.resize(data, fine_resolution, order=order, mode='edge', anti_aliasing=False)
        dayslice[np.isnan(landsea_mask)] = -32767
        nc.write_to_netcdf(dayslice.astype(np.int16), os.path.join(output_dir, f'ds_rrpcf_{crop}_{water}_{day}.nc'), extent=extent, compress=True, complevel=9, nodata_value=-32767) #type:ignore
    else:
        data[np.isnan(landsea_mask)] = -32767
        nc.write_to_netcdf(data.astype(np.int16), os.path.join(output_dir, f'ds_rrpcf_{crop}_{water}_{day}.nc'), extent=extent, compress=True, complevel=9, nodata_value=-32767) #type:ignore

def interpolate_rrpcf(config_file, extent, area_name, crops):
    output_dir = os.path.join(config_file['files']['output_dir']+'_downscaled', area_name)
    os.makedirs(output_dir, exist_ok=True)
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    #fine_dem, _ = dt.load_specified_lines(config_file['files']['fine_dem'], extent, False)
    #fine_resolution = fine_dem.shape
    fine_resolution = dt.get_resolution_array(config_file, extent, True)
    landsea_mask, _ = dt.load_specified_lines(config_file['files']['land_sea_mask'], extent, False)
    landsea_mask = np.asarray(landsea_mask).astype(np.float16)
    landsea_mask[landsea_mask == 0] = np.nan
    if landsea_mask.shape[0] != fine_resolution[0]:
        landsea_mask = dt.interpolate_nanmask(landsea_mask, fine_resolution)
    method = config_file['options'].get('rrpcf_interpolation_method', 'linear')

    for crop in crops:
        crop = os.path.splitext(crop)[0].lower()
        water = 'ir' if bool(int(config_file['options'].get('irrigation', False))) else 'rf'
        if not os.path.exists(os.path.join(climate_data_dir, f'rrpcf_{crop}_{water}.tif')):
            rrpcf_file = os.path.join(climate_data_dir, f'rrpcf_{crop}_{water}.nc')
        else:
            rrpcf_file = os.path.join(climate_data_dir, f'rrpcf_{crop}_{water}.tif')
        if not os.path.exists(rrpcf_file):
            continue

        print(f' -> Downscaling RRPCF data for {crop} under {water} conditions')

        if os.path.splitext(rrpcf_file)[1] == '.nc':
            data, _ = dt.load_specified_lines(rrpcf_file, extent, True)
        else:
            with rasterio.open(rrpcf_file, 'r') as src:
                data = src.read()
                bounds = src.bounds
                count = src.count
            if count == 1:
                data = dt.extract_domain_from_global_raster(data, [extent[1], extent[0], extent[3], extent[2]], [bounds.left, bounds.top, bounds.right, bounds.bottom])
            else:
                data = dt.extract_domain_from_global_3draster(data, [extent[1], extent[0], extent[3], extent[2]], [bounds.left, bounds.top, bounds.right, bounds.bottom])
            data = np.asarray(data, dtype=np.int16)
            data = data.transpose(1, 2, 0)
        
        count = len(data.shape)
        if count == 2:
            if os.path.exists(os.path.join(output_dir, f'ds_rrpcf_{crop}_{water}_0.nc')):
                continue
            if not data.shape == fine_resolution:
                order = 0 if method == 'nearest' else 1 if method == 'linear' else 3 if method == 'cubic' else 1
                dayslice = skt.resize(data, fine_resolution, order=order, mode='edge', anti_aliasing=False)
                dayslice[np.isnan(landsea_mask)] = -32767
                nc.write_to_netcdf(dayslice.astype(np.int16), os.path.join(output_dir, f'ds_rrpcf_{crop}_{water}_0.nc'), extent=extent, compress=True, complevel=9, nodata_value=-32767) #type:ignore
            else:
                data[np.isnan(landsea_mask)] = -32767
                nc.write_to_netcdf(data.astype(np.int16), os.path.join(output_dir, f'ds_rrpcf_{crop}_{water}_0.nc'), extent=extent, compress=True, complevel=9, nodata_value=-32767) #type:ignore
        else:
            if data.shape[:2] == fine_resolution:
                data[np.isnan(landsea_mask)] = -32767
                for day in range(data.shape[-1]):
                    if os.path.exists(os.path.join(output_dir, f'ds_rrpcf_{crop}_{water}_{day}.nc')):
                        continue
                    nc.write_to_netcdf(data[..., day].astype(np.int16), os.path.join(output_dir, f'ds_rrpcf_{crop}_{water}_{day}.nc'), extent=extent, compress=True, complevel=9, nodata_value=-32767) #type:ignore
            else:
                area = float((extent[0] - extent[2]) * (extent[3] - extent[1]))
                resolution_factor = {5: 1, 6: 0.25, 4: 5, 3: 10, 2: 12, 1: 30, 0: 60}
                resfact = resolution_factor.get(int(config_file['options'].get('resolution', 5)), 1)
                worker = np.clip(int((dt.get_cpu_ram()[1] / area) * 700 * resfact * resfact), 1, dt.get_cpu_ram()[0]-1)
                print(f'Using {worker} workers')

                """
                    ### DEBUG ###
                for day in range(365):
                    process_rrpcf_day(day, data[..., day], fine_resolution, landsea_mask, output_dir, crop, water, extent, method)
                """

                with ThreadPoolExecutor(max_workers=worker) as executor:
                    tasks = [executor.submit(process_rrpcf_day, day, data[..., day], fine_resolution, landsea_mask, output_dir, crop, water, extent, method) for day in range(365) if not os.path.exists(os.path.join(output_dir, f'ds_rrpcf_{crop}_{water}_{day}.nc'))]
                    for future in tasks:
                        future.result()
                
    return output_dir, True