import os
import numpy as np
import rasterio
import gc
import sys
from scipy.stats import linregress
from scipy.interpolate import NearestNDInterpolator
try:
    import data_tools as dt
    import nc_tools as nc
except:
    from src import data_tools as dt
    from src import nc_tools as nc
import concurrent.futures
import math
import warnings
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
warnings.filterwarnings("ignore")


def read_tif(fn):
    with rasterio.open(fn, 'r') as src:
        nodata = src.nodata
        data = src.read()
        bounds = src.bounds
    data = np.squeeze(data)
    data[data == nodata] = np.nan
    return data, bounds


def correct_unvalids(args):
    data, day = args
    mask = (data < -450) | (data > 450)
    if not np.any(mask):
        return data
    valid_coords = np.array(np.where(~mask)).T
    valid_values = data[~mask]
    interp = NearestNDInterpolator(valid_coords, valid_values)
    all_coords = np.indices(data.shape).reshape(data.ndim, -1).T
    data = interp(all_coords).reshape(data.shape)
    np.save(os.path.join(os.getcwd(), 'temp', f'{day}_corr.npy'), data)


def interpolate_temperature(config_file, coarse_dem, fine_dem, temp_data, domain, land_sea_mask, prec_data, full_domain=[0, 0, 0, 0]):
    """
        domain: [y_max, x_min, y_min, x_max]
    """
    interpolation_method = int(config_file['options']['temperature_downscaling_method'])
    try:
        world_clim_data_dir = config_file['files']['worldclim_temperature_data_dir']
    except:
        world_clim_data_dir = None

    if interpolation_method == 0:
        temp_data = temperature_interpolation_nearestneighbour(coarse_dem, fine_dem, temp_data)
    elif interpolation_method == 1:
        temp_data = temperature_interpolation_bilinear(fine_dem, temp_data)

    elif interpolation_method == 2:
        return
        fine_dem_full, _ = dt.load_specified_lines(config_file['files']['fine_dem'], full_domain, False)
        out_res = (abs(float(full_domain[1])) + abs(float(full_domain[3]))) / temp_data.shape[2]
        factors_file = calculate_temp_factors(fine_dem_full.shape, (int(np.round(temp_data.shape[2] * out_res)), np.shape(temp_data)[2]),
                                              world_clim_data_dir, full_domain)
        del fine_dem_full
        temp_data = temperature_interpolation_worldclim(factors_file, fine_dem.shape, temp_data, domain)

    elif interpolation_method == 3:
        temp_data = temperature_interpolation_height_regression(config_file, fine_dem, prec_data, coarse_dem, temp_data)
    
    temp_data[(temp_data < -500) | (temp_data > 500)] = -32767
    temp_data[np.isnan(land_sea_mask)] = -32767
    
    return temp_data.astype(np.int16)


def temperature_interpolation_nearestneighbour(coarse_dem, fine_dem, temp_data):
    ratio = fine_dem.shape[0] // coarse_dem.shape[0]
    temp_data = (temp_data * 10).astype(np.int16)
    new_temp = np.empty((fine_dem.shape + (365,)), dtype=temp_data.dtype)
    for day in range(temp_data.shape[0]):
        sys.stdout.write(f'     - Downscaling of temperature data for day #{day+1}                      '+'\r')
        sys.stdout.flush()
        new_temp[..., day] = np.repeat(np.repeat(temp_data[day], ratio, axis=0), ratio, axis=1)
    sys.stdout.write(f'   -> Downscaling of temperature data completed successfully                       '+'\r')
    sys.stdout.flush()
    return new_temp


def temperature_interpolation_bilinear(fine_dem, temp_data):
    temp_data = (temp_data * 10).astype(np.int16)
    new_temp = np.empty((fine_dem.shape + (365,)), dtype=temp_data.dtype)
    for day in range(temp_data.shape[0]):
        sys.stdout.write(f'     - Downscaling of temperature data for day #{day+1}                      '+'\r')
        sys.stdout.flush()
        new_temp[..., day] = (dt.resize_array_interp(temp_data[day], fine_dem.shape)[0]).astype(np.int16)
    sys.stdout.write(f'   -> Downscaling of temperature data completed successfully                       '+'\r')
    sys.stdout.flush()
    return new_temp  


def temperature_interpolation_height_regression(config_file, fine_dem, precipitation_files, coarse_dem, temp_data, domain, prec_dailyfiles=False):
    window_size = (int(config_file['options']['downscaling_window_size']), int(config_file['options']['downscaling_window_size'])) # Size of Moving Window
    use_gradient = config_file['options']['downscaling_use_temperature_gradient']
    dryadiabatic_gradient = float(config_file['options']['downscaling_dryadiabatic_gradient']) # K/m
    saturation_adiabatic_gradient = float(config_file['options']['downscaling_saturation_adiabatic_gradient']) # K/m
    bias_thres = float(config_file['options']['downscaling_temperature_bias_threshold']) # K
    precip_day_sum_thres = float(config_file['options']['downscaling_precipitation_per_day_threshold']) # mm/day

    temp_dir = os.path.join(os.getcwd(), 'temp')
    print(f'Temp Dir: {temp_dir}')
    [os.remove(os.path.join(temp_dir, fn)) for fn in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, fn))]

    precipitation = nc.read_area_from_netcdf_list(precipitation_files, overlap=False, extent=domain, dayslices=prec_dailyfiles)

    arguments = [(day, fine_dem, precipitation[..., day], precip_day_sum_thres, coarse_dem, window_size, temp_data[day],
                    use_gradient, saturation_adiabatic_gradient, dryadiabatic_gradient, bias_thres) for day in range(365)
              if not os.path.exists(os.path.join(os.getcwd(), 'temp', f'{day}.npy'))]
    
    no_threads, _ = dt.get_cpu_ram()
    max_proc = np.clip(int(no_threads-1), 1, no_threads)
    print(f'Limiting to {max_proc} cores')
    print(f'{len(arguments)} days have to be processed.')
    print('Allocating Threads...')

    """
    # DEBUG
    for argument in arguments:
        calculate_height_regression(argument)
    """

    if len(arguments) != 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_proc) as executor:
            executor.map(calculate_height_regression, arguments, chunksize=math.ceil(len(arguments) / max_proc))
    
    sys.stdout.write(f'Writing results to array                                           '+'\r')
    sys.stdout.flush()
    new_temp = np.empty((fine_dem.shape + (365,)), dtype=temp_data.dtype)
    for day in range(365):
        sys.stdout.write(f'     - reading {day}.npy                      '+'\r')
        sys.stdout.flush()
        new_temp[..., day] = np.load(os.path.join(os.getcwd(), 'temp', f'{day}.npy'))
    
    del temp_data, precipitation
    for i in range(365):
        os.remove(os.path.join(os.getcwd(), 'temp', f'{i}.npy'))
    return new_temp.astype(np.int16)
    

def calculate_height_regression(args):
    day, fine_dem, day_precip, precip_day_sum_thres, coarse_dem, window_size, day_temp, use_gradient, saturation_adiabatic_gradient, dryadiabatic_gradient, bias_thres = args
    """
    Process a day's climate data, interpolating precipitation and downscaling temperature.
    Parameters:
    - args (tuple): Tuple containing various data elements required for processing a day's data.
    Elements in the tuple include:
        - day (int): Day number.
        - day_precip (np.ndarray): Array containing precipitation data for day x.
        - day_temp (np.ndarray): Array containing temperature data for day x.
        - coarse_dem (numpy.ndarray): Coarse resolution digital elevation model.
        - fine_dem (numpy.ndarray): Fine resolution digital elevation model.
        - land_sea_mask (numpy.ndarray): Mask representing land and sea areas.
        - window_size (tuple): Size of the moving window for analysis.
        - precip_day_sum_thres (float): Threshold value for precipitation day sum.
        - precip_thres (float): Precipitation threshold.
        - use_gradient (bool): Flag indicating whether to use gradient in temperature downscaling.
        - saturation_adiabatic_gradient (float): Gradient value for saturation adiabatic temperature.
        - dryadiabatic_gradient (float): Gradient value for dry adiabatic temperature.
        - bias_thres (float): Threshold for temperature downscaled bias.
    This function processes climate data for a particular day, including interpolating precipitation data,
    downscaling temperature data, and handling various conditions for calculating temperature.
    """
    sys.stdout.write(f' -> Processing day #{day}                                       '+'\r')
    sys.stdout.flush()
    temp_arr = np.zeros((np.shape(fine_dem)[0], np.shape(fine_dem)[1]), dtype=np.float16)

    if window_size[0] >= day_temp.shape[0] or window_size[1] >= day_temp.shape[1]:
        window_size = (day_temp.shape[0] - 1, day_temp.shape[1] - 1)

    ### Temperature Downscaling ###
    dem_window = np.lib.stride_tricks.sliding_window_view(coarse_dem, window_size)
    temp_window = np.lib.stride_tricks.sliding_window_view(day_temp, window_size)
    slopes = np.zeros_like(day_temp)
    offsets = np.zeros_like(day_temp)

    # Calculate half window sizes
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

    # Bilinear interpolation of Slopes and Offsets, check for physical boundaries
    slopes, _ = dt.resize_array_interp(np.nan_to_num(slopes), fine_dem.shape)
    if use_gradient:
        grad = np.where(day_precip >= precip_day_sum_thres, saturation_adiabatic_gradient, dryadiabatic_gradient).astype(np.float16)
        slopes = np.clip(slopes, -grad, grad)
    offsets, _ = dt.resize_array_interp(np.nan_to_num(offsets), fine_dem.shape)

    # Calculate downscaled temperature
    temp_arr = slopes * fine_dem + offsets
    del slopes, offsets
    # Check for Residues and correct downscaled temperature
    temp_arr = dt.remove_residuals(temp_arr, day_temp, bias_thres, 10)
    temp_arr = (temp_arr * 10).astype(np.int16)
    np.save(os.path.join(os.getcwd(), 'temp', f'{day}.npy'), temp_arr)


def calculate_temp_factors(fine_dem_shape, coarse_dem_shape, world_clim_data_dir, full_domain):
    for idx, fn in enumerate(sorted([fname for fname in os.listdir(world_clim_data_dir) if fname.endswith('.tif') and not fname.startswith('factors')])):
        if os.path.exists(os.path.join(world_clim_data_dir, f'factors_month_{idx+1}.tif')):
            continue
        sys.stdout.write(f'     - Reading WorldClim data for month #{idx+1}                      '+'\r')
        sys.stdout.flush()

        world_clim_data, nan_value = dt.load_specified_lines(os.path.join(world_clim_data_dir, fn), full_domain, False)
        world_clim_data[world_clim_data == nan_value] = np.nan
        nan_mask = np.isnan(world_clim_data)
        world_clim_data = world_clim_data.astype(np.float16)
        world_clim_data = dt.fill_nan_nearest(world_clim_data)
        world_clim_data_coarse = dt.resize_array_mean(world_clim_data, coarse_dem_shape).astype(np.float16)
        world_clim_data_coarse = dt.resize_array_interp(world_clim_data_coarse, world_clim_data.shape)[0].astype(np.float16) #type:ignore

        # Calculate Factor
        if world_clim_data.shape[0] != fine_dem_shape[0]: #type:ignore
            dat, _ = dt.resize_array_interp(world_clim_data_coarse - world_clim_data, (fine_dem_shape[0], fine_dem_shape[1]))
            dat = dat.astype(np.float16)
        else:
            dat = (world_clim_data_coarse - world_clim_data).astype(np.float16)
        dat[nan_mask] = np.nan
        nc.write_to_netcdf(dat, os.path.join(world_clim_data_dir, f'factors_month_{idx+1}.nc'), extent=full_domain, compress=True, complevel=5, nodata_value=np.nan) #type:ignore
        #dt.write_geotiff(world_clim_data_dir, f'factors_month_{idx+1}.tif', dat, full_domain, nodata_value=np.nan, inhibit_message=True)
        del world_clim_data, world_clim_data_coarse, dat

    sys.stdout.write(f'   -> WorldClim data read successfully                       '+'\r')
    sys.stdout.flush()

    # Write to NetCDF
    #factors_file = dt.merge_geotiffs_to_multiband([os.path.join(world_clim_data_dir, f'factors_month_{idx+1}.tif') for idx in range(12)], factors_file)
    return world_clim_data_dir


def temperature_interpolation_worldclim(factors_file, fine_dem_shape, temp_data, domain, bias_thres = 0.0001):
    covered = dt.check_geotiff_extent(factors_file, domain)
    if covered:
        factors, _ = dt.load_specified_lines(factors_file, domain)
    factors[np.isnan(factors)] = 0
    nan_mask = np.isnan(temp_data)
    temp_data = (temp_data * 10).astype(np.int16)
    temp_data[nan_mask] = -32767
    del nan_mask
    new_temp = np.empty((fine_dem_shape + (365,)), dtype=temp_data.dtype)

    for day in range(365):
        sys.stdout.write(f'     - Downscaling of temperature data for day #{day+1}                      '+'\r')
        sys.stdout.flush()
        new_temp[..., day] = (dt.resize_array_interp(temp_data[day], fine_dem_shape))[0] - ((factors[int(day // (365 / 12))])*10) #type:ignore
    print('\n   ->  Correcting Data for Energy Conservation')

    del factors, temp_data
    gc.collect()

    return new_temp