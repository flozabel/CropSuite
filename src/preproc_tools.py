import os
import sys
try:
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'src'))
except Exception as e:
    print(f"Failed to modify system path: {e}")

import numpy as np
import xarray as xr
#from datetime import datetime
try:
    from src import data_tools as dt
    from src import read_climate_ini as rci
    from src import nc_tools as nc
    from src import read_plant_params as rpp
    #from src import downscaling as downs
    from src import temp_interpolation as ti
    from src import prec_interpolation as pi
    import data_tools as dt
    import read_climate_ini as rci
    import nc_tools as nc
    import read_plant_params as rpp
    #import downscaling as downs
    import temp_interpolation as ti
    import prec_interpolation as pi
except:
    import data_tools as dt
    import read_climate_ini as rci
    import nc_tools as nc
    
    import read_plant_params as rpp
    #import downscaling as downs
    import temp_interpolation as ti
    import prec_interpolation as pi
from scipy.integrate import cumulative_trapezoid
#from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ProcessPoolExecutor
import math
import netCDF4 as nc4
from skimage import transform as skt
from datetime import datetime

def get_time_range(nc_files):
    min_time = None
    max_time = None
    for file in nc_files:
        ds = xr.open_dataset(file)
        if 'time' in ds.coords:
            file_min_time = ds['time'].min().values
            file_max_time = ds['time'].max().values
            if min_time is None or file_min_time < min_time:
                min_time = file_min_time
            if max_time is None or file_max_time > max_time:
                max_time = file_max_time
        ds.close()
    min_year = np.datetime_as_string(min_time, unit='Y') #type:ignore
    max_year = np.datetime_as_string(max_time, unit='Y') #type:ignore
    return int(min_year), int(max_year)

def get_available_parameters(nc_files):
    params = list({var for file in nc_files for var in xr.open_dataset(file).data_vars.keys()})
    if params == ['tasmax', 'tasmin'] or params == ['tmax', 'tmin'] or params == ['tasmin', 'tasmax'] or params == ['tmin', 'tmax']:
        params = ['tas']
    return params

def calculate_daily_temp_mean(temp_files, out_file, start_year, end_year, extent = [0, 0, 0, 0]):
    print('    -> Reading temperature input files')

    params = list({var for file in temp_files for var in xr.open_dataset(file).data_vars.keys()})
    ds_list = [xr.open_dataset(nc) for nc in temp_files]
    if params == ['tasmax', 'tasmin'] or params == ['tmax', 'tmin'] or params == ['tasmin', 'tasmax'] or params == ['tmin', 'tmax']:
        print("    -> Calculating mean between tasmin and tasmax")
        if len(ds_list) != 2:
            raise ValueError("Expected exactly two NetCDF files for 'tasmin' and 'tasmax'.")
        ds_list = [((ds_list[0][list(ds_list[0].data_vars.keys())[0]] + ds_list[1][list(ds_list[1].data_vars.keys())[0]]) / 2).to_dataset(name="tas")]

    new_ds = []
    for ds in ds_list:
        if extent == [0, 0, 0, 0]:
            ds = ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))
        else:
            lat_slc = slice(float(extent.get('top')), float(extent.get('bottom'))) if ds.lat[0] > ds.lat[-1] else slice(float(extent.get('bottom')), float(extent.get('top')))
            lon_slc = slice(float(extent.get('left')), float(extent.get('right'))) if ds.lon[0] < ds.lon[-1] else slice(float(extent.get('right')), float(extent.get('left')))
            ds = ds.sel(time=~((ds['time.month'] == 2) & (ds['time.day'] == 29)))
            ds = ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'), lat=lat_slc, lon=lon_slc)
        if ds.dims['time'] > 0 and ds.dims['lat'] > 0 and ds.dims['lon'] > 0:
            new_ds.append(ds)
    print('    -> Selecting and concatenating required data')
    ds = new_ds[0] if len(new_ds) == 1 else xr.concat(new_ds, dim='time')
    varname = list(ds.data_vars)[0]
    print('    -> Calculating daily average temperature')
    ds = ds.groupby('time.dayofyear').mean('time')
    ds = ds.sel(dayofyear=slice(1, 365))
    if ds[varname].mean() > 100:
        ds[varname] -= 273.15
    encoding = {varname: {'zlib': True, 'complevel': 9, 'shuffle': True}}
    print('    -> Writing new NetCDF file')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    ds.to_netcdf(out_file, encoding=encoding)
    print('')

def calculate_daily_prec_sum(prec_files, out_file, start_year, end_year, extent = [0, 0, 0, 0]):
    print('    -> Reading precipitation input files')
    ds_list = [xr.open_dataset(nc) for nc in prec_files]
    new_ds = []
    for ds in ds_list:
        if extent == [0, 0, 0, 0]:
            ds = ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))
        else:
            lat_slc = slice(float(extent.get('top')), float(extent.get('bottom'))) if ds.lat[0] > ds.lat[-1] else slice(float(extent.get('bottom')), float(extent.get('top')))
            lon_slc = slice(float(extent.get('left')), float(extent.get('right'))) if ds.lon[0] < ds.lon[-1] else slice(float(extent.get('right')), float(extent.get('left')))
            ds = ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'), lat=lat_slc, lon=lon_slc)
        if ds.dims['time'] > 0 and ds.dims['lat'] > 0 and ds.dims['lon'] > 0:
            new_ds.append(ds)
    print('    -> Selecting and concatenating required data')
    ds = new_ds[0] if len(new_ds) == 1 else xr.concat(new_ds, dim='time')
    varname = list(ds.data_vars)[0]
    ds = ds.chunk({'time': 183})

    leap_idx = []
    for idx, tv in enumerate(list(ds['time'].values)):
        dt = tv.astype('M8[D]').astype(object)
        if dt.month == 2 and dt.day == 29:
            leap_idx.append(idx)
    ds = ds.isel(time=~np.isin(np.arange(ds.sizes['time']), np.asarray(leap_idx)))
    print('    -> Calculating daily average precipitation (optimized)')
    ds = ds.groupby('time.dayofyear').mean(dim='time', skipna=True)
    ds = ds.sel(dayofyear=slice(1, 365))
    if ds[varname].attrs.get('units', 'Units not found') != 'mm':
        print('    -> Converting units from non-mm to mm')
        ds[varname] = ds[varname] * 3600 * 24

    """
    leap_idx = []
    for idx, tv in enumerate(list(ds['time'].values)):
        dt = tv.astype('M8[D]').astype(object)
        if dt.month == 2 and dt.day == 29:
            leap_idx.append(idx)
    #ds = ds[varname].isel(time=[i for i in range(ds[varname].shape[0]) if i not in leap_idx])
    ds = ds.isel(time=[i for i in range(ds[varname].shape[0]) if i not in leap_idx])
    
    print('    -> Calculating daily average precipitation')
    #ds = ds.groupby('time.dayofyear').mean('time')
    #ds = ds.sel(dayofyear=slice(1, 365))
    ds = ds.chunk({'time': 183})
    ds = ds.resample(time='D').mean() 
    if ds[varname].attrs.get('units', 'Units not found') != 'mm':
        ds[varname] = ds[varname] * 3600 * 24
    """
    encoding = {varname: {'zlib': True, 'complevel': 9, 'shuffle': True}}

    print('    -> Writing new NetCDF file')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    ds.to_netcdf(out_file, encoding=encoding)
    print('')

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()


def process_slice(args):
    current_data_slice, extent, worldclim_factors, fine_resolution, day_index, output_dir, func_type, resampling_grid, landsea_mask, convert_units = args
    output_file = os.path.join(output_dir, f'ds_{func_type}_{day_index}.nc')
    if os.path.exists(output_file):
        return
    nan_usage = np.sum(np.isnan(current_data_slice)) > 0
    if nan_usage:
        current_data_slice = dt.fill_nan_nearest(current_data_slice)
    data = skt.resize(current_data_slice, fine_resolution, order=1, mode='edge', anti_aliasing=False)
    if convert_units:
        data = data - 273.15 if func_type == 'temp' else data * 3600 * 24
    if func_type == 'temp':
        data -= worldclim_factors
    elif func_type == 'prec':
        data *= worldclim_factors
    data[landsea_mask] = np.nan
    nc.write_to_netcdf(data, output_file, extent=extent, compress=True, complevel=9, nodata_value=np.nan)  # type: ignore


def parallel_processing(current_data, extent, worldclim_factors, fine_resolution, year, max_proc, output_dir, func_type, landsea_mask, world_clim_data_dir, convert_units):
    cpu, ram = dt.get_cpu_ram()
    area = (float(extent['top']) - float(extent['bottom'])) * (float(extent['right']) - float(extent['left']))
    max_proc = np.clip(int(ram / ((area / 50000) * 10)), 1, cpu-1)

    month_map = np.searchsorted(np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), np.arange(365))
    missing_days = [day for day in range(365) if not os.path.exists(os.path.join(output_dir, f'ds_{func_type}_{(year * 365) + day}.nc'))]
    if not missing_days:
        return
    
    month, worldclim_factors, month_tasks = -1, None, []
    for day in missing_days:
        current_month = month_map[day]
        if month != current_month and month_tasks:
            if len(month_tasks) > 0:
                """
                    ### DEBUG ###

                for task in month_tasks:
                    process_slice(task)
                """
                with ProcessPoolExecutor(max_workers=max_proc) as executor:
                    executor.map(process_slice, month_tasks, chunksize=int(math.ceil(len(month_tasks) / max_proc)))
                
                month_tasks = []
        if month != current_month:
            month = current_month
            worldclim_factors = nc.read_area_from_netcdf(os.path.join(world_clim_data_dir, f'factors_month_{current_month+1}.nc'),extent=extent)[0].astype(np.float16)
            worldclim_factors[np.isnan(worldclim_factors)] = 0
        month_tasks.append((current_data[day].astype(np.float16), extent, worldclim_factors, fine_resolution, year*365+day, output_dir, func_type, None, landsea_mask, convert_units))
    if month_tasks:
        if len(month_tasks) > 0:
            with ProcessPoolExecutor(max_workers=max_proc) as executor:
                executor.map(process_slice, month_tasks, chunksize=int(math.ceil(len(month_tasks) / max_proc)))


def get_nc_data(file_list, start_year, end_year, extent = [0, 0, 0, 0], downscaling=False, config={}, mode='temp', convert_units=True):
    print('    -> Reading NetCDF input files')

    params = list({var for file in file_list for var in xr.open_dataset(file).data_vars.keys()})
    ds_list = [xr.open_dataset(nc) for nc in file_list]
    varname = list(ds_list[0].data_vars)[0]
    unit = ds_list[0][varname].attrs.get('units', 'No units attribute found')
    convert_units = False if unit == 'Celcius degrees' or unit == 'mm' else True

    if params == ['tasmax', 'tasmin'] or params == ['tmax', 'tmin'] or params == ['tasmin', 'tasmax'] or params == ['tmin', 'tmax']:
        print("    -> Calculating mean between tasmin and tasmax")
        if len(ds_list) != 2:
            raise ValueError("Expected exactly two NetCDF files for 'tasmin' and 'tasmax'.")
        ds_list = [((ds_list[0][list(ds_list[0].data_vars.keys())[0]] + ds_list[1][list(ds_list[1].data_vars.keys())[0]]) / 2).to_dataset(name="tas")]

    new_ds = []
    for ds in ds_list:
        if extent == [0, 0, 0, 0]:
            ds = ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))
        else:
            lat_slc = slice(float(extent.get('top')), float(extent.get('bottom'))) if ds.lat[0] > ds.lat[-1] else slice(float(extent.get('bottom')), float(extent.get('top')))
            lon_slc = slice(float(extent.get('left')), float(extent.get('right'))) if ds.lon[0] < ds.lon[-1] else slice(float(extent.get('right')), float(extent.get('left')))
            ds = ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'), lat=lat_slc, lon=lon_slc)
        if ds.dims['time'] > 0 and ds.dims['lat'] > 0 and ds.dims['lon'] > 0:
            new_ds.append(ds)
    print('    -> Selecting and concatenating required data')
    ds = new_ds[0] if len(new_ds) == 1 else xr.concat(new_ds, dim='time')
    if varname == 'tasmax' or varname == 'tasmin':
        varname = 'tas'

    leap_idx = []
    for idx, tv in enumerate(list(ds['time'].values)):
        dateobj = tv.astype('M8[D]').astype(object)
        if dateobj.month == 2 and dateobj.day == 29:
            leap_idx.append(idx)
    ds = ds.isel(time=~np.isin(np.arange(ds.sizes['time']), np.asarray(leap_idx)))

    if downscaling:
        landsea_mask = dt.load_specified_lines(config['files']['land_sea_mask'], extent, False)[0] == 0
        fine_resolution = landsea_mask.shape #type:ignore
        output_dir = os.path.join(config['files'].get('climate_data_dir'), 'downscaled')
        os.makedirs(output_dir, exist_ok=True)
        total_days = ds.dims['time']
        max_proc = dt.get_cpu_ram()[0] - 1
        no_years = int(np.round(total_days / 365))
        
        if mode == 'temp':
            print('    -> Downscaling temperature data')
            dtshape =  ds[varname].values[1].shape
            world_clim_dir = config['files'].get('worldclim_temperature_data_dir')
            ext = nc.check_extent_load_file(os.path.join(world_clim_dir, f'factors_month_1.nc'), extent=extent)
            if not ext:
                [os.remove(file) for i in range(1, 13) if (file := os.path.join(world_clim_dir, f'factors_month_{i}.nc')) and os.path.exists(file)]
                ti.calculate_temp_factors(fine_resolution, dtshape, world_clim_dir, extent)
            print_progress_bar(0, no_years, prefix='       Progress', suffix='Complete', length=50)
            for year in range(no_years):
                current_data = ds[varname].values[(year * 365):((year + 1) * 365)]
                parallel_processing(current_data, extent, None, fine_resolution, year, max_proc, output_dir, 'temp', landsea_mask, world_clim_dir, convert_units)
                print_progress_bar(year, no_years, prefix='       Progress', suffix='Complete', length=50)            
            return [os.path.join(output_dir, f'ds_temp_{day}.nc') for day in range(len(os.listdir(output_dir)))]
        else:
            print('    -> Downscaling precipitation data')
            world_clim_dir = config['files'].get('worldclim_precipitation_data_dir')
            dtshape =  ds[varname].values[1].shape
            ext = nc.check_extent_load_file(os.path.join(world_clim_dir, f'factors_month_1.nc'), extent=extent)
            if not ext:
                [os.remove(file) for i in range(1, 13) if (file := os.path.join(world_clim_dir, f'factors_month_{i}.nc')) and os.path.exists(file)]
                pi.calculate_prec_factors(fine_resolution, dtshape, world_clim_dir, extent)
            print_progress_bar(0, no_years, prefix='       Progress', suffix='Complete', length=50)
            for year in range(no_years):
                current_data = ds[varname].values[(year * 365):((year + 1) * 365)]
                parallel_processing(current_data, extent, None, fine_resolution, year, max_proc, output_dir, 'prec', landsea_mask, world_clim_dir, convert_units)
                print_progress_bar(year, no_years, prefix='       Progress', suffix='Complete', length=50)
            return [os.path.join(output_dir, f'ds_prec_{day}.nc') for day in range(len(os.listdir(output_dir)))]
    else:
        return ds


def find_x_for_y_left(x, y, target_y):
    """
    Finds the x-axis values corresponding to a target y-value from the left and right.
    """
    if y[0] >= target_y:
        return x[0]
    x, y = np.array(x), np.array(y)
    left_idx = np.where(y >= target_y)[0][0]
    x_left = np.interp(target_y, [y[left_idx - 1], y[left_idx]], [x[left_idx - 1], x[left_idx]])
    return x_left

def find_right_x_for_y(x, y, target_y):
    if y[-1] >= target_y:
        return x[-1]
    x, y = np.array(x), np.array(y)
    for i in range(len(y) - 1, 0, -1):
        if y[i] < target_y <= y[i - 1]:
            return np.interp(target_y, [y[i], y[i - 1]], [x[i], x[i - 1]])

def get_limits(formulas, type, plant_params, no_stops = 100):
    x = np.linspace(formulas[type]['min_val'], formulas[type]['max_val'], no_stops)
    y = np.clip(formulas[type]['formula'](np.linspace(formulas[type]['min_val'], formulas[type]['max_val'], no_stops)), 0, 1)
    if y[-1] > 0:
        if x[-1] == 40 or x[-1] >= 4000:
            y[-1] = 0.
        else:
            step_size = x[1] - x[0]
            dest_val = 40 if type == 'temp' else (np.max(x) * 2)-1
            req = int((dest_val - np.nanmax(x)) // step_size)
            x = np.concatenate((x, [x[-1] + step_size * i for i in range(1, req)]))
            y = np.concatenate((y, [1 for i in range(1, req)])) #type: ignore
            y[-1] = 0.
    lower_bounds = find_x_for_y_left(x, y, 0.05)
    upper_bounds = find_right_x_for_y(x, y, 0.05)
    t = {'temp': 'temperature', 'prec': 'precipitation'}.get(type, '')
    u = {'temp': '°C', 'prec': 'mm'}.get(type, '')
    if type.lower() == 'temp' and 'hightemp_lim' in plant_params:
        upper_bounds = int(plant_params['hightemp_lim'][0])
    if type.lower() == 'temp' and 'lowtemp_lim' in plant_params:
        lower_bounds = int(plant_params['lowtemp_lim'][0])
    if type.lower() == 'prec' and 'highprec_lim' in plant_params:
        upper_bounds = int(plant_params['highprec_lim'][0])
    if type.lower() == 'prec' and 'lowprec_lim' in plant_params:
        lower_bounds = int(plant_params['lowprec_lim'][0])
    print(f' -> Lower {t} limit for extremes is {int(np.round(lower_bounds, 2))} {u}') #type:ignore
    print(f' -> Upper {t} limit for extremes is {int(np.round(upper_bounds, 2))} {u}') #type:ignore
    return lower_bounds, upper_bounds


def get_data_remove_leapyears(ds, varname):
    leap_idx = []
    for idx, tv in enumerate(list(ds['time'].values)):
        dt = tv.astype('M8[D]').astype(object)
        if dt.month == 2 and dt.day == 29:
            leap_idx.append(idx)
    
    arr = np.asarray(ds[varname])
    if len(leap_idx) > 0:
        return np.delete(arr, np.array(leap_idx), axis=0)
    else:
        return arr

def calculate_add_condition(entry, data):
    if entry[0] == 'Temperature':
        data_range = np.mean(data, axis=0)
    else:
        data_range = np.sum(data, axis=0)
    boolarr = np.full(data_range.shape, False, dtype=bool)
    if entry[3] == '<':
        boolarr[data_range >= float(entry[4])] = True
    elif entry[3] == '<=':
        boolarr[data_range > float(entry[4])] = True
    elif entry[3] == '>=':
        boolarr[data_range < float(entry[4])] = True
    elif entry[3] == '>':
        boolarr[data_range <= float(entry[4])] = True
    return boolarr


def calculate_crop_rrpcf(crop_list, crop_failure_code, temp_files, prec_files, config_dict, start_year, end_year, extent, varname_temp, varname_pr, downscaling):
    climate_data_dir = config_dict['files'].get('climate_data_dir')
    plant_param_dir = config_dict['files'].get('plant_param_dir', 'plant_params')
    plant_param_dict = rpp.read_crop_parameterizations_files(plant_param_dir)
    plant_params_formulas = rpp.get_plant_param_interp_forms_dict(plant_param_dict, config_dict)
    timerange = end_year - start_year + 1
    for crop in crop_list:
        print('')
        out_path_rf = os.path.join(climate_data_dir, f'{crop_failure_code}_{os.path.splitext(crop)[0].lower()}_rf.nc')
        out_path_ir = os.path.join(climate_data_dir, f'{crop_failure_code}_{os.path.splitext(crop)[0].lower()}_ir.nc')
        if os.path.exists(out_path_ir) and os.path.exists(out_path_rf):
            print(f' => Data for {os.path.splitext(crop)[0].capitalize()} already existing. Skipping.')
            continue
        print(f'--- Processing {os.path.splitext(crop)[0].capitalize()} --- ')
        crop_dict = plant_param_dict.get(os.path.splitext(crop.lower())[0])
        growing_cycle = int(crop_dict.get('growing_cycle', [365])[0]) #type:ignore
        crop_forms = plant_params_formulas.get(os.path.splitext(crop.lower())[0])
        # Wintercrop:
        if crop_dict.get('wintercrop', False): #type:ignore
            growing_cycle -= int(crop_dict.get('days_to_vernalization')[0]) #type:ignore

        # Consider additional Parameters:
        if int(crop_dict.get('consider_in_preproc', '0')[0]) == 1: #type:ignore
            additional_conditions = [cond for i in range(100) if (cond := crop_dict.get(f'AddCon:{i}')) is not None] #type:ignore
        else:
            additional_conditions = []

        # TEMPERATURE
        temp_lower, temp_upper = get_limits(crop_forms, 'temp', crop_dict)
        if not downscaling:
            temp_data = get_nc_data(temp_files, start_year, end_year, extent=extent, downscaling=downscaling, config=config_dict)
            varname = list(temp_data.data_vars)[0] #type:ignore
            temp_data = get_data_remove_leapyears(temp_data, varname)
            if np.nanmean(temp_data) > 100:
                temp_data -= 273.15
            shp = temp_data.shape[1:]
        else:
            filepaths = get_nc_data(temp_files, start_year, end_year, extent=extent, downscaling=downscaling, config=config_dict, mode='temp')
            shp = nc.get_netcdf_shape(filepaths[0])[nc.get_variable_name_from_nc(filepaths[0])]

        if growing_cycle >= 364:
            t_lower, t_upper = np.zeros(shp, dtype=np.int8), np.zeros(shp, dtype=np.int8)
            print('n    -> Calculation of limit value exceedances')
            for timeslice in range(timerange):
                if downscaling:
                    temp_data = nc.read_area_from_netcdf_list(filepaths[timeslice*growing_cycle:(timeslice+1)*growing_cycle], extent=extent, dayslices=True, transp=False)
                    curr_mean = np.mean(temp_data, axis=0) #type:ignore
                    add_conds = []
                    for entry in additional_conditions:
                        if entry[0] == 'Temperature':
                            temp_range = temp_data[int(entry[1]):int(entry[2])]
                            add_conds.append(calculate_add_condition(entry, temp_range))
                else:
                    curr_mean = np.mean(temp_data[timeslice*growing_cycle:(timeslice+1)*growing_cycle], axis=0) #type:ignore
                    add_conds = []
                    for entry in additional_conditions:
                        if entry[0] == 'Temperature':
                            temp_range = temp_data[timeslice*growing_cycle+int(entry[1]):(timeslice)*growing_cycle+int(entry[2])]
                            add_conds.append(calculate_add_condition(entry, temp_range))

                t_lower += curr_mean <= temp_lower
                t_upper += curr_mean >= temp_upper
                for add_cond in add_conds:
                    t_upper[day] = np.max([t_upper[day], add_cond], axis=0)
            t_lower = ((t_lower / timerange) * 100).astype(np.int8)
            t_upper = ((t_upper / timerange) * 100).astype(np.int8)
        else:
            t_lower, t_upper = np.zeros((365, *shp), dtype=np.int8), np.zeros((365, *shp), dtype=np.int8)
            print('\n    -> Calculation of limit value exceedances')

            for year in range(timerange):
                if downscaling:
                    temp_data = nc.read_area_from_netcdf_list(filepaths[year*365:((year+1)*365)+growing_cycle], extent=extent, dayslices=True, transp=False)
                for day in range(365):
                    if downscaling:
                        curr_mean = np.mean(temp_data[day:day+growing_cycle], axis=0) #type:ignore
                        add_conds = []
                        for entry in additional_conditions:
                            if entry[0] == 'Temperature':
                                temp_range = temp_data[day + int(entry[1]):day+int(entry[2])]
                                add_conds.append(calculate_add_condition(entry, temp_range))
                    else:
                        curr_mean = np.mean(temp_data[(year * 365) + day : (year*365) + day + growing_cycle], axis=0) #type:ignore
                        add_conds = []
                        for entry in additional_conditions:
                            if entry[0] == 'Temperature':
                                temp_range = temp_data[(year * 365) + day + int(entry[1]) : (year*365) + day + int(entry[2])]
                                add_conds.append(calculate_add_condition(entry, temp_range))
                    t_lower[day] += curr_mean <= temp_lower
                    t_upper[day] += curr_mean >= temp_upper
                    for add_cond in add_conds:
                        t_upper[day] = np.max([t_upper[day], add_cond], axis=0)
            t_lower = np.asarray((t_lower / timerange) * 100, dtype=np.int8)
            t_upper = np.asarray((t_upper / timerange) * 100, dtype=np.int8)
        del temp_data

        # PRECIPITATION

        print('')
        prec_lower, prec_upper = get_limits(crop_forms, 'prec', crop_dict)
        if not downscaling:
            prec_data = get_nc_data(prec_files, start_year, end_year, extent=extent, downscaling=downscaling, config=config_dict)
            varname = list(prec_data.data_vars)[0] #type:ignore
            prec_data = get_data_remove_leapyears(prec_data, varname)
            if np.nanmax(prec_data) < 5:
                prec_data *= 3600 * 24
            shp = prec_data.shape[1:]
        else:
            filepaths = get_nc_data(prec_files, start_year, end_year, extent=extent, downscaling=downscaling, config=config_dict, mode='prec')
            shp = nc.get_netcdf_shape(filepaths[0])[nc.get_variable_name_from_nc(filepaths[0])]
        if growing_cycle >= 364:
            p_lower, p_upper = np.zeros(shp, dtype=np.int8), np.zeros(shp, dtype=np.int8)
            print('    -> Calculation of limit value exceedances')
            for timeslice in range(timerange):
                if downscaling:
                    prec_data = nc.read_area_from_netcdf_list(filepaths[timeslice*growing_cycle:(timeslice+1)*growing_cycle], extent=extent, dayslices=True, transp=False)
                    curr_sum = np.sum(prec_data, axis=0) #type:ignore
                    add_conds = []
                    for entry in additional_conditions:
                        if entry[0] == 'Precipitation':
                            prec_range = prec_data[day+int(entry[1]):day+int(entry[2])]
                            add_conds.append(calculate_add_condition(entry, prec_range))

                else:
                    curr_sum = np.sum(prec_data[timeslice*growing_cycle:(timeslice+1)*growing_cycle], axis=0) #type:ignore
                    add_conds = []
                    for entry in additional_conditions:
                        if entry[0] == 'Precipitation':
                            prec_range = prec_data[timeslice*growing_cycle+entry[1]:(timeslice)*growing_cycle+entry[2]]
                            add_conds.append(calculate_add_condition(entry, prec_range))

                p_lower += curr_sum <= prec_lower
                p_upper += curr_sum >= prec_upper
                for add_cond in add_conds:
                    p_upper[day] = np.max([p_upper[day], add_cond], axis=0)
            p_lower = ((p_lower / timerange) * 100).astype(np.int8)
            p_upper = ((p_upper / timerange) * 100).astype(np.int8)
        else:
            p_lower, p_upper = np.zeros((365, *shp), dtype=np.int8), np.zeros((365, *shp), dtype=np.int8)
            print('    -> Calculation of limit value exceedances')
            for year in range(timerange):
                if downscaling:
                    prec_data = nc.read_area_from_netcdf_list(filepaths[year*365:((year+1)*365)+growing_cycle], extent=extent, dayslices=True, transp=False)
                for day in range(365):
                    add_conds = []
                    if downscaling:
                        curr_sum = np.sum(prec_data[day:day+growing_cycle], axis=0) #type:ignore
                        for entry in additional_conditions:
                            if entry[0] == 'Precipitation':
                                prec_range = prec_data[day+int(entry[1]):day+int(entry[2])]
                                add_conds.append(calculate_add_condition(entry, prec_range))
                    else:
                        curr_sum = np.sum(prec_data[(year * 365) + day : (year*365) + day + growing_cycle], axis=0) #type:ignore
                        for entry in additional_conditions:
                            if entry[0] == 'Precipitation':
                                prec_range = prec_data[(year * 365) + day + int(entry[1]) : (year*365) + day + int(entry[2])]
                                add_conds.append(calculate_add_condition(entry, prec_range))
                    p_lower[day] += curr_sum <= prec_lower
                    p_upper[day] += curr_sum >= prec_upper
                    for add_cond in add_conds:
                        p_upper[day] = np.max([p_upper[day], add_cond], axis=0)

            p_lower = np.asarray((p_lower / timerange) * 100, dtype=np.int8)
            p_upper = np.asarray((p_upper / timerange) * 100, dtype=np.int8)
        del prec_data
        # Rainfed:
        rrcpf = np.max(np.asarray([t_lower, t_upper, p_lower, p_upper]), axis=0)
        if rrcpf.ndim == 3:
            rrcpf = rrcpf.transpose(1, 2, 0)
            nc.write_to_netcdf(rrcpf, out_path_rf, ['lat', 'lon', 'time'], extent=extent, compress=True, complevel=9, var_name=crop_failure_code, nodata_value=-1) #type:ignore
        else:
            nc.write_to_netcdf(rrcpf, out_path_rf, ['lat', 'lon'], extent=extent, compress=True, complevel=9, var_name=crop_failure_code, nodata_value=-1) #type:ignore
        # Irrigated:
        rrcpf = np.max(np.asarray([t_lower, t_upper, p_upper]), axis=0)
        if rrcpf.ndim == 3:
            rrcpf = rrcpf.transpose(1, 2, 0)
            nc.write_to_netcdf(rrcpf, out_path_ir, ['lat', 'lon', 'time'], extent=extent, compress=True, complevel=9, var_name=crop_failure_code, nodata_value=-1) #type:ignore
        else:
            nc.write_to_netcdf(rrcpf, out_path_ir, ['lat', 'lon'], extent=extent, compress=True, complevel=9, var_name=crop_failure_code, nodata_value=-1) #type:ignore
        print(f' -> {os.path.splitext(crop)[0].capitalize()} complete!')    

def preprocessing_main(config_ini, temp_files, prec_files, time_range, extent, proc_varfiles, varname_temp='tas', varname_pr='pr', downscaling=False):
    # Read Config.ini
    config = rci.read_ini_file(config_ini)
    # Get Plant Parameterization Directory
    plant_param_dir = config['files'].get('plant_param_dir', 'plant_params')
    climate_data_dir = config['files'].get('climate_data_dir')
    # Get Crops
    crop_list = [f for f in os.listdir(plant_param_dir) if f.endswith('.inf')]
    crop_failure_code = 'rrpcf'
    if extent == 0:
        extent = {'top': config['extent'].get('upper_left_y'), 'left': config['extent'].get('upper_left_x'),
                  'bottom': config['extent'].get('lower_right_y'), 'right': config['extent'].get('lower_right_x')}
    else:
        extent = {'top': 90, 'left': -180, 'bottom': -90, 'right': 180}
    process_climvar_files = proc_varfiles == 1
    
    try:
        start_year, end_year = time_range
    except:
        start_year, end_year = tuple(map(int, time_range.strip('()').split(',')))
    # Temperature
    out_file = os.path.join(climate_data_dir, 'Temp_avg.nc')
    if not os.path.exists(out_file):
        print(' --- Calculation of daily mean temperature values for the selected period ---')
        calculate_daily_temp_mean(temp_files, out_file, start_year, end_year, extent = extent)
    # Precipitation
    out_file = os.path.join(climate_data_dir, 'Prec_avg.nc')
    if not os.path.exists(out_file):
        print(' --- Calculation of daily precipitation values for the selected period ---')
        calculate_daily_prec_sum(prec_files, out_file, start_year, end_year, extent = extent)
    if process_climvar_files:
        calculate_crop_rrpcf(crop_list, crop_failure_code, temp_files, prec_files, config, start_year, end_year, extent, varname_temp, varname_pr, downscaling)
    print(' === Completed! === ')


def parse_value(value):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
        if value:
            return [os.path.normpath(item.strip(" '\"")) for item in value.split(",")]
        return []
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.replace(".", "", 1).isdigit():
        return float(value) if "." in value else int(value)
    return value.strip(" '\"")

def read_pp_file(pp_file):
    variables = {}
    with open(pp_file, 'r') as rf:
        for line in rf:
            key, value = line.strip().split(" = ", 1)
            variables[key] = parse_value(value)
    return variables

def parse_inf_file(config_file):
    params = read_pp_file(config_file)
    autostart = params.get('autostart')
    config_ini = params.get('config_ini')
    temp_files = params.get('temp_files', [])
    temp_varname = params.get('temp_varname')
    prec_files = params.get('prec_files', [])
    prec_varname = params.get('prec_varname')
    time_range = params.get('time_range')
    extent = params.get('extent')
    proc_varfiles = params.get('proc_varfiles')
    downscaling = params.get('downscaling')
    return config_ini, temp_files, temp_varname, prec_files, prec_varname, time_range, extent, proc_varfiles, downscaling, autostart

if __name__ == '__main__':
    config_file = None
    for arg in sys.argv:
        if arg[:9].lower() == '-filename':
            config_file = arg[11:]
            break
    
    if config_file:
        if not os.path.exists(config_file):
            print('Specified config file does not exist.')
            sys.exit(1)
        config_ini, temp_files, temp_varname, prec_files, prec_varname, time_range, extent, proc_varfiles, downscaling, autostart = parse_inf_file(config_file)
        preprocessing_main(config_ini, temp_files, prec_files, tuple(map(int, time_range.strip('()').split(','))), extent, proc_varfiles, temp_varname, prec_varname, downscaling=downscaling==1) #type:ignore
    else:
        print("Usage: python preproc_tools.py -filename config.ini")
        exit()



