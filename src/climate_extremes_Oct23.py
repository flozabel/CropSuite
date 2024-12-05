import read_plant_params as rpp
import read_climate_ini as rci
import os
import numpy as np
import rasterio
import concurrent.futures
import data_tools as dt
from concurrent.futures import ThreadPoolExecutor
from psutil import virtual_memory
from multiprocessing import cpu_count
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning) 


def startup():
        print('''\
        
        =======================================================
        |                                                     |
        |                                                     |    
        |                      CropSuite                      |
        |                                                     |
        |                     Version  0.5                    |
        |                      2023-11-04                     |
        |                                                     |
        |       - Create Files for Climate Variability -      |
        |                                                     |
        |                      2023-11-04                     |
        |                                                     |
        |                                                     |
        |        Based on the work of Zabel et al. 2014       |
        |                                                     |
        |                                                     |
        |                     Florian Zabel                   |
        |                   Matthias Knüttel                  |
        |                         2023                        |
        |                                                     |
        |                Department of Geography              |      
        |        Ludwig Maximilians University of Munich      |
        |                                                     |
        |                                                     |
        |                © All rights reserved                |
        |                                                     |
        =======================================================
        
        ''')
        input('\n\nPress Any Key to Start\n\n')


def TicTocGenerator():
    ti, tf = 0, time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti
TicToc = TicTocGenerator()

def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print(f'Elapsed time: {tempTimeInterval} seconds.')

def tic():
    toc(False)



def get_file_paths_with_string(root_path, target_string):
    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.tif') and target_string.lower() in file.lower():
                file_paths.append(os.path.join(root, file))
    return file_paths


def get_opposite_file(file_path):
    base_name = os.path.basename(file_path)
    if '2041' in base_name:
        print('')
    if 'Tmax' in base_name:
        return file_path.replace('Tmax', 'Tmin')
    elif 'Tmin' in base_name:
        return file_path.replace('Tmin', 'Tmax')
    else:
        print('ERROR: No Tmax/Tmin File!\nExiting')
        exit()


def process_file(in_file1, in_file2, out_file):
    if not os.path.exists(out_file):   
        with rasterio.open(in_file1) as src1, rasterio.open(in_file2) as src2:
            profile = src1.profile
            data_array = np.empty((2, profile['height'], profile['width']), dtype=profile['dtype'])
            data_array[0, :, :] = src1.read(1)
            data_array[1, :, :] = src2.read(1)
            mean_data = np.nanmean(data_array, axis=0)
            profile.update(count=1, compress='lzw')
            with rasterio.open(out_file, 'w', **profile) as out:
                out.write(mean_data, 1)


def get_temp_dail_avg(temp_min, temp_max, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    def get_file_paths(directory):
        return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith('.tif')]

    tmax_files = sorted(get_file_paths(temp_max))
    tmin_files = sorted(get_file_paths(temp_min))
    files_to_process = [[tmax, tmin, os.path.join(out_folder, os.path.basename(tmin).replace('Tmin', 'Tavg'))] for tmax, tmin in zip(tmax_files, tmin_files)]
    """
    #DEBUG
    for file_paths in files_to_process:
        process_file(*file_paths)
        
    """
    with ThreadPoolExecutor() as executor:
        for file_paths in files_to_process:
            executor.submit(process_file, *file_paths)
    
    return out_folder

def get_temp_average_file(temp_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if os.path.exists(os.path.join(out_path, 'Temp_avg.tif')):
        return os.path.join(out_path, 'Temp_avg.tif')

    temp_avg2 = os.path.join(out_path, 'temp_avg2')
    if not os.path.exists(temp_avg2):
        os.makedirs(temp_avg2)
    
    file_list = []
    for month in range(12):
        for day in range(31):
            name_string = '.'+"{:02d}".format(month+1)+'.'+"{:02d}".format(day+1)+'.tif'
            returns = get_file_paths_with_string(os.path.join(temp_path), name_string)
            if len(returns) < 1:
                continue
            file_list.append(returns)

    """
    # DEBUG
    for fn in file_list:
        process_files(fn, temp_avg2)
    """
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_files, fn, temp_avg2) for fn in file_list}
        for future in futures:
            future.result()

    file_list = sorted([os.path.join(temp_avg2, f) for f in os.listdir(temp_avg2) if f.endswith('.tif')])
    with rasterio.open(file_list[0]) as met_f:
        out_meta = met_f.meta.copy()
        out_meta.update(count=len(file_list), compress='lzw')
        with rasterio.open(os.path.join(out_path, 'Temp_avg.tif'), 'w', **out_meta) as dst:
            for idx, layer in enumerate(file_list, start=1):
                with rasterio.open(layer) as src:
                    dst.write_band(idx, src.read(1))

    for f in file_list:
        os.remove(f)
    os.rmdir(os.path.dirname(file_list[0]))
    
    return os.path.join(out_path, 'Temp_avg.tif')


def process_files(files, out_path_temp):
    date = os.path.basename(files[0])[-9:-4]
    out_file = os.path.join(out_path_temp, f'Tavg_{date}.tif')
    if not os.path.exists(out_file):
        first_file_path = files[0]
        with rasterio.open(first_file_path) as src:
            profile = src.profile
            arr_shape = (len(files), profile['height'], profile['width'])
            arr_dtype = profile['dtype']
        data_array = np.empty(arr_shape, dtype=arr_dtype)
        for i, tif_file in enumerate(files):
            with rasterio.open(tif_file) as src:
                data_array[i, :, :] = src.read(1)

        mean_data = np.mean(data_array, axis=0)
        profile.update(count=1, compress='lzw')
        with rasterio.open(out_file, 'w', **profile) as out:
            out.write(mean_data, 1)


def process_files_prec(files, prec_path):
    try:
        date = os.path.basename(files[0])[-9:-4]
        out_file = os.path.join(prec_path, f'Pavg_{date}.tif')
        if not os.path.exists(out_file):
            first_file_path = files[0]
            with rasterio.open(first_file_path) as src:
                profile = src.profile
                arr_shape = (len(files), profile['height'], profile['width'])
                arr_dtype = profile['dtype']
            data_array = np.empty(arr_shape, dtype=arr_dtype)
            for i, tif_file in enumerate(files):
                with rasterio.open(tif_file) as src:
                    data_array[i, :, :] = src.read(1)
            mean_data = np.nanmean(data_array, axis=0)
            profile.update(count=1, compress='lzw')
            with rasterio.open(out_file, 'w', **profile) as out:
                out.write(mean_data, 1)
    except:
        pass


def get_prec_average(prec_path_in, climate_data_out):
    if not os.path.exists(climate_data_out):
        os.makedirs(climate_data_out)
    if os.path.exists(os.path.join(climate_data_out, 'Prec_avg.tif')):
        return os.path.join(climate_data_out, 'Prec_avg.tif')

    file_list = []
    for month in range(12):
        for day in range(31):
            name_string = '.'+"{:02d}".format(month+1)+'.'+"{:02d}".format(day+1)
            file_list.append(get_file_paths_with_string(prec_path_in, name_string))
    
    temp = os.path.join(climate_data_out, 'prec')
    if not os.path.exists(temp):
        os.makedirs(temp)

    """
    # DEBUG
    for fls in file_list:
        process_files_prec(fls, temp)
    """

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_files_prec, fls, temp) for fls in file_list}
        for future in futures:
            future.result()
    
    file_list = sorted([os.path.join(temp, f) for f in os.listdir(temp) if f.endswith('.tif')])

    with rasterio.open(file_list[0]) as met_f:
        out_meta = met_f.meta.copy()
        out_meta.update(count=len(file_list), compress='lzw')
        with rasterio.open(os.path.join(climate_data_out, 'Prec_avg.tif'), 'w', **out_meta) as dst:
            for idx, layer in enumerate(file_list, start=1):
                with rasterio.open(layer) as src:
                    dst.write_band(idx, src.read(1))

    for f in file_list:
        os.remove(f)
    os.rmdir(os.path.dirname(file_list[0]))
    return os.path.join(climate_data_out, 'Prec_avg.tif')


def get_tif_dimensions(file_path):
    try:
        with rasterio.open(file_path) as dataset:
            return dataset.width, dataset.height, dataset.count
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving dimensions of {file_path}: {e}')
        return [0, 0, 0]


def get_geotiff_extent(file_path):
    try:
        with rasterio.open(file_path) as dataset:
            return dataset.bounds
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving extent of {file_path}: {e}')
        return [0, 0, 0, 0]


def find_entries_with_string(input_list, target_string):
    return sorted([entry for entry in input_list if target_string in entry])

def find_tif_files(root_dir, target_string=None):
    data_lst = sorted([os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk(root_dir) for filename in filenames if filename.endswith('.tif') and (not target_string or target_string not in filename)])
    return [find_entries_with_string(data_lst, str(fn)[-9:-4]) for fn in data_lst if str(fn)[-9:-4] not in {str(day) for day in data_lst}]

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
            y = np.concatenate((y, [1 for i in range(1, req)]))
            y[-1] = 0.
    lower_bounds = np.percentile(x, 5)
    upper_bounds = np.percentile(x, 95)
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
    print(f' -> Lower {t} limit for extremes is {int(np.round(lower_bounds, 0))} {u}')
    print(f' -> Upper {t} limit for extremes is {int(np.round(upper_bounds, 0))} {u}')
    return lower_bounds, upper_bounds


def calculate_mean(tiff_file1, tiff_file2):
    with rasterio.open(tiff_file1) as src1, rasterio.open(tiff_file2) as src2:
        return (src1.read(1) + src2.read(1)) / 2.0, src1.profile


def save_geotiff(output_path, data, profile):
    try:
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)
    except Exception as e:
        print(f'Error: {e}')


def get_mean(temp_data, output):
    os.makedirs(output, exist_ok=True)

    def process_day(daydata, output):
        processed_years = set()
        for day in daydata:
            year = day[-14:-10]
            if year in processed_years:
                continue
            day = day[-14:-4]
            out_path = os.path.join(output, f'Tavg.{day}.tif')
            if not os.path.exists(out_path):
                data = find_entries_with_string(daydata, day)
                mean, profile = calculate_mean(data[0], data[1])
                profile.update(compress='lzw')
                save_geotiff(out_path, mean, profile)
            processed_years.add(year)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for daydata in temp_data:
            futures.append(executor.submit(process_day, daydata, output))
        for future in concurrent.futures.as_completed(futures):
            pass
    return output


def throw_exit_error(text):
    input(text+'\nExit')
    exit()


def read_tif_files_to_array(tif_list, arr_shape, timesteps, conversion_factor=1):
    out_arr = np.zeros((arr_shape[0], arr_shape[1], timesteps), dtype=np.int16)
    def read_and_process_tiff(idx, fn):
        with rasterio.open(fn, 'r') as src:
            data = src.read(1)
            if data.shape != arr_shape: data = np.pad(data, ((0, arr_shape[0] - data.shape[0]), (0, arr_shape[1] - data.shape[1])), mode='edge')
            out_arr[..., idx] = (data * conversion_factor).astype(np.int16)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_and_process_tiff, idx, fn) for idx, fn in enumerate(tif_list)]
        for future in futures:
            future.result()
    return out_arr


def process_yearly_extremes(arrshape, len_growing_cycle, limit_value, climate_data, limit, dataset, num_days = 365, conversion_factor=1):
    out_arr = np.zeros((arrshape[0], arrshape[1], num_days), dtype=np.int8)
    limit_value = limit_value * conversion_factor
    for day in range(num_days):
        if len_growing_cycle < climate_data.shape[-1]:
            if day + len_growing_cycle >= 365:
                curr_data = np.concatenate((climate_data[..., day:], climate_data[..., :day + len_growing_cycle - 365]), axis=2)
            else:
                curr_data = climate_data[..., day:day + len_growing_cycle]
        else:
            curr_data = climate_data  
        if limit == 'upper':
            out_arr[..., day] = (np.sum(curr_data >= limit_value, axis=2, dtype=np.int16) * 100) // len_growing_cycle
        else:
            out_arr[..., day] = (np.sum(curr_data <= limit_value, axis=2, dtype=np.int16) * 100) // len_growing_cycle
    return out_arr


def process_yearly_extremes2(arrshape, len_growing_cycle, limit_value, climate_data, limit, dataset, num_days=365, conversion_factor=1):
    out_arr = np.empty((arrshape[0], arrshape[1], num_days), dtype=np.uint8)
    limit_value *= conversion_factor
    if len_growing_cycle == 365:
        curr_data = np.cumsum(climate_data, axis=2)
    else:
        curr_data = np.cumsum(np.concatenate([climate_data, climate_data[..., :len_growing_cycle]], axis=2), axis=2)
    for day in range(num_days):
        if dataset == 'temp':
            if len_growing_cycle == 365:
                diff = curr_data[..., -1] // len_growing_cycle
            else:
                diff = (curr_data[..., day + len_growing_cycle] - curr_data[..., day]) // len_growing_cycle
        else:
            if len_growing_cycle == 365:
                diff = curr_data[..., -1]
            else:
                diff = curr_data[..., day + len_growing_cycle] - curr_data[..., day]
        if limit == 'upper':
            out_arr[..., day] = diff >= limit_value
        else:
            out_arr[..., day] = diff <= limit_value
        if len_growing_cycle == 365:
            return np.repeat(out_arr[..., 0][..., np.newaxis], 365, axis=2)
    return out_arr


def process_single_year_parallel(args):
    year, rows, cols, growing_cycles, temp_data, prec_data, temp_lower_lst, temp_upper_lst, \
    prec_lower_lst, prec_upper_lst, extent, climate_data_dir, factor, idx, crops, irrigation = args
    process_single_year_optimized(year, rows, cols, growing_cycles, temp_data, prec_data, temp_lower_lst,
                        temp_upper_lst, prec_lower_lst, prec_upper_lst, extent, climate_data_dir, factor, idx, crops, irrigation)
    


def load_temp_data_all(directory, temp_data, arr_shape):
    temp_data = sorted(temp_data)
    out_arr = np.zeros((arr_shape[0], arr_shape[1], len(temp_data)), dtype=np.int16)
    for idx, fn in enumerate(temp_data):
        if fn[-4:] == '.tif':
            with rasterio.open(os.path.join(directory, fn), 'r') as src:
                out_arr[..., idx] = np.asarray(src.read(1) * 100).astype(np.int16)
            print(f' {idx} read')
    print(out_arr.shape)
    return out_arr


def all_files_exist(file_list):
    for file_path in file_list:
        if not os.path.exists(file_path):
            return False
    return True


def custom_cumsum_secondaxis(array):
    for i in range(array.shape[2]):
        if i == 0:
            continue
        else:
            array[..., i] = array[..., i - 1] + array[..., i]
    return array


def process_single_year_optimized(year, rows, cols, growing_cycles, temp_data, prec_data, temp_lower_lst, temp_upper_lst,
                                  prec_lower_lst, prec_upper_lst, extent, climate_data, factor, year_idx, crops, irrigation=False):
    climdat_temp, climdat_prec = None, None
    if not all_files_exist([os.path.join(climate_data, f'Ex_temp_{crop}_{year}.tif') for crop in crops]):
        climdat_temp = read_tif_files_to_array([temp_data[i][year_idx] for i in range(365)], (rows, cols), 365, 10)
        for crop_idx, crop in enumerate(crops):
            if os.path.exists(os.path.join(climate_data, f'Ex_{crop}_{"ir" if irrigation else "rf"}.tif')):
                continue
            if not os.path.exists(os.path.join(climate_data, f'Ex_temp_{crop}_{year}.tif')):
                print(f'{crop} - Temperature - Current Year: {year}')
                temp_upper_lim, temp_lower_lim = temp_upper_lst[crop_idx], temp_lower_lst[crop_idx]
                len_growing_cycle = growing_cycles[crop_idx]
                if len_growing_cycle >= 365:
                    mean_values = np.mean(climdat_temp, axis=2)
                    var_temp = (mean_values < temp_lower_lim*10) | (mean_values > temp_upper_lim*10)
                    del mean_values
                else:
                    curr_data = np.cumsum(np.concatenate([climdat_temp, climdat_temp[..., :len_growing_cycle]], axis=2), axis=2)
                    #curr_data = custom_cumsum_secondaxis(np.concatenate([climdat_temp, climdat_temp[..., :len_growing_cycle]], axis=2).astype(np.int32))
                    var_temp = np.zeros(climdat_temp.shape, dtype=bool)
                    for day in range(365):
                        mean_values = (curr_data[..., day + len_growing_cycle] - curr_data[..., day]) // len_growing_cycle
                        var_temp[..., day] = (mean_values < temp_lower_lim*10) | (mean_values > temp_upper_lim*10)
                        del mean_values
                dt.write_geotiff(climate_data, f'Ex_temp_{crop}_{year}.tif', array=var_temp, extent=extent, dtype='bool')
                del var_temp
        del climdat_temp

    if not all_files_exist([os.path.join(climate_data, f'Ex_prec_{crop}_{year}.tif') for crop in crops]):
        climdat_prec = read_tif_files_to_array([prec_data[i][year_idx] for i in range(365)], (rows, cols), 365, 10)
        for crop_idx, crop in enumerate(crops):
            if os.path.exists(os.path.join(climate_data, f'Ex_{crop}_{"ir" if irrigation else "rf"}.tif')):
                continue
            if not os.path.exists(os.path.join(climate_data, f'Ex_prec_{crop}_{year}.tif')):
                print(f'{crop} - Precipitation - Current Year: {year}')
                prec_upper_lim, prec_lower_lim = prec_upper_lst[crop_idx], prec_lower_lst[crop_idx]
                len_growing_cycle = growing_cycles[crop_idx]
                if len_growing_cycle >= 365:
                    sum_values = np.sum(climdat_prec, axis=2)
                    if irrigation:
                        var_prec = sum_values > prec_upper_lim*10
                    else:
                        var_prec = (sum_values < prec_lower_lim*10) | (sum_values > prec_upper_lim*10)
                    del sum_values
                else:
                    curr_data = np.cumsum(np.concatenate([climdat_prec, climdat_prec[..., :len_growing_cycle]], axis=2), axis=2)
                    #curr_data = custom_cumsum_secondaxis(np.concatenate([climdat_prec, climdat_prec[..., :len_growing_cycle]], axis=2).astype(np.int32))
                    var_prec = np.zeros(climdat_prec.shape, dtype=bool)
                    for day in range(365):
                        sum_values = (curr_data[..., day + len_growing_cycle] - curr_data[..., day])
                        if irrigation:
                            var_prec[..., day] = sum_values > prec_upper_lim*10
                        else:
                            var_prec[..., day] = (sum_values < prec_lower_lim*10) | (sum_values > prec_upper_lim*10)
                        del sum_values
                dt.write_geotiff(climate_data, f'Ex_prec_{crop}_{year}.tif', array=var_prec, extent=extent, dtype='bool')
                del var_prec
        del climdat_prec


def find_matching_tif_files(directory, start_string):
    tif_files = []
    for filename in os.listdir(directory):
        if filename.startswith(start_string) and filename.endswith('.tif'):
            tif_files.append(os.path.join(directory, filename))
    return sorted(tif_files)


def calculate_mean_tif(tif_files, output_file):
    with rasterio.open(tif_files[0]) as src:
        meta = src.meta
        shape = src.shape
    mean_data_accumulator = np.zeros((shape[0], shape[1]), dtype=np.int16)
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            mean_data_accumulator += src.read(1)
    mean_data = (mean_data_accumulator / len(tif_files)).astype(np.int8)
    del mean_data_accumulator
    meta.update(dtype=np.int8)
    with rasterio.open(output_file, 'w', **meta) as dst:
        for band_index in range(365):
            dst.write(mean_data, band_index + 1)
    return output_file


def get_cpu_ram():
    return [cpu_count(), virtual_memory().available/1000000]

def merge_boolean_files(input_files_temp, input_files_prec, out_file, extent):
    input_files_temp, input_files_prec = sorted(input_files_temp), sorted(input_files_prec)
    with rasterio.open(input_files_temp[0]) as src1:
        height, width, bands = src1.height, src1.width, src1.count
        if bands == 1:
            combined_data = np.zeros((height, width, len(input_files_temp)), dtype=bool)
        else:
            combined_data = np.zeros((bands, height, width, len(input_files_temp)), dtype=bool)
            
    for idx in range(len(input_files_temp)):
        temp_file, prec_file = input_files_temp[idx], input_files_prec[idx]
        if bands == 1:
            with rasterio.open(temp_file) as tmp:
                with rasterio.open(prec_file) as prc:
                    combined_data[..., idx] = tmp.read(1) | prc.read(1)
        else:
            with rasterio.open(temp_file) as tmp:
                with rasterio.open(prec_file) as prc:
                    combined_data[..., idx] = tmp.read() | prc.read()
    """
    for idx, input_file in enumerate(input_files_temp):
        with rasterio.open(input_file) as src:
            if bands == 1:
                combined_data[..., idx] = src.read(1)
            else:
                combined_data[..., idx] = src.read()

    for idx, input_file in enumerate(input_files_prec):
        with rasterio.open(input_file) as src:
            if bands == 1:
                combined_data[..., idx] = combined_data[..., idx] | src.read(1)
            else:
                combined_data[..., idx] = combined_data[..., idx] | src.read()
    """
    if combined_data.ndim == 3:
        combined_data = np.sum(combined_data, axis=2, dtype=np.uint8) * (100 // combined_data.shape[2])
        dt.write_geotiff(os.path.dirname(out_file), os.path.basename(out_file), combined_data, extent=extent, dtype='int')
    else:
        combined_data = np.sum(combined_data, axis=3, dtype=np.uint8) * (100 // combined_data.shape[3])
        dt.write_geotiff(os.path.dirname(out_file), os.path.basename(out_file), combined_data.transpose(1, 2, 0), extent=extent, dtype='int')

def process_crops(crops, plant_params_formulas, climate_data_dir, plant_params, extent, irrigation, temperature_averages, precipitation, no_stops = 100):
    temp_lower_lst, temp_upper_lst, prec_lower_lst, prec_upper_lst = np.zeros(len(crops)), np.zeros(len(crops)), np.zeros(len(crops)), np.zeros(len(crops))
    growing_cycles = np.zeros(len(crops), dtype=np.uint16)

    for idx, crop in enumerate(crops):
        print(crop.capitalize())
        temp_lower_lst[idx], temp_upper_lst[idx] = get_limits(plant_params_formulas[crop], 'temp', plant_params[crop], no_stops)
        prec_lower_lst[idx], prec_upper_lst[idx] = get_limits(plant_params_formulas[crop], 'prec', plant_params[crop], no_stops)
        growing_cycles[idx] = int(plant_params[crop]['growing_cycle'][0])

    temp_data = find_tif_files(temperature_averages)
    prec_data = find_tif_files(precipitation)
    
    if len(temp_data) != len(prec_data):
        throw_exit_error(f'Number of data sets for temperature are unequal to those for precipitation: {len(temp_data)} vs {len(prec_data)}')

    cols, rows, _ = get_tif_dimensions(temp_data[0][0])

    years = np.unique([str(ds)[-14:-10] for ds in temp_data[0]]).tolist()   
    factor = 1
    
    year_args_list = [(year, rows, cols, growing_cycles, temp_data, prec_data, temp_lower_lst,
                    temp_upper_lst, prec_lower_lst, prec_upper_lst, extent, climate_data_dir, factor, idx, crops, irrigation)
                    for idx, year in enumerate(years)]
    
    cores, memory = get_cpu_ram()
    # 1 Worker Needs 15 GB RAM
    no_workers = int(np.min([cores-1, memory//15000]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=no_workers) as executor:
        executor.map(process_single_year_parallel, year_args_list)

    """
    # DEBUG
    for idx, year in enumerate(years):
        process_single_year_optimized(year, rows, cols, growing_cycles, temp_data, prec_data, temp_lower_lst,\
                            temp_upper_lst, prec_lower_lst, prec_upper_lst, extent, climate_data_dir, factor, idx, crops, irrigation)
    """
    
    for crop in crops:
        out_file = os.path.join(climate_data_dir, f'Ex_{crop}_{'ir' if irrigation else 'rf'}.tif')
        filelst_temp = find_matching_tif_files(climate_data_dir, f'Ex_temp_{crop}_')
        filelst_prec = find_matching_tif_files(climate_data_dir, f'Ex_prec_{crop}_')
        if not os.path.exists(out_file):
            if len(filelst_temp) >= 1:
                merge_boolean_files(filelst_temp, filelst_prec, out_file, extent)
        else:
            print(f' -> {os.path.basename(out_file)} already existing. Skipping.')
        for fn in filelst_prec+filelst_temp:
            os.remove(str(fn))
        print(f' -> {os.path.basename(out_file)} created')


def create_clim_extremes(temp_dir, prec_dir):
    config = rci.read_ini_file('config.ini')
    plant_params = rpp.read_crop_parameterizations_files(config['files']['plant_param_dir'])
    plant_params_formulas = rpp.get_plant_param_interp_forms_dict(plant_params, config)
    climate_data_dir = config['files']['climate_data_dir']
    irrigation = bool(int(config['options']['irrigation']))
    extent = get_geotiff_extent(os.path.join(temp_dir, os.listdir(temp_dir)[0]))
    crops = [crop for crop in plant_params_formulas]
    process_crops(crops, plant_params_formulas, climate_data_dir, plant_params, extent, irrigation, temp_dir, prec_dir, )


if __name__ == '__main__':
    startup()
    tic()
    config = rci.read_ini_file('config.ini')
    out_path = config['files']['climate_data_dir']
    orig_temp_min = config['files']['orig_temp_min_data']
    orig_temp_max = config['files']['orig_temp_max_data']
    orig_prec = config['files']['orig_prec_data']
    print('Calculating Temperature Means')   
    average_temp = get_temp_dail_avg(orig_temp_min, orig_temp_max, os.path.join(out_path, 'temp_avg'))
    temp_file = get_temp_average_file(average_temp, out_path)
    print('Temperature Means Created. Continuing with Precipitation')
    prec_file = get_prec_average(orig_prec, out_path)
    print('Means Created!')
    create_clim_extremes(average_temp, orig_prec)
    for f in os.listdir(average_temp):
        os.remove(f)
    os.rmdir(average_temp)
    toc()
    input('Completed!')
    exit()