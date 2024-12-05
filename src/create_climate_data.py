import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import rasterio
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning) 
import read_climate_ini as rci


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