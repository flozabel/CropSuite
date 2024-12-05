import os
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
import re
import numpy as np
import shutil
import sys
import data_tools as dt
import nc_tools as nc
from datetime import datetime


def process_geotiff(file_path):
    """
    Reads a GeoTIFF, fills all-NaN lines in the data, and writes it back to the same file.

    Parameters:
        file_path (str): Path to the GeoTIFF file to process.
    """

    with rasterio.open(file_path) as src:
        data = src.read()
        profile = src.profile
        data[data == src.nodata] = np.nan

    invalid_mask = np.all(np.isnan(data[0, :]) | (data[0, :] == 0), axis=1)
    to_delete = []
    for i in range(1, data.shape[1] - 1):
        if invalid_mask[i]:
            if not (invalid_mask[i + 1] or invalid_mask[i - 1]):
                to_delete.append(i)
    data = np.delete(data, to_delete, axis=1)

    """

    for i in range(data.shape[1]):
        if i == 5000:
            print(i)
        if np.all(np.isnan(data[0, i]) | (data[0, i] == 0)):
            if not np.all(np.isnan(data[0, i+1]) | (data[0, i+1 ] == 0)) and not np.all(np.isnan(data[0, i-1]) | (data[0, i-1 ] == 0)):
                data = np.delete(data, i, axis=1)
    """

    profile.update({'height': data.shape[1], 'width': data.shape[2], 'count': data.shape[0]})
    with rasterio.open(file_path, 'w', **profile) as dst:
        dst.write(data)


def get_num_bands(tif_file) -> int:
    with rasterio.open(tif_file, 'r') as src:
        return int(src.count)


def get_nodata_value(tif_file):
    with rasterio.open(tif_file, 'r') as src:
        return src.nodatavals[0]


def merge_outputs_no_overlap(results_path, config):
    print('Start Merging GeoTiffs')
    print(f' -> File Path: {results_path}')
    filenames = ['crop_limiting_factor.tif', 'crop_suitability.tif', 'multiple_cropping.tif', 'optimal_sowing_date.tif', 'suitable_sowing_days.tif', 
                 'climate_suitability.tif', 'multiple_cropping_sum.tif', 'optimal_sowing_date_mc_first.tif', 'optimal_sowing_date_mc_second.tif', 
                 'optimal_sowing_date_mc_third.tif', 'climate_suitability_mc.tif', 'all_suitability_vals.tif', 'optimal_sowing_date_vernalization.tif']
    areas = [d for d in next(os.walk(results_path))[1] if d.startswith('Area_')]
    print(' -> Found Areas: ')
    for area in areas:
        print(f'    * {area}')

    north_values = [int(value[:-1]) for item in areas for value in re.findall(r'(-?\d+N)', item)]
    east_values = [int(value[:-1]) for item in areas for value in re.findall(r'(-?\d+E)', item)]
    merged_result = os.path.join(results_path, f'Area_{max(north_values)}N{min(east_values)}E-{min(north_values)}N{max(east_values)}E')
    if os.path.isdir(merged_result):
        shutil.rmtree(merged_result)

    print(f' -> Output Directory: {merged_result}')
    os.makedirs(merged_result, exist_ok=True)

    soil_layers = [f for f in os.listdir(os.path.join(results_path, areas[0])) if f.endswith('.tif')]

    for soil_layer in soil_layers:
        directories = [os.path.join(results_path, area) for area in areas]
        tif_files = [file for file in [os.path.join(directory, soil_layer) for directory in directories] if os.path.exists(file)]
        nc_files = [file for file in [os.path.join(directory, soil_layer.replace('.tif', '.nc')) for directory in directories] if os.path.exists(file)]
        if len(tif_files) < 1 and len(nc_files) < 1:
            continue
        output_file = os.path.join(merged_result, soil_layer) if len(tif_files) > 1 else os.path.join(merged_result, soil_layer.replace('.tif', '.nc'))
        if len(tif_files) > 1:
            src_files_to_mosaic = []
            for fp in tif_files:
                src = rasterio.open(fp, 'r')
                src_files_to_mosaic.append(src)
            mosaic, out_trans = merge(src_files_to_mosaic)
            out_meta = src.meta.copy() # type: ignore
            out_meta.update({'driver': 'GTiff', 'height': mosaic.shape[1], 'width': mosaic.shape[2], 'transform': out_trans, 'compress': 'lzw'})
            try:
                with rasterio.open(output_file, 'w', **out_meta) as dest:
                    dest.write(mosaic[0], 1)
            except:
                with rasterio.open(output_file, 'w', **out_meta, BIGTIFF='yes') as dest:
                    dest.write(mosaic[0], 1)
        if len(nc_files) > 1:
            nc.merge_netcdf_files(nc_files, output_file, nodata_value=-1) #type:ignore

        sys.stdout.write(f'{soil_layer} created!                       '+'\r')
        sys.stdout.flush()
    
    crops = ['']
    for directory in [os.path.join(results_path, area) for area in areas][1:]:
        crops = set(next(os.walk([os.path.join(results_path, area) for area in areas][0]))[1])
    crops = list(crops)

    print(' -> Found Crops:')
    for crop in crops:
        print(f'    * {crop}')
    print('')

    for crop in crops:
        directories = [os.path.join(results_path, area, crop) for area in areas]
        output_dir = os.path.join(merged_result, crop)
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in filenames:
            tif_files = [file for file in [os.path.join(directory, filename) for directory in directories] if os.path.exists(file)]
            nc_files = [file for file in [os.path.join(directory, filename.replace('.tif', '.nc')) for directory in directories] if os.path.exists(file)]
            if len(tif_files) < 1 and len(nc_files) < 1:
                continue
            output_file = os.path.join(output_dir, filename) if len(tif_files) > 1 else os.path.join(output_dir, filename.replace('.tif', '.nc'))
            if len(tif_files) > 1:
                if not os.path.exists(tif_files[0]):
                    continue
                
                if get_num_bands(tif_files[0]) == 1:
                    src_files_to_mosaic = []
                    for fp in tif_files:
                        src = rasterio.open(fp)
                        src_files_to_mosaic.append(src)

                    mosaic, out_trans = merge(src_files_to_mosaic)
                    out_meta = src.meta.copy() # type: ignore
                    out_meta.update({'driver': 'GTiff', 'height': mosaic.shape[1], 'width': mosaic.shape[2], 'transform': out_trans, 'compress': 'lzw'})
                    try:
                        with rasterio.open(output_file, 'w', **out_meta) as dest:
                            dest.write(mosaic[0], 1)
                    except:
                        with rasterio.open(output_file, 'w', **out_meta, BIGTIFF='yes') as dest:
                            dest.write(mosaic[0], 1)

                else:
                    src_files_to_mosaic = [rasterio.open(file) for file in tif_files]
                    mosaic, out_trans = merge(src_files_to_mosaic)
                    out_meta = src_files_to_mosaic[0].meta.copy()
                    out_meta.update({'driver': 'GTiff',
                                    'height': mosaic.shape[1],
                                    'width': mosaic.shape[2],
                                    'transform': out_trans,
                                    'compress': 'lzw'})
                    try:
                        with rasterio.open(output_file, 'w', **out_meta) as dest:
                            dest.write(mosaic)
                    except:
                        with rasterio.open(output_file, 'w', **out_meta, BIGTIFF='yes') as dest:
                            dest.write(mosaic)

                if config['options']['output_format'].lower() == 'cog':
                    dt.create_cog_from_geotiff(output_file, output_file.replace('.tif', '_cog.tif'))

                if os.path.exists(output_file):
                    process_geotiff(output_file)
            
            if len(nc_files) > 1:
                nc.merge_netcdf_files(nc_files, output_file, nodata_value=-1) #type:ignore


            sys.stdout.write(f'{crop} {filename} created!                       '+'\r')
            sys.stdout.flush()

def merge_outputs(results_path):
    print('Start Merging GeoTiffs')
    print(f' -> File Path: {results_path}')
    filenames = ['crop_limiting_factor.tif', 'crop_suitability.tif', 'multiple_cropping.tif', 'optimal_sowing_date.tif', 'suitable_sowing_days.tif', 
                 'climate_suitability.tif', 'multiple_cropping_sum.tif', 'optimal_sowing_date_first.tif', 'optimal_sowing_date_second.tif', 
                 'optimal_sowing_date_third.tif', 'climate_suitability_mc.tif', 'soil_suitability.tif', 'optimal_sowing_date_vernalization.tif']
    ptr = 25
    areas = [d for d in next(os.walk(results_path))[1] if d.startswith('Area_')]
    
    print(' -> Found Areas: ')
    for area in areas:
        print(f'    * {area}')

    north_values = [int(value[:-1]) for item in areas for value in re.findall(r'(-?\d+N)', item)]
    east_values = [int(value[:-1]) for item in areas for value in re.findall(r'(-?\d+E)', item)]
    merged_result = os.path.join(results_path, f'Area_{max(north_values)}N{min(east_values)}E-{min(north_values)}N{max(east_values)}E')
    if os.path.isdir(merged_result):
        shutil.rmtree(merged_result)

    print(f' -> Output Directory: {merged_result}')

    crops = ['']
    for directory in [os.path.join(results_path, area) for area in areas][1:]:
        crops = set(next(os.walk([os.path.join(results_path, area) for area in areas][0]))[1]).intersection(set(next(os.walk(directory))[1]))
    crops = list(crops)

    print(' -> Found Crops:')
    for crop in crops:
        print(f'    * {crop}')

    for crop in crops:
        directories = [os.path.join(results_path, area, crop) for area in areas]
        output_dir = os.path.join(merged_result, crop)
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        for filename in filenames:
            tif_files = [file for file in [os.path.join(directory, filename) for directory in directories] if os.path.exists(file)]
            if len(tif_files) < 1:
                continue
            output_file = os.path.join(output_dir, filename)
            if not os.path.exists(tif_files[0]):
                continue
            
            crs = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
            src_files_to_mosaic = []
            for fp in tif_files:

                with rasterio.open(fp) as src:
                    height, width = src.height, src.width
                    dtype = src.dtypes[0]
                    x_res, y_res = src.res
                    left_new = src.bounds.left + (ptr * x_res)
                    right_new = src.bounds.right - (ptr * x_res)
                    top_new = src.bounds.top - (ptr * y_res)
                    bottom_new = src.bounds.bottom + (ptr * y_res)
                    data = src.read(1)[ptr:-ptr, ptr:-ptr]
                    height_new, width_new = height - (ptr * 2), width - (ptr * 2)
                    nodata_value = src.nodata
                os.remove(fp)
                transform_new = from_bounds(left_new, bottom_new, right_new, top_new, width_new, height_new)
                with rasterio.open(fp, 'w', driver='GTiff', height=height_new, width=width_new, count=1, crs=crs, transform=transform_new, nodata=int(nodata_value), dtype='int16') as dst:
                    dst.write(data.astype(np.int16), 1)
                src = rasterio.open(fp)
                src_files_to_mosaic.append(src)

            mosaic, out_trans = merge(src_files_to_mosaic)
            out_meta = src.meta.copy() # type: ignore
            out_meta.update({'driver': 'GTiff', 'height': mosaic.shape[1], 'width': mosaic.shape[2], 'transform': out_trans, 'compress': 'lzw'})
            with rasterio.open(output_file, 'w', **out_meta) as dest:
                dest.write(mosaic[0], 1)
            if os.path.exists(output_file):
                process_geotiff(output_file)
            print(f'{crop} {filename} created!')

    print('\n\nAll Output Files successfully merged!\n\n')

