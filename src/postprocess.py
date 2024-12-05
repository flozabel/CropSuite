import os
import rasterio 
import numpy as np
import sys

def get_geotiff_shape(geotiff_path) -> tuple:
    with rasterio.open(geotiff_path, 'r') as ds:
        return ds.height, ds.width


def list_tif_files(directory):
    tif_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif'):
                tif_files.append(os.path.join(root, file))    
    return tif_files


def fill_to_small(root_dir):
    print('\nCorrecting shape of sowing date files\n')
    for filename in list_tif_files(root_dir):
        if not filename.endswith('optimal_sowing_date_mc_third.tif') and not filename.endswith('optimal_sowing_date_mc_second.tif')\
            and not filename.endswith('optimal_sowing_date_mc_first.tif'):
            continue
        
        ref_file = os.path.join(os.path.dirname(filename), 'climate_suitability.tif')
        if get_geotiff_shape(filename) == get_geotiff_shape(ref_file):
            continue
        
        os.rename(filename, filename[:-4]+'_bak.tif')
        bak = filename[:-4]+'_bak.tif'
        
        #print(f' -> Processing {filename}')
        sys.stdout.write(f' -> Processing {filename} created!                       '+'\r')
        sys.stdout.flush()  
        with rasterio.open(ref_file, 'r') as first_ds:
            first_profile = first_ds.profile
            first_data = first_ds.read()
            first_data[first_data != first_ds.nodata] = 0
            
            with rasterio.open(bak, 'r') as second_ds:
                second_data = second_ds.read()
                window = first_ds.window(*second_ds.bounds)
                first_data[:, int(window.row_off):int(window.row_off + second_data.shape[1]), :] = second_data

                with rasterio.open(filename, 'w', **first_profile) as output_ds:
                    output_ds.write(first_data)
        os.remove(bak)
    sys.stdout.write(f'                                                          '+'\r')
    sys.stdout.flush()
    

def get_num_bands(tif_file) -> int:
    with rasterio.open(tif_file, 'r') as src:
        return int(src.count)


def adjust_datatype(root_dir):
    print('\nAdjusting Datatype to decrease file size\n')
    error_list = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.tif'):
                try:
                    crop = os.path.basename(subdir)
                    #print(f'-> Processing {crop} - {file}')
                    sys.stdout.write(f' -> Processing {crop} - {file}                        '+'\r')
                    sys.stdout.flush()
                    if get_num_bands(os.path.join(subdir, file)) == 1:
                        with rasterio.open(os.path.join(subdir, file), 'r') as src:
                            data = src.read(1)
                            dtype = src.dtypes[0]
                            transform = src.transform
                            width = src.width
                            height = src.height
                            crs = src.crs
                            nodata = src.nodata
                        dtype = np.int8 if np.nanmax(data) < 127 else np.int16
                        nodata = int(nodata)
                        data = data.astype(dtype)
                        os.remove(os.path.join(subdir, file))
                        with rasterio.open(os.path.join(subdir, file), 'w', driver='GTiff', height=height, width=width, count=1, dtype=dtype, crs=crs, transform=transform, compress='lzw', nodata=nodata) as dst:
                            dst.write(data, 1)
                    else:
                        with rasterio.open(os.path.join(subdir, file), 'r') as src:
                            data = src.read()
                            dtype = src.dtypes[0]
                            transform = src.transform
                            width = src.width
                            height = src.height
                            crs = src.crs
                            nodata = src.nodata
                        dtype = np.int8 if np.nanmax(data) < 127 else np.int16
                        data = data.astype(dtype)
                        os.remove(os.path.join(subdir, file))
                        with rasterio.open(os.path.join(subdir, file), 'w', driver='GTiff', height=height, width=width, count=data.shape[0], dtype=dtype, crs=crs, transform=transform, compress='lzw', nodata=nodata) as dst:
                            dst.write(data)
                except Exception as e:
                    error_list.append(f'Error: {crop} - {file} - {str(e)}')
    sys.stdout.write(f'                                                          '+'\r')
    sys.stdout.flush()
