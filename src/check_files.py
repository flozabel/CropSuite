import os
import rasterio
import numpy as np
import sys
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS
try:
    import nc_tools as nc
except:
    from src import nc_tools as nc

def get_soil_name(dirn) -> str:
    """
    Get the human-readable name for a soil parameter based on its directory code.

    Parameters:
    - dirn (str): Soil parameter directory code.

    Returns:
    - str: Human-readable name of the soil parameter.

    Note:
    - The function maps soil parameter directory codes to their corresponding human-readable names.
    """
    dict = {'bsat': 'Base Saturation',
            'cfvo': 'Coarse Fragments',
            'clay': 'Clay Content',
            'gyps': 'Gypsum',
            'ph': 'Soil pH',
            'sal': 'Electric Conductivity/Salinity',
            'sand': 'Sand Content',
            'slope': 'Slope',
            'soc': 'Soil Organic Carbon Content',
            'sod': 'Sodicity',
            'soildepth': 'Soildepth'           
            }
    
    if dirn not in dict:
        return dirn
    else:
        return dict[dirn]


def throw_exit_error(text):
    """
    Display an error message, prompt for input, and exit the program.

    This function prints the specified error message, prompts the user to press Enter, 
    and then exits the program using sys.exit().

    Parameters:
    - text (str): The error message to be displayed.

    Note:
    - The function is designed to inform the user about an error, request acknowledgment, 
      and terminate the program.
    """
    input(text+'\nExit')
    sys.exit()


def print_extent(extent):
    """
    Print a graphical representation of geographic extent.

    This function takes a geographic extent in the format [min_latitude, min_longitude, max_latitude, max_longitude]
    and prints a graphical representation of the extent.

    Parameters:
    - extent (list): Geographic extent in the format [min_latitude, min_longitude, max_latitude, max_longitude].

    Note:
    - The function represents the geographic extent with compass directions (N, S, E, W) and a simple grid.
    """
    n = str('%.2f' % abs(extent[0])) + ' S' if extent[0] < 0 else str('%.2f' % abs(extent[0])) + ' N'
    s = str('%.2f' % abs(extent[2])) + ' S' if extent[2] < 0 else str('%.2f' % abs(extent[2])) + ' N'
    w = str('%.2f' % abs(extent[1])) + ' W' if extent[1] < 0 else str('%.2f' % abs(extent[1])) + ' E'
    e = str('%.2f' % abs(extent[3])) + ' W' if extent[3] < 0 else str('%.2f' % abs(extent[3])) + ' E'

    max_label_width = max(len(w), len(e))
    padding = " " * (max_label_width - len(w))

    output = f'''
              {n}
            x---------x
            |         |
    {w} |         | {e}
            |         |
            x---------x
              {s}
    '''

    print(output)


def get_tif_dimensions(file_path) -> tuple:
    """
    Retrieve the dimensions (width, height, and band count) of a GeoTIFF file.

    This function opens a GeoTIFF file using rasterio and retrieves its width, height, and band count.

    Parameters:
    - file_path (str): The path to the GeoTIFF file.

    Returns:
    - tuple: A tuple containing the width, height, and band count of the GeoTIFF file.

    Note:
    - If an error occurs during the process, the function prints an error message and exits the program.
    """
    try:
        with rasterio.open(file_path) as dataset:
            width = dataset.width
            height = dataset.height
            count = dataset.count
            return width, height, count
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving dimensions of {file_path}: {e}')
        return 0, 0, 0


def get_geotiff_extent(file_path) -> tuple:
    """
    Retrieve the geographic extent (bounding box) of a GeoTIFF file.

    This function opens a GeoTIFF file using rasterio and retrieves its geographic extent.

    Parameters:
    - file_path (str): The path to the GeoTIFF file.

    Returns:
    - tuple: A tuple containing the bounding box coordinates (left, bottom, right, top).

    Note:
    - If an error occurs during the process, the function prints an error message and exits the program.
    """
    try:
        with rasterio.open(file_path) as dataset:
            bounds = dataset.bounds
            return bounds
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving extent of {file_path}: {e}')
        return 0, 0, 0, 0


def get_geotiff_resolution(file_path) -> tuple:
    """
    Retrieve the resolution (pixel size) of a GeoTIFF file.

    This function opens a GeoTIFF file using rasterio and retrieves its pixel size along the X and Y axes.

    Parameters:
    - file_path (str): The path to the GeoTIFF file.

    Returns:
    - tuple: A tuple containing the X and Y resolution of the GeoTIFF file.

    Note:
    - If an error occurs during the process, the function prints an error message and exits the program.
    """
    try:
        with rasterio.open(file_path) as dataset:
            x_res, y_res = dataset.res
            return x_res, y_res
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving resolution of {file_path}: {e}')
        return 0, 0


def get_geotiff_datatype(file_path) -> str:
    """
    Retrieve the data type of the first band in a GeoTIFF file.

    This function opens a GeoTIFF file using rasterio and retrieves the data type of the first band.

    Parameters:
    - file_path (str): The path to the GeoTIFF file.

    Returns:
    - str: The data type of the first band in the GeoTIFF file.

    Note:
    - If an error occurs during the process, the function prints an error message and exits the program.
    """
    try:
        with rasterio.open(file_path) as dataset:
            dtype = dataset.dtypes[0]
            return dtype
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving the data type of {file_path}: {e}')
        return ''


def get_geotiff_stats(file_path) -> list:
    """
    Retrieve basic statistics from the valid data in a GeoTIFF file.

    This function opens a GeoTIFF file using rasterio and calculates statistics such as minimum, mean, median,
    maximum, and the count of NaN values from the valid data (where mask is 255).

    Parameters:
    - file_path (str): The path to the GeoTIFF file.

    Returns:
    - list: A list containing the minimum, mean, median, maximum, and count of NaN values.

        Note:
    - If an error occurs during the process, the function prints an error message and exits the program.
    """
    try:
        with rasterio.open(file_path) as dataset:
            data = dataset.read(1)
            mask = dataset.read_masks(1)
            valid_data = data[mask == 255]
            stats = [np.nanmin(valid_data), np.nanmean(valid_data), np.nanmedian(valid_data), np.nanmax(valid_data), np.sum(np.isnan(data))]
            return stats
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving the statistics of {file_path}: {e}')
        return [0, 0, 0, 0, 0]


def get_geotiff_spatial_reference(file_path) -> str:
    """
    Retrieve the spatial reference of a GeoTIFF file.

    This function opens a GeoTIFF file using rasterio and retrieves the spatial reference in WKT format.

    Parameters:
    - file_path (str): The path to the GeoTIFF file.

    Returns:
    - str: The spatial reference of the GeoTIFF file in WKT format.

    Note:
    - If an error occurs during the process, the function prints an error message and exits the program.
    """
    try:
        with rasterio.open(file_path) as dataset:
            spatial_ref = dataset.crs.to_string()
            return spatial_ref
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving the spatial reference of {file_path}: {e}')
        return ''


def check_within_one(list1, list2) -> bool:
    """
    Check if corresponding elements of two lists are within one unit of difference.

    This function compares corresponding elements of two lists and returns True if the absolute
    difference between each pair of elements is at most one. Otherwise, it returns False.

    Parameters:
    - list1 (list): The first list for comparison.
    - list2 (list): The second list for comparison.

    Returns:
    - bool: True if corresponding elements are within one unit of difference, False otherwise.

    Note:
    - The function assumes that both lists have the same length. If the lengths are different,
      the function returns False.
    """
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if abs(list1[i] - list2[i]) > 1:
            return False
    return True


def get_minimum_extent(extent1, extent2) -> list:
    """
    Get the minimum extent that covers both input extents.

    This function takes two extents represented as [xmin, xmax, ymin, ymax] and returns a new
    extent that covers the minimum area required to encompass both input extents.

    Parameters:
    - extent1 (list): The first extent [xmin, xmax, ymin, ymax].
    - extent2 (list): The second extent [xmin, xmax, ymin, ymax].

    Returns:
    - list: The minimum extent covering both input extents.

    Note:
    - The function assumes that both extents are represented as [xmin, xmax, ymin, ymax].
    """
    return([np.min([extent1[0], extent2[0]]), np.max([extent1[1], extent2[1]]), np.max([extent1[2], extent2[2]]), np.min([extent1[3], extent2[3]])])


def swapPositions(list, pos1, pos2) -> list:
    """
    Swap the positions of elements at the specified indices in the input list.

    Parameters:
    - input_list (list): The input list.
    - pos1 (int): Index of the first element to swap.
    - pos2 (int): Index of the second element to swap.

    Returns:
    - list: The list with elements at pos1 and pos2 swapped.

    Note:
    - This function modifies the input list in-place.
    """
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list


def check_climate_data(options: dict) -> list:
    """
    Check climate data consistency and return the minimum extent of temperature and precipitation data.

    Parameters:
    - options (dict): Configuration options including file paths.

    Returns:
    - list: Minimum extent [min_lon, min_lat, max_lon, max_lat] of temperature and precipitation data.

    Note:
    - This function assumes that the temperature data is stored in 'Temp_avg.tif' and precipitation data in 'Prec_avg.tif'.
    - It checks if the resolutions, dimensions, and time slices match and returns the minimum extent.
    - It uses the provided helper functions: `get_tif_dimensions`, `get_geotiff_resolution`, `get_geotiff_extent`, `swap_positions`, and `get_minimum_extent`.
    - If there is an inconsistency, it exits with an error message.
    """
    clim_data_dir = options['files']['climate_data_dir']
    temp = os.path.join(clim_data_dir, 'Temp_avg.tif')
    prec = os.path.join(clim_data_dir, 'Prec_avg.tif')
    print('Climate Data')
    if os.path.exists(temp.replace('.tif', '.nc')) and os.path.exists(prec.replace('.tif', '.nc')):
        temp, prec = temp.replace('.tif', '.nc'), prec.replace('.tif', '.nc')
        temp_dims = nc.get_rows_cols(temp)
        prec_dims = nc.get_rows_cols(prec)

    else:
        temp_dims = get_tif_dimensions(temp)
        prec_dims = get_tif_dimensions(prec)
        x_res, y_res = get_geotiff_resolution(temp)
        print('   X-Resolution: {:.5f}'.format(x_res))
        print('   Y-Resolution: {:.5f}'.format(y_res))
        print(f'   Number of Time Slices: {temp_dims[2]}')

    print(f'   Number of Rows: {temp_dims[1]}')
    print(f'   Number of Columns: {temp_dims[0]}')

    if os.path.exists(temp.replace('.tif', '.nc')) and os.path.exists(prec.replace('.tif', '.nc')):
        temp, prec = temp.replace('.tif', '.nc'), prec.replace('.tif', '.nc')
        temp_extent = nc.get_netcdf_extent(temp)
        prec_extent = nc.get_netcdf_extent(prec)
        min_extent = get_minimum_extent(list(temp_extent.values()), list(prec_extent.values()))       
        return [min_extent[0], min_extent[2], min_extent[1], min_extent[3]]
    else:
        if check_within_one(temp_dims[:-1], prec_dims[:-1]) and (temp_dims[-1:] == prec_dims[-1:]):
            temp_extent = np.asarray(get_geotiff_extent(temp))
            prec_extent = np.asarray(get_geotiff_extent(prec))
            min_extent = get_minimum_extent(temp_extent, prec_extent)
            return min_extent
        else:
            throw_exit_error('Error: Temp_avg.tif and Prec_avg.tif do not have the same shape.')
            return []


def filter_list_by_ending(list, ending=('.tif', '.tiff')) -> list:
    """
    Filter a list of filenames based on the specified file endings.

    Parameters:
    - input_list (list): List of filenames.
    - ending (tuple): Tuple of file endings to filter.

    Returns:
    - list: Filtered list containing only filenames with the specified endings.

    Note:
    - This function filters filenames based on their endings, which is useful for selecting specific types of files.
    """
    return [entry for entry in list if entry.endswith(ending)]


def is_geotiff_readable(file_path) -> bool:
    """
    Check if a GeoTiff file is readable.

    Parameters:
    - file_path (str): Path to the GeoTiff file.

    Returns:
    - bool: True if the GeoTiff file is readable, False otherwise.

    Note:
    - This function attempts to open the GeoTiff file using rasterio to check if it is readable without raising an exception.
    - If the file is not readable, it prints an error message and returns False.
    """
    try:
        with rasterio.open(file_path) as src:
            _ = src.bounds
        return True
    except Exception as e:
        throw_exit_error(f'Error: {file_path} is no valid GeoTiff File')
        return False


def check_soil(path, weighting_method = 0) -> list:
    """
    Check soil GeoTiff files for validity and retrieve properties.

    Parameters:
    - path (str): Path to the soil GeoTiff file or directory.
    - weighting_method (int): Weighting method for combining multiple files. Default is 0.

    Returns:
    - soil_extent (list): Extent of the soil GeoTiff file.

    Note:
    - This function checks the validity of soil GeoTiff files, fetches their properties, and prints the information.
    - If the GeoTiff file is not readable or the properties cannot be fetched, an error message is printed, and the function returns None.
    """
    soil_datasets = []

    if not os.path.exists(path):
        throw_exit_error('Error: Path not existing:\n'+f'{path}')

    if os.path.isdir(path):
        for fn in os.listdir(path):
            if str(fn).lower().endswith('.tif') or str(fn).lower().endswith('.tiff'):
                soil_datasets.append(fn)
    elif os.path.isfile(path):
        if str(path).lower().endswith('.tif') or str(path).lower().endswith('.tiff'):
            soil_datasets.append(os.path.basename(path))
            path = os.path.dirname(path)
    
    if len(soil_datasets) == 0:
        throw_exit_error('Error: No valid geotiff files found\n'+f'Path: {path}')
    elif len(soil_datasets) == 1 and weighting_method > 0:
        throw_exit_error(f'Error: Only one geotiff found, but weighting method {weighting_method} requires three or six files')
    elif len(soil_datasets) > 1 and len(soil_datasets) < 6 and weighting_method == 2:
        throw_exit_error(f'Error: {len(soil_datasets)} geotiff files found, but weighting method {weighting_method} requires six files')
    elif len(soil_datasets) < 3 and weighting_method == 1:
        throw_exit_error(f'Error: geotiff found, but weighting method {weighting_method} requires three')
    elif len(soil_datasets) >= 6 or\
        (len(soil_datasets) >= 1 and weighting_method == 0) or\
        (len(soil_datasets) >= 3 and weighting_method == 1) or\
        (len(soil_datasets) >= 6 and weighting_method == 2):
        pass
    else:
        throw_exit_error('Error: Number of Geotiff files and Weighting Method does not match or Weighting Method is unknown.')

    if len(soil_datasets) > 1:
        soil_datasets = sorted(soil_datasets, key=lambda x: int(str(x).split('_')[1].split('-')[0]))

    soil_datasets = [os.path.join(path, soil_datasets[i]) for i in range(len(soil_datasets))]
    soil_dataset = soil_datasets[0]

    if not os.path.exists(soil_dataset):
        throw_exit_error(f'Error: {soil_dataset} is not existing')

    if is_geotiff_readable(soil_dataset):
        try:
            soil_extent = swapPositions(np.asarray(get_geotiff_extent(soil_dataset)), 2, 3)
            soil_dims = get_tif_dimensions(soil_dataset)
            x_res, y_res = get_geotiff_resolution(soil_dataset)
            dtype = get_geotiff_datatype(soil_dataset)
            crs = get_geotiff_spatial_reference(soil_dataset)
        except:
            soil_extent = [0, 0, 0, 0]
            soil_dims = [0, 0]
            x_res, y_res = 0, 0
            dtype = None
            crs = None
            throw_exit_error('Error: Invalid GeoTIFF data or unable to fetch raster properties\n'+f'{soil_dataset}')

        print(f'{get_soil_name(os.path.basename(os.path.dirname(soil_dataset)))} data')
        print('   Selected Files: ')
        if weighting_method == 0:
            print(f'      {soil_dataset}')
        else:
            for i in range(len(soil_datasets)):
                print(f'      {soil_datasets[i]}')
        print('   X-Resolution: {:.5f}'.format(x_res))
        print('   Y-Resolution: {:.5f}'.format(y_res))
        print(f'   Number of Rows: {soil_dims[1]}')
        print(f'   Number of Columns: {soil_dims[0]}')
        print(f'   Data Type: {dtype}')
        print(f'   Spatial Reference: {crs}')

        if crs != 'EPSG:4326':
            throw_exit_error(f'Error: The spatial reference system for the {soil_dataset} must be EPSG:4326 (World Geodetic System 1984, WGS1984)')

        print(f'   -> {get_soil_name(os.path.basename(os.path.dirname(soil_dataset)))} verified')
        return soil_extent
    else:
        return []

    

def calculate_area(extent) -> float:
    """
    Calculate the area based on the given extent.

    Parameters:
    - extent (list): Extent coordinates in the format [min_lon, min_lat, max_lon, max_lat].

    Returns:
    - area (float): Calculated area in square units.

    Note:
    - The extent should be in the format [min_lon, min_lat, max_lon, max_lat].
    - If the extent is not valid (min_lon >= max_lon or min_lat >= max_lat), the function returns 0.
    """
    if extent[0] <= extent[2] or extent[3] <= extent[1]:
        return 0
    return (extent[3] - extent[1]) * (extent[0] - extent[2])


def print_settings(options: dict):
    """
    Print the selected settings from the options dictionary.

    Parameters:
    - options (dict): Dictionary containing the configuration settings.
    """
    print('Selected settings')
    print('   Paths')
    print(f'   Output Path: {options["files"]["output_dir"]}')
    print(f'   Climate Data Path: {options["files"]["climate_data_dir"]}')
    print(f'   Plant Parameterization Path: {options["files"]["plant_param_dir"]}')
    print('')
    print('   Settings')
    print(f'   Irrigation: {options["options"]["irrigation"]}')
    print(f'   Output Format: {options["options"]["output_format"]}')
    print(f'   Output times as Day of the Year (doy) or Week of the Year (woy): {options["options"]["output_grow_cycle_as_doy"]}')
    print(f'   Downscaling: Moving Window Size: {options["options"]["downscaling_window_size"]}')
    print(f'   Downscaling: Use Temperature Gradient: {options["options"]["downscaling_use_temperature_gradient"]}')
    print(f'   Downscaling: Dryadiabatic Gradient: {options["options"]["downscaling_dryadiabatic_gradient"]}')
    print(f'   Downscaling: Saturation Adiabatic Gradient: {options["options"]["downscaling_saturation_adiabatic_gradient"]}')
    print(f'   Downscaling: Temperature BIAS Threshold: {options["options"]["downscaling_temperature_bias_threshold"]}')
    print(f'   Downscaling: Precipitation BIAS Threshold: {options["options"]["downscaling_precipitation_bias_threshold"]}')
    print(f'   Downscaling: Daily Precipitation Threshold: {options["options"]["downscaling_precipitation_per_day_threshold"]}')
    print('')
    print('   Membership Functions')
    print(f'   Plot all Membership Functions: {options["membershipfunctions"]["plot_for_each_crop"]}')


def get_id_list_start(dict, starts_with) -> list:
    """
    Get a list of keys from a dictionary that start with a specified prefix.

    Parameters:
    - dictionary (dict): The input dictionary.
    - starts_with (str): The prefix to match the keys.

    Returns:
    - list: A list of keys from the dictionary that start with the specified prefix.
    """
    lst = []
    for id, __ in dict.items():
        if str(id).startswith(starts_with):
            lst.append(id)
    return lst


def reproject_geotiff(input_file):
    with rasterio.open(input_file) as src:
        current_crs = src.crs
        target_crs = CRS.from_epsg(4326)
        if current_crs.to_string() != target_crs.to_string():
            output_file = os.path.splitext(input_file)[0] + "-wgs.tif"
            print(f"Reprojecting {os.path.basename(input_file)} from {current_crs} to EPSG:4326 (WGS 1984 / Plate Carree).")
            transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({'crs': target_crs, 'transform': transform, 'width': width, 'height': height, 'compress': 'lzw'})
            with rasterio.open(output_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i), src_transform=src.transform,
                        src_crs=src.crs, dst_transform=transform, dst_crs=target_crs, resampling=Resampling.nearest)
            print(f"Reprojection complete. Saved to {output_file}")

            os.remove(input_file)
            os.rename(output_file, input_file)


def check_all_inputs(options: dict) -> list:
    """
    Check and verify all input data specified in the configuration.

    Parameters:
    - options (dict): The configuration dictionary.

    Returns:
    - list: The verified minimum extent that covers all specified input data.
    """
    spec_extent = [float(options['extent']['upper_left_y']), float(options['extent']['upper_left_x']),\
              float(options['extent']['lower_right_y']), float(options['extent']['lower_right_x'])]
    
    print_settings(options)
    print('\nExtent specified in the config file')
    print_extent(extent=spec_extent)
    climate_extent = check_climate_data(options)
    climate_extent = [climate_extent[3], climate_extent[0], climate_extent[1], climate_extent[2]]
    min_extent = get_minimum_extent(spec_extent, climate_extent)

    # DGM, Slope, LandSeaMask
    soil_datasets = [entry.replace('parameters.', '', 1) if entry.startswith('parameters.') else entry for entry in get_id_list_start(options, 'parameters.')]
    print('')
    
    for soil in soil_datasets: 
        soil_path = options[f'parameters.{soil}']['data_directory']
        try:
            wm = int(options[f'parameters.{soil}']['weighting_method'])
        except:
            wm = 0
        
        soil_extent = check_soil(soil_path, weighting_method = wm)
        soil_extent = [soil_extent[2], soil_extent[0], soil_extent[1], soil_extent[3]]
        min_extent = get_minimum_extent(soil_extent, min_extent)
  
        for tif_file in [f for f in os.listdir(soil_path) if f.endswith('.tif') or f.endswith('.tiff')]:
            try:
                reproject_geotiff(os.path.join(soil_path, tif_file))
            except Exception as e:
                print(f'Error: Unable reprojecting {tif_file} - {e}')
        print('')

    try:
        reproject_geotiff(options['files'].get('fine_dem'))
    except Exception as e:
        print(f'Error: Unable reprojecting {tif_file} - {e}')

    try:
        reproject_geotiff(options['files'].get('land_sea_mask'))
    except Exception as e:
        print(f'Error: Unable reprojecting {tif_file} - {e}')

    if list((np.asarray(min_extent)*100).astype(int)/100) != list((np.asarray(spec_extent)*100).astype(int)/100):
        print('The specified extent cannot be created.\nData is only available for the following extent:')
        print_extent(extent=min_extent)
        #input('Continue anyway?')
    elif calculate_area(min_extent)<.01:
        throw_exit_error('The specified extent does not span any area.')
    else:
        print('All input data successfully verified')
    return min_extent

    

