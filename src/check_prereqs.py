import importlib

def check_libraries(required_libraries) -> bool:
    """
    Checks if a list of required libraries can be imported, and returns True if all libraries are available.
    If any library cannot be imported, an error message is printed listing the missing libraries.
    Args:
        required_libraries (list): A list of strings, where each string represents a required library.
    Returns:
        bool: True if all libraries are available, False otherwise.
    """
    missing_libraries = []
    for library in required_libraries:
        try:
            importlib.import_module(library)
        except ImportError:
            missing_libraries.append(library)

    if missing_libraries:
        error_msg = f"The following libraries are missing: {', '.join(missing_libraries)}."
        print(error_msg)
        return False
    else:
        return True
    

def is_directory_writable(directory_path) -> bool:
    """
    Check if a directory is writable by attempting to create a temporary file.

    Parameters:
    - directory_path (str): The path of the directory to be checked.

    Returns:
    - bool: True if the directory is writable, False otherwise.
    """
    import os
    try:
        os.makedirs(directory_path, exist_ok=True)
        temp_file_path = os.path.join(directory_path, 'temp_write_test.txt')
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write('This is a temporary file for write test.')
        os.remove(temp_file_path)
        os.rmdir(directory_path)
        return True
    except Exception as e:
        return False


# Kein Halt wenn Plant Param Dir/Climate Dir fehlt!

def check_config_file(conf, gui=False):
    """
    Check the validity of the configuration file.

    Parameters:
    - conf (dict): The configuration dictionary loaded from the configuration file.

    Returns:
    - None

    The function checks the specified paths, options, extent, and other necessary configurations.
    If any important entry is missing, it prints an error message and exits the program.
    """
    import os
    print('Checking Config File...')
    halt = False
    # Check File Paths
    if 'files' in conf:
        if not 'output_dir' in conf['files']:
            print(' • Output Directory not specified. Add "output_dir = /path/to/dir" to the FILES section')
            halt = False

        if 'climate_data_dir' in conf['files']:
            if not os.path.exists(os.path.join(conf['files']['climate_data_dir'])):
                print(' • Climate Data Directory specified but not found!')
                halt = False
            if not os.path.exists(os.path.join(conf['files']['climate_data_dir'], 'Temp_avg.tif')) and not os.path.exists(os.path.join(conf['files']['climate_data_dir'], 'Temp_avg.nc')):
                print(' • Climate data directory found but no usable Temp_avg.tif/.nc existing!')
                halt = False
            if not os.path.exists(os.path.join(conf['files']['climate_data_dir'], 'Prec_avg.tif')) and not os.path.exists(os.path.join(conf['files']['climate_data_dir'], 'Prec_avg.nc')):
                print(' • Climate data directory found but no usable Prec_avg.tif/.nc existing!')
                halt = False
        else:
            print(' • Climate data Directory not specified. Add "climate_data_dir = /path/to/dir" to the FILES section')
            halt = False
        
        if 'plant_param_dir' in conf['files']:
            if not os.path.exists(conf['files']['plant_param_dir']):
                print(' • Plant Parameterization Directory specified but not found!')
                halt = False
            elif len([fn for fn in os.listdir(conf['files']['plant_param_dir']) if fn.endswith('.inf')]) < 1:
                print(' • Plant Parameterization Directory specified but no parameterization .inf files found!')
                halt = False
        else:
            print(' • Plant Parameterization Directory not specified. Add "plant_param_dir = /path/to/dir" to the FILES section')
            halt = False

        if 'fine_dem' in conf['files']:
            if not os.path.exists(conf['files']['fine_dem']):
                print(' • Fine DEM specified but not found!')
                halt = False
        else:
            print(' • Fine DEM not specified. Add "fine_dem = /path/to/file" to the FILES section')
            halt = False
        
        if 'land_sea_mask' in conf['files']:
            if not os.path.exists(conf['files']['land_sea_mask']):
                print(' • Land-Sea-Mask specified but not found!')
                halt = False
        else:
            print(' • Land-Sea-Mask not specified. Add "land_sea_mask = /path/to/file" to the FILES section')
            halt = False
        
        if 'texture_classes' in conf['files']:
            if not os.path.exists(conf['files']['texture_classes']):
                print(' • Texture Classes Config specified but not found!')
                halt = False
        else:
            print(' • Texture Classes Config not specified. Add "texture_classes = /path/to/file" to the FILES section')
            halt = False
        
        if 'worldclim_precipitation_data_dir' in conf['files']:
            if not os.path.exists(conf['files']['worldclim_precipitation_data_dir']):
                print(' • WorldClim Data for Precipitation not found. Precipitation downscaling by using WorldClim data is not available!')
                halt = False
        else:
            print(' • WorldClim Precipitation Data Directory not specified. Add "worldclim_precipitation_data_dir = /path/to/directory" to the FILES section')
            halt = False
        if 'worldclim_temperature_data_dir' in conf['files']:
            if not os.path.exists(conf['files']['worldclim_temperature_data_dir']):
                print(' • WorldClim Data for Temperature not found. Temperature downscaling by using WorldClim data is not available!')
                halt = False
        else:
            print(' • WorldClim Temmperature Data Directory not specified. Add "worldclim_temperature_data_dir = /path/to/directory" to the FILES section')
            halt = False
        
    else:
        print(' • FILES Section of Config File is completely missing. Add "[Files]" Section to the config file. Refer to the manual.')

    # Check Options
    if 'options' in conf:
        if not 'use_scheduler' in conf['options']:
            print(' • Usage of Scheduler not specified. Add "use_scheduler = y / n" to the OPTIONS section')
            halt = True

        if not 'irrigation' in conf['options']:
            print(' • Irrigation or Rainfed conditions not specified. Add "irrigation = 0 / 1" to the OPTIONS section')
            halt = True
        
        if not 'precipitation_downscaling_method' in conf['options']:
            print(' • Downscaling method for precipitation not specified. Add "precipitation_downscaling_method = 0 / 1 / 2" to the OPTIONS section')
            halt = True
        
        if not 'temperature_downscaling_method' in conf['options']:
            print(' • Downscaling method for temperature not specified. Add "temperature_downscaling_method = 0 / 1 / 2 / 3" to the OPTIONS section')
            halt = True

        if not 'output_format' in conf['options']:
            print(' • Output Format not specified. Add "output_format = geotiff / netcdf" to the OPTIONS section')
            halt = True

        if not 'output_all_startdates' in conf['options']:
            print(' • Output All Startdates not specified. Add "output_all_startdates = y / n" to the OPTIONS section')
            halt = True
        
        if not 'output_grow_cycle_as_doy' in conf['options']:
            print(' • Output Startdate Format not specified. Add "output_grow_cycle_as_doy = y / n" to the OPTIONS section')
            halt = True

        if not 'downscaling_window_size' in conf['options']:
            print(' • Downscaling Window Size not specified. Add "downscaling_window_size = 0-100 (Default: 8)" to the OPTIONS section')
            halt = True
        
        if not 'downscaling_use_temperature_gradient' in conf['options']:
            print(' • Downscaling Temperature Gradient Checking not specified. Add "downscaling_use_temperature_gradient = y / n" to the OPTIONS section')
            halt = True

        if not 'downscaling_dryadiabatic_gradient' in conf['options']:
            print(' • Downscaling Dryadiabatic Temperature Gradient not specified. Add "downscaling_dryadiabatic_gradient = 0.001 - 0.01 (Default: 0.00976)" to the OPTIONS section')
            halt = True
        
        if not 'downscaling_saturation_adiabatic_gradient' in conf['options']:
            print(' • Downscaling Saturation Adiabatic Temperature Gradient not specified. Add "downscaling_saturation_adiabatic_gradient = 0.001 - 0.01 (Default: 0.007)" to the OPTIONS section')
            halt = True

        if not 'downscaling_temperature_bias_threshold' in conf['options']:
            print(' • Downscaling Temperature BIAS Threshold not specified. Add "downscaling_temperature_bias_threshold = 0.0001 - 0.01 (Default: 0.0005)" to the OPTIONS section')
            halt = True

        if not 'downscaling_precipitation_bias_threshold' in conf['options']:
            print(' • Downscaling Precipitation BIAS Threshold not specified. Add "downscaling_precipitation_bias_threshold = 0.0001 - 0.01 (Default: 0.0001)" to the OPTIONS section')
            halt = True

        if not 'downscaling_precipitation_per_day_threshold' in conf['options']:
            print(' • Drizzling Precipitation Threshold not specified. Add "downscaling_precipitation_per_day_threshold = 0.1 - 2.0 (Default: 0.5)" to the OPTIONS section')
            halt = True

        if not 'output_all_limiting_factors' in conf['options']:
            print(' • Output All Limiting Factors not specified. Add "output_all_limiting_factors = y / n" to the OPTIONS section')
            halt = True
        
        if not 'remove_interim_results' in conf['options']:
            print(' • Remove All Interim Results Option not specified. Add "remove_interim_results = y / n" to the OPTIONS section')
            halt = True

    else:
        print(' • OPTIONS Section of Config File is completely missing. Add "[Options]" Section to the config file. Refer to the manual.')
    
    # Check Extent
    if 'extent' in conf:
        if not 'upper_left_x' in conf['extent']:
            print(' • Upper Left Longitude Coordinate not specified. Add "upper_left_x = -180 - 180" to the EXTENT section')
            halt = True
        if not 'upper_left_y' in conf['extent']:
            print(' • Upper Left Latitude Coordinate not specified. Add "upper_left_y = -90 - 90" to the EXTENT section')
            halt = True
        if not 'lower_right_x' in conf['extent']:
            print(' • Upper Right Longitude Coordinate not specified. Add "lower_right_x = -180 - 180" to the EXTENT section')
            halt = True
        if not 'lower_right_y' in conf['extent']:
            print(' • Upper Right Latitude Coordinate not specified. Add "lower_right_y = -90 - 90" to the EXTENT section')
            halt = True
    else:
        print(' • EXTENT Section of Config File is completely missing. Add "[Extent]" Section to the config file. Refer to the manual.')

    # Check climatevariability
    if 'climatevariability' in conf:
        if not 'consider_variability' in conf['climatevariability']:
            print(' • Option to take climate variability into account not specified. Add "consider_variability = y / n" to the CLIMATEVARIABILITY section')
            halt = True
    
    if 'membershipfunctions' in conf:
        if not 'plot_for_each_crop' in conf['membershipfunctions']:
            print(' • Option to Plot Membership Functions for each Crop not specified. Add "plot_for_each_crop = y / n" to the MEMBERSHIPFUNCTIONS section')
            halt = True

    if not any(key.startswith('parameters.') for key in conf.keys()):
        print(' • No Parameter Sets defined. Add Parameter Files. Refer to the Manual.')
        halt = True

    halt=False 
    if halt:
        if gui:
            return False
        else:
            import sys
            input('Important entries are missing in the config file.\nPlease add the missing entries first!\nPress Enter to Exit')
            sys.exit()
    else:
        if gui:
            return True
        else:
            print('... Config File checked!')