import sys
import os
try:
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
except Exception as e:
    print(f"Failed to modify system path: {e}")
try:
    from src import check_prereqs as cp
except:
    import check_prereqs as cp
import time
import traceback

def run(silent_mode, config_file = None, gui=None):
    # Check required libraries
    req_libs = ['os', 'configparser', 'numpy', 'sys', 'multiprocessing', 'gc', 'numba', 'scipy', 'statistics',\
            'glob', 'rasterio', 'rio_cogeo', 'concurrent', 'psutil', 'matplotlib', 'math', 'xarray', 'numba', 'datetime']
    if not cp.check_libraries(req_libs):
        input('Exit')
        exit()
    else:
        if gui != None:
            gui.set_libraries_true()
            gui.update()
        print('\n All Required Libraries found\n')

    try:
        import os
        import sys
        sys.path.append(os.path.dirname(__file__))
        sys.path.append(os.getcwd())
        sys.path.append(os.path.join(os.getcwd(), 'src'))
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    except Exception as e:
        print(f"Failed to modify system path: {e}")
    
    try:
        from src import climate_suitability_main as csm
        from src import read_climate_ini as rci
        from src import read_plant_params as rpp
        from src import crop_suitability_main as crop_suit
        from src import check_files
        from src import data_tools as dt
        from src import merge_geotiff as mg
        from src import nc_tools as nc
        from src import downscaling as ds
    except:
        import climate_suitability_main as csm
        import read_climate_ini as rci
        import read_plant_params as rpp
        import crop_suitability_main as crop_suit
        import check_files
        import data_tools as dt
        import merge_geotiff as mg
        import nc_tools as nc
        import downscaling as ds        
    import numpy as np
    import math
    import shutil
    import gc
    import re

    print('''\
        
        =======================================================
        |                                                     |
        |                                                     |    
        |                      CropSuite                      |
        |                                                     |
        |                     Version  1.0                    |
        |                      2024-12-12                     |
        |                                                     |
        |                                                     |
        |                   Matthias Knüttel                  |
        |                     Florian Zabel                   |
        |                         2024                        |
        |                                                     |
        |                                                     |
        |         Departement of Environmental Sciences       |      
        |                 University of Basel                 |
        |                                                     |
        |                                                     |
        |              © 2024 All rights reserved             |
        |                                                     |
        =======================================================
        
        ''')

    # Read config_climate.ini into a dictionary
    if gui != None:
        config_file = gui.get_config_path()

    if config_file is None:
        if os.path.exists('config.ini'):
            if gui != None:
                gui.set_config_ini(r'.\config.ini')
                gui.update()
            climate_config = rci.read_ini_file('config.ini')
            if gui != None:
                gui.check_cfg_ini_true()
                gui.update()
        else:
            dt.throw_exit_error('Error: Config ini does not exist:\n'+str(os.path.join(os.getcwd(), 'config.ini')))
            exit(1)
    else:
        if os.path.exists(config_file):
            print(f'Custom Config File found at {config_file}')
            if gui != None:
                gui.set_config_ini(config_file)
                gui.update()
            print('')
            climate_config = rci.read_ini_file(config_file)
            if gui != None:
                gui.check_cfg_ini_true()
                gui.update()
        else:
            dt.throw_exit_error('Error: Config ini does not exist:\n'+str(config_file))
            exit(1)
    
    # Check if all required settings are readable in the config file
    cp.check_config_file(climate_config)

    if not silent_mode:
        input('\n\nPress Enter to Start\n\n')

    # Split big area into smaller ones
    #MaxY     MinX     MinY     MaxX    
    extent = check_files.check_all_inputs(climate_config)
    # extent = [17, -17, -3, 13] # West Africa
    # extent = [39, -25, -36, 55] # Africa
    # extent = [36, -10, 29, -1] # Morocco
    # extent = [12, -4, 4, 2] # Ghana
     
    if gui != None:
        gui.check_inpts_true()
        gui.update()
    time.sleep(0.1)

    area = abs((extent[3] - extent[1]) * (extent[0] - extent[2]))
    output_path = climate_config['files']['output_dir']
    area_name = f'Area_{int(extent[0])}N{int(extent[1])}E-{int(extent[2])}N{int(extent[3])}E'
    full_area_name = area_name
    temp = os.path.join(output_path, area_name)

    ##### DOWNSCALING #####

    print('\nDownscaling the climate data\n')
    prec_files, prec_dailyfiles = ds.interpolate_precipitation(climate_config, extent, area_name)
    temp_files, temp_dailyfiles = ds.interpolate_temperature(climate_config, extent, area_name)
    if climate_config['climatevariability'].get('consider_variability', True):
        ds.interpolate_rrpcf(climate_config, extent,  area_name,  [f for f in os.listdir(climate_config['files'].get('plant_param_dir', 'plant_params')) if f.endswith('.inf')])

    if not dt.extent_is_covered_by_second_extent(list(nc.get_maximum_extent_from_list(temp_files).values()), extent):
        [os.remove(f) for f in prec_files+temp_files]
        prec_files, prec_dailyfiles = ds.interpolate_precipitation(climate_config, extent, area_name)
        temp_files, temp_dailyfiles = ds.interpolate_temperature(climate_config, extent, area_name)

    if gui != None:
        gui.set_downscaling(completed=True)
        gui.update()


    ##### CLIMATE SUITABILITY #####
    no_tiles = np.clip(math.ceil(area / 700), 1, 100000) if climate_config['options']['use_scheduler'] else 1
    y_coords = np.around(np.linspace(extent[2], extent[0], num=int(no_tiles) + 1), 2) # type: ignore
    extents = [[y2, extent[1], y1, extent[3]] for y1, y2 in zip(y_coords[:-1], y_coords[1:])] # type: ignore

    for idx, extent in enumerate(extents):
        if gui != None:
            gui.set_extent(completed=False, extent=extent, no=idx+1, out_of=len(extents))
            gui.set_climsuit(completed=False, started=False)
            gui.set_cropsuit(completed=False, started=False)
            gui.update()

        area_name = f'Area_{int(extent[0])}N{int(extent[1])}E-{int(extent[2])}N{int(extent[3])}E'
        temp = os.path.join(output_path, area_name)
        tmp = [os.path.join(os.path.split(temp)[0]+'_var', os.path.split(temp)[1]), os.path.join(os.path.split(temp)[0]+'_novar', os.path.split(temp)[1])][1]
        
        plant_params = rpp.read_crop_parameterizations_files(climate_config['files']['plant_param_dir'])
        plant_params_formulas = rpp.get_plant_param_interp_forms_dict(plant_params, climate_config)
        plants = [plant for plant in plant_params]
        if os.path.exists(os.path.join(output_path+'_novar', area_name, plants[-1], 'crop_suitability.tif')):
            print(f'Data already existing. Skipping')
            continue
        
        if gui != None:
            gui.set_climsuit(completed=False, started=True)
            gui.update()
        print('\n'+f'Processing extent {extent} - {idx+1} out of {len(extents)}'+'\n')

        if climate_config['membershipfunctions']['plot_for_each_crop']:
            rpp.plot_all_parameterizations(plant_params_formulas, plant_params)
        print(' -> Plant data loaded')

        if climate_config['climatevariability'].get('consider_variability', True):
            climsuits = [os.path.join(pt, area_name, crop, 'climate_suitability.tif') for pt in [os.path.split(temp)[0]+'_var', os.path.split(temp)[0]+'_novar'] for crop in list(plant_params.keys())]
        else:
            climsuits = [os.path.join(os.path.split(temp)[0]+'_novar', area_name, crop, 'climate_suitability.tif') for crop in list(plant_params.keys())]
        if all(os.path.exists(clims) for clims in climsuits):
            print('\nClimate Suitability Data is already existing.\n -> Using existing data.\n')
        else:
            print(' -> Loading required climate data to memory...')
            land_sea_mask, _ = dt.load_specified_lines(climate_config['files']['land_sea_mask'], extent, False) #type:ignore
            temperature = nc.read_area_from_netcdf_list(temp_files, overlap=False, extent=extent, dayslices=temp_dailyfiles)
            temperature = dt.check_dimensions(land_sea_mask, temperature)
            precipitation = nc.read_area_from_netcdf_list(prec_files, overlap=False, extent=extent, dayslices=prec_dailyfiles)      
            precipitation = dt.check_dimensions(land_sea_mask, precipitation)

            if land_sea_mask.shape[0] != temperature.shape[0]: #type:ignore
                land_sea_mask = dt.interpolate_nanmask(land_sea_mask, temperature.shape[:2]) #type:ignore

            gc.collect()
            print(' -> Climate Data successfully loaded into memory')
            ret_paths = csm.climate_suitability(climate_config, extent, temperature, precipitation, land_sea_mask, plant_params, plant_params_formulas, temp, full_area_name)
            del temperature, precipitation
        if gui != None:
            gui.set_climsuit(completed=True, started=True)
            gui.update()
        gc.collect()

        ##### CROP SUITABILITY #####

        ret_paths = [os.path.join(os.path.split(temp)[0]+'_var', os.path.split(temp)[1]), os.path.join(os.path.split(temp)[0]+'_novar', os.path.split(temp)[1])]

        for idx, temp in enumerate(ret_paths):
            if os.path.exists(temp):
                if gui != None:
                    gui.set_cropsuit(completed=False, started=True)
                    gui.update()

                climsuit = np.dstack([dt.load_specified_lines(tif, extent, False)[0] for tif in [os.path.join(temp, crop, 'climate_suitability.tif') for crop in os.listdir(temp) if os.path.isdir(os.path.join(temp, crop))]])
                limiting = np.dstack([dt.load_specified_lines(tif, extent, False)[0] for tif in [os.path.join(temp, crop, 'limiting_factor.tif') for crop in os.listdir(temp) if os.path.isdir(os.path.join(temp, crop))]])

                if 'land_sea_mask' not in locals() and 'land_sea_mask' not in globals():
                    land_sea_mask, _ = dt.load_specified_lines(climate_config['files']['land_sea_mask'], extent, False)
                crop_suit.cropsuitability(climate_config, climsuit, limiting, plant_params_formulas, plant_params, extent, land_sea_mask, temp)
        print('Complete Current Extent')
        if gui != None:
            gui.set_cropsuit(completed=True, started=True)           
            gui.update()

        gc.collect()

    if gui != None:
        gui.set_extent(completed=True)  
        gui.set_climsuit(completed=True)         
        gui.set_cropsuit(completed=True)           
        gui.update()
    

    ##### MERGING OUTPUTS #####

    if gui != None:
        gui.set_merge(completed=False, merge_required=len(extents)>1, started=True)
        gui.update()

    for output_dir in [climate_config['files']['output_dir']+'_var', climate_config['files']['output_dir']+'_novar']:
        if not os.path.exists(output_dir):
            continue
        if len(extents) > 1:
            areas = [d for d in next(os.walk(output_dir))[1] if d.startswith('Area_')]
            north_values = [int(value[:-1]) for item in areas for value in re.findall(r'(-?\d+N)', item)]
            east_values = [int(value[:-1]) for item in areas for value in re.findall(r'(-?\d+E)', item)]
            merged_result = os.path.join(output_dir, f'Area_{max(north_values)}N{min(east_values)}E-{min(north_values)}N{max(east_values)}E')
            if not os.path.exists(merged_result):
                mg.merge_outputs_no_overlap(output_dir, climate_config)

    ##### CLEAN UP #####

    if climate_config['options']['remove_interim_results']:
        if len(extents) > 1:
            for extent in extents:
                for output_dir in [climate_config['files']['output_dir']+'_var', climate_config['files']['output_dir']+'_novar']:
                    if os.path.exists(output_dir):
                        shutil.rmtree(os.path.join(output_dir, f'Area_{int(extent[0])}N{int(extent[1])}E-{int(extent[2])}N{int(extent[3])}E'))

    if climate_config['options'].get('remove_downscaled_climate', False):
        [os.remove(f) for f in prec_files+temp_files]
        try:
            os.removedirs(os.path.dirname(temp_files[0]))
            os.removedirs(os.path.dirname(prec_files[0]))
        except:
            pass

    if gui != None:
        gui.set_merge(completed=True, merge_required=len(extents)>1, started=True)
        gui.update()

    if silent_mode:
        print('\n\nProgram successfully completed.\nAll datasets created\n\nExit')
    else:
        input('\n\nProgram successfully completed.\nAll datasets created\n\nPress Enter to Exit')
    
    if gui == None:
        exit()
    else:
        gui.set_finish()
        gui.update()
        while gui.update():
            pass


if __name__ == '__main__':
    if len(sys.argv[1:]) > 3:
        print("""Usage:
              -> python CropSuite.py
              -> python CropSuite.py -silent
              -> python CropSuite.py -silent -config "path_to_config.ini"
              -> python CropSuite.py -config "path_to_config.ini\"""")
        exit()
    else:
        silent_mode = '-silent' in sys.argv

    config_index = sys.argv.index('-config') if '-config' in sys.argv else None
    config_path = None

    if config_index is not None:
        if config_index + 1 < len(sys.argv):
            config_path = sys.argv[config_index + 1]
        else:
            print("Error: Path to configuration file is missing after -config option.")
            exit(1)

    try:
        if config_path is None:
            run(silent_mode)
        else:
            run(silent_mode, config_path)

    except Exception as e:
        print(traceback.format_exc())
        input('\nAn untreated critical error occurred\n\n'+str(e)+'\n\nPress Enter to Exit')
        exit()
