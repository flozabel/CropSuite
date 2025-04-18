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
    from src import check_versions as cv
    from src import check_prereqs as cp
except:
    import check_prereqs as cp
    import check_versions as cv
import time

class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

def run(silent_mode=False, config_file=None, gui = None):
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
        from src import preproc_tools as pti
        from src import crop_rotation as cro
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
        import preproc_tools as pti   
        import crop_rotation as cro  
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
        |                    Version 1.3.2                    |
        |                      2025-04-18                     |
        |                                                     |
        |                                                     |
        |                   Matthias Knüttel                  |
        |                     Florian Zabel                   |
        |                         2025                        |
        |                                                     |
        |                                                     |
        |          Department of Environmental Sciences       |      
        |                 University of Basel                 |
        |                                                     |
        |                                                     |
        |           © 2023-2025 All rights reserved           |
        |                                                     |
        =======================================================
        
        ''')

    # Read config_climate.ini into a dictionary
    if gui != None:
        config_file = gui.get_config_path()

    if config_file is None:
        config_file = os.path.abspath('config.ini')
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

    if not silent_mode:
        input('\n\nPress Enter to Start\n\n')

    pret_file = os.path.join(os.path.dirname(config_file), 'preproc.ini')
    if os.path.exists(pret_file):
        config_ini, temp_files, temp_varname, prec_files, prec_varname, time_range, pret_extent, proc_varfiles, downscaling, autostart = pti.parse_inf_file(pret_file)
        if autostart == 1:
            pti.preprocessing_main(config_ini, temp_files, prec_files, time_range, pret_extent, proc_varfiles, temp_varname, prec_varname, downscaling=downscaling==1) #type:ignore

    cp.check_config_file(climate_config)
    extent = check_files.check_all_inputs(climate_config)
     
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

    #if not dt.extent_is_covered_by_second_extent(list(nc.get_maximum_extent_from_list(temp_files).values()), extent):
    #    [os.remove(f) for f in prec_files+temp_files]
    #    prec_files, prec_dailyfiles = ds.interpolate_precipitation(climate_config, extent, area_name)
    #    temp_files, temp_dailyfiles = ds.interpolate_temperature(climate_config, extent, area_name)

    if gui != None:
        gui.set_downscaling(completed=True)
        gui.update()


    ##### CLIMATE SUITABILITY #####
    resolution_factor = {5: 1, 6: 0.25, 4: 5, 3: 10, 2: 12, 1: 30, 0: 60}
    no_tiles = np.clip(math.ceil(area / 700 / resolution_factor.get(int(climate_config['options'].get('resolution', 5)), 1) ** 2), 1, 100000) if climate_config['options']['use_scheduler'] else 1
    final_shape = dt.get_resolution_array(climate_config, extent, True)

    def adjust_extent_0(extent, resolution):
        return extent[2] + ((extent[0] - extent[2]) // resolution ) * resolution

    if final_shape[0] % no_tiles != 0:
        no_tiles = math.ceil(final_shape[0] / (final_shape[0] // no_tiles))

    resolution = (extent[3] - extent[1]) / final_shape[1]
    extent[0] = adjust_extent_0(extent, resolution)

    lst = [i * int(final_shape[0] / no_tiles) for i in range(no_tiles)] + [final_shape[0]]
    extents = [[extent[2] + lst[i+1] * resolution, extent[1], extent[2] + lst[i] * resolution, extent[3]]for i in range(no_tiles)]
    # extents = [[round(val, 4) for val in sublist] for sublist in extents]

    for idx, extent in enumerate(extents):
        if gui != None:
            gui.set_extent(completed=False, extent=extent, no=idx+1, out_of=len(extents))
            gui.set_climsuit(completed=False, started=False)
            gui.set_cropsuit(completed=False, started=False)
            gui.update()

        area_name = f'Area_{int(extent[0])}N{int(extent[1])}E-{int(extent[2])}N{int(extent[3])}E'
        temp = os.path.join(output_path, area_name)

        plant_params = rpp.read_crop_parameterizations_files(climate_config['files']['plant_param_dir'])
        plant_params_formulas = rpp.get_plant_param_interp_forms_dict(plant_params, climate_config)
        plants = [plant for plant in plant_params]
        if os.path.exists(os.path.join(output_path+'_novar', area_name, plants[-1], 'crop_suitability.tif')):
            print(f'Data already existing. Skipping')
            continue
        
        if gui != None:
            gui.set_climsuit(completed=False, started=True)
            gui.update()
        formatted_extent = [f"{float(e):.4f}" for e in extent]
        print(f"\nProcessing extent {formatted_extent} - {idx + 1} out of {len(extents)}\n")

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
            temperature = nc.read_area_from_netcdf_list(temp_files, overlap=False, extent=extent, dayslices=temp_dailyfiles)
            precipitation = nc.read_area_from_netcdf_list(prec_files, overlap=False, extent=extent, dayslices=prec_dailyfiles)
            fine_resolution = (temperature.shape[0], temperature.shape[1]) #type:ignore
            land_sea_mask, _ = dt.load_specified_lines(climate_config['files']['land_sea_mask'], extent, False)
            if land_sea_mask.shape != fine_resolution:
                land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_resolution)
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

                climsuit = np.dstack([
                    dt.load_specified_lines(
                        next(f for f in [os.path.join(temp, c, f'climate_suitability{ext}') for ext in ['.tif', '.nc', '.nc4']] if os.path.exists(f)),
                        extent, False
                    )[0].astype(np.int8)
                    for c in os.listdir(temp)
                    if c != 'crop_rotation' and os.path.isdir(os.path.join(temp, c))
                ])

                limiting = np.dstack([
                    dt.load_specified_lines(
                        next(f for f in [os.path.join(temp, c, f'limiting_factor{ext}') for ext in ['.tif', '.nc', '.nc4']] if os.path.exists(f)),
                        extent, False
                    )[0].astype(np.int8)
                    for c in os.listdir(temp)
                    if c != 'crop_rotation' and os.path.isdir(os.path.join(temp, c))
                ])

                #climsuit = np.dstack([(dt.load_specified_lines(tif, extent, False)[0]).astype(np.int8) for tif in [os.path.join(temp, crop, 'climate_suitability.tif') for crop in os.listdir(temp) if crop != 'crop_rotation' and os.path.isdir(os.path.join(temp, crop))]])
                #limiting = np.dstack([(dt.load_specified_lines(tif, extent, False)[0]).astype(np.int8) for tif in [os.path.join(temp, crop, 'limiting_factor.tif') for crop in os.listdir(temp) if crop != 'crop_rotation' and os.path.isdir(os.path.join(temp, crop))]])
                land_sea_mask, _ = dt.load_specified_lines(climate_config['files']['land_sea_mask'], extent, False)
                fine_resolution = (climsuit.shape[0], climsuit.shape[1])
                if land_sea_mask.shape != fine_resolution:
                    land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_resolution)
                crop_suit.cropsuitability(climate_config, climsuit, limiting, plant_params_formulas, plant_params, extent, land_sea_mask, temp)

        ##### CROP ROTATION #####
        if climate_config['options'].get('consider_crop_rotation', False):
            cro.crop_rotation(config_file)

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
        [os.remove(os.path.join(os.path.dirname(prec_files[0]), f)) for f in os.listdir(os.path.dirname(prec_files[0])) if os.path.isfile(os.path.join(os.path.dirname(prec_files[0]), f))]

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
    if cv.check_versions():
        try:
            args = sys.argv[1:]
            silent = "-silent" in args
            config = args[args.index("-config") + 1] if "-config" in args and len(args) > args.index("-config") + 1 else None
            debug = "-debug" in args
        except:
            print("""Usage:
                -> python CropSuite.py
                -> python CropSuite.py -silent
                -> python CropSuite.py -silent -config "path_to_config.ini"
                -> python CropSuite.py -config "path_to_config.ini\"""")
            exit()
        try:
            if debug:
                tee = Tee('logfile.log')
                original_stdout = sys.stdout
                sys.stdout = tee

            run(silent, config)

        finally:
            if debug:
                sys.stdout = original_stdout
                tee.close()

    sys.exit(1)
    exit()
