import os
import numpy as np
import rasterio
import xarray as xr


def write_rrpcf_day(config_dict, extent):
    v = config_dict['options'].get('irrigation', 'n')
    irrigation = (v.lower() == 'y' if isinstance(v, str) and v.lower() in ['y', 'n'] else bool(np.mean([int(c) for c in v]) > 0.5) if isinstance(v, str) and
                    all(c in '01' for c in v) else bool(v) if isinstance(v, (int, float)) else False)
    ir_code = 'ir' if irrigation else 'rf'
    crop_failure_code = 'rrpcf'
    crops = [os.path.splitext(f)[0] for f in os.listdir(config_dict['files'].get('plant_param_dir', 'plant_params')) if f.endswith('.inf')]
    area_name = f'Area_{int(extent[0])}N{int(extent[1])}E-{int(extent[2])}N{int(extent[3])}E'
    rrpcf_path = os.path.join(f'{config_dict["files"].get("output_dir")}_downscaled', area_name)

    for crop in crops:
        print(f' -> {crop.capitalize()}')
        if not os.path.exists(os.path.join(rrpcf_path, f'ds_{crop_failure_code}_{crop.lower()}_{ir_code}_0.nc')):
            print(f'{crop_failure_code.upper()} for {crop.capitalize()} missing. Skipping.')
            continue
        
        crop_dir = os.path.join(f'{config_dict["files"].get("output_dir")}_var', area_name, crop)
        if not os.path.exists(crop_dir):
            continue
        
        output_flags = {'optimal_sowing_date.tif': 'rrpcf', 'optimal_sowing_date_mc_first.tif': 'rrpcfmc1',
                        'optimal_sowing_date_mc_second.tif': 'rrpcfmc2','optimal_sowing_date_mc_third.tif': 'rrpcfmc3'}

        for sowing_file in ['optimal_sowing_date.tif', 'optimal_sowing_date_mc_first.tif', 'optimal_sowing_date_mc_second.tif', 'optimal_sowing_date_mc_third.tif']:
            key = output_flags.get(sowing_file)
            if key and config_dict.get('outputs', {}).get(key, True):

                sowing_date = os.path.join(crop_dir, sowing_file)
                suff = sowing_file.split("_mc_")[1].removesuffix(".tif") if "_mc_" in sowing_file else ""
                out_file = os.path.join(crop_dir, 'rrpcf'+f'{suff}'+'.tif')

                if not os.path.exists(sowing_date):
                    continue

                with rasterio.open(sowing_date, 'r') as src:
                    data = src.read(1)
                    profile = src.profile
                
                result = np.full_like(data, fill_value=-1, dtype=np.int8)
                if len(np.unique(data)) == 3:
                    filepath = os.path.join(rrpcf_path, f'ds_{crop_failure_code}_{crop.lower()}_{ir_code}_0.nc')
                    if not os.path.exists(filepath):
                        print(f"Missing: {os.path.basename(filepath)}")
                    else:
                        with xr.open_dataset(filepath) as ds:
                            result = ds[list(ds.data_vars)[0]].values
                else:
                    for uid in [int(uid) for uid in np.unique(data) if 0 <= uid <= 364]:
                        filepath = os.path.join(rrpcf_path, f'ds_{crop_failure_code}_{crop.lower()}_{ir_code}_{uid}.nc')
                        if not os.path.exists(filepath):
                            print(f"Missing: {os.path.basename(filepath)}")
                            continue
                        with xr.open_dataset(filepath) as ds:
                            mask = data == uid
                            result[mask] = ds[list(ds.data_vars)[0]].values[mask]
                profile.update({"dtype": result.dtype,"count": 1,"compress": "lzw", "nodata": -1})
                with rasterio.open(os.path.join(crop_dir, out_file), "w", **profile) as dst:
                    dst.write(result, 1)
    
    crop_rotation_path = os.path.join(f'{config_dict["files"].get("output_dir")}_var', area_name, 'crop_rotation')
    if not os.path.exists(crop_rotation_path):
        return
    combinations = os.listdir(crop_rotation_path) if len(crops) > 1 else []

    for combination in combinations:
        current_path = os.path.join(crop_rotation_path, combination)
        print(f' -> {combination}')
        for filename in os.listdir(current_path):
            if filename.endswith('_climate_sowingdate.tif'):
                crop = filename.split('-')[1].replace('_climate_sowingdate.tif', '')
            elif filename.endswith('_crop_sowingdate.tif'):
                crop = filename.split('-')[1].replace('_crop_sowingdate.tif', '')
            else:
                continue
            sowing_date = os.path.join(current_path, filename)

            if (filename.startswith('1_') and config_dict.get('outputs', {}).get('crrrpcf1', True)) or\
                (filename.startswith('2_') and config_dict.get('outputs', {}).get('crrrpcf2', True)):

                with rasterio.open(sowing_date, 'r') as src:
                    data = src.read(1)
                    profile = src.profile
                result = np.full_like(data, fill_value=-1, dtype=np.int8)
                for uid in [int(uid) for uid in np.unique(data) if 0 <= uid <= 364]:
                    filepath = os.path.join(rrpcf_path, f'ds_{crop_failure_code}_{crop.lower()}_{ir_code}_{uid}.nc')
                    if not os.path.exists(filepath):
                        print(f"Missing: {os.path.basename(filepath)}")
                        continue
                    with xr.open_dataset(filepath) as ds:
                        mask = data == uid
                        result[mask] = ds[list(ds.data_vars)[0]].values[mask]
                profile.update({"dtype": result.dtype,"count": 1,"compress": "lzw", "nodata": -1})
                with rasterio.open(os.path.join(current_path, f'rrpcf_{crop}.tif'), "w", **profile) as dst:
                    dst.write(result, 1)


