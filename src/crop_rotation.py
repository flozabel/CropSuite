import os
import numpy as np
import data_tools as dt
import read_climate_ini as rci
import read_plant_params as rpp
from numba import njit
from numba import prange
from concurrent.futures import ProcessPoolExecutor

def crop_rotation(config_file):
    config_ini = rci.read_ini_file(config_file)
    plant_params = rpp.read_crop_parameterizations_files(config_ini['files'].get('plant_param_dir', 'plant_params'))

    result_paths = [config_ini['files'].get('output_dir')+'_var', config_ini['files'].get('output_dir')+'_novar']
    for result_path in result_paths:
        if not os.path.exists(result_path):
            continue
        areas = [a for a in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, a))]
        for area in areas:
            crops = [c for c in os.listdir(os.path.join(result_path, area)) if os.path.isdir(os.path.join(os.path.join(result_path, area), c))] if areas else []

            if len(crops) < 1:
                print('Less than 1 crop found - crop rotation not possible')

            resting_period = int(config_ini['options'].get('multiple_cropping_turnaround_time', 21))
            rotation_dict = {crop: int(plant_params[crop].get('growing_cycle')[0]) for crop in crops if crop in plant_params.keys()}
            rotation_combinations = [[a, b] for a in crops for b in crops if a != b and a in rotation_dict.keys() and b in rotation_dict.keys() and (rotation_dict[a] + rotation_dict[b] + 2 * resting_period) <= 365]

            """
            # DEBUG

            for combination in rotation_combinations:
                compute_combinations(combination[0], combination[1], rotation_dict[combination[0]], rotation_dict[combination[1]], resting_period, os.path.join(result_path, area))
            """            
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(compute_combinations, combination[0], combination[1], rotation_dict[combination[0]],
                                           rotation_dict[combination[1]], resting_period, os.path.join(result_path, area)) for combination in rotation_combinations]
                for future in futures:
                    future.result()
            


@njit(parallel=True)
def calculate_suitabilities(suitabilities_a, suitabilities_b, resting, gc_a, gc_b):
    suits = np.empty((suitabilities_a.shape[0], suitabilities_a.shape[1], 2), dtype=np.int8)
    dates = np.empty((suitabilities_a.shape[0], suitabilities_a.shape[1], 2), dtype=np.int16)
    max_day = 365
    for x in prange(suitabilities_a.shape[1]):
        for y in range(suitabilities_a.shape[0]):
            timerange_a = suitabilities_a[y, x]
            suit_array = np.zeros((max_day, 4), dtype=np.int16)
            for day in range(max_day):
                if (day + gc_a + resting) > 365:
                    suit_array[day, 0] = -1
                    suit_array[day, 1] = -1
                    suit_array[day, 2] = day - 1
                    suit_array[day, 3] = day + gc_a + resting - 1
                else:
                    b_start = day + gc_a + resting
                    b_end = min(365, 365 + (day - resting - gc_b))
                    if b_start > 365 or b_end > 365:
                        print(f'Error: {day}, {gc_a}, {gc_b}, {resting}')
                    if b_end > b_start and b_start <= 365:
                        rng_b = suitabilities_b[y, x, b_start:b_end]
                        suit_b = np.max(rng_b)
                        idx_b = b_start + np.argmax(rng_b)
                        suit_array[day, 0] = timerange_a[day] if timerange_a[day] else -1
                        suit_array[day, 1] = suit_b if suit_b else -1
                        suit_array[day, 2] = day
                        suit_array[day, 3] = idx_b
                    else:
                        suit_array[day, 0] = -1
                        suit_array[day, 1] = -1
                        suit_array[day, 2] = day - 1
                        suit_array[day, 3] = day + gc_a + resting - 1
            suit_sum = np.where((suit_array[:, 0] > 0) & (suit_array[:, 1] > 0), suit_array[:, 0] + suit_array[:, 1], 0)
            idx = np.argmax(suit_sum)
            suits[y, x, 0] = suit_array[idx, 0]
            suits[y, x, 1] = suit_array[idx, 1]
            dates[y, x, 0] = suit_array[idx, 2]
            dates[y, x, 1] = suit_array[idx, 3]
    return suits, dates

@njit(parallel=True)
def calculate_suitabilities_test(suitabilities_a, suitabilities_b, resting, gc_a, gc_b):
    height, width = suitabilities_a.shape[:2]
    suits = np.full((height, width, 2), -1, dtype=np.int8)
    dates = np.full((height, width, 2), -1, dtype=np.int16)

    for y in prange(height):
        for x in range(width):
            timerange_a = suitabilities_a[y, x]
            timerange_b = suitabilities_b[y, x]
            max_val = -1
            best_i = -1
            best_j = -1
            for i in range(730):
                a_val = timerange_a[i]
                if a_val <= 0:
                    continue
                min_j = i + gc_a + resting + 1
                max_j = min(365, 365 + i - gc_b - resting)
                for j in range(min_j, max_j):
                    if j >= 365:
                        continue
                    b_val = timerange_b[j]
                    if b_val <= 0:
                        continue
                    total = a_val + b_val
                    if total > max_val:
                        max_val = total
                        best_i = i
                        best_j = j
            if best_i != -1:
                suits[y, x, 0] = timerange_a[best_i]
                suits[y, x, 1] = timerange_b[best_j]
                dates[y, x, 0] = best_i
                dates[y, x, 1] = best_j
    return suits, dates

def compute_combinations(crop_a, crop_b, gc_a, gc_b, resting, results_folder):
    results = os.path.join(results_folder, 'crop_rotation', f'{crop_a}_{crop_b}')
    if os.path.exists(results):
        print(f'-> {results} already exists')
        return

    print(f'-> Computing {crop_a} - {crop_b}')
    os.makedirs(results, exist_ok=True)
    extent = dt.get_geotiff_extent(os.path.join(results_folder, crop_a, 'climatesuitability_temp.tif'))
    suitabilities_a = dt.read_tif_file_with_bands(os.path.join(results_folder, crop_a, 'climatesuitability_temp.tif')).astype(np.int8)
    suitabilities_b = dt.read_tif_file_with_bands(os.path.join(results_folder, crop_b, 'climatesuitability_temp.tif')).astype(np.int8)
    suitabilities_a = np.concatenate([suitabilities_a, suitabilities_a], axis=2)
    suitabilities_b = np.concatenate([suitabilities_b, suitabilities_b], axis=2)
    
    suits, dates = calculate_suitabilities_test(suitabilities_a, suitabilities_b, resting, gc_a, gc_b)

    suit_a = suits[..., 0]
    suit_b = suits[..., 1]
    suit_a[suitabilities_a[..., 0] == -1] = -1
    suit_b[suitabilities_b[..., 0] == -1] = -1
    suit_a[np.max(suitabilities_a, axis=2) >= suit_a.astype(np.int16)+suit_b.astype(np.int16)] = -1
    suit_b[np.max(suitabilities_b, axis=2) >= suit_a.astype(np.int16)+suit_b.astype(np.int16)] = -1
    suit_a[suit_b <= 0] = -1
    suit_b[suit_a <= 0] = -1
    dt.write_geotiff(results, f'1-{crop_a}_climatesuitability.tif', suit_a, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)
    dt.write_geotiff(results, f'2-{crop_b}_climatesuitability.tif', suit_b, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)
    sm = suit_a.astype(np.int16) + suit_b.astype(np.int16)
    sm[sm <= 0] = -1
    dt.write_geotiff(results, f'1+2_climatesuitability.tif', sm, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)

    date_a = dates[..., 0]
    date_b = dates[..., 1]
    date_a[suit_a == -1] = -1
    date_b[suit_b == -1] = -1
    dt.write_geotiff(results, f'1-{crop_a}_climate_sowingdate.tif', date_a, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)
    dt.write_geotiff(results, f'2-{crop_b}_climate_sowingdate.tif', date_b, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)
    
    hv_date_a = date_a.copy()
    hv_date_b = date_b.copy()
    hv_date_a[hv_date_a > 0] = (hv_date_a[hv_date_a > 0] + gc_a) % 365
    hv_date_b[hv_date_b > 0] = (hv_date_b[hv_date_b > 0] + gc_b) % 365
    dt.write_geotiff(results, f'1-{crop_a}_climate_harvestdate.tif', hv_date_a, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)
    dt.write_geotiff(results, f'2-{crop_b}_climate_harvestdate.tif', hv_date_b, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)    
    del hv_date_a, hv_date_b

    soil_a = dt.read_raster_to_array(os.path.join(results_folder, crop_a, 'soil_suitability.tif')).astype(np.int8)
    suit_a = np.min([soil_a, suit_a], axis=0)
    suit_a[(soil_a == -1) | suitabilities_a[..., 0] == -1] = -1
    soil_b = dt.read_raster_to_array(os.path.join(results_folder, crop_b, 'soil_suitability.tif')).astype(np.int8)
    suit_b = np.min([soil_b, suit_b], axis=0)
    suit_b[(soil_b == -1) | suitabilities_b[..., 0] == -1] = -1
    suit_b[suit_a <= 0] = -1
    suit_a[suit_b <= 0] = -1
    suit_b[suit_a <= 0] = -1
    dt.write_geotiff(results, f'1-{crop_a}_cropsuitability.tif', suit_a, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)
    dt.write_geotiff(results, f'2-{crop_b}_cropsuitability.tif', suit_b, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)
    sm = suit_a.astype(np.int16) + suit_b.astype(np.int16)
    sm[sm <= 0] = -1
    dt.write_geotiff(results, f'1+2_cropsuitability.tif', sm, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)

    date_a[soil_a <= 0] = -1
    date_a[(suit_a == -1) | (suit_b < 0)] = -1
    date_b[soil_b <= 0] = -1
    date_b[(suit_b == -1) | (suit_a < 0)] = -1
    date_b[date_a < 0] = -1
    date_a[date_b < 0] = -1
    dt.write_geotiff(results, f'1-{crop_a}_crop_sowingdate.tif', date_a, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)
    dt.write_geotiff(results, f'2-{crop_b}_crop_sowingdate.tif', date_b, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)
    date_a[date_a > 0] = (date_a[date_a > 0] + gc_a) % 365
    date_b[date_b > 0] = (date_b[date_b > 0] + gc_b) % 365
    dt.write_geotiff(results, f'1-{crop_a}_crop_harvestdate.tif', date_a, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)
    dt.write_geotiff(results, f'2-{crop_b}_crop_harvestdate.tif', date_b, extent=extent, nodata_value=-1, dtype='int', inhibit_message=True)

    del dates, suits, soil_a, soil_b, date_a, date_b, suit_a, suit_b


if __name__ == '__main__':
    crop_rotation('brazil.ini')

