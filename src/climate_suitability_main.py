import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
import data_tools as dt
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')
from numba import jit
from gc import collect
import math
import nc_tools as nc
import re
from numba import njit
import re
    
def get_suitability_val_dict(forms, plant, form_type, value):
    """
    Calculates the suitability value based on a specified formula for a given plant form.

    Parameters:
    - forms (dict): Dictionary containing plant forms and their corresponding formulas, minimum and maximum values.
    Example structure: {'plant_1': {'form_type_1': {'formula': func_1, 'min_val': min_val_1, 'max_val': max_val_1}, ...}, ...}
    - plant (str): Name of the plant for which suitability value needs to be calculated.
    - form_type (str): Type of the plant form for which the suitability value is calculated.
    - value (Union[float, np.ndarray]): The input value or array of values for the formula calculation.

    Returns:
    - suitability_value (float): The calculated suitability value based on the specified formula for the given plant form.
    """
    func, min_val, max_val = forms[plant][form_type]['formula'], forms[plant][form_type]['min_val'], forms[plant][form_type]['max_val']
    if isinstance(value, np.ndarray):
        np.clip(value, min_val, max_val, out=value)
        return func(value)
    else:
        value = min(max_val, max(min_val, value))
        return func(value)
    

def calculate_day_length(lat, day_of_year):
    """
    Calculate the daylight duration in hours for a given latitude and day of the year.
    """
    decl = math.radians(23.44) * math.cos(math.radians((360/365) * (day_of_year + 10)))
    lat_rad = math.radians(lat)
    hour_angle = math.acos(-math.tan(lat_rad) * math.tan(decl))
    return 24 * hour_angle / np.pi


def calculate_average_sunshine(array_shape, lat_bounds, start_day, end_day):
    """
    Calculate the average sunshine duration over a given range of days for each pixel.
    
    Parameters:
    array_shape: tuple of (rows, cols) for the output array shape
    lat_bounds: tuple of (min_lat, max_lat) for latitude boundaries
    start_day: integer, starting day of the year (e.g., 124)
    end_day: integer, ending day of the year (e.g., 287)
    
    Returns:
    2D numpy array of average sunshine duration in hours.
    """
    rows, cols = array_shape
    min_lat, max_lat = lat_bounds
    lats = np.linspace(min_lat, max_lat, rows)
    avg_sunshine = np.zeros(array_shape, dtype=np.uint8)
    for i, lat in enumerate(lats):
        total_sunshine = 0
        num_days = end_day - start_day + 1
        for day in range(start_day, end_day + 1):
            try:
                total_sunshine += calculate_day_length(-lat, day)
            except:
                total_sunshine += 0
        avg_sunshine[i, :] = total_sunshine / num_days
    return avg_sunshine


def extract_season_data(temperature, precipitation, water_mask, day, growing_cycle, irrigation):
    if day + growing_cycle <= 365:
        temperature = temperature[..., day:day+growing_cycle]
        precipitation = precipitation[..., day:day+growing_cycle] if irrigation == 0 else np.zeros_like(temperature)
    elif growing_cycle == 365:
        temperature = temperature[...]
        precipitation = precipitation[...] if irrigation == 0 else np.zeros_like(temperature)
    else:
        temperature = np.concatenate((temperature[..., day:], temperature[..., :day+growing_cycle-365]), axis=2)
        precipitation = np.concatenate((precipitation[..., day:], precipitation[..., :day+growing_cycle-365]), axis=2) if irrigation == 0 else np.zeros_like(temperature)
    if temperature.shape[0] != water_mask.shape[0]:
        water_mask = water_mask[:temperature.shape[0], :temperature.shape[1]]
    return temperature, precipitation, water_mask


@njit
def compute_mean_water_masked(temperature, water_mask):
    lat, lon, t = temperature.shape
    result = np.zeros((lat, lon), dtype=np.int16)

    for i in range(lat):
        for j in range(lon):
            if water_mask[i, j] == 1:
                mean_val = 0.0
                for k in range(t):
                    mean_val += temperature[i, j, k]
                mean_val /= t
                result[i, j] = int(mean_val)
    return result


def apply_lethal_constraints(temperature, precipitation, temp, prec, water_mask, lethal_params, wintercrop, opt_start_date, dry_day_prec, max_consec_dry_days, max_prec_val, max_prec_dur, irrigation):
    def consecutive_condition(arr, threshold, cond_fn):
        cond = cond_fn(arr)
        cond = cond.astype(np.int16)
        reset_mask = ~cond.astype(bool)
        cond[reset_mask] = 0
        consec = np.zeros_like(cond[:, :, 0], dtype=np.int16)
        result = np.zeros_like(cond[:, :, 0], dtype=bool)
        for day in range(cond.shape[2]):
            consec = (consec + 1) * cond[:, :, day]
            result |= (consec >= threshold)
        return result

    min_dur, min_tmp = lethal_params[0], lethal_params[1]
    max_dur, max_tmp = lethal_params[2], lethal_params[3]

    if min_dur > 0:
        mask = consecutive_condition(temperature, min_dur, lambda arr: arr < (min_tmp * 10))
        if wintercrop:
            opt_start_date[mask] = -1
        temp[mask] = 0

    if max_dur > 0:
        mask = consecutive_condition(temperature, max_dur,lambda arr: arr > (max_tmp * 10))
        if wintercrop:
            opt_start_date[mask] = -1
        temp[mask] = 0

    if max_consec_dry_days > 0 and irrigation != 1:
        mask = consecutive_condition(precipitation, max_consec_dry_days, lambda arr: arr < (dry_day_prec * 10) if dry_day_prec != 0 else np.ones_like(arr, dtype=bool))
        prec[(water_mask == 1) & mask] = 0

    if max_prec_val > 0 and irrigation != 1:
        threshold = max_prec_dur if max_prec_dur > 0 else 3
        mask = consecutive_condition(precipitation, threshold, lambda arr: arr > (max_prec_val * 10) if max_prec_val != 0 else arr > 75)
        prec[(water_mask == 1) & mask] = 0

    return temp, prec


def smooth_curve(x, lower, upper, smoothness):
    x = np.asarray(x)
    half = smoothness / 2
    k = 2 * np.log(99) / smoothness
    rise = np.zeros_like(x, dtype=float)
    rise[x <= lower - half] = 0
    rise[x >= lower + half] = 100
    in_rise = (x > lower - half) & (x < lower + half)
    rise[in_rise] = 100 / (1 + np.exp(-k * (x[in_rise] - lower)))
    fall = np.zeros_like(x, dtype=float)
    fall[x <= upper - half] = 100
    fall[x >= upper + half] = 0
    in_fall = (x > upper - half) & (x < upper + half)
    fall[in_fall] = 100 / (1 + np.exp(k * (x[in_fall] - upper)))
    return np.minimum(rise, fall).astype(np.int8)

def process_day_climsuit_memopt(args):
    day, growing_cycle, temperature, precipitation, plant_params_forumlas, plant, wintercrop, water_mask, irrigation,\
        crop_failures, vernalization_params, lethal, lethal_params, sowprec_params, photoperiod, photoperiod_params,\
            max_consec_dry_days, dry_day_prec, dursowing, sowingtemp, max_prec_val, max_prec_dur, additional_conditions, phenology_params = args
    
    vern_period = 150

    temp_path = os.path.join(os.getcwd(), 'temp', f'{day}.tif')

    if os.path.exists(temp_path):
        sys.stdout.write(f' -> Skipping day #{day}              '+'\r')
        sys.stdout.flush()
        return
    sys.stdout.write(f' -> Processing day #{day}                                      '+'\r')
    sys.stdout.flush()

    if wintercrop:
        start_of_vernalization = day - vern_period
        if start_of_vernalization < 0:
            window_temp_vern = np.concatenate((temperature[..., start_of_vernalization:], temperature[..., :day]), axis=2)
        else:
            window_temp_vern = temperature[..., start_of_vernalization:day]  # Use view instead of copy

        # Vernalization temperature thresholds
        vernalization_min_temp = vernalization_params[2] * 10
        vernalization_max_temp = vernalization_params[1] * 10

        # Calculate cumulative conditions for vernalization in-place, minimize boolean array
        valid_temp_range = (window_temp_vern <= vernalization_max_temp) & (window_temp_vern >= vernalization_min_temp)
        cumulative_conditions = np.cumsum(valid_temp_range, axis=2, out=valid_temp_range.astype(np.int8))  # Reuse memory

        # Find where cumulative conditions meet vernalization requirements
        condition_met = cumulative_conditions == vernalization_params[0]

        # Initialize opt_start_date array, reuse cumulative_conditions for met_indices
        opt_start_date = np.full(cumulative_conditions.shape[:-1], -1, dtype=np.int16)
        met_indices = np.any(condition_met, axis=2)

        # Use in-place indexing and operations to update only where vernalization requirements are met
        first_50_days = np.argmax(condition_met, axis=2)
        opt_start_date[met_indices] = vern_period - first_50_days[met_indices]

        # Clean up temporary arrays
        del valid_temp_range, cumulative_conditions, first_50_days

        # Adjust opt_start_date
        opt_start_date = np.where(opt_start_date != -1, (day - opt_start_date) % 365, -1)

        # Frost resistance check
        if vernalization_params[3] != 0 and vernalization_params[4] != 0:
            frost_exceedance_days = np.sum(window_temp_vern < (vernalization_params[3] * 10), axis=2)
            opt_start_date[frost_exceedance_days > vernalization_params[4]] = -1
        del frost_exceedance_days
        del window_temp_vern

        days_to_vern = int(vernalization_params[5])

        temp_to_vern = np.full((temperature.shape[0], temperature.shape[1], days_to_vern), np.nan, dtype=np.float16)
        prec_to_vern = np.full((precipitation.shape[0], precipitation.shape[1], days_to_vern), 0, dtype=np.int16)

        vern_start = opt_start_date[..., None]
        start_indices = (vern_start - days_to_vern) % 365

        rows, cols = np.indices((temperature.shape[0], temperature.shape[1]), sparse=True)
        rows = rows[..., None]  # Make rows compatible for broadcasting
        cols = cols[..., None]
        range_indices = np.arange(days_to_vern)  # 1D array for days to vernalization

        # Extract temperature and precipitation data with modular arithmetic, avoiding array doubling
        start_indices_full = (start_indices + range_indices) % 365  # Wrap indices in range [0, 365)

        # Use advanced indexing to extract data
        temp_to_vern = temperature[rows, cols, start_indices_full]  # Extract temperature
        prec_to_vern = precipitation[rows, cols, start_indices_full]  # Extract precipitation

        vern_temp = np.nanmean(temp_to_vern, axis=2, dtype=np.float16)
        vern_temp = np.nan_to_num(vern_temp, nan=-32767).astype(np.int16)
        vern_temp[np.nanmean(temp_to_vern[..., :dursowing], axis=2) <= (sowingtemp * 10)] = -32767

        vern_prec = np.sum(prec_to_vern, axis=2, dtype=np.int16) / (days_to_vern / growing_cycle)
        vern_prec[np.sum(prec_to_vern[..., :sowprec_params[0]], axis=2, dtype=np.int16) <= sowprec_params[1] * 10] = 0
        del temp_to_vern, prec_to_vern

        opt_start_date[(vern_temp <= 0) | (vern_prec <= 0)] = -1

        with rasterio.open(temp_path[:-4]+'_osd.tif', 'w', driver='GTiff',  width=opt_start_date.shape[1], height=opt_start_date.shape[0], count=1, dtype=np.int16, compress='lzw') as dst:
            dst.write(opt_start_date, 1) 
        
        growing_cycle -= days_to_vern
    
    temperature, precipitation, water_mask = extract_season_data(temperature, precipitation, water_mask, day, growing_cycle, irrigation)
    temp = np.where(water_mask == 1, np.mean(temperature, axis=2).astype(np.int16), 0)

    if wintercrop:
        temp = ((temp * growing_cycle) + (vern_temp * vernalization_params[5])) / (growing_cycle + vernalization_params[5])
    
        ### WINTERCROP -> PHENOLOGY?

    if len(phenology_params) > 0 and not wintercrop and growing_cycle < 365:
        temp = []
        for entry in [entry for entry in phenology_params if entry[0] == 'temp']:
            if len(entry) > 6 and entry[6] != '[]' and entry[6] != []:
                s = ' '.join(entry[6:])
                xp, fp = zip(*[tuple(map(float, p.split())) for p in re.findall(r'\(([^)]+)\)', s)])
                xp = np.array(xp) * 10
                curr_temp = np.interp(np.clip(np.mean(temperature[..., entry[1]-1:entry[2]-1], axis=2, dtype=np.int16), min(xp), max(xp)), xp, fp)
            else:
                curr_temp = np.clip(np.mean(temperature[..., entry[1]-1:entry[2]-1], axis=2, dtype=np.int16), a_min=(float(entry[3])*10) - (float(entry[5])*5), a_max=float(entry[4])*10 + (float(entry[5])*5))
                curr_temp = smooth_curve(curr_temp, float(entry[3])*10, float(entry[4])*10, float(entry[5])*10)
            temp.append(curr_temp)
        temp = np.min(np.asarray(temp, dtype=np.int8), axis=0)
    else:
        temp[water_mask == 1] = (get_suitability_val_dict(plant_params_forumlas, plant, 'temp', temp[water_mask == 1] / 10) * 100)
        temp = temp.astype(np.int8)

    if not wintercrop and growing_cycle < 365:
        temp[np.nanmean(temperature[..., :dursowing], axis=2) <= (sowingtemp * 10)] = 0

    if irrigation == 0:
        prec = np.where(water_mask == 1, np.sum(precipitation, axis=2).astype(np.int32), 0)
        if wintercrop:
            prec = ((prec * growing_cycle) + (vern_prec * vernalization_params[5])) / (growing_cycle + vernalization_params[5])
        if len(phenology_params) > 0 and not wintercrop and growing_cycle < 365:
            prec = []
            for entry in [entry for entry in phenology_params if entry[0] == 'prec']:
                if len(entry) > 6 and entry[6] != '[]' and entry[6] != []:
                    s = ' '.join(entry[6:])
                    xp, fp = zip(*[tuple(map(float, p.split())) for p in re.findall(r'\(([^)]+)\)', s)])
                    xp = np.array(xp) * 10
                    curr_temp = np.interp(np.clip(np.sum(precipitation[..., entry[1]-1:entry[2]-1], axis=2, dtype=np.int16), min(xp), max(xp)), xp, fp)
                else:
                    curr_prec = np.clip(np.sum(precipitation[..., entry[1]-1:entry[2]-1], axis=2, dtype=np.int32), a_min=(float(entry[3])*10) - (float(entry[5])*5), a_max=float(entry[4])*10 + (float(entry[5])*5))
                    curr_prec = smooth_curve(curr_prec, float(entry[3])*10, float(entry[4])*10, float(entry[5])*10)
                prec.append(curr_prec)
            prec = np.min(np.asarray(prec, dtype=np.int8), axis=0)
        else:
            prec[water_mask == 1] = (get_suitability_val_dict(plant_params_forumlas, plant, 'prec', prec[water_mask == 1] / 10) * 100)
            prec = prec.astype(np.int8)
    else:
        prec = np.zeros_like(temp)

    if crop_failures.ndim > 1:
        crop_fail = np.empty((temperature.shape[0], temperature.shape[1]), dtype=np.int8)
        crop_fail[water_mask == 1] = (get_suitability_val_dict(plant_params_forumlas, plant, 'freqcropfail', crop_failures[water_mask == 1]/100) * 100).astype(np.int8)
    else:
        crop_fail = np.full_like(temp, 100)

    if irrigation == 0 and growing_cycle < 365 and not wintercrop:
        prec_sum = np.sum(precipitation[:, :, :sowprec_params[1]], axis=2, dtype=np.uint16)
        mask = (water_mask == 1) & (prec_sum < sowprec_params[0] * 10)
        prec[mask] = 0

    temp[np.mean(temperature, axis=2) < 0] = 0

    if photoperiod and growing_cycle < 365:
        min_h, max_h, min_y, max_y = photoperiod_params
        sunshine_hours = calculate_average_sunshine((temp.shape[0], temp.shape[1]), (min_y, max_y), day, day+growing_cycle)
        sunshine_hours = np.where((sunshine_hours >= min_h) & (sunshine_hours <= max_h), 100, 0)
    else:
        sunshine_hours = np.full_like(temp, 100, dtype = temp.dtype)

    if lethal:
        temp, prec = apply_lethal_constraints(temperature, precipitation, temp, prec, water_mask, lethal_params, wintercrop,
                                              opt_start_date if wintercrop else None, dry_day_prec, max_consec_dry_days,
                                              max_prec_val, max_prec_dur, irrigation)

    if additional_conditions:
        for cond in additional_conditions:
            start_day, end_day = int(cond[1]), int(cond[2])
            val = int(float(cond[4]) * 10)
            if cond[0].lower() == 'temperature':
                data_slice = np.nanmean(temperature[..., start_day:end_day], axis=2)
                comp_map = {'>=': data_slice >= val, '>':  data_slice > val, '<':  data_slice < val, '<=': data_slice <= val}
                mask = comp_map.get(cond[3])
                if mask is not None:
                    temp[mask] = 0

            elif cond[0].lower() == 'precipitation':
                data_slice = np.nansum(precipitation[..., start_day:end_day], axis=2)
                comp_map = {'>=': data_slice < val, '>':  data_slice <= val, '<':  data_slice >= val, '<=': data_slice > val}
                mask = comp_map.get(cond[3])
                if mask is not None:
                    prec[mask] = 0

    with rasterio.open(temp_path, 'w', driver='GTiff',  width=temp.shape[1], height=temp.shape[0], count=4, dtype=np.int8, compress='lzw') as dst:
        dst.write(temp, 1)
        dst.write(prec.fill(100) if irrigation else prec, 2)
        dst.write(crop_fail, 3)
        dst.write(sunshine_hours, 4)

    if not os.path.exists(temp_path):
        process_day_climsuit_memopt(args)
    else:
        if irrigation == 0:
            del temp, prec, sunshine_hours
        else:
            del temp, sunshine_hours




def read_and_process(day):
    with rasterio.open(os.path.join(os.getcwd(), 'temp', f'{day}.tif')) as src:
        return np.transpose(src.read(), (1, 2, 0))


def read_tif_data_to_tempprecfail_arr(shape):
    """
    Reads temperature, precipitation, and failure suitability arrays from GeoTIFF files.

    Parameters:
    - shape (tuple): Shape of the output arrays in the format (height, width, depth).

    Returns:
    - temp_arr (np.ndarray): Array containing temperature values for each day.
    - prec_arr (np.ndarray): Array containing precipitation values for each day.
    - fail_arr (np.ndarray): Array containing failure suitability values for each day.

    Note: This function reads GeoTIFF files for each day and extracts temperature, precipitation, and failure suitability data.
    """
    with rasterio.open(os.path.join(os.getcwd(), 'temp', '0.tif')) as src:
        dtype = src.dtypes[0]
    temp_arr = np.empty(shape, dtype=dtype)
    for day in range(365):
        sys.stdout.write(f'     - reading {day}.tif                      '+'\r')
        sys.stdout.flush()
        with rasterio.open(os.path.join(os.getcwd(), 'temp', f'{day}.tif')) as src:
            temp_arr[..., day, :] = np.transpose(src.read(), (1, 2, 0))
    sys.stdout.write(f'   -> All files read in successfully                       '+'\r')
    sys.stdout.flush()
    collect()
    return temp_arr[..., 0], temp_arr[..., 1], temp_arr[..., 2], temp_arr[..., 3]


def read_tif_data_to_opt_sow_date_arr(shape):
    with rasterio.open(os.path.join(os.getcwd(), 'temp', '0_osd.tif')) as src:
        dtype = src.dtypes[0]
    temp_arr = np.empty(shape, dtype=dtype)
    for day in range(365):
        sys.stdout.write(f'     - reading {day}.tif                      '+'\r')
        sys.stdout.flush()
        with rasterio.open(os.path.join(os.getcwd(), 'temp', f'{day}_osd.tif')) as src:
            temp_arr[..., day] = src.read(1)
    sys.stdout.write(f'   -> All files read in successfully                       '+'\r')
    sys.stdout.flush()
    collect()
    return temp_arr

def read_tif_data_to_opt_sow_date_arr_start(shape):
    with rasterio.open(os.path.join(os.getcwd(), 'temp', '0_osd_start.tif')) as src:
        dtype = src.dtypes[0]
    temp_arr = np.empty(shape, dtype=dtype)
    for day in range(365):
        sys.stdout.write(f'     - reading {day}.tif                      '+'\r')
        sys.stdout.flush()
        with rasterio.open(os.path.join(os.getcwd(), 'temp', f'{day}_osd_start.tif')) as src:
            temp_arr[..., day] = src.read(1)
    sys.stdout.write(f'   -> All files read in successfully                       '+'\r')
    sys.stdout.flush()
    collect()
    return temp_arr

@jit(nopython=True) #type:ignore
def find_max_sum_new(suit_vals, span, harvests) -> tuple:
    """
    Finds the optimal combination of days to maximize the sum of suitability values for harvesting.

    Parameters:
    - suit_vals (list): A list containing the suit values for each day of the year.
    - span (int): The number of consecutive days required for a single harvest.
    - harvests (int): The number of harvests to perform (2 or 3).

    Returns:
    - tuple: A tuple containing two elements:
        - list: The indices of the selected days for harvesting.
        - int: The maximum sum of suit values achievable with the selected days.

    Note:
    - The function assumes that the length of suit_vals is 365, corresponding to each day of the year.
    """
    max_sum = 0
    max_indices = [-1, -1] if harvests == 2 else [-1, -1, -1]

    if harvests == 2:
        for day1 in range(365-span):
            suit1 = suit_vals[day1]
            for day2 in range((day1 + span) % 365, (day1 - span) % 365):
                suit2 = suit_vals[day2]
                curr_sum = suit1 + suit2
                if curr_sum > max_sum:
                    max_sum, max_indices = curr_sum, [day1, day2]
    else:
        for day1 in range(365-span):
            suit1 = suit_vals[day1]
            for day2 in range((day1 + span) % 365, (day1 - (2 * span)) % 365):
                suit2 = suit_vals[day2]
                for day3 in range((day2 + span) % 365, (day1 - span) % 365):
                    suit3 = suit_vals[day3]
                    curr_sum = suit1 + suit2 + suit3
                    if curr_sum > max_sum:
                        max_sum, max_indices = curr_sum, [day1, day2, day3]
        max_sum_2hv = 0
        max_indices_2hv = [-1, -1]
        for day1 in range(365-span):
            suit1 = suit_vals[day1]
            for day2 in range((day1 + span) % 365, (day1 - span) % 365):
                suit2 = suit_vals[day2]
                curr_sum = suit1 + suit2
                if curr_sum > max_sum:
                    max_sum_2hv, max_indices_2hv = curr_sum, [day1, day2]
        if max_sum_2hv >= max_sum:
            return max_indices_2hv, max_sum_2hv
    return sorted(max_indices), max_sum  


@njit
def find_longest_run_center(arr3d):
    ny, nx, nt = arr3d.shape
    result = np.zeros((ny, nx), dtype=np.int32)
    for i in range(ny):
        for j in range(nx):
            ts = arr3d[i, j, :]
            max_val = np.max(ts)
            if max_val == 0:
                result[i, j] = 0
                continue
            max_mask = ts == max_val
            best_len = 0
            best_start = 0
            k = 0
            while k < nt:
                if max_mask[k]:
                    start = k
                    while k < nt and max_mask[k]:
                        k += 1
                    end = k  # exclusive
                    length = end - start
                    if length > best_len:
                        best_len = length
                        best_start = start
                else:
                    k += 1
            center = best_start + (best_len - 1) // 2
            result[i, j] = center
    return result


def climsuit_new(climate_config, extent, temperature, precipitation, land_sea_mask, plant_params, plant_params_formulas, results_path, plant, area_name) -> list:
    """
    Runs the CLIMSUIT part to assess climate suitability for crop cultivation.

    Parameters:
    - climate_config (dict): Configuration settings for the CLIMSUITE model.
    - extent (list): List containing the spatial extent of the analysis area in the format [min_lon, min_lat, max_lon, max_lat].
    - climatedata (np.ndarray): 4D array containing climate data in the format (X, Y, Day, Plant).
    - land_sea_mask (np.ndarray): 2D array defining land (1) and sea (0) locations.
    - plant_params (dict): Plant-specific parameters.
    - plant_params_formulas (dict): Plant-specific formulas for CLIMSUITE calculations.
    - results_path (str): Path to the directory where the results will be saved.
    - plant (str): Name of the plant for which CLIMSUITE is being run.
    - winter_crops (list): List of winter crops.

    Note: This function processes climate data, assesses climate suitability, and saves the results in the specified format.
    """

    v = climate_config['options'].get('irrigation', 'n')
    irrigation = (v.lower() == 'y' if isinstance(v, str) and v.lower() in ['y', 'n'] else bool(np.mean([int(c) for c in v]) > 0.5) if isinstance(v, str) and
                    all(c in '01' for c in v) else bool(v) if isinstance(v, (int, float)) else False)
    crop_failure_code = 'rrpcf'

    rrpcf_files = sorted([os.path.join(os.path.join(climate_config['files']['output_dir'] + '_downscaled', area_name), f) for f in\
                          os.listdir(os.path.join(climate_config['files']['output_dir'] + '_downscaled', area_name)) if\
                            f.startswith(f'ds_{crop_failure_code}_{plant.lower()}_{"ir" if irrigation else "rf"}_')],\
                                key=lambda f: int(re.search(r'_(\d+)\.nc$', f).group(1))) #type:ignore
    """
    if len(rrpcf_files) and climate_config['climatevariability'].get('consider_variability', False):
        if os.path.exists(os.path.join(os.path.split(results_path)[0]+'_var', os.path.split(results_path)[1], plant, 'climate_suitability.tif'))\
            and os.path.exists(os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1], plant,'climate_suitability.tif')):
            return [os.path.join(os.path.split(results_path)[0]+'_var', os.path.split(results_path)[1]),\
                    os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1])]
    elif os.path.exists(os.path.join(os.path.split(results_path)[0]+'_novar', plant, 'climate_suitability.tif')):
        return [os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1])]
    """

    base, sub = os.path.split(results_path)
    val = int(climate_config['climatevariability'].get('consider_variability', 0))

    if rrpcf_files:
        tags = ['_var', '_novar'] if val == 2 else ['_var' if val == 1 else '_novar']
        paths = [os.path.join(base + t, sub) for t in tags]
        tif_paths = [os.path.join(p, plant, 'climate_suitability.tif') for p in paths]
        if all(os.path.exists(p) for p in tif_paths):
            return paths

    final_shape = temperature.shape

    # Dims: X, Y, Day, Plant
    length_of_growing_period = np.zeros((final_shape[0], final_shape[1]), dtype=np.int16)
    multiple_cropping = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
    start_growing_cycle = np.zeros((final_shape[0], final_shape[1]), dtype=np.int16)
    fuzzy_clim = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
    limiting_factor = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)

    water_mask = (land_sea_mask == 1)
    
    len_growing_cycle = int(plant_params[plant]['growing_cycle'][0])

    if 'wintercrop' in plant_params[plant].keys():
        wintercrop = plant_params[plant].get('wintercrop')[0].lower().strip() == 'y'
        if wintercrop:
            vernalization_period = int(plant_params[plant].get('vernalization_effective_days')[0])
            vernalization_tmax = float(plant_params[plant].get('vernalization_tmax')[0])
            vernalization_tmin = float(plant_params[plant].get('vernalization_tmin')[0])
            vernalization_days = float(plant_params[plant].get('days_to_vernalization')[0])
            if 'frost_resistance' in plant_params[plant].keys():
                frost_resistance = float(plant_params[plant].get('frost_resistance')[0])
                frost_resistance_period = int(plant_params[plant].get('frost_resistance_days')[0])
                vernalization_params = [vernalization_period, vernalization_tmax, vernalization_tmin, frost_resistance, frost_resistance_period, vernalization_days]
            else:
                vernalization_params = [vernalization_period, vernalization_tmax, vernalization_tmin, 0, 0, vernalization_days]
        else:
            vernalization_params = [0, 0, 0, 0, 0, 0]
    else:
        wintercrop, vernalization_params = False, [0, 0, 0, 0, 0, 0]
    
    if plant_params[plant].get('lethal_thresholds', [0])[0] == 1:
        try:
            lethal_params = [
                int(plant_params[plant].get('lethal_min_temp_duration', [0])[0]),
                int(plant_params[plant].get('lethal_min_temp', [0])[0]),
                int(plant_params[plant].get('lethal_max_temp_duration', [0])[0]),
                int(plant_params[plant].get('lethal_max_temp', [0])[0])]
            lethal = True
        except (ValueError, TypeError, IndexError):
            lethal, lethal_params = False, [0, 0, 0, 0]
    else:
        lethal, lethal_params = False, [0, 0, 0, 0]

    if 'photoperiod' in plant_params[plant].keys():
        photoperiod = plant_params[plant].get('photoperiod')[0] == 1
        if photoperiod:
            photoperiod_params = [int(plant_params[plant].get('minimum_sunlight_hours')[0]), int(plant_params[plant].get('maximum_sunlight_hours')[0]), extent[2], extent[0]]
        else:
            photoperiod_params = [0, 24, extent[1], extent[3]]
    else:
        photoperiod, photoperiod_params = False, [0, 0, 0, 0]

    if 'prec_req_after_sow' in plant_params[plant].keys():
        sowprec_params = [int(plant_params[plant].get('prec_req_after_sow')[0]), int(plant_params[plant].get('prec_req_days')[0])]
    else:
        sowprec_params = [20, 15]

    if 'lethal_min_prec_duration' in plant_params[plant].keys() and 'lethal_min_prec' in plant_params[plant].keys():
        max_consec_dry_days = int(plant_params[plant].get('lethal_min_prec_duration', 0)[0])
        dry_day_prec = int(plant_params[plant].get('lethal_min_prec', 0)[0])
    else:
        max_consec_dry_days = 0
        dry_day_prec = 0

    if 'lethal_max_prec' in plant_params[plant].keys() and 'lethal_max_prec_duration' in plant_params[plant].keys():
        max_prec_val = int(plant_params[plant].get('lethal_max_prec', 0)[0])
        max_prec_dur = int(plant_params[plant].get('lethal_max_prec_duration', 0)[0])
    else:
        max_prec_val = 0
        max_prec_dur = 0

    if 'temp_for_sow_duration' in plant_params[plant].keys() and 'temp_for_sow' in plant_params[plant].keys():
        dursowing = int(plant_params[plant].get('temp_for_sow_duration', 7)[0])
        sowingtemp = int(plant_params[plant].get('temp_for_sow', 5)[0])
    else:
        dursowing, sowingtemp = 7, 5

    additional_conditions = [cond for i in range(100) if (cond := plant_params[plant].get(f'AddCon:{i}')) is not None]
    for adcon in additional_conditions:
        if int(adcon[5]) == 0:
            additional_conditions.remove(adcon)

    no_threads, av_ram = dt.get_cpu_ram()
    pixeltoprocess = temperature.shape[0] * temperature.shape[1]
    wintercrop_factor = 0.5 if wintercrop else 1
    system_factor = {'darwin': 3, 'win32': 0.8}.get(sys.platform, 0.5)
    max_proc = int(np.clip(math.ceil(av_ram / (pixeltoprocess / 1e6)) * system_factor * wintercrop_factor, 1, no_threads-1))

    ### Climate Extremes / Climate Variability Module ###
    
    if (len(rrpcf_files) > 0) and int(climate_config['climatevariability'].get('consider_variability', False)) >= 1:
        use_cropfailures = True
        print('Module for the consideration of interannual climate variability is activated')
        print(f' -> Reading {"irrigated" if irrigation else "rainfed"} climate variability file for {plant}')
        crop_failures = nc.read_area_from_netcdf_list(rrpcf_files, overlap=False, extent=extent, dayslices=True)
        crop_failures = np.transpose(crop_failures, axes=(2, 0, 1))
        if crop_failures.shape[0] == 1:
            crop_failures = crop_failures[0]
            if crop_failures.shape != temperature.shape[:2]:
                crop_failures = dt.interpolate_array(crop_failures, temperature.shape, order=0)
        else:
            if crop_failures.transpose(1, 2, 0).shape != temperature.shape:
                crop_failures = dt.interpolate_array(crop_failures.transpose(1, 2, 0), temperature.shape, order=0).transpose(2, 0, 1)
        if np.max(crop_failures) <= 1: #type:ignore
            crop_failures *= 100 #type:ignore
        print(' -> Data loaded')
    else:
        use_cropfailures = False
        print('Module for the consideration of interannual climate variability is deactivated')
        crop_failures = np.zeros((365))

    phenology_params = []
    for key, values in plant_params[plant].items():
        if key.startswith('phen'):
            _, range_str, var_type = key.split('_')
            start, end = map(int, range_str.split('-'))
            phenology_params.append([var_type, start, end] + values)    

    if len([entry for entry in phenology_params if entry[0] == 'temp']) > 0:
        print('Temperature: Phenology definitions available - Overriding membership function')
    if len([entry for entry in phenology_params if entry[0] == 'prec']) > 0:
        print('Precipitation: Phenology definitions available - Overriding membership function')
        
    tmp = os.path.join(os.getcwd(), 'temp')
    os.makedirs(tmp, exist_ok=True)
    for fn in os.listdir(tmp):
        try:
            os.remove(os.path.join(tmp, fn))
        except:
            pass

    ### - ###

    def process_day_concfut(day):
        if crop_failures.ndim == 2: # type: ignore
            process_day_climsuit_memopt([day, len_growing_cycle, temperature, precipitation, plant_params_formulas, plant, wintercrop, water_mask,
                                         int(climate_config['options']['irrigation']), crop_failures, vernalization_params, lethal, lethal_params, sowprec_params,
                                         photoperiod, photoperiod_params, max_consec_dry_days, dry_day_prec, dursowing, sowingtemp, max_prec_val, max_prec_dur,
                                         additional_conditions, phenology_params])
        else:
            process_day_climsuit_memopt([day, len_growing_cycle, temperature, precipitation, plant_params_formulas, plant, wintercrop, water_mask,
                                         int(climate_config['options']['irrigation']), crop_failures[day], vernalization_params, lethal, lethal_params, sowprec_params,
                                         photoperiod, photoperiod_params, max_consec_dry_days, dry_day_prec, dursowing, sowingtemp, max_prec_val, max_prec_dur,
                                         additional_conditions, phenology_params])
    

    if len_growing_cycle >= 365:
        process_day_concfut(0)
        with rasterio.open(os.path.join(tmp, '0.tif')) as src:
            data = src.read()
            temperature = data[0].astype(np.int8)
            precipitation = data[1].astype(np.int8)
            failuresuit = data[2].astype(np.int8)
            sunshinesuit = data[3].astype(np.int8)
        del data
        
    else:
        cpl = False
        while not cpl:
            print(f'Limiting to {max_proc} cores')
            while True:
                if len(os.listdir(tmp)) >= 365:
                    break
                if max_proc == 1:
                    for day in range(365):
                        process_day_concfut(day) 
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_proc) as executor:
                        executor.map(process_day_concfut, range(365), chunksize=math.ceil(365 / max_proc))     
                break
            try:
                temperature, precipitation, failuresuit, sunshinesuit = read_tif_data_to_tempprecfail_arr((final_shape[0], final_shape[1], 365, 4))
                cpl = True
            except:
                print('Error reading climate suitability data from daily geotiff files. Retrying.')
                [os.remove(os.path.join(tmp, f)) for f in os.listdir(tmp)]
    del crop_failures

    
    if wintercrop:
        len_growing_cycle -= vernalization_days

    ret_paths = []
    for variability in [False] if int(climate_config['climatevariability'].get('consider_variability', 1)) == 0 else [True] if int(climate_config['climatevariability'].get('consider_variability', 1)) == 1 else [True, False]:
        if variability:
            if not use_cropfailures:
                continue
            print('Calculation of Climate Suitability with Consideration of Crop Failure Frequency')
            curr_fail = failuresuit 
            res_path = os.path.join(os.path.split(results_path)[0]+'_var', os.path.split(results_path)[1], plant)
        else:
            print('Calculation of Climate Suitability without Consideration of Crop Failure Frequency')
            curr_fail = np.full_like(temperature, 100)
            res_path = os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1], plant)

        os.makedirs(res_path, exist_ok=True)
        ret_paths.append(res_path)

        if os.path.exists(os.path.join(res_path, plant, 'climate_suitability.tif')):
            print(f' -> Suitability for {plant} already existing. Continuing.')
            continue

        print(' -> Calculating Suitability')
        fuzzy_clim_growing_cycle_wdoy = np.min([temperature, precipitation, curr_fail, sunshinesuit], axis=0)

        if climate_config['options'].get('consider_crop_rotation', False) and len_growing_cycle < 365:
            writesuit = np.copy(fuzzy_clim_growing_cycle_wdoy).astype(np.int8)
            writesuit[land_sea_mask == 0] = -1
            writesuit = np.transpose(writesuit, (1, 2, 0))
            top, left, bottom, right = extent
            height, width, depth = writesuit.shape
            transform = from_bounds(left, bottom, right, top, width, height)
            with rasterio.open(os.path.join(res_path, 'climatesuitability_temp.tif'), 'w', driver='GTiff', height=height, width=width, count=depth, dtype=rasterio.int8,
                               crs="EPSG:4326", transform=transform, nodata=-1,compress='LZW') as dst:
                for i in range(depth):
                    dst.write(writesuit[:, :, i], i + 1)
            del writesuit

        print(' -> Calculating Suitable Sowing Days')
        length_of_growing_period = ((fuzzy_clim_growing_cycle_wdoy > 0).astype(np.int16) * (365 if len_growing_cycle >= 365 else 1))
        if len_growing_cycle < 365:
            length_of_growing_period = length_of_growing_period.sum(axis=2)
            
        #growing_cycle_wdoy = len_growing_cycle
        
        if len_growing_cycle >= 365:
            fuzzy_clim[water_mask] = fuzzy_clim_growing_cycle_wdoy[water_mask]
            print(' -> Calculating Optimal Sowing Date')                
            start_growing_cycle = np.full_like(fuzzy_clim_growing_cycle_wdoy, 0)
            print(' -> Calculating Limiting Factor')
            limiting_factor = np.argmin([temperature, precipitation, curr_fail, sunshinesuit], axis=0).astype(np.int8)
            temp_suit = temperature
            prec_suit = precipitation
            cffq_suit = curr_fail
            photoperiod_suit = sunshinesuit
        else:
            print(' -> Calculating Optimal Sowing Date')
            if wintercrop:
                # Read Start of Vernalization Data to 3D Array
                optimal_sowing_date = read_tif_data_to_opt_sow_date_arr((final_shape[0], final_shape[1], 365))
                fuzzy_clim_growing_cycle_wdoy[optimal_sowing_date == -1] = 0

                fuzzy_clim[water_mask] = np.max(fuzzy_clim_growing_cycle_wdoy[water_mask], axis=1)
                start_growing_cycle[water_mask] = np.argmax(fuzzy_clim_growing_cycle_wdoy[water_mask, :], axis=1)

                # Save Start of Growing Cycle
                start_growing_cycle_without_vern = start_growing_cycle
                # Get optimal sowing date 
                start_growing_cycle = optimal_sowing_date[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle]
                start_growing_cycle[fuzzy_clim <= 0] = -1
                # Get sowing Date
                start_growing_cycle[start_growing_cycle > -1] = (start_growing_cycle[start_growing_cycle > -1] - vernalization_params[5]) % 365
                start_growing_cycle_without_vern[fuzzy_clim <= 0] = -1

                start_growing_cycle_without_vern[start_growing_cycle < 0] = -1
                fuzzy_clim[start_growing_cycle < 0] = 0

                del optimal_sowing_date
            else:
                fuzzy_clim[water_mask] = np.max(fuzzy_clim_growing_cycle_wdoy[water_mask], axis=1)
                start_growing_cycle = find_longest_run_center(fuzzy_clim_growing_cycle_wdoy)
                start_growing_cycle[~water_mask] = -1
                
                #start_growing_cycle[water_mask] = np.argmax(fuzzy_clim_growing_cycle_wdoy[water_mask, :], axis=1)

            # For determination of limiting factor:
            suit_sum = temperature.astype(np.int16) + precipitation.astype(np.int16) + curr_fail.astype(np.int16) + sunshinesuit.astype(np.int16)
            start_growing_cycle[water_mask & (start_growing_cycle <= 0)] = np.argmax(suit_sum[water_mask & (start_growing_cycle <= 0)], axis=1)

            print(' -> Calculating Limiting Factor')
            temp_suit = temperature[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
            prec_suit = precipitation[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
            cffq_suit = curr_fail[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
            photoperiod_suit = sunshinesuit[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
            limiting_factor = np.argmin([temp_suit, prec_suit, cffq_suit, photoperiod_suit], axis=0).astype(np.int8)

        start_growing_cycle[fuzzy_clim == 0] = -1

        limiting_factor[~water_mask] = -1
        collect()
            
        fuzzy_clim = np.clip(fuzzy_clim, 0, 100)
        
        # Wintercrops: Growing_Cycle = Growing_cycle + Vernalisation Period
        len_growing_cycle = len_growing_cycle + vernalization_params[0] if wintercrop else len_growing_cycle

        threshold_time = 365
        if climate_config['options'].get('consider_multiple_cropping', False):
            if not wintercrop:
                print(' -> Calculting Potential Multiple Cropping')
                if 'multiple_cropping_turnaround_time' in climate_config['options']:
                    turnaround_time = int(climate_config['options'].get('multiple_cropping_turnaround_time'), 21)
                else:
                    turnaround_time = 21
                multiple_cropping[length_of_growing_period < 4 * len_growing_cycle] = 3
                multiple_cropping[length_of_growing_period < 3 * len_growing_cycle] = 2
                multiple_cropping[length_of_growing_period < 2 * len_growing_cycle] = 1
                multiple_cropping[length_of_growing_period == 0] = 0
                multiple_cropping[multiple_cropping >= 3] = 3

                if (3 * len_growing_cycle + 3 * turnaround_time) > threshold_time:
                    multiple_cropping[multiple_cropping >= 2] = 2

                if (2 * len_growing_cycle + 2 * turnaround_time) <= 365 and climate_config['options'].get('output_all_startdates', True):
                    print(' -> Calculation of Sowing Days for Multiple Cropping')
                    start_days = np.empty(start_growing_cycle.shape + (4,), dtype=np.int16)       

                    def process_index(idx):
                        i, j = idx
                        suit_vals = fuzzy_clim_growing_cycle_wdoy[i, j].astype(np.int16)
                        start_idx, max_sum = find_max_sum_new(suit_vals, len_growing_cycle + turnaround_time, multiple_cropping[i, j])
                        
                        multiple_cropping[i, j] = min(1 if np.max(suit_vals) >= np.sum(suit_vals[start_idx]) else np.sum(suit_vals[start_idx] > 1), multiple_cropping[i, j])
                        if multiple_cropping[i, j] > 1:
                            if len(start_idx) == 2:
                                start_days[i, j, :] = [max_sum, start_idx[0], start_idx[1], -1]
                                multiple_cropping[i, j] = 2
                            else:
                                start_days[i, j, :] = [max_sum, start_idx[0], start_idx[1], start_idx[2]]
                                multiple_cropping[i, j] = 3
                    
                    valid_indices = np.argwhere(multiple_cropping >= 2)
                    print(f' -> Processing {len(valid_indices)} pixels...')

                    """
                    # DEBUG
                    for indices in valid_indices:
                        process_index(indices)
                    """
                    with concurrent.futures.ThreadPoolExecutor(max_workers=no_threads) as executor:
                        list(executor.map(process_index, valid_indices, chunksize=len(valid_indices)//no_threads))

                    start_days[..., 1:] += 1
                    start_days[multiple_cropping < 2, 1] = -1 
                    start_days[multiple_cropping < 2, 2] = -1
                    start_days[multiple_cropping < 3, 3] = -1 
                    start_days[multiple_cropping < 2, 0] = fuzzy_clim[multiple_cropping < 2]
                    start_days[land_sea_mask == 0, 0] = -1 

                    if climate_config['options']['output_format'] == 'geotiff' or climate_config['options']['output_format'] == 'cog':
                        if climate_config.get('outputs', {}).get('optimal_sowing_date_mc_first', 1):
                            dt.write_geotiff(res_path, 'optimal_sowing_date_mc_first.tif', start_days[..., 1]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
                        if climate_config.get('outputs', {}).get('optimal_sowing_date_mc_second', 1):
                            dt.write_geotiff(res_path, 'optimal_sowing_date_mc_second.tif', start_days[..., 2]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
                        if np.nanmax(start_days[..., 3]) > 0 and climate_config.get('outputs', {}).get('optimal_sowing_date_mc_third', 1):
                            dt.write_geotiff(res_path, 'optimal_sowing_date_mc_third.tif', start_days[..., 3]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
                        if climate_config.get('outputs', {}).get('climate_suitability_mc', 1):
                            dt.write_geotiff(res_path, 'climate_suitability_mc.tif', start_days[..., 0]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
                    elif climate_config['options']['output_format'] == 'netcdf4':
                        if climate_config.get('outputs', {}).get('optimal_sowing_date_mc_first', 1):
                            nc.write_to_netcdf(start_days[..., 1]*land_sea_mask, os.path.join(res_path, 'optimal_sowing_date_mc_first.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_mc_first', nodata_value=-1) #type:ignore
                        if climate_config.get('outputs', {}).get('optimal_sowing_date_mc_second', 1):
                            nc.write_to_netcdf(start_days[..., 2]*land_sea_mask, os.path.join(res_path, 'optimal_sowing_date_mc_second.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_mc_second', nodata_value=-1) #type:ignore
                        if np.nanmax(start_days[..., 3]) > 0 and climate_config.get('outputs', {}).get('optimal_sowing_date_mc_third', 1):
                            nc.write_to_netcdf(start_days[..., 3]*land_sea_mask, os.path.join(res_path, 'optimal_sowing_date_mc_third.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_mc_third', nodata_value=-1) #type:ignore
                        if climate_config.get('outputs', {}).get('climate_suitability_mc', 1):
                            nc.write_to_netcdf(start_days[..., 0]*land_sea_mask, os.path.join(res_path, 'climate_suitability_mc.nc'), extent=extent, compress=True, var_name='climate_suitability_mc', nodata_value=-1) #type:ignore
            else:
                print(' -> Skipping Calculation of Potential Multiple Cropping')
            
        fuzzy_clim[land_sea_mask == 0] = -1
        start_growing_cycle += 1
        start_growing_cycle[land_sea_mask == 0] = -1
        multiple_cropping[land_sea_mask == 0] = -1
        length_of_growing_period[land_sea_mask == 0] = -1
        limiting_factor[land_sea_mask == 0] = -1
        temp_suit[land_sea_mask == 0] = -1
        prec_suit[land_sea_mask == 0] = -1
        cffq_suit[land_sea_mask == 0] = -1
        photoperiod_suit[land_sea_mask == 0] = -1

        if climate_config['options']['output_all_limiting_factors']:
            all_array = np.asarray([temp_suit, prec_suit, cffq_suit, photoperiod_suit])
            all_array[np.isnan(all_array)] = -1
            all_array = all_array.astype(np.int8)
            dt.write_geotiff(res_path, 'all_climlim_factors.tif', np.transpose(all_array, (1, 2, 0)), extent, nodata_value=-1)
            del all_array

        if climate_config['options']['output_format'] == 'geotiff' or climate_config['options']['output_format'].lower() == 'cog':
            dt.write_geotiff(res_path, 'limiting_factor.tif', limiting_factor, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
            dt.write_geotiff(res_path, 'climate_suitability.tif', fuzzy_clim, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
            if wintercrop:
                if climate_config.get('outputs', {}).get('optimal_sowing_date_with_vernalization', True):
                    dt.write_geotiff(res_path, 'optimal_sowing_date_with_vernalization.tif', start_growing_cycle, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
                if climate_config.get('outputs', {}).get('start_growing_cycle_after_vernalization', True):
                    dt.write_geotiff(res_path, 'start_growing_cycle_after_vernalization.tif', start_growing_cycle_without_vern, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
            else:
                if climate_config.get('outputs', {}).get('optimal_sowing_date', True):
                    dt.write_geotiff(res_path, 'optimal_sowing_date.tif', start_growing_cycle, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
            if climate_config.get('outputs', {}).get('multiple_cropping', True):
                dt.write_geotiff(res_path, 'multiple_cropping.tif', multiple_cropping, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
            if climate_config.get('outputs', {}).get('suitable_sowing_days', True):
                dt.write_geotiff(res_path, 'suitable_sowing_days.tif', length_of_growing_period, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
        elif climate_config['options']['output_format'] == 'netcdf4':
            nc.write_to_netcdf(fuzzy_clim, os.path.join(res_path, 'climate_suitability.nc'), extent=extent, compress=True, var_name='climate_suitability', nodata_value=-1) #type:ignore
            nc.write_to_netcdf(limiting_factor.astype(np.uint8)+1, os.path.join(res_path, 'limiting_factor.nc'), extent=extent, compress=True, var_name='limiting_factor', nodata_value=-1) #type:ignore
            if wintercrop:
                if climate_config.get('outputs', {}).get('optimal_sowing_date_with_vernalization', True):
                    nc.write_to_netcdf(start_growing_cycle, os.path.join(res_path, 'optimal_sowing_date_with_vernalization.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_with_vernalization', nodata_value=-1) #type:ignore
                if climate_config.get('outputs', {}).get('start_growing_cycle_after_vernalization', True):
                    nc.write_to_netcdf(start_growing_cycle_without_vern, os.path.join(res_path, 'start_growing_cycle_after_vernalization.nc'), extent=extent, compress=True, var_name='start_growing_cycle_after_vernalization', nodata_value=-1) #type:ignore
            else:
                if climate_config.get('outputs', {}).get('optimal_sowing_date', True):
                    nc.write_to_netcdf(start_growing_cycle, os.path.join(res_path, 'optimal_sowing_date.nc'), extent=extent, compress=True, var_name='optimal_sowing_date', nodata_value=-1) #type:ignore
            if climate_config.get('outputs', {}).get('multiple_cropping', True):
                nc.write_to_netcdf(multiple_cropping, os.path.join(res_path, 'multiple_cropping.nc'), extent=extent, compress=True, var_name='multiple_cropping', nodata_value=-1) #type:ignore
            if climate_config.get('outputs', {}).get('suitable_sowing_days', True):
                nc.write_to_netcdf(length_of_growing_period, os.path.join(res_path, 'suitable_sowing_days.nc'), extent=extent, compress=True, var_name='suitable_sowing_days', nodata_value=-1) #type:ignore
        else:
            print('No output format specified.')

        length_of_growing_period = np.zeros((final_shape[0], final_shape[1]), dtype=np.int16)
        multiple_cropping = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
        start_growing_cycle = np.zeros((final_shape[0], final_shape[1]), dtype=np.int16)
        fuzzy_clim = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
        limiting_factor = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
        del curr_fail

    del temperature, precipitation, failuresuit
    for i in range(365):
        try:
            os.remove(os.path.join(tmp, f'{i}.npy'))
        except:
            pass
        try:
            os.remove(os.path.join(tmp, f'{i}.tif'))
            os.remove(os.path.join(tmp, f'{i}_osd.tif'))
        except:
            pass
    collect()

    ret_paths = [os.path.split(pt)[0] for pt in ret_paths]
    return ret_paths


def climate_suitability(climate_config, extent, temperature, precipitation, land_sea_mask, plant_params, plant_params_formulas, results_path, area_name) -> list:
    """
    Calculates climate suitability for multiple plants based on the CropSuite model.

    Parameters:
    - climate_config (dict): Configuration settings for the CLIMSUITE model.
    - extent (list): List containing the spatial extent of the analysis area in the format [min_lon, min_lat, max_lon, max_lat].
    - climatedata (np.ndarray): 4D array containing climate data in the format (X, Y, Day, Plant).
    - land_sea_mask (np.ndarray): 2D array defining land (1) and sea (0) locations.
    - plant_params (dict): Plant-specific parameters.
    - plant_params_formulas (dict): Plant-specific formulas for CLIMSUITE calculations.
    - results_path (str): Path to the directory where the results will be saved.

    Returns:
    - fuzzy_clim (np.ndarray): Array containing the fuzzy climate suitability values for each plant.
    - start_growing_cycle (np.ndarray): Array containing the optimal sowing dates for each plant.
    - length_of_growing_period (np.ndarray): Array containing the suitable sowing days for each plant.
    - limiting_factor (np.ndarray): Array containing the limiting factors for each plant.
    - multiple_cropping (np.ndarray): Array containing information about multiple cropping for each plant.

    Note: This function processes climate data for multiple plants using the CropSuite model and aggregates the results.
    """
    plant_list = [plant for plant in plant_params]
    ret_paths = []

    for idx, plant in enumerate(plant_params):
        val = int(climate_config['climatevariability'].get('consider_variability', 0))
        base, sub = os.path.split(results_path)
        tags = ['_novar', '_var'] if val == 2 else ['_novar' if val == 0 else '_var']
        ret_paths = [os.path.join(base + t, sub) for t in tags]
        if os.path.exists(os.path.join(ret_paths[-1], plant, 'climate_suitability.tif')):
            print(f' -> {plant} already created. Skipping')
            continue

        print(f'\nProcessing {plant} - {idx+1} out of {len(plant_list)} crops\n')
        climsuit_new(climate_config, extent, temperature, precipitation, land_sea_mask, plant_params, plant_params_formulas, results_path, plant, area_name)
        collect()

    print('Climate suitability calculation finished!\n\n')
    return ret_paths

    
