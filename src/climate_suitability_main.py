import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
import data_tools as dt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import rasterio
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')
from numba import jit
from gc import collect
import math
import nc_tools as nc

    
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


def process_day_climsuit_memopt(args):
    """
    Processes climate suitability data for a specific day and saves the results to a GeoTIFF file.

    Parameters:
    - args (tuple): Tuple containing the following elements:
    - day (int): Current day for processing climate data.
    - growing_cycle (int): Duration of the growing cycle.
    - climatedata (np.ndarray): Climate data array containing temperature and precipitation information.
    - plant_params_forumlas (dict): Dictionary containing plant formulas for temperature, precipitation, and failure frequency.
    - plant_list (list): List of plant names.
    - plant (str): Name of the plant for which suitability data is processed.
    - winter_crops (bool): winter crop
    - temp_mask (np.ndarray): Mask for temperature data.
    - irrigation (int): Flag indicating whether irrigation is applied (0 for no irrigation, 1 for irrigation).
    - crop_failures (np.ndarray): Array indicating frequency of crop failures.
    - vernalization_params: [vernalization_period, vernalization_tmax, vernalization_tmin, frost_resistance, frost_resistance_period, time_to_vernalization]
    Returns:
    - None

    Note: This function processes climate suitability data for a specific day, calculates suitability values based on
        specified formulas, and saves the results to a GeoTIFF file. It also handles memory optimization during processing.
    """
    day, growing_cycle, temperature, precipitation, plant_params_forumlas, plant, wintercrop, water_mask, irrigation,\
        crop_failures, vernalization_params, lethal, lethal_params, sowprec_params, photoperiod, photoperiod_params,\
            max_consec_dry_days, dursowing, sowingtemp, additional_conditions = args
    # vern_period is the time period, for which the vernalization requirements of the effective vernalization days are searched
    vern_period = 150
    temp_path = os.path.join(os.getcwd(), 'temp', f'{day}.tif')

    if os.path.exists(temp_path):
        sys.stdout.write(f' -> Skipping day #{day}              '+'\r')
        sys.stdout.flush()
        return

    sys.stdout.write(f' -> Processing day #{day}                                      '+'\r')
    sys.stdout.flush()

    if wintercrop:
        """
        start_of_vernalization = day - vern_period
        if start_of_vernalization < 0:
            window_temp_vern = np.concatenate((temperature[..., start_of_vernalization:], temperature[..., :day]), axis=2)
        else:
            window_temp_vern = temperature[..., start_of_vernalization:day]

        # Vernalization temperature thresholds
        vernalization_min_temp, vernalization_max_temp = vernalization_params[2] * 10, vernalization_params[1] * 10
        cumulative_conditions = np.cumsum((window_temp_vern <= vernalization_max_temp) & (window_temp_vern >= vernalization_min_temp), axis=2)

        # Find the optimal start date where cumulative_conditions first reaches vernalization requiremets
        opt_start_date = np.full(cumulative_conditions.shape[:-1], -1, dtype=np.int16)
        first_50_days = np.argmax(cumulative_conditions == vernalization_params[0], axis=2).astype(np.uint8)

        # Correct opt_start_date only where 50 days are met
        opt_start_date[np.any(cumulative_conditions >= vernalization_params[0], axis=2)] = vern_period - first_50_days[np.any(cumulative_conditions >= vernalization_params[0], axis=2)]
        del first_50_days, cumulative_conditions
        """

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

        """
        # Vectorized approach
        vern_start = opt_start_date[..., None]
        start_indices = (vern_start - days_to_vern) % 365

        # Create a full year array to handle wrap-around using concatenation
        temperature_full = np.concatenate((temperature, temperature), axis=2)
        precipitation_full = np.concatenate((precipitation, precipitation), axis=2)

        # Calculate indices for vectorized extraction
        rows, cols = np.indices((temperature.shape[0], temperature.shape[1]))
        rows = rows[..., None]
        cols = cols[..., None]
        start_indices_full = (start_indices + 365) % 365
        range_indices = np.arange(days_to_vern)

        # Extract temperature and precipitation data
        temp_to_vern = temperature_full[rows, cols, start_indices_full + range_indices]
        prec_to_vern = precipitation_full[rows, cols, start_indices_full + range_indices]
        """

        # Vectorized approach without duplicating arrays
        vern_start = opt_start_date[..., None]
        start_indices = (vern_start - days_to_vern) % 365

        # Calculate rows, cols, and range_indices once, without creating full arrays
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
        #vern_temp[water_mask == 1] = (get_suitability_val_dict(plant_params_forumlas, plant, 'temp', vern_temp[water_mask == 1] / 10) * 100).astype(np.int8)
        
        vern_temp[np.nanmean(temp_to_vern[..., :dursowing], axis=2) <= (sowingtemp * 10)] = -32767

        vern_prec = np.sum(prec_to_vern, axis=2, dtype=np.int16) / (days_to_vern / growing_cycle) # hypothetical precipitation over full growing cycle for shorter pre-vernalization

        # If no precipitation within the first 15 days after sowing -> BBCH1 Emergence: Set Suitability to 0
        # sowprec_params: [prec_req_after_sow, prec_req_days]
        vern_prec[np.sum(prec_to_vern[..., :sowprec_params[0]], axis=2, dtype=np.int16) <= sowprec_params[1] * 10] = 0
        del temp_to_vern, prec_to_vern

        opt_start_date[(vern_temp <= 0) | (vern_prec <= 0)] = -1

        with rasterio.open(temp_path[:-4]+'_osd.tif', 'w', driver='GTiff',  width=opt_start_date.shape[1], height=opt_start_date.shape[0], count=1, dtype=np.int16, compress='lzw') as dst:
            dst.write(opt_start_date, 1) 
        
        growing_cycle -= days_to_vern
        
    if day + growing_cycle <= 365:
        temperature = temperature[..., day:day+growing_cycle]
        precipitation = precipitation[..., day:day+growing_cycle] if irrigation == 0 else np.zeros_like(temperature)
    elif growing_cycle == 365:
        temperature = temperature[...]
        precipitation = precipitation[...] if irrigation == 0 else np.zeros_like(temperature)
    else:
        temperature = np.concatenate((temperature[..., day:], temperature[..., :day+growing_cycle-365]), axis=2)
        precipitation = np.concatenate((precipitation[..., day:], precipitation[..., :day+growing_cycle-365]), axis=2) if irrigation == 0 else np.zeros_like(temperature)

    temp = np.empty((temperature.shape[0], temperature.shape[1]), dtype=np.int16)
    temp[water_mask == 1] = np.mean(temperature[water_mask == 1], axis=1).astype(np.int16)

    if wintercrop:
        temp = ((temp * growing_cycle) + (vern_temp * vernalization_params[5])) / (growing_cycle + vernalization_params[5])

    temp[water_mask == 1] = (get_suitability_val_dict(plant_params_forumlas, plant, 'temp', temp[water_mask == 1] / 10) * 100)
    temp = temp.astype(np.int8)
    if not wintercrop and growing_cycle < 365:
        temp[np.nanmean(temperature[..., :dursowing], axis=2) <= (sowingtemp * 10)] = 0

    if irrigation == 0:
        prec = np.empty((temperature.shape[0], temperature.shape[1]), dtype=np.int32)
        prec[water_mask == 1] = np.sum(precipitation[water_mask == 1], axis=1)
        if wintercrop:
            prec = ((prec * growing_cycle) + (vern_prec * vernalization_params[5])) / (growing_cycle + vernalization_params[5])
        prec[water_mask == 1] = (get_suitability_val_dict(plant_params_forumlas, plant, 'prec', prec[water_mask == 1] / 10) * 100).astype(np.int8)
        prec = prec.astype(np.int8)
    else:
        prec = np.zeros_like(temp)

    if crop_failures.ndim > 1:
        failure_suit = np.empty((temperature.shape[0], temperature.shape[1]), dtype=np.int8)
        failure_suit[water_mask == 1] = (get_suitability_val_dict(plant_params_forumlas, plant, 'freqcropfail', crop_failures[water_mask == 1]/100) * 100).astype(np.int8)
    else:
        failure_suit = np.full_like(temp, 100)


    if irrigation == 0 and growing_cycle < 365 and not wintercrop:
        # sowprec_params: [prec_req_after_sow, prec_req_days]
        prec_after_sowing = np.empty((temperature.shape[0], temperature.shape[1]), dtype=np.uint16)
        prec_after_sowing[water_mask == 1] = np.sum(precipitation[water_mask == 1, :sowprec_params[1]], axis=1, dtype=np.uint16)
        prec[(water_mask == 1) & (prec_after_sowing < sowprec_params[1]*10)] = 0
        del prec_after_sowing
        collect()

    # Permafrost    
    temp[np.mean(temperature, axis=2) < 0] = 0
    collect()

    if photoperiod and growing_cycle < 365:
        # photoperiod_params [min_hours, max_hours, min_lat, max_lat]
        min_hours, max_hours, min_y, max_y = photoperiod_params
        sunshine_hours = calculate_average_sunshine((temp.shape[0], temp.shape[1]), (min_y, max_y), day, day+growing_cycle)
        sunshine_hours[(sunshine_hours > max_hours) | (sunshine_hours < min_hours)] = 0
        sunshine_hours[sunshine_hours > 0] = 100
    else:
        sunshine_hours = np.full_like(temp, 100, dtype = temp.dtype)
    if lethal:
        min_dur, min_tmp = lethal_params[0], lethal_params[1]
        max_dur, max_tmp = lethal_params[2], lethal_params[3]

        if min_dur != 0:
            bool_arr = temperature < min_tmp * 10
            sliding_windows = sliding_window_view(bool_arr, window_shape=min_dur, axis=2)
            mask = np.any(np.sum(sliding_windows, axis=-1) == min_dur, axis=-1)
            if wintercrop:
                opt_start_date[mask] = -1
            temp[mask] = 0

        if max_dur != 0:
            bool_arr = temperature > max_tmp * 10
            sliding_windows = sliding_window_view(bool_arr, window_shape=max_dur, axis=2)
            mask = np.any(np.sum(sliding_windows, axis=-1) == max_dur, axis=-1)
            if wintercrop:
                opt_start_date[mask] = -1
            temp[mask] = 0
    
    #if wintercrop:
        #vern_suit = (get_suitability_val_dict(plant_params_forumlas, plant, 'temp', vern_temp[water_mask == 1] / 10) * 100).astype(np.int8)
        #temp[temp > 0] = ((temp[temp > 0] * (growing_cycle)) + (vern_temp[temp > 0] * vernalization_params[5])) / (growing_cycle + vernalization_params[5])
        #temp[vern_temp <= 0] = 0
        #prec[prec > 0] = ((prec[prec > 0] * (growing_cycle)) + (vern_prec[prec > 0] * vernalization_params[5])) / (growing_cycle + vernalization_params[5])
        #prec[vern_prec <= 0] = 0
        #temp[opt_start_date == -1] = 0
        #del vern_prec, vern_temp

    if max_consec_dry_days > 0 and not irrigation == 1:
        bool_arr = precipitation <= 10 # less than 1 mm
        sliding_windows = sliding_window_view(bool_arr, window_shape=max_consec_dry_days, axis=2)
        consecutive_drydays = np.any(np.sum(sliding_windows, axis=-1) == max_consec_dry_days, axis=-1)
        prec[(water_mask == 1) & consecutive_drydays] = 0

    if irrigation == 1:
        prec.fill(100)

    if len(additional_conditions) > 0:
        for cond in additional_conditions:
            tp = cond[0].lower()
            start_day, end_day = int(cond[1]), int(cond[2])
            val = int(cond[4] * 10)
            operator = cond[3]

            if tp == 'temperature':
                if operator == '>=':
                    temp[np.nanmean(temperature[..., start_day:end_day], axis=2) >= val] = 0
                elif operator == '>':
                    temp[np.nanmean(temperature[..., start_day:end_day], axis=2) > val] = 0
                elif operator == '<':
                    temp[np.nanmean(temperature[..., start_day:end_day], axis=2) < val] = 0
                elif operator == '<=':
                    temp[np.nanmean(temperature[..., start_day:end_day], axis=2) <= val] = 0
            elif tp == 'precipitation':
                if operator == '>=':
                    prec[np.nansum(precipitation[..., start_day:end_day], axis=2) < val] = 0
                elif operator == '>':
                    prec[np.nansum(precipitation[..., start_day:end_day], axis=2) <= val] = 0
                elif operator == '<':
                    prec[np.nansum(precipitation[..., start_day:end_day], axis=2) >= val] = 0
                elif operator == '<=':
                    prec[np.nansum(precipitation[..., start_day:end_day], axis=2) > val] = 0

    with rasterio.open(temp_path, 'w', driver='GTiff',  width=temp.shape[1], height=temp.shape[0], count=4, dtype=np.int8, compress='lzw') as dst:
        dst.write(temp, 1)
        dst.write(prec, 2)
        dst.write(failure_suit, 3)
        dst.write(sunshine_hours, 4)

    if not os.path.exists(temp_path):
        collect()
        process_day_climsuit_memopt(args)
    else:
        if irrigation == 0:
            del temp, prec, sunshine_hours
        else:
            del temp, sunshine_hours
    collect()


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
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(read_and_process, range(365)))

    for day, result in enumerate(results):
        temp_arr[..., day, :] = result
    """
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

    irrigation = bool(int(climate_config['options']['irrigation']))
    crop_failure_code = 'rrpcf'
    rrpcf_files = [os.path.join(climate_config['files']['output_dir']+'_downscaled', area_name, f) for f in os.listdir(os.path.join(climate_config['files']['output_dir']+'_downscaled', area_name)) if f.startswith(f'ds_{crop_failure_code}_{plant.lower()}_{"ir" if irrigation else "rf"}_')]

    if len(rrpcf_files) and climate_config['climatevariability'].get('consider_variability', False):
        if os.path.exists(os.path.join(os.path.split(results_path)[0]+'_var', os.path.split(results_path)[1], plant, 'climate_suitability.tif'))\
            and os.path.exists(os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1], plant,'climate_suitability.tif')):
            return [os.path.join(os.path.split(results_path)[0]+'_var', os.path.split(results_path)[1]),\
                    os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1])]
    elif os.path.exists(os.path.join(os.path.split(results_path)[0]+'_novar', plant, 'climate_suitability.tif')):
        return [os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1])]


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

    if 'lethal_thresholds' in plant_params[plant].keys():
        lethal = plant_params[plant].get('lethal_thresholds')[0] == 1
        if lethal:
            min_dur = int(plant_params[plant].get('lethal_min_duration')[0])
            min_tmp = int(plant_params[plant].get('lethal_min_temp')[0])
            max_dur = int(plant_params[plant].get('lethal_max_duration')[0])
            max_tmp = int(plant_params[plant].get('lethal_max_temp')[0])
            lethal_params = [min_dur, min_tmp, max_dur, max_tmp]
        else:
            lethal_params = [0, 0, 0, 0]
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

    if 'consecutive_dry_days' in plant_params[plant].keys():
        max_consec_dry_days = int(plant_params[plant].get('consecutive_dry_days')[0])
    else:
        max_consec_dry_days = 0

    if 'temp_for_sow_duration' in plant_params[plant].keys() and 'temp_for_sow' in plant_params[plant].keys():
        dursowing = int(plant_params[plant].get('temp_for_sow_duration')[0])
        sowingtemp = int(plant_params[plant].get('temp_for_sow')[0])
    else:
        dursowing, sowingtemp = 7, 5
    additional_conditions = [cond for i in range(100) if (cond := plant_params[plant].get(f'AddCon:{i}')) is not None]

    no_threads, av_ram = dt.get_cpu_ram()

    area = abs((extent[3] - extent[1]) * (extent[0] - extent[2]))
    if wintercrop:
        factor = {'darwin': 10, 'win32': 20}.get(sys.platform, 60)
    else:
        factor = {'darwin': 6, 'win32': 12}.get(sys.platform, 10)
    if wintercrop:
        max_proc = int(np.clip(np.min([(no_threads - 1), ((1000 * av_ram) // (factor * area))]).astype(int), 1, no_threads))
    else:
        max_proc = np.clip(np.min([(no_threads - 1), ((1000 * av_ram) // (factor * area))]).astype(int), 1, no_threads)


    ### Climate Extremes / Climate Variability Module ###
    
    if (len(rrpcf_files) > 0) and climate_config['climatevariability'].get('consider_variability', False):
        use_cropfailures = True
        print('Module for the consideration of climate extremes is activated')
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
        print('Module for the consideration of climate extremes is deactivated')
        crop_failures = np.zeros((365))

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
                                         photoperiod, photoperiod_params, max_consec_dry_days, dursowing, sowingtemp, additional_conditions])
        else:
            process_day_climsuit_memopt([day, len_growing_cycle, temperature, precipitation, plant_params_formulas, plant, wintercrop, water_mask,
                                         int(climate_config['options']['irrigation']), crop_failures[day], vernalization_params, lethal, lethal_params, sowprec_params,
                                         photoperiod, photoperiod_params, max_consec_dry_days, dursowing, sowingtemp, additional_conditions])
    

    if len_growing_cycle >= 365:
        process_day_concfut(0)
        with rasterio.open(os.path.join(tmp, '0.tif')) as src:
            data = src.read()
            temperature = data[0].astype(np.int8)
            precipitation = data[1].astype(np.int8)
            failuresuit = data[2].astype(np.int8)
            sunshinesuit = data[3].astype(np.int8)

    else:
        print(f'Limiting to {max_proc} cores')
        while True:
            if len(os.listdir(tmp)) >= 365:
                break
            
            """
            #DEBUG
            for day in range(365):
                process_day_concfut(day) 
            """

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_proc) as executor:
                executor.map(process_day_concfut, range(365), chunksize=math.ceil(365 / max_proc))
            
            collect()        
            break
        try:
            temperature, precipitation, failuresuit, sunshinesuit = read_tif_data_to_tempprecfail_arr((final_shape[0], final_shape[1], 365, 4))
        except:
            dt.throw_exit_error('Error reading climate suitability data from daily geotiff files. Exit.')
        collect()

    del crop_failures
    collect()

    if wintercrop:
        len_growing_cycle -= vernalization_days

    if climate_config['options']['output_grow_cycle_as_doy']:
        fuzzy_temp_growing_cycle_wdoy = temperature
        fuzzy_precip_growing_cycle_wdoy = precipitation
        fuzzy_fail_growing_cycle_wdoy = failuresuit
        fuzzy_photop_growing_cycle_wdoy = sunshinesuit
    else:
        if len_growing_cycle >= 365:
            fuzzy_temp_growing_cycle_wdoy = temperature
            fuzzy_precip_growing_cycle_wdoy = precipitation
            fuzzy_fail_growing_cycle_wdoy = failuresuit
            fuzzy_photop_growing_cycle_wdoy = sunshinesuit
        else:
            fuzzy_temp_growing_cycle_wdoy = np.squeeze(np.mean(temperature[..., :364].reshape((final_shape[0], final_shape[1], 52, 7, -1)), axis=3)) # type:ignore
            fuzzy_precip_growing_cycle_wdoy = np.squeeze(np.mean(precipitation[..., :364].reshape((final_shape[0], final_shape[1], 52, 7, -1)), axis=3)) # type:ignore
            fuzzy_fail_growing_cycle_wdoy = np.squeeze(np.mean(failuresuit[..., :364].reshape((final_shape[0], final_shape[1], 52, 7, -1)), axis=3)) # type:ignore
            fuzzy_photop_growing_cycle_wdoy = np.squeeze(np.mean(sunshinesuit[..., :364].reshape((final_shape[0], final_shape[1], 52, 7, -1)), axis=3)) # type:ignore
    del temperature, precipitation, failuresuit, sunshinesuit
    collect()
    
    ret_paths = []

    for variability in ~np.unique([climate_config['climatevariability']['consider_variability'], False]) if len(np.unique([climate_config['climatevariability']['consider_variability'], False])) > 1 else np.unique([climate_config['climatevariability']['consider_variability'], False]):
        if variability:
            if not use_cropfailures:
                continue
            print('Calculation of Climate Suitability with Consideration of Crop Failure Frequency')
            curr_fail = fuzzy_fail_growing_cycle_wdoy 
            res_path = os.path.join(os.path.split(results_path)[0]+'_var', os.path.split(results_path)[1], plant)
        else:
            print('Calculation of Climate Suitability without Consideration of Crop Failure Frequency')
            curr_fail = np.full_like(fuzzy_temp_growing_cycle_wdoy, 100)
            res_path = os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1], plant)

        os.makedirs(res_path, exist_ok=True)
        ret_paths.append(res_path)

        if os.path.exists(os.path.join(res_path, plant, 'climate_suitability.tif')):
            print(f' -> Suitability for {plant} already existing. Continuing.')
            continue

        print(' -> Calculating Suitability')
        fuzzy_clim_growing_cycle_wdoy = np.min([fuzzy_temp_growing_cycle_wdoy, fuzzy_precip_growing_cycle_wdoy, curr_fail, fuzzy_photop_growing_cycle_wdoy], axis=0)
        print(' -> Calculating Suitable Sowing Days')
        if len_growing_cycle >= 365:
            length_of_growing_period = np.zeros_like(fuzzy_clim_growing_cycle_wdoy, dtype=np.int16)
            if climate_config['options']['output_grow_cycle_as_doy']:
                length_of_growing_period[fuzzy_clim_growing_cycle_wdoy > 0] = 365
            else:
                length_of_growing_period[fuzzy_clim_growing_cycle_wdoy > 0] = 52
        else:
            length_of_growing_period = (fuzzy_clim_growing_cycle_wdoy > 0).sum(axis=2).astype(np.int16)
        
        growing_cycle_wdoy = len_growing_cycle if climate_config['options']['output_grow_cycle_as_doy'] else len_growing_cycle // 7
        
        if len_growing_cycle >= 365:
            fuzzy_clim[water_mask] = fuzzy_clim_growing_cycle_wdoy[water_mask]
            print(' -> Calculating Optimal Sowing Date')                
            start_growing_cycle = np.full_like(fuzzy_clim_growing_cycle_wdoy, 0)
            print(' -> Calculating Limiting Factor')
            limiting_factor = np.argmin([fuzzy_temp_growing_cycle_wdoy, fuzzy_precip_growing_cycle_wdoy, curr_fail, fuzzy_photop_growing_cycle_wdoy], axis=0).astype(np.int8)
            temp_suit = fuzzy_temp_growing_cycle_wdoy
            prec_suit = fuzzy_precip_growing_cycle_wdoy
            cffq_suit = curr_fail
            photoperiod_suit = fuzzy_photop_growing_cycle_wdoy
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
                start_growing_cycle[water_mask] = np.argmax(fuzzy_clim_growing_cycle_wdoy[water_mask, :], axis=1)

            # For determination of limiting factor:
            suit_sum = fuzzy_temp_growing_cycle_wdoy.astype(np.int16) + fuzzy_precip_growing_cycle_wdoy.astype(np.int16) + curr_fail.astype(np.int16) + fuzzy_photop_growing_cycle_wdoy.astype(np.int16)
            start_growing_cycle[water_mask & (start_growing_cycle <= 0)] = np.argmax(suit_sum[water_mask & (start_growing_cycle <= 0)], axis=1)

            print(' -> Calculating Limiting Factor')
            temp_suit = fuzzy_temp_growing_cycle_wdoy[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
            prec_suit = fuzzy_precip_growing_cycle_wdoy[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
            cffq_suit = curr_fail[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
            photoperiod_suit = fuzzy_photop_growing_cycle_wdoy[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
            limiting_factor = np.argmin([temp_suit, prec_suit, cffq_suit, photoperiod_suit], axis=0).astype(np.int8)

        start_growing_cycle[fuzzy_clim == 0] = -1

        limiting_factor[~water_mask] = -1
        collect()
            
        fuzzy_clim = np.clip(fuzzy_clim, 0, 100)
        
        # Wintercrops: Growing_Cycle = Growing_cycle + Vernalisation Period
        growing_cycle_wdoy = growing_cycle_wdoy + vernalization_params[0] if wintercrop else growing_cycle_wdoy

        
        threshold_time = 365 if climate_config['options']['output_grow_cycle_as_doy'] else 52
        if not wintercrop:
            print(' -> Calculting Potential Multiple Cropping')
            if 'multiple_cropping_turnaround_time' in climate_config['options']:
                try:
                    turnaround_time = int(climate_config['options'].get('multiple_cropping_turnaround_time')) #type:ignore
                except:
                    turnaround_time = 21 if climate_config['options']['output_grow_cycle_as_doy'] else 3
            else:
                turnaround_time = 21 if climate_config['options']['output_grow_cycle_as_doy'] else 3
            multiple_cropping[length_of_growing_period < 4 * growing_cycle_wdoy] = 3
            multiple_cropping[length_of_growing_period < 3 * growing_cycle_wdoy] = 2
            multiple_cropping[length_of_growing_period < 2 * growing_cycle_wdoy] = 1
            multiple_cropping[length_of_growing_period == 0] = 0
            multiple_cropping[multiple_cropping >= 3] = 3

            if (3 * growing_cycle_wdoy + 3 * turnaround_time) > threshold_time:
                multiple_cropping[multiple_cropping >= 2] = 2

            if (2 * growing_cycle_wdoy + 2 * turnaround_time) <= 365 and climate_config['options']['output_all_startdates']:
                print(' -> Calculation of Sowing Days for Multiple Cropping')
                start_days = np.empty(start_growing_cycle.shape + (4,), dtype=np.int16)       

                def process_index(idx):
                    i, j = idx
                    suit_vals = fuzzy_clim_growing_cycle_wdoy[i, j].astype(np.int16)
                    start_idx, max_sum = find_max_sum_new(suit_vals, growing_cycle_wdoy + turnaround_time, multiple_cropping[i, j])
                    
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
                    dt.write_geotiff(res_path, 'optimal_sowing_date_mc_first.tif', start_days[..., 1]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
                    dt.write_geotiff(res_path, 'optimal_sowing_date_mc_second.tif', start_days[..., 2]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
                    if np.nanmax(start_days[..., 3]) > 0:
                        dt.write_geotiff(res_path, 'optimal_sowing_date_mc_third.tif', start_days[..., 3]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
                    dt.write_geotiff(res_path, 'climate_suitability_mc.tif', start_days[..., 0]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
                elif climate_config['options']['output_format'] == 'netcdf4':
                    nc.write_to_netcdf(start_days[..., 1]*land_sea_mask, os.path.join(res_path, 'optimal_sowing_date_mc_first.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_mc_first', nodata_value=-1) #type:ignore
                    nc.write_to_netcdf(start_days[..., 2]*land_sea_mask, os.path.join(res_path, 'optimal_sowing_date_mc_second.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_mc_second', nodata_value=-1) #type:ignore
                    if np.nanmax(start_days[..., 3]) > 0:
                        nc.write_to_netcdf(start_days[..., 3]*land_sea_mask, os.path.join(res_path, 'optimal_sowing_date_mc_third.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_mc_third', nodata_value=-1) #type:ignore
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

        #np.save(os.path.join(res_path, 'fuzzy_clim.npy'), fuzzy_clim)
        #np.save(os.path.join(res_path, 'optimal_sowing_date.npy'), start_growing_cycle)
        #np.save(os.path.join(res_path, 'multiple_cropping.npy'), multiple_cropping)
        #np.save(os.path.join(res_path, 'suitable_sowing_days.npy'), length_of_growing_period)
        #np.save(os.path.join(res_path, 'limiting_factor.npy'), limiting_factor)

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
                dt.write_geotiff(res_path, 'optimal_sowing_date_with_vernalization.tif', start_growing_cycle, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
                dt.write_geotiff(res_path, 'start_growing_cycle_after_vernalization.tif', start_growing_cycle_without_vern, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
            else:
                dt.write_geotiff(res_path, 'optimal_sowing_date.tif', start_growing_cycle, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
            dt.write_geotiff(res_path, 'multiple_cropping.tif', multiple_cropping, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
            dt.write_geotiff(res_path, 'suitable_sowing_days.tif', length_of_growing_period, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
        elif climate_config['options']['output_format'] == 'netcdf4':
            nc.write_to_netcdf(fuzzy_clim, os.path.join(res_path, 'climate_suitability.nc'), extent=extent, compress=True, var_name='climate_suitability', nodata_value=-1) #type:ignore
            nc.write_to_netcdf(limiting_factor.astype(np.uint8)+1, os.path.join(res_path, 'limiting_factor.nc'), extent=extent, compress=True, var_name='limiting_factor', nodata_value=-1) #type:ignore
            if wintercrop:
                nc.write_to_netcdf(start_growing_cycle, os.path.join(res_path, 'optimal_sowing_date_with_vernalization.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_with_vernalization', nodata_value=-1) #type:ignore
                nc.write_to_netcdf(start_growing_cycle_without_vern, os.path.join(res_path, 'start_growing_cycle_after_vernalization.nc'), extent=extent, compress=True, var_name='start_growing_cycle_after_vernalization', nodata_value=-1) #type:ignore
            else:
                nc.write_to_netcdf(start_growing_cycle, os.path.join(res_path, 'optimal_sowing_date.nc'), extent=extent, compress=True, var_name='optimal_sowing_date', nodata_value=-1) #type:ignore
            nc.write_to_netcdf(multiple_cropping, os.path.join(res_path, 'multiple_cropping.nc'), extent=extent, compress=True, var_name='multiple_cropping', nodata_value=-1) #type:ignore
            nc.write_to_netcdf(length_of_growing_period, os.path.join(res_path, 'suitable_sowing_days.nc'), extent=extent, compress=True, var_name='suitable_sowing_days', nodata_value=-1) #type:ignore
        else:
            print('No output format specified.')

        length_of_growing_period = np.zeros((final_shape[0], final_shape[1]), dtype=np.int16)
        multiple_cropping = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
        start_growing_cycle = np.zeros((final_shape[0], final_shape[1]), dtype=np.int16)
        fuzzy_clim = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
        limiting_factor = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
        del curr_fail

    del fuzzy_temp_growing_cycle_wdoy, fuzzy_precip_growing_cycle_wdoy, fuzzy_fail_growing_cycle_wdoy
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
    final_shape = temperature.shape
    ret_paths = []

    for idx, plant in enumerate(plant_params):
        res_path = os.path.join(results_path, plant)
        if climate_config['climatevariability'].get('consider_variability', False):
            ret_paths = [os.path.join(os.path.split(results_path)[0]+'_var',\
                                    os.path.split(results_path)[1]), os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1])]
            if os.path.exists(os.path.join(ret_paths[0], plant, 'climate_suitability.tif')) and os.path.exists(os.path.join(ret_paths[1], plant, 'climate_suitability.tif')):
                print(f' -> {plant} already created. Skipping')
                continue
        else:
            ret_paths = [os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1])]
            if os.path.exists(os.path.join(ret_paths[0], plant, 'climate_suitability.tif')):
                print(f' -> {plant} already created. Skipping')
                continue

        print(f'\nProcessing {plant} - {idx+1} out of {len(plant_list)} crops\n')
        climsuit_new(climate_config, extent, temperature, precipitation, land_sea_mask, plant_params, plant_params_formulas, results_path, plant, area_name)
        collect()

    print('Climate suitability calculation finished!\n\n')
    return ret_paths

    
