from scipy.interpolate import interp1d, interp2d, CubicSpline, PPoly
from scipy.ndimage import zoom
from rasterio.transform import from_bounds
import os
import numpy as np
import rasterio
import data_tools as dt
from matplotlib import path
import nc_tools as nc


def get_formula(x_vals, y_vals, method):
    """
        Given two arrays of numerical values x_vals and y_vals representing data points, and a method integer value,
        this function returns a formula that approximates the data points based on the selected interpolation method.

        Args:
            x_vals (array-like): Array of numerical values representing the x-coordinates of the data points.
            y_vals (array-like):Array of numerical values representing the y-coordinates of the data points.
            method (int):  Integer value that determines the interpolation method to be used. The available methods are:
                0 - Linear interpolation
                1 - Cubic interpolation
                2 - Quadratic interpolation
                3 - Cubic spline interpolation
                4 - Piecewise polynomial interpolation
                5 - Spline interpolation
        Returns:
            formula (array-like): An array containing the formula generated by the selected interpolation method, the minimum value of x_vals,
                and the maximum value of x_vals. If an error occurs during the interpolation process, a linear interpolation
                formula is returned instead.
    """
    if len(x_vals) > 3:
        try:
            if method == 0:  
                return [interp1d(x_vals, y_vals, kind='linear'), min(x_vals), max(x_vals)]
            elif method == 4:
                return [PPoly(x_vals, y_vals), min(x_vals), max(x_vals)]
            elif method == 1:
                return [interp1d(x_vals, y_vals, kind='cubic'), min(x_vals), max(x_vals)]
            elif method == 2:
                return [interp1d(x_vals, y_vals, kind='quadratic'), min(x_vals), max(x_vals)]
            elif method == 3:
                return [CubicSpline(x_vals, y_vals), min(x_vals), max(x_vals)]
            elif method == 5:
                return [interp1d(x_vals, y_vals, kind='slinear'), min(x_vals), max(x_vals)]
        except:
            return [interp1d(x_vals, y_vals, kind='linear'), min(x_vals), max(x_vals)]
    else:
        return [interp1d(x_vals, y_vals, kind='linear'), min(x_vals), max(x_vals)]


def get_shape_of_raster(raster_file):
    """
    Returns the shape (dimensions) of a raster file.

    Parameters:
    - raster_file (str): Path to the raster file.

    Returns:
    - shape (tuple): Tuple representing the shape (dimensions) of the raster file.
    """
    return(np.shape(raster_file))


def extract_domain_from_global_raster(raster_dataset, domain, raster_extent=[-180, 90, 180, -90]):
    """
    Extracts a subdomain from a global raster dataset based on specified domain coordinates.

    Parameters:
    - raster_dataset (np.ndarray): Global raster dataset.
    - domain (list): List representing the domain coordinates in the format [left, upper, right, lower].
    - raster_extent (list): List representing the extent of the global raster in the format [left, upper, right, lower].

    Returns:
    - subdomain (np.ndarray): Subdomain extracted from the global raster dataset.

    Note: The function calculates the pixel coordinates of the specified subdomain within the global raster dataset
    and extracts the corresponding subdomain.
    """
    if len(np.shape(raster_dataset)) == 2:
        big_rows, big_cols = get_shape_of_raster(raster_dataset) # Großes Raster Zeilen und Spalten
    else:
        big_rows, big_cols = get_shape_of_raster(raster_dataset)[1:3]

    big_uy = raster_extent[1] # Großes Raster oben
    big_by = raster_extent[3] # Großes Raster unten
    big_lx = raster_extent[0] # Großes Raster links
    big_rx = raster_extent[2] # Großes Raster rechts

    small_uy = domain[1] # Kleines Raster oben
    small_by = domain[3] # Kleines Raster unten
    small_lx = domain[0] # Kleines Raster links
    small_rx = domain[2] # Kleines Raster rechts

    pix_top = int(np.round((big_rows/(big_uy - big_by)) * (small_uy-big_by)))
    pix_bottom = int(np.round((big_rows/(big_uy - big_by)) * (small_by-big_by)))

    pix_left = int(np.round((big_cols/(big_rx - big_lx)) * (small_lx - big_lx)))
    pix_right = int(np.round((big_cols/(big_rx - big_lx)) * (small_rx - big_lx)))
    
    if len(np.shape(raster_dataset)) == 2:
        return raster_dataset[big_rows-pix_top:big_rows-pix_bottom, pix_left:pix_right]
    else:
        return raster_dataset[:, big_rows-pix_top:big_rows-pix_bottom, pix_left:pix_right]


def resize_array(array, shape):
    """
    Resizes a 2D array to the specified shape using mean pooling.

    Parameters:
    - array (np.ndarray): 2D array to be resized.
    - shape (tuple): Tuple representing the target shape (dimensions) in the format (rows, columns).

    Returns:
    - resized_array (np.ndarray): Resized array with the specified shape.

    Note: The function reshapes the input array by applying mean pooling to achieve the target shape.
    """
    sh = shape[0], array.shape[0]//shape[0], shape[1], array.shape[1]//shape[1]
    return array.reshape(sh).mean(-1).mean(1)


def fill_nodata_nearest(array, nodata):
    mask = array == nodata
    array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask])
    return array, mask


def interpolate_nanmask(array, new_shape):
    """
    Interpolate a binary mask with NaN values to a new specified shape.

    Parameters:
    - array (numpy.ndarray): Input binary mask with NaN values to be interpolated.
    - new_shape (tuple): A tuple specifying the new shape of the interpolated mask (height, width).

    Returns:
    - numpy.ndarray: Interpolated binary mask with the specified shape.

    Note:
    - The function assumes 'array' is a 2D numpy array representing a binary mask with NaN values.
    - It performs linear interpolation using the 'interp2d' function from scipy.
    - The resulting interpolated values are thresholded to create a binary mask.
    """
    return dt.interpolate_nanmask(array, new_shape)


def resize_array_interp(array, new_shape, nodata=None, limit=(-9999, -9999), method='linear'):
    """
    Resizes a 2D array to the specified new shape using linear interpolation.

    Parameters:
    - array (np.ndarray): 2D array to be resized.
    - new_shape (tuple): Tuple representing the target shape (dimensions) in the format (rows, columns).
    - limit (tuple, optional): Tuple containing the lower and upper limits for the resized array values. Default is (-9999, -9999).

    Returns:
    - resized_array (np.ndarray): Resized array with the specified new shape using linear interpolation.

    Note: The function uses linear interpolation to resize the input array to the new shape.
    Values outside the specified limits are set to 0 if limits are provided.
    """
    """
    if not nodata is None:
        array, nanmask = fill_nodata_nearest(array, nodata)

    h, w = array.shape

    if method == 'nearest':
        zoom_factors = (new_shape[0] / h, new_shape[1] / w)
        ret = zoom(array, zoom_factors, order=0).astype(float)
    else:
        interp_func = interp2d(np.arange(w), np.arange(h), array, kind='linear')
        ret = interp_func(np.linspace(0, w - 1, int(new_shape[1])), np.linspace(0, h - 1, int(new_shape[0])))
    if limit[0] != -9999: ret[ret<=-limit[0]] = 0
    if limit[1] != -9999: ret[ret>=limit[1]] = 0

    if not nodata is None:
        nanmask = interpolate_nanmask(nanmask, new_shape) # type: ignore
        ret += np.where(nanmask, np.nan, 0.0)
    return ret, None if not nodata else np.nan
    """

    return dt.resize_array_interp(array, new_shape, nodata=None, limit=(-9999, -9999))


def get_suitability_val(forms, plant_id, form_type, value):
    """
    Calculates the suitability value based on the provided formulas for a specific plant and form type.

    Parameters:
    - forms (dict): Dictionary containing formulas for different plants and form types.
    Structure: {plant_id: {form_type: (function, min_value, max_value)}}
    - plant_id (str): Identifier for the plant.
    - form_type (str): Type of the form for which the suitability value is calculated.
    - value (float or np.ndarray): Input value or array of values to be used in the formula.

    Returns:
    - suitability_value (float or np.ndarray): Suitability value calculated based on the provided formula.

    Note: The function retrieves the formula, minimum, and maximum values from the forms dictionary
    and applies them to the input value or array to calculate the suitability value.
    """
    func, min_val, max_val = forms[plant_id][form_type]
    if isinstance(value, np.ndarray):
        value = np.clip(value, min_val, max_val)
        return func(value)
    else:
        value = min(max_val, max(min_val, value))
        return func(value)


def get_suitability_val_dict(forms, plant, form_type, value):
    """
    Calculate the suitability value for a given form type and value based on predefined formulas.

    Parameters:
    - forms (dict): A dictionary containing predefined formulas for different forms, plants, and form types.
    - plant (str): The plant for which the suitability value is calculated.
    - form_type (str): The type of form for which the suitability value is calculated.
    - value (float or numpy.ndarray): The input value or array of values for the specified form type.

    Returns:
    - numpy.ndarray: The suitability values calculated based on the predefined formula.

    Note:
    - The function uses predefined formulas stored in the 'forms' dictionary to calculate suitability values.
    - It handles both scalar and array input for the 'value' parameter.
    - The output is clipped between 0 and 1 to ensure it falls within the suitability range.
    - If 'value' is a numpy array, NaN values are preserved using a nanmask during the calculations.
    """
    replacements = {'base_saturation': 'base_sat', 'coarse_fragments': 'coarsefragments', 'salinity': 'elco',
                    'sodicity': 'esp', 'pH': 'ph', 'soil_organic_carbon': 'organic_carbon'}
    if form_type in replacements:
        form_type = replacements[form_type]
    func, min_val, max_val = forms[plant][form_type]['formula'], forms[plant][form_type]['min_val'], forms[plant][form_type]['max_val']
    
    if isinstance(value, np.ndarray):
        nanmask = np.isnan(value)
        value = np.clip(value, min_val, max_val)
        value[value >= max_val] = max_val
        value[value <= min_val] = min_val
        try:
            value = np.clip(func(value), 0, 1)
        except:
            value = np.full_like(value, 1)
        value[nanmask] = -0.01
        return value
    else:
        value = min(max_val, max(min_val, value))
        return np.clip(func(value), 0, 1)
    

def get_geotiff_extent(filepath):
    """
    Retrieve the geographic extent (bounding box) of a GeoTIFF file.

    Parameters:
    - filepath (str): The path to the GeoTIFF file.

    Returns:
    - list: A list representing the geographic extent in the format [left, top, right, bottom].

    Note:
    - The function uses the 'rasterio' library to open the GeoTIFF file and extract its bounds.
    - The extent is returned in the format [left, top, right, bottom].
    """
    with rasterio.open(filepath) as dataset:
        bounds = dataset.bounds
        extent = [bounds.left, bounds.top, bounds.right, bounds.bottom]
    return extent


def aggregate_soil_raster(path, domain, final_shape, type=1, conversion_factor = 1, weighting = [2., 1.5, 1., 0.75, 0.5, 0.25], resolution = 1000, output_tif = False):
    output_tif = True
    """
    Aggregate soil raster data based on specified criteria.

    Parameters:
    - path (str): The path to the directory containing soil raster files.
    - domain (str): The geographic domain or area of interest.
    - final_shape (tuple): A tuple specifying the final shape of the aggregated array (rows, columns).
    - type (int): The type of aggregation to perform (0, 1, 2, or 3).
    - conversion_factor (float): A conversion factor to apply to the aggregated result.
    - weighting (list): A list of weighting factors for different soil layers (used in type 2 aggregation).
    - resolution (int): The resolution of the raster data.

    Returns:
    - numpy.ndarray: The aggregated soil data.

    Note:
    - The function aggregates soil data based on specified criteria and different types of aggregation.
    - 'type' parameter determines the aggregation strategy: 0 for single layer, 1 for mean of three layers, 2 for weighted sum of six layers, 3 for custom aggregation.
    """
    resolution = str(resolution)
    tif_list = filter_list_by_ending(os.listdir(path))

    if type == 3:
        if len(tif_list) >= 6:
            tif_list = sorted(tif_list, key=lambda x: int(x.split('_')[1].split('-')[0]))
            type = 2
        elif len(tif_list) >= 3:
            tif_list = sorted(tif_list, key=lambda x: int(x.split('_')[1].split('-')[0]))
            type = 1
        elif len(tif_list) == 1:
            type = 0
        else:
            input('ERROR: Missing Soil Files!\nExit')
            exit()

    if type == 0:
        fn = os.path.join(path, tif_list[0])
        extent = get_geotiff_extent(fn)
        with rasterio.open(fn) as src:
            layer = src.read(1)
            layer = extract_domain_from_global_raster(layer, domain, extent)
        if layer.shape != final_shape:
            layer, _ = resize_array_interp(layer, final_shape, method='nearest')
        return layer / conversion_factor #type:ignore

    elif type == 1:
        layer_files = [os.path.join(path, tif) for tif in tif_list[:3]]
        extent = get_geotiff_extent(layer_files[0])
        layers = []
        for layer_file in layer_files:
            with rasterio.open(layer_file) as src:
                layer = src.read(1)
                layer = extract_domain_from_global_raster(layer, domain, extent)
                layers.append(layer)
        mean = np.nanmean(layers, axis=0)
        if mean.shape != final_shape:
            mean = resize_array_interp(mean, final_shape, method='nearest')
        return mean / conversion_factor # type: ignore
    
    elif type == 2:
        layer_files = [os.path.join(path, tif) for tif in tif_list[:6]]
        layers = []
        extent = get_geotiff_extent(layer_files[0])
        for layer_file in layer_files:
            with rasterio.open(layer_file) as src:
                layer = src.read(1)
                layers.append(extract_domain_from_global_raster(layer, domain, extent))
        new_file = np.zeros((layers[0].shape))
        new_file = new_file.astype(float)
        new_file = layers[0] * 5 * weighting[0] # 0-5
        new_file += layers[1] * 10 * weighting[0] # 5-15
        new_file += layers[2] * 10 * weighting[0] # 15-25
        new_file += layers[2] * 5 * weighting[1] # 25-30
        new_file += layers[3] * 20 * weighting[1] # 30-50
        new_file += layers[3] * 10 * weighting[2] # 50-60
        new_file += layers[4] * 15 * weighting[2] # 60-75
        new_file += layers[4] * 25 * weighting[3] # 75-100
        new_file += layers[5] * 25 * weighting[4] # 100-125
        new_file += layers[5] * 25 * weighting[5] # 125-150
        if new_file.shape != final_shape:
            new_file = resize_array_interp(new_file, final_shape, method='nearest')

        if output_tif:
            ret_soil = new_file / 150 / conversion_factor #type:ignore
            dt.write_geotiff(path, tif_list[0].split('_')[0].split('.')[0]+'_combined.tif', ret_soil, domain)
            return ret_soil
        else:
            return new_file / 150 / conversion_factor  # type: ignore

    else:
        return np.zeros(final_shape)
    
def throw_exit_error(text):
    """
    Display an error message, wait for user input, and exit the program.

    Parameters:
    - text (str): The error message to be displayed.

    Note:
    - The function displays the specified error message, waits for user input, and then exits the program.
    """
    input(text+'\nExit')
    exit()

def get_id_list_start(dict, starts_with):
    """
    Get a list of IDs from a dictionary where the ID starts with a specified prefix.

    Parameters:
    - dictionary (dict): The input dictionary containing IDs as keys.
    - starts_with (str): The prefix to filter IDs.

    Returns:
    - list: A list of IDs from the dictionary that start with the specified prefix.

    Note:
    - The function iterates through the keys of the dictionary and includes IDs that start with the specified prefix.
    """
    lst = []
    for id, __ in dict.items():
        if str(id).startswith(starts_with):
            lst.append(id)
    return lst

def filter_list_by_ending(list, ending=('.tif', '.tiff')):
    """
    Filter a list of strings to include only those ending with specified endings.

    Parameters:
    - input_list (list): The input list of strings to be filtered.
    - ending (tuple): A tuple of string endings to filter by.

    Returns:
    - list: A filtered list containing only strings that end with the specified endings.

    Note:
    - The function uses list comprehension to filter the input list based on specified string endings.
    """
    return [entry for entry in list if entry.endswith(ending)]

def get_soil_data(climate_config, current_parameter, domain, shape, itype):
    """
    Get soil data based on specified climate configuration parameters.

    Parameters:
    - climate_config (dict): The climate configuration dictionary containing parameter settings.
    - current_parameter (str): The current parameter for which soil data is retrieved.
    - domain (str): The geographic domain or area of interest.
    - shape (tuple): A tuple specifying the shape of the output data (rows, columns).
    - itype: The numpy data type for the output array.

    Returns:
    - numpy.ndarray: The soil data array.

    Note:
    - The function retrieves soil data based on specified climate configuration parameters.
    - It handles various configurations, including data directories, weighting methods, conversion factors, and nodata values.
    - The retrieved soil data is aggregated using the 'aggregate_soil_raster_lst' function from your module.
    """
    try:
        current_dict = climate_config[f'parameters.{current_parameter}']
    except:
        throw_exit_error(f'{current_parameter} is not defined')
        current_dict = {}
    
    data_dir = current_dict['data_directory']
    weighting_method = int(current_dict['weighting_method'])
    weighting_factors = np.asarray(str(current_dict['weighting_factors']).split(',')).astype(float)
    conversion_factor = float(current_dict['conversion_factor'])

    try:
        no_data = float(current_dict['no_data'])
    except:
        no_data = False

    param_datasets = []

    if os.path.isdir(data_dir):
        for fn in os.listdir(data_dir):
            if str(fn).lower().endswith('.tif') or str(fn).lower().endswith('.tiff'):
                param_datasets.append(fn)
    elif os.path.isfile(data_dir):
        if str(data_dir).lower().endswith('.tif') or str(data_dir).lower().endswith('.tiff'):
            param_datasets.append(os.path.basename(data_dir))
            data_dir = os.path.dirname(data_dir)

    if len(param_datasets) > 1:
        param_datasets = sorted(param_datasets, key=lambda x: int(str(x).split('_')[1].split('-')[0]))

    param_datasets = [os.path.join(data_dir, param_datasets[i]) for i in range(len(param_datasets))]

    dataset, no_data = aggregate_soil_raster_lst(param_datasets, domain, shape, weighting_method, conversion_factor, weighting_factors)
    if no_data and np.issubdtype(dataset.dtype, np.floating):
        dataset[dataset<0] = np.nan
        dataset[dataset == no_data] = np.nan 
    return dataset.astype(itype)

def get_valid_dtype(itype):
    dtype_map = {
        'uint8': 'uint8',
        'int8': 'int8',
        'uint16': 'uint16',
        'int16': 'int16',
        'uint32': 'uint32',
        'int32': 'int32',
        'float32': 'float32',
        'float64': 'float64'
    }
    try:
        return dtype_map.get(itype.type, 'float32')
    except:
        try:
            return dtype_map.get(itype.name, 'float32')
        except:
            return dtype_map.get(itype, 'float32')
        
def output_param_data(param_arr, param_list, output_dir, domain):
    for idx, param in enumerate(param_list):
        out_f = f'{param}_combined.tif'
        dataset = param_arr[..., idx]
        valid_dtype = get_valid_dtype(param_arr[..., idx].dtype)
        try:
            os.remove(os.path.join(output_dir, out_f))
        except:
            pass
    
        crs = 'EPSG:4326'
        transform = from_bounds(domain[0], domain[3], domain[2], domain[1], dataset.shape[1], dataset.shape[0])
        metadata = {'driver': 'GTiff', 'height': dataset.shape[0], 'width': dataset.shape[1], 'count': 1, 'dtype': valid_dtype, 'crs': crs, 'transform': transform, 'nodata': np.nan}
        with rasterio.open(os.path.join(output_dir, out_f), 'w', **metadata) as dst:
            dst.write(dataset.astype(valid_dtype), 1)

def aggregate_soil_raster_lst(file_list, domain, final_shape, weighting_method = 0, conversion_factor = 1., weighting = [2., 1.5, 1., 0.75, 0.5, 0.25]):
    """
    Aggregate soil raster data from a list of files based on specified parameters.

    Parameters:
    - file_list (list): A list of file paths to the soil raster data.
    - domain (list): A list containing the geographic domain coordinates [min_longitude, min_latitude, max_longitude, max_latitude].
    - final_shape (tuple): A tuple specifying the final shape of the aggregated array (rows, columns).
    - weighting_method (int): The method used for weighting during aggregation (0, 1, 2).
    - conversion_factor (float): A conversion factor applied to the aggregated result.
    - weighting (list): A list of weighting factors used in the aggregation process.

    Returns:
    - tuple: A tuple containing the aggregated soil data array and the nodata value.

    Note:
    - The function aggregates soil raster data from a list of files based on the specified weighting method.
    - 'weighting_method' determines the aggregation strategy: 0 for single layer, 1 for mean of multiple layers, 2 for custom weighted sum.
    - The 'conversion_factor' scales the aggregated result.
    - The function returns a tuple containing the aggregated soil data array and the nodata value.
    """
    nodata = None
    output_tif = False
    if weighting_method == 0:
        layer, nodata = dt.load_specified_lines(file_list[0], [domain[1], domain[0], domain[3], domain[2]], all_bands=False) #type:ignore
        if layer.shape != final_shape:
            layer, nodata = resize_array_interp(layer, final_shape, nodata=nodata, method='nearest')

        return layer / conversion_factor, nodata  # type: ignore
    elif weighting_method == 1:
        layers = []
        for layer_file in file_list:
            layer, nodata = dt.load_specified_lines(layer_file, [domain[1], domain[0], domain[3], domain[2]], all_bands=False)
            layers.append(layer)
        mean = np.nanmean(layers, axis=0)
        if mean.shape != final_shape:
            mean, nodata = resize_array_interp(mean, final_shape, nodata=nodata, method='nearest')

        return mean / conversion_factor, nodata  # type: ignore

    elif weighting_method == 2:
        layers = []
        for layer_file in file_list:
            layer, nodata = dt.load_specified_lines(layer_file, [domain[1], domain[0], domain[3], domain[2]], all_bands=False)
            layers.append(layer)

        for idx, layer in enumerate(layers):
            if layer.shape != final_shape:
                layers[idx], nodata = resize_array_interp(layer, final_shape, nodata=nodata, method='nearest')

        new_file = np.zeros((layers[0].shape))
        new_file = new_file.astype(float)
        new_file = layers[0] * 5 * weighting[0] # 0-5
        new_file += layers[1] * 10 * weighting[0] # 5-15
        new_file += layers[2] * 10 * weighting[0] # 15-25
        new_file += layers[2] * 5 * weighting[1] # 25-30
        new_file += layers[3] * 20 * weighting[1] # 30-50
        new_file += layers[3] * 10 * weighting[2] # 50-60
        new_file += layers[4] * 15 * weighting[2] # 60-75
        new_file += layers[4] * 25 * weighting[3] # 75-100
        new_file += layers[5] * 25 * weighting[4] # 100-125
        new_file += layers[5] * 25 * weighting[5] # 125-150
        
        return new_file / 150 / conversion_factor, nodata  # type: ignore
        
    else:
        throw_exit_error('Unkown Weighting Method')
        return np.zeros(final_shape), 0


def get_unique_prefixes_in_order(lst):
    """
    Get unique prefixes from a list of strings in the order of their first occurrence.

    Parameters:
    - lst (list): The input list of strings.

    Returns:
    - list: A list of unique prefixes in the order of their first occurrence.

    Note:
    - The function iterates through the input list and extracts the prefix from each string.
    - It maintains the order of the first occurrence for each unique prefix.
    """
    prefixes = []
    seen = set()
    for item in lst:
        prefix = item.split('_')[0]
        if prefix not in seen:
            prefixes.append(prefix)
            seen.add(prefix)
    return prefixes


def getTable(fn):
    """
    Parse a file to create a table of class names and associated sand-clay boundaries.

    Parameters:
    - fn (str): File path to the input file.

    Returns:
    - dict: A dictionary where keys are class names and values are matplotlib.path.Path objects representing sand-clay boundaries.

    Note:
    - The function reads the content of the file, extracts class names, sand, and clay limits, and creates a dictionary with matplotlib.path.Path objects.
    """
    with open(fn, 'r') as f:
        x = f.readlines()
    nbClasses = int(x[0])
    classNamesRaw = x[1 + nbClasses*4 + 1: 1 + nbClasses*4 + 1 + nbClasses + 1]
    classNames = dict(item.strip().split('=') for item in classNamesRaw)
    table = {}
    for i in range(nbClasses):
        className = x[1 + i*4].strip()
        sandLimits = list(map(float, x[3 + i*4].split()))
        clayLimits = list(map(float, x[4 + i*4].split()))
        verts = np.column_stack((sandLimits, clayLimits)) # type: ignore
        table[classNames[className]] = path.Path(verts)
    return table


def get_texture_class(sand, clay, config):
    """
    Get the texture class based on sand and clay percentages using USDA soil texture boundaries.

    Parameters:
    - sand (numpy.ndarray): Array containing sand percentages.
    - clay (numpy.ndarray): Array containing clay percentages.
    - config (dict): Configuration dictionary containing file paths.

    Returns:
    - numpy.ndarray: Array containing texture class indices.

    Note:
    - The function uses USDA soil texture boundaries from a specified file to classify sand and clay percentages into texture classes.
    - The result is an array of indices corresponding to different texture classes.
    ```
    {1: 'heavy clay', 2: 'silty clay', 3: 'clay', 4: 'silty clay loam', 5: 'clay loam',
     6: 'silt', 7: 'silt loam', 8: 'sandy clay', 9: 'loam', 10: 'sandy clay loam',
     11: 'sandy loam', 12: 'loamy sand', 13: 'sand'}
    ```
    """
    usda_table = getTable(os.path.join(config['files']['texture_classes']))
    texture_dict = {'heavy clay': 1,
                    'silty clay': 2,
                    'clay': 3,
                    'silty clay loam': 4,
                    'clay loam': 5,
                    'silt': 6,
                    'silt loam': 7,
                    'sandy clay': 8,
                    'loam': 9,
                    'sandy clay loam': 10,
                    'sandy loam': 11,
                    'loamy sand': 12,
                    'sand': 13}
    sand = np.array(sand)
    clay = np.array(clay)
    texture = np.zeros_like(sand, dtype=np.int8)
    for key, p in usda_table.items():
        mask = p.contains_points(np.column_stack((sand.flatten(), clay.flatten())))
        texture[mask.reshape(sand.shape)] = int(texture_dict[str(key)])
    return texture


def calculate_slope(dem_path, output_shape, extent):
    if output_shape == (0, 0) or extent == [0, 0, 0, 0]:
        with rasterio.open(dem_path, 'r') as dem:
            transform = dem.transform
            cell_size = transform.a
            dem_array = dem.read(1)
    else:
        cell_size = ((extent[2] - extent[0]) / output_shape[1])
        domain = [extent[1], extent[0], extent[3], extent[2]]
        dem_array = dt.load_specified_lines(dem_path, domain)[0][0]
        dem_array = dem_array[:output_shape[0], :output_shape[1]]
    R = 6371000
    resolution_rad = np.deg2rad(cell_size)
    resolution_m = R * resolution_rad
    dz_dy, dz_dx = np.gradient(dem_array, resolution_m, edge_order=2)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_degrees = np.rad2deg(slope)
    return slope_degrees


def cropsuitability(config, clim_suit, lim_factor, plant_formulas, plant_params, extent, land_sea_mask, results_path, multiple_cropping_sum=None):
    """
    Calculate soil suitability for crop plants based on climate suitability and soil parameters.

    Parameters:
    - config (dict): Configuration dictionary containing file paths and options.
    - clim_suit (numpy.ndarray): Array containing climate suitability values for each plant.
    - lim_factor (numpy.ndarray): Array containing limiting factors for each plant.
    - plant_formulas (dict): Dictionary containing formulas for calculating plant suitability.
    - plant_params (dict): Dictionary containing plant parameters.
    - extent (list): List containing the geographical extent for analysis.
    - land_sea_mask (numpy.ndarray): Array representing land-sea mask.
    - results_path (str): Path to the directory where results will be stored.

    Returns:
    - None

    Note:
    - The function calculates soil suitability for each plant based on climate suitability, soil parameters, and limiting factors.
    - It processes data according to the specified configuration and stores results in the specified directory.
    """
    print('Calculate the Soil Suitability')
    domain = [-180, 90, 180, -90]
    if extent:
        domain = [extent[1], extent[0], extent[3], extent[2], 0, 0]

    plant_list = [plant for plant in plant_params]
    if os.path.exists(os.path.join(results_path, plant_list[-1], 'crop_suitability.tif')):
        return
    #suitability = np.zeros((clim_suit.shape[0], clim_suit.shape[1], len(plant_list)), dtype=np.float16)
    parameter_list = [entry.replace('parameters.', '', 1) if entry.startswith('parameters.') else entry for entry in get_id_list_start(config, 'parameters.')] + ['slope']
    parameter_array = np.empty((clim_suit.shape[0], clim_suit.shape[1], len(parameter_list)), dtype=np.float16)
    parameter_dictionary = {parameter_list[parameter_id]: config[f'parameters.{parameter_list[parameter_id]}']['rel_member_func'] for parameter_id in range(len(parameter_list)) if f'parameters.{parameter_list[parameter_id]}' in config}
    parameter_dictionary['slope'] = 'slope'

    for counter, parameter in enumerate(parameter_list):
        print(f' -> Loading {parameter} data')
        if parameter == 'slope':
            parameter_array[..., counter] = calculate_slope(config['files'].get('fine_dem'), (clim_suit.shape[0], clim_suit.shape[1]), domain)
        else:
            parameter_array[..., counter] = get_soil_data(config, parameter, domain, (clim_suit.shape[0], clim_suit.shape[1]), np.float16)
    print(' -> Converting sand and clay content to texture class')
    parameter_array[..., parameter_list.index('sand_content')] = get_texture_class(parameter_array[..., parameter_list.index('sand_content')],\
                                                                                   parameter_array[..., parameter_list.index('clay_content')],\
                                                                                    config)
    parameter_array = np.delete(parameter_array, parameter_list.index('clay_content'), axis=2)
    parameter_list[parameter_list.index('sand_content')] = 'texture'
    del parameter_list[parameter_list.index('clay_content')]

    if config['options'].get('output_soil_data', 0) == 1:
        output_param_data(parameter_array, parameter_list, results_path, domain[:4])

    formulas = [plant for plant in plant_formulas[plant_list[0]]]
    formulas = dict(zip(formulas, np.arange(0, len(formulas))))

    for plant_idx, plant in enumerate([p for p in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, p))]):
        res_path = os.path.join(results_path, plant)
        if not os.path.exists(os.path.join(res_path, 'climate_suitability.tif')) and not os.path.exists(os.path.join(res_path, 'climate_suitability.nc')):
            continue
        print('\n'+f'Processing suitability for {plant}')
        
        os.makedirs(res_path, exist_ok=True)
        if os.path.exists(os.path.join(res_path, 'crop_suitability.tif')):
            continue
        #plant_idx = plant_list.index(plant)
        suitability_array = np.empty((clim_suit.shape[0], clim_suit.shape[1], len(parameter_list)), dtype=np.float16)        

        max_ind_val_climate = 3

        order_file = os.path.join(res_path, 'limiting_factor.inf')
        with open(order_file, 'w') as write_file:
            write_file.write('value - limiting factor\n')
            write_file.write('0 - temperature\n')
            write_file.write('1 - precipitation\n')
            write_file.write('2 - climate variability\n')
            write_file.write('3 - photoperiod\n')
            for counter, parameter in enumerate(parameter_list):
                if parameter == 'texture' and 'texture' in plant_formulas[plant]:
                    suitability_array[..., counter] = (get_suitability_val_dict(plant_formulas, plant, 'texture', parameter_array[..., counter])*100).astype(np.float16)
                elif parameter not in parameter_dictionary or parameter_dictionary[parameter] not in plant_formulas[plant]:
                    print(f' -> {plant} has no parameter {parameter}. Skipping {parameter}.')
                    suitability_array[..., counter] = (np.ones_like(suitability_array[..., counter])*100).astype(np.float16)
                else:
                    suitability_array[..., counter] = (get_suitability_val_dict(plant_formulas, plant, parameter_dictionary[parameter], parameter_array[..., counter])*100).astype(np.float16)
                write_file.write(f'{counter+max_ind_val_climate+1} - {parameter}'+'\n')

        curr_climsuit = clim_suit[..., plant_idx]
        soil_suitablility = np.min(suitability_array, axis=2).astype(np.int8)
        suitability_array = np.concatenate((curr_climsuit[..., np.newaxis], suitability_array), axis=2)
        suitability = np.clip(np.min(suitability_array, axis=2), -1, 100) 
        suitability[np.isnan(land_sea_mask)] = -1
        suitability = suitability.astype(np.int8)

        min_indices = (np.argmin(suitability_array, axis=2) + max_ind_val_climate).astype(np.int8)
        min_indices[min_indices == max_ind_val_climate] = lim_factor[min_indices == max_ind_val_climate, plant_idx]
        min_indices[np.isnan(land_sea_mask)] = -1
        nan_mask = suitability == -1
        min_indices[nan_mask] = -1
        
        suitability_array[np.isnan(suitability_array)] = -1
        suitability_array = suitability_array.astype(np.int8)

        if config['options']['output_all_limiting_factors'] and os.path.exists(os.path.join(res_path, 'all_climlim_factors.tif')):
            clim_lims = dt.read_tif_file_with_bands(os.path.join(res_path, 'all_climlim_factors.tif'))
            suitability_array = np.concatenate([np.transpose(clim_lims, (1, 2, 0)), suitability_array[..., 1:]], axis=2)
            suitability_array[nan_mask, :] = -1
            dt.write_geotiff(res_path, 'all_suitability_vals.tif', suitability_array, extent, dtype='int', nodata_value=-1, cog=config['options']['output_format'].lower() == 'cog')
            
        if config['options']['output_format'] == 'geotiff' or config['options']['output_format'] == 'cgo':
            dt.write_geotiff(res_path, 'crop_suitability.tif', suitability, extent, nodata_value=-1, cog=config['options']['output_format'].lower() == 'cog')
            dt.write_geotiff(res_path, 'crop_limiting_factor.tif', min_indices, extent, nodata_value=-1, cog=config['options']['output_format'].lower() == 'cog')
            dt.write_geotiff(res_path, 'soil_suitability.tif', soil_suitablility, extent, dtype='int', nodata_value=-1, cog=config['options']['output_format'].lower() == 'cog')
        elif config['options']['output_format'] == 'netcdf4':
            nc.write_to_netcdf(suitability, os.path.join(res_path, 'crop_suitability.nc'), extent=extent, compress=True, var_name='crop_suitability', nodata_value=-1) #type:ignore
            nc.write_to_netcdf(min_indices, os.path.join(res_path, 'crop_limiting_factor.nc'), extent=extent, compress=True, var_name='crop_limiting_factor', nodata_value=-1) #type:ignore
            nc.write_to_netcdf(soil_suitablility, os.path.join(res_path, 'soil_suitability.nc'), extent=extent, compress=True, var_name='soil_suitability', nodata_value=-1) #type:ignore
        else:
            print('No output format specified.')

        del suitability_array, suitability, soil_suitablility, clim_lims, nan_mask, min_indices, curr_climsuit

    print('\nSuitability data created')