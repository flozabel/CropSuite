import numpy as np
import os
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from rasterio.windows import Window
import glob
from psutil import virtual_memory
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from gc import collect
from scipy.interpolate import RegularGridInterpolator
import warnings
import shutil
warnings.filterwarnings("ignore")
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import scipy.ndimage
import sys
try:
    import nc_tools as nc
except:
    from src import nc_tools as nc
from scipy.ndimage import zoom
from collections import namedtuple


def get_extent(filepath):
    BoundingBox = namedtuple("BoundingBox", ["left", "bottom", "right", "top"])
    if filepath.endswith((".tif", ".tiff")):
        with rasterio.open(filepath, 'r') as src:
            left, bottom, right, top = src.bounds
    elif filepath.endswith((".nc", ".nc4")):
        with xr.open_dataset(filepath) as ds:
            lon = ds.coords.get("lon", ds.coords.get("longitude", None))
            lat = ds.coords.get("lat", ds.coords.get("latitude", None))
            if lon is None or lat is None:
                raise ValueError("Longitude and Latitude coordinates not found.")
            left, right = float(lon.min()), float(lon.max())
            bottom, top = float(lat.min()), float(lat.max())
    else:
        raise ValueError("Unsupported file format.")

    return BoundingBox(left, bottom, right, top)

def get_npy_file_shape(file_path) -> list:
    """
    Retrieve the shape of a NumPy array stored in a .npy file.

    Parameters:
    - file_path (str): Path to the .npy file.

    Returns:
    list: A list representing the shape of the NumPy array in the file. If an error occurs, returns [-1, -1, -1].

    This function reads the header of a .npy file to extract the shape information of the stored NumPy array.
    """
    try:
        with open(file_path, 'rb') as file:
            header = np.lib.format.read_magic(file)
            shape, _, _ = np.lib.format.read_array_header_1_0(file)
        return shape
    except Exception as e:
        print(f"Error reading the .npy file header: {e}")
        return [-1, -1, -1]


def get_tif_dimensions(file_path) -> tuple:
    """
    Retrieve the dimensions (width, height, and count) of a GeoTIFF file.

    Parameters:
    - file_path (str): Path to the GeoTIFF file.

    Returns:
    tuple: A tuple (width, height, count) representing the dimensions of the GeoTIFF file.
           If an error occurs, returns (0, 0, 0).

    This function uses the rasterio library to open the GeoTIFF file and extract its dimensions.
    """
    try:
        with rasterio.open(file_path, 'r') as dataset:
            width = dataset.width
            height = dataset.height
            count = dataset.count
            return width, height, count
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving dimensions of {file_path}: {e}')
        return 0, 0, 0


def get_geotiff_resolution(file_path) -> tuple:
    """
    Retrieve the resolution (pixel size) of a GeoTIFF file.

    Parameters:
    - file_path (str): Path to the GeoTIFF file.

    Returns:
    tuple: A tuple (x_resolution, y_resolution) representing the pixel size of the GeoTIFF file.
           If an error occurs, returns (0, 0).

    This function uses the rasterio library to open the GeoTIFF file and extract its resolution.
    """
    try:
        with rasterio.open(file_path, 'r') as dataset:
            x_res, y_res = dataset.res
            return x_res, y_res
    except Exception as e:
        print(f'An error occurred while retrieving resolution of {file_path}: {e}')
        return 0, 0


def get_geotiff_datatype(file_path) -> str:
    """
    Retrieve the data type of a GeoTIFF file.

    Parameters:
    - file_path (str): Path to the GeoTIFF file.

    Returns:
    str: The data type of the GeoTIFF file.
         If an error occurs, returns an empty string.

    This function uses the rasterio library to open the GeoTIFF file and extract its data type.
    """
    try:
        with rasterio.open(file_path, 'r') as dataset:
            dtype = dataset.dtypes[0]
            return dtype
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving the data type of {file_path}: {e}')
        return ''


def get_geotiff_stats(file_path) -> list:
    """
    Retrieve statistics from a GeoTIFF file.

    Parameters:
    - file_path (str): Path to the GeoTIFF file.

    Returns:
    list: A list containing the minimum, mean, median, maximum values, and the number of NaN values.
          If an error occurs, returns [0, 0, 0, 0, 0].

    This function uses the rasterio library to open the GeoTIFF file, read its data and mask, and then
    calculates statistics (minimum, mean, median, maximum) from the valid data (non-NaN) values.
    """
    try:
        with rasterio.open(file_path, 'r') as dataset:
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

    Parameters:
    - file_path (str): Path to the GeoTIFF file.

    Returns:
    str: Spatial reference in WKT (Well-Known Text) format.
         If an error occurs, returns an empty string.

    This function uses the rasterio library to open the GeoTIFF file and retrieve its
    coordinate reference system (CRS) information in the WKT format.
    """
    try:
        with rasterio.open(file_path, 'r') as dataset:
            spatial_ref = dataset.crs.to_string()
            return spatial_ref
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving the spatial reference of {file_path}: {e}')
        return ''


def replace_with_nan(data, nodata_value=.0) -> np.ndarray:
    """
    Replace specified NoData values with NaN in the input array.

    Parameters:
    - data (numpy.ndarray): Input array containing NoData values.
    - nodata_value (float): Value in the array to be treated as NoData (default is 0.0).

    Returns:
    numpy.ndarray: New array with NoData values replaced by NaN.

    This function replaces all occurrences of the specified NoData value with NaN
    in the input array.
    """
    data[data == nodata_value] = np.nan
    return data


def read_coarse_dem(coarse_dem, coarse_cols, coarse_rows=300, itype='float32') -> np.ndarray:
    """
    Reads the coarse digital elevation model (DEM) from a binary file with the specified data type.
    Args:
        coarse_dem (str): The name of the binary file containing the coarse DEM.
        coarse_cols (int): The number of columns in the coarse DEM.
        coarse_rows (int): The number of rows in the coarse DEM.
        itype (str, optional): The data type of the values in the binary file. Defaults to 'float32'.
    Returns:
        numpy.ndarray: A 2D array containing the coarse DEM values.
    """
    ras = np.fromfile(coarse_dem+'.ras', dtype=itype)
    ras = np.array(ras).reshape(300, coarse_cols)
    ras = replace_with_nan(ras)
    return np.append(ras, np.zeros((60, ras.shape[1])), axis=0)


def read_coordinates(lon_path, lat_path) -> tuple[list[float], list[float]]:
    """
    Reads longitude and latitude values from two text files and returns them as lists of floats.
    Args:
        lon_path (str): The path to the file containing longitude values, with one value per line.
        lat_path (str): The path to the file containing latitude values, with one value per line.
    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists of floats, representing the longitude
        and latitude values, respectively.
    """
    with open(lon_path, 'r') as f:
        lines = f.readlines()
    f.close() 
    lon = [float(line.strip()) for line in lines if line.strip()]
    with open(lat_path, 'r') as f:
        lines = f.readlines()
    f.close() 
    lat = [float(line.strip()) for line in lines if line.strip()]
    return lon, lat


def read_latitudes_echam(lat_path) -> list[float]:
    """
    Reads latitude values from a text file and returns them as a list of floats for use with the ECHAM model.
    Args:
        lat_path (str): The path to the file containing latitude values, with one value per line.
    Returns:
        List[float]: A list of floats representing the latitude values.
    """
    with open(lat_path, 'r') as f:
        lines = f.readlines()
    f.close() 
    lat_echam = [float(line.strip()) for line in lines if line.strip()]
    return [float(line.strip()) for line in lines if line.strip()]


def read_land_sea_mask(land_sea_mask, line_number=-1, ras_suffix = '.ras', header_suffix = '.rhd', itype=np.uint8) -> np.ndarray:
    """
    Reads a binary raster file containing land/sea mask data, along with its associated header file. The file location
    and name are specified in the land_sea_mask argument. If a line number is specified, the function reads only that line
    of the raster.
    Args:
    - land_sea_mask (str): the file location and name of the land/sea mask file, without the file extension.
    - line_number (int): the line number to read from the raster file. Default value is -1, which reads the entire file.
    - ras_suffix (str): the suffix of the binary raster file, default value is '.ras'.
    - header_suffix (str): the suffix of the header file, default value is '.rhd'.
    - itype (numpy.dtype): the data type of the raster file, default value is np.uint8.
    Returns:
    - ras (numpy.ndarray): a 1D or 2D numpy array containing the land/sea mask data. If a line number is specified, the 
      returned array will be 1D.
    """
    with open(land_sea_mask + header_suffix, 'r') as f:
        header = [line.strip() for line in f.readlines()]
    y, x = map(int, header[2].split())
    if line_number < 0:
        ras = np.fromfile(land_sea_mask + ras_suffix, dtype=itype)
        ras = ras.reshape(y, x)
    else:
        offset = (line_number-1) * x * itype().itemsize
        with open(land_sea_mask + ras_suffix, 'rb') as f:
            f.seek(offset)
            ras = np.fromfile(f, dtype=itype, count=x)
    print('land-sea-mask read')
    return ras


def read_land_sea_mask_tif(climate_config, itype=np.uint8) -> tuple:
    """
    Read land-sea mask from a GeoTIFF file.

    Parameters:
    - climate_config (dict): Configuration dictionary containing file paths.
    - itype (numpy.dtype): Data type to use when reading the GeoTIFF (default is np.uint8).

    Returns:
    tuple: A tuple containing the land-sea mask array and the bounds.

    This function reads the land-sea mask from the specified GeoTIFF file
    in the climate configuration. It returns a tuple containing the land-sea mask array
    and the bounds of the GeoTIFF.
    """
    land_sea_mask = os.path.join(climate_config['files']['land_sea_mask'])
    with rasterio.open(land_sea_mask, 'r') as src:
        bounds = src.bounds
        data = src.read(1)
    bounds = swapPositions(np.asarray(bounds), 2, 3)
    print('land-sea-mask read')
    return data, bounds


def read_soildepth(soildepth, line_number=-1, ras_suffix = '.bil', header_suffix = '.hdr', itype=np.uint8) -> np.ndarray:
    """
    Read soil depth data from binary file (.bil) with an associated header file (.hdr).

    Parameters:
    - soildepth (str): Path to the soil depth file without file extension.
    - line_number (int): Specific line number to read from the file (default is -1, which reads the entire file).
    - ras_suffix (str): Suffix for the binary raster file (default is '.bil').
    - header_suffix (str): Suffix for the header file (default is '.hdr').
    - itype (numpy.dtype): Data type to use when reading the binary file (default is np.uint8).

    Returns:
    numpy.ndarray: Array containing the soil depth data.

    This function reads soil depth data from a binary file (.bil) using an associated header file (.hdr).
    If line_number is provided, it reads a specific line from the binary file; otherwise, it reads the entire file.
    """
    with open(soildepth + header_suffix, 'r') as f:
        header = [line.strip() for line in f.readlines()]
    y = int(header[2].split()[1])
    x = int(header[3].split()[1])
    if line_number < 0:
        ras = np.fromfile(soildepth + ras_suffix, dtype=itype)
        ras = ras.reshape(y, x)
    else:
        offset = (line_number-1) * x * itype().itemsize
        with open(soildepth + ras_suffix, 'rb') as f:
            f.seek(offset)
            ras = np.fromfile(f, dtype=itype, count=x)
    print('soildepth read')
    return ras


def read_raster_to_array(raster_file, nodata=-9999.) -> np.ndarray:
    """
    Read a raster file into a NumPy array.

    Parameters:
    - raster_file (str): Path to the raster file.
    - nodata (float): NoData value in the raster file (default is -9999.).

    Returns:
    numpy.ndarray: Array containing the raster data with NoData values replaced by NaN.

    This function uses rasterio to read a raster file into a NumPy array.
    NoData values are replaced with NaN if the data type is float (float16, float32, or float).
    """
    with rasterio.open(raster_file, 'r') as src:
        data = src.read(1)
    if data.dtype == np.float16 or data.dtype == float or data.dtype == np.float32:
        data[data == nodata] = np.nan
    return data


def read_irrigation(irrigation_ras, line_number=-1, irrigation_flag=2, ras_suffix='.ras', header_suffix='.rhd', itype=np.uint8) -> np.ndarray:
    """
    Read irrigation raster data from a file into a NumPy array.

    Parameters:
    - irrigation_ras (str): Path to the irrigation raster file.
    - line_number (int): Line number to read from the raster file (-1 to read the entire raster).
    - irrigation_flag (int): Irrigation flag value to consider (default is 2).
    - ras_suffix (str): Suffix for the raster data file (default is '.ras').
    - header_suffix (str): Suffix for the header file (default is '.rhd').
    - itype (numpy.dtype): Data type to use for reading the raster file (default is np.uint8).

    Returns:
    numpy.ndarray: Irrigation raster data in the specified line or the entire raster.

    This function reads irrigation raster data from a file into a NumPy array.
    It supports reading the entire raster or a specific line based on the line number.
    """
    with open(irrigation_ras + header_suffix, 'r') as f:
        header = [line.strip() for line in f.readlines()]
    y, x = map(int, header[2].split())

    if line_number < 0:
        ras = np.fromfile(irrigation_ras + ras_suffix, dtype=itype)
        ras = ras.reshape(y, x)
    else:
        offset = (line_number-1) * x * itype().itemsize
        with open(irrigation_ras + ras_suffix, 'rb') as f:
            f.seek(offset)
            ras = np.fromfile(f, dtype=itype, count=x)
    return ras


def read_tif_file_with_bands(fn) -> np.ndarray:
    """
    Read a GeoTIFF file with multiple bands into a NumPy array.

    Parameters:
    - fn (str): Path to the GeoTIFF file.

    Returns:
    numpy.ndarray: Array containing the bands from the GeoTIFF file.

    This function uses the rasterio library to read a GeoTIFF file with multiple bands
    into a NumPy array. The returned array has shape (num_bands, height, width).
    """
    with rasterio.open(fn, 'r') as src:
        data = src.read()
    return data


def get_cpu_ram() -> list:
    """
    Get information about the CPU and available RAM.

    Returns:
    list: A list containing the number of CPU cores and the available RAM in gigabytes.

    This function uses the psutil library to retrieve information about the CPU and
    available RAM. The returned list has two elements: the number of CPU cores and
    the available RAM in gigabytes.
    """
    return [cpu_count(), virtual_memory().available/1000000000]


def read_bias_factors(biasfactors_path, line_number, pixelsize=0.008333333, filename_temp='biasfactor_temp',\
                      filename_prcip='biasfactor_precip', suffix='.bil', itype=np.float32, nodata=-9999.) -> list:
    """
    Reads bias factors from binary files for a given line number.
    Args:
        biasfactors_path (str): Path to the directory containing the bias factor files.
        line_number (int): Line number to extract from the bias factor files.
        pixelsize (float, optional): Pixel size in degrees. Defaults to 0.008333333.
        filename_temp (str, optional): Prefix of the temperature bias factor file names. Defaults to 'biasfactor_temp'.
        filename_prcip (str, optional): Prefix of the precipitation bias factor file names. Defaults to 'biasfactor_precip'.
        suffix (str, optional): Suffix of the bias factor file names. Defaults to '.bil'.
        itype (numpy.dtype, optional): Data type of the bias factor files. Defaults to numpy.uint16.
    Returns:
        List of 12 pairs of temperature and precipitation bias factors for the given line number.
    """
    bias_arr = []
    x = int(360./pixelsize)
    for month in range(12):
        # print(f'   reading bias factors for month {month+1}')
        temp_f = os.path.join(biasfactors_path, filename_temp+'_'+str(month+1)+suffix)
        prec_f = os.path.join(biasfactors_path, filename_prcip+'_'+str(month+1)+suffix)
        offset = (line_number-1) * x * itype().itemsize
        with open(temp_f, 'rb') as f:
            f.seek(offset)
            temp_bias = np.fromfile(f, dtype=itype, count=x)
        with open(prec_f, 'rb') as f:
            f.seek(offset)
            prec_bias = np.fromfile(f, dtype=itype, count=x)
        bias_arr.append([temp_bias, prec_bias])
    bias_arr[bias_arr == nodata] = np.nan
    return bias_arr


def read_fine_dem_line(climate_config: dict, line_number: int=-1, suffix: str='.ras', itype=np.uint16) -> np.ndarray:
    """
    Reads a binary raster file containing fine digital elevation model data. The file location and name are specified
    in the climate_config dictionary. If a line number is specified, the function reads only that line of the raster.
    Args:
    - climate_config (dict): a dictionary containing the file location and name, as well as options for the raster file.
    - line_number (int): the line number to read from the raster file. Default value is -1, which reads the entire file.
    - suffix (str): the suffix of the file, default value is '.ras'.
    - itype (numpy.dtype): the data type of the raster file, default value is np.uint8.
    Returns:
    - ras (numpy.ndarray): a 2D numpy array containing the raster data. If a line number is specified, the returned array
      will be 1D.
    """
    fine_dem = os.path.join(climate_config['files']['fine_dem'])
    x, y = int(climate_config['options']['n_cols_fine']), int(climate_config['options']['n_rows_fine'])
    
    if line_number < 0:
        ras = np.fromfile(fine_dem+suffix, dtype=itype)
        ras = np.array(ras).reshape(y, x)
        print('Elevation data loaded')
    else:
        offset = (line_number-1) * x * itype().itemsize
        with open(fine_dem+suffix, 'rb') as f:
            f.seek(offset)
            ras = np.fromfile(f, dtype=itype, count=x)
    return ras


def fill_nan_nearest(array, nodata=np.nan, return_nanmask=False):
    if np.isnan(nodata):
        mask = np.isnan(array)
    else:
        mask = array == nodata
    array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask])
    if return_nanmask:
        return array, mask
    else:
        return array


def interpolate_array(data, target_shape, order=1):
    """
    Interpolates a 3D array to the specified target shape using spline interpolation.

    Parameters:
    - data: 3D numpy array of shape (365, 16, 18)
    - target_shape: tuple of the target shape (e.g., (365, 960, 1080))

    Returns:
    - Interpolated array with shape (365, 960, 1080)
    """
    # Calculate the zoom factors for each axis
    zoom_factors = [target_dim / original_dim for target_dim, original_dim in zip(target_shape, data.shape)]

    # Apply zoom interpolation
    interpolated_data = zoom(data, zoom_factors, order=order)  # order=1 for bilinear interpolation

    return interpolated_data


def resize_3darray_interp(array, new_shape):
    """
    Resizes a 3D array to a new shape using interpolation for each slice along the first axis.

    Parameters:
    - array (numpy.ndarray): Input 3D array to be resized.
    - new_shape (tuple): Desired shape of each 2D slice in the format (rows, columns).

    Returns:
    - resized_array (numpy.ndarray): Resized 3D array with the specified new shape for each slice
      obtained using interpolation.
    """
    new_arr = np.empty((array.shape[0], new_shape[0], new_shape[1]), dtype=array.dtype)
    for slice in range(array.shape[0]):
        new_arr[slice, ...], _ = resize_array_interp(array[slice], (new_shape[0], new_shape[1]))
    return new_arr


def resize_slice(slice, new_shape, idx, dtype=np.int16):
    np.save(os.path.join('temp', f'{idx}.npy'), (resize_array_interp(slice, (new_shape[0], new_shape[1]))[0]).astype(dtype))
    

def resize_3darray_interp_multiprocessing(array, new_shape, temp_dir='temp'):
    os.makedirs(temp_dir, exist_ok=True)
    no_threads = get_cpu_ram()[0] - 1

    with ProcessPoolExecutor(max_workers=no_threads) as executor:
        # Calculate the chunksize
        chunksize = max(1, array.shape[0] // no_threads)
        # Map the resize_slice function to each slice index
        executor.map(resize_slice, array, [new_shape]*array.shape[0], range(array.shape[0]), chunksize=chunksize)

    new_arr = np.empty((array.shape[0], new_shape[0], new_shape[1]), dtype=array.dtype)
    for slice in range(array.shape[0]):
        new_arr[slice] = np.load(os.path.join('temp', f'{slice}.npy'))
        os.remove(os.path.join('temp', f'{slice}.npy'))

    return new_arr


def resize_viewer_data(array, new_shape, nodata):
    nan_mask = np.asarray(array == nodata, dtype=np.uint8)
    nan_mask, _ = resize_array_interp(nan_mask, new_shape)
    
    filled = fill_nan_nearest(array, nodata)
    filled, _ = resize_array_interp(array, new_shape)

    filled[nan_mask > 0.1] = np.nan

    return filled  



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
    if not nodata is None:
        array, nanmask = fill_nan_nearest(array, nodata, return_nanmask=True)

    h, w = array.shape

    # Create an interpolating function using RegularGridInterpolator
    interp_func = RegularGridInterpolator(
        (np.arange(h), np.arange(w)),
        array,
        method=method
    )

    # Define the target points for the new shape
    target_x = np.linspace(0, h - 1, int(new_shape[0]))
    target_y = np.linspace(0, w - 1, int(new_shape[1]))
    target_points = np.array(np.meshgrid(target_x, target_y, indexing='ij')).reshape(2, -1).T

    # Interpolate to get the resized array
    ret = interp_func(target_points).reshape(int(new_shape[0]), int(new_shape[1]))
    if limit[0] != -9999: ret[ret<=-limit[0]] = 0
    if limit[1] != -9999: ret[ret>=limit[1]] = 0
    if not nodata is None:
        nanmask = interpolate_nanmask(nanmask, new_shape) # type: ignore
        ret += np.where(nanmask, np.nan, 0.0)
    return ret, None if not nodata else np.nan
'''

def resize_array_interp(array, new_shape, nodata=None, limit=(-9999, -9999)):
    """
    Resizes a 2D array to the specified new shape using skimage's resize function.

    Parameters:
    - array (np.ndarray): 2D array to be resized.
    - new_shape (tuple): Tuple representing the target shape (dimensions) in the format (rows, columns).
    - nodata (float, optional): Value representing no-data (NaN) in the array.
    - limit (tuple, optional): Tuple containing the lower and upper limits for the resized array values. Default is (-9999, -9999).

    Returns:
    - resized_array (np.ndarray): Resized array with the specified new shape using linear interpolation.
    - nodata (float or None): Returns None if nodata is not used, otherwise returns np.nan.

    Note: Values outside the specified limits are set to 0 if limits are provided.
    """
    if nodata is not None:
        array, nanmask = fill_nan_nearest(array, nodata, return_nanmask=True)

    # Resize the array using skimage's resize (linear interpolation by default)
    # resized_array = skt.resize(array, new_shape, order=1, mode='edge', anti_aliasing=False)

    h, w = array.shape

    # Create an interpolating function using RegularGridInterpolator
    interp_func = RegularGridInterpolator(
        (np.arange(h), np.arange(w)),
        array,
        method='linear'
    )

    # Define the target points for the new shape
    target_x = np.linspace(0, h - 1, int(new_shape[0]))
    target_y = np.linspace(0, w - 1, int(new_shape[1]))
    target_points = np.array(np.meshgrid(target_x, target_y, indexing='ij')).reshape(2, -1).T

    # Interpolate to get the resized array
    resized_array = interp_func(target_points).reshape(int(new_shape[0]), int(new_shape[1]))

    # Apply limits
    if limit[0] != -9999:
        resized_array[resized_array <= -limit[0]] = 0
    if limit[1] != -9999:
        resized_array[resized_array >= limit[1]] = 0

    # Restore NaN mask if nodata was provided
    if nodata is not None:
        nanmask = skt.resize(nanmask.astype(float), new_shape, order=0, mode='reflect') > 0.5  # Nearest-neighbor resize
        resized_array[nanmask] = np.nan

    return resized_array, None if nodata is None else np.nan
'''

def remove_residuals(fine_data, coarse_data, threshold=0.0005, iterations=10):
    residuals_coarse = resize_array_mean(fine_data, coarse_data.shape) - coarse_data # Coarse Resolution
    cntr = 0
    while abs(np.nanmean(residuals_coarse)) >= threshold and cntr < iterations:
        residuals_fine, _ = resize_array_interp(np.nan_to_num(residuals_coarse), fine_data.shape) # Fine Resolution
        fine_data -= residuals_fine.astype(fine_data.dtype)
        residuals_coarse = resize_array_mean(fine_data, coarse_data.shape) - coarse_data # Coarse Resolution
        cntr += 1
    return fine_data


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
    """
    h, w = array.shape
    array = array.astype(float)
    interp_func = interp2d(np.arange(w), np.arange(h), array, kind='linear')
    ret = interp_func(np.linspace(0, w - 1, int(new_shape[1])), np.linspace(0, h - 1, int(new_shape[0])))
    ret = (ret >= 0.5).astype(bool)
    return ret
    """
    
    h, w = array.shape
    array = array.astype(float)
    
    # Define the grid points for the original array
    x = np.arange(w)
    y = np.arange(h)
    
    # Create the interpolation function
    interp_func = RegularGridInterpolator((y, x), array, method='linear')
    
    # Generate the new grid points for interpolation
    new_x = np.linspace(0, w - 1, int(new_shape[1]))
    new_y = np.linspace(0, h - 1, int(new_shape[0]))
    
    # Create a meshgrid for the new points
    new_grid = np.meshgrid(new_y, new_x, indexing='ij')
    new_points = np.array(new_grid).reshape(2, -1).T
    
    # Perform the interpolation
    ret = interp_func(new_points).reshape(new_shape)
    
    # Apply threshold and convert to boolean
    ret = (ret >= 0.5).astype(bool)
    
    return ret



def resize_array_summarize(array, shape):
    """
    Resizes the input array to the specified shape by averaging blocks of pixels.

    Parameters:
    - array (numpy.ndarray): Input array to be resized.
    - shape (tuple): Desired shape of the output array in the format (rows, columns).

    Returns:
    - resized_array (numpy.ndarray): Resized array with the specified shape obtained by averaging blocks
      of pixels from the input array.
    """
    sh = shape[0], array.shape[0]//shape[0], shape[1], array.shape[1]//shape[1]
    return array.reshape(sh).sum(-1).sum(1)


def resize_array_mean(array, shape):
    """
    Resizes the input array to the specified shape by averaging blocks of pixels.

    Parameters:
    - array (numpy.ndarray): Input array to be resized.
    - shape (tuple): Desired shape of the output array in the format (rows, columns).

    Returns:
    - resized_array (numpy.ndarray): Resized array with the specified shape obtained by averaging blocks
      of pixels from the input array.
    """
    sh = shape[0], array.shape[0]//shape[0], shape[1], array.shape[1]//shape[1]
    try:
        return array.reshape(sh).mean(-1).mean(1)
    except:
        ret, _ = resize_array_interp(array, shape)
        return ret


def resample_array_with_nans_custom(arr, factor):
    shape = arr.shape
    pad_height = (factor - shape[0] % factor) % factor
    pad_width = (factor - shape[1] % factor) % factor
    if pad_height > 0 or pad_width > 0:
        arr = np.pad(arr, ((0, pad_height), (0, pad_width)), constant_values=np.nan)
    reshaped = arr.reshape(arr.shape[0] // factor, factor, arr.shape[1] // factor, factor)
    result = np.nanmean(reshaped, axis=(1, 3))
    return result


def resize_array_with_nans_ndimage(arr, factor):
    shape = arr.shape
    pad_height = (factor[0] - shape[0] % factor[0]) % factor[0]
    pad_width = (factor[1] - shape[1] % factor[1]) % factor[1]
    if pad_height > 0 or pad_width > 0:
        arr = np.pad(arr, ((0, pad_height), (0, pad_width)), constant_values=np.nan)
    reshaped = arr.reshape(arr.shape[0] // factor[0], factor[0], arr.shape[1] // factor[1], factor[1])
    result = np.nanmean(reshaped, axis=(1, 3))
    return result


def resize_array_ndimage(arr, new_shape):
    return np.asarray(scipy.ndimage.zoom(arr, (new_shape[0] / arr.shape[0], new_shape[1] / arr.shape[1]), order=3))


def resize_array_max(array, shape):
    """
    Resizes the input array to the specified shape by averaging blocks of pixels.

    Parameters:
    - array (numpy.ndarray): Input array to be resized.
    - shape (tuple): Desired shape of the output array in the format (rows, columns).

    Returns:
    - resized_array (numpy.ndarray): Resized array with the specified shape obtained by averaging blocks
      of pixels from the input array.
    """
    sh = shape[0], array.shape[0]//shape[0], shape[1], array.shape[1]//shape[1]
    return array.reshape(sh).max(-1).max(1)


def throw_exit_error(text):
    """
    Display an error message, wait for user input, and then exit the program.

    Args:
    text (str): The error message to be displayed.

    This function is designed to be used when a critical error occurs. It prints
    the provided error message, waits for the user to press Enter, and then exits
    the program.
    """
    input(text+'\nExit')
    exit()


def get_geotiff_extent(file_path) -> tuple:
    """
    Get the spatial extent (bounding box) of a GeoTIFF file.

    Args:
    file_path (str): The path to the GeoTIFF file.

    Returns:
    tuple: A tuple (minx, miny, maxx, maxy) representing the bounding box.

    This function uses rasterio to open the GeoTIFF file and retrieve its spatial
    extent. If an error occurs, it displays an error message and exits the program.
    """
    try:
        with rasterio.open(file_path, 'r') as dataset:
            bounds = dataset.bounds
            return bounds
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving extent of {file_path}: {e}')
        return 0, 0, 0, 0


def read_slope(climate_config, slope_path, itype=np.float32) -> np.ndarray:
    """
    Read slope data from a binary file and reshape it based on the provided climate configuration.

    Args:
    climate_config (dict): Climate configuration containing 'options' with 'n_cols_fine' and 'n_rows_fine'.
    slope_path (str): Path to the binary slope file.
    itype (numpy.dtype): Data type of the binary file.

    Returns:
    numpy.ndarray: Reshaped slope data.

    Reads the binary slope file and reshapes it according to the number of columns ('n_cols_fine') and
    rows ('n_rows_fine') specified in the climate configuration. The resulting numpy array is returned.
    """
    x, y = int(climate_config['options']['n_cols_fine']), int(climate_config['options']['n_rows_fine'])
    ras = np.fromfile(slope_path, dtype=itype)
    return np.array(ras).reshape(y, x)


def read_fine_dem(climate_config: dict, nofill=False, line_number: int=-1, suffix: str='.ras', itype=np.uint16) -> np.ndarray:
    """
    Reads a binary raster file containing fine digital elevation model data. The file location and name are specified
    in the climate_config dictionary. If a line number is specified, the function reads only that line of the raster.
    Args:
    - climate_config (dict): a dictionary containing the file location and name, as well as options for the raster file.
    - line_number (int): the line number to read from the raster file. Default value is -1, which reads the entire file.
    - suffix (str): the suffix of the file, default value is '.ras'.
    - itype (numpy.dtype): the data type of the raster file, default value is np.uint8.
    Returns:
    - ras (numpy.ndarray): a 2D numpy array containing the raster data. If a line number is specified, the returned array
      will be 1D.
    """
    fine_dem = os.path.join(climate_config['files']['fine_dem'])
    x, y = int(climate_config['options']['n_cols_fine']), int(climate_config['options']['n_rows_fine'])
    
    if line_number < 0:
        ras = np.fromfile(fine_dem+suffix, dtype=itype)
        ras = np.array(ras).reshape(y, x)
        print('Elevation data loaded')
    else:
        offset = (line_number-1) * x * itype().itemsize
        with open(fine_dem+suffix, 'rb') as f:
            f.seek(offset)
            ras = np.fromfile(f, dtype=itype, count=x)
    ras = np.array(ras).astype(np.float32)
    if nofill:
        return ras
    else:
        return np.append(ras, np.zeros((int((ras.shape[1] / 2)) - ras.shape[0], ras.shape[1])), axis=0)


def swapPositions(list, pos1, pos2) -> list:
    """
    Swap positions of two elements in a list.

    Args:
    input_list (list): The input list.
    pos1 (int): Position of the first element to swap.
    pos2 (int): Position of the second element to swap.

    Returns:
    list: List with positions of the specified elements swapped.

    Swaps the positions of elements at positions 'pos1' and 'pos2' in the input list and returns the modified list.
    """
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list


def read_fine_dem_tif(climate_config: dict) -> tuple:
    """
    Read the fine resolution Digital Elevation Model (DEM) from a GeoTIFF file.

    Args:
    climate_config (dict): Configuration dictionary containing file paths.

    Returns:
    tuple: A tuple containing the DEM data and its bounds.

    Reads the fine resolution DEM from the specified GeoTIFF file and returns a tuple containing the data and bounds.
    """
    fine_dem = os.path.join(climate_config['files']['fine_dem'])
    with rasterio.open(fine_dem, 'r') as src:
        bounds = src.bounds
        data = src.read(1)
    bounds = swapPositions(np.asarray(bounds), 2, 3)
    return data, bounds


def write_netcdf(filepath, filename, array, extent, crs='+proj=longlat +datum=WGS84 +no_defs +type=crs', nodata=np.nan):
    """
    Write a NumPy array to a NetCDF file.

    Args:
    filepath (str): Path to the directory where the NetCDF file will be saved.
    filename (str): Name of the NetCDF file.
    array (numpy.ndarray): 2D NumPy array to be saved.
    extent (tuple): Tuple containing the extent (xmin, ymin, xmax, ymax) of the data.
    crs (str): Coordinate reference system string.
    nodata (float): NoData value in the array.

    Writes the input array to a NetCDF file using xarray.
    """
    print(f' -> Writing {filename}')
    output_path = os.path.join(filepath, filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    height, width = array.shape
    xmin, ymin, xmax, ymax = extent
    xres = (xmax - xmin) / width
    yres = (ymax - ymin) / height
    transform = (xmin, xres, 0, ymax, 0, -yres)
    array = array.astype(float)
    da = xr.DataArray(array, dims=('y', 'x'), attrs={'transform': transform, 'crs': crs})
    da = da.where(da != nodata, other=np.nan)
    ds = xr.Dataset({'data': da})
    ds.to_netcdf(output_path, format='NETCDF4', engine='netcdf4')


def write_geotiff_string(filepath, filename, array, extent, crs='+proj=longlat +datum=WGS84 +no_defs +type=crs', cgo=False):
    """
    Write a NumPy array to a GeoTIFF file.

    Args:
    filepath (str): Path to the directory where the GeoTIFF file will be saved.
    filename (str): Name of the GeoTIFF file.
    array (numpy.ndarray): 2D NumPy array to be saved.
    extent (tuple): Tuple containing the extent (xmin, ymin, xmax, ymax) of the data.
    crs (str): Coordinate reference system string.

    Writes the input array to a GeoTIFF file using rasterio.
    """
    height, width = array.shape
    transform = from_bounds(extent[1], extent[2], extent[3], extent[0], width, height)
    with rasterio.open(os.path.join(filepath, filename), 'w', driver='GTiff', height=height, width=width, count=1, dtype='uint8', crs=crs, transform=transform) as dst:
        dst.write(array, 1)


def write_rgb_geotiff(filepath, filename, array, extent, crs='+proj=longlat +datum=WGS84 +no_defs +type=crs', nodata_value=-1):
    """
    Write a 3D NumPy array (RGB image) to a GeoTIFF file.

    Args:
    filepath (str): Path to the directory where the GeoTIFF file will be saved.
    filename (str): Name of the GeoTIFF file.
    array (numpy.ndarray): 3D NumPy array representing an RGB image (bands, height, width).
    extent (tuple or rasterio.bounds.BoundingBox): Tuple or BoundingBox containing the extent (xmin, ymin, xmax, ymax) of the data.
    crs (str): Coordinate reference system string.
    nodata_value (float): NoData value for the GeoTIFF.

    Writes the RGB image to a GeoTIFF file using rasterio.
    """
    print(f' -> Writing {filename}')
    output_path = os.path.join(filepath, filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    array = np.asarray(array)
    num_bands, height, width = array.shape
    try:
        transform = from_bounds(extent.left, extent.bottom, extent.right, extent.top, width, height)
    except:
        transform = from_bounds(extent[1], extent[2], extent[3], extent[0], width, height)
    
    with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=num_bands, dtype=array.dtype, crs=crs, transform=transform, compress='lzw') as dst:
        for band in range(num_bands):
            dst.write_band(band+1, array[band])


def write_geotiff(filepath, filename, array, extent, crs='+proj=longlat +datum=WGS84 +no_defs +type=crs', nodata_value=-1., dtype='float', cog=False, inhibit_message=False):
    """
    Write a NumPy array to a GeoTIFF file.

    Args:
    filepath (str): Path to the directory where the GeoTIFF file will be saved.
    filename (str): Name of the GeoTIFF file.
    array (numpy.ndarray): NumPy array to be written to the GeoTIFF.
    extent (tuple or rasterio.bounds.BoundingBox): Tuple or BoundingBox containing the extent (xmin, ymin, xmax, ymax) of the data.
    crs (str): Coordinate reference system string.
    nodata_value (float): NoData value for the GeoTIFF.
    dtype (str): Data type of the array ('float', 'int', 'bool').

    Writes the array to a GeoTIFF file using rasterio.

    """
    if not inhibit_message:
        print(f' -> Writing {filename}')
    output_path = os.path.join(filepath, filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    if array.ndim == 2:
        height, width = array.shape
        try:
            transform = from_bounds(extent.left, extent.bottom, extent.right, extent.top, width, height)
        except:
            try:
                transform = from_bounds(extent[1], extent[2], extent[3], extent[0], width, height)
            except:
                transform = from_bounds(float(extent['left']), float(extent['bottom']), float(extent['right']), float(extent['top']), width, height)
        if dtype == 'float':
            array = array.astype(float)
            array[np.isnan(array)] = nodata_value
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=str(array.dtype), crs=crs, transform=transform, nodata=nodata_value, compress='lzw') as dst:
                dst.write(array, 1)
        if dtype == 'int':
            nodata_value = int(nodata_value)
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=array.dtype, crs=crs, transform=transform, compress='lzw', nodata=nodata_value) as dst:
                dst.write(array, 1)
        if dtype == 'bool':
            with rasterio.open(output_path, 'w', driver='GTiff', nbits=1, height=height, width=width, crs=crs, count=1, dtype=np.uint8, transform=transform, compress='lzw') as dst:
                dst.write(array.astype(np.uint8), 1)
    elif array.ndim == 3:
        height, width, num_bands = array.shape
        try:
            transform = from_bounds(extent.left, extent.bottom, extent.right, extent.top, width, height)
        except:
            transform = from_bounds(extent[1], extent[2], extent[3], extent[0], width, height)
        if dtype == 'float':
            array = array.astype(float)
            array[np.isnan(array)] = nodata_value
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=num_bands, dtype=str(array.dtype), crs=crs, transform=transform, nodata=nodata_value, compress='lzw') as dst:
                for band in range(num_bands):
                    dst.write(array[..., band], band+1)
        if dtype == 'int':
            nodata_value = int(nodata_value)
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=num_bands, dtype=array.dtype, crs=crs, transform=transform, compress='lzw', nodata=nodata_value) as dst:
                for band in range(num_bands):
                    dst.write_band(band+1, array[..., band])
        if dtype == 'bool':
            with rasterio.open(output_path, 'w', driver='GTiff', nbits=1, height=height, width=width, count=num_bands, crs=crs, dtype=np.uint8, transform=transform, compress='lzw') as dst:
                for band in range(num_bands):
                    dst.write_band(band+1, array[..., band].astype(np.uint8))
        
    else:
        raise ValueError('Array dimensions should be 2 or 3.')
    
    if cog:
        create_cog_from_geotiff(output_path, output_path.replace('.tif', '_cog.tif'))


def geotiff_to_smallest_datatype(geotiff_list):
    min_val, max_val = 0, 0
    for input_file in geotiff_list:
        with rasterio.open(input_file, 'r') as src:
            data = src.read()
            min_val = np.min([min_val, np.min(data)])
            max_val = np.max([max_val, np.max(data)])
            if min_val > src.nodata:
                min_val = src.nodata
            src.close()
    if min_val >= -128 and max_val <= 128:
        dtype = np.int8
    elif min_val >= 0 and max_val <= 255:
        dtype = np.uint8
    elif min_val >= 0 and max_val <= 65535:
        dtype = np.uint16
    elif min_val >= -32768 and max_val <= 32767:
        dtype = np.int16
    elif min_val >= 0 and max_val <= 4294967295:
        dtype = np.uint32
    elif min_val >= -2147483648 and max_val <= 2147483647:
        dtype = np.int32
    else:
        dtype = np.float32
    print(f' -> Using {dtype} for {os.path.basename(geotiff_list[0])}')
    for input_file in geotiff_list:
        if not os.path.exists(input_file):
            continue

        temp_file = 'temp.tif'
        with rasterio.open(input_file) as src:
            data = src.read()
            meta = src.meta.copy()
            data_converted = data.astype(dtype)
            meta.update({'dtype': np.dtype(dtype).name})
        
        with rasterio.open(temp_file, 'w', **meta) as dst:
            dst.write(data_converted)
        shutil.move(temp_file, input_file)

        """
        with rasterio.open(input_file, "r+") as src:
            data = src.read()
            min_val, max_val = np.min(data), np.max(data)
            if min_val >= -128 and max_val <= 128:
                dtype = np.int8
            elif min_val >= 0 and max_val <= 255:
                dtype = np.uint8
            elif min_val >= 0 and max_val <= 65535:
                dtype = np.uint16
            elif min_val >= -32768 and max_val <= 32767:
                dtype = np.int16
            elif min_val >= 0 and max_val <= 4294967295:
                dtype = np.uint32
            elif min_val >= -2147483648 and max_val <= 2147483647:
                dtype = np.int32
            else:
                dtype = np.float32
            data = data.astype(dtype)
            try:
                src.dtype = [np.dtype(dtype).name] * src.count
            except:
                src.dtypes = [np.dtype(dtype).name] * src.count
            src.write(data)
        """

def merge_geotiffs_to_multiband(geotiff_files, output_file):
    with rasterio.open(geotiff_files[0]) as src0:
        meta = src0.meta
    meta.update(count=len(geotiff_files), compress='lzw')
    
    with rasterio.open(output_file, 'w', **meta) as dst:
        for idx, geotiff_file in enumerate(geotiff_files):
            with rasterio.open(geotiff_file) as src:
                data = src.read(1)
                dst.write(data, idx + 1)

    for f in geotiff_files:
        os.remove(f)

    return output_file


def create_cog_from_geotiff(src_path, dst_path, profile="deflate", profile_options={}, **options):
    """Convert image to COG."""
    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile_options)
    config = dict(GDAL_NUM_THREADS="ALL_CPUS", GDAL_TIFF_INTERNAL_MASK=True, GDAL_TIFF_OVR_BLOCKSIZE="128")
    cog_translate(src_path, dst_path, output_profile, config=config, in_memory=False, quiet=True, **options)


def extent_is_covered_by_second_extent(A, B):
    """
        A: Big Extent
        B: Small Extent
    """
    A = [round(x, 2) for x in A]
    B = [round(x, 2) for x in B]
    return A[0] >= B[0] and A[1] >= B[1] and A[2] <= B[2] and A[3] <= B[3]


def merge_downscaled_data_memmap(in_files, in_extents, out_file) -> tuple:
    """
    Merge downscaled data from multiple input files into a single output file using NumPy's memmap functionality.

    Parameters:
    - in_files (List[str]): List of paths to input files containing downscaled data.
    - in_extents (List[Tuple[float, float, float, float]]): List of extents, each represented as a tuple (xmin, ymin, xmax, ymax).
    - out_file (str): Path to the output file where the merged data will be stored.

    Returns:
    Tuple[str, Tuple[int, int, int, int], bool, Tuple[int, ...]]: A tuple containing:
        - The path to the output file.
        - The maximum extent of the merged data (xmin, ymin, xmax, ymax).
        - A boolean indicating whether the operation was successful.
        - The shape of the final merged data as a tuple.
    """
    in_extents, in_files = zip(*sorted(list(zip(in_extents, in_files)), key=lambda x: x[0][0], reverse=True))
    data_shape = np.load(in_files[0], mmap_mode='r').shape
    y_res = int(np.around(data_shape[1] // (in_extents[0][3] - in_extents[0][1]), 0))

    max_extent = get_max_from_extents(in_extents)
    final_lines = y_res * (max_extent[0]-max_extent[2])

    print(f' -> Creating output file')
    final_shape = (final_lines, *data_shape[1:])
    out_data = np.memmap(out_file, dtype=np.load(in_files[0], mmap_mode='r').dtype, mode='w+', shape=final_shape)
    current_size = 0
    for idx, filename in enumerate(in_files):
        print(f' -> Loading {os.path.basename(os.path.dirname(filename))}')
        temp_data = np.load(filename, mmap_mode='r')
        if idx == 0:
            out_data[current_size:current_size + (temp_data.shape[0] - y_res), ...] = temp_data[:-y_res, ...]
            current_size += temp_data.shape[0] - y_res
            del temp_data
        elif idx + 1 == len(in_files):
            out_data[current_size:current_size + (temp_data.shape[0] - y_res), ...] = temp_data[y_res:, ...]
            current_size += temp_data.shape[0] - y_res
            del temp_data
        else:
            out_data[current_size:current_size + (temp_data.shape[0] - 2 * y_res), ...] = temp_data[y_res:-y_res, ...]
            current_size += temp_data.shape[0] - 2 * y_res
            del temp_data
        collect()
    out_data = out_data[:current_size, ...]
    print(' -> Flushing merged dataset to harddisk')
    out_data.flush() # type: ignore
    header = np.lib.format.header_data_from_array_1_0(out_data)
    print(' -> Writing file header')
    with open(out_file, 'r+b') as f:
        np.lib.format.write_array_header_1_0(f, header)
    del out_data
    return out_file, get_max_from_extents(in_extents), True, final_shape


def load_specified_lines(filepath, extent, all_bands = True):
    try:
        y_max, y_min = float(extent.get('top')), float(extent.get('bottom'))
        x_max, x_min = float(extent.get('right')), float(extent.get('left'))
    except:
        y_max, y_min = np.max([extent[0], extent[2]]), np.min([extent[0], extent[2]])
        x_max, x_min = np.max([extent[1], extent[3]]), np.min([extent[1], extent[3]])

    if filepath.endswith('.nc'):
        var_name = nc.get_variable_name_from_nc(filepath)
        subset_data, nodata = nc.read_area_from_netcdf(filepath, extent, var_name)
    else:
        with rasterio.open(filepath, 'r') as src:
            window = src.window(x_min, y_min, x_max, y_max)
            try:
                window = Window(window.col_off, np.round(window.row_off), window.width, np.round(window.height)) #type:ignore
            except:
                pass
            if isinstance(all_bands, bool):
                if all_bands:
                    if src.count == 1:
                        subset_data = src.read(window=window)
                    else:
                        shp = src.read(1, window=window).shape
                        subset_data = np.empty((src.count, shp[0], shp[1]), dtype=src.dtypes[1])
                        for i in range(src.count):
                            subset_data[i] = src.read(i+1, window=window)
                    #subset_data = src.read(window=window)
                else:
                    subset_data = src.read(1, window=window)
            elif isinstance(all_bands, int):
                subset_data = src.read(all_bands, window=window)
            nodata = src.nodata
    return subset_data, nodata


def check_geotiff_extent(geotiff_file, extent):
    # extent: [top, left, bottom, right]
    if not os.path.exists(geotiff_file):
        return False
    top, left, bottom, right = extent
    with rasterio.open(geotiff_file) as src:
        tiff_bounds = src.bounds
        tiff_left, tiff_bottom, tiff_right, tiff_top = tiff_bounds.left, tiff_bounds.bottom, tiff_bounds.right, tiff_bounds.top
    return (left >= tiff_left and right <= tiff_right and bottom >= tiff_bottom and top <= tiff_top)


def check_dimensions(reference, data):
    if reference.shape != data.shape[:2]:
        if reference.shape[0] > data.shape[0]:
            return np.pad(data, ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        if reference.shape[0] < data.shape[0]:
            return data[:-1]
    else:
        return data
        

def get_max_from_extents(extents) -> list:
    extents = np.asarray(extents)
    return [np.max(extents[..., 0]), np.min(extents[..., 1]), np.min(extents[..., 2]), np.max(extents[..., 3])]


def merge_downscaled_data(in_files, in_extents, out_file) -> tuple:
    in_extents, in_files = zip(*sorted(list(zip(in_extents, in_files)), key=lambda x: x[0][0], reverse=True))
    data_shape = np.load(in_files[0], mmap_mode='r').shape
    y_res = int(np.around(data_shape[1] // (in_extents[0][3] - in_extents[0][1]), 0))
    data = None

    if len(in_files) == 1:
        shutil.move(in_files[0], out_file)
    else:
        for idx, filename in enumerate(in_files):
            print(f' -> Loading {os.path.basename(os.path.dirname(filename))}')
            if idx == 0:
                data = np.load(filename, mmap_mode='r')[:-y_res, ...]
            else:
                if data is None:
                    throw_exit_error('Error merging Downscaled Climatedata: No data to merge found!')
                if idx+1 == len(in_files):
                    temp_data = np.load(filename, mmap_mode='r')[y_res:, ...]
                    data = np.concatenate((data, temp_data), axis=0) #type:ignore
                    del temp_data
                else:
                    temp_data = np.load(filename, mmap_mode='r')[y_res:-y_res, ...]
                    data = np.concatenate((data, temp_data), axis=0) #type:ignore
                    del temp_data
        print(' -> Writing merged dataset')
        np.save(out_file, data) #type:ignore
        del data
    return out_file, get_max_from_extents(in_extents)


def get_array_from_npy(npy_file, extent, npy_extent) -> np.ndarray:
    data_shape = np.load(npy_file, mmap_mode='r').shape
    y_res = data_shape[1] // (npy_extent[3] - npy_extent[1])
    start_line = int((npy_extent[0] - extent[0]) * y_res)
    end_line = int((npy_extent[0] - extent[2]) * y_res)
    if start_line > end_line:
        start_line, end_line = end_line, start_line
    if end_line > data_shape[0] or start_line < 0:
        end_line = data_shape[0]
        start_line = 0
        print('Warning: Selected Extent not covered by climate data!\nClipping to existing climate data')
    npz_file = np.load(npy_file, mmap_mode='r')
    block_data = npz_file[start_line:end_line]
    del npz_file
    return block_data


def remove_files(directory, pattern):
    npy_files = glob.glob(os.path.join(directory, '**', pattern), recursive=True)
    for file_path in npy_files:
        try:
            os.remove(file_path)
        except:
            pass


def get_domain(climate_config):
    """
    Extracts domain information from climate configuration.

    Parameters:
    - climate_config (dict): Dictionary containing climate configuration data.

    Returns:
    - domain (numpy.ndarray): Array of floats representing domain information in the format
      [min_x, max_y, max_x, min_y, no_cols, no_rows]. Raises an error and exits if there's a problem
      reading extent from the config file.
    """
    try:
        min_x = climate_config['extent']['upper_left_x']
        max_y = climate_config['extent']['upper_left_y']
        max_x = climate_config['extent']['lower_right_x']
        min_y= climate_config['extent']['lower_right_y']
        no_cols = 0
        no_rows = 0
        return np.array([min_x, max_y, max_x, min_y, no_cols, no_rows]).astype(float)
    except Exception as e:
        input('Error: Problem reading extent from config file: \n'+str(e)+'\nExit.')
        sys.exit()


def get_shape_of_raster(raster_file):
    """
    Get the shape of a raster file.

    Parameters:
    - raster_file: Numpy array or raster file object. Input raster data.

    Returns:
    - shape: Tuple. Shape of the raster data in the format (rows, columns).
    """
    return(np.shape(raster_file))



def extract_domain_from_global_3draster(raster_dataset, domain, raster_extent=[-180, 90, 180, -90], axis=0):
    """
    Extracts a specific domain from a global 3D raster dataset.

    Parameters:
    - raster_dataset (numpy.ndarray): Global 3D raster dataset.
    - domain (list or numpy.ndarray): Domain to be extracted in the format [left, top, right, bottom].
    - raster_extent (list, optional): Extent of the entire global raster in the format [left, top, right, bottom].
      Defaults to [-180, 90, 180, -90].
    - axis (int, optional): Axis along which the domain is extracted. Defaults to 0.

    Returns:
    - extracted_domain (numpy.ndarray): Extracted domain from the global 3D raster dataset.
    """
    if raster_dataset.ndim == 2:
        big_rows, big_cols = get_shape_of_raster(raster_dataset) # Großes Raster Zeilen und Spalten
    else:
        big_rows, big_cols = get_shape_of_raster(raster_dataset)[1:3]
    try:
        big_uy = raster_extent.top # Großes Raster oben
        big_by = raster_extent.bottom # Großes Raster unten
        big_lx = raster_extent.left # Großes Raster links
        big_rx = raster_extent.right # Großes Raster rechts
    except:
        big_uy = raster_extent[1] # Großes Raster oben
        big_by = raster_extent[3] # Großes Raster unten
        big_lx = raster_extent[0] # Großes Raster links
        big_rx = raster_extent[2] # Großes Raster rechts

    try:
        small_uy = domain.top # Kleines Raster oben
        small_by = domain.bottom # Kleines Raster unten
        small_lx = domain.left # Kleines Raster links
        small_rx = domain.right # Kleines Raster rechts
    except:
        small_uy = domain[1] # Kleines Raster oben
        small_by = domain[3] # Kleines Raster unten
        small_lx = domain[0] # Kleines Raster links
        small_rx = domain[2] # Kleines Raster rechts
    pix_top = int(np.round((big_rows/(big_uy - big_by)) * (small_uy-big_by)))
    pix_bottom = int(np.round((big_rows/(big_uy - big_by)) * (small_by-big_by)))
    pix_left = int(np.round((big_cols/(big_rx - big_lx)) * (small_lx - big_lx)))
    pix_right = int(np.round((big_cols/(big_rx - big_lx)) * (small_rx - big_lx)))
    if axis == 0:
        return raster_dataset[:, big_rows-pix_top:big_rows-pix_bottom, pix_left:pix_right]
    elif axis == 1:
        return raster_dataset[big_rows-pix_top:big_rows-pix_bottom, :, pix_left:pix_right]
    elif axis == 2:
        return raster_dataset[big_rows-pix_top:big_rows-pix_bottom, pix_left:pix_right, ...]
    else:
        return []

def extract_domain_from_global_raster(raster_dataset, domain, raster_extent=[-180, 90, 180, -90]):
    """
    Extracts a specific domain from a global raster dataset.

    Parameters:
    - raster_dataset (numpy.ndarray): Global raster dataset.
    - domain (list or numpy.ndarray): Domain to be extracted in the format [Left, Top, Right, Bottom].
    - raster_extent (list, optional): Extent of the entire global raster in the format [Left, Top, Right, Bottom].
      Defaults to [-180, 90, 180, -90].

    Returns:
    - extracted_domain (numpy.ndarray): Extracted domain from the global raster dataset.
    """

    if raster_dataset.ndim == 2:
        big_rows, big_cols = get_shape_of_raster(raster_dataset) # Großes Raster Zeilen und Spalten
    else:
        big_rows, big_cols = get_shape_of_raster(raster_dataset)[1:3]
    try:
        big_uy = raster_extent.top # Großes Raster oben
        big_by = raster_extent.bottom # Großes Raster unten
        big_lx = raster_extent.left # Großes Raster links
        big_rx = raster_extent.right # Großes Raster rechts
    except:
        big_uy = raster_extent[1] # Großes Raster oben
        big_by = raster_extent[3] # Großes Raster unten
        big_lx = raster_extent[0] # Großes Raster links
        big_rx = raster_extent[2] # Großes Raster rechts

    try:
        small_uy = domain.top # Kleines Raster oben
        small_by = domain.bottom # Kleines Raster unten
        small_lx = domain.left # Kleines Raster links
        small_rx = domain.right # Kleines Raster rechts
    except:
        small_uy = domain[1] # Kleines Raster oben
        small_by = domain[3] # Kleines Raster unten
        small_lx = domain[0] # Kleines Raster links
        small_rx = domain[2] # Kleines Raster rechts

    pix_top = int(np.round((big_rows/(big_uy - big_by)) * (small_uy-big_by)))
    pix_bottom = int(np.round((big_rows/(big_uy - big_by)) * (small_by-big_by)))

    pix_left = int(np.round((big_cols/(big_rx - big_lx)) * (small_lx - big_lx)))
    pix_right = int(np.round((big_cols/(big_rx - big_lx)) * (small_rx - big_lx)))
    
    if raster_dataset.ndim == 2:
        return raster_dataset[big_rows-pix_top:big_rows-pix_bottom, pix_left:pix_right]
    else:
        return raster_dataset[:, big_rows-pix_top:big_rows-pix_bottom, pix_left:pix_right]
    


def split_area_into_rows(extent, num_rows, overlap=0):
    if overlap != 0:
        max_y, min_x, min_y, max_x = extent
        row_height = ((max_y - min_y) + overlap * (num_rows - 1)) // num_rows
        row_extents = []
        for i in range(num_rows):
            row_min_y = min_y + (row_height - overlap) * i
            row_max_y = row_min_y + row_height
            if i == num_rows:
                row_max_y = max_y
            row_extents.append([row_max_y, min_x, row_min_y, max_x])
        uniques = []
        for extent in row_extents:
            if not extent in uniques:
                uniques.append(extent)
        return sorted(uniques)
    else:
        lat_range = abs(extent[0]) + abs(extent[2])
        lat_size = lat_range / num_rows
        extents = []
        for i in range(num_rows):
            bottom = extent[2] + (i * lat_size)
            top = extent[2] + ((1 + i) * lat_size)
            #extents.append({'top': top, 'left': extent[1], 'bottom': bottom, 'right': extent[3]})
            extents.append([top, extent[1], bottom, extent[3]])
        return extents


def get_resolution_array(config_dictionary, extent, only_shape=False, climate=False):
    resolution_value = int(config_dictionary['options'].get('resolution', 5))
    if climate:
        resolution_value = np.min([resolution_value, 5])
    resolution_dict = {0: 0.5, 1: 0.25, 2: 0.1, 3: 0.08333333333333, 4: 0.041666666666666, 5: 0.008333333333333, 6: 0.00208333333333333}

    try:
        y_max, y_min = float(extent.get('top')), float(extent.get('bottom'))
        x_max, x_min = float(extent.get('right')), float(extent.get('left'))
    except:
        y_max, y_min = np.max([extent[0], extent[2]]), np.min([extent[0], extent[2]])
        x_max, x_min = np.max([extent[1], extent[3]]), np.min([extent[1], extent[3]])

    resolution = resolution_dict.get(resolution_value, 0.00833333333333)

    px_y = (y_max - y_min) / resolution
    px_x = (x_max - x_min) / resolution

    if (px_y - int(px_y)) >= 0.5:
        px_y = int(px_y) + 1
    else:
        px_y = int(px_y)

    if (px_x - int(px_x)) >= 0.5:
        px_x = int(px_x) + 1
    else:
        px_x = int(px_x)

    if only_shape:
        return (px_y, px_x)
    else:
        return np.empty((px_y, px_x))

def resize_nearest_bool(array, final_shape, true_value = 1):
    new_array = np.empty(final_shape, dtype=array.dtype)
    y_res = new_array.shape[0] // array.shape[0]
    x_res = new_array.shape[1] // array.shape[1]
    if new_array.shape > array.shape:
        for y in array.shape[0]:
            for x in array.shape[1]:
                new_array[y*y_res:(y+1)*y_res, x*x_res:(x+1)*x_res] = array[y, x]

if __name__ == '__main__':
    pass