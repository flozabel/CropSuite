import numpy as np
import os
import xarray as xr
import shutil
from dask.distributed import Client
from dask.diagnostics import ProgressBar #type:ignore
from datetime import datetime
import rasterio
from dask.distributed import Client
import netCDF4 as nc4
from datetime import datetime
import math

def read_area_from_netcdf(filename, extent, variable='data', day_range=[-1, -1]):
    # extent = y_max, x_min, y_min, x_max
    ds = xr.open_dataset(filename, engine='netcdf4')

    if variable == '':
        for var_name in ds.data_vars:
            variable = var_name
            break
        if variable is None:
            raise ValueError("No data variables found in the NetCDF file.")
    try:
        ds[variable]
    except:
        variable = get_variable_name_from_nc(filename)

    dimensions = list(ds.dims.keys())

    if 'lat' in dimensions and 'lon' in dimensions:
        lat_dim, lon_dim = 'lat', 'lon'
    elif 'latitude' in dimensions and 'longitude' in dimensions:
        lat_dim, lon_dim = 'latitude', 'longitude'
    if len(dimensions) > 2:
        time_dim = [dim for dim in dimensions if dim not in ['lat', 'lon', 'latitude', 'longitude']][0]

    try:
        y_max, x_min, y_min, x_max = float(extent.get('top')), float(extent.get('left')), float(extent.get('bottom')), float(extent.get('right'))
    except:
        y_max, x_min, y_min, x_max = extent

    try:
        nodata = ds.attrs['_FillValue']
    except:
        nodata = False
    
    if isinstance(day_range, list):
        lat_slc = slice(y_max, y_min) if ds.lat[0] > ds.lat[-1] else slice(y_min, y_max)
        lon_slc = slice(x_min, x_max) if ds.lon[0] < ds.lon[-1] else slice(x_max, x_min)
        if day_range[1] > -1:
            data = ds.sel(**{lat_dim: lat_slc, lon_dim: lon_slc, time_dim: slice(day_range[0]+1, day_range[1]+1)}) #type:ignore
        else:
            data = ds.sel(**{lat_dim: lat_slc, lon_dim: lon_slc}) #type:ignore

    elif isinstance(day_range, int):
        lat_slc = slice(y_max, y_min) if ds.lat[0] > ds.lat[-1] else slice(y_min, y_max)
        lon_slc = slice(x_min, x_max) if ds.lon[0] < ds.lon[-1] else slice(x_max, x_min)
        if day_range > -1:
            data = ds.sel(**{lat_dim: lat_slc, lon_dim: lon_slc, time_dim: day_range+1}) #type:ignore
        else:
            data = ds.sel(**{lat_dim: lat_slc, lon_dim: lon_slc}) #type:ignore

    else:
        print('error')
    data = np.asarray(data[variable])
    return data, nodata

def get_netcdf_extent(file_path):
    """
    Get the spatial extent (min and max) of the latitude and longitude in a NetCDF file.

    Parameters:
    file_path (str): Path to the NetCDF file.

    Returns:
    dict: A dictionary with keys 'lon_min', 'lon_max', 'lat_min', and 'lat_max'.
    """
    # Open the NetCDF file
    ds = xr.open_dataset(file_path)
    
    # Check if 'latitude' and 'longitude' exist, otherwise, look for other possible names
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        lat = ds['latitude']
        lon = ds['longitude']
    elif 'lat' in ds.coords and 'lon' in ds.coords:
        lat = ds['lat']
        lon = ds['lon']
    else:
        raise ValueError("Could not find latitude and longitude coordinates in the dataset.")
    
    # Get the extent
    extent = {
        'left': float(lon.min()),
        'right': float(lon.max()),
        'bottom': float(lat.min()),
        'top': float(lat.max())
    }
    
    return extent

def check_extent_load_file(filepath, extent):
    # extent: [top, left, bottom, right]
    extent_dict = {k: float(extent[k]) for k in ('top', 'left', 'bottom', 'right')} if isinstance(extent, dict) else dict(zip(('top', 'left', 'bottom', 'right'), map(float, extent)))

    extent_dict = {
        'top': math.floor(extent_dict['top'] * 10) / 10,
        'right': math.floor(extent_dict['right'] * 10) / 10,
        'bottom': math.ceil(extent_dict['bottom'] * 10) / 10,
        'left': math.ceil(extent_dict['left'] * 10) / 10
    }

    if not os.path.exists(filepath):
        return False
    else:
        nc_extent = get_netcdf_extent(filepath)
        return (extent_dict.get('top') <= nc_extent.get('top')) and (extent_dict.get('left') >= nc_extent.get('left')) and (extent_dict.get('bottom') >= nc_extent.get('bottom')) and (extent_dict.get('right') <= nc_extent.get('right')) #type:ignore

def get_rows_cols(filename):
    dataset = xr.open_dataset(filename)
    return (dataset.dims['lat'], dataset.dims['lon'])

def get_variable_name_from_nc(ds):
    if isinstance(ds, str):
        ds = xr.open_dataset(ds)
    return list(ds.data_vars)[0]

def get_y_resolution(nc_file_path):
    dataset = xr.open_dataset(nc_file_path)
    lat = dataset.get('lat', dataset.get('latitude'))
    if lat is None:
        raise ValueError("Latitude variable not found")
    return abs(lat.diff(dim='lat').mean().item())


def get_netcdf_dtypes(ds, variable = None):
    """
    Get the data types of variables in a NetCDF file.

    Parameters:
    file_path (str): Path to the NetCDF file.

    Returns:
    dict: A dictionary where keys are variable names and values are their data types.
    """
    if isinstance(ds, str):
        ds = xr.open_dataset(ds)
    
    # Create a dictionary to store the data types
    if variable == None:
        variable_dtypes = {var: ds[var].dtype for var in ds.data_vars}
    else:
        variable_dtypes = ds[variable].dtype
    return variable_dtypes

def get_netcdf_shape(ds, variable=None):
    """
    Get the shape of variables in a NetCDF file.

    Parameters:
    file_path (str): Path to the NetCDF file.

    Returns:
    dict: A dictionary where keys are variable names and values are their shapes.
    """
    if isinstance(ds, str):
        ds = xr.open_dataset(ds)
    
    if variable == None:
        # Create a dictionary to store the shapes
        variable_shapes = {var: ds[var].shape for var in ds.data_vars}
    else:
        variable_shapes = ds[variable].shape
    return variable_shapes


def geotiff_to_netcdf(geotiff_files, netcdf_filename):
    data_arrays = []
    for idx, geotiff_file in enumerate(geotiff_files):
        with rasterio.open(geotiff_file) as src:
            data = src.read(1)
            lon, lat = np.meshgrid(
                np.linspace(src.bounds.left, src.bounds.right, src.width),
                np.linspace(src.bounds.top, src.bounds.bottom, src.height)
            )
            
        month_data = xr.DataArray(
            data,
            dims=("lat", "lon"),
            coords={"lat": lat[:, 0], "lon": lon[0, :]},
            name=f"month_{idx + 1}"  # name for each month's data
        )
        
        data_arrays.append(month_data)

    combined_data = xr.concat(data_arrays, dim="month")
    combined_data = combined_data.assign_coords(month=np.arange(1, len(geotiff_files) + 1))
    encoding = dict(zlib=True, complevel=4)
    ds = xr.Dataset({"data": combined_data})

    ds['lat'].attrs['units'] = 'degrees_north'
    ds['lon'].attrs['units'] = 'degrees_east'
    ds['lat'].attrs['long_name'] = 'latitude'
    ds['lon'].attrs['long_name'] = 'longitude'
    ds['lat'].attrs['axis'] = 'Y'
    ds['lon'].attrs['axis'] = 'X'

    ds.to_netcdf(netcdf_filename, encoding=encoding) #type:ignore
    
    print(f"NetCDF file created: {netcdf_filename}")
    for f in geotiff_files:
        os.remove(f)


def check_netcdf_extent(nc_file, extent, variable=False):
    var = variable if variable else 'data'
    if not os.path.exists(nc_file):
        return False, []
    try:
        # Open the NetCDF file
        ds = xr.open_dataset(nc_file, engine='netcdf4')
        # Get the latitude and longitude values
        try:
            lat, lon = ds['latitude'].values, ds['longitude'].values
        except:
            lat, lon = ds['lat'].values, ds['lon'].values
        # Check if the extent matches
        y_max, x_min, y_min, x_max = extent
        lat_min, lat_max = min(lat), max(lat)
        lon_min, lon_max = min(lon), max(lon)
        if (lat_min == y_min and lat_max == y_max and
            lon_min == x_min and lon_max == x_max):
            return True, np.asarray(ds[var])
        else:
            return False, []
    except Exception as e:
        try:
            ds.close()
        except:
            pass
        os.remove(nc_file)
        return False, []
    finally:
        ds.close()


def get_maximum_extent_from_list(downscaled_files):
    extents = [get_netcdf_extent(f) for f in downscaled_files]
    bottom, top = extents[0]['bottom'], extents[0]['top']
    left, right = extents[0]['left'], extents[0]['right']

    for extent in extents:
        bottom = min(bottom, extent['bottom'])
        top = max(top, extent['top'])
        left = min(left, extent['left'])
        right = max(right, extent['right'])
    return {'top': top, 'left': left, 'bottom': bottom, 'right': right}


def sort_coordinatelist(filelist):
    lst = list(zip(filelist, [int(str(os.path.basename(os.path.dirname(f))).split('_')[1].split('N')[0]) for f in filelist]))
    lst.sort(key=lambda x: x[1])    
    return [f[0] for f in lst]


def process_file(fn, extent, var_name):
    ds = xr.open_dataset(fn)
    ds_data = ds.sel(lat=slice(extent.get('top'), extent.get('bottom')), 
                     lon=slice(extent.get('left'), extent.get('right')))[var_name]
    ds.close()
    return np.asarray(ds_data)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()



def load_data(fn, extent, var_name):
    try:
        with xr.open_dataset(fn) as ds:
            ds_data = ds.sel(lat=slice(extent['top'], extent['bottom']),
                             lon=slice(extent['left'], extent['right']))[var_name]  # type:ignore
            return np.asarray(ds_data)
    except Exception as e:
        print(f"Error processing {fn}: {e}")
        return None
    

def read_area_from_netcdf_list(downscaled_files, overlap = False, var_name = 'data', extent = [0, 0, 0, 0], timestep=-1, dayslices=False, transp=True):
    """
        downscaled_files: list of netcdf files
        overlap: In Degree
        extent: [North, Left, South, Right]
    """

    if dayslices:
        downscaled_files = [f for f in downscaled_files if os.path.exists(f)]
        ds_list = []
        total = len(downscaled_files)
        try:
            extent = {'top': extent[0], 'left': extent[1], 'bottom': extent[2], 'right': extent[3]}
        except:
            pass

        for idx, fn in enumerate(downscaled_files):
            try:
                ds = xr.open_dataset(fn)
                ds_data = ds.sel(lat=slice(float(extent.get('top')), float(extent.get('bottom'))), lon=slice(float(extent.get('left')), float(extent.get('right'))))[var_name] #type:ignore
                ds.close()
                ds_list.append(np.asarray(ds_data))
            except:
                pass
            try:
                print_progress_bar(idx, total-1, prefix='       Progress', suffix='Complete', length=50)
            except:
                pass

        if transp:
            return np.transpose(np.asarray(ds_list), (1, 2, 0))
        else:
            return np.asarray(ds_list)

    else:
        if overlap:
            extent = {'top': extent[0] + overlap, 'left': extent[1], 'bottom': extent[2] - overlap, 'right': extent[3]}
        else:
            extent = {'top': extent[0], 'left': extent[1], 'bottom': extent[2], 'right': extent[3]}

        ds_list = []
        downscaled_files = sort_coordinatelist(downscaled_files)
        for fn in reversed(downscaled_files):
            current_extent = get_netcdf_extent(fn)
            if not (((current_extent['top'] >= extent['top']) and (current_extent['bottom'] <= extent['top'])) or\
                    ((current_extent['top'] >= extent['bottom']) and current_extent['bottom'] <= extent['bottom'])):
                continue

            current_area = {'top': np.min([current_extent.get('top'), extent.get('top')]), #type:ignore
                'left': np.max([current_extent.get('left'), extent.get('left')]), #type:ignore
                'bottom': np.max([current_extent.get('bottom'), extent.get('bottom')]), #type:ignore
                'right': np.min([current_extent.get('right'), extent.get('right')])} #type:ignore

            ds = xr.open_dataset(fn)
            if timestep == -1:
                ds_data = ds.sel(lat=slice(current_area.get('top'), current_area.get('bottom')), lon=slice(current_area.get('left'), current_area.get('right')))[var_name]
            else:
                ds_data = ds.sel(lat=slice(current_area.get('top'), current_area.get('bottom')), lon=slice(current_area.get('left'), current_area.get('right')), day=timestep)[var_name]
            ds.close() 
            ds_list.append(np.asarray(ds_data))

        if len(ds_list) == 1:
            return ds_list[0]
        else:
            if overlap:
                return []
            else:
                return np.concatenate(ds_list, axis=0)

"""
def write_to_netcdf(data, filename, dimensions=['lat', 'lon'], extent=None, compress=False, complevel=4, info_text=False, var_name='data', nodata_value=False, unlimited=None):
    '''
    Write data to a NetCDF4 file using xarray.

    Parameters:
    - data (ndarray): The input data array with dimensions (latitude, longitude, [day, ...])
    - filename (str): The name of the NetCDF file to be created.
    - extent (list or tuple, optional): The spatial extent [y_max, x_min, y_min, x_max].
    - compress (bool, optional): Whether to apply compression. Default is True.
    - complevel (int, optional): Compression level (1-9). Default is 4.

    Returns:
    - None
    '''    

    if len(dimensions) != data.ndim:
        print(f'Specified Dimensions {dimensions} not matching data dimensions {data.shape}')

    if extent:
        try:
            latitudes = np.linspace(float(extent.get('top')), float(extent.get('bottom')), data.shape[0])
            longitudes = np.linspace(float(extent.get('left')), float(extent.get('right')), data.shape[1])            
        except:
            try:
                extent = list(extent.values())
            except:
                pass
            latitudes = np.linspace(np.max([extent[0], extent[2]]), np.min([extent[0], extent[2]]), data.shape[0])
            longitudes = np.linspace(np.min([extent[1], extent[3]]), np.max([extent[1], extent[3]]), data.shape[1])
    else:
        latitudes, longitudes = range(data.shape[0]), range(data.shape[1])
    coords = {'lat': latitudes, 'lon': longitudes}

    if data.ndim > 2:
        for dim in range(2, data.ndim):
            coords[dimensions[dim]] = range(data.shape[dim])

    if data.dtype == np.float16:
        data = data.astype(np.float32)
    elif data.dtype == np.int8:
        data = data.astype(np.int16)
        
    ds = xr.Dataset(data_vars={var_name: (dimensions, data)}, coords=coords)

    if extent is not None:
        ds['lat'].attrs['units'] = 'degrees_north'
        ds['lon'].attrs['units'] = 'degrees_east'
        ds['lat'].attrs['long_name'] = 'latitude'
        ds['lon'].attrs['long_name'] = 'longitude'
        ds['lat'].attrs['axis'] = 'Y'
        ds['lon'].attrs['axis'] = 'X'

    
    ds[var_name].encoding['dtype'] = data.dtype
    
    if nodata_value:
        ds.attrs['_FillValue'] = nodata_value
    else:
        if np.count_nonzero(np.isnan(data)) > 0:
            ds.attrs['_FilLValue'] = np.nan
        elif np.min(data) == -32767:
            ds.attrs['_FillValue'] = -32767
        else:
            ds.attrs['_FillValue'] = 0
            
    ds.attrs['Institution'] = 'University of Basel, Department of Environmental Sciences'
    ds.attrs['Contact'] = 'Florian Zabel & Matthias Knuettel, florian.zabel@unibas.ch'
    ds.attrs['Creation_Time'] = f'{datetime.now().strftime("%d.%m.%Y - %H:%M")}'
    ds.attrs['Info'] = 'Created by CropSuite v1'
    if isinstance(info_text, str):
        ds.attrs['Info'] = info_text

    if compress:
        encoding = {var_name: {'zlib': compress, 'complevel': complevel}}
        if unlimited != None:
            ds.to_netcdf(filename, format='NETCDF4', engine='netcdf4', encoding=encoding, unlimited_dims=unlimited)
        else:
            ds.to_netcdf(filename, format='NETCDF4', engine='netcdf4', encoding=encoding)
    else:
        if unlimited != None:
            ds.to_netcdf(filename, format='NETCDF4', engine='netcdf4', unlimited_dims=unlimited)
        else:
            ds.to_netcdf(filename, format='NETCDF4', engine='netcdf4')
    return filename
"""

def write_to_netcdf(data, filename, dimensions=['lat', 'lon'], extent=None, compress=False, complevel=4, info_text=False, var_name='data', nodata_value=False, unlimited=None):
    if len(dimensions) != data.ndim:
        raise ValueError(f'Specified dimensions {dimensions} do not match data dimensions {data.shape}')
    
    if extent:
        try:
            latitudes = np.linspace(float(extent.get('top')), float(extent.get('bottom')), data.shape[0])
            longitudes = np.linspace(float(extent.get('left')), float(extent.get('right')), data.shape[1])            
        except:
            try:
                extent = list(extent.values())
            except:
                pass
            latitudes = np.linspace(np.max([extent[0], extent[2]]), np.min([extent[0], extent[2]]), data.shape[0])
            longitudes = np.linspace(np.min([extent[1], extent[3]]), np.max([extent[1], extent[3]]), data.shape[1])
    else:
        latitudes, longitudes = np.arange(data.shape[0]), np.arange(data.shape[1])
    
    # Ensure correct data type
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    elif data.dtype == np.int8:
        data = data.astype(np.int16)
    
    fill_value = 0
    if nodata_value:
        fill_value = nodata_value
    elif np.isnan(data).any():
        fill_value = np.nan
    elif np.min(data) == -32767:
        fill_value = -32767
    
    with nc4.Dataset(filename, 'w', format='NETCDF4') as ds: #type:ignore
        
        # Create dimensions
        ds.createDimension('lat', data.shape[0])
        ds.createDimension('lon', data.shape[1])
        for dim in range(2, data.ndim):
            ds.createDimension(dimensions[dim], data.shape[dim])
        
        # Create coordinate variables
        lat_var = ds.createVariable('lat', 'f4', ('lat',))
        lon_var = ds.createVariable('lon', 'f4', ('lon',))
        lat_var[:] = latitudes
        lon_var[:] = longitudes
        
        # Add attributes to coordinates
        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        lat_var.long_name = 'latitude'
        lon_var.long_name = 'longitude'
        lat_var.axis = 'Y'
        lon_var.axis = 'X'
        
        # Create data variable
        var_dims = tuple(dimensions)
        var = ds.createVariable(var_name, data.dtype, var_dims, zlib=compress, complevel=complevel, fill_value=fill_value) #type:ignore
        var[:] = data
        
        # Add global attributes
        ds.setncattr('Institution', 'University of Basel, Department of Environmental Sciences')
        ds.setncattr('Contact', 'Florian Zabel & Matthias Knuettel, florian.zabel@unibas.ch')
        ds.setncattr('Creation_Time', datetime.now().strftime("%d.%m.%Y - %H:%M"))
        ds.setncattr('Info', 'Created by CropSuite v1.5.0')
        if isinstance(info_text, str):
            ds.setncattr('Info', info_text)
    return filename



def create_append_netcdf(filename, data, dimensions=['lat', 'lon'], extent=None, var_name='data', overlap = 0, nodata_value = False, info_text = False, unlimited=None):
    """
    Write data to a NetCDF4 file using xarray.

    Parameters:
    - data (ndarray): The input data array with dimensions (latitude, longitude, [day, ...])
    - filename (str): The name of the NetCDF file to be created.
    - extent (list or tuple, optional): The spatial extent [y_max, x_min, y_min, x_max].
    - compress (bool, optional): Whether to apply compression. Default is True.
    - complevel (int, optional): Compression level (1-9). Default is 4.

    Returns:
    - None
    """    

    if len(dimensions) != data.ndim:
        print(f'Specified Dimensions {dimensions} not matching data dimensions {data.shape}')

    if extent:
        latitudes = np.linspace(np.max([extent[0], extent[2]]), np.min([extent[0], extent[2]]), data.shape[0])
        longitudes = np.linspace(np.min([extent[1], extent[3]]), np.max([extent[1], extent[3]]), data.shape[1])
    else:
        latitudes, longitudes = range(data.shape[0]), range(data.shape[1])
    coords = {'lat': latitudes, 'lon': longitudes}

    if data.ndim > 2:
        for dim in range(2, data.ndim):
            coords[dimensions[dim]] = range(data.shape[dim])

    ds = xr.Dataset(data_vars={var_name: (dimensions, data)}, coords=coords)

    if extent is not None:
        ds['lat'].attrs['units'] = 'degrees_north'
        ds['lon'].attrs['units'] = 'degrees_east'
        ds['lat'].attrs['long_name'] = 'latitude'
        ds['lon'].attrs['long_name'] = 'longitude'
        ds['lat'].attrs['axis'] = 'Y'
        ds['lon'].attrs['axis'] = 'X'

    ds[var_name].encoding['dtype'] = data.dtype
    
    if nodata_value:
        ds.attrs['_FillValue'] = nodata_value
    else:
        if np.count_nonzero(np.isnan(data)) > 0:
            ds.attrs['_FilLValue'] = np.nan
        elif np.min(data) == -32767:
            ds.attrs['_FillValue'] = -32767
        else:
            ds.attrs['_FillValue'] = 0
            
    ds.attrs['Institution'] = 'University of Basel, Department of Environmental Sciences'
    ds.attrs['Contact'] = 'Florian Zabel & Matthias Knuettel, florian.zabel@unibas.ch'
    ds.attrs['Creation_Time'] = f'{datetime.now().strftime("%d.%m.%Y - %H:%M")}'
    ds.attrs['Info'] = 'Created by CropSuite v1.5.0'
    if isinstance(info_text, str):
        ds.attrs['Info'] = info_text

    encoding = {var_name: {'zlib': True, 'complevel': 4}}
    print(f"Writing to {filename}")

    if os.path.exists(filename):
        existing_ds = xr.open_dataset(filename, mode='a')
        
        
        
        #existing_ds[var_name].loc[{'lat': slice(None)}] = xr.concat([existing_ds[var_name], ds], dim='lat')

    else:
        if unlimited != None:
            ds.to_netcdf(filename, format='NETCDF4', engine='netcdf4', encoding=encoding, unlimited_dims=unlimited)
        else:
            ds.to_netcdf(filename, format='NETCDF4', engine='netcdf4', encoding=encoding)


    print(f'{os.path.basename(filename)} successfully written!')
    return filename

        

        
    
    



def merge_netcdf_files(file_list, output_file, overlap=0, nodata_value=False, info_text=False):
    """
    Merge multiple NetCDF files based on latitude and longitude coordinates

    Parameters:
    - file_list (list): List of filenames of NetCDF files covering different latitude ranges.
    - output_file (str): The filename for the merged NetCDF file covering the area from 17°N to 3°S.
    - compress (bool, optional): Whether to apply compression. Default is True.
    - complevel (int, optional): Compression level (1-9). Default is 4.

    Returns:
    - output_file (str)
    """
    
    if len(file_list) == 1:
        shutil.copy(file_list[0], output_file)
        return output_file

    print(f'Merging {os.path.basename(output_file)}')
    # Open the NetCDF files as Dask-backed datasets
    datasets = [xr.open_dataset(file, chunks={'lat': 'auto', 'lon': 'auto'}) for file in file_list]
    try:
        sorted_datasets = sorted(datasets, key=lambda ds: ds.latitude.min(), reverse=True)
    except:
        sorted_datasets = sorted(datasets, key=lambda ds: ds.lat.min(), reverse=True)

    y_max, y_min = 0, 0
    for ds in datasets:
        if y_max < ds.lat.max():
            y_max = float(ds.lat.max())
        if y_min > ds.lat.min():
            y_min = float(ds.lat.min())

    if overlap > 0:
        selected_datasets = []
        for i, ds in enumerate(sorted_datasets):
            if i == 0:
                selected_datasets.append(ds.sel(lat=slice(ds.lat.max(), ds.lat.min() + overlap/2)))
            elif i == len(sorted_datasets) - 1:
                selected_datasets.append(ds.sel(lat=slice(ds.lat.max() - overlap/2, ds.lat.min())))
            else:
                selected_datasets.append(ds.sel(lat=slice(ds.lat.max() - overlap/2, ds.lat.min() + overlap/2)))
        merged_data = xr.concat(selected_datasets, dim='lat')
    else:
        merged_data = xr.concat(sorted_datasets, dim='lat')

    if isinstance(nodata_value, (int, float)):
        merged_data.attrs['_FillValue'] = nodata_value

    ds.attrs['Institution'] = 'University of Basel, Department of Environmental Sciences'
    ds.attrs['Contact'] = 'Florian Zabel, florian.zabel@unibas.ch'
    ds.attrs['Creation_Time'] = f'{datetime.now().strftime("%d.%m.%Y - %H:%M")}'
    ds.attrs['Info'] = 'Created by CropSuite v1.5.0'
    if isinstance(info_text, str):
        ds.attrs['Info'] = info_text

    merged_data['lat'] = ('lat', np.linspace(y_max, y_min, int(merged_data.dims['lat'])))
    encoding = {var: {'zlib': True, 'complevel': 4} for var in merged_data.data_vars}

    client = Client(n_workers=12)
    
    write_job = merged_data.to_netcdf(output_file, encoding=encoding, compute=False)
    print(f"Writing to {output_file}")
    with ProgressBar():
        write_job.compute() #type:ignore

    return output_file


def get_nodata_value(nc_file_path, variable='data'):
    try:
        ds = xr.open_dataset(nc_file_path)
        var = ds[variable]
        nodata_value = var.attrs['_FillValue'] 
        ds.close()
    except:
        nodata_value = -9999
    return nodata_value