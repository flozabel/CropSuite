import numpy as np
from math import modf
from scipy.interpolate import interp2d
from scipy.ndimage import uniform_filter


def get_coarse_dem_pixel(pixel_no, y_line_number, fine_dem_dims, coarse_dem_dims) -> tuple:
    """
    Map a pixel from a fine DEM to its corresponding pixel in a coarse DEM.

    Parameters:
    - pixel_no (int): Pixel number in the fine DEM.
    - y_line_number (int): Line number in the fine DEM.
    - fine_dem_dims (tuple): Dimensions of the fine DEM (no_cols, no_rows).
    - coarse_dem_dims (tuple): Dimensions of the coarse DEM (no_cols, no_rows).

    Returns:
    - tuple: Coarse pixel indices (aspalte_coarse, aspalte_coarse_r, azeile_coarse, azeile_coarse_r).
    """
    """
    pixel_no
    y_line_number
    fine_dem_dims = [no_cols, no_rows]
    coarse_dem_dims = [no_cols, no_rows]
    """
    aspalte_coarse = aspalte_coarse_r = azeile_coarse = azeile_coarse_r = 0

    n_rows_fine, n_cols_fine = fine_dem_dims
    n_rows_coarse, n_cols_coarse = coarse_dem_dims

    aspalte_coarse_r, aspalte_coarse = modf(pixel_no / n_cols_fine * n_cols_coarse)
    azeile_coarse_r, azeile_coarse = modf(y_line_number / n_rows_fine * n_rows_coarse)

    if aspalte_coarse_r > 0.5:
        aspalte_coarse += 1
        aspalte_coarse_r -= 1
    if azeile_coarse_r > 0.5:
        azeile_coarse += 1
        azeile_coarse_r -= 1

    return int(aspalte_coarse), float(aspalte_coarse_r), int(azeile_coarse), float(azeile_coarse_r)


def get_dem_rectangle(x1, x2, y1, y2, dem) -> np.ndarray:
    """
    Extract a rectangular region from a Digital Elevation Model (DEM).

    Parameters:
    - x1 (int): Starting column index.
    - x2 (int): Ending column index.
    - y1 (int): Starting row index.
    - y2 (int): Ending row index.
    - dem (numpy.ndarray): Digital Elevation Model array.

    Returns:
    - numpy.ndarray: Extracted rectangular region from the DEM.
    """
    return dem[x1:x2, y1:y2]


def resize_array(array, shape) -> np.ndarray:
    """
    Resize an input array using block averaging.

    Parameters:
    - array (numpy.ndarray): Input array to be resized.
    - shape (tuple): Desired shape of the resized array in the format (new_rows, new_columns).

    Returns:
    - numpy.ndarray: Resized array.
    """
    sh = shape[0], array.shape[0]//shape[0], shape[1], array.shape[1]//shape[1]
    return array.reshape(sh).mean(-1).mean(1)


def calc_regression_precip(coarse_dem, coarse_vals, aspalte_coarse_r, azeile_coarse_r, fine_dem_line, pixel_no, gradient = -9999., interp_flag = True) -> float:
    """
    Calculate precipitation values through regression using both coarse and fine-resolution data.

    Parameters:
    - coarse_dem (numpy.ndarray): 2D array representing the coarse-resolution digital elevation model.
    - coarse_vals (numpy.ndarray): 2D array representing the precipitation values corresponding to the coarse-resolution DEM.
    - aspalte_coarse_r (float): Relative pixel position in the x-direction within the fine-resolution grid.
    - azeile_coarse_r (float): Relative pixel position in the y-direction within the fine-resolution grid.
    - fine_dem_line (numpy.ndarray): 1D array representing the fine-resolution digital elevation model for a specific row.
    - pixel_no (int): Pixel index within the fine-resolution grid.
    - gradient (float, optional): Physical limits for the slope in the regression. Default is -9999.
    - interp_flag (bool, optional): Flag indicating whether to use bilinear interpolation. Default is True.

    Returns:
    - float: Precipitation value calculated through regression.

    Notes:
    - If interp_flag is True, the function performs bilinear interpolation on coarse_vals.
    - If interp_flag is False, the function calculates regression slopes, variances, and covariance for neighboring pixels.
    - The regression parameters are then used to estimate precipitation values for the given fine-resolution pixel.

    """
    slope_all = np.zeros(9)
    axis_sec = np.zeros(9)

    if interp_flag:
        arr = uniform_filter(coarse_vals, size=3, mode='reflect')
        ret_val = arr[3, 3]
        """
        interpolation_values = coarse_vals
        mask = ~np.isnan(interpolation_values[4:7, 4:7])
        sum_masked = np.sum(interpolation_values[4:7, 4:7][mask])
        count_masked = np.count_nonzero(mask)
        interpolation_values[interpolation_values == -9999] = sum_masked / max(1, count_masked)
        interpolation_values_n = interpolation_values[4:7, 4:7].flatten()
        ret_val = bilinear_interpolation(interpolation_values_n, aspalte_coarse_r, azeile_coarse_r)
        """
        return ret_val if ret_val >= 0. else 0. 

    
    for loop in range(0, 9):
        if loop < 8:
            x_range, y_range = range(loop % 3, loop % 3 + 5), range(loop // 3, loop // 3 + 5)
        else:
            x_range = y_range = range(0, 7), range(0, 7)

        table = np.empty((25, 2))
        table[:, 0] = coarse_dem[np.min([*x_range]):np.max([*x_range])+1, np.min([*y_range]):np.max([*y_range])+1].flatten()
        table[:, 1] = coarse_vals[np.min([*x_range]):np.max([*x_range])+1, np.min([*y_range]):np.max([*y_range])+1].flatten()
        table = np.delete(table, np.where((np.isnan(table)) | (table == 0.0)), axis=0)
        
        if len(table) > 0:
            # Calculate slope, variances and covariance
            slope = np.cov(table[:, 0], table[:, 1])[0, 1] / np.var(table[:, 0])
            # Check if slope is within physical limits
            if gradient != -9999:
                slope = np.clip(slope, -gradient, gradient)
            slope_all[loop], axis_sec[loop] = slope, np.mean(table[:, 1]) - slope * np.mean(table[:, 0])

    slope = bilinear_interpolation(slope_all, aspalte_coarse_r, azeile_coarse_r)
    axis_sec = bilinear_interpolation(axis_sec, aspalte_coarse_r, azeile_coarse_r)
    
    coarse_reg = np.zeros((5, 5))
    valid_indices = (slice(1, 6), slice(1, 6))
    valid_mask = ~np.isnan(coarse_dem[valid_indices]) & (coarse_dem[valid_indices] != -9999.)
    coarse_reg = slope * coarse_dem[valid_indices] + axis_sec
    coarse_reg[~valid_mask] = 0.0

    coarse_residuals = np.zeros(coarse_reg.shape)
    valid_mask = ~np.isnan(coarse_vals[valid_indices]) & (coarse_vals[valid_indices] != -9999.)
    coarse_residuals = coarse_reg - coarse_vals[valid_indices]
    coarse_residuals[~valid_mask] = 0.0

    fine_reg = slope * fine_dem_line[pixel_no] + axis_sec
    return fine_reg if fine_reg > 0. else 0.


def interpolate_radiation(coarse_vals, aspalte_coarse_r, azeile_coarse_r) -> float:
    """
    Interpolate radiation values based on the given coarse values and interpolation parameters.

    Parameters:
    - coarse_vals (numpy.ndarray): 2D array of coarse radiation values.
    - aspalte_coarse_r (float): Fractional part of the column coordinate for interpolation.
    - azeile_coarse_r (float): Fractional part of the row coordinate for interpolation.

    Returns:
    float: Interpolated radiation value.

    The function performs bilinear interpolation using the provided coarse values and interpolation
    parameters to estimate the radiation value at a specific point. If the resulting value is less
    than zero, it is clamped to zero.

    Note: The function assumes that valid radiation values are non-negative (>= 0), and invalid
    values are represented as -9999.
    """
    interpolation_values = coarse_vals
    mask = interpolation_values[3:6, 3:6] >= 0.
    sum_masked = np.sum(interpolation_values[3:6, 3:6][mask])
    count_masked = np.count_nonzero(mask)
    interpolation_values[interpolation_values == -9999] = sum_masked / max(1, count_masked)
    interpolation_values_n = interpolation_values[3:6, 3:6].flatten()
    ret_val = bilinear_interpolation(interpolation_values_n[0:9], aspalte_coarse_r, azeile_coarse_r)
    return ret_val if ret_val >= 0. else 0. 
    

def calc_regression(coarse_dem, coarse_vals, aspalte_coarse_r, azeile_coarse_r, fine_dem_line, pixel_no, gradient=-9999.) -> float:
    """
    Calculate regression values for fine DEM based on coarse DEM and corresponding values.

    Parameters:
    - coarse_dem (numpy.ndarray): 2D array representing the coarse DEM.
    - coarse_vals (numpy.ndarray): 2D array representing values associated with the coarse DEM.
    - aspalte_coarse_r (float): Fractional part of the column coordinate for interpolation.
    - azeile_coarse_r (float): Fractional part of the row coordinate for interpolation.
    - fine_dem_line (numpy.ndarray): 1D array representing a line of the fine DEM.
    - pixel_no (int): Pixel number in the fine DEM line for which to calculate the regression.
    - gradient (float, optional): Maximum allowed slope. Defaults to -9999.

    Returns:
    float: Calculated regression value for the specified pixel in the fine DEM.

    The function performs regression analysis based on the given coarse DEM and values to estimate
    the regression value for a specific pixel in the fine DEM. If the resulting value is less than
    zero, it is clamped to zero.
    """
    slope_all = np.zeros(9)
    axis_sec = np.zeros(9)

    for loop in range(0, 9):
        if loop == 0:
            x_range, y_range = range(0, 5), range(0, 5)
        elif loop == 1:
            x_range, y_range = range(1, 6), range(0, 5)
        elif loop == 2:
            x_range, y_range = range(2, 7), range(0, 5)
        elif loop == 3:
            x_range, y_range = range(0, 5), range(1, 6)
        elif loop == 4:
            x_range, y_range = range(1, 6), range(1, 6)
        elif loop == 5:
            x_range, y_range = range(2, 7), range(1, 6)
        elif loop == 6:
            x_range, y_range = range(0, 5), range(2, 7)
        elif loop == 7:
            x_range, y_range = range(1, 6), range(2, 7)
        elif loop == 8:
            x_range, y_range = range(2, 7), range(2, 7)
        else:
            x_range = y_range = range(0, 7), range(0, 7)

        table = np.empty((25, 2))
        table[:, 0] = coarse_dem[np.min([*x_range]):np.max([*x_range])+1, np.min([*y_range]):np.max([*y_range])+1].flatten()
        table[:, 1] = coarse_vals[np.min([*x_range]):np.max([*x_range])+1, np.min([*y_range]):np.max([*y_range])+1].flatten()
        table = np.delete(table, np.where((np.isnan(table)) | (table == 0.0)), axis=0)
        
        if len(table) > 0:
            # Calculate slope, variances and covariance
            slope = np.cov(table[:, 0], table[:, 1])[0, 1] / np.var(table[:, 0])
            # Check if slope is within physical limits
            if gradient != -9999:
                slope = np.clip(slope, -gradient, gradient)
            slope_all[loop], axis_sec[loop] = slope, np.mean(table[:, 1]) - slope * np.mean(table[:, 0])

    # ? Wenn aspalte_coarse_r und azeile_coarse_r == 0, wird einfach steigung und achsenabschnitt des "mittleren" Pixels zurückgegeben?
    # ? Was passiert an den Rändern mit der Regression?
    slope = bilinear_interpolation(slope_all, aspalte_coarse_r, azeile_coarse_r)
    axis_sec = bilinear_interpolation(axis_sec, aspalte_coarse_r, azeile_coarse_r)
    
    # Calculate values for climate model from regression
    coarse_reg = np.zeros((5, 5))
    valid_indices = (slice(1, 6), slice(1, 6))
    valid_mask = ~np.isnan(coarse_dem[valid_indices]) & (coarse_dem[valid_indices] != -9999.)
    coarse_reg = slope * coarse_dem[valid_indices] + axis_sec
    coarse_reg[~valid_mask] = 0.0

    coarse_residuals = np.zeros(coarse_reg.shape)
    valid_mask = ~np.isnan(coarse_vals[valid_indices]) & (coarse_vals[valid_indices] != -9999.)
    coarse_residuals = coarse_reg - coarse_vals[valid_indices]
    coarse_residuals[~valid_mask] = 0.0

    # residuals_interpolated = interpolate_residuals(aspalte_coarse_r=aspalte_coarse_r, azeile_coarse_r=azeile_coarse_r, residuals=coarse_residuals)

    fine_reg = slope * fine_dem_line[pixel_no] + axis_sec
    return fine_reg if fine_reg > 0. else 0.


def interpolate_residuals(aspalte_coarse_r: float, azeile_coarse_r: float, residuals: np.ndarray) -> float:
    """
    Interpolate residuals based on fractional column and row coordinates.

    Parameters:
    - aspalte_coarse_r (float): Fractional part of the column coordinate for interpolation.
    - azeile_coarse_r (float): Fractional part of the row coordinate for interpolation.
    - residuals (numpy.ndarray): 2D array of residuals for interpolation.

    Returns:
    float: Interpolated residual value.

    The function performs bilinear interpolation of residuals based on the fractional column and row
    coordinates. It extracts the appropriate submatrix from the residuals matrix and calculates the
    interpolated value using the given fractional coordinates.
    """
    if aspalte_coarse_r >= 0 and azeile_coarse_r >= 0:
        submatrix = residuals[2:4, 2:4]
    elif aspalte_coarse_r < 0 and azeile_coarse_r >= 0:
        submatrix = residuals[2:4, 1:3]
    elif aspalte_coarse_r >= 0 and azeile_coarse_r < 0:
        submatrix = residuals[1:3, 2:4]
    elif aspalte_coarse_r < 0 and azeile_coarse_r < 0:
        submatrix = residuals[1:3, 1:3]
    else:
        raise ValueError("aspalte_coarse_r and azeile_coarse_r must be non-negative")
    a, b, c, d = submatrix.flat
    dx = abs(aspalte_coarse_r)
    dy = abs(azeile_coarse_r)
    res_interpol = (1-dx)*(1-dy)*a + dx*(1-dy)*b + (1-dx)*dy*c + dx*dy*d
    return res_interpol


def interpolate_value(werte_in, azeile_coarse_r, aspalte_coarse_r, interp_func) -> float:
    """
    Interpolate a value from a given set of values based on fractional row and column coordinates.

    Parameters:
    - werte_in (list or np.ndarray): List or array of input values for interpolation.
    - azeile_coarse_r (float): Fractional part of the row coordinate for interpolation.
    - aspalte_coarse_r (float): Fractional part of the column coordinate for interpolation.
    - interp_func (callable): Interpolation function used to calculate the interpolated value.

    Returns:
    float: Interpolated value.

    The function checks if fractional row coordinate (azeile_coarse_r) is provided. If not, it returns
    the middle value of the input set. Otherwise, it uses the provided interpolation function (interp_func)
    to calculate the interpolated value based on fractional row and column coordinates.
    """
    if not azeile_coarse_r:
        return werte_in[len(werte_in)//2]
    else:
        return float(interp_func(aspalte_coarse_r, azeile_coarse_r))
 

def bilinear_interpolation(werte_in, aspalte_coarse_r, azeile_coarse_r, method='linear'):
    """
    Performs bilinear interpolation on a 3x3 grid of input values to estimate the value at a specified point.
    Parameters:
        werte_in (3x3 array): The input values as a 3x3 array.
        aspalte_coarse_r (float): The x-coordinate of the point to interpolate.
        azeile_coarse_r (float): The y-coordinate of the point to interpolate.
        method  {'linear', 'nearest', 'cubic'}: The interpolation method to use. Can be one of 'linear', 'nearest', or 'cubic'. Default is 'linear'.
    Returns:
        float:  The interpolated value at the specified point.
    Methods:
        'linear': Linear interpolation.
        'nearest': Nearest-neighbor interpolation.
        'cubic': Cubic interpolation.
    """
    if aspalte_coarse_r == 0. and azeile_coarse_r == 0.:
        return werte_in[4]
    interp_func = interp2d(np.array([-1, 0, 1]), np.array([-1, 0, 1]), werte_in.reshape((3, 3)), kind=method)
    return interpolate_value(werte_in, azeile_coarse_r, aspalte_coarse_r, interp_func)




def downscale_line(climate_config, landsea_line, coarse_dem, temp_data, precip_data, fine_dem_line, y_line_number, fine_dem_dims) -> np.ndarray:
    """
    Downscale climate data for a single row of pixels.

    Parameters:
    - climate_config (dict): Configuration settings for climate data processing.
    - landsea_line (np.ndarray): Boolean array indicating land-sea mask for the current row.
    - coarse_dem (np.ndarray): Coarse-resolution digital elevation model.
    - temp_data (np.ndarray): Temperature data for downscaling.
    - precip_data (np.ndarray): Precipitation data for downscaling.
    - fine_dem_line (np.ndarray): Fine-resolution digital elevation model for the current row.
    - y_line_number (int): Line number of the current row.
    - fine_dem_dims (tuple): Dimensions of the fine-resolution DEM (no_cols, no_rows).

    Returns:
    np.ndarray: Downscaled climate data for the current row of pixels.

    The function processes each pixel in the row based on the land-sea mask. For land pixels,
    it extracts the relevant region from the coarse DEM, performs downscaling for temperature and precipitation
    data, and returns the downscaled climate data for the row.

    Note: The function assumes that the inputs are consistent and properly aligned.

    """
    no_pixels = len(fine_dem_line)
    print(f'processing {np.count_nonzero(landsea_line)} pixel out of {no_pixels} pixel in the current row')

    no_variables = 2

    line = np.zeros((no_pixels, 365, no_variables))
    for pixel in range(no_pixels):
        if landsea_line[pixel] == False:
            continue
        else:

            coarse_x, coarse_x_rest, coarse_y, coarse_y_rest = get_coarse_dem_pixel(pixel, y_line_number, fine_dem_dims, [np.shape(coarse_dem)[0], np.shape(coarse_dem)[1]])
            x1, x2 = np.clip([coarse_x - 3, coarse_x + 4], 1, np.shape(coarse_dem)[0])
            y1, y2 = np.clip([coarse_y - 3, coarse_y + 4], 1, np.shape(coarse_dem)[1])
            
            if coarse_x <= 3:
                x1, x2 = 0, 7
            if coarse_y <= 3:
                y1, y2 = 0, 7

            if x2 == np.shape(coarse_dem)[0]:
                x1 = x2 - 7
            if y2 == np.shape(coarse_dem)[1]:
                y1 = y2 - 7

            rect_dem = get_dem_rectangle(y1, y2, x1, x2, coarse_dem)
            if np.shape(rect_dem) != (7, 7):
                print('Error')

            if np.nansum(rect_dem) == 0:
                continue

            new_arr = np.zeros((365, no_variables))

            for timestep in range(1):
                # Temperature
                new_arr[timestep, 0] = calc_regression(rect_dem, temp_data[timestep, y1:y2, x1:x2], coarse_x_rest, coarse_y_rest, fine_dem_line, pixel, 0.00976)
                # Precipitation
                if precip_data:
                    new_arr[timestep, 1] = calc_regression_precip(rect_dem, precip_data[timestep, y1:y2, x1:x2], coarse_x_rest, coarse_y_rest, fine_dem_line, pixel, -9999, climate_config['options']['interpolate_precip_without_elevation'])

            line[pixel] = new_arr
            #gtd(st)
    line[line == 0.] = np.nan
    return line
