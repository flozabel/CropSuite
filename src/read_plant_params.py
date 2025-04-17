import os
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.interpolate import interp1d, CubicSpline, PPoly, BarycentricInterpolator, Akima1DInterpolator
import sys

def get_crops_from_dict(plant_params):
    lst = []
    for key, value in plant_params.items():
        if not key in lst:
            lst.append(key)
    return lst

def plot_plant_params(form_arr, crop, sz = 12, no_stops = 50, no_cols = 4, len_growing_cycle=365):
    params = [param for param in form_arr]
    params_dict = {'temp': 'Temperature', 'prec': 'Precipitation', 'slope': 'Slope', 'soildepth': 'Soil Depth', 'texture': 'Texture Class',
                   'coarsefragments': 'Coarse Fragments', 'gypsum': 'Gypsum', 'base_sat': 'Base Saturation', 'ph': 'Soil pH', 'organic_carbon': 'Organic Carbon',
                   'elco': 'Electric Conductivity (Salinity)', 'esp': 'ESP (Sodicity)', 'freqcropfail': 'Recurrence rate of potential crop failures'}
    unit_dict = {'temp': '[°C]', 'prec': '[mm/growing period]', 'slope': '[°]', 'soildepth': '[m]', 'texture': '[1]', 'coarsefragments': '[Volume-%]',
                 'gypsum': '[% CaSO4]', 'base_sat': '[%]', 'ph': '[1]', 'organic_carbon': '[%]', 'elco': '[dS/m]', 'esp': '[%]', 'freqcropfail': '[%]'}

    fig = plt.figure(figsize=(sz, sz*5/6))
    fig.suptitle(f'{crop.capitalize()} - Growing Cycle: {len_growing_cycle}') 

    no_rows = math.ceil(len(form_arr) / 4)

    for i, param in enumerate(params):
        plt.subplot(no_cols, no_rows, i+1)
        plt.plot(np.linspace(form_arr[param]['min_val'], form_arr[param]['max_val'], no_stops), np.clip(100*form_arr[param]['formula'](np.linspace(form_arr[param]['min_val'], form_arr[param]['max_val'], no_stops)), 0, 100), 'k')
        if param in params_dict:
            plt.title(params_dict[param].capitalize(), fontsize='small', loc='left')
        else:
            plt.title(param.capitalize(), fontsize='small', loc='left')
        plt.ylabel('Suitability [%]', fontsize='small')
        plt.xticks(fontsize='small')
        plt.yticks(np.linspace(0, 100, 11), fontsize='small')
        if param in unit_dict:
            plt.xlabel(unit_dict[param], fontsize='small')
        plt.grid()
    fig.tight_layout(pad=0.75) #type:ignore
    if not os.path.exists(os.path.join(os.getcwd(), 'parameterization_plots')):
        os.mkdir(os.path.join(os.getcwd(), 'parameterization_plots'))
    plt.close()
    fig.savefig(os.path.join(os.getcwd(), 'parameterization_plots', str(crop)+'.png'), dpi=150) #type:ignore

def plot_plant_params_mult(form_arr, no_stops=50, sz=12, no_cols = 4):
    fig = plt.figure(figsize=(sz, sz*5/6))
    fig.suptitle('All Crops') 

    
    crops = [crop for crop in form_arr]
    params = [param for param in form_arr[crops[0]]]
    no_rows = math.ceil(len(params) / 4)
    params_dict = {'temp': 'Temperature', 'prec': 'Precipitation', 'slope': 'Slope', 'soildepth': 'Soil Depth', 'texture': 'Texture Class',
                   'coarsefragments': 'Coarse Fragments', 'gypsum': 'Gypsum', 'base_sat': 'Base Saturation', 'ph': 'pH', 'organic_carbon': 'Organic Carbon',
                   'elco': 'Electric Conductivity (Salinity)', 'esp': 'ESP (Sodicity)'}
    unit_dict = {'temp': '[°C]', 'prec': '[mm/growing period]', 'slope': '[°]', 'soildetph': '[m]', 'texture': '[]', 'coarsefragments': '[Volume-%]',
                 'gypsum': '[% CaSO4]', 'base_sat': '[%]', 'ph': '[]', 'organic_carbon': '[%]', 'elco': '[dS/m]', 'esp': '[%]'}
    
    #x_labs = ['Temperature', 'Precipitation', 'Slope', 'Soil Depth', 'Texture Class', 'Coarse Fragments', 'Gypsum', 'Base Saturation', 'pH', 'Organic Carbon', 'Electric Conductivity (Salinity)', 'ESP (Sodicity)']
    #units = ['[°C]', '[mm/growing period]', '[°]', '[m]', '[]', '[Volume-%]', '[% CaSO4]', '[%]', '[]', '[%]', '[dS/m]', '[%]']

    for i, param in enumerate(params):
        plt.subplot(no_cols, no_rows, i+1)
        for crop in form_arr:
            plt.plot(np.linspace(form_arr[crop][param]['min_val'], form_arr[crop][param]['max_val'], no_stops),\
                     np.clip(100*form_arr[crop][param]['formula'](np.linspace(form_arr[crop][param]['min_val'],\
                                                                              form_arr[crop][param]['max_val'], no_stops)), 0, 100), 'k', linewidth=.5)
        if param in params_dict:
            plt.title(params_dict[param], fontsize='small', loc='left')
        else:
            plt.title(param.captialize(), fontsize='small', loc='left')
        plt.ylabel('Suitability', fontsize='small')
        plt.xticks(fontsize='small')
        plt.yticks(np.linspace(0, 100, 11), fontsize='small')
        if param in unit_dict:
            plt.xlabel(unit_dict[param], fontsize='small')
        plt.grid()
    fig.tight_layout(pad=0.75) #type:ignore
    if not os.path.exists(os.path.join(os.getcwd(), 'parameterization_plots')):
        os.mkdir(os.path.join(os.getcwd(), 'parameterization_plots'))
    plt.close()
    fig.savefig(os.path.join(os.getcwd(), 'parameterization_plots', 'overview.png'), dpi=150) #type:ignore

def plot_all_parameterizations(form_arr, crop_dict):
    for crop in crop_dict:
        plot_plant_params(form_arr[crop], crop, len_growing_cycle=int(crop_dict[crop]['growing_cycle'][0]))
    print('Control plots of the membership functions of the crop parameterisations created')

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
    """
    if len(x_vals) > 3:
        try:
            if method == 0:  
                interpolator = lambda xi: np.interp(xi, x_vals, y_vals)
                return [interpolator, min(x_vals), max(x_vals)]
                #return [interp1d(x_vals, y_vals, kind='linear'), min(x_vals), max(x_vals)]
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
        interpolator = lambda xi: np.interp(xi, x_vals, y_vals)
        return [interpolator, min(x_vals), max(x_vals)]
        #return [interp1d(x_vals, y_vals, kind='linear'), min(x_vals), max(x_vals)]
    """

    interpolator = None
    try:
        if len(x_vals) <= 3:
            interpolator = lambda xi: np.interp(xi, x_vals, y_vals)
        elif method == 0:
            interpolator = lambda xi: np.interp(xi, x_vals, y_vals)
        elif method == 1:
            interpolator = CubicSpline(x_vals, y_vals, bc_type='not-a-knot')
        elif method == 2:
            interpolator = BarycentricInterpolator(x_vals, y_vals)
        elif method == 3:
            interpolator = CubicSpline(x_vals, y_vals)
        elif method == 4:
            interpolator = PPoly(y_vals, x_vals)  # Only if y_vals are coefficients!
        elif method == 5:
            interpolator = Akima1DInterpolator(x_vals, y_vals, method="makima")
        else:
            interpolator = lambda xi: np.interp(xi, x_vals, y_vals)
    except Exception:
        interpolator = lambda xi: np.interp(xi, x_vals, y_vals)

    return [interpolator, min(x_vals), max(x_vals)]

def get_plant_param_interp_forms(plant_params, methods):
    formula_list = []
    cnt = 0
    crop_list = get_crops_from_dict(plant_params)
    for section in plant_params.values():
        section_list = []
        curr_crop = crop_list[cnt]
        for param_name, x_vals in section.items():
            if '_vals' in param_name:
                y_vals = section.get(param_name.replace('_vals', '_suit'))
                try:
                    assert len(x_vals) == len(y_vals), f"Unequal lengths for {param_name}"
                except Exception as e:
                    input(f'Error reading file '+str(curr_crop)+'\n'+str(e))
                if 'temp_' in param_name:
                    if y_vals[-1] > 0:
                        y_vals.append(0)
                        x_vals.append(40)
                section_list.append(get_formula(x_vals, y_vals, method=int(methods[param_name[:-5]])))
        formula_list.append(section_list)
        cnt += 1
    return formula_list

def get_id_list_start(dict, starts_with):
    lst = []
    for id, __ in dict.items():
        if str(id).startswith(starts_with):
            lst.append(id)
    return lst

def get_plant_param_interp_forms_dict(plant_params, config):
    formula_dict = {}
    for crop_name, section in plant_params.items():
        section_dict = {}
        parameter_list = [entry.replace('parameters.', '', 1) if entry.startswith('parameters.') else entry for entry in get_id_list_start(config, 'parameters.')]
        parameter_dictionary = {config[f'parameters.{parameter_list[parameter_id]}']['rel_member_func']: parameter_list[parameter_id] for parameter_id in range(len(parameter_list))}
        for param_name, x_vals in section.items():
            if '_vals' in param_name:
                y_vals = section.get(param_name.replace('_vals', '_suit'))
                if param_name.replace('_vals', '') not in parameter_dictionary:
                    method = 1 if param_name in ['freqcropfail'] else 0
                else:
                    parameter_name = parameter_dictionary[param_name.replace('_vals', '')]
                    method = int(config[f'parameters.{parameter_name}']['interpolation_method'])
                try:
                    formula, min_val, max_val = get_formula(x_vals, y_vals, method)
                except Exception as e:
                    print(f'Parameterization error in {crop_name}:')
                    print(f' -> Parameter {param_name.replace("_vals", "")} is incorrect: ')
                    print(f'    * {param_name} = {x_vals}')
                    print(f'    * {param_name.replace("_vals", "_suit")} = {y_vals}')
                    input('Please correct the parameterization and try again.\nPress any Key to Exit')
                    sys.exit()
                section_dict[param_name.replace('_vals', '')] = {'formula': formula, 'min_val': min_val, 'max_val': max_val}
        formula_dict[crop_name] = section_dict
    return formula_dict

def read_crop_parameterizations_files(folder_path, suffix='.inf'):
    """
    Reads and parses crop parameterization files from a specified folder path.
    Args:
        folder_path (str): The path to the folder containing the crop parameterization files.
        suffix (str, optional): The file extension for the crop parameterization files. Defaults to '.inf'.
    Returns:
        dict: A dictionary where the key is the crop name and the value is a dictionary containing the crop's
              parameterization data.
    """
    data = {}
    for filename in os.listdir(folder_path):
        if not filename.endswith(suffix):
            continue
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        section = {}
        section_name = ''
        for line in lines:
            key, value = line.strip().split('=', 1)
            key = key.strip()
            value = value.strip()
            if key == 'name':
                section_name = value
            else:
                section[key] = value.split(',')
                try:
                    section[key] = [float(val) for val in section[key]]
                except ValueError:
                    try:
                        section[key] = [int(val) for val in section[key]]
                    except ValueError:
                        section[key] = [str(val) for val in section[key]]
        data[section_name] = section
    print_crop_param_output(data)
    return data

def read_single_crop_parameterizations_files(filepath, suffix='.inf'):
    data = {}
    with open(os.path.join(str(filepath)), 'r') as f:
        lines = f.readlines()
    section = {}
    section_name = ''
    for line in lines:
        key, value = line.strip().split('=', 1)
        key = key.strip()
        value = value.strip()
        if key == 'name':
            section_name = value
        else:
            section[key] = value.split(',')
            try:
                section[key] = [float(val) for val in section[key]]
            except ValueError:
                try:
                    section[key] = [int(val) for val in section[key]]
                except ValueError:
                    section[key] = [str(val) for val in section[key]]
            if len(section[key]) == 1:
                section[key] = section[key][0]
    data[section_name] = section
    return data

def print_sections(dictionary):
    """
    Prints the keys of a given dictionary as a list of sections or items.
    Args:
        dictionary (dict): A dictionary containing the items or sections to be printed.
    Returns:
        None
    """
    print('Loaded crops:')
    for key in dictionary:
        print(' * ', key)


def print_crop_param_output(dictionary):
    """
    Prints the number of crop parameterizations found and the keys of a given dictionary as a list of sections or items.
    Args:
        dictionary (dict): A dictionary containing the items or sections to be printed.
    Returns:
        None
    """
    print(f"{len(dictionary)} crop parameterizations found and successfully read:")
    print_sections(dictionary)