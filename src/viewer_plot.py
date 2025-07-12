import rasterio
import numpy as np
import os
import cartopy.crs as ccrs
import tkinter as tk
from tkinter import ttk 
import xarray as xr

class ToggledFrame(tk.Frame):
    def __init__(self, parent, text="", *args, **options):
        tk.Frame.__init__(self, parent, *args, **options)

        self.show = tk.IntVar()
        self.show.set(0)

        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill="x", expand=1)

        ttk.Label(self.title_frame, text=text).pack(side="left", fill="x", expand=1)

        self.toggle_button = ttk.Checkbutton(self.title_frame, width=2, text='↓', command=self.toggle,
                                            variable=self.show, style='Toolbutton')
        self.toggle_button.pack(side="left")

        self.sub_frame = tk.Frame(self, relief="sunken", borderwidth=1)

    def toggle(self):
        if bool(self.show.get()):
            self.sub_frame.pack(fill="x", expand=1)
            self.toggle_button.configure(text='↑')
        else:
            self.sub_frame.forget()
            self.toggle_button.configure(text='↓')

def get_models(filepath):
    curr_path = os.path.normpath(os.path.join(os.path.dirname(filepath), '../../..'))
    curr_crop = os.path.basename(os.path.dirname(filepath))
    curr_fnme = os.path.basename(filepath)
    curr_area = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    curr_paths = [f for f in os.listdir(curr_path) if os.path.isdir(os.path.join(curr_path, f)) and (f.endswith('_var') or f.endswith('_novar'))]

    if len(curr_paths) == 0:
        return []

    ret_lst = []
    for d in curr_paths:
        act_pt = os.path.join(curr_path, d, curr_area, curr_crop, curr_fnme)
        if act_pt == filepath:
            continue
        if os.path.exists(act_pt):
            ret_lst.append(act_pt)
    return ret_lst

def get_years_from_name(file_list):
    years = []
    for f in file_list:
        fname = os.path.splitext(os.path.basename(f))[0]
        yearname = fname[-9:]
        years.append(int(yearname[:4]))
        years.append(int(yearname[-4:]))
    
    years = np.asarray(years)
    return np.arange(np.min(years), np.max(years)+1, 1).astype(int)

def get_variable_names(netcdf_file):
    ds = xr.open_dataset(netcdf_file)
    coord_vars = list(ds.coords)
    var_names = [var for var in ds.variables if var not in coord_vars]
    ds.close()
    return var_names

def get_netcdf_over_time_range(netcdf_files, start_year, end_year, tp='mean'):
    datasets = []
    for file in netcdf_files:
        ds = xr.open_dataset(file)
        coord_vars = list(ds.coords)
        var_name = [var for var in ds.variables if var not in coord_vars][0]
        if 'time' in ds.variables:
            if not np.issubdtype(ds['time'].dtype, np.datetime64):
                ds['time'] = xr.decode_cf(ds)['time']
            filtered_ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
            datasets.append(filtered_ds)
        else:
            raise ValueError(f"Time variable not found in {file}")
    combined_ds = xr.concat(datasets, dim="time")
    if tp == 'mean':
        data = combined_ds[var_name].mean(dim="time")
    elif tp == 'sum':
        data = combined_ds[var_name].sum(dim="time")
    else:
        data = []
    return data

def read_geotiff(filepath, resolution, resolution_mode=50, day=-1):
    if day < 0:
        with rasterio.open(filepath, 'r') as dataset:
            width = dataset.width
            height = dataset.height
            nodata = dataset.nodata
            bounds = dataset.bounds

            if resolution_mode == 0:
                data = dataset.read(1)
            else:
                if resolution_mode == 25:
                    factor = min(4, width // (resolution * 4))
                elif resolution_mode == 50:
                    factor = width // resolution
                elif resolution_mode == 75:
                    factor = width // (resolution // 2)
                else:
                    factor = width // 256

                if factor < 1:
                    data = dataset.read(1)
                else:
                    data = dataset.read(1, out_shape=(1, height // factor, width // factor))
    else:
        with rasterio.open(filepath, 'r') as dataset:
            width = dataset.width
            height = dataset.height
            nodata = dataset.nodata
            bounds = dataset.bounds

            if resolution_mode == 0:
                data = dataset.read()
            else:
                if resolution_mode == 25:
                    factor = min(4, width // (resolution * 4))
                elif resolution_mode == 50:
                    factor = width // resolution
                elif resolution_mode == 75:
                    factor = width // (resolution // 2)
                else:
                    factor = width // 256

                if factor < 1:
                    data = dataset.read()
                else:
                    data = dataset.read(out_shape=(dataset.bounds, height // factor, width // factor))
        data = data[..., day]
    del dataset
    data = data.astype(np.float16)
    data[data == nodata] = np.nan
    print(f' -> Data loaded with shape {data.shape} and nodata value {nodata}')
    return data, bounds

def read_geotiff_mean(filepath, resolution, resolution_mode=50):
    with rasterio.open(filepath, 'r') as dataset:
        width = dataset.width
        height = dataset.height
        nodata = dataset.nodata
        bounds = dataset.bounds

        if resolution_mode == 0:
            data = dataset.read()
        else:
            if resolution_mode == 25:
                factor = min(4, width // (resolution * 4))
            elif resolution_mode == 50:
                factor = width // resolution
            elif resolution_mode == 75:
                factor = width // (resolution // 2)
            else:
                factor = width // 256

            if factor < 1:
                data = dataset.read()
            else:
                data = dataset.read(out_shape=(dataset.bounds, height // factor, width // factor))
        data = np.nanmean(data, axis=(0, 1))
    del dataset
    data = data.astype(np.float16)
    data[data == nodata] = np.nan
    print(f' -> Data loaded with shape {data.shape} and nodata value {nodata}')
    return data, bounds

def get_layers(filepath):
    with rasterio.open(filepath) as src:
        if src.count > 1:
            return src.shape[1]
        return 1
    
def get_limiting_factors(filename):
    filename = os.path.join(os.path.dirname(filename), 'limiting_factor.inf')
    if os.path.exists(filename):
        labels = []
        with open(filename, 'r') as file:
            for line in file:
                if "value - limiting factor" in line:
                    continue
                if " - " in line:
                    labels.append(line.strip().split(" - ", 1)[1].title().replace("_", " "))
    else:
        labels = ['Temperature', 'Precipitation', 'Climate Variability', 'Photoperiod', 'Base Saturation', 'Coarse Fragments', 'Gypsum Content', 'Soil pH', 'Salinity (EC)',\
                    'Texture Class', 'Soil Organic Carbon Content', 'Sodicity (ESP)', 'Soil Depth', 'Slope', 'No Data']
    return labels


class ProjectionGetter:
    def __init__(self):
        self.proj_list = {'Projection: Plate Carrée': ccrs.PlateCarree(),
                 'Projection: Robinson': ccrs.Robinson(),
                 'Projection: Eckert IV': ccrs.EckertIV(),
                 'Projection: AlbersEqualArea': ccrs.AlbersEqualArea(),
                 'Projection: Mercator': ccrs.Mercator(),
                 'Projection: Mollweide': ccrs.Mollweide(),
                 'Projection: InterruptedGoodeHomolosine': ccrs.InterruptedGoodeHomolosine(),
                 'Projection: NearsidePerspective': ccrs.NearsidePerspective(),
                 'Projection: Aitoff': ccrs.Aitoff(),
                 'Projection: EqualEarth': ccrs.EqualEarth(),
                 'Projection: Hammer': ccrs.Hammer()}
    
    def get_projection(self, name):
        if name in self.proj_list:
            return self.proj_list[name]
        else:
            raise ValueError(f"Projetion name '{name}' not found.")
        
    def get_all_projections(self):
        return list(self.proj_list.keys())

class ColormapGetter:
    def __init__(self):
        self.colormap_list = {'suitability': ['white', 'darkgray', 'darkred', 'yellow', 'greenyellow', 'limegreen', 'darkgreen', 'darkslategray'],
                            'suitability_2': ['#f5f5f5', '#a9a9a9', '#fde725', '#a0da39', '#4ac16d', '#1fa187', '#277f8e', '#365c8d', '#46327e', '#440154'],
                            'suitability_colorblind': ['#f5f5f5', '#a6611a', '#dfc27d', '#80cdc1', '#018571'],
                            'multiple_cropping': ['white', '#eceeba', '#a7c55c', '#0b3d00'],
                            'suitable_sowing_days': ['palegoldenrod', 'forestgreen', 'darkgreen', 'teal'],
                            'suitable_sowing_days_colorblind': ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'],
                            'sowing_date': ['blue', 'green', 'yellow', 'red', 'blue'],
                            'sowing_date_colorblind': ['gray', '#648fff', '#dc267f', '#ffb000', 'gray'],
                            'white-green-black': ['white', '#d0d392', '#a7c55c', '#287200', '#025012'],
                            'temperature': ['rebeccapurple', 'mediumblue', 'turquoise', 'lightgreen', 'orange', 'firebrick', 'saddlebrown'],
                            'precipitation': ['aquamarine', 'darkturquoise', 'cornflowerblue', 'mediumblue', 'darkviolet'],
                            'base_saturation': ['#563209', '#50380b', '#494b16', '#3c6025', '#266f30', '#0a713b', '#00684a', '#005856', '#00475e', '#043863'],
                            'coarse_fragments': ['darkgreen', 'limegreen', 'yellow', 'red', 'darkred'],
                            'gypsum': ['darkgreen', 'limegreen', 'yellow', 'red', 'darkred'],
                            'pH': ['darkcyan', 'aquamarine', 'darkgreen', 'limegreen', 'coral', 'darkred'],
                            'salinity': ['darkgreen', 'limegreen', 'yellow', 'orange', 'darkred'],
                            'slope': ['darkgreen', 'limegreen', 'orange', 'darkred'],
                            'sodicity': ['darkgreen', 'limegreen', 'darkred'],
                            'organic_carbon': ['darkred', 'yellow', 'limegreen', 'darkgreen', 'darkslategrey'],
                            'soildepth': ['darkred', 'yellow', 'limegreen', 'darkgreen', 'darkslategrey'],
                            'texture': ['#f3f3d9', '#d5bb76', '#a78834', '#a17a10', '#76531b', '#987a4d', '#736142', '#8a652b', '#8a4f2b', '#a66b4a', '#a6494a', '#8f4a80', '#64303b'],
                            'crop_limiting': ['#FF7F7F', '#0060cc', '#ffaa00', '#ccca16', '#00e6a9', '#cd8966','#e1e1e1', '#0084a8', '#ffbee8', '#d7c29e', '#267300', '#f57ab6', '#a87000', '#686868', 'white'],
                            'diff': ['darkred', 'coral', 'white', 'cornflowerblue', 'darkblue'],
                            'diff_r': ['darkblue', 'cornflowerblue', 'white', 'coral', 'darkred'],
                            'sowdate_diff': ['darkred', 'coral', 'palegoldenrod', 'white', 'paleturquoise', 'cornflowerblue', 'darkblue'],
                            'limiting_factor': ['#FF7F7F', '#0060cc', '#ffaa00', '#ccca16'],
                            'magma': 'magma', 'inferno': 'inferno', 'plasma': 'plasma', 'viridis': 'viridis', 'cividis': 'cividis', 'twilight': 'twilight', 'twilight_shifted': 'twilight_shifted', 'turbo': 'turbo',
                            'Blues': 'Blues', 'BrBG': 'BrBG', 'BuGn': 'BuGn', 'BuPu': 'BuPu', 'CMRmap': 'CMRmap', 'GnBu': 'GnBu', 'Greens': 'Greens', 'Greys': 'Greys', 'OrRd': 'OrRd',
                            'Oranges': 'Oranges', 'PRGn': 'PRGn', 'PiYG': 'PiYG', 'PuBu': 'PuBu', 'PuBuGn': 'PuBuGn', 'PuOr': 'PuOr', 'PuRd': 'PuRd', 'Purples': 'Purples', 'RdBu': 'RdBu',
                            'RdGy': 'RdGy', 'RdPu': 'RdPu', 'RdYlBu': 'RdYlBu', 'RdYlGn': 'RdYlGn', 'Reds': 'Reds', 'Spectral': 'Spectral', 'Wistia': 'Wistia', 'YlGn': 'YlGn', 'YlGnBu': 'YlGnBu',
                            'YlOrBr': 'YlOrBr', 'YlOrRd': 'YlOrRd', 'afmhot': 'afmhot', 'autumn': 'autumn', 'binary': 'binary', 'bone': 'bone', 'brg': 'brg', 'bwr': 'bwr', 'cool': 'cool',
                            'coolwarm': 'coolwarm', 'copper': 'copper', 'cubehelix': 'cubehelix', 'flag': 'flag', 'gist_earth': 'gist_earth', 'gist_gray': 'gist_gray', 'gist_heat': 'gist_heat',
                            'gist_ncar': 'gist_ncar', 'gist_rainbow': 'gist_rainbow', 'gist_stern': 'gist_stern', 'gist_yarg': 'gist_yarg', 'gnuplot': 'gnuplot', 'gnuplot2': 'gnuplot2',
                            'gray': 'gray', 'hot': 'hot', 'hsv': 'hsv', 'jet': 'jet', 'nipy_spectral': 'nipy_spectral', 'ocean': 'ocean', 'pink': 'pink', 'prism': 'prism', 'rainbow': 'rainbow',
                            'seismic': 'seismic', 'spring': 'spring', 'summer': 'summer', 'terrain': 'terrain', 'winter': 'winter', 'Accent': 'Accent', 'Dark2': 'Dark2', 'Paired': 'Paired',
                            'Pastel1': 'Pastel1', 'Pastel2': 'Pastel2', 'Set1': 'Set1', 'Set2': 'Set2', 'Set3': 'Set3', 'tab10': 'tab10', 'tab20': 'tab20', 'tab20b': 'tab20b', 'tab20c': 'tab20c',
                            'magma_r': 'magma_r', 'inferno_r': 'inferno_r', 'plasma_r': 'plasma_r', 'viridis_r': 'viridis_r', 'cividis_r': 'cividis_r', 'twilight_r': 'twilight_r',
                            'twilight_shifted_r': 'twilight_shifted_r', 'turbo_r': 'turbo_r', 'Blues_r': 'Blues_r', 'BrBG_r': 'BrBG_r', 'BuGn_r': 'BuGn_r', 'BuPu_r': 'BuPu_r',
                            'CMRmap_r': 'CMRmap_r', 'GnBu_r': 'GnBu_r', 'Greens_r': 'Greens_r', 'Greys_r': 'Greys_r', 'OrRd_r': 'OrRd_r', 'Oranges_r': 'Oranges_r', 'PRGn_r': 'PRGn_r',
                            'PiYG_r': 'PiYG_r', 'PuBu_r': 'PuBu_r', 'PuBuGn_r': 'PuBuGn_r', 'PuOr_r': 'PuOr_r', 'PuRd_r': 'PuRd_r', 'Purples_r': 'Purples_r', 'RdBu_r': 'RdBu_r',
                            'RdGy_r': 'RdGy_r', 'RdPu_r': 'RdPu_r', 'RdYlBu_r': 'RdYlBu_r', 'RdYlGn_r': 'RdYlGn_r', 'Reds_r': 'Reds_r', 'Spectral_r': 'Spectral_r', 'Wistia_r': 'Wistia_r',
                            'YlGn_r': 'YlGn_r', 'YlGnBu_r': 'YlGnBu_r', 'YlOrBr_r': 'YlOrBr_r', 'YlOrRd_r': 'YlOrRd_r', 'afmhot_r': 'afmhot_r', 'autumn_r': 'autumn_r', 'binary_r': 'binary_r',
                            'bone_r': 'bone_r', 'brg_r': 'brg_r', 'bwr_r': 'bwr_r', 'cool_r': 'cool_r', 'coolwarm_r': 'coolwarm_r', 'copper_r': 'copper_r', 'cubehelix_r': 'cubehelix_r',
                            'flag_r': 'flag_r', 'gist_earth_r': 'gist_earth_r', 'gist_gray_r': 'gist_gray_r', 'gist_heat_r': 'gist_heat_r', 'gist_ncar_r': 'gist_ncar_r',
                            'gist_rainbow_r': 'gist_rainbow_r', 'gist_stern_r': 'gist_stern_r', 'gist_yarg_r': 'gist_yarg_r', 'gnuplot_r': 'gnuplot_r', 'gnuplot2_r': 'gnuplot2_r',
                            'gray_r': 'gray_r', 'hot_r': 'hot_r', 'hsv_r': 'hsv_r', 'jet_r': 'jet_r', 'nipy_spectral_r': 'nipy_spectral_r', 'ocean_r': 'ocean_r', 'pink_r': 'pink_r',
                            'prism_r': 'prism_r', 'rainbow_r': 'rainbow_r', 'seismic_r': 'seismic_r', 'spring_r': 'spring_r', 'summer_r': 'summer_r', 'terrain_r': 'terrain_r',
                            'winter_r': 'winter_r', 'Accent_r': 'Accent_r', 'Dark2_r': 'Dark2_r', 'Paired_r': 'Paired_r', 'Pastel1_r': 'Pastel1_r', 'Pastel2_r': 'Pastel2_r', 'Set1_r': 'Set1_r',
                            'Set2_r': 'Set2_r', 'Set3_r': 'Set3_r', 'tab10_r': 'tab10_r', 'tab20_r': 'tab20_r', 'tab20b_r': 'tab20b_r', 'tab20c_r': 'tab20c_r',
                            'rrpcf': ['darkgreen', 'yellow', 'orange', 'darkred', 'darkred', 'darkred', 'darkred', 'darkred']}
    
    def get_colormap(self, name):
        if name in self.colormap_list:
            return self.colormap_list[name]
        else:
            raise ValueError(f"Colormap name '{name}' not found.")
        
    def get_all_colormaps(self):
        return list(self.colormap_list.keys())
    
    def get_colormap_index(self, name):
        print(list(self.colormap_list.keys()).index(name))